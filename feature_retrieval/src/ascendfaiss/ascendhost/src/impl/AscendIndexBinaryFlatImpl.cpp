/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * IndexSDK is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */

#include "ascendhost/include/impl/AscendIndexBinaryFlatImpl.h"
#include "common/utils/DataType.h"

namespace faiss {
namespace ascend {
bool AscendIndexBinaryFlatImpl::isRemoveFast = false;
AscendIndexBinaryFlatImpl::AscendIndexBinaryFlatImpl(const faiss::IndexBinaryFlat *index,
                                                     AscendIndexBinaryFlatConfig config, bool usedFloat)
    : resourceSize(config.resourceSize)
{
    CheckParam(config);
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "index is nullptr");
    d = index->d;
    code_size = d / BINARY_BYTE_SIZE;
    ntotal = 0;
    verbose = false;
    is_trained = true;
    metric_type = faiss::METRIC_L2;
    deviceId = config.deviceList.at(0);
    isUsedFloat = usedFloat;
    copyFrom(index);
}

AscendIndexBinaryFlatImpl::AscendIndexBinaryFlatImpl(const faiss::IndexBinaryIDMap *index,
                                                     AscendIndexBinaryFlatConfig config, bool usedFloat)
    : resourceSize(config.resourceSize)
{
    CheckParam(config);
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "index is nullptr");
    d = index->d;
    code_size = d / BINARY_BYTE_SIZE;
    ntotal = 0;
    verbose = false;
    is_trained = true;
    metric_type = faiss::METRIC_L2;
    deviceId = config.deviceList.at(0);
    isUsedFloat = usedFloat;

    copyFrom(index);
}

AscendIndexBinaryFlatImpl::AscendIndexBinaryFlatImpl(int dims, AscendIndexBinaryFlatConfig config, bool usedFloat)
    : resourceSize(config.resourceSize)
{
    CheckParam(config);
    FAISS_THROW_IF_NOT_MSG(std::find(DIMS.begin(), DIMS.end(), dims) != DIMS.end(),
                           "Unsupported dims, should be in { 256, 512, 1024 }.");
    if (dims == LARGE_DIM) {
        largeDimSetting();
    }
    d = dims;
    code_size = d / BINARY_BYTE_SIZE;
    ntotal = 0;
    verbose = false;
    is_trained = true;
    metric_type = faiss::METRIC_L2;
    deviceId = config.deviceList.at(0);
    isUsedFloat = usedFloat;
}

void AscendIndexBinaryFlatImpl::CheckParam(AscendIndexBinaryFlatConfig config)
{
    FAISS_THROW_IF_NOT_MSG(config.deviceList.size() == 1, "devices size should be 1.");
    FAISS_THROW_IF_NOT_FMT(config.resourceSize >= BINARY_FLAT_DEFAULT_MEM && config.resourceSize <= BINARY_FLAT_MAX_MEM,
                           "resource size should be in range [%ld, %ld]", BINARY_FLAT_DEFAULT_MEM, BINARY_FLAT_MAX_MEM);
    auto socName = aclrtGetSocName();
    FAISS_THROW_IF_NOT_MSG(socName != nullptr, "aclrtGetSocName() return nullptr. please check your environment ");
    std::string deviceName(socName);
    FAISS_THROW_IF_NOT_MSG(deviceName.find("Ascend310P") != std::string::npos,
                           "AscendIndexBinaryFlat can only be used on Ascend310P* device.");
}

void AscendIndexBinaryFlatImpl::Initialize()
{
    APP_LOG_INFO("AscendIndexBinaryFlat Initialize operation started.\n");
    FAISS_THROW_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE);
    pResources = std::unique_ptr<AscendResourcesProxy>(new AscendResourcesProxy());
    pResources->setTempMemory(resourceSize);
    pResources->initialize();
    resetTopkCompOp();
    resetTopkFloatCompOp();
    resetDistCompOp();
    resetDistFloatCompOp();
    APP_LOG_INFO("AscendIndexBinaryFlat Initialize operation end.\n");
}

void AscendIndexBinaryFlatImpl::setRemoveFast(bool useRemoveFast)
{
    isRemoveFast = useRemoveFast;
}

void AscendIndexBinaryFlatImpl::addVectors(int n, const uint8_t *x)
{
    APP_LOG_INFO("AscendIndexBinaryFlat addVectors operation started.\n");
    FAISS_THROW_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE);

    std::string opName = "TransdataShaped";
    auto &mem = pResources->getMemoryManager();
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();

    AscendTensor<uint8_t, DIMS_2> data(mem, {n, this->code_size}, stream);

    auto ret = aclrtMemcpy(data.data(), data.getSizeInBytes(), x, n * this->code_size, ACL_MEMCPY_HOST_TO_DEVICE);
    FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Mem operator error %d", (int)ret);

    int blockNum = utils::divUp(this->ntotal + n, BLOCK_SIZE);

    AscendTensor<int64_t, DIMS_2> attrs(mem, {blockNum, aicpu::TRANSDATA_SHAPED_ATTR_IDX_COUNT}, stream);

    int reshapeDim1 = utils::divUp(BLOCK_SIZE, zRegionHeight);
    int reshapeDim2 = this->d / CUBE_ALIGN;
    int align1 = zRegionHeight;
    int align2 = HAMMING_CUBE_ALIGN;
    if (isUsedFloat) {
        SetShapeDim(reshapeDim1, reshapeDim2, align1, align2);
    }
    size_t blockSize = static_cast<size_t>(BLOCK_SIZE);
    for (size_t i = 0; i < static_cast<size_t>(n);) {
        size_t total = static_cast<size_t>(this->ntotal) + i;
        size_t offsetInBlock = total % blockSize;
        size_t leftInBlock = blockSize - offsetInBlock;
        size_t leftInData = static_cast<size_t>(n) - i;
        size_t copyCount = std::min(leftInBlock, leftInData);
        size_t blockIdx = total / blockSize;

        int copy = static_cast<int>(copyCount);
        AscendTensor<uint8_t, DIMS_2> src(data[i].data(), {copy, this->code_size});
        AscendTensor<uint8_t, DIMS_4> dst(baseShaped[blockIdx]->data(),
                                          {reshapeDim1, reshapeDim2, align1, align2});
        AscendTensor<int64_t, DIMS_1> attr = attrs[blockIdx].view();
        attr[aicpu::TRANSDATA_SHAPED_ATTR_NTOTAL_IDX] = offsetInBlock;

        i += copyCount;

        LaunchOpTwoInOneOut<uint8_t, DIMS_2, ACL_UINT8, int64_t, DIMS_1, ACL_INT64, uint8_t, DIMS_4, ACL_UINT8>(
            opName, stream, src, attr, dst);
    }

    ret = synchronizeStream(stream);
    this->ntotal += n;
    FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "synchronizeStream addVectors stream failed: %i\n", ret);
    APP_LOG_INFO("AscendIndexBinaryFlat addVectors operation end.\n");
}

void AscendIndexBinaryFlatImpl::add_with_ids_inner(idx_t n, const uint8_t *x, const idx_t *xids)
{
    APP_LOG_INFO("AscendIndexBinaryFlat add_with_ids operation started.\n");
    FAISS_THROW_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE);

    FAISS_THROW_IF_NOT_FMT((n > 0) && (n <= MAX_N), "n must be > 0 and <= %ld", MAX_N);
    FAISS_THROW_IF_NOT_FMT(this->ntotal + n <= MAX_N, "ntotal must be <= %ld", MAX_N);
    FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
    idx_t ntotalNew = this->ntotal + n;
    if (xids != nullptr) {
        ids.insert(ids.end(), xids, xids + n);
    } else {
        ids.reserve(ntotalNew);
        for (idx_t idx = this->ntotal; idx < ntotalNew; ++idx) {
            ids.emplace_back(idx);
        }
    }

    if (!isRemoveFast) {
        SeparateAdd(n, x);
    } else {
        std::thread insertLabel([this, n, ntotalNew] {
            for (idx_t i = ntotalNew - n; i < ntotalNew; ++i) {
                label2IdxMap[this->ids[i]] = i;
            }
        });
        try {
            SeparateAdd(n, x);
            insertLabel.join();
        } catch (std::exception &e) {
            insertLabel.join();
            ASCEND_THROW_FMT("wait for add functor failed %s", e.what());
        }
    }

    APP_LOG_INFO("AscendIndexBinaryFlat add_with_ids operation end.\n");
}
void AscendIndexBinaryFlatImpl::SeparateAdd(idx_t n, const uint8_t *x)
{
    APP_LOG_INFO("AscendIndexBinaryFlatImpl SeparateAdd operation started.\n");

    size_t offset = 0;
    size_t addTotal =  static_cast<size_t>(n);
    size_t singleAddMax = static_cast<size_t>(utils::divDown(SINGLE_ADD_MAX, BLOCK_SIZE) * BLOCK_SIZE);
    while (addTotal > 0) {
        auto singleAdd = std::min(addTotal, singleAddMax);
        addWithIdsImpl(singleAdd, x + offset * this->code_size);
        offset += singleAdd;
        addTotal -= singleAdd;
    }

    APP_LOG_INFO("AscendIndexBinaryFlatImpl SeparateAdd operation finished.\n");
}

void AscendIndexBinaryFlatImpl::add_with_ids(idx_t n, const uint8_t *x, const idx_t *xids)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    add_with_ids_inner(n, x, xids);
}

void AscendIndexBinaryFlatImpl::addWithIdsImpl(idx_t n, const uint8_t *x)
{
    APP_LOG_INFO("AscendIndexBinaryFlat addWithIdsImpl operation started.\n");

    // Be aware that n is narrowing down from int64_t
    int blockNum = utils::divUp(this->ntotal, BLOCK_SIZE);
    int addBlockNum = utils::divUp(this->ntotal + n, BLOCK_SIZE) - blockNum;
    for (int i = 0; i < addBlockNum; ++i) {
        baseShaped.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<uint8_t>, MemorySpace::DEVICE_HUGEPAGE));
        baseShaped.at(blockNum + i)->resize(BLOCK_SIZE * this->code_size, true);
    }

    addVectors(n, x);

    APP_LOG_INFO("AscendIndexBinaryFlat addWithIdsImpl operation end.\n");
}

void AscendIndexBinaryFlatImpl::postProcess(idx_t searchNum, int topK, float16_t *outDistances, int32_t *distances,
                                            idx_t *labels)
{
    APP_LOG_INFO("AscendIndexBinaryFlat postProcess operation started.\n");

// Speed up post-processing with multithreading
#pragma omp parallel for if (searchNum > 1) num_threads(CommonUtils::GetThreadMaxNums())
    for (idx_t i = 0; i < searchNum; ++i) {
        std::transform(outDistances + i * topK, outDistances + (i + 1) * topK, distances + i * topK,
                       [&](const float16_t temp) -> int32_t {
                           return (this->d - static_cast<float>(fp16(temp))) / 2.0; // 汉明距离计算公式常量
                       });
        std::transform(labels + i * topK, labels + (i + 1) * topK, labels + i * topK,
                       [&](const idx_t temp) -> idx_t { return temp == -1 ? -1 : this->ids.at(temp); });
    }
    APP_LOG_INFO("AscendIndexBinaryFlat postProcess operation end.\n");
}

void AscendIndexBinaryFlatImpl::postProcess(int searchNum, int topK, float16_t *outDistances, float *distances,
                                            idx_t *labels)
{
    APP_LOG_INFO("AscendIndexBinaryFloat postProcess operation started.\n");
 
    for (int i = 0; i < searchNum; ++i) {
        std::transform(outDistances + i * topK, outDistances + (i + 1) * topK, distances + i * topK,
                       [&](const float16_t temp) -> float { return static_cast<float>(fp16(temp)); });
 
        std::transform(labels + i * topK, labels + (i + 1) * topK, labels + i * topK,
                       [&](const idx_t temp) -> idx_t { return temp == -1 ? -1 : this->ids.at(temp); });
    }
    APP_LOG_INFO("AscendIndexBinaryFloat postProcess operation end.\n");
}

void AscendIndexBinaryFlatImpl::searchPaged(int pageIdx,
                                            int batch,
                                            const uint8_t *x,
                                            int topK,
                                            AscendTensor<float16_t, DIMS_2> &outDistanceOnDevice,
                                            AscendTensor<idx_t, DIMS_2> &outIndicesOnDevice)
{
    APP_LOG_INFO("AscendIndexBinaryFlat searchPaged operation started.\n");
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = pResources->getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    auto &mem = pResources->getMemoryManager();
    int totalBlocks = utils::divUp(this->ntotal, BLOCK_SIZE);
    int pageBlocks = static_cast<int>((1.0 * resourceSize / BINARY_FLAT_DEFAULT_MEM) * PAGE_BLOCKS);
    int totalPages = utils::divUp(totalBlocks, pageBlocks);
    int lastPageBlocks = totalBlocks % pageBlocks == 0 ? pageBlocks : totalBlocks % pageBlocks;
    int fullPageFeatures = BLOCK_SIZE * pageBlocks;
    int lastBlockFeatures = this->ntotal % BLOCK_SIZE == 0 ? BLOCK_SIZE : this->ntotal % BLOCK_SIZE;
    int blockNum = pageIdx == totalPages - 1 ? lastPageBlocks : pageBlocks;

    AscendTensor<uint8_t, DIMS_2> queries(mem, {batch, this->code_size}, stream);
    auto ret =
        aclrtMemcpy(queries.data(), queries.getSizeInBytes(), x, batch * this->code_size, ACL_MEMCPY_HOST_TO_DEVICE);
    FAISS_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Failed to copy to device");

    AscendTensor<float16_t, DIMS_3> distResult(mem, {blockNum, batch, BLOCK_SIZE}, stream);
    AscendTensor<float16_t, DIMS_3> maxDistResult(mem, {blockNum, batch, BLOCK_SIZE / burstLen * 2}, stream);

    AscendTensor<uint32_t, DIMS_3> opSize(mem, {blockNum, ACTUAL_NUM_SIZE, 1}, stream);
    AscendTensor<uint16_t, DIMS_3> opFlag(mem, {blockNum, FLAG_SIZE, 1}, stream);
    opFlag.zero();

    // attrs: [0]asc, [1]k, [2]burstLen, [3]block_num
    AscendTensor<int64_t, DIMS_1> attrsInput(mem, {aicpu::TOPK_FLAT_ATTR_IDX_COUNT}, stream);
    std::vector<int64_t> attrs(aicpu::TOPK_FLAT_ATTR_IDX_COUNT);
    attrs[aicpu::TOPK_FLAT_ATTR_ASC_IDX] = 0;
    attrs[aicpu::TOPK_FLAT_ATTR_K_IDX] = topK;
    attrs[aicpu::TOPK_FLAT_ATTR_BURST_LEN_IDX] = burstLen;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_NUM_IDX] = blockNum;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_IDX] = pageIdx;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_NUM_IDX] = totalPages;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_SIZE_IDX] = fullPageFeatures;
    // In the big topk scenario, performance is better to use QUICK_HEAP only for the first page.
    attrs[aicpu::TOPK_FLAT_ATTR_QUICK_HEAP] = (topK >= BIG_TOPK_START && pageIdx == 0) ? 1 : 0;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_SIZE] = BLOCK_SIZE;
    ret = aclrtMemcpy(attrsInput.data(), attrsInput.getSizeInBytes(), attrs.data(), attrs.size() * sizeof(int64_t),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    FAISS_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Failed to copy to device");

    runTopkCompute(distResult, maxDistResult, opSize, opFlag, attrsInput, outDistanceOnDevice, outIndicesOnDevice,
                   streamAicpu);

    // opSize Host to Device,reduce communication
    std::vector<uint32_t> opSizeHost(blockNum * ACTUAL_NUM_SIZE);
    for (int i = 0; i < blockNum; ++i) {
        opSizeHost[ACTUAL_NUM_SIZE * i] = (pageIdx == totalPages - 1) && (i == blockNum - 1)?
            static_cast<uint32_t>(lastBlockFeatures):
            static_cast<uint32_t>(BLOCK_SIZE);
    }

    ret = aclrtMemcpy(opSize.data(), opSize.getSizeInBytes(), opSizeHost.data(), opSizeHost.size() * sizeof(uint32_t),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    FAISS_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Failed to copy opsize to device");

    int reshapeDim1 = utils::divUp(BLOCK_SIZE, zRegionHeight);
    int reshapeDim2 = this->d / CUBE_ALIGN;
    for (int i = 0; i < blockNum; ++i) {
        int baseShapedIdx = pageIdx * pageBlocks + i;
        AscendTensor<uint8_t, DIMS_4> shaped(baseShaped[baseShapedIdx]->data(),
                                             {reshapeDim1, reshapeDim2, zRegionHeight, HAMMING_CUBE_ALIGN});
        auto dist = distResult[i].view();
        auto maxDist = maxDistResult[i].view();
        auto flag = opFlag[i].view();
        auto actualSize = opSize[i].view();
        runDistCompute(queries, shaped, actualSize, dist, maxDist, flag, stream);
    }

    ret = synchronizeStream(stream);
    FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "synchronizeStream default stream: %i\n", ret);

    ret = synchronizeStream(streamAicpu);
    FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "synchronizeStream aicpu stream failed: %i\n", ret);

    APP_LOG_INFO("AscendIndexBinaryFlat searchPaged operation end.\n");
}
 
void AscendIndexBinaryFlatImpl::searchPaged(int pageIdx,
                                            int batch,
                                            AscendTensor<float16_t, DIMS_2> &queries,
                                            int topK,
                                            AscendTensor<float16_t, DIMS_2> &outDistanceOnDevice,
                                            AscendTensor<idx_t, DIMS_2> &outIndicesOnDevice)
{
    APP_LOG_INFO("AscendIndexBinaryFloat searchPaged operation started.\n");
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = pResources->getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    auto &mem = pResources->getMemoryManager();
    int totalBlocks = utils::divUp(this->ntotal, BLOCK_SIZE);
    int pageBlocks = static_cast<int>((1.0 * resourceSize / BINARY_FLAT_DEFAULT_MEM) * PAGE_BLOCKS);
    int totalPages = utils::divUp(totalBlocks, pageBlocks);
    int lastPageBlocks = totalBlocks % pageBlocks == 0 ? pageBlocks : totalBlocks % pageBlocks;
    int fullPageFeatures = BLOCK_SIZE * pageBlocks;
    int pageOffset = pageIdx * fullPageFeatures;
    int computeNum = std::min(static_cast<int>(this->ntotal) - pageOffset, fullPageFeatures);
    int blockNum = pageIdx == totalPages - 1 ? lastPageBlocks : pageBlocks;
    int burstLen = BURST_LEN_HIGH;
    auto curBurstsOfBlock = GetBurstsOfBlock(batch, BLOCK_SIZE, burstLen);

    AscendTensor<float16_t, DIMS_3> distResult(mem, { blockNum, batch, BLOCK_SIZE }, stream);
    AscendTensor<float16_t, DIMS_3> maxDistResult(mem, { blockNum, batch, curBurstsOfBlock }, stream);

    uint32_t opFlagSize = static_cast<uint32_t>(blockNum * CORE_NUM * FLAG_SIZE) * sizeof(uint16_t);
    uint32_t attrsSize = aicpu::TOPK_FLAT_ATTR_IDX_COUNT * sizeof(int64_t);
    uint32_t opSizeLen = static_cast<uint32_t>(blockNum * CORE_NUM * SIZE_ALIGN) * sizeof(uint32_t);
    uint32_t continuousMemSize = opFlagSize + attrsSize + opSizeLen;
    // 1) aclrtMemcpy比AscendTensor::zero更高效
    // 2) 使用连续内存来减少aclrtMemcpy的调用次数
    AscendTensor<uint8_t, DIMS_1, uint32_t> continuousMem(mem, { continuousMemSize }, stream);
    std::vector<uint8_t> continuousValue(continuousMemSize, 0);
    uint8_t *data = continuousValue.data();

    int64_t *attrs = reinterpret_cast<int64_t *>(data + opFlagSize + opSizeLen);
    attrs[aicpu::TOPK_FLAT_ATTR_ASC_IDX] = 0;
    attrs[aicpu::TOPK_FLAT_ATTR_K_IDX] = topK;
    attrs[aicpu::TOPK_FLAT_ATTR_BURST_LEN_IDX] = burstLen;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_NUM_IDX] = blockNum;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_IDX] = pageIdx;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_NUM_IDX] = totalPages;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_SIZE_IDX] = fullPageFeatures;
    attrs[aicpu::TOPK_FLAT_ATTR_QUICK_HEAP] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_SIZE] = BLOCK_SIZE;
    uint32_t *opSizeData = reinterpret_cast<uint32_t *>(data + opFlagSize);
    for (int i = 0; i < blockNum; ++i) {
        int offset = i * BLOCK_SIZE;
        int opSizeHostIdx = i * CORE_NUM * SIZE_ALIGN;
        opSizeData[opSizeHostIdx] = std::min(static_cast<uint32_t>(computeNum - offset),
            static_cast<uint32_t>(BLOCK_SIZE));
    }

    auto ret = aclrtMemcpy(continuousMem.data(), continuousMem.getSizeInBytes(),
        continuousValue.data(), continuousValue.size() * sizeof(uint8_t), ACL_MEMCPY_HOST_TO_DEVICE);

    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "AclrtMemcpy failed to copy to device %d", ret);

    uint16_t *opFlagMem = reinterpret_cast<uint16_t *>(continuousMem.data());
    AscendTensor<uint16_t, DIMS_3> opFlag(opFlagMem, { blockNum, CORE_NUM, FLAG_SIZE });
    uint32_t *opSizeMem = reinterpret_cast<uint32_t *>(continuousMem.data() + opFlagSize);
    AscendTensor<uint32_t, DIMS_3> opSize(opSizeMem, { blockNum, CORE_NUM, SIZE_ALIGN });
    int64_t *attrMem = reinterpret_cast<int64_t *>(continuousMem.data() + opFlagSize + opSizeLen);
    AscendTensor<int64_t, DIMS_1> attrsInput(attrMem, { aicpu::TOPK_FLAT_ATTR_IDX_COUNT });

    runTopkFloatCompute(distResult, maxDistResult, opSize, opFlag, attrsInput, outDistanceOnDevice, outIndicesOnDevice,
                        streamAicpu);

    int reshapeDim1 = utils::divUp(BLOCK_SIZE, zRegionHeight);
    int reshapeDim2 = utils::divUp(this->d, CUBE_ALIGN);
    int align1 = zRegionHeight;
    int align2 = HAMMING_CUBE_ALIGN;
    if (isUsedFloat) {
        SetShapeDim(reshapeDim1, reshapeDim2, align1, align2);
    }
    for (int i = 0; i < blockNum; ++i) {
        int baseShapedIdx = pageIdx * pageBlocks + i;
        AscendTensor<uint8_t, DIMS_4> shaped(baseShaped[baseShapedIdx]->data(),
                                             {reshapeDim1, reshapeDim2, align1, align2});

        auto dist = distResult[i].view();
        auto maxDist = maxDistResult[i].view();
        auto flag = opFlag[i].view();
        auto actualSize = opSize[i].view();
        runDistFloatCompute(queries, shaped, actualSize, dist, maxDist, flag, stream);
    }

    ret = synchronizeStream(stream);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "synchronizeStream default stream: %i\n", ret);

    ret = synchronizeStream(streamAicpu);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "synchronizeStream aicpu stream failed: %i\n", ret);

    APP_LOG_INFO("AscendIndexBinaryFloat searchPaged operation end.\n");
}

void AscendIndexBinaryFlatImpl::searchBatch(int batch, const uint8_t *x, int topK, int32_t *distances, idx_t *labels)
{
    APP_LOG_INFO("AscendIndexBinaryFlat searchBatch operation started.\n");
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = pResources->getMemoryManager();
    AscendTensor<float16_t, DIMS_2> outDistanceOnDevice(mem, {batch, topK}, stream);
    AscendTensor<idx_t, DIMS_2> outIndicesOnDevice(mem, {batch, topK}, stream);

    int pageBlocks = static_cast<int>((1.0 * resourceSize / BINARY_FLAT_DEFAULT_MEM) * PAGE_BLOCKS);
    int pageNum = utils::divUp(this->ntotal,  static_cast<idx_t>(BLOCK_SIZE) *  static_cast<idx_t>(pageBlocks));
    for (int pageIdx = 0; pageIdx < pageNum; ++pageIdx) {
        searchPaged(pageIdx, batch, x, topK, outDistanceOnDevice, outIndicesOnDevice);
    }

    std::vector<float16_t> outDistances(batch * topK);
    auto ret = aclrtMemcpy(outDistances.data(), batch * topK * sizeof(float16_t), outDistanceOnDevice.data(),
                           outDistanceOnDevice.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    FAISS_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Failed to copy back to host");
    ret = aclrtMemcpy(labels, batch * topK * sizeof(idx_t), outIndicesOnDevice.data(),
                      outIndicesOnDevice.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    FAISS_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Failed to copy back to host");

    postProcess(batch, topK, outDistances.data(), distances, labels);

    APP_LOG_INFO("AscendIndexBinaryFlat searchBatch operation end.\n");
}

void AscendIndexBinaryFlatImpl::searchBatch(int batch, const float *x, int topK, float *distances, idx_t *labels)
{
    APP_LOG_INFO("AscendIndexBinaryFloat searchBatch operation started.\n");
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = pResources->getMemoryManager();
    AscendTensor<float16_t, DIMS_2> outDistanceOnDevice(mem, { batch, topK }, stream);
    AscendTensor<int64_t, DIMS_2> outIndicesOnDevice(mem, { batch, topK }, stream);

    // convert query data from float to fp16, device use fp16 data to search
    std::vector<uint16_t> query(batch * this->d, 0);
    transform(x, x + batch * this->d, begin(query), [](float temp) { return fp16(temp).data; });
    AscendTensor<float16_t, DIMS_2> queries(mem, { batch, this->d }, stream);
    auto ret =
        aclrtMemcpy(queries.data(), queries.getSizeInBytes(),
                    query.data(), batch * this->d * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "AclrtMemcpy failed to copy to device");
    int pageBlocks = static_cast<int>((1.0 * resourceSize / BINARY_FLAT_DEFAULT_MEM) * PAGE_BLOCKS);
    int pageNum = utils::divUp(this->ntotal, static_cast<int64_t>(BLOCK_SIZE) * static_cast<int64_t>(pageBlocks));
    for (int pageIdx = 0; pageIdx < pageNum; ++pageIdx) {
        searchPaged(pageIdx, batch, queries, topK, outDistanceOnDevice, outIndicesOnDevice);
    }

    std::vector<float16_t> outDistances(batch * topK);
    ret = aclrtMemcpy(outDistances.data(), batch * topK * sizeof(float16_t), outDistanceOnDevice.data(),
                      outDistanceOnDevice.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    ASCEND_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Failed to copy back to host");
 
    ret = aclrtMemcpy(labels, batch * topK * sizeof(int64_t), outIndicesOnDevice.data(),
                      outIndicesOnDevice.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    ASCEND_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Failed to copy back to host");

    postProcess(batch, topK, outDistances.data(), distances, labels);

    APP_LOG_INFO("AscendIndexBinaryFloat searchBatch operation end.\n");
}
 
 
void AscendIndexBinaryFlatImpl::search(idx_t n, const uint8_t *x, idx_t k, int32_t *distances, idx_t *labels)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    APP_LOG_INFO("AscendIndexBinaryFlat search operation started.\n");
    FAISS_THROW_IF_NOT_FMT((n > 0) && (n <= MAX_N), "n(%ld) must be > 0 and <= %ld", n, MAX_N);
    FAISS_THROW_IF_NOT_FMT((k > 0) && (k <= MAX_TOPK), "k(%ld) must be > 0 and <= %d", k, MAX_TOPK);
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "x can not be nullptr");
    FAISS_THROW_IF_NOT_MSG(distances != nullptr, "distances can not be nullptr");
    FAISS_THROW_IF_NOT_MSG(labels != nullptr, "labels can not be nullptr");
    FAISS_THROW_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE);

    if (this->ntotal == 0) {
        return;
    }
    idx_t offset = 0;
    for (auto batch : BATCH_SIZES) {
        while (n >= batch) {
            searchBatch(batch, x + offset * this->code_size, static_cast<int>(k), distances + offset * k,
                        labels + offset * k);
            offset +=  static_cast<idx_t>(batch);
            n -=  static_cast<idx_t>(batch);
        }
    }

    APP_LOG_INFO("AscendIndexBinaryFlat search operation end.\n");
}

void AscendIndexBinaryFlatImpl::search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    APP_LOG_INFO("AscendIndexBinaryFloat search operation started.\n");
    ASCEND_THROW_IF_NOT_FMT((n > 0) && (n <= MAX_N), "n(%ld) must be > 0 and <= %ld", n, MAX_N);
    ASCEND_THROW_IF_NOT_FMT((k > 0) && (k <= MAX_TOPK), "k(%d) must be > 0 and <= %d", k, MAX_TOPK);
    ASCEND_THROW_IF_NOT_MSG(x != nullptr, "x can not be nullptr");
    ASCEND_THROW_IF_NOT_MSG(distances != nullptr, "distances can not be nullptr");
    ASCEND_THROW_IF_NOT_MSG(labels != nullptr, "labels can not be nullptr");
    ASCEND_THROW_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE);

    if (this->ntotal == 0) {
        return;
    }

    int64_t offset = 0;
    for (auto batch : BATCH_SIZES) {
        while (n >= batch) {
            searchBatch(batch, x + offset * this->d, k,
                        distances + offset * static_cast<int64_t>(k),
                        labels + offset * static_cast<int64_t>(k));
            offset += static_cast<int64_t>(batch);
            n -= static_cast<int64_t>(batch);
        }
    }

    APP_LOG_INFO("AscendIndexBinaryFloat search operation end.\n");
}

void AscendIndexBinaryFlatImpl::runDistCompute(AscendTensor<uint8_t, DIMS_2> &queryVecs,
                                               AscendTensor<uint8_t, DIMS_4> &shapedData,
                                               AscendTensor<uint32_t, DIMS_2> &size,
                                               AscendTensor<float16_t, DIMS_2> &outDistances,
                                               AscendTensor<float16_t, DIMS_2> &outMaxDistances,
                                               AscendTensor<uint16_t, DIMS_2> &flag,
                                               aclrtStream stream)
{
    APP_LOG_INFO("AscendIndexBinaryFlat runDistCompute operation started.\n");
    FAISS_THROW_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE);

    AscendOperator *op = nullptr;
    int batch = queryVecs.getSize(0);
    if (distComputeOps.find(batch) != distComputeOps.end()) {
        op = distComputeOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    distOpInput->emplace_back(aclCreateDataBuffer(queryVecs.data(), queryVecs.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(shapedData.data(), shapedData.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(size.data(), size.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(outDistances.data(), outDistances.getSizeInBytes()));
    distOpOutput->emplace_back(aclCreateDataBuffer(outMaxDistances.data(), outMaxDistances.getSizeInBytes()));
    distOpOutput->emplace_back(aclCreateDataBuffer(flag.data(), flag.getSizeInBytes()));

    op->exec(*distOpInput, *distOpOutput, stream);
    APP_LOG_INFO("AscendIndexBinaryFlat runDistCompute operation end.\n");
}
 
void AscendIndexBinaryFlatImpl::runDistFloatCompute(AscendTensor<float16_t, DIMS_2> &queryVecs,
                                                    AscendTensor<uint8_t, DIMS_4> &shapedData,
                                                    AscendTensor<uint32_t, DIMS_2> &size,
                                                    AscendTensor<float16_t, DIMS_2> &outDistances,
                                                    AscendTensor<float16_t, DIMS_2> &outMaxDistances,
                                                    AscendTensor<uint16_t, DIMS_2> &flag,
                                                    aclrtStream stream)
{
    APP_LOG_INFO("AscendIndexBinaryFloat runDistFloatCompute operation started.\n");
    ASCEND_THROW_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE);

    AscendOperator *op = nullptr;
    int batch = queryVecs.getSize(0);
    if (distComputeFloatOps.find(batch) != distComputeFloatOps.end()) {
        op = distComputeFloatOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    distOpInput->emplace_back(aclCreateDataBuffer(queryVecs.data(), queryVecs.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(shapedData.data(), shapedData.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(size.data(), size.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(outDistances.data(), outDistances.getSizeInBytes()));
    distOpOutput->emplace_back(aclCreateDataBuffer(outMaxDistances.data(), outMaxDistances.getSizeInBytes()));
    distOpOutput->emplace_back(aclCreateDataBuffer(flag.data(), flag.getSizeInBytes()));

    op->exec(*distOpInput, *distOpOutput, stream);
    APP_LOG_INFO("AscendIndexBinaryFloat runDistFloatCompute operation end.\n");
}

void AscendIndexBinaryFlatImpl::resetDistCompOp()
{
    APP_LOG_INFO("AscendIndexBinaryFlat resetDistCompOp operation started.\n");
    auto distCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("DistanceFlatHamming");
        std::vector<int64_t> input_shape0{batch, this->code_size};
        std::vector<int64_t> input_shape1{BLOCK_SIZE / zRegionHeight, this->d / CUBE_ALIGN, zRegionHeight,
                                          HAMMING_CUBE_ALIGN};
        std::vector<int64_t> input_shape2{ACTUAL_NUM_SIZE};
        std::vector<int64_t> output_shape0{BLOCK_SIZE * batch};
        std::vector<int64_t> output_shape1{BLOCK_SIZE * batch / burstLen * 2}; // vmax and indice, so we multiply by 2
        std::vector<int64_t> output_shape2{FLAG_SIZE};

        desc.addInputTensorDesc(ACL_UINT8, input_shape0.size(), input_shape0.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT8, input_shape1.size(), input_shape1.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, input_shape2.size(), input_shape2.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT16, output_shape0.size(), output_shape0.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, output_shape1.size(), output_shape1.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT16, output_shape2.size(), output_shape2.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : BATCH_SIZES) {
        distComputeOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        FAISS_THROW_IF_NOT_MSG(distCompOpReset(distComputeOps[batch], batch), "op init failed");
    }
    APP_LOG_INFO("AscendIndexBinaryFlat resetDistCompOp operation end.\n");
}

void AscendIndexBinaryFlatImpl::resetDistFloatCompOp()
{
    APP_LOG_INFO("AscendIndexBinaryFloat resetDistFloatCompOp operation started.\n");
    auto distCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("DistanceBinaryFloat");
        int reshapeDim1 = utils::divUp(BLOCK_SIZE, zRegionHeight);
        int reshapeDim2 = utils::divUp(this->d, CUBE_ALIGN);
        int align1 = zRegionHeight;
        int align2 = HAMMING_CUBE_ALIGN;
        if (isUsedFloat) {
            SetShapeDim(reshapeDim1, reshapeDim2, align1, align2);
        }
        int burstLen = BURST_LEN_HIGH;
        auto curBurstsOfBlock = GetBurstsOfBlock(batch, BLOCK_SIZE, burstLen);
        std::vector<int64_t> queryShape{ batch, this->d };
        std::vector<int64_t> coarseCentroidsShape{ reshapeDim1, reshapeDim2, align1, align2 };
        std::vector<int64_t> sizeShape{ CORE_NUM, SIZE_ALIGN };
        std::vector<int64_t> distResultShape{ batch, BLOCK_SIZE };
        std::vector<int64_t> maxResultShape{ batch, curBurstsOfBlock };
        std::vector<int64_t> flagShape{ CORE_NUM, FLAG_SIZE };

        desc.addInputTensorDesc(ACL_FLOAT16, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT8, coarseCentroidsShape.size(), coarseCentroidsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT16, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, maxResultShape.size(), maxResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : BATCH_SIZES) {
        distComputeFloatOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        ASCEND_THROW_IF_NOT_MSG(distCompOpReset(distComputeFloatOps[batch], batch), "op init failed");
    }
    APP_LOG_INFO("AscendIndexBinaryFloat resetDistFloatCompOp operation end.\n");
}

void AscendIndexBinaryFlatImpl::runTopkCompute(
    AscendTensor<float16_t, DIMS_3> &dists, AscendTensor<float16_t, DIMS_3> &maxDists,
    AscendTensor<uint32_t, DIMS_3> &sizes, AscendTensor<uint16_t, DIMS_3> &flags, AscendTensor<int64_t, DIMS_1> &attrs,
    AscendTensor<float16_t, DIMS_2> &outDists, AscendTensor<int64_t, DIMS_2> &outLabel, aclrtStream stream)
{
    APP_LOG_INFO("AscendIndexBinaryFlat runTopkCompute operation started.\n");
    AscendOperator *op = nullptr;
    int batch = dists.getSize(1);
    if (topkComputeOps.find(batch) != topkComputeOps.end()) {
        op = topkComputeOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(maxDists.data(), maxDists.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(sizes.data(), sizes.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(flags.data(), flags.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(attrs.data(), attrs.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(outDists.data(), outDists.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(outLabel.data(), outLabel.getSizeInBytes()));
 
    op->exec(*topkOpInput, *topkOpOutput, stream);
    APP_LOG_INFO("AscendIndexBinaryFlat runTopkCompute operation end.\n");
}

void AscendIndexBinaryFlatImpl::runTopkFloatCompute(
    AscendTensor<float16_t, DIMS_3> &dists, AscendTensor<float16_t, DIMS_3> &maxDists,
    AscendTensor<uint32_t, DIMS_3> &sizes, AscendTensor<uint16_t, DIMS_3> &flags, AscendTensor<int64_t, DIMS_1> &attrs,
    AscendTensor<float16_t, DIMS_2> &outDists, AscendTensor<int64_t, DIMS_2> &outLabel, aclrtStream stream)
{
    APP_LOG_INFO("AscendIndexBinaryFloat runTopkCompute operation started.\n");
    AscendOperator *op = nullptr;
    int batch = dists.getSize(1);
    if (topkComputeUint8Ops.find(batch) != topkComputeUint8Ops.end()) {
        op = topkComputeUint8Ops[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(maxDists.data(), maxDists.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(sizes.data(), sizes.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(flags.data(), flags.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(attrs.data(), attrs.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(outDists.data(), outDists.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(outLabel.data(), outLabel.getSizeInBytes()));
 
    op->exec(*topkOpInput, *topkOpOutput, stream);
    APP_LOG_INFO("AscendIndexBinaryFloat runTopkCompute operation end.\n");
}

void AscendIndexBinaryFlatImpl::resetTopkCompOp()
{
    APP_LOG_INFO("AscendIndexBinaryFlat resetTopkCompOp operation started.\n");
    auto topkCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("TopkFlat");
        std::vector<int64_t> shape0{0, batch, BLOCK_SIZE};
        std::vector<int64_t> shape1{0, batch, BLOCK_SIZE / burstLen * 2}; // vmax and indice, so we multiply by 2
        // topk 算子用shape2第二维长度作为core num, opFlag的core num位全部置为1就开始计算
        std::vector<int64_t> shape2{0, 1, ACTUAL_NUM_SIZE};
        std::vector<int64_t> shape3{0, 1, FLAG_SIZE}; // topk 算子用第二维长度作为flag判断的offset, 设为1
        std::vector<int64_t> shape4{aicpu::TOPK_FLAT_ATTR_IDX_COUNT};
        std::vector<int64_t> shape5{batch, 0};

        desc.addInputTensorDesc(ACL_FLOAT16, shape0.size(), shape0.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, shape1.size(), shape1.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, shape2.size(), shape2.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, shape3.size(), shape3.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape4.size(), shape4.data(), ACL_FORMAT_ND);
 
        desc.addOutputTensorDesc(ACL_FLOAT16, shape5.size(), shape5.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_INT64, shape5.size(), shape5.data(), ACL_FORMAT_ND);
 
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : BATCH_SIZES) {
        topkComputeOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        FAISS_THROW_IF_NOT_MSG(topkCompOpReset(topkComputeOps[batch], batch), "topk op init failed");
    }

    APP_LOG_INFO("AscendIndexBinaryFlat resetTopkCompOp operation end.\n");
}

void AscendIndexBinaryFlatImpl::resetTopkFloatCompOp()
{
    APP_LOG_INFO("AscendIndexBinaryFloat resetTopkFloatCompOp operation started.\n");
    auto topkCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("TopkFlat");
        int burstLen = BURST_LEN_HIGH;
        auto curBurstsOfBlock = GetBurstsOfBlock(batch, BLOCK_SIZE, burstLen);

        std::vector<int64_t> shape0{ 0, batch, BLOCK_SIZE };
        std::vector<int64_t> shape1{ 0, batch, curBurstsOfBlock };
        std::vector<int64_t> shape2{ 0, CORE_NUM, SIZE_ALIGN };
        std::vector<int64_t> shape3{ 0, CORE_NUM, FLAG_SIZE };
        std::vector<int64_t> shape4{ aicpu::TOPK_FLAT_ATTR_IDX_COUNT };
        std::vector<int64_t> shape5{ batch, 0 };

        desc.addInputTensorDesc(ACL_FLOAT16, shape0.size(), shape0.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, shape1.size(), shape1.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, shape2.size(), shape2.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, shape3.size(), shape3.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape4.size(), shape4.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT16, shape5.size(), shape5.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_INT64, shape5.size(), shape5.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : BATCH_SIZES) {
        topkComputeUint8Ops[batch] = std::unique_ptr<AscendOperator>(nullptr);
        ASCEND_THROW_IF_NOT_MSG(topkCompOpReset(topkComputeUint8Ops[batch], batch), "topk op init failed");
    }

    APP_LOG_INFO("AscendIndexBinaryFloat resetTopkFloatCompOp operation end.\n");
}

void AscendIndexBinaryFlatImpl::resetInner()
{
    APP_LOG_INFO("AscendIndexBinaryFlat reset operation started.\n");
    ids.clear();
    label2IdxMap.clear();

    int dvSize = utils::divUp((int)this->ntotal, BLOCK_SIZE);
    for (int i = 0; i < dvSize; ++i) {
        baseShaped.at(i)->clear();
    }

    this->ntotal = 0;
    APP_LOG_INFO("AscendIndexBinaryFlat reset operation end.\n");
}

void AscendIndexBinaryFlatImpl::reset()
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    resetInner();
}

void AscendIndexBinaryFlatImpl::copyFrom(const faiss::IndexBinaryFlat *index)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndexBinaryFlat copyFrom operation started.\n");
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "Invalid index nullptr.");
    FAISS_THROW_IF_NOT_MSG(std::find(DIMS.begin(), DIMS.end(), index->d) != DIMS.end(),
                           "Unsupported dims, should be in { 256, 512, 1024 }.");
    FAISS_THROW_IF_NOT_FMT(index->code_size == index->d / BINARY_BYTE_SIZE, "code_size[%d] must be equal [%d].",
        index->code_size, index->d / BINARY_BYTE_SIZE);
    FAISS_THROW_IF_NOT_FMT(index->ntotal >= 0 && static_cast<size_t>(index->ntotal) <= MAX_N,
                           "ntotal must be >= 0 and <= %ld", MAX_N);

    resetInner();
    this->d = index->d;
    this->code_size = index->code_size;
    if (index->d == LARGE_DIM) {
        largeDimSetting();
    }

    // copy cpu index's codes and preCompute to ascend index
    copyCode(index);
    APP_LOG_INFO("AscendIndexBinaryFlat copyFrom operation finished.\n");
}
void AscendIndexBinaryFlatImpl::copyFrom(const faiss::IndexBinaryIDMap *index)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndexBinaryFlat copyFrom operation started.\n");
    FAISS_THROW_IF_NOT_MSG(index != nullptr && index->index != nullptr, "Invalid index nullptr.");
    // copy cpu index's codes and preCompute to ascend index
    auto binaryPtr = dynamic_cast<const faiss::IndexBinaryFlat *>(index->index);
    FAISS_THROW_IF_NOT_MSG(binaryPtr != nullptr, "Invalid sqIndex nullptr.");
    FAISS_THROW_IF_NOT_MSG(std::find(DIMS.begin(), DIMS.end(), binaryPtr->d) != DIMS.end(),
                           "Unsupported dims, should be in { 256, 512, 1024 }.");
    FAISS_THROW_IF_NOT_FMT(binaryPtr->code_size == binaryPtr->d / BINARY_BYTE_SIZE, "code_size[%d] must be equal [%d].",
        binaryPtr->code_size, binaryPtr->d / BINARY_BYTE_SIZE);
    FAISS_THROW_IF_NOT_FMT(binaryPtr->ntotal >= 0 && static_cast<size_t>(binaryPtr->ntotal) <= MAX_N,
                           "ntotal must be >= 0 and <= %ld", MAX_N);

    resetInner();
    this->d = binaryPtr->d;
    this->code_size = binaryPtr->code_size;
    if (binaryPtr->d == LARGE_DIM) {
        largeDimSetting();
    }

    if (index->id_map.data() != nullptr) {
        FAISS_THROW_IF_NOT_MSG(index->id_map.size() == static_cast<size_t>(binaryPtr->ntotal),
                               "The size of id_map must be equal to ntotal.\n");
    }
    copyCode(binaryPtr, index->id_map.data());
    APP_LOG_INFO("AscendIndexBinaryFlat copyFrom operation finished.\n");
}

void AscendIndexBinaryFlatImpl::copyCode(const faiss::IndexBinaryFlat *index, const idx_t *xids)
{
    APP_LOG_INFO("AscendIndexBinaryFlatImpl copyCode operation started.\n");
    if (index->xb.size() == 0) {
        return;
    }
    FAISS_THROW_IF_NOT_FMT(index->xb.size()
                               == static_cast<size_t>(index->ntotal) * static_cast<size_t>(this->code_size),
                           "Invalid index->xb.size is %zu, expected is %zu.\n", index->xb.size(),
                           static_cast<size_t>(index->ntotal) * static_cast<size_t>(this->code_size));
    add_with_ids_inner(index->ntotal, index->xb.data(), xids);
    APP_LOG_INFO("AscendIndexBinaryFlatImpl copyCode operation finished.\n");
}

void AscendIndexBinaryFlatImpl::copyTo(faiss::IndexBinaryFlat *index) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    APP_LOG_INFO("AscendIndexBinaryFlat copyTo operation started.\n");

    FAISS_THROW_IF_NOT(index != nullptr);
    index->reset();
    index->metric_type = this->metric_type;
    index->d = this->d;
    index->ntotal = this->ntotal;
    index->is_trained = this->is_trained;
    index->code_size = this->code_size;

    if (this->ntotal > 0) {
        std::vector<uint8_t> codes(this->ntotal * this->code_size);
        getVectors(0, this->ntotal, codes);
        index->xb = std::move(codes);
    }
    APP_LOG_INFO("AscendIndexBinaryFlat copyTo operation finished.\n");
}

void AscendIndexBinaryFlatImpl::getVectors(uint32_t offset, uint32_t num, std::vector<uint8_t> &xb) const
{
    APP_LOG_INFO("AscendIndexBinaryFlat getVectors operation started.\n");

    FAISS_THROW_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE);

    FAISS_THROW_IF_NOT_FMT(offset + num <= this->ntotal, "offset + num must be <= (%zu).", this->ntotal);

    if (num * static_cast<size_t>(this->code_size) > xb.size()) {
        xb.resize(num * static_cast<size_t>(this->code_size));
    }

    std::string opName = "TransdataRaw";
    auto &mem = pResources->getMemoryManager();
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();

    int blockNum = utils::divUp(this->ntotal, BLOCK_SIZE);
    AscendTensor<int64_t, DIMS_2> attrs(mem, {blockNum, aicpu::TRANSDATA_RAW_ATTR_IDX_COUNT}, stream);
    int reshapeDim1 = utils::divUp(BLOCK_SIZE, zRegionHeight);
    int reshapeDim2 = this->d / CUBE_ALIGN;
    int align1 = zRegionHeight;
    int align2 = HAMMING_CUBE_ALIGN;
    if (isUsedFloat) {
        SetShapeDim(reshapeDim1, reshapeDim2, align1, align2);
    }
    size_t blockSize = static_cast<size_t>(BLOCK_SIZE);

    for (size_t i = 0; i < num;) {
        size_t total = i + offset;
        size_t offsetInBlock = total % blockSize;
        size_t leftInBlock = blockSize - offsetInBlock;
        size_t leftInData = num - i;
        size_t copyCount = std::min(leftInBlock, leftInData);
        size_t blockIdx = total / blockSize;
        int copy = static_cast<int>(copyCount);

        AscendTensor<uint8_t, DIMS_2> dst(mem, {copy, this->code_size}, stream);
        AscendTensor<uint8_t, DIMS_4> src(baseShaped[blockIdx]->data(),
                                          {reshapeDim1, reshapeDim2, align1, align2});

        AscendTensor<int64_t, DIMS_1> attr = attrs[blockIdx].view();
        attr[aicpu::TRANSDATA_RAW_ATTR_OFFSET_IDX] = offsetInBlock;

        LaunchOpTwoInOneOut<uint8_t, DIMS_4, ACL_UINT8, int64_t, DIMS_1, ACL_INT64, uint8_t, DIMS_2, ACL_UINT8>(
            opName, stream, src, attr, dst);

        auto ret = synchronizeStream(stream);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "synchronizeStream getVectors stream failed: %i\n", ret);

        ret = aclrtMemcpy(xb.data() + i * this->code_size, copy * this->code_size, dst.data(), dst.getSizeInBytes(),
                          ACL_MEMCPY_DEVICE_TO_HOST);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Mem operator error %d", (int)ret);
        i += copyCount;
    }
    APP_LOG_INFO("AscendIndexBinaryFlat getVectors operation end..\n");
}

void AscendIndexBinaryFlatImpl::copyTo(faiss::IndexBinaryIDMap *index) const
{
    auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
    APP_LOG_INFO("AscendIndexInt8Flat copyTo operation started.\n");
    FAISS_THROW_IF_NOT_MSG(index != nullptr, "Invalid index nullptr.");

    copyTo(dynamic_cast<faiss::IndexBinaryFlat *>(index->index));

    index->id_map = std::move(ids);
    APP_LOG_INFO("AscendIndexInt8Flat copyTo operation finished.\n");
}

size_t AscendIndexBinaryFlatImpl::remove_ids(const faiss::IDSelector &sel)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
    APP_LOG_INFO("AscendIndexBinaryFlat remove_ids operation start.\n");
    FAISS_THROW_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE);

    std::vector<idx_t> removeIds;
    if (isRemoveFast) {
        if (auto rangeSel = dynamic_cast<const IDSelectorBatch *>(&sel)) {
            removeBatch(rangeSel, removeIds);
        } else if (auto rangeSel = dynamic_cast<const IDSelectorRange *>(&sel)) {
            removeRange(rangeSel, removeIds);
        } else {
            APP_LOG_WARNING("Invalid IDSelector.\n");
            return 0;
        }
    } else {
        for (idx_t pos = this->ntotal - 1; pos >= 0; --pos) {
            if (sel.is_member(ids[pos])) {
                removeIds.emplace_back(pos);
            }
        }
    }

    if (removeIds.empty()) {
        return 0;
    }
    removeIdx(removeIds);

    APP_LOG_INFO("AscendIndexBinaryFlat remove_ids operation finished.\n");
    return removeIdsImpl(removeIds);
}

void AscendIndexBinaryFlatImpl::removeBatch(const IDSelectorBatch *rangeSel, std::vector<idx_t> &removeIds)
{
    size_t removeSize = rangeSel->set.size();
    FAISS_THROW_IF_NOT_FMT(removeSize <= static_cast<size_t>(ntotal),
        "the size of removed codes should be in range [0, %ld], actual=%zu.", ntotal, removeSize);
    std::vector<idx_t> removeBatch(removeSize);
    transform(begin(rangeSel->set), end(rangeSel->set), begin(removeBatch),
        [](idx_t temp) { return static_cast<idx_t>(temp); });

    for (auto delLabel : removeBatch) {
        removeSingle(removeIds, delLabel);
    }
}

void AscendIndexBinaryFlatImpl::removeRange(const IDSelectorRange *rangeSel, std::vector<idx_t> &removeIds)
{
    for (auto delLabel = rangeSel->imin; delLabel < rangeSel->imax; ++delLabel) {
        removeSingle(removeIds, delLabel);
    }
}

void AscendIndexBinaryFlatImpl::removeIdx(const std::vector<idx_t> &removeIds)
{
    APP_LOG_INFO("AscendIndexBinaryFlat removeIdx operation started.\n");

    int lastIdx = this->ntotal - 1;
    for (auto pos : removeIds) {
        int64_t delLabel = ids[pos];
        ids[pos] = ids[lastIdx];
        if (isRemoveFast) {
            label2IdxMap[ids[lastIdx]] = pos;
            label2IdxMap.erase(delLabel);
        }
        --lastIdx;
    }
    ids.resize(lastIdx + 1);

    APP_LOG_INFO("AscendIndexBinaryFlat removeIdx operation finished.\n");
}

void AscendIndexBinaryFlatImpl::removeSingle(std::vector<idx_t> &removes, idx_t delLabel)
{
    APP_LOG_INFO("AscendIndexBinaryFlatImpl removeSingle operation started.\n");

    auto it = label2IdxMap.find(delLabel);
    if (it != label2IdxMap.end()) {
        removes.emplace_back(it->second);
    }

    APP_LOG_INFO("AscendIndexBinaryFlatImpl removeSingle operation finished.\n");
}

void AscendIndexBinaryFlatImpl::removeInvalidData(int oldTotal, int remove)
{
    APP_LOG_INFO("AscendIndexBinaryFlat removeInvalidData operation start. \n");
    FAISS_THROW_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE);
    int oldVecSize = utils::divUp(oldTotal, BLOCK_SIZE);
    int vecSize = utils::divUp(oldTotal - remove, BLOCK_SIZE);

    for (int i = oldVecSize - 1; i >= vecSize; --i) {
        baseShaped.at(i)->clear();
    }
    baseShaped.resize(vecSize);
    APP_LOG_INFO("AscendIndexBinaryFlat removeInvalidData operation finished. \n");
}

void AscendIndexBinaryFlatImpl::moveShapedForward(idx_t srcIdx, idx_t dstIdx)
{
    APP_LOG_INFO("AscendIndexBinaryFlat moveShapedForward operation start. \n");
    FAISS_THROW_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE);

    ASCEND_THROW_IF_NOT(srcIdx >= dstIdx);
    int srcIdx1 = (int)srcIdx / BLOCK_SIZE;
    int srcIdx2 = (int)srcIdx % BLOCK_SIZE;

    int dstIdx1 = (int)dstIdx / BLOCK_SIZE;
    int dstIdx2 = (int)dstIdx % BLOCK_SIZE;

    int dim2 = utils::divUp(this->d, CUBE_ALIGN);

    uint8_t *srcDataPtr = baseShaped[srcIdx1]->data()
        + (srcIdx2 / zRegionHeight) * (dim2 * zRegionHeight * HAMMING_CUBE_ALIGN)
        + (srcIdx2 % zRegionHeight) * HAMMING_CUBE_ALIGN;
    uint8_t *dstDataPtr = baseShaped[dstIdx1]->data()
        + (dstIdx2 / zRegionHeight) * (dim2 * zRegionHeight * HAMMING_CUBE_ALIGN)
        + (dstIdx2 % zRegionHeight) * HAMMING_CUBE_ALIGN;

    for (int i = 0; i < dim2; i++) {
        auto err =
            aclrtMemcpy(dstDataPtr, HAMMING_CUBE_ALIGN, srcDataPtr, HAMMING_CUBE_ALIGN, ACL_MEMCPY_DEVICE_TO_DEVICE);
        ASCEND_THROW_IF_NOT_FMT(err == EOK, "Mem error %d", (int)err);
        dstDataPtr += zRegionHeight * HAMMING_CUBE_ALIGN;
        srcDataPtr += zRegionHeight * HAMMING_CUBE_ALIGN;
    }
    APP_LOG_INFO("AscendIndexBinaryFlat moveShapedForward operation finished. \n");
}

size_t AscendIndexBinaryFlatImpl::removeIdsImpl(const std::vector<idx_t> &indices)
{
    APP_LOG_INFO("AscendIndexBinaryFlat removeIdsImpl operation start. \n");
    // move the end data to the locate of delete data
    int dimAlignSize = utils::divUp(this->d, CUBE_ALIGN);
    int removedCnt = static_cast<int>(indices.size());
    std::vector<uint64_t> srcAddr(removedCnt);
    std::vector<uint64_t> dstAddr(removedCnt);
    std::string opName = "RemovedataShaped";
    auto &mem = pResources->getMemoryManager();
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    int align1 = zRegionHeight;
    int align2 = HAMMING_CUBE_ALIGN;
    if (isUsedFloat) {
        dimAlignSize = this->code_size / CUBE_ALIGN;
        align1 = CUBE_ALIGN;
        align2 = CUBE_ALIGN;
    }
    for (int i = 0; i < removedCnt; i++) {
        int srcIdx = static_cast<int>(this->ntotal - i - 1);
        int srcIdx1 = srcIdx / BLOCK_SIZE;
        int srcIdx2 = srcIdx % BLOCK_SIZE;
        int dstIdx = indices[i];
        int dstIdx1 = dstIdx / BLOCK_SIZE;
        int dstIdx2 = dstIdx % BLOCK_SIZE;

        uint8_t *srcDataPtr = baseShaped[srcIdx1]->data() + (srcIdx2 / align1) *
            (dimAlignSize * align1 * align2) + (srcIdx2 % align1) * align2;
        uint8_t *dstDataPtr = baseShaped[dstIdx1]->data() + (dstIdx2 / align1) *
            (dimAlignSize * align1 * align2) + (dstIdx2 % align1) * align2;
        srcAddr[i] = reinterpret_cast<uint64_t>(srcDataPtr);
        dstAddr[i] = reinterpret_cast<uint64_t>(dstDataPtr);
    }

    AscendTensor<uint64_t, DIMS_1> srcInput(mem, { removedCnt }, stream);
    AscendTensor<uint64_t, DIMS_1> dstInput(mem, { removedCnt }, stream);
    auto ret = aclrtMemcpy(srcInput.data(), srcInput.getSizeInBytes(), srcAddr.data(),
        srcAddr.size() * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to copy to device (error %d)", (int)ret);
    ret = aclrtMemcpy(dstInput.data(), dstInput.getSizeInBytes(), dstAddr.data(), dstAddr.size() * sizeof(uint64_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to copy to device (error %d)", (int)ret);
    AscendTensor<int64_t, DIMS_1> attrsInput(mem, { aicpu::REMOVEDATA_SHAPED_ATTR_IDX_COUNT }, stream);
    setRemoveAttr(attrsInput, dimAlignSize, align1, align2);
    LaunchOpTwoInOneOut<uint64_t, DIMS_1, ACL_UINT64, int64_t, DIMS_1, ACL_INT64, uint64_t, DIMS_1, ACL_UINT64>(opName,
        stream, srcInput, attrsInput, dstInput);

    ret = synchronizeStream(stream);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to acl Synchronize Streame (error %d)", (int)ret);
    // release the space of unusage
    removeInvalidData(this->ntotal, removedCnt);
    this->ntotal -= removedCnt;
    APP_LOG_INFO("AscendIndexBinaryFlat removeIdsImpl operation finished. \n");
    return removedCnt;
}

void AscendIndexBinaryFlatImpl::setRemoveAttr(AscendTensor<int64_t, DIMS_1> &attrsInput, int dimAlignSize,
    int align1, int align2) const
{
    APP_LOG_INFO("AscendIndexBinaryFlat setRemoveAttr operation start. \n");
    std::vector<int64_t> attrs(aicpu::REMOVEDATA_SHAPED_ATTR_IDX_COUNT);
    attrs[aicpu::REMOVEDATA_SHAPED_ATTR_DATA_TYPE] = faiss::ascend::UINT8;
    attrs[aicpu::REMOVEDATA_SHAPED_ATTR_ZREGION_HEIGHT] = align1;
    attrs[aicpu::REMOVEDATA_SHAPED_ATTR_DIM_ALIGN_NUM] = dimAlignSize;
    attrs[aicpu::REMOVEDATA_SHAPED_ATTR_CUBE_ALIGN] = align2;
    auto ret = aclrtMemcpy(attrsInput.data(), attrsInput.getSizeInBytes(), attrs.data(), attrs.size() * sizeof(int64_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to copy to device (error %d)", ret);
    APP_LOG_INFO("AscendIndexBinaryFlat setRemoveAttr operation finished. \n");
}

} // namespace ascend
} // namespace faiss
