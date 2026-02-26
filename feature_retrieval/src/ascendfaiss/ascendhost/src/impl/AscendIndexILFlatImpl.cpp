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

 
#include "ascendhost/include/impl/AscendIndexILFlatImpl.h"
 
#include <faiss/impl/FaissAssert.h>
#include "threadpool/AscendThreadPool.h"
#include "ascenddaemon/utils/StaticUtils.h"
#include "ascenddaemon/utils/AscendUtils.h"
#include "ascenddaemon/utils/AscendRWLock.h"
#include "ascenddaemon/utils/Limits.h"
#include "common/utils/OpLauncher.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"
#include "ascend/utils/fp16.h"
using namespace ascend;
 
namespace faiss {
namespace ascend {
std::vector<int> ASCEND_ILFLAT_DIMS = {32, 64, 128, 256, 384, 512};
std::vector<int> ASCEND_ILFLAT_SEARCH_BATCHES = {64, 48, 32, 30, 18, 16, 8, 6, 4, 2, 1};
std::vector<int> ASCEND_ILFLAT_COMPUTE_BATCHES = {256, 128, 64, 48, 32, 30, 18, 16, 8, 6, 4, 2, 1};
std::vector<int> ASCEND_ILFLAT_COMPUTE_IDX_BATCHES = {256, 128, 64, 48, 32, 16, 8, 6, 4, 2, 1};
 
const int ZREGION_HEIGHT = 2;
const int IDX_BURST_LEN = 64;
const int TABLE_LEN = 10048; // mapping table with redundancy of 48
const int FLAT_BLOCK_SIZE = 16384 * 16;
const int FLAT_COMPUTE_PAGE = FLAT_BLOCK_SIZE * 16;
const int LARGE_BATCH_BURST_LEN = 32;
const int LARGE_BATCH_THRESHOLD = 64;
const int FLAG_SIZE = 16;
const int CORE_NUM = faiss::ascend::SocUtils::GetInstance().GetCoreNum();
AscendIndexILFlatImpl::AscendIndexILFlatImpl(int dim, int capacity, int deviceId, int64_t resourceSize)
    : dim(dim), capacity(capacity), deviceId(deviceId), resourceSize(resourceSize), isInitialized(false) {}
 
AscendIndexILFlatImpl::~AscendIndexILFlatImpl() {}
 
APP_ERROR AscendIndexILFlatImpl::Init()
{
    APP_LOG_INFO("AscendIndexILFlatImpl Init operation started.\n");
    APPERR_RETURN_IF_NOT_FMT(std::find(ASCEND_ILFLAT_DIMS.begin(), ASCEND_ILFLAT_DIMS.end(),
        this->dim) != ASCEND_ILFLAT_DIMS.end(), APP_ERR_INVALID_PARAM,
        "Illegal dim(%d), given dim should be in {32, 64, 128, 256, 384, 512}.", this->dim);
 
    APPERR_RETURN_IF_NOT_FMT(static_cast<size_t>(capacity) <= GetMaxCapacity(),
        APP_ERR_INVALID_PARAM,
        "Illegal capacity for dim %d, the upper boundary of capacity is: %zu, ", dim, GetMaxCapacity());
 
    auto ret = aclrtSetDevice(deviceId);
    APP_LOG_INFO("AscendIndexILFlatImpl Init on device(%d).\n", deviceId);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_ACL_SET_DEVICE_FAILED, "failed to set device(%d)", ret);
    this->ntotal = 0;
    this->pageSize = FLAT_COMPUTE_PAGE;
    this->computeBatchSizes = ASCEND_ILFLAT_COMPUTE_BATCHES;
    this->searchBatchSizes = ASCEND_ILFLAT_SEARCH_BATCHES;
    this->computeByIdxBatchSizes = ASCEND_ILFLAT_COMPUTE_IDX_BATCHES;
    this->blockSize = FLAT_BLOCK_SIZE;
    this->fakeBaseSizeInBytes =
        static_cast<size_t>(FAKE_HUGE_BASE) * static_cast<size_t>(this->dim) * sizeof(float16_t);
    this->blockMaskSize = utils::divUp(this->blockSize, SIZE_ALIGN);
    pResources = CREATE_UNIQUE_PTR(AscendResourcesProxy);
    pResources->setTempMemory(resourceSize);
    pResources->initialize();

    ret = InitOps();
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "InitOps init failed !!!");
 
    size_t blockAlign = static_cast<size_t>(utils::divUp(capacity, this->blockSize));
    size_t blockSizeAlign = static_cast<size_t>(utils::divUp(this->blockSize, CUBE_ALIGN));
    int dimAlign = utils::divUp(this->dim, CUBE_ALIGN);
    try {
        auto paddedSize = blockAlign * blockSizeAlign * static_cast<size_t>(dimAlign * CUBE_ALIGN * CUBE_ALIGN);
        this->baseSpace = CREATE_UNIQUE_PTR(DeviceVector<float16_t>, MemorySpace::DEVICE_HUGEPAGE);
        this->baseSpace->resize(paddedSize, true);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_NOT_FMT(
            false, APP_ERR_ACL_BAD_ALLOC, "Malloc device base memory failed. , error msg[%s]", e.what());
    }
    isInitialized = true;
 
    APP_LOG_INFO("AscendIndexILFlatImpl Init operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR AscendIndexILFlatImpl::InitOps()
{
    auto ret = ResetDistCompIdxWithTableOp();
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
        "ResetDistCompIdxWithTableOp init failed !!!");
    ret = ResetDistCompIdxOp();
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
        "ResetDistCompIdxOp init failed !!!");
    ret = ResetTopkCompOp();
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
        "ResetTopkCompOp init failed !!!");
    ret = ResetDistCompOp(FLAT_BLOCK_SIZE);
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
        "ResetDistCompOp init failed !!!");
    ret = ResetDistanceWithTableOp();
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
        "ResetDistanceWithTableOp init failed !!!");
    ret = ResetDistanceFlatIpOp();
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
        "ResetDistanceFlatIpOp init failed !!!");
    return APP_ERR_OK;
}

void AscendIndexILFlatImpl::Finalize()
{
    if (isInitialized) {
        baseSpace->clear();
        isInitialized = false;
    }
}

APP_ERROR AscendIndexILFlatImpl::CheckComputeMaxNum(int n, const int *num, const idx_t *indices, int& maxNum)
{
    for (int i = 0; i < n; i++) {
        int tmpNum = *(num + i);
        APPERR_RETURN_IF_NOT_LOG(tmpNum >= 0 && tmpNum <= this->ntotal, APP_ERR_INVALID_PARAM,
            "The num of query idx is invalid, it should be in range [0, ntotal]. ");
        maxNum = std::max(maxNum, tmpNum);
    }
    if (maxNum == 0) {
        return APP_ERR_OK;
    }

    if (!isInputDevice) {
        for (int i = 0; i < n; i++) {
            int tmpNum = *(num + i);
            for (int j = 0; j < tmpNum; j++) {
                APPERR_RETURN_IF_NOT_LOG(*(indices + i * maxNum + j) < static_cast<size_t>(this->ntotal),
                    APP_ERR_INVALID_PARAM, "The given indice to compare with should be smaller than ntotal");
            }
        }
    }

    return APP_ERR_OK;
}

size_t AscendIndexILFlatImpl::GetMaxCapacity() const
{
    size_t vectorSize = static_cast<size_t>(dim) * sizeof(float16_t);
    return MAX_BASE_SPACE / vectorSize;
}
 
APP_ERROR AscendIndexILFlatImpl::AddPageFp16(int num, const float16_t *featuresFloat, idx_t indice)
{
    APP_LOG_INFO("AscendIndexILFlatImpl Add operation start.\n");
 
    std::string opName = "TransdataShaped";
    auto &mem = pResources->getMemoryManager();
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
 
    AscendTensor<float16_t, DIMS_2> baseData(mem, {num, this->dim}, stream);
    auto ret = aclrtMemcpy(baseData.data(),
        baseData.getSizeInBytes(), featuresFloat,
        static_cast<uint64_t>(num) * static_cast<uint64_t>(this->dim) * sizeof(float16_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy feature to device error(%d)", ret);
 
    int dimAlign = utils::divUp(this->dim, CUBE_ALIGN);
    int blockAlign = utils::divUp(capacity, this->blockSize);
    int blockSizeAlign = utils::divUp(this->blockSize, ZREGION_HEIGHT);
    int64_t paddedSize = blockAlign * blockSizeAlign * dimAlign * CUBE_ALIGN * ZREGION_HEIGHT;
    APPERR_RETURN_IF_NOT_LOG(paddedSize != 0, APP_ERR_INNER_ERROR, "paddedSize should not be zero.");
    int offsetInBlock = static_cast<int64_t>(indice) % paddedSize;
 
    AscendTensor<int64_t, DIMS_1> attr(mem, {aicpu::TRANSDATA_SHAPED_ATTR_IDX_COUNT}, stream);
    AscendTensor<float16_t, DIMS_2> src(baseData.data(), {num, this->dim});
    AscendTensor<float16_t, DIMS_4> dst(this->baseSpace->data(),
        {utils::divUp(capacity, ZREGION_HEIGHT), utils::divUp(this->dim, CUBE_ALIGN),
         ZREGION_HEIGHT, CUBE_ALIGN});
    attr[aicpu::TRANSDATA_SHAPED_ATTR_NTOTAL_IDX] = offsetInBlock;
 
    LaunchOpTwoInOneOut<float16_t, DIMS_2, ACL_FLOAT16, int64_t, DIMS_1, ACL_INT64, float16_t, DIMS_4, ACL_FLOAT16>(
        opName, stream, src, attr, dst);
 
    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(
        ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream addVector stream failed: %i\n", ret);
    this->ntotal += num;
    APP_LOG_INFO("AscendIndexILFlatImpl Add operation end.\n");
    return APP_ERR_OK;
}
 
APP_ERROR AscendIndexILFlatImpl::AddPageFp16(int num, const float *featuresFloat, idx_t indice)
{
    APP_LOG_INFO("AscendIndexILFlatImpl Add operation start.\n");
    uint64_t featureSize = static_cast<uint64_t>(num) * static_cast<uint64_t>(this->dim);
    std::vector<float16_t> features(featureSize);
    std::transform(
        featuresFloat, featuresFloat + featureSize, features.data(), [](float temp) { return fp16(temp).data; });
    
    return AddPageFp16(num, features.data(), indice);
    APP_LOG_INFO("AscendIndexILFlatImpl Add operation end.\n");
}

APP_ERROR AscendIndexILFlatImpl::Update(int n, const float16_t *features, const idx_t *indices)
{
    APPERR_RETURN_IF_NOT_LOG(
        (isInitialized), APP_ERR_INVALID_PARAM, "Illegal operation, please initialize the index first. ");
    APPERR_RETURN_IF_NOT_FMT((n) > 0 && (n) <= this->capacity,
        APP_ERR_INVALID_PARAM,
        "The number n should be in range (0, %d]",
        this->capacity);
    APPERR_RETURN_IF_NOT_LOG(features, APP_ERR_INVALID_PARAM, "Features can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(indices, APP_ERR_INVALID_PARAM, "indices can not be nullptr.");
    return UpdateFp16(n, features, indices);
}

APP_ERROR AscendIndexILFlatImpl::UpdateFp16(int n, const float16_t *features, const idx_t *indices)
{
    auto ret = aclrtSetDevice(deviceId);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_ACL_SET_DEVICE_FAILED, "failed to set device(%d)", ret);
    std::vector<std::future<void>> functorRet;
    AscendThreadPool pool(MAX_THREAD_NUM);
    idx_t maxIndice = 0;
    auto updateData = [this, &maxIndice](const idx_t *indices, const float16_t *features, size_t i,
        size_t dimAlign) {
        auto ret = aclrtSetDevice(deviceId);
        ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "failed to set device(%d)", ret);
        auto seq = *(indices + i);
        maxIndice = std::max(maxIndice, seq);
        ASCEND_THROW_IF_NOT_FMT(seq < static_cast<size_t>(this->capacity),
            "Invalid feature to get, the indice %u should be smaller than capacity\n", seq);
        float16_t *dataPtr = this->baseSpace->data() + seq / ZREGION_HEIGHT * dimAlign * (ZREGION_HEIGHT * CUBE_ALIGN) +
            seq % ZREGION_HEIGHT * CUBE_ALIGN;

        for (size_t j = 0; j < dimAlign; j++) {
            size_t getOffset = i * static_cast<size_t>(this->dim) + j * CUBE_ALIGN;
            size_t cpyNum = (j == dimAlign - 1) ? (static_cast<size_t>(this->dim) - j * CUBE_ALIGN) : CUBE_ALIGN;
            ret = aclrtMemcpy(dataPtr + j * ZREGION_HEIGHT * CUBE_ALIGN, cpyNum * sizeof(float16_t),
                features + getOffset, cpyNum * sizeof(float16_t),
                ACL_MEMCPY_HOST_TO_DEVICE);
            ASCEND_THROW_IF_NOT_FMT(ret == EOK, "aclrtMemcpy error, err=%d\n", ret);
        }
    };
    size_t dimAlign = static_cast<size_t>(utils::divUp(this->dim, CUBE_ALIGN));
    for (size_t i = 0; i < static_cast<size_t>(n); i++) {
        functorRet.emplace_back(pool.Enqueue(updateData, indices, features, i, dimAlign));
    }

    uint32_t seartchWait = 0;
    try {
        for (std::future<void> &ret : functorRet) {
            seartchWait++;
            ret.get();
        }
    } catch (std::exception &e) {
        for_each(functorRet.begin() + seartchWait, functorRet.end(), [](std::future<void> &ret) { ret.wait(); });
        APPERR_RETURN_IF_NOT_FMT(
            false, APP_ERR_INNER_ERROR, "for_each fail, error msg[%s]", e.what());
    }
    this->ntotal = std::max(this->ntotal, static_cast<int>(maxIndice) + 1);
    return APP_ERR_OK;
}


APP_ERROR AscendIndexILFlatImpl::Update(int n, const float *features, const idx_t *indices)
{
    APPERR_RETURN_IF_NOT_LOG(
        (isInitialized), APP_ERR_INVALID_PARAM, "Illegal operation, please initialize the index first. ");
    APPERR_RETURN_IF_NOT_FMT((n) > 0 && (n) <= this->capacity,
        APP_ERR_INVALID_PARAM,
        "The number n should be in range (0, %d]",
        this->capacity);
    APPERR_RETURN_IF_NOT_LOG(features, APP_ERR_INVALID_PARAM, "Features can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(indices, APP_ERR_INVALID_PARAM, "indices can not be nullptr.");
    size_t total = static_cast<size_t>(n) * static_cast<size_t>(this->dim);
    std::vector<float16_t> featuresData(total, 0);
    transform(features, features + total, begin(featuresData), [](float temp) { return fp16(temp).data; });
    return UpdateFp16(n, featuresData.data(), indices);
}

APP_ERROR AscendIndexILFlatImpl::TableMapping(int n, float *distances, size_t tableLen,
    const float *table, int topk)
{
    APP_LOG_INFO("AscendIndexILFlatImpl TableMapping operation start.\n");
    if (tableLen > 0) {
        int numEachQuery = 0;
        numEachQuery = std::min(topk, this->ntotal);
        int tableIndex = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < numEachQuery; ++j) {
                int offset = i * topk + j;
                // index = (cos + 1) / 2 * tableLen, 加上0.5，取整，实现四舍五入
                tableIndex = static_cast<int>((*(distances + offset) + 1) / 2 * tableLen + 0.5);
                APPERR_RETURN_IF_NOT_LOG(
                    tableIndex >= 0 && tableIndex < static_cast<int>(TABLE_LEN), APP_ERR_INVALID_TABLE_INDEX,
                    "Invalid index for table mapping, please ensure the correctness of vector normalization.\n");
                *(distances + offset) = *(table + tableIndex);
            }
        }
    }
    APP_LOG_INFO("AscendIndexILFlatImpl TableMapping operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR AscendIndexILFlatImpl::SearchImpl(int n, const float *query, int topk, idx_t *indices, float *distances)
{
    // convert query data from float to fp16, device use fp16 data to search
    std::vector<float16_t> queryData(n * this->dim, 0);
    transform(query, query + n * this->dim, begin(queryData), [](float temp) { return fp16(temp).data; });
 
    return searchImplFp16(n, queryData.data(), topk, indices, distances);
}
 
APP_ERROR AscendIndexILFlatImpl::SearchImpl(int n, const float16_t *query, int topk, idx_t *indices, float *distances)
{
    return searchImplFp16(n, query, topk, indices, distances);
}
 
APP_ERROR AscendIndexILFlatImpl::searchImplFp16(int n, const float16_t *query, int topk,
    idx_t *indices, float *distances)
{
    APP_LOG_INFO("AscendIndexILFlatImpl searchImplFp16 operation start.\n");
    auto streamPtr = this->pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = this->pResources->getMemoryManager();
 
    AscendTensor<float16_t, DIMS_2> queries(mem, {n, this->dim}, stream);
    auto ret = aclrtMemcpy(queries.data(), queries.getSizeInBytes(), query, n * this->dim * sizeof(float16_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy queryIdxArr to device fail(%d)!", ret);
 
    AscendTensor<idx_t, DIMS_2> outIndices(mem, { n, topk }, stream);
    AscendTensor<float16_t, DIMS_2> outDistances(mem, { n, topk }, stream);
 
    int pageNum = utils::divUp(this->ntotal, this->pageSize);
    for (int pageId = 0; pageId < pageNum; ++pageId) {
        APP_ERROR ret = SearchPaged(pageId, queries, pageNum, outIndices, outDistances);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR,
            "AscendIndexILFlatImpl SearchPaged failed(%d)", ret);
    }

    std::vector<float16_t> tmpDistances(n * topk);
    ret = aclrtMemcpy(tmpDistances.data(), tmpDistances.size() * sizeof(float16_t), outDistances.data(),
        outDistances.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy outDistances back to host(%d)", ret);
 
    std::transform(tmpDistances.begin(), tmpDistances.end(), distances,
        [](float16_t temp) { return static_cast<float>(fp16(temp)); });
 
    ret = aclrtMemcpy(indices, n * topk * sizeof(idx_t), outIndices.data(), outIndices.getSizeInBytes(),
        ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy outIndices back to host(%d)", ret);
    APP_LOG_INFO("AscendIndexILFlatImpl searchImplFp16 operation end.\n");
    return APP_ERR_OK;
}
 
APP_ERROR AscendIndexILFlatImpl::InitAttr(int topk, int burstLen, int blockNum, int pageId, int pageNum,
    AscendTensor<int64_t, DIMS_1>& attrsInput)
{
    std::vector<int64_t> attrs(aicpu::TOPK_FLAT_ATTR_IDX_COUNT);
    attrs[aicpu::TOPK_FLAT_ATTR_ASC_IDX] = 0;
    attrs[aicpu::TOPK_FLAT_ATTR_K_IDX] = topk;
    attrs[aicpu::TOPK_FLAT_ATTR_BURST_LEN_IDX] = burstLen;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_NUM_IDX] = blockNum;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_IDX] = pageId;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_NUM_IDX] = pageNum;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_SIZE_IDX] = this->pageSize;
    attrs[aicpu::TOPK_FLAT_ATTR_QUICK_HEAP] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_SIZE] = this->blockSize;
    auto ret = aclrtMemcpy(attrsInput.data(), attrsInput.getSizeInBytes(), attrs.data(), attrs.size() * sizeof(int64_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy attrsInput to device(%d)", ret);
    return APP_ERR_OK;
}
 
APP_ERROR AscendIndexILFlatImpl::SearchPaged(int pageId, AscendTensor<float16_t, DIMS_2> &queries, int pageNum,
    AscendTensor<idx_t, DIMS_2> &outIndices, AscendTensor<float16_t, DIMS_2> &outDistances)
{
    APP_LOG_INFO("AscendIndexILFlatImpl SearchPaged operation start.\n");
    auto streamPtr = this->pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = this->pResources->getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
 
    auto &mem = this->pResources->getMemoryManager();
    int batch = queries.getSize(0);
    int topk = outDistances.getSize(1);
    int pageOffset = pageId * this->pageSize;
    int blockOffset = pageOffset / this->blockSize;
    int computeNum = std::min(this->ntotal - pageOffset, this->pageSize);
    int blockNum = utils::divUp(computeNum, this->blockSize);
 
    auto burstLen = (batch >= LARGE_BATCH_THRESHOLD) ? LARGE_BATCH_BURST_LEN : BURST_LEN;
    // 乘以2，是和算子生成时的shape保持一致
    auto burstOfBlock = (FLAT_BLOCK_SIZE + burstLen - 1) / burstLen * 2;
 
    AscendTensor<float16_t, DIMS_3> distResult(mem, { blockNum, batch, this->blockSize }, stream);
    AscendTensor<float16_t, DIMS_3> maxDistResult(mem, { blockNum, batch, burstOfBlock }, stream);
    AscendTensor<uint32_t, DIMS_3> opSize(mem, { blockNum, CORE_NUM, SIZE_ALIGN }, stream);
    AscendTensor<uint16_t, DIMS_3> opFlag(mem, { blockNum, CORE_NUM, FLAG_SIZE }, stream);
    opFlag.zero();
 
    AscendTensor<uint8_t, DIMS_2> mask(mem, { batch, this->blockMaskSize }, stream);
 
    AscendTensor<int64_t, DIMS_1> attrsInput(mem, { aicpu::TOPK_FLAT_ATTR_IDX_COUNT }, stream);
    auto ret = InitAttr(topk, burstLen, blockNum, pageId, pageNum, attrsInput);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "InitAttr failed(%d)", ret);
    runTopkCompute(distResult, maxDistResult, opSize, opFlag, attrsInput, outDistances, outIndices, streamAicpu);
 
    const int dim1 = utils::divUp(this->blockSize, ZREGION_HEIGHT);
    const int dim2 = utils::divUp(this->dim, CUBE_ALIGN);

    for (int i = 0; i < blockNum; i++) {
        auto baseOffset = static_cast<int64_t>(blockOffset + i) * this->blockSize * dim2 * CUBE_ALIGN;
        AscendTensor<float16_t, DIMS_4> shaped(this->baseSpace->data() + baseOffset,
            { dim1, dim2, ZREGION_HEIGHT, CUBE_ALIGN });
        auto dist = distResult[i].view();
        auto maxDist = maxDistResult[i].view();
        auto actualSize = opSize[i].view();
        auto flag = opFlag[i].view();
 
        int offset = i * this->blockSize;
        auto actual = std::min(static_cast<uint32_t>(computeNum - offset), static_cast<uint32_t>(this->blockSize));
        actualSize[0][0] = actual;
        actualSize[0][3] = 0;  // 3 for IDX_USE_MASK, not use mask
 
        ComputeBlockDist(queries, mask, shaped, actualSize, dist, maxDist, flag, stream);
    }

    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream failed: %i\n", ret);

    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream aicpu failed: %i\n", ret);
    APP_LOG_INFO("AscendIndexILFlatImpl SearchPaged operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR AscendIndexILFlatImpl::ComputeDistByIdxImpl(int n, const float *queries, float *distances,
    std::tuple<int, const int *, const idx_t *> idxInfo, std::tuple<unsigned int, const float *> tableInfo)
{
    if (isInputDevice) {
        std::vector<float> queryDataFp32(n * this->dim, 0);
        auto ret = aclrtMemcpy(queryDataFp32.data(), queryDataFp32.size() * sizeof(float), queries,
            n * this->dim * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy queryDataFp32 fail(%d)!", ret);

        std::vector<float16_t> queryData(n * this->dim, 0);
        transform(queryDataFp32.data(), queryDataFp32.data() + n * this->dim, begin(queryData),
            [](float temp) { return fp16(temp).data; });

        AscendTensor<float16_t, DIMS_2> queryTensor({ n, this->dim });
        ret = aclrtMemcpy(queryTensor.data(), queryTensor.getSizeInBytes(), queryData.data(),
            n * this->dim * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR,
            "copy queryTensor to device fail(%d)!", ret);
        return ComputeDistByIdxImplFp16(n, queryTensor.data(), distances, idxInfo, tableInfo);
    }

    std::vector<float16_t> queryData(n * this->dim, 0);
    transform(queries, queries + n * this->dim, begin(queryData), [](float temp) { return fp16(temp).data; });

    AscendTensor<float16_t, DIMS_2> queryTensor({ n, this->dim });
    auto ret = aclrtMemcpy(queryTensor.data(), queryTensor.getSizeInBytes(), queryData.data(),
        n * this->dim * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy queryTensor to device fail(%d)!", ret);
    
    return ComputeDistByIdxImplFp16(n, queryTensor.data(), distances, idxInfo, tableInfo);
}

APP_ERROR AscendIndexILFlatImpl::ComputeDistByIdxImpl(int n, const float16_t *queries, float *distances,
    std::tuple<int, const int *, const idx_t *> idxInfo, std::tuple<unsigned int, const float *> tableInfo)
{
    if (isInputDevice) {
        return ComputeDistByIdxImplFp16(n, queries, distances, idxInfo, tableInfo);
    }

    AscendTensor<float16_t, DIMS_2> queryTensor({ n, this->dim });
    auto ret = aclrtMemcpy(queryTensor.data(), queryTensor.getSizeInBytes(), queries, n * this->dim * sizeof(float16_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy queryTensor to device fail(%d)!", ret);
    return ComputeDistByIdxImplFp16(n, queryTensor.data(), distances, idxInfo, tableInfo);
}

APP_ERROR AscendIndexILFlatImpl::MoveData(int& idxCopyNum, int maxNum, int n, const idx_t *indice,
    AscendTensor<idx_t, DIMS_3>& idxTensor)
{
    int idxSliceNum = utils::divUp(maxNum, IDX_BURST_LEN);
    for (int i = 0; i < idxSliceNum; i++) {
        for (int j = 0; j < n; j++) {
            idxCopyNum = (i == idxSliceNum - 1) ? (maxNum - i * IDX_BURST_LEN) : IDX_BURST_LEN;
            auto err = aclrtMemcpy(idxTensor[i][j].data(), idxCopyNum * sizeof(idx_t),
                indice + static_cast<size_t>(j) * maxNum + i * IDX_BURST_LEN, idxCopyNum * sizeof(idx_t),
                ACL_MEMCPY_HOST_TO_DEVICE);
            APPERR_RETURN_IF_NOT_FMT((err == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy idxTensor fail(%d)!", err);
        }
    }
    return APP_ERR_OK;
}

APP_ERROR AscendIndexILFlatImpl::CopyDistData(int maxNum, int idxCopyNum, float *distances, int n,
    AscendTensor<float, DIMS_3>& distResult)
{
    int idxSliceNum = utils::divUp(maxNum, IDX_BURST_LEN);
    for (int i = 0; i < idxSliceNum; i++) {
        for (int j = 0; j < n; j++) {
            idxCopyNum = (i == idxSliceNum - 1) ? (maxNum - i * IDX_BURST_LEN) : IDX_BURST_LEN;
            auto err = aclrtMemcpy(distances + i * IDX_BURST_LEN + j * maxNum, idxCopyNum * sizeof(float),
                distResult[i][j].data(), idxCopyNum * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            APPERR_RETURN_IF_NOT_FMT(err == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                "memcpy error, i = %d, j = %d. err = %d\n", i, j, err);
        }
    }
    return APP_ERR_OK;
}

APP_ERROR AscendIndexILFlatImpl::ComputeDistByIdxImplFp16(int n, const float16_t *queries, float *distances,
    std::tuple<int, const int *, const idx_t *> idxInfo, std::tuple<unsigned int, const float *> tableInfo)
{
    auto streamPtr = this->pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = this->pResources->getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();

    auto &mem = this->pResources->getMemoryManager();
    int maxNum = std::get<0>(idxInfo);
    const idx_t *indice = std::get<2>(idxInfo);
    unsigned int tableLen = std::get<0>(tableInfo);
    const float *table = std::get<1>(tableInfo);
    int idxSliceNum = utils::divUp(maxNum, IDX_BURST_LEN);

    AscendTensor<float16_t, DIMS_2> queryTensor(const_cast<float16_t *>(queries), {n, this->dim});

    AscendTensor<idx_t, DIMS_2> indiceTensor(const_cast<idx_t *>(indice), { n, maxNum });
    AscendTensor<float, DIMS_3> distResult(mem, { idxSliceNum, n, IDX_BURST_LEN }, stream);
    AscendTensor<idx_t, DIMS_3> idxTensor(mem, { idxSliceNum, n, IDX_BURST_LEN }, stream);

    int idxCopyNum = 0;
    if (isInputDevice) {
        std::string opName = "TransdataIdx";
        LaunchOpOneInOneOut<idx_t, DIMS_2, ACL_UINT32,
                            idx_t, DIMS_3, ACL_UINT32>(opName, streamAicpu, indiceTensor, idxTensor);
        auto ret = synchronizeStream(streamAicpu);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
            "synchronizeStream LaunchOpOneInOneOut streamAicpu failed: %i\n", ret);
    } else {
        auto ret = MoveData(idxCopyNum, maxNum, n, indice, idxTensor);
        APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "MoveData failed(%d)", ret);
    }

    int blockNum = utils::divUp(this->capacity, this->blockSize);
    const int dim1 = utils::divUp(this->blockSize, ZREGION_HEIGHT);
    const int dim2 = utils::divUp(this->dim, CUBE_ALIGN);

    // 每次调用算子，都传入全量的底库
    AscendTensor<float16_t, DIMS_4> shaped(
        this->baseSpace->data(), { blockNum * dim1, dim2, ZREGION_HEIGHT, CUBE_ALIGN });
    AscendTensor<uint32_t, DIMS_2> sizeTensorList(mem, {idxSliceNum, SIZE_ALIGN}, stream);

    for (int i = 0; i < idxSliceNum; i++) {
        auto index = idxTensor[i].view();
        auto dist = distResult[i].view();
        auto sizeTensor = sizeTensorList[i].view();
        sizeTensor[0] = (i == idxSliceNum - 1) ? (maxNum - i * IDX_BURST_LEN) : IDX_BURST_LEN;
        if (tableLen > 0) {
            AscendTensor<float, DIMS_1> tableTensor(mem, {TABLE_LEN}, stream);
            auto ret = aclrtMemcpy(tableTensor.data(), tableTensor.getSizeInBytes(), table, TABLE_LEN * sizeof(float),
                ACL_MEMCPY_HOST_TO_DEVICE);
            APPERR_RETURN_IF_NOT_FMT(
                (ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy tableTensor to device fail(%d) !!!", ret);
            ComputeDistWholeBaseWithTable(queryTensor, sizeTensor, shaped, index, dist, tableTensor, stream);
        } else {
            ComputeDistWholeBase(queryTensor, sizeTensor, shaped, index, dist, stream);
        }
    }
    auto ret = synchronizeStream(stream);

    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream default stream failed, error code: %i\n", ret);

    if (isOutputDevice) {
        AscendTensor<float, DIMS_2> distanceTensor(distances, { n, maxNum });
        std::string opName = "TransdataDist";
        LaunchOpOneInOneOut<float, DIMS_3, ACL_FLOAT,
                            float, DIMS_2, ACL_FLOAT>(opName, streamAicpu, distResult, distanceTensor);
        ret = synchronizeStream(streamAicpu);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
            "synchronizeStream LaunchOpOneInOneOut streamAicpu failed: %i\n", ret);
    } else {
        ret = CopyDistData(maxNum, idxCopyNum, distances, n, distResult);
        APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "CopyDistData failed(%d)", ret);
    }

    return ret;
}

void AscendIndexILFlatImpl::ComputeDistWholeBaseWithTable(AscendTensor<float16_t, DIMS_2> &queryTensor,
    AscendTensor<uint32_t, DIMS_1> &sizeTensor, AscendTensor<float16_t, DIMS_4> &shapedData,
    AscendTensor<idx_t, DIMS_2> &idxTensor, AscendTensor<float, DIMS_2> &distanceTensor,
    AscendTensor<float, DIMS_1> &tableTensor, aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batchSize = queryTensor.getSize(0);
    if (this->distComputeIdxWithTableOps.find(batchSize) != this->distComputeIdxWithTableOps.end()) {
        op = this->distComputeIdxWithTableOps[batchSize].get();
    }
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);

    // construct fake huge input size for basevectors, FLAT_BLOCK_SIZE * IDX_BURST_LEN, the index is 2
    distOpInput->emplace_back(aclCreateDataBuffer(queryTensor.data(), queryTensor.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(idxTensor.data(), idxTensor.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(sizeTensor.data(), sizeTensor.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(shapedData.data(), fakeBaseSizeInBytes));
    distOpInput->emplace_back(aclCreateDataBuffer(tableTensor.data(), tableTensor.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(distanceTensor.data(), distanceTensor.getSizeInBytes()));

    op->exec(*distOpInput, *distOpOutput, stream);
}

void AscendIndexILFlatImpl::ComputeDistWholeBase(AscendTensor<float16_t, DIMS_2> &queryTensor,
    AscendTensor<uint32_t, DIMS_1> &sizeTensor, AscendTensor<float16_t, DIMS_4> &shapedData,
    AscendTensor<idx_t, DIMS_2> &idxTensor, AscendTensor<float, DIMS_2> &distanceTensor, aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batchSize = queryTensor.getSize(0);
    if (this->distComputeIdxOps.find(batchSize) != this->distComputeIdxOps.end()) {
        op = this->distComputeIdxOps[batchSize].get();
    }
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);

    // construct fake huge input size for basevectors, FLAT_BLOCK_SIZE * IDX_BURST_LEN, the index is 2
    distOpInput->emplace_back(aclCreateDataBuffer(queryTensor.data(), queryTensor.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(idxTensor.data(), idxTensor.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(sizeTensor.data(), sizeTensor.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(shapedData.data(), fakeBaseSizeInBytes));

    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(distanceTensor.data(), distanceTensor.getSizeInBytes()));

    op->exec(*distOpInput, *distOpOutput, stream);
}


void AscendIndexILFlatImpl::ComputeBlockDist(AscendTensor<float16_t, DIMS_2> &queryTensor,
    AscendTensor<uint8_t, DIMS_2> &mask,
    AscendTensor<float16_t, DIMS_4> &shapedData, AscendTensor<uint32_t, DIMS_2> &size,
    AscendTensor<float16_t, DIMS_2> &outDistances, AscendTensor<float16_t, DIMS_2> &maxDistances,
    AscendTensor<uint16_t, DIMS_2> &flag, aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batchSize = queryTensor.getSize(0);
    if (this->distComputeOps.find(batchSize) != this->distComputeOps.end()) {
        op = this->distComputeOps[batchSize].get();
    }
    ASCEND_THROW_IF_NOT(op);
 
    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    distOpInput->emplace_back(aclCreateDataBuffer(queryTensor.data(), queryTensor.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(mask.data(), mask.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(shapedData.data(), shapedData.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(size.data(), size.getSizeInBytes()));
 
    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(outDistances.data(), outDistances.getSizeInBytes()));
    distOpOutput->emplace_back(aclCreateDataBuffer(maxDistances.data(), maxDistances.getSizeInBytes()));
    distOpOutput->emplace_back(aclCreateDataBuffer(flag.data(), flag.getSizeInBytes()));
 
    op->exec(*distOpInput, *distOpOutput, stream);
}
 
void AscendIndexILFlatImpl::runTopkCompute(AscendTensor<float16_t, DIMS_3> &dists,
                                           AscendTensor<float16_t, DIMS_3> &maxdists,
                                           AscendTensor<uint32_t, DIMS_3> &sizes,
                                           AscendTensor<uint16_t, DIMS_3> &flags,
                                           AscendTensor<int64_t, DIMS_1> &attrs,
                                           AscendTensor<float16_t, DIMS_2> &outdists,
                                           AscendTensor<idx_t, DIMS_2> &outlabel,
                                           aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batch = dists.getSize(1);
    if (topkComputeOps.find(batch) != topkComputeOps.end()) {
        op = topkComputeOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);
 
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(maxdists.data(), maxdists.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(sizes.data(), sizes.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(flags.data(), flags.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(attrs.data(), attrs.getSizeInBytes()));
 
    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(outdists.data(), outdists.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(outlabel.data(), outlabel.getSizeInBytes()));
 
    op->exec(*topkOpInput, *topkOpOutput, stream);
}
 
 
APP_ERROR AscendIndexILFlatImpl::ComputeDistImplBatched(int n, const float16_t *queries, float *distances,
    unsigned int tableLen, const float *table)
{
    return ComputeDistImplBatchedFp16(n, queries, distances, tableLen, table);
}
 
APP_ERROR AscendIndexILFlatImpl::ComputeDistImplBatched(int n, const float *queries, float *distances,
    unsigned int tableLen, const float *table)
{
    std::vector<float16_t> queryData(n * this->dim, 0);
    transform(queries, queries + n * this->dim, begin(queryData), [](float temp) { return fp16(temp).data; });
    return ComputeDistImplBatchedFp16(n, queryData.data(), distances, tableLen, table);
}
 
 
APP_ERROR AscendIndexILFlatImpl::ComputeDistImplBatchedFp16(int n, const float16_t *queries, float *distances,
    unsigned int tableLen, const float *table)
{
    auto streamPtr = this->pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = this->pResources->getMemoryManager();
    bool needMapping = tableLen > 0 ? true : false;
 
    AscendTensor<float16_t, DIMS_2> queryTensor(mem, {n, this->dim}, stream);
    auto ret = aclrtMemcpy(queryTensor.data(), queryTensor.getSizeInBytes(), queries, n * this->dim * sizeof(float16_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy queryTensor to device fail(%d)!", ret);
 
    AscendTensor<uint32_t, DIMS_1> baseSize(mem, { SIZE_ALIGN }, stream);
    int ntotalPad = utils::divUp(this->ntotal, CUBE_ALIGN) * CUBE_ALIGN;
    AscendTensor<float, DIMS_1> distanceTensor(mem, {n * ntotalPad}, stream);
    baseSize[0] = ntotalPad;
 
    const int dim1 = utils::divUp(this->ntotal, ZREGION_HEIGHT);
    const int dim2 = utils::divUp(this->dim, CUBE_ALIGN);
    AscendTensor<float16_t, DIMS_4> shapedData(this->baseSpace->data(), {dim1, dim2, ZREGION_HEIGHT, CUBE_ALIGN});
 
    if (needMapping) {
        AscendTensor<float, DIMS_1> tableTensor(mem, {TABLE_LEN}, stream);
        ret = aclrtMemcpy(tableTensor.data(), tableTensor.getSizeInBytes(), table, TABLE_LEN * sizeof(float),
            ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(
            (ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy tableTensor to device fail(%d) !!!", ret);
 
        ComputeBlockDistWholeBaseWithTable(queryTensor, shapedData, baseSize, tableTensor, distanceTensor, stream);
    } else {
        ComputeBlockDistWholeBase(queryTensor, shapedData, baseSize, distanceTensor, stream);
    }
    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream failed: %i\n", ret);
 
    ret = aclrtMemcpy(distances, n * ntotalPad * sizeof(float), distanceTensor.data(),
        distanceTensor.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy outDistances back to host(%d)", ret);
    return APP_ERR_OK;
}
 
void AscendIndexILFlatImpl::ComputeBlockDistWholeBaseWithTable(AscendTensor<float16_t, DIMS_2> &queryTensor,
    AscendTensor<float16_t, DIMS_4> &shapedData, AscendTensor<uint32_t, DIMS_1> &baseSize,
    AscendTensor<float, DIMS_1> &tableTensor, AscendTensor<float, DIMS_1> &distanceTensor, aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batchSize = queryTensor.getSize(0);
    if (this->distanceFlatIpWithTableOps.find(batchSize) != this->distanceFlatIpWithTableOps.end()) {
        op = this->distanceFlatIpWithTableOps[batchSize].get();
    }
    ASCEND_THROW_IF_NOT(op);
 
    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    distOpInput->emplace_back(aclCreateDataBuffer(queryTensor.data(), queryTensor.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(shapedData.data(), this->fakeBaseSizeInBytes));
    distOpInput->emplace_back(aclCreateDataBuffer(baseSize.data(), baseSize.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(tableTensor.data(), tableTensor.getSizeInBytes()));
 
    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(
        distanceTensor.data(), static_cast<size_t>(batchSize) * FAKE_HUGE_BASE * sizeof(float)));
 
    op->exec(*distOpInput, *distOpOutput, stream);
}
 
void AscendIndexILFlatImpl::ComputeBlockDistWholeBase(AscendTensor<float16_t, DIMS_2> &queryTensor,
    AscendTensor<float16_t, DIMS_4> &shapedData, AscendTensor<uint32_t, DIMS_1> &baseSize,
    AscendTensor<float, DIMS_1> &distanceTensor, aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batchSize = queryTensor.getSize(0);
    if (this->distanceFlatIpOps.find(batchSize) != this->distanceFlatIpOps.end()) {
        op = this->distanceFlatIpOps[batchSize].get();
    }
    ASCEND_THROW_IF_NOT(op);
 
    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    distOpInput->emplace_back(aclCreateDataBuffer(queryTensor.data(), queryTensor.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(shapedData.data(), this->fakeBaseSizeInBytes));
    distOpInput->emplace_back(aclCreateDataBuffer(baseSize.data(), baseSize.getSizeInBytes()));
    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(
        distanceTensor.data(), static_cast<size_t>(batchSize) * FAKE_HUGE_BASE * sizeof(float)));
 
    op->exec(*distOpInput, *distOpOutput, stream);
}
 
APP_ERROR AscendIndexILFlatImpl::Remove(int n, const idx_t *indices)
{
    APPERR_RETURN_IF_NOT_LOG(
        (isInitialized), APP_ERR_INVALID_PARAM, "Illegal operation, please initialize the index first. ");
 
    size_t dimAlign = static_cast<size_t>(utils::divUp(this->dim, CUBE_ALIGN));
 
    for (size_t i = 0; i < static_cast<size_t>(n); i++) {
        auto seq = static_cast<size_t>(*(indices + i));
        APPERR_RETURN_IF_NOT_LOG(seq < static_cast<size_t>(this->ntotal), APP_ERR_INVALID_PARAM,
            "Invalid feature to remove, the indice should not be greater than ntotal\n");
        size_t offset1 = seq / ZREGION_HEIGHT * dimAlign;
        size_t offset2 = seq % ZREGION_HEIGHT;
        size_t offset = offset1 * ZREGION_HEIGHT * CUBE_ALIGN + offset2 * CUBE_ALIGN;
        auto dataptr = this->baseSpace->data() + offset;
 
        for (size_t j = 0; j < dimAlign; j++) {
            auto err = aclrtMemset(dataptr, CUBE_ALIGN * sizeof(float16_t), 0x0, CUBE_ALIGN * sizeof(float16_t));
            APPERR_RETURN_IF_NOT_FMT(err == EOK, APP_ERR_INNER_ERROR, "aclrtMemset error, err=%d\n", err);
            dataptr += ZREGION_HEIGHT * CUBE_ALIGN;
        }
    }
 
    return APP_ERR_OK;
}

APP_ERROR AscendIndexILFlatImpl::Get(int n, float16_t *features, const idx_t *indices) const
{
    AscendTensor<float16_t, DIMS_2> queries(features, {n, this->dim});
    auto ret = GetDevice(n, queries.data(), indices);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "GetDevice failed: %i\n", ret);

    ret = aclrtMemcpy(features, n * this->dim * sizeof(float16_t), queries.data(),
        n * this->dim * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy features back to host(%d)", ret);

    return APP_ERR_OK;
}

APP_ERROR AscendIndexILFlatImpl::Get(int n, float *features, const idx_t *indices) const
{
    size_t total = static_cast<size_t>(n) * static_cast<size_t>(this->dim);
    std::vector<float16_t> featuresData(total);
    auto ret = Get(n, featuresData.data(), indices);
    APPERR_RETURN_IF_NOT_FMT((ret == APP_ERR_OK), APP_ERR_INNER_ERROR, "Get fail(%d)!", ret);
    std::transform(featuresData.data(), featuresData.data() + total, features,
        [](float16_t temp) { return static_cast<float>(fp16(temp)); });
    return APP_ERR_OK;
}

APP_ERROR AscendIndexILFlatImpl::GetDevice(int n, float16_t *features, const idx_t *indices) const
{
    APPERR_RETURN_IF_NOT_FMT(n >= 0 && n <= this->capacity, APP_ERR_INVALID_PARAM,
        "The number n should be in range [0, %d]", this->capacity);
    APPERR_RETURN_IF_NOT_LOG(features, APP_ERR_INVALID_PARAM, "Features can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(indices, APP_ERR_INVALID_PARAM, "Indices can not be nullptr.");
    auto streamPtr = this->pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = this->pResources->getMemoryManager();

    std::string opName = "TransdataGet";
    AscendTensor<idx_t, DIMS_1> attr(mem, { n }, stream);
    auto ret = aclrtMemcpy(attr.data(), attr.getSizeInBytes(), indices, n * sizeof(idx_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "copy attr to device fail(%d) !!!", ret);

    AscendTensor<float16_t, DIMS_2> dst(features, {n, this->dim});
    AscendTensor<float16_t, DIMS_4> src(this->baseSpace->data(),
        {utils::divUp(this->capacity, ZREGION_HEIGHT), utils::divUp(this->dim, CUBE_ALIGN),
        ZREGION_HEIGHT, CUBE_ALIGN});
    LaunchOpTwoInOneOut<float16_t, DIMS_4, ACL_FLOAT16,
                        idx_t, DIMS_1, ACL_UINT32,
                        float16_t, DIMS_2, ACL_FLOAT16>(opName, stream, src, attr, dst);
    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream LaunchOpTwoInOneOut stream failed: %i\n", ret);
    return APP_ERR_OK;
}

APP_ERROR AscendIndexILFlatImpl::GetDevice(int n, float *features, const idx_t *indices) const
{
    APPERR_RETURN_IF_NOT_FMT(n >= 0 && n <= this->capacity, APP_ERR_INVALID_PARAM,
        "The number n should be in range [0, %d]", this->capacity);
    APPERR_RETURN_IF_NOT_LOG(features, APP_ERR_INVALID_PARAM, "Features can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(indices, APP_ERR_INVALID_PARAM, "Indices can not be nullptr.");
    size_t total = static_cast<size_t>(n) * static_cast<size_t>(this->dim);
    std::vector<float> featuresData(total);

    auto ret = Get(n, featuresData.data(), indices);
    ret = aclrtMemcpy(features, total * sizeof(float), featuresData.data(),
        featuresData.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR,
        "copy featuresData to device fail(%d)!", ret);

    return APP_ERR_OK;
}

void AscendIndexILFlatImpl::SetNTotal(int n)
{
    this->ntotal = n;
}
 
int AscendIndexILFlatImpl::GetNTotal() const
{
    return this->ntotal;
}
 
APP_ERROR AscendIndexILFlatImpl::ResetDistanceFlatIpOp()
{
    auto distanceFlatIpOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("DistanceFlatIp");
        std::vector<int64_t> queryShape({ batch, this->dim });
        std::vector<int64_t> coarseCentroidsShape({ utils::divUp(FAKE_HUGE_BASE, ZREGION_HEIGHT),
            utils::divUp(this->dim, CUBE_ALIGN), ZREGION_HEIGHT, CUBE_ALIGN });
        std::vector<int64_t> sizeShape({ SIZE_ALIGN });
        std::vector<int64_t> distResultShape({ batch * utils::divUp(FAKE_HUGE_BASE, ZREGION_HEIGHT) * ZREGION_HEIGHT });
 
        desc.addInputTensorDesc(ACL_FLOAT16, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, coarseCentroidsShape.size(), coarseCentroidsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);
        op.reset();
        op = std::make_unique<AscendOperator>(desc);
        return op->init();
    };
 
    for (auto batch : this->computeBatchSizes) {
        distanceFlatIpOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(distanceFlatIpOpReset(distanceFlatIpOps[batch], batch),
            APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "distanceFlatIpOpReset init failed");
    }
    return APP_ERR_OK;
}
 
APP_ERROR AscendIndexILFlatImpl::ResetDistanceWithTableOp()
{
    auto distanceFlatIpWithTableOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("DistanceFlatIpWithTable");
        std::vector<int64_t> queryShape({ batch, this->dim });
        std::vector<int64_t> coarseCentroidsShape({ utils::divUp(FAKE_HUGE_BASE, ZREGION_HEIGHT),
            utils::divUp(this->dim, CUBE_ALIGN), ZREGION_HEIGHT, CUBE_ALIGN });
        std::vector<int64_t> sizeShape({ SIZE_ALIGN });
        std::vector<int64_t> distResultShape({ batch * utils::divUp(FAKE_HUGE_BASE, ZREGION_HEIGHT) * ZREGION_HEIGHT });
        std::vector<int64_t> tableShape({TABLE_LEN});
 
        desc.addInputTensorDesc(ACL_FLOAT16, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, coarseCentroidsShape.size(), coarseCentroidsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, tableShape.size(), tableShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);
        op.reset();
        op = std::make_unique<AscendOperator>(desc);
        return op->init();
    };
 
    for (auto batch : this->computeBatchSizes) {
        distanceFlatIpWithTableOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(distanceFlatIpWithTableOpReset(distanceFlatIpWithTableOps[batch], batch),
            APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "distanceFlatIpWithTableOpReset init failed");
    }
    return APP_ERR_OK;
}
 
APP_ERROR AscendIndexILFlatImpl::ResetTopkCompOp()
{
    auto topkCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("TopkFlat");
        auto burstLen = (batch >= LARGE_BATCH_THRESHOLD) ? LARGE_BATCH_BURST_LEN : BURST_LEN;
        auto burstOfBlock = (FLAT_BLOCK_SIZE + burstLen - 1) / burstLen * 2;
        std::vector<int64_t> shape0 { 0, batch, this->blockSize };
        std::vector<int64_t> shape1 { 0, batch, burstOfBlock };
        std::vector<int64_t> shape2 { 0, CORE_NUM, SIZE_ALIGN };
        std::vector<int64_t> shape3 { 0, CORE_NUM, FLAG_SIZE };
        std::vector<int64_t> shape4 { aicpu::TOPK_FLAT_ATTR_IDX_COUNT };
        std::vector<int64_t> shape5 { batch, 0 };
 
        desc.addInputTensorDesc(ACL_FLOAT16, shape0.size(), shape0.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, shape1.size(), shape1.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, shape2.size(), shape2.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, shape3.size(), shape3.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape4.size(), shape4.data(), ACL_FORMAT_ND);
 
        desc.addOutputTensorDesc(ACL_FLOAT16, shape5.size(), shape5.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT32, shape5.size(), shape5.data(), ACL_FORMAT_ND);
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };
 
    for (auto batch : this->searchBatchSizes) {
        topkComputeOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(topkCompOpReset(topkComputeOps[batch], batch),
                                 APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "topk op init failed");
    }
    return APP_ERR_OK;
}
 
APP_ERROR AscendIndexILFlatImpl::ResetDistCompOp(int numLists)
{
    auto distCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        auto burstLen = (batch >= LARGE_BATCH_THRESHOLD) ? LARGE_BATCH_BURST_LEN : BURST_LEN;
        // 乘以2，是和算子生成时的shape保持一致
        auto burstOfBlock = (FLAT_BLOCK_SIZE + burstLen - 1) / burstLen * 2;
        std::string opName = (batch >= LARGE_BATCH_THRESHOLD) ? "DistanceFlatIPMaxsBatch" : "DistanceFlatIPMaxs";
        AscendOpDesc desc(opName);
        std::vector<int64_t> queryShape({ batch, this->dim });
        std::vector<int64_t> maskShape({ batch, blockMaskSize });
        std::vector<int64_t> coarseCentroidsShape({ utils::divUp(numLists, ZREGION_HEIGHT),
            utils::divUp(this->dim, CUBE_ALIGN), ZREGION_HEIGHT, CUBE_ALIGN });
        std::vector<int64_t> sizeShape({ CORE_NUM, SIZE_ALIGN });
        std::vector<int64_t> distResultShape({ batch, numLists });
        std::vector<int64_t> maxResultShape({ batch, burstOfBlock });
        std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });
 
        desc.addInputTensorDesc(ACL_FLOAT16, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT8, maskShape.size(), maskShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, coarseCentroidsShape.size(), coarseCentroidsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, maxResultShape.size(), maxResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);
 
        op.reset();
        op = std::make_unique<AscendOperator>(desc);
        return op->init();
    };
 
    for (auto batch : this->searchBatchSizes) {
        distComputeOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(distCompOpReset(distComputeOps[batch], batch),
            APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "ResetDistCompOp init failed");
    }
    return APP_ERR_OK;
}

APP_ERROR AscendIndexILFlatImpl::ResetDistCompIdxOp()
{
    auto distCompIdxOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("DistanceFlatIpByIdx");
        std::vector<int64_t> queryShape({ batch, this->dim });
        std::vector<int64_t> indexShape({ batch, IDX_BURST_LEN });
        std::vector<int64_t> sizeShape({ SIZE_ALIGN });
        std::vector<int64_t> coarseCentroidsShape(
            { utils::divUp(FAKE_HUGE_BASE, ZREGION_HEIGHT), utils::divUp(this->dim, CUBE_ALIGN),
            ZREGION_HEIGHT, CUBE_ALIGN });
        std::vector<int64_t> distResultShape({ batch, IDX_BURST_LEN });

        desc.addInputTensorDesc(ACL_FLOAT16, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, indexShape.size(), indexShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, coarseCentroidsShape.size(), coarseCentroidsShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);

        op.reset();
        op = std::make_unique<AscendOperator>(desc);
        return op->init();
    };
    for (auto batch : computeByIdxBatchSizes) {
        distComputeIdxOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(distCompIdxOpReset(distComputeIdxOps[batch], batch),
            APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "distCompIdxOpReset init failed");
    }
    return APP_ERR_OK;
}

APP_ERROR AscendIndexILFlatImpl::ResetDistCompIdxWithTableOp()
{
    auto distCompIdxWithTableOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("DistanceFlatIpByIdxWithTable");
        std::vector<int64_t> queryShape({ batch, this->dim });
        std::vector<int64_t> indexShape({ batch, IDX_BURST_LEN });
        std::vector<int64_t> sizeShape({ SIZE_ALIGN });
        std::vector<int64_t> coarseCentroidsShape(
            { utils::divUp(FAKE_HUGE_BASE, ZREGION_HEIGHT), utils::divUp(this->dim, CUBE_ALIGN),
            ZREGION_HEIGHT, CUBE_ALIGN });
        std::vector<int64_t> tableShape({ TABLE_LEN });
        std::vector<int64_t> distResultShape({ batch, IDX_BURST_LEN });

        desc.addInputTensorDesc(ACL_FLOAT16, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, indexShape.size(), indexShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, sizeShape.size(), sizeShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, coarseCentroidsShape.size(), coarseCentroidsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, tableShape.size(), tableShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);

        op.reset();
        op = std::make_unique<AscendOperator>(desc);
        return op->init();
    };
    for (auto batch : computeByIdxBatchSizes) {
        distComputeIdxWithTableOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(distCompIdxWithTableOpReset(distComputeIdxWithTableOps[batch], batch),
            APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "distCompIdxWithTableOpReset init failed");
    }
    return APP_ERR_OK;
}

} // namespace ascend
}