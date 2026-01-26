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


#include "ascenddaemon/impl/IndexInt8Flat.h"

#include <set>
#include <algorithm>
#include <vector>

#include "ascenddaemon/impl/AuxIndexStructures.h"
#include "ascenddaemon/utils/DeviceVector.h"
#include "ascenddaemon/utils/StaticUtils.h"
#include "ascenddaemon/utils/AscendTensor.h"
#include "ascenddaemon/utils/Limits.h"
#include "common/utils/LogUtils.h"
#include "common/utils/AscendAssert.h"
#include "common/utils/CommonUtils.h"
#include "common/utils/OpLauncher.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"

namespace ascend {
namespace {
const int THREADS_CNT = faiss::ascend::SocUtils::GetInstance().GetThreadsCnt();
const int BURST_LEN = 64;
const int L2NORM_COMPUTE_BATCH = 16384; // each blocksize = 16384 * 16
const int PAGE_BLOCKS = 32;
}

// 正常来说全局算子资源应该调用resetOp、runOp接口来使用，从而保证线程安全；
// 这里没有按照正常的使用方法，因此需要单独加锁
static std::mutex multiSearchTopkMtx;

IndexSearchTensorShare::IndexSearchTensorShare(AscendResourcesProxy &resources, IndexSearchContext &ctx,
    const HeteroBlockGroupMgr &grpSpliter, const IndexSchemaBase &indexSchema)
    : groupCount(grpSpliter.Count())
{
    if (groupCount == 0) {
        return;
    }

    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    int nq = ctx.queries.getSize(0);
    queriesNorm = std::shared_ptr<AscendTensor<float16_t, DIMS_1>>(
        new AscendTensor<float16_t, DIMS_1>(mem, { utils::roundUp(nq, CUBE_ALIGN) }, stream));
    actualNum = std::shared_ptr<AscendTensor<uint32_t, DIMS_2>>(
        new AscendTensor<uint32_t, DIMS_2>(mem, { utils::divUp(nq, L2NORM_COMPUTE_BATCH), SIZE_ALIGN }, stream));
    if (grpSpliter.GetGroupSize() > SIZE_MAX - static_cast<size_t>(indexSchema.getBlockSize())) {
        return;
    }
    size_t maxBlockCount = utils::divUp(grpSpliter.GetGroupSize(), indexSchema.getBlockSize());
    // groupCount=1的场景使用lastDistResult，lastMinDistResult
    if (groupCount > 1) {
        distResult = std::shared_ptr<AscendTensor<float16_t, DIMS_3, size_t>>(
            new AscendTensor<float16_t, DIMS_3, size_t>(mem, { maxBlockCount,
                static_cast<size_t>(nq), static_cast<size_t>(indexSchema.getBlockSize())}, stream));
        minDistResult = std::shared_ptr<AscendTensor<float16_t, DIMS_3, size_t>>(
            new AscendTensor<float16_t, DIMS_3, size_t>
                (mem, { maxBlockCount, static_cast<size_t>(nq),
                        static_cast<size_t>(indexSchema.getBurstsOfBlock())}, stream));
    }

    size_t grpId = groupCount - 1;
    auto &grp = grpSpliter.At(grpId);
    int blockNum = static_cast<int>(grp.blocks.size());
    lastDistResult = std::shared_ptr<AscendTensor<float16_t, DIMS_3, size_t>>(
        new AscendTensor<float16_t, DIMS_3, size_t>
            (mem, { static_cast<size_t>(blockNum), static_cast<size_t>(nq),
                    static_cast<size_t>(indexSchema.getBlockSize())}, stream));
    lastMinDistResult = std::shared_ptr<AscendTensor<float16_t, DIMS_3, size_t>>(
        new AscendTensor<float16_t, DIMS_3, size_t>
            (mem, { static_cast<size_t>(blockNum), static_cast<size_t>(nq),
                    static_cast<size_t>(indexSchema.getBurstsOfBlock())}, stream));

    for (grpId = 0; grpId < groupCount; ++grpId) {
        auto &grp = grpSpliter.At(grpId);
        blockNum = static_cast<int>(grp.blocks.size());
        opFlagVec.emplace_back(std::shared_ptr<AscendTensor<uint16_t, DIMS_3>>(
            new AscendTensor<uint16_t, DIMS_3>(mem, { blockNum, FLAG_NUM, FLAG_SIZE }, stream)));
        opSizeVec.emplace_back(std::shared_ptr<AscendTensor<uint32_t, DIMS_3>>(
            new AscendTensor<uint32_t, DIMS_3>(mem, { blockNum, CORE_NUM, SIZE_ALIGN }, stream)));
        attrsInputVec.emplace_back(std::shared_ptr<AscendTensor<int64_t, DIMS_1>>(
            new AscendTensor<int64_t, DIMS_1>(mem, { aicpu::TOPK_FLAT_ATTR_IDX_COUNT }, stream)));
    }
}

AscendTensor<float16_t, DIMS_3, size_t> &IndexSearchTensorShare::getDistResult(size_t grpId)
{
    if (grpId + 1 < groupCount) {
        return *distResult;
    }
    return *lastDistResult;
}

AscendTensor<float16_t, DIMS_3, size_t> &IndexSearchTensorShare::getMinDistResult(size_t grpId)
{
    if (grpId + 1 < groupCount) {
        return *minDistResult;
    }
    return *lastMinDistResult;
}

IndexSearchTensorShare::~IndexSearchTensorShare()
{
    // 这里的释放顺序，需要和构造顺序相反：AscendTensor用了stack模式来申请和释放内存，即先申请的后释放
    size_t groupCount = attrsInputVec.size();
    for (size_t i = 0; i < groupCount; ++i) {
        if (!attrsInputVec.empty()) {
            attrsInputVec.pop_back();
        }
        if (!opSizeVec.empty()) {
            opSizeVec.pop_back();
        }
        if (!opFlagVec.empty()) {
            opFlagVec.pop_back();
        }
    }
}

template <typename P>
IndexInt8Flat<P>::IndexInt8Flat(int dim, MetricType metric, int64_t resourceSize, int dBlockSize)
    : IndexInt8(dim, metric, resourceSize), codeBlockSize(dBlockSize)
{
    ASCEND_THROW_IF_NOT(this->dims % CUBE_ALIGN_INT8 == 0);

    // supported batch size
    if (faiss::ascend::SocUtils::GetInstance().IsAscend910B() && (metric == METRIC_INNER_PRODUCT)) {
        searchBatchSizes = { 128, 112, 96, 80, 64, 48, 36, 32, 24, 18, 16, 12, 8, 6, 4, 2, 1 };
    } else {
        searchBatchSizes = { 64, 48, 36, 32, 24, 18, 16, 12, 8, 6, 4, 2, 1 };
    }

    // no need train for flat index
    this->isTrained = true;

    this->blockMaskSize = utils::divUp(this->codeBlockSize, BINARY_BYTE_SIZE);
    this->pageSize = this->codeBlockSize * DEFAULT_PAGE_BLOCK_NUM;
    int dim1 = utils::divUp(this->codeBlockSize, CUBE_ALIGN);
    int dim2 = utils::divUp(this->dims, CUBE_ALIGN_INT8);
    this->devVecCapacity = dim1 * dim2 * CUBE_ALIGN * CUBE_ALIGN_INT8;

    // align by 2
    burstsOfBlock = (this->codeBlockSize + BURST_LEN - 1) / BURST_LEN * 2;

    this->int8L2Norm = CREATE_UNIQUE_PTR(Int8L2Norm, dim); // int8L2Norm is inited by child class
    flagNum = faiss::ascend::SocUtils::GetInstance().IsAscend910B() ? CORE_NUM : FLAG_NUM;
    isNeedCleanMinDist = faiss::ascend::SocUtils::GetInstance().IsAscend910B();
}

template<typename P>
IndexInt8Flat<P>::~IndexInt8Flat() {}

template<typename P>
APP_ERROR IndexInt8Flat<P>::setHeteroParam(uint32_t deviceId, size_t deviceCapacity,
    size_t deviceBuffer, size_t hostCapacity)
{
    size_t blockMem = static_cast<size_t>(devVecCapacity) * sizeof(int8_t);
    // 缓冲区至少要预留2个block的大小，用于交替缓冲使用
    const size_t devBufferMin = 2 * blockMem;
    APPERR_RETURN_IF_NOT_FMT(deviceBuffer >= devBufferMin, APP_ERR_INVALID_PARAM,
        "deviceBuffer %zu should >= %zu", deviceBuffer, devBufferMin);

    return deviceMemMng.SetHeteroParam(deviceId, deviceCapacity, deviceBuffer, hostCapacity, blockMem);
}

template <typename P>
APP_ERROR IndexInt8Flat<P>::tryToSychResultAdvanced(int &hasCopiedCount, int &indexDoneCount, int indexId, int n,
    int batchSize, int k, float16_t *distances, idx_t *labels, AscendTensor<float16_t, DIMS_3, size_t> &srcDistances,
    AscendTensor<idx_t, DIMS_3, size_t> &srcIndices)
{
    for (int j = hasCopiedCount; j < indexDoneCount; ++j) {
        size_t totalIndex = static_cast<size_t>(j) * static_cast<size_t>(n) * static_cast<size_t>(k);
        size_t totalBatchSize = static_cast<size_t>(batchSize) * static_cast<size_t>(k);

        auto ret = aclrtMemcpy(distances + totalIndex, totalBatchSize * sizeof(float16_t),
            srcDistances[j].data(), totalBatchSize * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

        ret = aclrtMemcpy(labels + totalIndex, totalBatchSize * sizeof(idx_t),
            srcIndices[j].data(), totalBatchSize * sizeof(idx_t), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
        ++hasCopiedCount;
    }

    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    auto ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream aicpu stream failed: %i\n", ret);
    indexDoneCount = indexId - 1;

    return APP_ERR_OK;
}

template<typename P>
APP_ERROR IndexInt8Flat<P>::addVectorsAicpu(size_t startOffset, AscendTensor<int8_t, DIMS_2> &rawData)
{
    int n = rawData.getSize(0);
    std::string opName = "TransdataShaped";
    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    AscendTensor<int8_t, DIMS_2> data;
    if (faiss::ascend::SocUtils::GetInstance().IsRunningInHost()) {
        AscendTensor<int8_t, DIMS_2> hostData(mem, {n, dims}, stream);
        data = std::move(hostData);
        auto ret = aclrtMemcpy(data.data(), data.getSizeInBytes(),
            rawData.data(), rawData.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
    } else {
        AscendTensor<int8_t, DIMS_2> deviceData(rawData.data(), {n, dims});
        data = std::move(deviceData);
    }

    int blockNum = utils::divUp(static_cast<int>(startOffset) + n, codeBlockSize);
    AscendTensor<int64_t, DIMS_2> attrs(mem, {blockNum, aicpu::TRANSDATA_SHAPED_ATTR_IDX_COUNT}, stream);

    size_t blockSize = static_cast<size_t>(this->codeBlockSize);
    for (size_t i = 0; i < static_cast<size_t>(n);) {
        size_t total = startOffset + i;
        size_t offsetInBlock = total % blockSize;
        size_t leftInBlock = blockSize - offsetInBlock;
        size_t leftInData = static_cast<size_t>(n) - i;
        size_t copyCount = std::min(leftInBlock, leftInData);
        size_t blockIdx = total / blockSize;

        int copy = static_cast<int>(copyCount);
        AscendTensor<int8_t, DIMS_2> src(data[i].data(), {copy, dims});
        AscendTensor<int8_t, DIMS_4> dst(baseShaped[blockIdx]->data(),
            {utils::divUp(this->codeBlockSize, CUBE_ALIGN), utils::divUp(dims, CUBE_ALIGN_INT8),
            CUBE_ALIGN, CUBE_ALIGN_INT8});
        AscendTensor<int64_t, DIMS_1> attr = attrs[blockIdx].view();
        attr[aicpu::TRANSDATA_SHAPED_ATTR_NTOTAL_IDX] = offsetInBlock;

        LaunchOpTwoInOneOut<int8_t, DIMS_2, ACL_INT8,
                            int64_t, DIMS_1, ACL_INT64,
                            int8_t, DIMS_4, ACL_INT8>(opName, stream, src, attr, dst);

        i += copyCount;

        deviceMemMng.DevVecFullProc(baseShaped, blockIdx, (copyCount == leftInBlock), stream);
    }

    auto ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream addVector stream failed: %i\n", ret);

    return APP_ERR_OK;
}

template<typename P>
APP_ERROR IndexInt8Flat<P>::addVectors(AscendTensor<int8_t, DIMS_2> &rawData,
    int num, int dim, int vecSize, int addVecNum)
{
    for (int i = 0; i < addVecNum; ++i) {
        this->baseShaped.emplace_back(deviceMemMng.CreateDeviceVector<int8_t>(MemorySpace::DEVICE_HUGEPAGE));
        this->normBase.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<P>));
        int64_t newVecSize = static_cast<int64_t>(vecSize) + static_cast<int64_t>(i);
        if (newVecSize > 0) {
            this->baseShaped.at(newVecSize)->resize(this->devVecCapacity, true);
            this->normBase.at(newVecSize)->resize(this->codeBlockSize, true);
        }
    }

    // resize little to avoid waste for idx 0
    if (ntotal < static_cast<idx_t>(this->codeBlockSize)) {
        auto capacity = getVecCapacity(this->ntotal + num, this->baseShaped.at(0)->size());
        this->baseShaped.at(0)->resize(capacity, true);
        this->normBase.at(0)->resize(utils::divUp(capacity, static_cast<size_t>(dim)), true);
    }
    return addVectorsAicpu(ntotal, rawData);
}

template<typename P>
void IndexInt8Flat<P>::resizeBaseShaped(int num)
{
    int newVecSize = static_cast<int>(utils::divUp(this->ntotal + num, this->codeBlockSize));
    int vecSize = static_cast<int>(utils::divUp(this->ntotal, this->codeBlockSize));
    int addVecNum = static_cast<int>(utils::divUp(this->ntotal + num, this->codeBlockSize)) - vecSize;

    // 1. adapt old block size
    if (vecSize > 0) {
        int vecId = vecSize - 1;
        if (addVecNum > 0) {
            // there is a new block needed, then the old last one block must be fulled
            this->baseShaped.at(vecId)->resize(this->devVecCapacity, true);
            this->normBase.at(vecId)->resize(this->codeBlockSize, true);
        } else {
            auto capacity = getVecCapacity(
                this->ntotal - static_cast<size_t>(vecId * this->codeBlockSize) + static_cast<size_t>(num),
                this->baseShaped.at(vecId)->size());
            this->baseShaped.at(vecId)->resize(capacity, true);
            this->normBase.at(vecId)->resize(utils::divUp(capacity, static_cast<size_t>(dims)), true);
        }
    }

    for (int i = 0; i < addVecNum; ++i) {
        this->baseShaped.emplace_back(deviceMemMng.CreateDeviceVector<int8_t>(MemorySpace::DEVICE_HUGEPAGE));
        this->normBase.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<P>));
        int64_t newVecId = static_cast<int64_t>(vecSize) + static_cast<int64_t>(i);
        if (newVecId == newVecSize - 1) {
            auto capacity = getVecCapacity(
                this->ntotal - static_cast<size_t>(newVecId * this->codeBlockSize) + static_cast<size_t>(num),
                this->baseShaped.at(newVecId)->size());
            this->baseShaped.at(newVecId)->resize(capacity, true);
            this->normBase.at(newVecId)->resize(utils::divUp(capacity, static_cast<size_t>(dims)), true);
        } else {
            this->baseShaped.at(newVecId)->resize(this->devVecCapacity, true);
            this->normBase.at(newVecId)->resize(this->codeBlockSize, true);
        }
    }
}

template<typename P>
APP_ERROR IndexInt8Flat<P>::addVectors(AscendTensor<int8_t, DIMS_2> &rawData)
{
    int num = rawData.getSize(0);
    int dim = rawData.getSize(1);
    APPERR_RETURN_IF_NOT_FMT(num >= 0, APP_ERR_INVALID_PARAM, "the number of vectors added is %d", num);
    APPERR_RETURN_IF_NOT_FMT(
        dim == this->dims, APP_ERR_INVALID_PARAM, "the dim of add vectors is %d, not equal to base", dim);

    int vecSize = utils::divUp((int)this->ntotal, this->codeBlockSize);
    int addVecNum = utils::divUp((int)this->ntotal + num, this->codeBlockSize) - vecSize;

    // ock方案使用内存优化方案会导致性能下降，使用原有方案
    if (deviceMemMng.GetStrategy() == DevMemStrategy::HETERO_MEM) {
        return addVectors(rawData, num, dim, vecSize, addVecNum);
    }

    resizeBaseShaped(num);

    return copyAndSaveVectors(ntotal, rawData);
}

template<typename P>
APP_ERROR IndexInt8Flat<P>::copyAndSaveVectors(size_t startOffset, AscendTensor<int8_t, DIMS_2> &rawData)
{
    if (faiss::ascend::SocUtils::GetInstance().IsZZCodeFormat()) {
        return addVectorsAicpu(startOffset, rawData);
    } else {
        return AddCodeNDFormat(rawData, startOffset, codeBlockSize, baseShaped);
    }
}

template<typename P>
size_t IndexInt8Flat<P>::getVecCapacity(size_t vecNum, size_t size) const
{
    const size_t minCapacity = 512 * 1024;
    // The value of need size is aligned based on 1024 records.
    // Because the maximum number of records that can be operated in batches by the int8 operator is 1024.
    const size_t needSize = utils::roundUp(vecNum, 1024) * dims;
    if (needSize < minCapacity) {
        return minCapacity;
    }

    if (needSize <= size) {
        return size;
    }

    return std::min(needSize, static_cast<size_t>(this->devVecCapacity));
}

template<typename P>
void IndexInt8Flat<P>::getBaseEnd()
{
    // 释放用于getBase申请的device内存
    dataVec.clear();
    attrsVec.clear();
}

template<typename P>
void IndexInt8Flat<P>::setPageSize(uint16_t pageBlockNum)
{
    this->pageSize = this->codeBlockSize * static_cast<int>(pageBlockNum);
}

template<typename P>
APP_ERROR IndexInt8Flat<P>::getVectorsAiCpu(uint32_t offset, uint32_t num, std::vector<int8_t> &vectors)
{
    if (faiss::ascend::SocUtils::GetInstance().IsAscend910B()) {
        size_t blockSize = static_cast<size_t>(this->codeBlockSize);
        size_t total = offset;
        size_t blockIdx = total / blockSize;
        size_t offsetInBlock = total % blockSize;
        int srcOffset = offsetInBlock * dim;
        auto ret = aclrtMemcpy(vectors.data(), vectors.size() * sizeof(int8_t),
            baseShaped[BlockIdx]->data() + srcOffset, vectors.size() * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);
    } else {
        std::string opName = "TransdataRaw";
        auto streamPtr = resources.getAlternateStreams().back();
        auto stream = streamPtr->GetStream();
        int blockNum = utils::divUp(static_cast<int>(ntotal), codeBlockSize);
        // dataVec、attrsVec需要使用完后清理
        dataVec.resize(static_cast<size_t>(num) * static_cast<size_t>(dims), true);
        attrsVec.resize(blockNum * aicpu::TRANSDATA_RAW_ATTR_IDX_COUNT, true);

        AscendTensor<int8_t, DIMS_2> data(dataVec.data(), {static_cast<int>(num), dims});
        AscendTensor<int64_t, DIMS_2> attrs(attrsVec.data(), {blockNum, aicpu::TRANSDATA_RAW_ATTR_IDX_COUNT});

        size_t blockSize = static_cast<size_t>(this->codeBlockSize);
        for (size_t i = 0; i < num;) {
            size_t total = offset + i;
            size_t offsetInBlock = total % blockSize;
            size_t leftInBlock = blockSize - offsetInBlock;
            size_t leftInData = num - i;
            size_t copyCount = std::min(leftInBlock, leftInData);
            size_t blockIdx = total / blockSize;

            int copy = static_cast<int>(copyCount);
            AscendTensor<int8_t, DIMS_2> dst(data[i].data(), {copy, dims});
            AscendTensor<int8_t, DIMS_4> src(baseShaped[blockIdx]->data(),
                {utils::divUp(this->codeBlockSize, CUBE_ALIGN), utils::divUp(dims, CUBE_ALIGN_INT8),
                CUBE_ALIGN, CUBE_ALIGN_INT8});
            AscendTensor<int64_t, DIMS_1> attr = attrs[blockIdx].view();
            attr[aicpu::TRANSDATA_RAW_ATTR_OFFSET_IDX] = offsetInBlock;

            LaunchOpTwoInOneOut<int8_t, DIMS_4, ACL_INT8,
                                int64_t, DIMS_1, ACL_INT64,
                                int8_t, DIMS_2, ACL_INT8>(opName, stream, src, attr, dst);

            i += copyCount;
        }
        auto ret = synchronizeStream(stream);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
            "synchronizeStream addVector stream failed: %i\n", ret);

        ret = aclrtMemcpy(vectors.data(), vectors.size() * sizeof(int8_t),
        data.data(), data.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);
    }
    return APP_ERR_OK;
}

template<typename P>
APP_ERROR IndexInt8Flat<P>::getVectors(uint32_t offset, uint32_t num, std::vector<int8_t> &vectors)
{
    uint32_t actualNum;
    if (offset >= this->ntotal) {
        actualNum = 0;
    } else if (offset + num >= this->ntotal) {
        actualNum = this->ntotal - offset;
    } else {
        actualNum = num;
    }

    vectors.resize(actualNum * this->dims);

    if (faiss::ascend::SocUtils::GetInstance().IsZZCodeFormat()) {
        return getVectorsAiCpu(offset, actualNum, vectors);
    }
    return GetVectorsNDFormat(offset, actualNum, codeBlockSize, baseShaped, vectors);
}

template<typename P>
void IndexInt8Flat<P>::reset()
{
    int dvSize = utils::divUp((int)this->ntotal, this->codeBlockSize);
    for (int i = 0; i < dvSize; ++i) {
        baseShaped.at(i)->clear();
        normBase.at(i)->clear();
    }
    this->ntotal = 0;
}

template<typename P>
APP_ERROR IndexInt8Flat<P>::searchImpl(int n, const int8_t *x, int k, float16_t *distances, idx_t *labels)
{
    // 1. init output data
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    AscendTensor<int8_t, DIMS_2> queries(const_cast<int8_t *>(x), { n, dims });
    AscendTensor<float16_t, DIMS_2> outDistances(distances, { n, k });
    AscendTensor<idx_t, DIMS_2> outIndices(labels, { n, k });

    // 这里申请的内存要512字对齐，此时dist算子性能最优；
    constexpr size_t alignLen = 512;
    size_t minDistancesLen = static_cast<size_t>(n) * static_cast<size_t>(k) * sizeof(float16_t);
    size_t minDistancesMemLen = utils::roundUp(minDistancesLen, alignLen);
    AscendTensor<int8_t, DIMS_1, size_t> minDistancesMem(mem, { minDistancesMemLen }, stream);
    AscendTensor<float16_t, DIMS_2> minDistances(reinterpret_cast<float16_t *>(minDistancesMem.data()), { n, k });

    size_t minIndicesLen = static_cast<size_t>(n) * static_cast<size_t>(k) * sizeof(int64_t);
    size_t minIndicesMemLen = utils::roundUp(minIndicesLen, alignLen);
    AscendTensor<int8_t, DIMS_1, size_t>minIndicesMem(mem, { minIndicesMemLen }, stream);
    AscendTensor<int64_t, DIMS_2> minIndices(reinterpret_cast<int64_t *>(minIndicesMem.data()), { n, k });

    int idxMaskLen = static_cast<int>(utils::divUp(this->ntotal, BINARY_BYTE_SIZE));
    // operator has verification, mask must be blockMaskSize at least.
    int idxMaskLen2 = this->blockMaskSize;
    if (this->maskData != nullptr) {
        idxMaskLen2 = std::max(idxMaskLen, this->blockMaskSize);
    }
    AscendTensor<uint8_t, DIMS_2> mask(mem, { n, idxMaskLen2 }, stream);
    if (this->maskData != nullptr) {
        auto ret = aclrtMemcpy(mask.data(), static_cast<size_t>(n) * static_cast<size_t>(idxMaskLen2),
                               this->maskData + this->maskSearchedOffset,
                               static_cast<size_t>(n) * static_cast<size_t>(idxMaskLen), ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "aclrtMemcpy copy mask failed: %i\n", ret);
    }

    // 2. compute distance by code page
    int pageNum = static_cast<int>(utils::divUp(this->ntotal, (size_t)pageSize));
    for (int pageId = 0; pageId < pageNum; ++pageId) {
        APP_ERROR ret = searchPaged(pageId, queries, k, minDistances, minIndices, mask);
        APPERR_RETURN_IF(ret, ret);
    }
    // memcpy data back from dev to host
    auto ret = aclrtMemcpy(outDistances.data(), outDistances.getSizeInBytes(), minDistances.data(),
                           minDistances.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy outDistances back to host");

    ret = aclrtMemcpy(outIndices.data(), outIndices.getSizeInBytes(), minIndices.data(), minIndices.getSizeInBytes(),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy outDistances back to host");
    return APP_ERR_OK;
}

template<typename P>
APP_ERROR IndexInt8Flat<P>::searchPaged(int pageId, AscendTensor<int8_t, DIMS_2> &queries, int k,
    AscendTensor<float16_t, DIMS_2> &minDistances, AscendTensor<int64_t, DIMS_2> &minIndices,
    AscendTensor<uint8_t, DIMS_2> &mask)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    int nq = queries.getSize(0);
    int pageOffset = pageId * this->pageSize;
    int blockOffset = pageId * this->pageSize / codeBlockSize;
    int computeNum = std::min(this->ntotal - pageOffset, static_cast<idx_t>(this->pageSize));
    int blockNum = utils::divUp(computeNum, this->codeBlockSize);

    AscendTensor<float16_t, DIMS_3, size_t> distResult(
        mem, {static_cast<size_t>(blockNum), static_cast<size_t>(nq), static_cast<size_t>(codeBlockSize)}, stream);
    AscendTensor<float16_t, DIMS_3, size_t> minDistResult(mem,
        {static_cast<size_t>(blockNum), static_cast<size_t>(nq), static_cast<size_t>(this->burstsOfBlock)},
        stream);
    // 这里必须在minDistResult后申请一个缓存内存，原因如下：
    // ascendc算子中DataCopy一次拷贝32字节，但极值采用一次拷贝16字节的方式拷贝到结果内存中，在最后一次拷贝时，
    // 极值内存仅占16字节，导致内存拷贝会越界，超过极值内存16字节，超过的16字节全部写0，而极值内存与opSize内存相邻，
    // 越界后污染opSize内容，导致结果问题，这里申请一个512(共享内存512对齐)字节的缓存区域，保证后续内存数据正常
    constexpr uint16_t paddingSize = 512;
    AscendTensor<uint8_t, DIMS_1> paddingMem(mem, {paddingSize}, stream);
    AscendTensor<uint32_t, DIMS_3> opSize(mem, {blockNum, CORE_NUM, SIZE_ALIGN}, stream);

    uint32_t opFlagSize = static_cast<size_t>((blockNum * flagNum * FLAG_SIZE)) * sizeof(uint16_t);
    uint32_t attrsSize = aicpu::TOPK_FLAT_ATTR_IDX_COUNT * sizeof(int64_t);
    uint32_t continuousMemSize = opFlagSize + attrsSize;
    AscendTensor<uint8_t, DIMS_1, uint32_t> continuousMem(mem, { continuousMemSize }, stream);
    std::vector<uint8_t> continuousValue(continuousMemSize, 0);
    uint8_t *data = continuousValue.data();
    int64_t *attrs = reinterpret_cast<int64_t *>(data + opFlagSize);
    // attrs: [0]asc, [1]k, [2]burst_len, [3]block_num
    int pageNum = static_cast<int>(utils::divUp(this->ntotal, static_cast<idx_t>(this->pageSize)));
    attrs[aicpu::TOPK_FLAT_ATTR_ASC_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_K_IDX] = k;
    attrs[aicpu::TOPK_FLAT_ATTR_BURST_LEN_IDX] = BURST_LEN;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_NUM_IDX] = blockNum;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_IDX] = pageId;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_NUM_IDX] = pageNum;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_SIZE_IDX] = this->pageSize;
    attrs[aicpu::TOPK_FLAT_ATTR_QUICK_HEAP] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_SIZE] = this->codeBlockSize;
    auto ret = aclrtMemcpy(continuousMem.data(), continuousMem.getSizeInBytes(), continuousValue.data(),
        continuousValue.size() * sizeof(uint8_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

    uint16_t *opFlagMem = reinterpret_cast<uint16_t *>(continuousMem.data());
    AscendTensor<uint16_t, DIMS_3> opFlag(opFlagMem, { blockNum, flagNum, FLAG_SIZE });
    int64_t *attrMem = reinterpret_cast<int64_t *>(continuousMem.data() + opFlagSize);
    AscendTensor<int64_t, DIMS_1> attrsInput(attrMem, { aicpu::TOPK_FLAT_ATTR_IDX_COUNT });

    // 1. run the topk operator to wait for distance result and compute topk
    runTopkCompute(distResult, minDistResult, opSize, opFlag, attrsInput, minDistances, minIndices, streamAicpu);

    // 2. run the disance operator to compute the distance
    uint32_t idxMaskLen = static_cast<uint32_t>(utils::divUp(this->ntotal, BINARY_BYTE_SIZE));
    // opSize Host to Device,reduce communication
    std::vector<uint32_t> opSizeHost(blockNum * CORE_NUM * SIZE_ALIGN);
    int opSizeHostOffset = CORE_NUM * SIZE_ALIGN;
    int opSizeHostIdx = 0;
    int offset = 0;
    uint32_t idxUseMask = (this->maskData != nullptr) ? 1 : 0;
    for (int i = 0; i < blockNum; i++) {
        opSizeHost[opSizeHostIdx + IDX_ACTUAL_NUM] =
            std::min(static_cast<uint32_t>(computeNum - offset), static_cast<uint32_t>(this->codeBlockSize));
        opSizeHost[opSizeHostIdx + IDX_COMP_OFFSET] = static_cast<uint32_t>(pageOffset) + static_cast<uint32_t>(offset);
        opSizeHost[opSizeHostIdx + IDX_MASK_LEN] = idxMaskLen;
        opSizeHost[opSizeHostIdx + IDX_USE_MASK] = idxUseMask;
        opSizeHostIdx += opSizeHostOffset;
        offset += this->codeBlockSize;
    }
    ret = aclrtMemcpy(opSize.data(), opSize.getSizeInBytes(), opSizeHost.data(), opSizeHost.size() * sizeof(uint32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy op size to device");

    opSizeHostIdx = 0;
    const int dim1 = utils::divUp(this->codeBlockSize, CUBE_ALIGN);
    const int dim2 = utils::divUp(this->dims, CUBE_ALIGN_INT8);
    for (int i = 0; i < blockNum; ++i) {
        AscendTensor<int8_t, DIMS_4> shaped(baseShaped[blockOffset + i]->data(),
            { dim1, dim2, CUBE_ALIGN, CUBE_ALIGN_INT8 });
        AscendTensor<P, DIMS_1> norm(normBase[blockOffset + i]->data(), { codeBlockSize });
        auto dist = distResult[i].view();
        auto minDist = minDistResult[i].view();
        auto flag = opFlag[i].view();
        auto actualSize = opSize[i].view();
        uint32_t actualNum = opSizeHost[opSizeHostIdx + IDX_ACTUAL_NUM];
        if (isNeedCleanMinDist) {
            minDist.zero();
        }

        std::vector<const AscendTensorBase *> input {&queries, &mask, &shaped, &norm, &actualSize};
        std::vector<const AscendTensorBase *> output {&dist, &minDist, &flag};
        runDistCompute(nq, input, output, stream, actualNum);
        opSizeHostIdx += opSizeHostOffset;
    }

    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream default stream: %i\n",
        ret);

    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream aicpu stream failed: %i\n", ret);

    return APP_ERR_OK;
}

template<typename P>
APP_ERROR IndexInt8Flat<P>::searchImpl(std::vector<IndexInt8 *> indexes, int n, int batchSize, const int8_t *x, int k,
    float16_t *distances, idx_t *labels)
{
    if (!this->isSupportMultiSearch) { return APP_ERR_OK; }

    int indexSize = static_cast<int64_t>(indexes.size());
    std::vector<idx_t> ntotals(indexSize);
    std::vector<idx_t> offsetBlocks(indexSize + 1, 0);
    for (int i = 0; i < indexSize; ++i) {
        ntotals[i] = indexes[i]->ntotal;
        offsetBlocks[i + 1] = offsetBlocks[i] + utils::divUp(ntotals[i], static_cast<idx_t>(codeBlockSize));
    }

    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    int blockNum = static_cast<int>(offsetBlocks[indexSize]);

    // 1. costruct the distance operator param
    // 1.1 aicore params
    AscendTensor<int8_t, DIMS_2> queries(const_cast<int8_t *>(x), { batchSize, dims });
    AscendTensor<uint32_t, DIMS_3> opSize(mem, { PAGE_BLOCKS, CORE_NUM, SIZE_ALIGN }, stream);
    AscendTensor<float16_t, DIMS_1> queriesNorm;
    if (this->int8FlatIndexType == Int8FlatIndexType::INT8_FLAT_COS) {
        AscendTensor<float16_t, DIMS_1> tmpQueriesNorm(mem, { utils::roundUp(batchSize, CUBE_ALIGN) }, stream);
        AscendTensor<uint32_t, DIMS_2> actualNum(mem, { utils::divUp(batchSize, L2NORM_COMPUTE_BATCH), SIZE_ALIGN },
            stream);
        int8L2Norm->dispatchL2NormTask(queries, tmpQueriesNorm, actualNum, stream);
        auto ret = synchronizeStream(stream);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
            "synchronizeStream default stream: %i", ret);
        queriesNorm = std::move(tmpQueriesNorm);
    }
    AscendTensor<float16_t, DIMS_3, size_t> distResult(mem,
        { static_cast<size_t>(PAGE_BLOCKS), static_cast<size_t>(batchSize),
        static_cast<size_t>(codeBlockSize) }, stream);
    AscendTensor<float16_t, DIMS_3, size_t> mvDistResult(mem,
        { static_cast<size_t>(PAGE_BLOCKS), static_cast<size_t>(batchSize),
        static_cast<size_t>(burstsOfBlock) }, stream);
    AscendTensor<float16_t, DIMS_2> topKThreshold;

    // 1.2 aicpu params
    AscendTensor<uint32_t, DIMS_1> indexOffsetInputs(mem, { blockNum }, streamAicpu);
    AscendTensor<uint32_t, DIMS_1> labelOffsetInputs(mem, { blockNum }, streamAicpu);
    AscendTensor<uint16_t, DIMS_1> reorderFlagInputs(mem, { blockNum }, streamAicpu);

    uint32_t opFlagSize = static_cast<uint32_t>(blockNum * flagNum * FLAG_SIZE) * sizeof(uint16_t);
    uint32_t attrsSize = aicpu::TOPK_MULTISEARCH_ATTR_IDX_COUNT * sizeof(int64_t);
    uint32_t continuousMemSize = opFlagSize + attrsSize;
    // 1) aclrtMemcpy比AscendTensor::zero更高效
    // 2) 使用连续内存来减少aclrtMemcpy的调用次数
    AscendTensor<uint8_t, DIMS_1, uint32_t> continuousMem(mem, { continuousMemSize }, stream);
    std::vector<uint8_t> continuousValue(continuousMemSize, 0);
    uint8_t *data = continuousValue.data();
    int64_t *attrs = reinterpret_cast<int64_t *>(data + opFlagSize);
    attrs[aicpu::TOPK_MULTISEARCH_ATTR_ASC_IDX] =
        this->int8FlatIndexType == Int8FlatIndexType::INT8_FLAT_COS ? 0 : 1;
    attrs[aicpu::TOPK_MULTISEARCH_ATTR_K_IDX] = k;
    attrs[aicpu::TOPK_MULTISEARCH_ATTR_BURST_LEN_IDX] = BURST_LEN;
    attrs[aicpu::TOPK_MULTISEARCH_ATTR_INDEX_NUM_IDX] = indexSize;
    attrs[aicpu::TOPK_MULTISEARCH_ATTR_PAGE_BLOCK_NUM_IDX] = PAGE_BLOCKS;
    attrs[aicpu::TOPK_MULTISEARCH_ATTR_QUICK_HEAP] = 1;
    attrs[aicpu::TOPK_MULTISEARCH_ATTR_BLOCK_SIZE] = this->codeBlockSize;

    auto ret = aclrtMemcpy(continuousMem.data(), continuousMem.getSizeInBytes(),
                           continuousValue.data(), continuousValue.size() * sizeof(uint8_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %i", ret);

    uint16_t *opFlagMem = reinterpret_cast<uint16_t *>(continuousMem.data());
    AscendTensor<uint16_t, DIMS_3> opFlag(opFlagMem, { blockNum, flagNum, FLAG_SIZE });
    int64_t *attrMem = reinterpret_cast<int64_t *>(continuousMem.data() + opFlagSize);
    AscendTensor<int64_t, DIMS_1> attrsInputs(attrMem, { aicpu::TOPK_MULTISEARCH_ATTR_IDX_COUNT });

    APPERR_RETURN_IF_NOT_OK(
        computeMultisearchTopkParam(indexOffsetInputs, labelOffsetInputs, reorderFlagInputs, ntotals, offsetBlocks));
    // 1.3 resultTemp
    AscendTensor<float16_t, DIMS_3, size_t> mvDistances(mem,
        { static_cast<size_t>(indexSize), static_cast<size_t>(batchSize), static_cast<size_t>(k) },
        streamAicpu);
    AscendTensor<idx_t, DIMS_3, size_t> mvIndices(mem,
        { static_cast<size_t>(indexSize), static_cast<size_t>(batchSize), static_cast<size_t>(k) },
        streamAicpu);
    APPERR_RETURN_IF_NOT_OK(initResult(mvDistances,  mvIndices));

    // 2. run the disance operators and topk operators
    AscendTensor<uint8_t, DIMS_2> mask(mem, { batchSize, this->blockMaskSize }, stream);
    const int blockSizeShapNum = utils::divUp(this->codeBlockSize, CUBE_ALIGN);
    const int dimsShapeNum = utils::divUp(this->dims, CUBE_ALIGN_INT8);
    int blockOffset = 0;
    int indexDoneCount = 0;
    int copyCount = 0;
    for (int indexId = 0; indexId < indexSize; ++indexId) {
        IndexInt8Flat *index = dynamic_cast<IndexInt8Flat *>(indexes[indexId]);

        int blocks = static_cast<int64_t>(utils::divUp(ntotals[indexId], this->codeBlockSize));
        for (int i = 0; i < blocks; ++i) {
            int offset = i * this->codeBlockSize;
            int blockIdx = (static_cast<int64_t>(offsetBlocks[indexId]) + i) % PAGE_BLOCKS;
            if (blockIdx == 0 && (static_cast<int64_t>(offsetBlocks[indexId]) + i) > 0) {
                APP_ERROR syncRetCode = tryToSychResultAdvanced(copyCount, indexDoneCount, indexId, n, batchSize, k,
                                                                distances, labels, mvDistances, mvIndices);
                APPERR_RETURN_IF_NOT_FMT(syncRetCode == APP_ERR_OK, syncRetCode,
                    "try to copy result in advanced failed:%i", syncRetCode);
            }
            if (blockIdx == 0) {
                // under topk operators
                int actualPageBlocks = std::min(blockNum - blockOffset, PAGE_BLOCKS);
                attrsInputs[aicpu::TOPK_MULTISEARCH_ATTR_PAGE_BLOCK_NUM_IDX] = actualPageBlocks;
                AscendTensor<uint32_t, DIMS_1> indexOffset(indexOffsetInputs.data() + blockOffset,
                    { actualPageBlocks });
                AscendTensor<uint32_t, DIMS_1> labelOffset(labelOffsetInputs.data() + blockOffset,
                    { actualPageBlocks });
                AscendTensor<uint16_t, DIMS_1> reorderFlag(reorderFlagInputs.data() + blockOffset,
                    { actualPageBlocks });
                AscendTensor<uint16_t, DIMS_3> flag(opFlag.data() + blockOffset * flagNum * FLAG_SIZE,
                    { actualPageBlocks, flagNum, FLAG_SIZE });

                runMultisearchTopkCompute(distResult,  mvDistResult, opSize, flag, attrsInputs, indexOffset,
                    labelOffset, reorderFlag,  mvDistances,  mvIndices, streamAicpu);
                blockOffset += actualPageBlocks;
            }

            // under distance operators
            AscendTensor<int8_t, DIMS_4> shaped(index->baseShaped[i]->data(),
                { blockSizeShapNum, dimsShapeNum, CUBE_ALIGN, CUBE_ALIGN_INT8 });
            AscendTensor<uint32_t, DIMS_2> actualSize(opSize[blockIdx].data(), { CORE_NUM, SIZE_ALIGN });
            AscendTensor<float16_t, DIMS_2, size_t> result = distResult[blockIdx].view();
            AscendTensor<float16_t, DIMS_2, size_t> mvResult = mvDistResult[blockIdx].view();
            AscendTensor<P, DIMS_1> codesNorm(index->normBase[i]->data(), { this->codeBlockSize });
            AscendTensor<uint16_t, DIMS_2> flag(opFlag[offsetBlocks[indexId] + i].data(), { flagNum, FLAG_SIZE });
            if (isNeedCleanMinDist) {
                mvResult.zero();
            }

            uint32_t actualNum = std::min(static_cast<uint32_t>(index->ntotal - offset),
                                          static_cast<uint32_t>(codeBlockSize));
            std::vector<uint32_t> actualVec(SIZE_ALIGN, 0);
            // only use IDX_ACTUAL_NUM, IDX_USE_MASK in operator when do multi search, and IDX_USE_MASK value is 0.
            actualVec[IDX_ACTUAL_NUM] = actualNum;
            actualVec[IDX_USE_MASK] = 0; // not use mask
            auto ret = aclrtMemcpy(actualSize[0].data(), static_cast<size_t>(SIZE_ALIGN) * sizeof(uint32_t),
                actualVec.data(), static_cast<size_t>(SIZE_ALIGN) * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
            APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                "Failed to copy actualSize[0] to device");

            std::vector<const AscendTensorBase *> input {&queries, &mask, &shaped,
                                                         &codesNorm, &actualSize};
            if (this->int8FlatIndexType == Int8FlatIndexType::INT8_FLAT_COS) {
                int posAtInput = 3;
                input.insert(input.begin() + posAtInput, &queriesNorm);
            }
            if (this->int8FlatIndexType == Int8FlatIndexType::INT8_FLAT_COS
                && deviceMemMng.GetStrategy() == DevMemStrategy::HETERO_MEM) {
                input.push_back(&topKThreshold);
            }
            std::vector<const AscendTensorBase *> output {&result, &mvResult, &flag};
            runDistCompute(batchSize, input, output, stream, actualNum);
        }
    }

    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream aicpu stream failed: %i", ret);
    // memcpy data back from dev to host
    for (int indexId = copyCount; indexId < indexSize; ++indexId) {
        ret = aclrtMemcpy(distances + static_cast<size_t>(indexId) * static_cast<size_t>(n) * static_cast<size_t>(k),
            static_cast<size_t>(batchSize) * static_cast<size_t>(k) * sizeof(float16_t),  mvDistances[indexId].data(),
            static_cast<size_t>(batchSize) * static_cast<size_t>(k) * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %i", ret);

        ret = aclrtMemcpy(labels + static_cast<size_t>(indexId) * static_cast<size_t>(n) * static_cast<size_t>(k),
            static_cast<size_t>(batchSize) * static_cast<size_t>(k) * sizeof(idx_t),  mvIndices[indexId].data(),
            static_cast<size_t>(batchSize) * static_cast<size_t>(k) * sizeof(idx_t), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %i", ret);
    }

    return APP_ERR_OK;
}

template<typename P>
APP_ERROR IndexInt8Flat<P>::initResult(AscendTensor<float16_t, DIMS_3, size_t> &distances,
    AscendTensor<idx_t, DIMS_3, size_t> &indices) const
{
    std::vector<float16_t> distancesInit(distances.getSizeInBytes() / sizeof(float16_t),
        this->int8FlatIndexType == Int8FlatIndexType::INT8_FLAT_COS ? Limits<float16_t>::getMin()
            : Limits<float16_t>::getMax());
    std::vector<idx_t> indicesInit(indices.getSizeInBytes() / sizeof(idx_t), std::numeric_limits<idx_t>::max());

    auto ret = aclrtMemcpy(distances.data(), distances.getSizeInBytes(), distancesInit.data(),
        distancesInit.size() * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %i", ret);
    ret = aclrtMemcpy(indices.data(), indices.getSizeInBytes(), indicesInit.data(),
        indicesInit.size() * sizeof(idx_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %i", ret);

    return APP_ERR_OK;
}

// the method is not used for IndexIntFlatCos
template<typename P>
void IndexInt8Flat<P>::computeNorm(AscendTensor<int8_t, DIMS_2> &rawData)
{
    int num = rawData.getSize(0);
    int dim = rawData.getSize(1);

    bool isFirst = true;
    int idx = 0;
    for (int i = 0; i < num; i++) {
        int idx1 = ((int)this->ntotal + i) / codeBlockSize;
        int idx2 = ((int)this->ntotal + i) % codeBlockSize;

        // if the baseShapedSlice is full or reach the last
        if (idx2 + 1 == codeBlockSize || i == num - 1) {
            P *pNormBaseSlice = normBase[idx1]->data();

            // calc y^2 (the first time is different)
            if (isFirst) {
                ivecNormsL2sqr(pNormBaseSlice + (int)this->ntotal % codeBlockSize,
                    rawData[idx][0].data(), dim, i + 1);
                idx += (i + 1);
                isFirst = false;
            } else {
                ivecNormsL2sqr(pNormBaseSlice, rawData[idx][0].data(), dim, idx2 + 1);
                idx += (idx2 + 1);
            }
        }
    }
}

template<typename P>
P IndexInt8Flat<P>::ivecNormL2sqr(const int8_t *x, size_t d)
{
    P res = 0;
    for (size_t i = 0; i < d; i++) {
        res += x[i] * x[i];
    }
    return res;
}

template<typename P>
void IndexInt8Flat<P>::ivecNormsL2sqr(P *nr, const int8_t *x, size_t d, size_t nx)
{
#ifdef HOSTCPU
    std::vector<P> norms(nx);
#pragma omp parallel for num_threads(CommonUtils::GetThreadMaxNums())
    for (size_t i = 0; i < nx; i++) {
        norms[i] = ivecNormL2sqr(x + i * d, d);
    }
    auto err = aclrtMemcpy(nr, nx * sizeof(P), norms.data(), nx * sizeof(P), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(err == EOK, "Mem operator error %d", (int)err);
#else
#pragma omp parallel for num_threads(CommonUtils::GetThreadMaxNums())
    for (size_t i = 0; i < nx; i++) {
        nr[i] = ivecNormL2sqr(x + i * d, d);
    }
#endif
}

template<typename P>
void IndexInt8Flat<P>::moveNormForward(idx_t srcIdx, idx_t dstIdx)
{
    ASCEND_THROW_IF_NOT(srcIdx >= dstIdx);
    size_t blockSizeLong = (size_t)this->codeBlockSize;
    size_t srcIdx1 = srcIdx / blockSizeLong;
    size_t srcIdx2 = srcIdx % blockSizeLong;
    size_t dstIdx1 = dstIdx / blockSizeLong;
    size_t dstIdx2 = dstIdx % blockSizeLong;
    
    auto err = aclrtMemcpy(normBase[dstIdx1]->data() + dstIdx2, sizeof(P),
                           normBase[srcIdx1]->data() + srcIdx2, sizeof(P),
                           ACL_MEMCPY_DEVICE_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(err == EOK, "Mem error %d", (int)err);
}

template<typename P>
void IndexInt8Flat<P>::moveShapedForward(idx_t srcIdx, idx_t dstIdx)
{
    ASCEND_THROW_IF_NOT(srcIdx >= dstIdx);
    int srcIdx1 = (int)srcIdx / this->codeBlockSize;
    int srcIdx2 = (int)srcIdx % this->codeBlockSize;

    int dstIdx1 = (int)dstIdx / this->codeBlockSize;
    int dstIdx2 = (int)dstIdx % this->codeBlockSize;

    if (!faiss::ascend::SocUtils::GetInstance().IsZZCodeFormat()) {
        RemoveForwardParam param = {
            static_cast<size_t>(srcIdx1), static_cast<size_t>(srcIdx2),
            static_cast<size_t>(dstIdx1), static_cast<size_t>(dstIdx2)
        };
        auto ret = RemoveForwardNDFormat(param, dims, baseShaped);
        ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "RemoveForwardNDFormat error %d", ret);
        return;
    }

    int dim2 = utils::divUp(this->dims, CUBE_ALIGN_INT8);

    int srcOffset = (srcIdx2 / CUBE_ALIGN) * (dim2 * CUBE_ALIGN * CUBE_ALIGN_INT8) +
        (srcIdx2 % CUBE_ALIGN) * (CUBE_ALIGN_INT8);
    int dstOffset = (dstIdx2 / CUBE_ALIGN) * (dim2 * CUBE_ALIGN * CUBE_ALIGN_INT8) +
        (dstIdx2 % CUBE_ALIGN) * (CUBE_ALIGN_INT8);
    for (int i = 0; i < dim2; i++) {
        auto err = deviceMemMng.MemoryCopy(*baseShaped[dstIdx1], static_cast<size_t>(dstOffset),
            *baseShaped[srcIdx1], static_cast<size_t>(srcOffset), CUBE_ALIGN_INT8);
        ASCEND_THROW_IF_NOT_FMT(err == EOK, "Mem error %d", (int)err);
        dstOffset += CUBE_ALIGN * CUBE_ALIGN_INT8;
        srcOffset += CUBE_ALIGN * CUBE_ALIGN_INT8;
    }
}

template<typename P>
size_t IndexInt8Flat<P>::removeIdsBatch(const std::vector<idx_t> &indices)
{
    // 1. filter the same id
    std::set<idx_t> filtered;
    for (auto idx : indices) {
        if (idx < this->ntotal) {
            filtered.insert(idx);
        }
    }

    // 2. sort by DESC, then delete from the big to small
    std::vector<idx_t> sortData(filtered.begin(), filtered.end());
    std::sort(sortData.begin(), sortData.end(), std::greater<idx_t>());

    // 3. move the end data to the locate of delete data
    idx_t oldTotal = this->ntotal;
    for (const auto index : sortData) {
        moveVectorForward(this->ntotal - 1, index);
        --this->ntotal;
    }

    // 4. release the space of unusage
    size_t removedCnt = filtered.size();
    removeInvalidData(oldTotal, removedCnt);

    return removedCnt;
}

template<typename P>
size_t IndexInt8Flat<P>::removeIdsRange(idx_t min, idx_t max)
{
    // 1. check param
    if (min >= max || min >= this->ntotal) {
        return 0;
    }

    if (max > this->ntotal) {
        max = this->ntotal;
    }

    // 2. move the end data to the locate of delete data(delete from back to front)
    size_t oldTotal = this->ntotal;
    for (idx_t i = 1; i <= max - min; ++i) {
        moveVectorForward(this->ntotal - 1, max - i);
        --this->ntotal;
    }

    // 3. release the space of unusage
    size_t removeCnt = max - min;
    removeInvalidData(oldTotal, removeCnt);

    return removeCnt;
}

template<typename P>
size_t IndexInt8Flat<P>::removeIdsImpl(const ascend::IDSelector &sel)
{
    size_t removeCnt = 0;

    try {
        const ascend::IDSelectorBatch &batch = dynamic_cast<const ascend::IDSelectorBatch &>(sel);
        std::vector<idx_t> buf(batch.set.begin(), batch.set.end());
        removeCnt = removeIdsBatch(buf);
    } catch (std::bad_cast &e) {
        // ignore
    }

    try {
        const ascend::IDSelectorRange &range = dynamic_cast<const ascend::IDSelectorRange &>(sel);
        removeCnt = removeIdsRange(range.imin, range.imax);
    } catch (std::bad_cast &e) {
        // ignore
    }

    return removeCnt;
}

template<typename P>
void IndexInt8Flat<P>::removeInvalidData(int oldTotal, int remove)
{
    int oldVecSize = utils::divUp(oldTotal, this->codeBlockSize);
    int vecSize = utils::divUp(oldTotal - remove, this->codeBlockSize);

    for (int i = oldVecSize - 1; i >= vecSize; --i) {
        this->baseShaped.at(i)->clear();
        this->normBase.at(i)->clear();
    }
}

template<typename P>
size_t IndexInt8Flat<P>::calcShapedBaseSize(idx_t totalNum)
{
    size_t numBatch = utils::divUp(totalNum, (size_t)codeBlockSize);
    int dim1 = utils::divUp(codeBlockSize, CUBE_ALIGN);
    int dim2 = utils::divUp(this->dims, CUBE_ALIGN_INT8);
    return numBatch * (size_t)(dim1 * dim2 * CUBE_ALIGN * CUBE_ALIGN_INT8);
}

template<typename P>
size_t IndexInt8Flat<P>::calcNormBaseSize(idx_t totalNum)
{
    size_t numBatch = utils::divUp(totalNum, codeBlockSize);
    return numBatch * (size_t)codeBlockSize;
}

template<typename P>
APP_ERROR IndexInt8Flat<P>::resetTopkCompOp()
{
    auto topkCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("TopkFlat");
        std::vector<int64_t> shapeOfIndists { 0, batch, this->codeBlockSize };
        std::vector<int64_t> shapeOfVmdists { 0, batch, this->burstsOfBlock };
        std::vector<int64_t> shapeOfSize { 0, CORE_NUM, SIZE_ALIGN };
        std::vector<int64_t> sizeOfOpflag { 0, flagNum, FLAG_SIZE };
        std::vector<int64_t> shapeOfAttr { aicpu::TOPK_FLAT_ATTR_IDX_COUNT };
        std::vector<int64_t> shapeOfOutputs { batch, 0 };

        desc.addInputTensorDesc(ACL_FLOAT16, shapeOfIndists.size(), shapeOfIndists.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, shapeOfVmdists .size(), shapeOfVmdists .data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, shapeOfSize.size(), shapeOfSize.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, sizeOfOpflag.size(), sizeOfOpflag.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shapeOfAttr.size(), shapeOfAttr.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT16, shapeOfOutputs.size(), shapeOfOutputs.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_INT64, shapeOfOutputs.size(), shapeOfOutputs.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : searchBatchSizes) {
        topkComputeOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(topkCompOpReset(topkComputeOps[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
            "topk op init failed");
    }

    return APP_ERR_OK;
}
 
template<typename P>
void IndexInt8Flat<P>::runTopkCompute(AscendTensor<float16_t, DIMS_3, size_t> &dists,
    AscendTensor<float16_t, DIMS_3, size_t> &mindists, AscendTensor<uint32_t, DIMS_3> &sizes,
    AscendTensor<uint16_t, DIMS_3> &flags, AscendTensor<int64_t, DIMS_1> &attrs,
    AscendTensor<float16_t, DIMS_2> &outdists, AscendTensor<int64_t, DIMS_2> &outlabel, aclrtStream stream)
{
    AscendOperator *op = nullptr;
    int batch = static_cast<int64_t>(dists.getSize(1));
    if (topkComputeOps.find(batch) != topkComputeOps.end()) {
        op = topkComputeOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);
 
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(mindists.data(), mindists.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(sizes.data(), sizes.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(flags.data(), flags.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(attrs.data(), attrs.getSizeInBytes()));
 
    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(outdists.data(), outdists.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(outlabel.data(), outlabel.getSizeInBytes()));
 
    op->exec(*topkOpInput, *topkOpOutput, stream);
}

template<typename P>
void IndexInt8Flat<P>::runMultisearchTopkCompute(AscendTensor<float16_t, DIMS_3, size_t> &dists,
    AscendTensor<float16_t, DIMS_3, size_t> &maxDists,
    AscendTensor<uint32_t, DIMS_3> &sizes,
    AscendTensor<uint16_t, DIMS_3> &flags,
    AscendTensor<int64_t, DIMS_1> &attrs,
    AscendTensor<uint32_t, DIMS_1> &indexOffset,
    AscendTensor<uint32_t, DIMS_1> &pageOffset,
    AscendTensor<uint16_t, DIMS_1> &reorderFlag,
    AscendTensor<float16_t, DIMS_3, size_t> &outDists,
    AscendTensor<idx_t, DIMS_3, size_t> &outLabel,
    aclrtStream stream)
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(multiSearchTopkMtx);
    int batch = static_cast<int>(dists.getSize(1));
    std::map<OpsMngKey, std::unique_ptr<AscendOperator>>& multiSearchtopkComputeOps =
        DistComputeOpsManager::getInstance().getDistComputeOps(IndexTypeIdx::ITI_TOPK_MULTISEARCH);
    std::vector<int> keys({batch, codeBlockSize});
    OpsMngKey opsKey(keys);
    AscendOperator *op = multiSearchtopkComputeOps[opsKey].get();
    ASCEND_THROW_IF_NOT(op);
 
    std::shared_ptr<std::vector<const aclDataBuffer *>> topkOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    topkOpInput->emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(maxDists.data(), maxDists.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(sizes.data(), sizes.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(flags.data(), flags.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(attrs.data(), attrs.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(indexOffset.data(), indexOffset.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(pageOffset.data(), pageOffset.getSizeInBytes()));
    topkOpInput->emplace_back(aclCreateDataBuffer(reorderFlag.data(), reorderFlag.getSizeInBytes()));
 
    std::shared_ptr<std::vector<aclDataBuffer *>> topkOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    topkOpOutput->emplace_back(aclCreateDataBuffer(outDists.data(), outDists.getSizeInBytes()));
    topkOpOutput->emplace_back(aclCreateDataBuffer(outLabel.data(), outLabel.getSizeInBytes()));
 
    op->exec(*topkOpInput, *topkOpOutput, stream);
}

template<typename P>
APP_ERROR IndexInt8Flat<P>::computeMultisearchTopkParam(AscendTensor<uint32_t, DIMS_1> &indexOffsetInputs,
    AscendTensor<uint32_t, DIMS_1> &labelOffsetInputs, AscendTensor<uint16_t, DIMS_1> &reorderFlagInputs,
    std::vector<idx_t> &ntotals, std::vector<idx_t> &offsetBlocks) const
{
    size_t indexSize = ntotals.size();
    idx_t blockNum = offsetBlocks[indexSize];
    std::vector<uint32_t> indexOffset(blockNum);
    std::vector<uint32_t> labelOffset(blockNum);
    std::vector<uint16_t> reorderFlag(blockNum);
 
    for (size_t indexId = 0; indexId < indexSize; ++indexId) {
        idx_t blocks = utils::divUp(ntotals[indexId], static_cast<idx_t>(codeBlockSize));
        for (idx_t i = 0; i < blocks; ++i) {
            idx_t blockIdx = (offsetBlocks[indexId] + i);
            indexOffset[blockIdx] = static_cast<uint32_t>(indexId);
            labelOffset[blockIdx] = static_cast<uint32_t>(i);
            reorderFlag[blockIdx] = (i == blocks - 1) ? 1 : 0;
        }
    }
 
    auto ret = aclrtMemcpy(indexOffsetInputs.data(), indexOffsetInputs.getSizeInBytes(), indexOffset.data(),
        indexOffset.size() * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);
    ret = aclrtMemcpy(labelOffsetInputs.data(), labelOffsetInputs.getSizeInBytes(), labelOffset.data(),
        labelOffset.size() * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);
    ret = aclrtMemcpy(reorderFlagInputs.data(), reorderFlagInputs.getSizeInBytes(), reorderFlag.data(),
        reorderFlag.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);
 
    return APP_ERR_OK;
}

template<typename P>
APP_ERROR IndexInt8Flat<P>::resetMultisearchTopkCompOp()
{
    auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(multiSearchTopkMtx);
    auto topkCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("TopkMultisearch");
        std::vector<int64_t> shape0 { 0, batch, this->codeBlockSize };
        std::vector<int64_t> shape1 { 0, batch, this->burstsOfBlock };
        std::vector<int64_t> shape2 { 0, CORE_NUM, SIZE_ALIGN };
        std::vector<int64_t> shape3 { 0, flagNum, FLAG_SIZE };
        std::vector<int64_t> shape4 { aicpu::TOPK_MULTISEARCH_ATTR_IDX_COUNT };
        std::vector<int64_t> shape5 { 0 };
        std::vector<int64_t> shape6 { 0, batch, 0 };
 
        desc.addInputTensorDesc(ACL_FLOAT16, shape0.size(), shape0.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, shape1.size(), shape1.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, shape2.size(), shape2.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, shape3.size(), shape3.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape4.size(), shape4.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, shape5.size(), shape5.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, shape5.size(), shape5.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, shape5.size(), shape5.data(), ACL_FORMAT_ND);
 
        desc.addOutputTensorDesc(ACL_FLOAT16, shape6.size(), shape6.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_INT64, shape6.size(), shape6.data(), ACL_FORMAT_ND);
 
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };
    std::map<OpsMngKey, std::unique_ptr<AscendOperator>>& topkComputeOps =
        DistComputeOpsManager::getInstance().getDistComputeOps(IndexTypeIdx::ITI_TOPK_MULTISEARCH);
    for (auto batch : searchBatchSizes) {
        std::vector<int> keys({batch, codeBlockSize});
        OpsMngKey opsKey(keys);
        topkComputeOps[opsKey] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(topkCompOpReset(topkComputeOps[opsKey], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
            "topk op init failed");
    }
    return APP_ERR_OK;
}

template class IndexInt8Flat<int32_t>;
template class IndexInt8Flat<float16_t>;
} // namespace ascend
