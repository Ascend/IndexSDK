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

#include <iostream>
#include "acl/acl_op_compiler.h"
#include "index/IndexFlatL2Aicpu.h"

#include "ascend/AscendIndex.h"
#include "ascenddaemon/utils/Limits.h"
#include "ascenddaemon/utils/AscendTensor.h"
#include "common/utils/CommonUtils.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"

namespace ascend {
namespace {
// opSize offset
constexpr int IDX_ACTUAL_NUM = 0;
constexpr int IDX_COMP_OFFSET = 1; // mask offset of code block
constexpr int IDX_MASK_LEN = 2;    // mask len of each query
constexpr int IDX_USE_MASK = 3;    // use mask flag
const int PAGE_BLOCKS = 32;
const int DIM_LOWER = 64;
const int DIM_UPPER = 1024;
}

IndexFlatL2Aicpu::IndexFlatL2Aicpu(int dim, int64_t resourceSize) : IndexFlat(dim, resourceSize) {}

IndexFlatL2Aicpu::~IndexFlatL2Aicpu()
{
    for (auto& pair: disL2OpInputDesc) {
        std::vector<aclTensorDesc*>& inputDesc = pair.second;
        for (auto item: inputDesc) {
            aclDestroyTensorDesc(item);
        }
    }
    for (auto& pair: disL2OpOutputDesc) {
        std::vector<aclTensorDesc*>& outputDesc = pair.second;
        for (auto item: outputDesc) {
            aclDestroyTensorDesc(item);
        }
    }
}

APP_ERROR IndexFlatL2Aicpu::init()
{
    if (faiss::ascend::SocUtils::GetInstance().IsAscend910B()) {
        searchBatchSizes = {96, 80, 64, 48, 36, 32, 30, 24, 18, 16, 12, 8, 6, 4, 2, 1};
        distOpName = "DistanceFlatL2";
        flagNum = CORE_NUM;
        isNeedCleanMinDist = true;
    } else {
        // 2048维度不支持大batch优化,32维无优化效果
        if (dims >= DIM_LOWER && dims <= DIM_UPPER) {
            searchBatchSizes = {96, 80, 64, 48, 36, 32, 30, 24, 18, 16, 12, 8, 6, 4, 2, 1};
        }
        distOpName = "DistanceComputeFlatMin64";
        flagNum = FLAG_NUM;
        isNeedCleanMinDist = false;
    }

    APP_ERROR ret = resetTopkCompOp();
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, ret, "failed to reset topk op");
    ret = resetMultisearchTopkCompOp();
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INVALID_PARAM, "failed to reset multisearch topk op", ret);
    return resetDistCompOp(blockSize);
}

APP_ERROR IndexFlatL2Aicpu::reset()
{
    int dvSize = utils::divUp(static_cast<int>(this->ntotal), this->blockSize);
    for (int i = 0; i < dvSize; ++i) {
        normBase.at(i)->clear();
    }

    return IndexFlat::reset();
}

APP_ERROR IndexFlatL2Aicpu::addVectors(AscendTensor<float16_t, DIMS_2> &rawData)
{
    // 1. save the rawData to shaped data
    APP_ERROR ret = IndexFlat::addVectors(rawData);
    APPERR_RETURN_IF(ret, ret);

    // 2. compute the norm data
    ret = computeNormHostBuffer(rawData);
    APPERR_RETURN_IF(ret, ret);

    this->ntotal += (idx_t)rawData.getSize(0);
    return APP_ERR_OK;
}

APP_ERROR IndexFlatL2Aicpu::computeNormHostBuffer(AscendTensor<float16_t, DIMS_2> &rawData)
{
    int num = rawData.getSize(0);
    int dim = rawData.getSize(1);
    int vecSize = utils::divUp((int)ntotal, this->blockSize);
    int addVecNum = utils::divUp((int)ntotal + num, this->blockSize) - vecSize;
    // 1. resize normBase
    for (int i = 0; i < addVecNum; ++i) {
        this->normBase.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<float16_t>, MemorySpace::DEVICE_HUGEPAGE));
        this->normBase.at(vecSize + i)->resize(this->blockSize, true);
    }

    // 2. compute the norm data
    bool isFirst = true;
    int idx = 0;
    for (int i = 0; i < num; i++) {
        int idx1 = ((int)ntotal + i) / blockSize;
        int idx2 = ((int)ntotal + i) % blockSize;

        // if the baseShapedSlice is full or reach the last
        if (idx2 + 1 == blockSize || i == num - 1) {
            float16_t *pNormBaseSlice;
            int nx;
            // calc y^2 (the first time is different)
            if (isFirst) {
                pNormBaseSlice = normBase[idx1]->data() + (int)ntotal % blockSize;
                nx = i + 1;
                isFirst = false;
            } else {
                pNormBaseSlice = normBase[idx1]->data();
                nx = idx2 + 1;
            }
            auto streamPtr = resources.getDefaultStream();
            auto stream = streamPtr->GetStream();
            auto &mem = resources.getMemoryManager();
            AscendTensor<float16_t, DIMS_1> nr(pNormBaseSlice, {nx});
            AscendTensor<float16_t, DIMS_2> x(mem, {nx, dim}, stream);
            
            auto ret = aclrtMemcpy(x.data(), x.getSizeInBytes(), rawData[idx][0].data(),
                (size_t)nx * (size_t)dim * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);
            
            fvecNormsL2sqrAicpu(nr, x);

            idx += nx;
        }
    }
    return APP_ERR_OK;
}

APP_ERROR IndexFlatL2Aicpu::searchImpl(AscendTensor<float16_t, DIMS_2> &queries, int k,
    AscendTensor<float16_t, DIMS_2> &outDistances, AscendTensor<idx_t, DIMS_2> &outIndices)
{
    // 1. generate result variable
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    int nq = queries.getSize(0);

    AscendTensor<float16_t, DIMS_2> minDistances(mem, {nq, k}, stream);
    AscendTensor<int64_t, DIMS_2> minIndices(mem, {nq, k}, stream);

    // 2. compute distance by code page
    size_t pageNum = utils::divUp(this->ntotal, pageSize);
    for (size_t pageId = 0; pageId < pageNum; ++pageId) {
        APP_ERROR ret = searchPaged(pageId, pageNum, queries, minDistances, minIndices);
        APPERR_RETURN_IF(ret, ret);
    }

    // memcpy data back from dev to host
    auto ret = aclrtMemcpy(outDistances.data(), outDistances.getSizeInBytes(),
                           minDistances.data(), minDistances.getSizeInBytes(),
                           ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy outDistances back to host");

    ret = aclrtMemcpy(outIndices.data(), outIndices.getSizeInBytes(),
                      minIndices.data(), minIndices.getSizeInBytes(),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy outIndices back to host");

    return APP_ERR_OK;
}

APP_ERROR IndexFlatL2Aicpu::searchPaged(size_t pageId, size_t pageNum, AscendTensor<float16_t, DIMS_2> &queries,
    AscendTensor<float16_t, DIMS_2> &minDistances, AscendTensor<int64_t, DIMS_2> &minIndices)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    int nq = queries.getSize(0);
    int k = minDistances.getSize(1);
    size_t pageOffset = pageId * (size_t)this->pageSize;
    size_t blockOffset = pageId * (size_t)this->pageSize / (size_t)blockSize;
    int computeNum = std::min(this->ntotal - pageOffset, static_cast<idx_t>(this->pageSize));
    int blockNum = utils::divUp(computeNum, this->blockSize);
    int burstLen = BURST_LEN_HIGH;
    auto curBurstsOfBlock = GetBurstsOfBlock(nq, this->blockSize, burstLen);
    AscendTensor<float16_t, DIMS_3, size_t> distResult(mem,
        { (size_t)blockNum, (size_t)nq, (size_t)blockSize }, stream);
    AscendTensor<float16_t, DIMS_3, size_t> minDistResult(mem,
        { (size_t)blockNum, (size_t)nq, (size_t)curBurstsOfBlock }, stream);

    // 这里必须在minDistResult后申请一个缓存内存，原因如下：
    // ascendc算子中DataCopy一次拷贝32字节，但极值采用一次拷贝16字节的方式拷贝到结果内存中，在最后一次拷贝时，
    // 极值内存仅占16字节，导致内存拷贝会越界，超过极值内存16字节，超过的16字节全部写0，而极值内存与opFlag内存相邻，
    // 越界后污染opFlag内容，导致结果问题，这里申请一个512(共享内存512对齐)字节的缓存区域，保证后续内存数据正常
    constexpr uint16_t paddingSize = 512;
    AscendTensor<uint8_t, DIMS_1> paddingMem(mem, {paddingSize}, stream);

    uint32_t opFlagSize = static_cast<uint32_t>(blockNum * flagNum * FLAG_SIZE) * sizeof(uint16_t);
    uint32_t attrsSize = aicpu::TOPK_FLAT_ATTR_IDX_COUNT * sizeof(int64_t);
    uint32_t opSizeLen = static_cast<uint32_t>(blockNum * CORE_NUM * SIZE_ALIGN) * sizeof(uint32_t);
    uint32_t continuousMemSize = opFlagSize + attrsSize + opSizeLen;
    // 1) aclrtMemcpy比AscendTensor::zero更高效
    // 2) 使用连续内存来减少aclrtMemcpy的调用次数
    AscendTensor<uint8_t, DIMS_1, uint32_t> continuousMem(mem, { continuousMemSize }, stream);
    std::vector<uint8_t> continuousValue(continuousMemSize, 0);
    uint8_t *data = continuousValue.data();

    // attrs: [0]asc, [1]k, [2]burst_len, [3]block_num [4]special page: -1:first page, 0:mid page, 1:last page
    int64_t *attrs = reinterpret_cast<int64_t *>(data + opFlagSize + opSizeLen);
    attrs[aicpu::TOPK_FLAT_ATTR_ASC_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_K_IDX] = k;
    attrs[aicpu::TOPK_FLAT_ATTR_BURST_LEN_IDX] = burstLen;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_NUM_IDX] = blockNum;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_IDX] = static_cast<int64_t>(pageId);
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_NUM_IDX] = static_cast<int64_t>(pageNum);
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_SIZE_IDX] = this->pageSize;
    attrs[aicpu::TOPK_FLAT_ATTR_QUICK_HEAP] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_SIZE] = this->blockSize;
    uint32_t *opSizeData = reinterpret_cast<uint32_t *>(data + opFlagSize);

    uint32_t idxUseMask = (this->maskData != nullptr) ? 1 : 0;
    for (int i = 0; i < blockNum; ++i) {
        int offset = i * this->blockSize;
        int opSizeHostIdx = i * CORE_NUM * SIZE_ALIGN;
        opSizeData[opSizeHostIdx + IDX_ACTUAL_NUM] = std::min(static_cast<uint32_t>(computeNum - offset),
                                                              static_cast<uint32_t>(this->blockSize));
        opSizeData[opSizeHostIdx + IDX_COMP_OFFSET] =
            static_cast<uint32_t>(pageOffset) + static_cast<uint32_t>(offset);
        opSizeData[opSizeHostIdx + IDX_MASK_LEN] = maskLen;
        opSizeData[opSizeHostIdx + IDX_USE_MASK] = idxUseMask;
    }
    auto ret = aclrtMemcpy(continuousMem.data(), continuousMem.getSizeInBytes(),
                           continuousValue.data(), continuousValue.size() * sizeof(uint8_t), ACL_MEMCPY_HOST_TO_DEVICE);

    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy attr to device");

    uint16_t *opFlagMem = reinterpret_cast<uint16_t *>(continuousMem.data());
    AscendTensor<uint16_t, DIMS_3> opFlag(opFlagMem, { blockNum, flagNum, FLAG_SIZE });
    uint32_t *opSizeMem = reinterpret_cast<uint32_t *>(continuousMem.data() + opFlagSize);
    AscendTensor<uint32_t, DIMS_3> opSize(opSizeMem, { blockNum, CORE_NUM, SIZE_ALIGN });
    int64_t *attrMem = reinterpret_cast<int64_t *>(continuousMem.data() + opFlagSize + opSizeLen);
    AscendTensor<int64_t, DIMS_1> attrsInput(attrMem, { aicpu::TOPK_FLAT_ATTR_IDX_COUNT });

    // 1. run the topk operator to wait for distance result and compute topk
    runTopkCompute(distResult, minDistResult, opSize, opFlag, attrsInput, minDistances, minIndices, streamAicpu);

    AscendTensor<uint8_t, DIMS_2> mask;
    if (idxUseMask == 1) {
        int32_t alignLen = utils::roundUp(static_cast<int32_t>(maskLen), CUBE_ALIGN_INT8);
        int32_t maxLen = std::max(alignLen, this->blockMaskSize);
        mask = AscendTensor<uint8_t, DIMS_2>(this->maskData, { nq, maxLen });
    } else {
        mask = AscendTensor<uint8_t, DIMS_2>(mem, { nq, this->blockMaskSize }, stream);
    }

    // 2. run the disance operator to compute the distance
    const int dim1 = utils::divUp(this->blockSize, CUBE_ALIGN);
    const int dim2 = utils::divUp(this->dims, CUBE_ALIGN);
    for (int i = 0; i < blockNum; ++i) {
        AscendTensor<float16_t, DIMS_4> shaped(baseShaped[blockOffset + (size_t)i]->data(),
            { dim1, dim2, CUBE_ALIGN, CUBE_ALIGN });
        AscendTensor<float16_t, DIMS_1> norm(normBase[blockOffset + i]->data(), { blockSize });
        auto dist = distResult[i].view();
        auto minDist = minDistResult[i].view();
        auto actualSize = opSize[i].view();
        auto flag = opFlag[i].view();
        if (isNeedCleanMinDist) {
            minDist.zero();
        }

        std::vector<const AscendTensorBase *> input {&queries, &mask, &shaped, &norm, &actualSize};
        std::vector<const AscendTensorBase *> output {&dist, &minDist, &flag};
        runDistCompute(nq, input, output, stream);
    }

    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream default stream: %i\n", ret);

    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream aicpu stream failed: %i\n", ret);

    return APP_ERR_OK;
}

APP_ERROR IndexFlatL2Aicpu::searchImpl(std::vector<Index *> indexes, int n, int batchSize, const float16_t *x,
    int k, float16_t *distances, idx_t *labels)
{
    int indexSize =  static_cast<int>(indexes.size()); // max indexes num is 10000
    std::vector<idx_t> ntotals(indexSize);
    std::vector<idx_t> offsetBlocks(indexSize + 1, 0);
    for (int i = 0; i < indexSize; ++i) {
        ntotals[i] = indexes[i]->ntotal;
        offsetBlocks[i + 1] = offsetBlocks[i] + utils::divUp(ntotals[i], blockSize);
    }
    int burstLen = BURST_LEN_HIGH;
    auto curBurstsOfBlock = GetBurstsOfBlock(batchSize, this->blockSize, burstLen);
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    int blockNum = static_cast<int>(offsetBlocks[indexSize]);

    // 1. construct the distance operator param
    // 1.1 aicore params
    AscendTensor<float16_t, DIMS_2> queries(const_cast<float16_t *>(x), { batchSize, dims });
    AscendTensor<uint16_t, DIMS_3> opFlag(mem, { blockNum, flagNum, FLAG_SIZE }, stream);
    AscendTensor<uint32_t, DIMS_3> opSize(mem, { PAGE_BLOCKS, CORE_NUM, SIZE_ALIGN }, stream);
    AscendTensor<float16_t, DIMS_3, size_t> distResult(mem,
        { static_cast<size_t>(PAGE_BLOCKS), static_cast<size_t>(batchSize), static_cast<size_t>(blockSize) },
        stream);
    AscendTensor<float16_t, DIMS_3, size_t> minDistResult(mem,
        { static_cast<size_t>(PAGE_BLOCKS), static_cast<size_t>(batchSize), static_cast<size_t>(curBurstsOfBlock) },
        stream);
    opFlag.zero();

    // 1.2 aicpu params
    AscendTensor<int64_t, DIMS_1> attrsInputs(mem, { aicpu::TOPK_MULTISEARCH_ATTR_IDX_COUNT }, streamAicpu);
    AscendTensor<uint32_t, DIMS_1> indexOffsetInputs(mem, { blockNum }, streamAicpu);
    AscendTensor<uint32_t, DIMS_1> labelOffsetInputs(mem, { blockNum }, streamAicpu);
    AscendTensor<uint16_t, DIMS_1> reorderFlagInputs(mem, { blockNum }, streamAicpu);
    attrsInputs[aicpu::TOPK_MULTISEARCH_ATTR_ASC_IDX] = 1;
    attrsInputs[aicpu::TOPK_MULTISEARCH_ATTR_K_IDX] = k;
    attrsInputs[aicpu::TOPK_MULTISEARCH_ATTR_BURST_LEN_IDX] = burstLen;
    attrsInputs[aicpu::TOPK_MULTISEARCH_ATTR_INDEX_NUM_IDX] = indexSize;
    attrsInputs[aicpu::TOPK_MULTISEARCH_ATTR_PAGE_BLOCK_NUM_IDX] = PAGE_BLOCKS;
    attrsInputs[aicpu::TOPK_MULTISEARCH_ATTR_QUICK_HEAP] = 1;
    attrsInputs[aicpu::TOPK_MULTISEARCH_ATTR_BLOCK_SIZE] = blockSize;
    APPERR_RETURN_IF_NOT_OK(
        computeMultisearchTopkParam(indexOffsetInputs, labelOffsetInputs, reorderFlagInputs, ntotals, offsetBlocks));
 
    // 1.3 resultTemp
    AscendTensor<float16_t, DIMS_3, size_t> minDistances(mem, { static_cast<size_t>(indexSize),
        static_cast<size_t>(batchSize), static_cast<size_t>(k) }, streamAicpu);
    AscendTensor<idx_t, DIMS_3, size_t> minIndices(mem, { static_cast<size_t>(indexSize),
        static_cast<size_t>(batchSize), static_cast<size_t>(k) }, streamAicpu);
    APPERR_RETURN_IF_NOT_OK(initResult(minDistances, minIndices));
 
    AscendTensor<uint8_t, DIMS_2> mask(mem, { batchSize, this->blockMaskSize }, stream);

    // 2. run the disance operators and topk operators
    const int blockSizeShapNum = utils::divUp(this->blockSize, CUBE_ALIGN);
    const int dimsShapeNum = utils::divUp(this->dims, CUBE_ALIGN);
    int blockOffset = 0;
    int indexDoneCount = 0;
    int copyCount = 0;
    for (int indexId = 0; indexId < indexSize; ++indexId) {
        IndexFlat *index = dynamic_cast<IndexFlat *>(indexes[indexId]);
 
        int blocks = static_cast<int>(utils::divUp(ntotals[indexId], blockSize));
        for (int i = 0; i < blocks; ++i) {
            int offset = i * blockSize;
            int blockIdx = (static_cast<int>(offsetBlocks[indexId]) + i) % PAGE_BLOCKS;
            if (blockIdx == 0 && (static_cast<int>(offsetBlocks[indexId]) + i) > 0) {
                APP_ERROR syncRetCode = tryToSychResultAdvanced(copyCount, indexDoneCount, indexId, n, batchSize, k,
                                                                distances, labels, minDistances, minIndices);
                APPERR_RETURN_IF_NOT_FMT(syncRetCode == APP_ERR_OK, syncRetCode,
                                         "try to copy result in advanced failed:%d.\n", syncRetCode);
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
                topkMultisearchParams params = topkMultisearchParams(batchSize, &distResult, &minDistResult, &opSize,
                                                                     &flag, &attrsInputs, &indexOffset, &labelOffset,
                                                                     &reorderFlag, &minDistances, &minIndices);

                calculateTopkMultisearch(params, streamAicpu);
                blockOffset += actualPageBlocks;
            }
 
            // under distance operators
            AscendTensor<float16_t, DIMS_4> shaped(index->getBaseShaped()[i]->data(),
                { blockSizeShapNum, dimsShapeNum, CUBE_ALIGN, CUBE_ALIGN });
            AscendTensor<uint32_t, DIMS_2> actualSize(opSize[blockIdx].data(), { CORE_NUM, SIZE_ALIGN });
            AscendTensor<float16_t, DIMS_2, size_t> result = distResult[blockIdx].view();
            AscendTensor<float16_t, DIMS_2, size_t> minResult = minDistResult[blockIdx].view();
            AscendTensor<float16_t, DIMS_1> norm(index->getNormBase()[i]->data(), { blockSize });
            AscendTensor<uint16_t, DIMS_2> flag(opFlag[offsetBlocks[indexId] + i].data(), { flagNum, FLAG_SIZE });
 
            actualSize[0][IDX_ACTUAL_NUM] = std::min(static_cast<uint32_t>(index->ntotal - offset),
                                                     static_cast<uint32_t>(blockSize));
            actualSize[0][IDX_USE_MASK] = 0; // not use mask

            if (isNeedCleanMinDist) {
                minResult.zero();
            }

            std::vector<const AscendTensorBase *> input {&queries, &mask, &shaped, &norm, &actualSize};
            std::vector<const AscendTensorBase *> output {&result, &minResult, &flag};
            runDistCompute(batchSize, input, output, stream);
        }
    }

    auto ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream aicpu stream failed: %i\n", ret);
 
    // memcpy data back from dev to host
    for (int indexId = copyCount; indexId < indexSize; ++indexId) {
        size_t totalIndex = static_cast<size_t>(indexId) * static_cast<size_t>(n) * static_cast<size_t>(k);
        size_t totalBatchSize = static_cast<size_t>(batchSize) * static_cast<size_t>(k);

        ret = aclrtMemcpy(distances + totalIndex, totalBatchSize * sizeof(float16_t), minDistances[indexId].data(),
            totalBatchSize * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy outDistances back to host");

        ret = aclrtMemcpy(labels + totalIndex, totalBatchSize * sizeof(idx_t), minIndices[indexId].data(),
            totalBatchSize * sizeof(idx_t), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy outIndices back to host");
    }
 
    return APP_ERR_OK;
}

void IndexFlatL2Aicpu::calculateTopkMultisearch(topkMultisearchParams &opParams, aclrtStream streamAicpu)
{
    std::vector<const AscendTensorBase *> input {opParams.distResult, opParams.minDistResult, opParams.opSize,
                                                 opParams.flag,  opParams.attrsInputs, opParams.indexOffset,
                                                 opParams.labelOffset, opParams.reorderFlag};
    std::vector<const AscendTensorBase *> output {opParams.minDistances, opParams.minIndices};
    if (isUseOnlineOp()) {
        runMultisearchTopkOnline(opParams, input, output, streamAicpu);
    } else {
        runMultisearchTopkCompute(opParams.batch, input, output, streamAicpu);
    }
}

APP_ERROR IndexFlatL2Aicpu::initResult(AscendTensor<float16_t, DIMS_3, size_t> &distances,
    AscendTensor<idx_t, DIMS_3, size_t> &indices) const
{
    std::vector<float16_t> distancesInit(distances.getSizeInBytes() / sizeof(float16_t), Limits<float16_t>::getMax());
    std::vector<idx_t> indicesInit(indices.getSizeInBytes() / sizeof(idx_t), std::numeric_limits<idx_t>::max());

    auto ret = aclrtMemcpy(distances.data(), distances.getSizeInBytes(), distancesInit.data(),
        distancesInit.size() * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
    ret = aclrtMemcpy(indices.data(), indices.getSizeInBytes(), indicesInit.data(),
        indicesInit.size() * sizeof(idx_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
 
    return APP_ERR_OK;
}

APP_ERROR IndexFlatL2Aicpu::tryToSychResultAdvanced(int &hasCopiedCount, int &indexDoneCount, int indexId,
    int n, int batchSize, int k,
    float16_t *distances, idx_t *labels,
    AscendTensor<float16_t, DIMS_3, size_t> &minDistances, AscendTensor<idx_t, DIMS_3, size_t> &minIndices)
{
    for (int j = hasCopiedCount; j < indexDoneCount; ++j) {
        size_t totalIndex = static_cast<size_t>(j) * static_cast<size_t>(n) * static_cast<size_t>(k);
        size_t totalBatchSize = static_cast<size_t>(batchSize) * static_cast<size_t>(k);

        auto ret = aclrtMemcpy(distances + totalIndex, totalBatchSize * sizeof(float16_t),
            minDistances[j].data(), totalBatchSize * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
        
        ret = aclrtMemcpy(labels + totalIndex, totalBatchSize * sizeof(idx_t),
            minIndices[j].data(), totalBatchSize * sizeof(idx_t), ACL_MEMCPY_DEVICE_TO_HOST);
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

void IndexFlatL2Aicpu::runTopkCompute(AscendTensor<float16_t, DIMS_3, size_t> &dists,
                                      AscendTensor<float16_t, DIMS_3, size_t> &maxdists,
                                      AscendTensor<uint32_t, DIMS_3> &sizes,
                                      AscendTensor<uint16_t, DIMS_3> &flags,
                                      AscendTensor<int64_t, DIMS_1> &attrs,
                                      AscendTensor<float16_t, DIMS_2> &outdists,
                                      AscendTensor<int64_t, DIMS_2> &outlabel,
                                      aclrtStream stream)
{
    uint32_t batch = dists.getSize(1);
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

    if (!isUseOnlineOp()) {
        AscendOperator *op = nullptr;
        if (topkComputeOps.find(batch) != topkComputeOps.end()) {
            op = topkComputeOps[batch].get();
        }
        ASCEND_THROW_IF_NOT(op);
        op->exec(*topkOpInput, *topkOpOutput, stream);
    } else {
        topkOpParams params(dists, maxdists, sizes, flags, attrs, outdists, outlabel);
        auto ret = runTopkOnlineOp(batch, flagNum, params, stream);
        ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run online TopkFlat operator failed: %i\n", ret);
    }
}
void IndexFlatL2Aicpu::runMultisearchTopkOnline(topkMultisearchParams &opParams,
                                                const std::vector<const AscendTensorBase *> &inputData,
                                                const std::vector<const AscendTensorBase *> &outputData,
                                                aclrtStream streamAicpu)
{
    AscendOpDesc desc("TopkMultisearch");
    int burstLen = BURST_LEN_HIGH;
    int batch = opParams.batch;
    auto curBurstsOfBlock = GetBurstsOfBlock(batch, this->blockSize, burstLen);
    std::vector<int64_t> shape0 { static_cast<int64_t>(opParams.distResult->getSize(0)), batch, this->blockSize };
    std::vector<int64_t> shape1 { static_cast<int64_t>(opParams.minDistResult->getSize(0)), batch, curBurstsOfBlock };
    std::vector<int64_t> shape2 { static_cast<int64_t>(opParams.opSize->getSize(0)), CORE_NUM, SIZE_ALIGN };
    std::vector<int64_t> shape3 { static_cast<int64_t>(opParams.flag->getSize(0)), flagNum, FLAG_SIZE };
    std::vector<int64_t> shape4 { aicpu::TOPK_MULTISEARCH_ATTR_IDX_COUNT };
    std::vector<int64_t> shape5 { static_cast<int64_t>(opParams.indexOffset->getSize(0)) };
    std::vector<int64_t> shape6 { static_cast<int64_t>(opParams.labelOffset->getSize(0)) };
    std::vector<int64_t> shape7 { static_cast<int64_t>(opParams.reorderFlag->getSize(0)) };
    std::vector<int64_t> shape8 { static_cast<int64_t>(opParams.minDistances->getSize(0)), batch,
                                  static_cast<int64_t>(opParams.minDistances->getSize(2)) };
    std::vector<int64_t> shape9 { static_cast<int64_t>(opParams.minIndices->getSize(0)), batch,
                                  static_cast<int64_t>(opParams.minIndices->getSize(2)) };
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
    std::vector<aclDataBuffer *> topkOpInput;
    for (auto &data : inputData) {
        topkOpInput.emplace_back(aclCreateDataBuffer(data->getVoidData(), data->getSizeInBytes()));
    }
    std::vector<aclDataBuffer *> topkOpOutput;
    for (auto &data : outputData) {
        topkOpOutput.emplace_back(aclCreateDataBuffer(data->getVoidData(), data->getSizeInBytes()));
    }
    aclopAttr *opAttr = aclopCreateAttr();
    auto ret = aclopExecuteV2("TopkMultisearch", desc.inputDesc.size(),
                              const_cast<aclTensorDesc **>(desc.inputDesc.data()), topkOpInput.data(),
                              desc.outputDesc.size(), const_cast<aclTensorDesc **>(desc.outputDesc.data()),
                              topkOpOutput.data(), opAttr, streamAicpu);
    aclopDestroyAttr(opAttr);
    for (auto item: topkOpInput) {
        aclDestroyDataBuffer(item);
    }
    for (auto item: topkOpOutput) {
        aclDestroyDataBuffer(item);
    }
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run online TopkMultisearch operator failed: %i\n", ret);
}

APP_ERROR IndexFlatL2Aicpu::resetMultisearchTopkCompOp()
{
    if (isUseOnlineOp()) {
        return resetOnlineMultisearchTopk();
    } else {
        return resetOfflineMultisearchTopk(IndexTypeIdx::ITI_FLAT_L2_TOPK_MULTISEARCH, flagNum);
    }
}

APP_ERROR IndexFlatL2Aicpu::resetTopkOffline()
{
    auto topkCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("TopkFlat");
        int burstLen = BURST_LEN_HIGH;
        auto curBurstsOfBlock = GetBurstsOfBlock(batch, this->blockSize, burstLen);
        std::vector<int64_t> shape0 { 0, batch, this->blockSize };
        std::vector<int64_t> shape1 { 0, batch, curBurstsOfBlock };
        std::vector<int64_t> shape2 { 0, CORE_NUM, SIZE_ALIGN };
        std::vector<int64_t> shape3 { 0, flagNum, FLAG_SIZE };
        std::vector<int64_t> shape4 { aicpu::TOPK_FLAT_ATTR_IDX_COUNT };
        std::vector<int64_t> shape5 { batch, 0 };

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

    for (auto batch : searchBatchSizes) {
        topkComputeOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(topkCompOpReset(topkComputeOps[batch], batch),
                                 APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "topk op init failed");
    }
    return APP_ERR_OK;
}

APP_ERROR IndexFlatL2Aicpu::resetTopkCompOp()
{
    if (isUseOnlineOp()) {
        return resetTopkOnline();
    } else {
        return resetTopkOffline();
    }
}

void IndexFlatL2Aicpu::moveVectorForward(idx_t srcIdx, idx_t dstIdx)
{
    ASCEND_THROW_IF_NOT(srcIdx >= dstIdx);
    // 1. move code
    IndexFlat::moveVectorForward(srcIdx, dstIdx);

    // 2. move precompute
    idx_t srcIdx1 = srcIdx / static_cast<idx_t>(this->blockSize);
    idx_t srcIdx2 = srcIdx % static_cast<idx_t>(this->blockSize);
    idx_t dstIdx1 = dstIdx / static_cast<idx_t>(this->blockSize);
    idx_t dstIdx2 = dstIdx % static_cast<idx_t>(this->blockSize);

    auto err = aclrtMemcpy(normBase[dstIdx1]->data() + dstIdx2, sizeof(float16_t),
                           normBase[srcIdx1]->data() + srcIdx2, sizeof(float16_t),
                           ACL_MEMCPY_DEVICE_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(err == EOK, "Mem error %d", err);
}

void IndexFlatL2Aicpu::releaseUnusageSpace(int oldTotal, int remove)
{
    IndexFlat::releaseUnusageSpace(oldTotal, remove);

    int oldVecSize = utils::divUp(oldTotal, this->blockSize);
    int vecSize = utils::divUp(oldTotal - remove, this->blockSize);

    for (int i = oldVecSize - 1; i >= vecSize; --i) {
        this->normBase.at(i)->clear();
    }
}

void IndexFlatL2Aicpu::runDistCompute(int batch,
                                      const std::vector<const AscendTensorBase *> &input,
                                      const std::vector<const AscendTensorBase *> &output,
                                      aclrtStream stream)
{
    if (!isUseOnlineOp()) {
        IndexTypeIdx indexType = IndexTypeIdx::ITI_FLAT_L2;
        std::vector<int> keys({batch, dims});
        OpsMngKey opsKey(keys);
        auto ret = DistComputeOpsManager::getInstance().runOp(indexType, opsKey, input, output, stream);
        ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run offline operator failed: %i\n", ret);
    } else {
        std::vector<aclTensorDesc *> inputDesc = disL2OpInputDesc.at(batch);
        std::vector<aclTensorDesc *> outputDesc = disL2OpOutputDesc.at(batch);
        std::vector<aclDataBuffer *> distOpInput;
        for (auto &data : input) {
            distOpInput.emplace_back(aclCreateDataBuffer(data->getVoidData(), data->getSizeInBytes()));
        }
        std::vector<aclDataBuffer *> distOpOutput;
        for (auto &data : output) {
            distOpOutput.emplace_back(aclCreateDataBuffer(data->getVoidData(), data->getSizeInBytes()));
        }
        const char *opType = distOpName.c_str();
        aclopAttr *opAttr = aclopCreateAttr();
        auto ret = aclopExecuteV2(opType, inputDesc.size(),
                                  const_cast<aclTensorDesc **>(inputDesc.data()),
                                  distOpInput.data(), outputDesc.size(),
                                  const_cast<aclTensorDesc **>(outputDesc.data()),
                                  distOpOutput.data(),
                                  opAttr, stream);
        aclopDestroyAttr(opAttr);
        for (auto item: distOpInput) {
            aclDestroyDataBuffer(item);
        }
        for (auto item: distOpOutput) {
            aclDestroyDataBuffer(item);
        }
        ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run online operator failed: %i\n", ret);
    }
}

APP_ERROR IndexFlatL2Aicpu::resetDistOnlineOp(int batch,
                                              std::vector<std::pair<aclDataType, std::vector<int64_t>>> &input,
                                              std::vector<std::pair<aclDataType, std::vector<int64_t>>> &output)
{
    // 在线转换算子
    const char *opType = distOpName.c_str();
    int numInputs = static_cast<int>(input.size());
    for (int i = 0; i < numInputs; i++) {
        aclTensorDesc *desc = aclCreateTensorDesc(input[i].first, input[i].second.size(),
                                                  input[i].second.data(), ACL_FORMAT_ND);
        if (desc == nullptr) {
            return APP_ERR_INNER_ERROR;
        }
        disL2OpInputDesc[batch].emplace_back(desc);
    }
    int numOutputs = static_cast<int>(output.size());
    for (int i = 0; i < numOutputs; i++) {
        aclTensorDesc *desc = aclCreateTensorDesc(output[i].first, output[i].second.size(),
                                                  output[i].second.data(), ACL_FORMAT_ND);
        if (desc == nullptr) {
            return APP_ERR_INNER_ERROR;
        }
        disL2OpOutputDesc[batch].emplace_back(desc);
    }
    aclopAttr *opAttr = aclopCreateAttr();
    auto ret = aclSetCompileopt(ACL_OP_JIT_COMPILE, "enable");
    if (ret != APP_ERR_OK) {
        aclopDestroyAttr(opAttr);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "enable jit compile fail opName:%s : %i\n", opType, ret);
    }
    ret = aclopCompile(opType, numInputs, disL2OpInputDesc[batch].data(), numOutputs,
                       disL2OpOutputDesc[batch].data(), opAttr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, nullptr);
    aclopDestroyAttr(opAttr);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "op init failed: %i", ret);
    return APP_ERR_OK;
}

APP_ERROR IndexFlatL2Aicpu::resetDistCompOp(int numLists)
{
    IndexTypeIdx indexType = IndexTypeIdx::ITI_FLAT_L2;
    for (auto batch : searchBatchSizes) {
        int burstLen = BURST_LEN_HIGH;
        auto curBurstsOfBlock = GetBurstsOfBlock(batch, this->blockSize, burstLen);
        std::vector<int64_t> queryShape({ batch, dims });
        std::vector<int64_t> maskShape({ batch, blockMaskSize });
        std::vector<int64_t> coarseCentroidsShape({ utils::divUp(numLists, CUBE_ALIGN),
            utils::divUp(dims, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN });
        std::vector<int64_t> preNormsShape({ numLists });
        std::vector<int64_t> sizeShape({ CORE_NUM, SIZE_ALIGN });
        std::vector<int64_t> distResultShape({ batch, numLists });
        std::vector<int64_t> minResultShape({ batch, curBurstsOfBlock });
        std::vector<int64_t> flagShape({ flagNum, FLAG_SIZE });

        std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
            { ACL_FLOAT16, queryShape },
            { ACL_UINT8, maskShape },
            { ACL_FLOAT16, coarseCentroidsShape },
            { ACL_FLOAT16, preNormsShape },
            { ACL_UINT32, sizeShape }
        };

        std::vector<std::pair<aclDataType, std::vector<int64_t>>> output {
            { ACL_FLOAT16, distResultShape },
            { ACL_FLOAT16, minResultShape },
            { ACL_UINT16, flagShape }
        };
        if (isUseOnlineOp()) {
            auto ret = resetDistOnlineOp(batch, input, output);

            APPERR_RETURN_IF_NOT_FMT(APP_ERR_OK == ret, APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                                     "compile distance op init failed:%i, batch: %d\n", ret, batch);
        } else {
            std::vector<int> keys({batch, dims});
            OpsMngKey opsKey(keys);
            auto ret = DistComputeOpsManager::getInstance().resetOp(distOpName, indexType, opsKey, input, output);
            APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "op init failed: %i", ret);
        }
    }
    return APP_ERR_OK;
}

size_t IndexFlatL2Aicpu::calcNormBaseSize(idx_t totalNum) const
{
    size_t numBatch = static_cast<size_t>(utils::divUp(static_cast<int>(totalNum), blockSize));
    return numBatch * static_cast<size_t>(blockSize);
}
} // namespace ascend
