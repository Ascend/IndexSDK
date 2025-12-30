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

#include "index/IndexSQL2Aicpu.h"

#include <algorithm>

#include "ascend/AscendIndex.h"
#include "ascenddaemon/utils/AscendTensor.h"
#include "ascenddaemon/utils/Limits.h"
#include "common/utils/CommonUtils.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"

namespace ascend {
namespace {
const int IDX_USE_MASK = 3;
const int IDX_CODE_SIZE = 4;
}
IndexSQL2Aicpu::IndexSQL2Aicpu(int dim, bool filterable, int64_t resourceSize, int blockSize)
    : IndexSQ(dim, filterable, resourceSize, blockSize) {}

IndexSQL2Aicpu::~IndexSQL2Aicpu() {}

APP_ERROR IndexSQL2Aicpu::init()
{
    IndexTypeIdx indexType =
        (this->dims != 64) ? IndexTypeIdx::ITI_SQ_DIST_L2 : IndexTypeIdx::ITI_SQ_DIST_DIM64_L2;
    std::string opTypeName = "DistanceSQ8L2Mins";
    auto ret = resetSqDistOperator(opTypeName, indexType);
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, ret, "failed to reset dist op");

    IndexTypeIdx indexMaskType =
        (this->dims != 64) ? IndexTypeIdx::ITI_SQ_DIST_MASK_L2 : IndexTypeIdx::ITI_SQ_DIST_MASK_DIM64_L2;
    std::string opMaskTypeName = "DistanceMaskedSQ8L2Mins";
    ret = resetSqDistMaskOperator(opMaskTypeName, indexMaskType);
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, ret, "failed to reset dist mask op");
    return IndexSQ::init();
}

APP_ERROR IndexSQL2Aicpu::addVectors(size_t numVecs, const uint8_t *data, const float *preComp)
{
    // 0. add code
    APPERR_RETURN_IF_NOT_OK(IndexSQ::addVectors(numVecs, data, preComp));

    // 1. resize DeviceVector to save the data x
    int newVecSize = static_cast<int>(utils::divUp(this->ntotal + numVecs, this->computeBlockSize));
    int vecSize = static_cast<int>(utils::divUp(this->ntotal, this->computeBlockSize));
    for (int i = 0; i < newVecSize - vecSize; ++i) {
        this->preCompute.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<float>));
        if (vecSize + i > 0) {
            this->preCompute.at(vecSize + i)->resize(this->codes[i]->size() / this->dims, true);
        }
    }

    // resize little to avoid waste for idx 0
    if (ntotal < static_cast<idx_t>(computeBlockSize)) {
        this->preCompute.at(0)->resize(this->codes[0]->size() / this->dims, true);
    }

    // 2. save preCompute if not null
    APPERR_RETURN_IF_NOT_OK(savePreComputeHostCpu(numVecs, preComp));

    // 3. modify ntotal
    this->ntotal += numVecs;

    return APP_ERR_OK;
}

APP_ERROR IndexSQL2Aicpu::savePreComputeHostCpu(int numVecs, const float *preCompute)
{
    int vecBefore = static_cast<int>(this->ntotal) / this->computeBlockSize;
    int dVecBefore = static_cast<int>(this->ntotal) % this->computeBlockSize;

    ASCEND_THROW_IF_NOT(static_cast<int>(this->preCompute.size()) > vecBefore);
    ASCEND_THROW_IF_NOT(this->computeBlockSize > dVecBefore);

    if ((dVecBefore + numVecs) > this->computeBlockSize) {
        int vecIndex = utils::divUp((numVecs + dVecBefore), this->computeBlockSize);
        int dVecLast = (numVecs + dVecBefore) % this->computeBlockSize;

        for (int i = 0; i < vecIndex; i++) {
            const float *SrcPtr = preCompute;
            float *DstPtr = this->preCompute[vecBefore]->data() + dVecBefore;
            int sizeLen = this->computeBlockSize - dVecBefore;

            if (i > 0) {
                SrcPtr = preCompute + (computeBlockSize - dVecBefore) + (i - 1) * computeBlockSize;
                DstPtr = this->preCompute[vecBefore + i]->data();
                if (i == vecIndex - 1) {
                    sizeLen = (dVecLast == 0) ? this->computeBlockSize : dVecLast;
                } else {
                    sizeLen = this->computeBlockSize;
                }
            }
            auto ret = aclrtMemcpy(DstPtr, sizeLen * sizeof(float),
                                   SrcPtr, sizeLen * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);
        }
    } else {
        auto ret = aclrtMemcpy(this->preCompute[vecBefore]->data() + dVecBefore, numVecs * sizeof(float), preCompute,
            numVecs * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);
    }

    return APP_ERR_OK;
}

void IndexSQL2Aicpu::moveVectorForward(idx_t srcIdx, idx_t dstIdx)
{
    ASCEND_THROW_IF_NOT(srcIdx >= dstIdx);
    // 1. move code
    IndexSQ::moveVectorForward(srcIdx, dstIdx);

    // 2. move precompute
    int srcIdx1 = static_cast<int>(srcIdx) / this->computeBlockSize;
    int srcIdx2 = static_cast<int>(srcIdx) % this->computeBlockSize;
    int dstIdx1 = static_cast<int>(dstIdx) / this->computeBlockSize;
    int dstIdx2 = static_cast<int>(dstIdx) % this->computeBlockSize;

    auto err = aclrtMemcpy(this->preCompute[dstIdx1]->data() + dstIdx2, sizeof(float),
        this->preCompute[srcIdx1]->data() + srcIdx2, sizeof(float), ACL_MEMCPY_DEVICE_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(err == EOK, "Memcpy error %d", (int)err);
}

APP_ERROR IndexSQL2Aicpu::searchImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    AscendTensor<float16_t, DIMS_2> outDistances(distances, { n, k });
    AscendTensor<idx_t, DIMS_2> outIndices(labels, { n, k });
    AscendTensor<float16_t, DIMS_2> minDistances(mem, { n, k }, stream);
    AscendTensor<int64_t, DIMS_2> minIndices(mem, { n, k }, stream);
    // compute distance and topk by code page
    size_t pageNum = utils::divUp(this->ntotal, this->pageSize);
    for (size_t pageId = 0; pageId < pageNum; ++pageId) {
        APP_ERROR ret = searchPaged(pageId, pageNum, x, minDistances, minIndices);
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

APP_ERROR IndexSQL2Aicpu::searchPaged(size_t pageId, size_t pageNum, const float16_t *x,
    AscendTensor<float16_t, DIMS_2> &minDistances, AscendTensor<int64_t, DIMS_2> &minIndices)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    int n = minDistances.getSize(0);
    int k = minDistances.getSize(1);
    int pageOffset = static_cast<int>(pageId) * this->pageSize;
    int blockOffset = static_cast<int>(pageId) * this->pageSize / computeBlockSize;
    int computeNum = std::min(this->ntotal - pageOffset, static_cast<idx_t>(this->pageSize));
    int blockNum = utils::divUp(computeNum, this->computeBlockSize);

    // costruct the SQ distance operator param
    AscendTensor<float16_t, DIMS_3, size_t> distResult(mem,
        { (size_t)blockNum, (size_t)n, (size_t)this->codeBlockSize }, stream);
    AscendTensor<float16_t, DIMS_3, size_t> minDistResult(mem,
        { (size_t)blockNum, (size_t)n, (size_t)this->burstsOfBlock }, stream);

    AscendTensor<float16_t, DIMS_2> tensorDevQueries(const_cast<float16_t *>(x), { n, dims });

    uint32_t opFlagSize = static_cast<uint32_t>(blockNum * FLAG_NUM * FLAG_SIZE) * sizeof(uint16_t);
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
    attrs[aicpu::TOPK_FLAT_ATTR_BURST_LEN_IDX] = BURST_LEN;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_NUM_IDX] = blockNum;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_IDX] = static_cast<int64_t>(pageId);
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_NUM_IDX] = static_cast<int64_t>(pageNum);
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_SIZE_IDX] = this->pageSize;
    attrs[aicpu::TOPK_FLAT_ATTR_QUICK_HEAP] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_SIZE] = this->computeBlockSize;
    uint32_t *opSizeData = reinterpret_cast<uint32_t *>(data + opFlagSize);
    for (int i = 0; i < blockNum; ++i) {
        int offset = i * this->computeBlockSize;
        opSizeData[i * CORE_NUM * SIZE_ALIGN + IDX_ACTUAL_NUM] =
            std::min(static_cast<uint32_t>(computeNum - offset), static_cast<uint32_t>(this->computeBlockSize));
        opSizeData[i * CORE_NUM * SIZE_ALIGN + IDX_CODE_SIZE] = this->codes[i]->size() / this->dims;
    }

    auto ret = aclrtMemcpy(continuousMem.data(), continuousMem.getSizeInBytes(), continuousValue.data(),
        continuousValue.size() * sizeof(uint8_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT((ret == ACL_SUCCESS), APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

    uint16_t *opFlagMem = reinterpret_cast<uint16_t *>(continuousMem.data());
    AscendTensor<uint16_t, DIMS_3> opFlag(opFlagMem, { blockNum, FLAG_NUM, FLAG_SIZE });
    uint32_t *opSizeMem = reinterpret_cast<uint32_t *>(continuousMem.data() + opFlagSize);
    AscendTensor<uint32_t, DIMS_3> opSize(opSizeMem, { blockNum, CORE_NUM, SIZE_ALIGN });
    int64_t *attrMem = reinterpret_cast<int64_t *>(continuousMem.data() + opFlagSize + opSizeLen);
    AscendTensor<int64_t, DIMS_1> attrsInput(attrMem, { aicpu::TOPK_FLAT_ATTR_IDX_COUNT });

    // 1. run the topk operator to wait for distance result and compute topk
    std::vector<const AscendTensorBase *> input {&distResult, &minDistResult, &opSize, &opFlag, &attrsInput};
    std::vector<const AscendTensorBase *> output {&minDistances, &minIndices};
    runTopkCompute(n, input, output, streamAicpu);

    // 2. run the disance operator to compute the distance
    for (int i = 0; i < blockNum; ++i) {
        AscendTensor<uint8_t, DIMS_4> pageCode(this->codes[blockOffset + i]->data(),
            { this->codeBlockSize / CUBE_ALIGN, this->dims / CUBE_ALIGN, CUBE_ALIGN, CUBE_ALIGN });
        AscendTensor<float, DIMS_1> batchPreComp(this->preCompute[blockOffset + i]->data(), { this->codeBlockSize });
        AscendTensor<uint16_t, DIMS_2> flag(opFlag[i].data(), { FLAG_NUM, FLAG_SIZE });
        AscendTensor<uint32_t, DIMS_2> actualSize(opSize[i].data(), { CORE_NUM, SIZE_ALIGN });
        AscendTensor<float16_t, DIMS_2, size_t> result = distResult[i].view();
        AscendTensor<float16_t, DIMS_2, size_t> minResult = minDistResult[i].view();

        std::vector<const AscendTensorBase *> input {
            &tensorDevQueries, &pageCode, &batchPreComp, &(this->vDiff), &(this->vMin), &actualSize};
        std::vector<const AscendTensorBase *> output {&result, &minResult, &flag};
        runSqDistOperator(n, input, output, stream);
    }

    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream default stream: %i\n",
        ret);

    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream aicpu stream failed: %i\n", ret);

    return APP_ERR_OK;
}

APP_ERROR IndexSQL2Aicpu::searchFilterImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels,
    uint8_t *masks, uint32_t maskLen)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    int blockNum = static_cast<int>(utils::divUp(this->ntotal, this->computeBlockSize));
    ASCEND_THROW_IF_NOT(static_cast<int>(this->codes.size()) == blockNum);

    // costruct the SQ distance operator param
    AscendTensor<uint16_t, DIMS_3> opFlag(mem, { blockNum, FLAG_NUM, FLAG_SIZE }, stream);
    AscendTensor<uint32_t, DIMS_3> opSize(mem, { blockNum, CORE_NUM, SIZE_ALIGN }, stream);
    AscendTensor<float16_t, DIMS_3, size_t> distResult(mem, { (size_t)blockNum, (size_t)n,
        (size_t)this->codeBlockSize }, stream);
    AscendTensor<float16_t, DIMS_3, size_t> minDistResult(mem,
        { (size_t)blockNum, (size_t)n, (size_t)this->burstsOfBlock }, stream);
    opFlag.zero();

    AscendTensor<float16_t, DIMS_2> tensorDevQueries(const_cast<float16_t *>(x), { n, dims });

    AscendTensor<float16_t, DIMS_2> minDistances(mem, { n, k }, stream);
    AscendTensor<int64_t, DIMS_2> minIndices(mem, { n, k }, stream);

    // attrs: [0]asc, [1]k, [2]burst_len, [3]block_num [4]special page: -1:first page, 0:mid page, 1:last page
    AscendTensor<int64_t, DIMS_1> attrsInput(mem, { aicpu::TOPK_FLAT_ATTR_IDX_COUNT }, stream);
    std::vector<int64_t> attrs(aicpu::TOPK_FLAT_ATTR_IDX_COUNT);
    attrs[aicpu::TOPK_FLAT_ATTR_ASC_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_K_IDX] = k;
    attrs[aicpu::TOPK_FLAT_ATTR_BURST_LEN_IDX] = BURST_LEN;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_NUM_IDX] = blockNum;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_IDX] = 0; // set as last page to reorder
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_NUM_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_SIZE_IDX] = 0;
    attrs[aicpu::TOPK_FLAT_ATTR_QUICK_HEAP] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_SIZE] = this->computeBlockSize;

    auto ret = aclrtMemcpy(attrsInput.data(), attrsInput.getSizeInBytes(), attrs.data(), attrs.size() * sizeof(int64_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);

    // 1. run the topk operator to wait for distance result and compute topk
    std::vector<const AscendTensorBase *> input {&distResult, &minDistResult, &opSize, &opFlag, &attrsInput};
    std::vector<const AscendTensorBase *> output {&minDistances, &minIndices};
    runTopkCompute(n, input, output, streamAicpu);

    // 2. run the disance operator to compute the distance
    const int maskSize = static_cast<int>(utils::divUp(this->codeBlockSize, BIT_OF_UINT8));
    for (int i = 0; i < blockNum; ++i) {
        AscendTensor<uint8_t, DIMS_4> batchCode(this->codes[i]->data(),
            { this->codeBlockSize / CUBE_ALIGN, this->dims / CUBE_ALIGN, CUBE_ALIGN, CUBE_ALIGN });
        AscendTensor<float, DIMS_1> batchPreComp(this->preCompute[i]->data(), { this->codeBlockSize });
        AscendTensor<uint16_t, DIMS_2> flag(opFlag[i].data(), { FLAG_NUM, FLAG_SIZE });
        AscendTensor<uint32_t, DIMS_2> actualSize(opSize[i].data(), { CORE_NUM, SIZE_ALIGN });
        AscendTensor<float16_t, DIMS_2, size_t> result = distResult[i].view();
        AscendTensor<float16_t, DIMS_2, size_t> minResult = minDistResult[i].view();

        int offset = i * this->computeBlockSize;
        AscendTensor<uint8_t, DIMS_2> mask(masks, { n, maskSize });

        actualSize[0][IDX_ACTUAL_NUM] =
            std::min(static_cast<uint32_t>(this->ntotal - offset), static_cast<uint32_t>(this->computeBlockSize));
        actualSize[0][IDX_COMP_OFFSET] = offset; // offset of each blockSize
        actualSize[0][IDX_MASK_LEN] = maskLen;
        actualSize[0][IDX_USE_MASK] = 1;
        actualSize[0][IDX_CODE_SIZE] = static_cast<uint32_t>(this->codes[i]->size() / static_cast<size_t>(this->dims));

        std::vector<const AscendTensorBase *> input {
            &tensorDevQueries, &mask, &batchCode, &batchPreComp, &(this->vDiff), &(this->vMin), &actualSize};
        std::vector<const AscendTensorBase *> output {&result, &minResult, &flag};
        runSqDistMaskOperator(n, input, output, stream);
    }

    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream default stream: %i\n",
        ret);

    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream aicpu stream failed: %i\n", ret);

    // memcpy data back from dev to host
    ret = aclrtMemcpy(distances, (size_t)n * (size_t)k * sizeof(float16_t), minDistances.data(),
        minDistances.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);

    ret = aclrtMemcpy(labels, (size_t)n * (size_t)k * sizeof(idx_t), minIndices.data(), minIndices.getSizeInBytes(),
        ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);

    return APP_ERR_OK;
}

APP_ERROR IndexSQL2Aicpu::initResult(AscendTensor<float16_t, DIMS_3, size_t> &distances,
    AscendTensor<idx_t, DIMS_3, size_t> &indices) const
{
    std::vector<float16_t> distancesInit(distances.getSizeInBytes() / sizeof(float16_t), Limits<float16_t>::getMax());
    std::vector<idx_t> indicesInit(indices.getSizeInBytes() / sizeof(idx_t), std::numeric_limits<idx_t>::max());

    auto ret = aclrtMemcpy(distances.data(), distances.getSizeInBytes(), distancesInit.data(),
        distancesInit.size() * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);
    ret = aclrtMemcpy(indices.data(), indices.getSizeInBytes(), indicesInit.data(),
        indicesInit.size() * sizeof(idx_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);

    return APP_ERR_OK;
}

APP_ERROR IndexSQL2Aicpu::searchImpl(std::vector<Index *> indexes, int n, int batchSize, const float16_t *x, int k,
    float16_t *distances, idx_t *labels)
{
    int indexSize =  static_cast<int>(indexes.size());
    std::vector<idx_t> ntotals(indexSize);
    std::vector<idx_t> offsetBlocks(indexSize + 1, 0);
    for (int i = 0; i < indexSize; ++i) {
        ntotals[i] = indexes[i]->ntotal;
        offsetBlocks[i + 1] = offsetBlocks[i] + utils::divUp(ntotals[i], computeBlockSize);
    }

    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    int blockNum = static_cast<int>(offsetBlocks[indexSize]);

    // 1. costruct the SQ distance operator param
    // aicore params
    AscendTensor<float16_t, DIMS_2> queries(const_cast<float16_t *>(x), { batchSize, dims });
    AscendTensor<uint32_t, DIMS_3> opSize(mem, { PAGE_BLOCKS, CORE_NUM, SIZE_ALIGN }, stream);
    AscendTensor<float16_t, DIMS_3, size_t> distResult(mem,
        { (size_t)PAGE_BLOCKS, (size_t)batchSize, (size_t)codeBlockSize }, stream);
    AscendTensor<float16_t, DIMS_3, size_t>  minDistResult(mem,
        { (size_t)PAGE_BLOCKS, (size_t)batchSize, (size_t)burstsOfBlock }, stream);

    // aicpu params
    AscendTensor<uint32_t, DIMS_1> indexOffsetInputs(mem, { blockNum }, streamAicpu);
    AscendTensor<uint32_t, DIMS_1> labelOffsetInputs(mem, { blockNum }, streamAicpu);
    AscendTensor<uint16_t, DIMS_1> reorderFlagInputs(mem, { blockNum }, streamAicpu);

    uint32_t opFlagSize = static_cast<uint32_t>(blockNum * FLAG_NUM * FLAG_SIZE) * sizeof(uint16_t);
    uint32_t attrsSize = aicpu::TOPK_MULTISEARCH_ATTR_IDX_COUNT * sizeof(int64_t);
    uint32_t continuousMemSize = opFlagSize + attrsSize;
    // 1) aclrtMemcpy比AscendTensor::zero更高效
    // 2) 使用连续内存来减少aclrtMemcpy的调用次数
    AscendTensor<uint8_t, DIMS_1, uint32_t> continuousMem(mem, { continuousMemSize }, stream);
    std::vector<uint8_t> continuousValue(continuousMemSize, 0);
    uint8_t *data = continuousValue.data();
    int64_t *attrs = reinterpret_cast<int64_t *>(data + opFlagSize);
    attrs[aicpu::TOPK_MULTISEARCH_ATTR_ASC_IDX] = 1;
    attrs[aicpu::TOPK_MULTISEARCH_ATTR_K_IDX] = k;
    attrs[aicpu::TOPK_MULTISEARCH_ATTR_BURST_LEN_IDX] = BURST_LEN;
    attrs[aicpu::TOPK_MULTISEARCH_ATTR_INDEX_NUM_IDX] = indexSize;
    attrs[aicpu::TOPK_MULTISEARCH_ATTR_PAGE_BLOCK_NUM_IDX] = PAGE_BLOCKS;
    attrs[aicpu::TOPK_MULTISEARCH_ATTR_QUICK_HEAP] = 1;
    attrs[aicpu::TOPK_MULTISEARCH_ATTR_BLOCK_SIZE] = computeBlockSize;

    auto ret = aclrtMemcpy(continuousMem.data(), continuousMem.getSizeInBytes(),
                           continuousValue.data(), continuousValue.size() * sizeof(uint8_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);

    uint16_t *opFlagMem = reinterpret_cast<uint16_t *>(continuousMem.data());
    AscendTensor<uint16_t, DIMS_3> opFlag(opFlagMem, { blockNum, FLAG_NUM, FLAG_SIZE });
    int64_t *attrMem = reinterpret_cast<int64_t *>(continuousMem.data() + opFlagSize);
    AscendTensor<int64_t, DIMS_1> attrsInputs(attrMem, { aicpu::TOPK_MULTISEARCH_ATTR_IDX_COUNT });

    APPERR_RETURN_IF_NOT_OK(
        computeMultisearchTopkParam(indexOffsetInputs, labelOffsetInputs, reorderFlagInputs, ntotals, offsetBlocks));

    // resultTemp
    AscendTensor<float16_t, DIMS_3, size_t>  minDistances(mem,
        { static_cast<size_t>(indexSize), static_cast<size_t>(batchSize), static_cast<size_t>(k) }, streamAicpu);
    AscendTensor<idx_t, DIMS_3, size_t>  minIndices(mem,
        { static_cast<size_t>(indexSize), static_cast<size_t>(batchSize), static_cast<size_t>(k) }, streamAicpu);
    APPERR_RETURN_IF_NOT_OK(initResult(minDistances,  minIndices));

    // 2. run the disance operators and topk operators
    int blockOffset = 0;
    int indexDoneCount = 0;
    int copyCount = 0;
    for (int indexId = 0; indexId < indexSize; ++indexId) {
        auto index = dynamic_cast<IndexSQ *>(indexes[indexId]);
        if (index == nullptr) {
            APP_LOG_ERROR("the index cast to IndexSQ failed, indexId=%d\n", indexId);
            continue;
        }
        int blocks =  static_cast<int>(utils::divUp(ntotals[indexId], static_cast<idx_t>(computeBlockSize)));
        for (int i = 0; i < blocks; ++i) {
            int offset = i * computeBlockSize;
            int blockIdx =  static_cast<int>((offsetBlocks[indexId] +  static_cast<idx_t>(i)) % PAGE_BLOCKS);

            if (blockIdx == 0) {
                if ((offsetBlocks[indexId] + static_cast<idx_t>(i)) > 0) {
                    for (int j = copyCount; j < indexDoneCount; ++j) {
                        ret = aclrtMemcpy(distances + (size_t)j * (size_t)n * (size_t)k,
                            (size_t)batchSize * (size_t)k * sizeof(float16_t), minDistances[j].data(),
                            (size_t)batchSize * (size_t)k * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_HOST);
                        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d",
                            (int)ret);
                        ret = aclrtMemcpy(labels + (size_t)j * (size_t)n * (size_t)k,
                            (size_t)batchSize * (size_t)k * sizeof(idx_t), minIndices[j].data(),
                            (size_t)batchSize * (size_t)k * sizeof(idx_t), ACL_MEMCPY_DEVICE_TO_HOST);
                        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d",
                            (int)ret);
                        ++copyCount;
                    }
                    ret = synchronizeStream(streamAicpu);
                    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                        "synchronizeStream aicpu stream failed: %i\n", ret);
                    indexDoneCount = indexId - 1;
                }
                // under topk operators
                int actualPageBlocks = std::min(blockNum - blockOffset, PAGE_BLOCKS);
                if (actualPageBlocks < PAGE_BLOCKS) {
                    attrsInputs[aicpu::TOPK_MULTISEARCH_ATTR_PAGE_BLOCK_NUM_IDX] = actualPageBlocks;
                }
                AscendTensor<uint32_t, DIMS_1> indexOffset(indexOffsetInputs.data() + blockOffset,
                    { actualPageBlocks });
                AscendTensor<uint32_t, DIMS_1> labelOffset(labelOffsetInputs.data() + blockOffset,
                    { actualPageBlocks });
                AscendTensor<uint16_t, DIMS_1> reorderFlag(reorderFlagInputs.data() + blockOffset,
                    { actualPageBlocks });
                AscendTensor<uint16_t, DIMS_3> flag(opFlag.data() + blockOffset * FLAG_NUM * FLAG_SIZE,
                    { actualPageBlocks, FLAG_NUM, FLAG_SIZE });

                std::vector<const AscendTensorBase *> input {&distResult, &minDistResult, &opSize, &flag, &attrsInputs,
                    &indexOffset, &labelOffset, &reorderFlag};
                std::vector<const AscendTensorBase *> output {&minDistances, &minIndices};
                runMultisearchTopkCompute(batchSize, input, output, streamAicpu);

                blockOffset += actualPageBlocks;
            }

            // under distance operators
            AscendTensor<uint8_t, DIMS_4> batchCode(index->codes[i]->data(),
                { codeBlockSize / CUBE_ALIGN, dims / CUBE_ALIGN, CUBE_ALIGN, CUBE_ALIGN });
            AscendTensor<float, DIMS_1> batchPreComp(index->preCompute[i]->data(), { this->codeBlockSize });
            AscendTensor<float16_t, DIMS_1> vdiff(index->vDiff.data(), { dims });
            AscendTensor<float16_t, DIMS_1> vmin(index->vMin.data(), { dims });
            AscendTensor<uint32_t, DIMS_2> actualSize(opSize[blockIdx].data(), { CORE_NUM, SIZE_ALIGN });
            AscendTensor<float16_t, DIMS_2, size_t> result = distResult[blockIdx].view();
            AscendTensor<float16_t, DIMS_2, size_t> minResult = minDistResult[blockIdx].view();
            AscendTensor<uint16_t, DIMS_2> flag(opFlag[offsetBlocks[indexId] + i].data(), { FLAG_NUM, FLAG_SIZE });

            actualSize[0][IDX_ACTUAL_NUM] =
                std::min(static_cast<uint32_t>(index->ntotal - offset), static_cast<uint32_t>(computeBlockSize));
            actualSize[0][IDX_CODE_SIZE] =
                static_cast<uint32_t>(index->codes[i]->size()) / static_cast<size_t>(index->dims);

            std::vector<const AscendTensorBase *> input {
                &queries, &batchCode, &batchPreComp, &vdiff, &vmin, &actualSize};
            std::vector<const AscendTensorBase *> output {&result, &minResult, &flag};
            runSqDistOperator(batchSize, input, output, stream);
        }
    }

    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream aicpu stream failed: %i\n", ret);

    // memcpy data back from dev to host
    for (int indexId = copyCount; indexId < indexSize; ++indexId) {
        ret = aclrtMemcpy(distances + (size_t)indexId * (size_t)n * (size_t)k,
            (size_t)batchSize * (size_t)k * sizeof(float16_t),  minDistances[indexId].data(),
            (size_t)batchSize * (size_t)k * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);

        ret = aclrtMemcpy(labels + (size_t)indexId * (size_t)n * (size_t)k,
            (size_t)batchSize * (size_t)k * sizeof(idx_t),  minIndices[indexId].data(),
            (size_t)batchSize * (size_t)k * sizeof(idx_t), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);
    }

    return APP_ERR_OK;
}

APP_ERROR IndexSQL2Aicpu::searchFilterImpl(std::vector<Index*> indexes, int n, int batchSize, const float16_t *x, int k,
    float16_t *distances, idx_t *labels, AscendTensor<uint8_t, DIMS_2, int64_t>& maskData,
    AscendTensor<int, DIMS_1>& maskOffset)
{
    int indexSize = static_cast<int>(indexes.size());
    std::vector<idx_t> ntotals(indexSize);
    std::vector<idx_t> offsetBlocks(indexSize + 1, 0);
    for (int i = 0; i < indexSize; ++i) {
        ntotals[i] = indexes[i]->ntotal;
        offsetBlocks[i + 1] = offsetBlocks[i] + utils::divUp(ntotals[i], computeBlockSize);
    }

    auto &mem = resources.getMemoryManager();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    int blockNum = offsetBlocks[indexSize];

    // 1. costruct the SQ distance operator param
    // aicore params
    AscendTensor<float16_t, DIMS_2> queries(const_cast<float16_t *>(x), { batchSize, dims });
    AscendTensor<uint32_t, DIMS_3> opSize(mem, { PAGE_BLOCKS, CORE_NUM, SIZE_ALIGN }, stream);
    AscendTensor<float16_t, DIMS_3, size_t> distResult(mem,
        { (size_t)PAGE_BLOCKS, (size_t)batchSize, (size_t)codeBlockSize }, stream);
    AscendTensor<float16_t, DIMS_3, size_t>  minDistResult(mem,
        { (size_t)PAGE_BLOCKS, (size_t)batchSize, (size_t)burstsOfBlock }, stream);

    // aicpu params
    AscendTensor<uint32_t, DIMS_1> indexOffsetInputs(mem, { blockNum }, streamAicpu);
    AscendTensor<uint32_t, DIMS_1> labelOffsetInputs(mem, { blockNum }, streamAicpu);
    AscendTensor<uint16_t, DIMS_1> reorderFlagInputs(mem, { blockNum }, streamAicpu);

    uint32_t opFlagSize = static_cast<uint32_t>(blockNum * FLAG_NUM * FLAG_SIZE) * sizeof(uint16_t);
    uint32_t attrsSize = aicpu::TOPK_MULTISEARCH_ATTR_IDX_COUNT * sizeof(int64_t);
    uint32_t continuousMemSize = opFlagSize + attrsSize;
    // 1) aclrtMemcpy比AscendTensor::zero更高效
    // 2) 使用连续内存来减少aclrtMemcpy的调用次数
    AscendTensor<uint8_t, DIMS_1, uint32_t> continuousMem(mem, { continuousMemSize }, stream);
    std::vector<uint8_t> continuousValue(continuousMemSize, 0);
    uint8_t *data = continuousValue.data();
    int64_t *attrs = reinterpret_cast<int64_t *>(data + opFlagSize);

    attrs[aicpu::TOPK_MULTISEARCH_ATTR_ASC_IDX] = 1;
    attrs[aicpu::TOPK_MULTISEARCH_ATTR_K_IDX] = k;
    attrs[aicpu::TOPK_MULTISEARCH_ATTR_BURST_LEN_IDX] = BURST_LEN;
    attrs[aicpu::TOPK_MULTISEARCH_ATTR_INDEX_NUM_IDX] = indexSize;
    attrs[aicpu::TOPK_MULTISEARCH_ATTR_PAGE_BLOCK_NUM_IDX] = PAGE_BLOCKS;
    attrs[aicpu::TOPK_MULTISEARCH_ATTR_QUICK_HEAP] = 1;
    attrs[aicpu::TOPK_MULTISEARCH_ATTR_BLOCK_SIZE] = computeBlockSize;

    auto ret = aclrtMemcpy(continuousMem.data(), continuousMem.getSizeInBytes(),
                           continuousValue.data(), continuousValue.size() * sizeof(uint8_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);

    uint16_t *opFlagMem = reinterpret_cast<uint16_t *>(continuousMem.data());
    AscendTensor<uint16_t, DIMS_3> opFlag(opFlagMem, { blockNum, FLAG_NUM, FLAG_SIZE });
    int64_t *attrMem = reinterpret_cast<int64_t *>(continuousMem.data() + opFlagSize);
    AscendTensor<int64_t, DIMS_1> attrsInputs(attrMem, { aicpu::TOPK_MULTISEARCH_ATTR_IDX_COUNT });

    APPERR_RETURN_IF_NOT_OK(
        computeMultisearchTopkParam(indexOffsetInputs, labelOffsetInputs, reorderFlagInputs, ntotals, offsetBlocks));

    // resultTemp
    AscendTensor<float16_t, DIMS_3, size_t>  minDistances(mem,
        { static_cast<size_t>(indexSize), static_cast<size_t>(batchSize), static_cast<size_t>(k) }, streamAicpu);
    AscendTensor<idx_t, DIMS_3, size_t>  minIndices(mem,
        { static_cast<size_t>(indexSize), static_cast<size_t>(batchSize), static_cast<size_t>(k) }, streamAicpu);
    APPERR_RETURN_IF_NOT_OK(initResult(minDistances,  minIndices));

    // 2. run the disance operators and topk operators
    const int maskSize = static_cast<int>(utils::divUp(this->codeBlockSize, BIT_OF_UINT8));
    int blockOffset = 0;
    int indexDoneCount = 0;
    int copyCount = 0;
    for (int indexId = 0; indexId < indexSize; ++indexId) {
        auto index = dynamic_cast<IndexSQ *>(indexes[indexId]);
        if (index == nullptr) {
            APP_LOG_ERROR("the index cast to IndexSQ failed, indexId=%d\n", indexId);
            continue;
        }
        int blocks = static_cast<int>(utils::divUp(ntotals[indexId], static_cast<idx_t>(computeBlockSize)));
        for (int i = 0; i < blocks; ++i) {
            int offset = i * computeBlockSize;
            int blockIdx = (static_cast<int>(offsetBlocks[indexId]) + i) % PAGE_BLOCKS;

            if (blockIdx == 0) {
                if ((static_cast<int>(offsetBlocks[indexId]) + i) > 0) {
                    for (int j = copyCount; j < indexDoneCount; ++j) {
                        ret = aclrtMemcpy(distances + (size_t)j * (size_t)n * (size_t)k,
                            (size_t)batchSize * (size_t)k * sizeof(float16_t), minDistances[j].data(),
                            (size_t)batchSize * (size_t)k * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_HOST);
                        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d",
                            (int)ret);
                        ret = aclrtMemcpy(labels + (size_t)j * (size_t)n * (size_t)k,
                            (size_t)batchSize * (size_t)k * sizeof(idx_t), minIndices[j].data(),
                            (size_t)batchSize * (size_t)k * sizeof(idx_t), ACL_MEMCPY_DEVICE_TO_HOST);
                        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d",
                            (int)ret);
                        ++copyCount;
                    }
                    auto ret = synchronizeStream(streamAicpu);
                    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                        "synchronizeStream aicpu stream failed: %i\n", ret);
                    indexDoneCount = indexId - 1;
                }
                // under topk operators
                int actualPageBlocks = std::min(blockNum - blockOffset, PAGE_BLOCKS);
                if (actualPageBlocks < PAGE_BLOCKS) {
                    attrsInputs[aicpu::TOPK_MULTISEARCH_ATTR_PAGE_BLOCK_NUM_IDX] = actualPageBlocks;
                }
                AscendTensor<uint32_t, DIMS_1> indexOffset(indexOffsetInputs.data() + blockOffset,
                    { actualPageBlocks });
                AscendTensor<uint32_t, DIMS_1> labelOffset(labelOffsetInputs.data() + blockOffset,
                    { actualPageBlocks });
                AscendTensor<uint16_t, DIMS_1> reorderFlag(reorderFlagInputs.data() + blockOffset,
                    { actualPageBlocks });
                AscendTensor<uint16_t, DIMS_3> flag(opFlag.data() + blockOffset * FLAG_NUM * FLAG_SIZE,
                    { actualPageBlocks, FLAG_NUM, FLAG_SIZE });

                std::vector<const AscendTensorBase *> input {&distResult, &minDistResult, &opSize, &flag, &attrsInputs,
                    &indexOffset, &labelOffset, &reorderFlag};
                std::vector<const AscendTensorBase *> output {&minDistances, &minIndices};
                runMultisearchTopkCompute(batchSize, input, output, streamAicpu);

                blockOffset += actualPageBlocks;
            }

            // under distance operators
            AscendTensor<uint8_t, DIMS_4> batchCode(index->codes[i]->data(),
                { codeBlockSize / CUBE_ALIGN, dims / CUBE_ALIGN, CUBE_ALIGN, CUBE_ALIGN });
            AscendTensor<float, DIMS_1> batchPreComp(index->preCompute[i]->data(), { this->codeBlockSize });
            AscendTensor<float16_t, DIMS_1> vdiff(index->vDiff.data(), { dims });
            AscendTensor<float16_t, DIMS_1> vmin(index->vMin.data(), { dims });
            AscendTensor<uint32_t, DIMS_2> actualSize(opSize[blockIdx].data(), { CORE_NUM, SIZE_ALIGN });
            AscendTensor<float16_t, DIMS_2, size_t> result = distResult[blockIdx].view();
            AscendTensor<float16_t, DIMS_2, size_t> minResult = minDistResult[blockIdx].view();
            AscendTensor<uint16_t, DIMS_2> flag(opFlag[offsetBlocks[indexId] + i].data(), { FLAG_NUM, FLAG_SIZE });
            AscendTensor<uint8_t, DIMS_2> mask(maskData.data(), { batchSize, maskSize });

            actualSize[0][IDX_ACTUAL_NUM] =
                std::min(static_cast<uint32_t>(index->ntotal - offset), static_cast<uint32_t>(computeBlockSize));
            actualSize[0][IDX_COMP_OFFSET] = maskOffset[indexId] + offset; // offset of each blockSize
            actualSize[0][IDX_MASK_LEN] = maskData.getSize(1);
            actualSize[0][IDX_USE_MASK] = 1;
            actualSize[0][IDX_CODE_SIZE] = index->codes[i]->size() / static_cast<size_t>(index->dims);

            std::vector<const AscendTensorBase *> input {
                &queries, &mask, &batchCode, &batchPreComp, &vdiff, &vmin, &actualSize};
            std::vector<const AscendTensorBase *> output {&result, &minResult, &flag};
            runSqDistMaskOperator(batchSize, input, output, stream);
        }
    }

    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream aicpu stream failed: %i\n", ret);

    // memcpy data back from dev to host
    for (int indexId = copyCount; indexId < indexSize; ++indexId) {
        ret = aclrtMemcpy(distances + (size_t)indexId * (size_t)n * (size_t)k,
            (size_t)batchSize * (size_t)k * sizeof(float16_t),  minDistances[indexId].data(),
            (size_t)batchSize * (size_t)k * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);

        ret = aclrtMemcpy(labels + (size_t)indexId * (size_t)n * (size_t)k,
            (size_t)batchSize * (size_t)k * sizeof(idx_t),  minIndices[indexId].data(),
            (size_t)batchSize * (size_t)k * sizeof(idx_t), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);
    }

    return APP_ERR_OK;
}

void IndexSQL2Aicpu::runSqDistOperator(int batch,
                                       const std::vector<const AscendTensorBase *> &input,
                                       const std::vector<const AscendTensorBase *> &output,
                                       aclrtStream stream) const
{
    IndexTypeIdx indexType =
        (this->dims != 64) ? IndexTypeIdx::ITI_SQ_DIST_L2 : IndexTypeIdx::ITI_SQ_DIST_DIM64_L2;
    std::vector<int> keys({batch, dims, codeBlockSize});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().runOp(indexType, opsKey, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
}

void IndexSQL2Aicpu::runSqDistMaskOperator(int batch,
                                           const std::vector<const AscendTensorBase *> &input,
                                           const std::vector<const AscendTensorBase *> &output,
                                           aclrtStream stream) const
{
    IndexTypeIdx indexMaskType =
        (this->dims != 64) ? IndexTypeIdx::ITI_SQ_DIST_MASK_L2 : IndexTypeIdx::ITI_SQ_DIST_MASK_DIM64_L2;
    std::vector<int> keys({batch, dims, codeBlockSize});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().runOp(indexMaskType, opsKey, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
}

} // namespace ascend