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


#include "index/IndexFlatIPAicpu.h"
#include "ascend/AscendIndex.h"
#include "common/utils/CommonUtils.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"
#include "ascenddaemon/utils/Limits.h"

namespace ascend {
namespace {
// opSize offset
constexpr int IDX_ACTUAL_NUM = 0;
constexpr int IDX_COMP_OFFSET = 1; // mask offset of code block
constexpr int IDX_MASK_LEN = 2;    // mask len of each query
constexpr int IDX_USE_MASK = 3;    // use mask flag
const int PAGE_BLOCKS = 32;
}

static std::mutex mtx;

static std::string GetDistOpName(int nq)
{
    std::string opName = "DistanceFlatIP";
    if (faiss::ascend::SocUtils::GetInstance().IsAscend910B()) {
        opName = "DistanceFlatIP";
    } else {
        opName = (nq > OPTIMIZE_BATCH_THRES) ? "DistanceFlatIPMaxsBatch" : "DistanceFlatIPMaxs";
    }
    return opName;
}

IndexFlatIPAicpu::IndexFlatIPAicpu(int dim, int64_t resourceSize) : IndexFlat(dim, resourceSize) {}

IndexFlatIPAicpu::~IndexFlatIPAicpu()
{
    for (auto& pair: disIPOpInputDesc) {
        std::vector<aclTensorDesc*>& inputDesc = pair.second;
        for (auto item: inputDesc) {
            aclDestroyTensorDesc(item);
        }
    }
    for (auto& pair: disIPOpOutputDesc) {
        std::vector<aclTensorDesc*>& outputDesc = pair.second;
        for (auto item: outputDesc) {
            aclDestroyTensorDesc(item);
        }
    }
}

APP_ERROR IndexFlatIPAicpu::init()
{
    searchBatchSizes = {128, 64, 48, 36, 32, 30, 24, 18, 16, 12, 8, 6, 4, 2, 1};
    APP_ERROR ret = resetTopkCompOp();
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, ret, "failed to reset topk op");
    ret = resetMultisearchTopkCompOp();
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INVALID_PARAM, "failed to reset multisearch topk op", ret);
    return resetDistCompOp(this->blockSize);
}

APP_ERROR IndexFlatIPAicpu::addVectors(AscendTensor<float16_t, DIMS_2> &rawData)
{
    APP_ERROR ret = IndexFlat::addVectors(rawData);
    APPERR_RETURN_IF(ret, ret);

    int num = rawData.getSize(0);
    this->ntotal += static_cast<idx_t>(num);

    return APP_ERR_OK;
}

APP_ERROR IndexFlatIPAicpu::searchImpl(AscendTensor<float16_t, DIMS_2> &queries, int k,
    AscendTensor<float16_t, DIMS_2> &outDistances, AscendTensor<idx_t, DIMS_2> &outIndices)
{
    // 1. generate result variable
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    int nq = queries.getSize(0);

    AscendTensor<float16_t, DIMS_2> maxDistances(mem, {nq, k}, stream);
    AscendTensor<int64_t, DIMS_2> maxIndices(mem, {nq, k}, stream);

    // 2. compute distance by code page
    size_t pageNum = utils::divUp(this->ntotal, pageSize);
    for (size_t pageId = 0; pageId < pageNum; ++pageId) {
        APP_ERROR ret = searchPaged(pageId, pageNum, queries, maxDistances, maxIndices);
        APPERR_RETURN_IF(ret, ret);
    }

    // memcpy data back from dev to host
    auto ret = aclrtMemcpy(outDistances.data(), outDistances.getSizeInBytes(),
                           maxDistances.data(), maxDistances.getSizeInBytes(),
                           ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy outDistances back to host");

    ret = aclrtMemcpy(outIndices.data(), outIndices.getSizeInBytes(),
                      maxIndices.data(), maxIndices.getSizeInBytes(),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy outIndices back to host");

    return APP_ERR_OK;
}

APP_ERROR IndexFlatIPAicpu::searchPaged(size_t pageId, size_t pageNum, AscendTensor<float16_t, DIMS_2> &queries,
    AscendTensor<float16_t, DIMS_2> &maxDistances, AscendTensor<int64_t, DIMS_2> &maxIndices)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    int nq = queries.getSize(0);
    int k = maxDistances.getSize(1);
    size_t pageOffset = pageId * (size_t)this->pageSize;
    size_t blockOffset = pageId * (size_t)this->pageSize / (size_t)blockSize;
    int computeNum = std::min(this->ntotal - pageOffset, static_cast<idx_t>(this->pageSize));
    int blockNum = utils::divUp(computeNum, this->blockSize);
    int burstLen = BURST_LEN_HIGH;
    auto curBurstsOfBlock = GetBurstsOfBlock(nq, this->blockSize, burstLen);

    AscendTensor<float16_t, DIMS_3, size_t> distResult(mem, { static_cast<size_t>(blockNum), static_cast<size_t>(nq),
        static_cast<size_t>(blockSize) }, stream);
    AscendTensor<float16_t, DIMS_3, size_t> maxDistResult(mem,
        { static_cast<size_t>(blockNum), static_cast<size_t>(nq), static_cast<size_t>(curBurstsOfBlock) }, stream);

    uint32_t opFlagSize = static_cast<uint32_t>(blockNum * CORE_NUM * FLAG_SIZE) * sizeof(uint16_t);
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
    attrs[aicpu::TOPK_FLAT_ATTR_ASC_IDX] = 0;
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
    AscendTensor<uint16_t, DIMS_3> opFlag(opFlagMem, { blockNum, CORE_NUM, FLAG_SIZE });
    uint32_t *opSizeMem = reinterpret_cast<uint32_t *>(continuousMem.data() + opFlagSize);
    AscendTensor<uint32_t, DIMS_3> opSize(opSizeMem, { blockNum, CORE_NUM, SIZE_ALIGN });
    int64_t *attrMem = reinterpret_cast<int64_t *>(continuousMem.data() + opFlagSize + opSizeLen);
    AscendTensor<int64_t, DIMS_1> attrsInput(attrMem, { aicpu::TOPK_FLAT_ATTR_IDX_COUNT });

    // 1. run the topk operator to wait for distance result and compute topk
    runTopkCompute(distResult, maxDistResult, opSize, opFlag, attrsInput, maxDistances, maxIndices, streamAicpu);

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
        auto dist = distResult[i].view();
        auto maxDist = maxDistResult[i].view();
        auto actualSize = opSize[i].view();
        auto flag = opFlag[i].view();

        std::vector<const AscendTensorBase *> input {&queries, &mask, &shaped, &actualSize};
        std::vector<const AscendTensorBase *> output {&dist, &maxDist, &flag};
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

void IndexFlatIPAicpu::runTopkCompute(AscendTensor<float16_t, DIMS_3> &dists,
                                      AscendTensor<float16_t, DIMS_3> &maxdists,
                                      AscendTensor<uint32_t, DIMS_3> &sizes,
                                      AscendTensor<uint16_t, DIMS_3> &flags,
                                      AscendTensor<int64_t, DIMS_1> &attrs,
                                      AscendTensor<float16_t, DIMS_2> &outdists,
                                      AscendTensor<int64_t, DIMS_2> &outlabel,
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

APP_ERROR IndexFlatIPAicpu::resetTopkOfflineOp()
{
    auto topkCompOpReset = [&](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("TopkFlat");
        int burstLen = BURST_LEN_HIGH;
        auto curBurstsOfBlock = GetBurstsOfBlock(batch, this->blockSize, burstLen);
        std::vector<int64_t> shape0 { 0, batch, this->blockSize };
        std::vector<int64_t> shape1 { 0, batch, curBurstsOfBlock };
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

APP_ERROR IndexFlatIPAicpu::resetTopkCompOp()
{
    if (isUseOnlineOp()) {
        return resetTopkOnline();
    } else {
        return resetTopkOfflineOp();
    }
}

void IndexFlatIPAicpu::runTopkCompute(AscendTensor<float16_t, DIMS_3, size_t> &dists,
                                      AscendTensor<float16_t, DIMS_3, size_t> &maxdists,
                                      AscendTensor<uint32_t, DIMS_3> &sizes,
                                      AscendTensor<uint16_t, DIMS_3> &flags,
                                      AscendTensor<int64_t, DIMS_1> &attrs,
                                      AscendTensor<float16_t, DIMS_2> &outdists,
                                      AscendTensor<int64_t, DIMS_2> &outlabel,
                                      aclrtStream stream)
{
    int batch = static_cast<int>(dists.getSize(1));
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
        auto ret = runTopkOnlineOp(batch, params.flags.getSize(1), params, stream);
        ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run online TopkFlat operator failed: %i\n", ret);
    }
}

APP_ERROR IndexFlatIPAicpu::resetMultisearchTopkCompOp()
{
    if (isUseOnlineOp()) {
        return resetOnlineMultisearchTopk();
    } else {
        return resetOfflineMultisearchTopk(IndexTypeIdx::ITI_FLAT_IP_TOPK_MULTISEARCH, CORE_NUM);
    }
}

APP_ERROR IndexFlatIPAicpu::searchImpl(std::vector<Index *> indexes, int n, int batchSize, const float16_t *x,
    int k, float16_t *distances, idx_t *labels)
{
    size_t indexSize = indexes.size();
    std::vector<idx_t> ntotals(indexSize);
    std::vector<idx_t> offsetBlocks(indexSize + 1, 0);
    for (size_t i = 0; i < indexSize; ++i) {
        ntotals[i] = indexes[i]->ntotal;
        offsetBlocks[i + 1] = offsetBlocks[i] + utils::divUp(ntotals[i], blockSize);
    }

    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    int blockNum = static_cast<int>(offsetBlocks[indexSize]);
    int burstLen = BURST_LEN_HIGH;
    auto curBurstsOfBlock = GetBurstsOfBlock(batchSize, this->blockSize, burstLen);
    // 1. costruct the distance operator param
    // aicore params
    AscendTensor<float16_t, DIMS_2> queries(const_cast<float16_t *>(x), { batchSize, dims });
    AscendTensor<uint32_t, DIMS_3> opSize(mem, { PAGE_BLOCKS, CORE_NUM, SIZE_ALIGN }, stream);
    AscendTensor<float16_t, DIMS_3, size_t> distResult(mem,
        { static_cast<size_t>(PAGE_BLOCKS), static_cast<size_t>(batchSize),
        static_cast<size_t>(blockSize) }, stream);
    AscendTensor<float16_t, DIMS_3, size_t> maxDistResult(mem,
        { static_cast<size_t>(PAGE_BLOCKS), static_cast<size_t>(batchSize),
        static_cast<size_t>(curBurstsOfBlock) }, stream);

    // aicpu params
    AscendTensor<uint32_t, DIMS_1> indexOffsetInputs(mem, { blockNum }, streamAicpu);
    AscendTensor<uint32_t, DIMS_1> labelOffsetInputs(mem, { blockNum }, streamAicpu);
    AscendTensor<uint16_t, DIMS_1> reorderFlagInputs(mem, { blockNum }, streamAicpu);

    uint32_t opFlagSize = static_cast<uint32_t>(blockNum * CORE_NUM * FLAG_SIZE) * sizeof(uint16_t);
    uint32_t attrsSize = aicpu::TOPK_MULTISEARCH_ATTR_IDX_COUNT * sizeof(int64_t);
    uint32_t continuousMemSize = opFlagSize + attrsSize;
    // 1) aclrtMemcpy比AscendTensor::zero更高效
    // 2) 使用连续内存来减少aclrtMemcpy的调用次数
    AscendTensor<uint8_t, DIMS_1, uint32_t> continuousMem(mem, { continuousMemSize }, stream);
    std::vector<uint8_t> continuousValue(continuousMemSize, 0);
    uint8_t *data = continuousValue.data();
    int64_t *attrs = reinterpret_cast<int64_t *>(data + opFlagSize);
    attrs[aicpu::TOPK_MULTISEARCH_ATTR_ASC_IDX] = 0;
    attrs[aicpu::TOPK_MULTISEARCH_ATTR_K_IDX] = k;
    attrs[aicpu::TOPK_MULTISEARCH_ATTR_BURST_LEN_IDX] = burstLen;
    attrs[aicpu::TOPK_MULTISEARCH_ATTR_INDEX_NUM_IDX] = static_cast<int64_t>(indexSize);
    attrs[aicpu::TOPK_MULTISEARCH_ATTR_PAGE_BLOCK_NUM_IDX] = PAGE_BLOCKS;
    attrs[aicpu::TOPK_MULTISEARCH_ATTR_QUICK_HEAP] = 1;
    attrs[aicpu::TOPK_MULTISEARCH_ATTR_BLOCK_SIZE] = blockSize;
    
    auto ret = aclrtMemcpy(continuousMem.data(), continuousMem.getSizeInBytes(),
        continuousValue.data(), continuousValue.size() * sizeof(uint8_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

    uint16_t *opFlagMem = reinterpret_cast<uint16_t *>(continuousMem.data());
    AscendTensor<uint16_t, DIMS_3> opFlag(opFlagMem, { blockNum, CORE_NUM, FLAG_SIZE });
    int64_t *attrMem = reinterpret_cast<int64_t *>(continuousMem.data() + opFlagSize);
    AscendTensor<int64_t, DIMS_1> attrsInputs(attrMem, { aicpu::TOPK_MULTISEARCH_ATTR_IDX_COUNT });
    APPERR_RETURN_IF_NOT_OK(
        computeMultisearchTopkParam(indexOffsetInputs, labelOffsetInputs, reorderFlagInputs,
            ntotals, offsetBlocks));

    // resultTemp
    AscendTensor<float16_t, DIMS_3, size_t> maxDistances(mem, { static_cast<size_t>(indexSize),
        static_cast<size_t>(batchSize), static_cast<size_t>(k) }, streamAicpu);
    AscendTensor<idx_t, DIMS_3, size_t> maxIndices(mem, { static_cast<size_t>(indexSize),
        static_cast<size_t>(batchSize), static_cast<size_t>(k) }, streamAicpu);
    APPERR_RETURN_IF_NOT_OK(initResult(maxDistances, maxIndices));

    AscendTensor<uint8_t, DIMS_2> mask(mem, { batchSize, this->blockMaskSize }, stream);

    // 2. run the disance operators and topk operators
    int blockOffset = 0;
    int indexDoneCount = 0;
    int copyCount = 0;
    const int dim1 = utils::divUp(this->blockSize, CUBE_ALIGN);
    const int dim2 = utils::divUp(this->dims, CUBE_ALIGN);
    for (size_t indexId = 0; indexId < indexSize; ++indexId) {
        auto index = dynamic_cast<IndexFlat *>(indexes[indexId]);
        if (index == nullptr) {
            APP_LOG_ERROR("the index cast to Index failed, indexId=%d\n", indexId);
            continue;
        }
        int blocks = static_cast<int>(utils::divUp(ntotals[indexId], blockSize));
        for (int i = 0; i < blocks; ++i) {
            int offset = i * blockSize;
            int blockIdx = (static_cast<int64_t>(offsetBlocks[indexId]) + i) % PAGE_BLOCKS;

            if (blockIdx == 0 && (offsetBlocks[indexId] + static_cast<idx_t>(i)) > 0) {
                for (int j = copyCount; j < indexDoneCount; ++j) {
                    ret = aclrtMemcpy(distances + static_cast<size_t>(j) * static_cast<size_t>(n) *
                        static_cast<size_t>(k), static_cast<size_t>(batchSize) *
                        static_cast<size_t>(k) * sizeof(float16_t), maxDistances[j].data(),
                        static_cast<size_t>(batchSize) * static_cast<size_t>(k) * sizeof(float16_t),
                        ACL_MEMCPY_DEVICE_TO_HOST);
                    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d",
                        ret);
                    ret = aclrtMemcpy(labels + static_cast<size_t>(j) * static_cast<size_t>(n) *
                        static_cast<size_t>(k), static_cast<size_t>(batchSize) *
                        static_cast<size_t>(k) * sizeof(idx_t), maxIndices[j].data(),
                        static_cast<size_t>(batchSize) * static_cast<size_t>(k) * sizeof(idx_t),
                        ACL_MEMCPY_DEVICE_TO_HOST);
                    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
                    ++copyCount;
                }
                auto ret = synchronizeStream(streamAicpu);
                APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                    "synchronizeStream aicpu stream failed: %i\n", ret);
                indexDoneCount = static_cast<int>(indexId) - 1;
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
                AscendTensor<uint16_t, DIMS_3> flag(opFlag.data() + blockOffset * CORE_NUM * FLAG_SIZE,
                    { actualPageBlocks, CORE_NUM, FLAG_SIZE });
                std::vector<const AscendTensorBase *> input {&distResult, &maxDistResult, &opSize, &flag,
                                                             &attrsInputs, &indexOffset, &labelOffset, &reorderFlag};
                std::vector<const AscendTensorBase *> output {&maxDistances, &maxIndices};
                topkFlatIpMultisearchParams params(batchSize, distResult, maxDistResult, opSize,
                                                   flag, attrsInputs, indexOffset, labelOffset,
                                                   reorderFlag, maxDistances, maxIndices);
                calculateTopkMultisearch(params, streamAicpu);
                blockOffset += actualPageBlocks;
            }

            // under distance operators
            AscendTensor<float16_t, DIMS_4> shaped(index->getBaseShaped()[i]->data(),
                { dim1, dim2, CUBE_ALIGN, CUBE_ALIGN });
            auto dist = distResult[blockIdx].view();
            auto maxDist = maxDistResult[blockIdx].view();
            auto actualSize = opSize[blockIdx].view();
            auto flag = opFlag[offsetBlocks[indexId] + i].view();
            
            actualSize[0][IDX_ACTUAL_NUM] =
                std::min(static_cast<uint32_t>(index->ntotal - offset), static_cast<uint32_t>(blockSize));
            actualSize[0][IDX_USE_MASK] = 0; // not use mask

            std::vector<const AscendTensorBase *> input {&queries, &mask, &shaped, &actualSize};
            std::vector<const AscendTensorBase *> output {&dist, &maxDist, &flag};
            runDistCompute(batchSize, input, output, stream);
        }
    }

    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream default stream: %i\n", ret);
    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream aicpu stream failed: %i\n", ret);

    // memcpy data back from dev to host
    for (size_t indexId = static_cast<size_t>(copyCount); indexId < indexSize; ++indexId) {
        size_t totalIndex = static_cast<size_t>(indexId) * static_cast<size_t>(n) * static_cast<size_t>(k);
        size_t totalBatchSize = static_cast<size_t>(batchSize) * static_cast<size_t>(k);

        ret = aclrtMemcpy(distances + totalIndex, totalBatchSize * sizeof(float16_t), maxDistances[indexId].data(),
            totalBatchSize * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

        ret = aclrtMemcpy(labels + totalIndex, totalBatchSize * sizeof(idx_t), maxIndices[indexId].data(),
            totalBatchSize * sizeof(idx_t), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
    }

    return APP_ERR_OK;
}

APP_ERROR IndexFlatIPAicpu::initResult(AscendTensor<float16_t, DIMS_3, size_t> &distances,
    AscendTensor<idx_t, DIMS_3, size_t> &indices) const
{
    std::vector<float16_t> distancesInit(distances.getSizeInBytes() / sizeof(float16_t), Limits<float16_t>::getMin());
    std::vector<idx_t> indicesInit(indices.getSizeInBytes() / sizeof(idx_t), std::numeric_limits<idx_t>::max());
 
    auto ret = aclrtMemcpy(distances.data(), distances.getSizeInBytes(), distancesInit.data(),
        distancesInit.size() * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
    ret = aclrtMemcpy(indices.data(), indices.getSizeInBytes(), indicesInit.data(),
        indicesInit.size() * sizeof(idx_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
 
    return APP_ERR_OK;
}

void IndexFlatIPAicpu::runMultisearchTopkOnline(topkFlatIpMultisearchParams &opParams,
                                                const std::vector<const AscendTensorBase *> &input,
                                                const std::vector<const AscendTensorBase *> &output,
                                                aclrtStream streamAicpu)
{
    AscendOpDesc desc("TopkMultisearch");
    int burstLen = BURST_LEN_HIGH;
    int batch = opParams.batch;
    auto curBurstsOfBlock = GetBurstsOfBlock(batch, this->blockSize, burstLen);
    std::vector<int64_t> shape0 { static_cast<int64_t>(opParams.distResult.getSize(0)), batch, this->blockSize };
    std::vector<int64_t> shape1 { static_cast<int64_t>(opParams.maxDistResult.getSize(0)), batch, curBurstsOfBlock };
    std::vector<int64_t> shape2 { static_cast<int64_t>(opParams.opSize.getSize(0)), CORE_NUM, SIZE_ALIGN };
    std::vector<int64_t> shape3 { static_cast<int64_t>(opParams.flag.getSize(0)),
                                  static_cast<int64_t>(opParams.flag.getSize(1)), FLAG_SIZE };
    std::vector<int64_t> shape4 { aicpu::TOPK_MULTISEARCH_ATTR_IDX_COUNT };
    std::vector<int64_t> shape5 { static_cast<int64_t>(opParams.indexOffset.getSize(0)) };
    std::vector<int64_t> shape6 { static_cast<int64_t>(opParams.labelOffset.getSize(0)) };
    std::vector<int64_t> shape7 { static_cast<int64_t>(opParams.reorderFlag.getSize(0)) };
    std::vector<int64_t> shape8 { static_cast<int64_t>(opParams.maxDistances.getSize(0)), batch,
                                  static_cast<int64_t>(opParams.maxDistances.getSize(2)) };
    std::vector<int64_t> shape9 { static_cast<int64_t>(opParams.maxIndices.getSize(0)), batch,
                                  static_cast<int64_t>(opParams.maxIndices.getSize(2)) };
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
    for (auto &data : input) {
        topkOpInput.emplace_back(aclCreateDataBuffer(data->getVoidData(), data->getSizeInBytes()));
    }
    std::vector<aclDataBuffer *> topkOpOutput;
    for (auto &data : output) {
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

void IndexFlatIPAicpu::calculateTopkMultisearch(topkFlatIpMultisearchParams &opParams, aclrtStream streamAicpu)
{
    std::vector<const AscendTensorBase *> input {&opParams.distResult, &opParams.maxDistResult, &opParams.opSize,
                                                 &opParams.flag,  &opParams.attrsInputs, &opParams.indexOffset,
                                                 &opParams.labelOffset, &opParams.reorderFlag};
    std::vector<const AscendTensorBase *> output {&opParams.maxDistances, &opParams.maxIndices};
    if (isUseOnlineOp()) {
        runMultisearchTopkOnline(opParams, input, output, streamAicpu);
    } else {
        runMultisearchTopkCompute(opParams.batch, input, output, streamAicpu);
    }
}

void IndexFlatIPAicpu::runDistCompute(int batch,
                                      const std::vector<const AscendTensorBase *> &input,
                                      const std::vector<const AscendTensorBase *> &output,
                                      aclrtStream stream)
{
    if (!isUseOnlineOp()) {
        IndexTypeIdx indexType = IndexTypeIdx::ITI_FLAT_IP;
        std::vector<int> keys({batch, dims});
        OpsMngKey opsKey(keys);
        auto ret = DistComputeOpsManager::getInstance().runOp(indexType, opsKey, input, output, stream);
        ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
    } else {
        std::vector<aclTensorDesc *> inputDesc = disIPOpInputDesc.at(batch);
        std::vector<aclTensorDesc *> outputDesc = disIPOpOutputDesc.at(batch);
        std::vector<aclDataBuffer *> distOpInput;
        for (auto &data : input) {
            distOpInput.emplace_back(aclCreateDataBuffer(data->getVoidData(), data->getSizeInBytes()));
        }
        std::vector<aclDataBuffer *> distOpOutput;
        for (auto &data : output) {
            distOpOutput.emplace_back(aclCreateDataBuffer(data->getVoidData(), data->getSizeInBytes()));
        }
        std::string opName = GetDistOpName(batch);
        const char *opType = opName.c_str();
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

APP_ERROR IndexFlatIPAicpu::resetDistOnlineOp(int batch,
                                              std::vector<std::pair<aclDataType, std::vector<int64_t>>> &input,
                                              std::vector<std::pair<aclDataType, std::vector<int64_t>>> &output)
{
    // 在线转换算子
    std::string opName = GetDistOpName(batch);
    const char *opType = opName.c_str();
    int numInputs = static_cast<int>(input.size());
    for (int i = 0; i < numInputs; i++) {
        aclTensorDesc *desc = aclCreateTensorDesc(input[i].first, input[i].second.size(),
                                                  input[i].second.data(), ACL_FORMAT_ND);
        if (desc == nullptr) {
            return APP_ERR_INNER_ERROR;
        }
        disIPOpInputDesc[batch].emplace_back(desc);
    }
    int numOutputs = static_cast<int>(output.size());
    for (int i = 0; i < numOutputs; i++) {
        aclTensorDesc *desc = aclCreateTensorDesc(output[i].first, output[i].second.size(),
                                                  output[i].second.data(), ACL_FORMAT_ND);
        if (desc == nullptr) {
            return APP_ERR_INNER_ERROR;
        }
        disIPOpOutputDesc[batch].emplace_back(desc);
    }
    aclopAttr *opAttr = aclopCreateAttr();
    auto ret = aclSetCompileopt(ACL_OP_JIT_COMPILE, "enable");
    if (ret != APP_ERR_OK) {
        aclopDestroyAttr(opAttr);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "enable jit compile fail opName:%s, %i\n", opType, ret);
    }
    ret = aclopCompile(opType, numInputs, disIPOpInputDesc[batch].data(), numOutputs,
                       disIPOpOutputDesc[batch].data(), opAttr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, nullptr);
    aclopDestroyAttr(opAttr);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "op init failed: %i", ret);
    return APP_ERR_OK;
}

APP_ERROR IndexFlatIPAicpu::resetDistCompOp(int numLists)
{
    IndexTypeIdx indexType = IndexTypeIdx::ITI_FLAT_IP;
    for (auto batch : searchBatchSizes) {
        int burstLen = BURST_LEN_HIGH;
        auto curBurstsOfBlock = GetBurstsOfBlock(batch, this->blockSize, burstLen);
        std::string opTypeName = GetDistOpName(batch);

        std::vector<int64_t> queryShape({ batch, dims });
        std::vector<int64_t> maskShape({ batch, blockMaskSize });
        std::vector<int64_t> coarseCentroidsShape({ utils::divUp(numLists, CUBE_ALIGN),
            utils::divUp(dims, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN });
        std::vector<int64_t> sizeShape({ CORE_NUM, SIZE_ALIGN });
        std::vector<int64_t> distResultShape({ batch, numLists });
        std::vector<int64_t> maxResultShape({ batch, curBurstsOfBlock });
        std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });

        std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
            { ACL_FLOAT16, queryShape },
            { ACL_UINT8, maskShape },
            { ACL_FLOAT16, coarseCentroidsShape },
            { ACL_UINT32, sizeShape }
        };
        std::vector<std::pair<aclDataType, std::vector<int64_t>>> output {
            { ACL_FLOAT16, distResultShape },
            { ACL_FLOAT16, maxResultShape },
            { ACL_UINT16, flagShape }
        };

        if (isUseOnlineOp()) {
            auto ret = resetDistOnlineOp(batch, input, output);
            APPERR_RETURN_IF_NOT_FMT(APP_ERR_OK == ret, APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                                     "compile distance op init failed:%i, batch: %d\n", ret, batch);
        } else {
            std::vector<int> keys({batch, dims});
            OpsMngKey opsKey(keys);
            auto ret = DistComputeOpsManager::getInstance().resetOp(opTypeName, indexType, opsKey, input, output);
            APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "op init failed: %i", ret);
        }
    }

    return APP_ERR_OK;
}
} // namespace ascend
