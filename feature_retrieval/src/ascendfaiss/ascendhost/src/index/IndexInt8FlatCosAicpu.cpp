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

#include <algorithm>
#include "ascenddaemon/utils/AscendTensor.h"
#include "ascenddaemon/utils/Limits.h"
#include "common/utils/CommonUtils.h"
#include "common/utils/SocUtils.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"
#include "index/IndexInt8FlatCosAicpu.h"

namespace ascend {
namespace {
const int L2NORM_COMPUTE_BATCH = 16384;
const int BURST_LEN = 64;
const int IDX_CODE_SIZE = 4;
const int PAGE_BLOCKS = 32;
const int FP16_ALGIN = 16;
const int64_t BIG_BATCH_MEM_SIZE_THRESHOLD = static_cast<int64_t>(2) * 1024 * 1024 * 1024;
}

IndexInt8FlatCosAicpu::IndexInt8FlatCosAicpu(int dim, int64_t resourceSize, int blockSize)
    : IndexInt8Flat<float16_t>(dim, MetricType::METRIC_INNER_PRODUCT, resourceSize, blockSize)
{
    this->isSupportMultiSearch = true;
    this->int8FlatIndexType = Int8FlatIndexType::INT8_FLAT_COS;
    // TIK算子做了大batch优化（AscendC也做了但是方案不一样，因此支持的大batch也不同），
    // 但是batch224场景经计算分析最大会占用2G的共享内存，如果共享内存过小反而会因为申请了
    // 非共享内存而导致性能下降，因此要限制下共享内存大小来决定是否开启大batch性能优化
    if (!faiss::ascend::SocUtils::GetInstance().IsAscend910B() && resourceSize >= BIG_BATCH_MEM_SIZE_THRESHOLD) {
        this->searchBatchSizes = { 224, 192, 128, 112, 96, 64, 48, 36, 32, 24, 18, 16, 12, 8, 6, 4, 2, 1 };
    }
    APP_LOG_DEBUG("using strategy %d\n", static_cast<int>(deviceMemMng.GetStrategy()));
}

IndexInt8FlatCosAicpu::~IndexInt8FlatCosAicpu() {}

APP_ERROR IndexInt8FlatCosAicpu::init()
{
    APP_ERROR ret = resetTopkCompOp();
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, ret, "failed to reset topk op");
    ret = resetMultisearchTopkCompOp();
    APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, ret, "failed to reset multisearch topk op");
    APPERR_RETURN_IF_NOT_OK(int8L2Norm->init());
    if (faiss::ascend::SocUtils::GetInstance().IsAscend910B()) {
        APPERR_RETURN_IF_NOT_OK(resetAscendcDistCompOp(codeBlockSize));
    } else {
        APPERR_RETURN_IF_NOT_OK(resetDistCompOp(codeBlockSize));
    }
    return APP_ERR_OK;
}

APP_ERROR IndexInt8FlatCosAicpu::calL2norm(int num, AscendTensor<int8_t, DIMS_2> &rawTensor,
                                           AscendTensor<float16_t, 1> &precompData)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    AscendTensor<uint32_t, DIMS_2> actualNum(mem, { utils::divUp(num, L2NORM_COMPUTE_BATCH), SIZE_ALIGN }, stream);
    AscendTensor<int8_t, DIMS_2> codesData(mem, { num, this->dims }, stream);
    if (faiss::ascend::SocUtils::GetInstance().IsRunningInHost()) {
        auto retAcl = aclrtMemcpy(codesData.data(), codesData.getSizeInBytes(), rawTensor.data(),
            rawTensor.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(retAcl == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", retAcl);
        int8L2Norm->dispatchL2NormTask(codesData, precompData, actualNum, stream);
    } else {
        int8L2Norm->dispatchL2NormTask(rawTensor, precompData, actualNum, stream);
    }
    auto ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "synchronizeStream failed: %i\n", ret);
    return APP_ERR_OK;
}

APP_ERROR IndexInt8FlatCosAicpu::copyNormByIndice(int64_t startIndice, int64_t length, int64_t normOffset,
                                                  AscendTensor<float16_t, DIMS_1> &precompData)
{
    int64_t normVecId = startIndice / this->codeBlockSize;
    int64_t noremVecOffset = startIndice % this->codeBlockSize;
    int64_t mantainLen = length;
    int64_t precompDataOffset = normOffset;
    while (mantainLen > 0) {
        int64_t curVeclest = static_cast<int64_t>(this->normBase.at(normVecId)->size()) - noremVecOffset;
        int64_t cpyNum = std::min(curVeclest, mantainLen);
        auto ret = aclrtMemcpy(this->normBase.at(normVecId)->data() + noremVecOffset,
            cpyNum * sizeof(float16_t),
            precompData.data() + precompDataOffset, cpyNum * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "copy precompData failed %d", ret);
        mantainLen -= cpyNum;
        noremVecOffset = (noremVecOffset + cpyNum) % this->codeBlockSize;
        precompDataOffset += cpyNum;
        if (noremVecOffset + cpyNum >= this->codeBlockSize) {
            normVecId++;
        }
    };
    return APP_ERR_OK;
}

APP_ERROR IndexInt8FlatCosAicpu::addVectors(AscendTensor<int8_t, DIMS_2> &rawData)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();

    // 1. dispatch the task of compute the l2 norm of code data
    int num = rawData.getSize(0);
    AscendTensor<float16_t, 1> precompData(mem, { static_cast<int>(utils::roundUp(num, CUBE_ALIGN)) }, stream);
    auto l2Ret = calL2norm(num, rawData, precompData);
    APPERR_RETURN_IF_NOT_FMT(APP_ERR_OK == l2Ret, APP_ERR_INNER_ERROR,
                             "calculate L2norm for rawData failed! %d", l2Ret);

    // 2. save code
    APP_ERROR ret = IndexInt8Flat::addVectors(rawData);
    APPERR_RETURN_IF(ret, ret);

    int vecSize = static_cast<int>(utils::divUp(this->ntotal, static_cast<idx_t>(this->codeBlockSize)));
    int newVecSize = static_cast<int>(utils::divUp(this->ntotal + num, static_cast<idx_t>(this->codeBlockSize)));
    idx_t lastOffset = ntotal % static_cast<idx_t>(this->codeBlockSize);
    int offset = 0;

    // if normBase[vecSize - 1] is not full
    if (lastOffset != 0) {
        int cpyNum = std::min(num, this->codeBlockSize * vecSize - static_cast<int>(ntotal));
        APPERR_RETURN_IF_NOT_FMT(vecSize >= 1, APP_ERR_INNER_ERROR, "vecSize[%d], error!", vecSize);
        auto ret = aclrtMemcpy(this->normBase.at(vecSize - 1)->data() + lastOffset,
                               this->codeBlockSize * sizeof(float16_t),
                               precompData.data(), cpyNum * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
        offset += cpyNum;
    }

    for (int i = vecSize; i < newVecSize; ++i) {
        int cpyNum = std::min(num - offset, this->codeBlockSize);
        auto ret = aclrtMemcpy(this->normBase.at(i)->data(), this->codeBlockSize * sizeof(float16_t),
                               precompData.data() + offset, cpyNum * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
        offset += cpyNum;
    }
    this->ntotal += static_cast<idx_t>(num);
    return APP_ERR_OK;
}

APP_ERROR IndexInt8FlatCosAicpu::groupSearchImpl(int n, const int8_t *x, int k, float16_t *distances, idx_t *labels)
{
    // 1. init output data
    AscendTensor<int8_t, DIMS_2> queries(const_cast<int8_t *>(x), { n, dims });
    AscendTensor<float16_t, DIMS_2> outDistances(distances, { n, k });
    AscendTensor<idx_t, DIMS_2> outIndices(labels, { n, k });
    AscendTensor<float16_t, DIMS_2> minDistances({ n, k });
    AscendTensor<int64_t, DIMS_2> minIndices({ n, k });

    // 2. compute distance by code page
    AscendTensor<uint8_t, DIMS_2> mask;
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto ret = prepareMaskData(n, stream, mask);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "prepareMaskData error %d", ret);
    APP_LOG_INFO("groupSearchImpl n=%d, k=%d, maxDeviceCapacity=%lld, dims=%d, maxDeviceDataNum=%zu, shapeSize=%u", n,
                 k, deviceMemMng.GetDeviceBuffer(), dims,
                 utils::divDown(deviceMemMng.GetDeviceCapacity(), static_cast<size_t>(dims)), baseShaped.size());
    DeviceStorgeInfo deviceStorgeInfo(utils::divDown(deviceMemMng.GetDeviceCapacity(), static_cast<size_t>(dims)),
                                      DFT_MAX_DEVICE_BLOCK_COUNT, DFT_MAX_HOST_BLOCK_COUNT);
    auto grpSpliter = HeteroBlockGroupMgr::Create(ntotal, pageSize, codeBlockSize, &deviceStorgeInfo, baseShaped);
    IndexSearchContext ctx(queries, k, minDistances, minIndices);
    auto retSearch = searchInGroups(*grpSpliter, ctx, mask); // 任务分组
    APPERR_RETURN_IF(retSearch, retSearch);

    // 3. memcpy data back from dev to host
    ret = aclrtMemcpy(outDistances.data(), outDistances.getSizeInBytes(), minDistances.data(),
        minDistances.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

    ret = aclrtMemcpy(outIndices.data(), outIndices.getSizeInBytes(), minIndices.data(), minIndices.getSizeInBytes(),
        ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

    grpSpliter->OrderIds2IndexIds(n, k, labels);
    return APP_ERR_OK;
}

APP_ERROR IndexInt8FlatCosAicpu::searchInGroup(size_t grpId, const HeteroBlockGroupMgr &grpSpliter,
                                               IndexSearchContext &ctx, IndexSearchTensorShare &tensorCond,
                                               AscendTensor<uint8_t, DIMS_2> &mask)
{
    auto &grp = grpSpliter.At(grpId);
    if (grp.blocks.empty()) {
        APP_LOG_INFO("searchInGroup %d, empty", grpId);
        return APP_ERR_OK;
    }
    APP_LOG_INFO("searchInGroup grpId=%d, blockCount=%d", grpId, grp.blocks.size());
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    size_t blockNum = grp.blocks.size();
    auto &mem = resources.getMemoryManager();
    auto nq = ctx.queries.getSize(0);

    numThresholds = calNumThresholds(nq, ctx.queries.getSize(1), BURST_LEN);
    AscendTensor<float16_t, DIMS_2> topKThreshold(mem, { nq, numThresholds }, stream);
    APPERR_RETURN_IF_NOT_OK(getThreshold(topKThreshold, nq, ctx.topN, grpId, ctx.minDistances));

    // 1. run the topk operator to wait for distance result and csompute topk
    runTopkCompute(tensorCond.getDistResult(grpId), tensorCond.getMinDistResult(grpId), *tensorCond.opSizeVec.at(grpId),
        *tensorCond.opFlagVec.at(grpId), *tensorCond.attrsInputVec.at(grpId), ctx.minDistances, ctx.minIndices,
        streamAicpuPtr->GetStream());

    // 2. run the disance operator to compute the distance
    const int dimBlock = utils::divUp(this->codeBlockSize, CUBE_ALIGN);
    const int dimFeature = utils::divUp(this->dims, CUBE_ALIGN_INT8);

    for (size_t i = 0; i < blockNum; i++) {
        AscendTensor<int8_t, DIMS_4> shaped(baseShaped[grp.blocks.at(i).blockId]->data(),
            { dimBlock, dimFeature, CUBE_ALIGN, CUBE_ALIGN_INT8 });
        AscendTensor<float16_t, DIMS_1> codesNorm(normBase[grp.blocks.at(i).blockId]->data(), { codeBlockSize });
        auto dist = tensorCond.getDistResult(grpId)[i].view();
        auto minDist = tensorCond.getMinDistResult(grpId)[i].view();
        auto flag = tensorCond.opFlagVec.at(grpId)->operator[](i).view();
        auto actualSize = tensorCond.opSizeVec.at(grpId)->operator[](i).view();

        std::vector<const AscendTensorBase *> input {&(ctx.queries), &mask, &shaped, tensorCond.queriesNorm.get(),
                                                     &codesNorm, &actualSize};
        if (deviceMemMng.GetStrategy() == DevMemStrategy::HETERO_MEM) {
            input.push_back(&topKThreshold);
        }
        std::vector<const AscendTensorBase *> output {&dist, &minDist, &flag};
        runDistCompute(nq, input, output, stream);

        if ((grp.groupType == HeteroBlockGroupType::BGT_PURE_HOST && (i + 1) % DFT_MAX_HOST_BLOCK_COUNT == 0) ||
            (i + 1 == blockNum)) {
            auto ret = synchronizeStream(stream);
            APPERR_RETURN_IF_FMT(ret != ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream stream error: %i\n", ret);
            for (size_t nextInvalidateNum = 0; nextInvalidateNum <= i; ++nextInvalidateNum) {
                baseShaped[grp.blocks.at(nextInvalidateNum).blockId]->pushData(false);
            }
        }
    }

    auto ret = synchronizeStream(streamAicpuPtr->GetStream());
    APPERR_RETURN_IF_FMT(ret != ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream aicpu stream failed: %i\n", ret);

    APP_LOG_INFO("searchInGroup grpId=%d, blockCount=%d complete", grpId, grp.blocks.size());
    return APP_ERR_OK;
}

APP_ERROR IndexInt8FlatCosAicpu::prepareIndexSearchTensorShare(const HeteroBlockGroupMgr &grpSpliter,
                                                               IndexSearchContext &ctx, IndexSearchTensorShare &outCond)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    int8L2Norm->dispatchL2NormTask(ctx.queries, *outCond.queriesNorm, *outCond.actualNum, stream);

    for (size_t grpId = 0; grpId < grpSpliter.Count(); ++grpId) {
        auto &grp = grpSpliter.At(grpId);
        size_t blockNum = grp.blocks.size();
        outCond.opFlagVec.at(grpId)->zero();

        std::vector<int64_t> attrs(aicpu::TOPK_FLAT_ATTR_IDX_COUNT);
        attrs[aicpu::TOPK_FLAT_ATTR_ASC_IDX] = 0;
        attrs[aicpu::TOPK_FLAT_ATTR_K_IDX] = ctx.topN;
        attrs[aicpu::TOPK_FLAT_ATTR_BURST_LEN_IDX] = BURST_LEN;
        attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_NUM_IDX] = static_cast<int64_t>(blockNum);
        attrs[aicpu::TOPK_FLAT_ATTR_PAGE_IDX] = (int)grpId;
        attrs[aicpu::TOPK_FLAT_ATTR_PAGE_NUM_IDX] = (int)grpSpliter.Count();
        attrs[aicpu::TOPK_FLAT_ATTR_PAGE_SIZE_IDX] = (int)grpSpliter.GetGroupSize();
        attrs[aicpu::TOPK_FLAT_ATTR_QUICK_HEAP] = 1;
        attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_SIZE] = this->codeBlockSize;
        auto ret = aclrtMemcpy(outCond.attrsInputVec.at(grpId)->data(),
                               outCond.attrsInputVec.at(grpId)->getSizeInBytes(), attrs.data(),
                               attrs.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "aclrtMemcpy grpId=%d attrsInput: %i\n",
            grpId, ret);

        std::vector<uint32_t> opSizeHost(blockNum * CORE_NUM * SIZE_ALIGN);
        int opSizeHostOffset = CORE_NUM * SIZE_ALIGN;
        int opSizeHostIdx = 0;

        int idxMaskLen = static_cast<int>(utils::divUp(this->ntotal, BINARY_BYTE_SIZE));
        int idxUseMask = (this->maskData != nullptr) ? 1 : 0;

        for (size_t i = 0; i < blockNum; i++) {
            opSizeHost[opSizeHostIdx + IDX_ACTUAL_NUM] = grp.blocks.at(i).dataNum;
            opSizeHost[opSizeHostIdx + IDX_COMP_OFFSET] =
                (uint32_t)(grp.blocks.at(i).blockId * this->codeBlockSize);
            opSizeHost[opSizeHostIdx + IDX_MASK_LEN] = (size_t)idxMaskLen;
            opSizeHost[opSizeHostIdx + IDX_USE_MASK] = (size_t)idxUseMask;
            opSizeHost[opSizeHostIdx + IDX_CODE_SIZE] = this->baseShaped[grp.blocks.at(i).blockId]->size() /
                                                                         static_cast<size_t>(this->dims);
            opSizeHostIdx += opSizeHostOffset;
        }

        ret = aclrtMemcpy(outCond.opSizeVec.at(grpId)->data(), outCond.opSizeVec.at(grpId)->getSizeInBytes(),
                          opSizeHost.data(), opSizeHost.size() * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "aclrtMemcpy grpId=%d opSizeHostInput: %i\n",
            grpId, ret);
    }
    return APP_ERR_OK;
}

APP_ERROR IndexInt8FlatCosAicpu::searchInGroups(const HeteroBlockGroupMgr &grpSpliter, IndexSearchContext &ctx,
                                                AscendTensor<uint8_t, DIMS_2> &mask)
{
    IndexSearchTensorShare tensorCond(resources, ctx, grpSpliter, *this);
    prepareIndexSearchTensorShare(grpSpliter, ctx, tensorCond);
    for (size_t grpId = 0; grpId < grpSpliter.Count(); ++grpId) {
        APP_ERROR ret = searchInGroup(grpId, grpSpliter, ctx, tensorCond, mask);
        APPERR_RETURN_IF(ret, ret);
    }
    return APP_ERR_OK;
}

int IndexInt8FlatCosAicpu::calNumThresholds(int n, int dims, int burstLen) const
{
    if (deviceMemMng.GetStrategy() == DevMemStrategy::PURE_DEVICE_MEM) {
        return 0;
    }
    ASCEND_THROW_IF_NOT(n != 0 && dims != 0);

    // centroidsNumEachLoop 以及 numThresholds的值
    // 均按照distance_int8_cos_maxs.py算子中queries_num_each_loop 以及 centroids_num_each_loop来进行计算
    int centroidsNumEachLoop = std::min(std::min((48 / std::min(48, n)) * 512, 1024), 512 * 1024 / dims);
    int numThresholds = (centroidsNumEachLoop + burstLen - 1) / burstLen * 2;
    APP_LOG_DEBUG("in searchImpl, numThresholds = %d\n", numThresholds);
    return numThresholds;
}

void IndexInt8FlatCosAicpu::initSearchResult(int indexesSize, int n, int k, float16_t *distances, idx_t *labels)
{
    AscendTensor<float16_t, DIMS_3> outDistances(distances, { indexesSize, n, k });
    AscendTensor<idx_t, DIMS_3> outIndices(labels, { indexesSize, n, k });
    outDistances.initValue(Limits<float16_t>::getMin());
    outIndices.initValue(std::numeric_limits<idx_t>::max());
}

APP_ERROR IndexInt8FlatCosAicpu::prepareMaskData(int n, const aclrtStream &stream,
    AscendTensor<uint8_t, DIMS_2> &retMask)
{
    auto &mem = resources.getMemoryManager();
    int idxMaskLen = static_cast<int>(utils::divUp(this->ntotal, BINARY_BYTE_SIZE));
    if (this->maskData != nullptr && this->maskOnDevice) {
        // maskData is dynamic generated by one batch, and which is on device
        // operator has verification, mask must be blockMaskSize at least.
        retMask = std::move(AscendTensor<uint8_t, DIMS_2>(this->maskData,
            { n, std::max(idxMaskLen, this->blockMaskSize) }));
        return APP_ERR_OK;
    }
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
        if (ret != ACL_SUCCESS) {
            APP_LOG_ERROR("aclrtMemcpy mask error");
            return ret;
        }
    }
    retMask = std::move(mask);
    return APP_ERR_OK;
}

APP_ERROR IndexInt8FlatCosAicpu::searchImpl(int n, const int8_t *x, int k, float16_t *distances, idx_t *labels)
{
    deviceMemMng.Prefetch(baseShaped);
    if (deviceMemMng.UsingGroupSearch()) {
        return groupSearchImpl(n, x, k, distances, labels);
    }
    // 1. init output data
    AscendTensor<int8_t, DIMS_2> queries(const_cast<int8_t *>(x), { n, dims });
    AscendTensor<float16_t, DIMS_2> outDistances(distances, { n, k });
    AscendTensor<idx_t, DIMS_2> outIndices(labels, { n, k });
    AscendTensor<float16_t, DIMS_2> minDistances({ n, k });
    AscendTensor<int64_t, DIMS_2> minIndices({ n, k });

    // 2. compute distance by code page
    numThresholds = calNumThresholds(n, dims, BURST_LEN);

    int pageNum = static_cast<int>(utils::divUp(this->ntotal, pageSize));
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    AscendTensor<uint8_t, DIMS_2> mask;
    auto ret = prepareMaskData(n, stream, mask);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "prepareMaskData error %d", ret);
    auto &mem = resources.getMemoryManager();
    AscendTensor<float16_t, DIMS_1> queriesNorm(mem, { utils::roundUp(n, CUBE_ALIGN) }, stream);
    AscendTensor<uint32_t, DIMS_2> actualNum(mem, { utils::divUp(n, L2NORM_COMPUTE_BATCH), SIZE_ALIGN }, stream);
    int8L2Norm->dispatchL2NormTask(queries, queriesNorm, actualNum, stream);
    for (int pageId = 0; pageId < pageNum; ++pageId) {
        APP_ERROR ret = searchPaged(pageId, queries, k, minDistances, minIndices, mask, queriesNorm);
        APPERR_RETURN_IF(ret, ret);
    }

    // 3. memcpy data back from dev to host
    ret = aclrtMemcpy(outDistances.data(), outDistances.getSizeInBytes(), minDistances.data(),
        minDistances.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);

    ret = aclrtMemcpy(outIndices.data(), outIndices.getSizeInBytes(), minIndices.data(), minIndices.getSizeInBytes(),
        ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", (int)ret);

    return APP_ERR_OK;
}

APP_ERROR IndexInt8FlatCosAicpu::searchPaged(int pageId, AscendTensor<int8_t, DIMS_2> &queries, int k,
    AscendTensor<float16_t, DIMS_2> &minDistances, AscendTensor<int64_t, DIMS_2> &minIndices,
    AscendTensor<uint8_t, DIMS_2> &mask, AscendTensor<float16_t, DIMS_1> &queriesNorm)
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

    AscendTensor<float16_t, DIMS_3, size_t> distResult(mem, { (size_t)blockNum, (size_t)nq, (size_t)codeBlockSize },
        stream);
    AscendTensor<float16_t, DIMS_3, size_t> minDistResult(mem,
        { (size_t)blockNum, (size_t)nq, (size_t)this->burstsOfBlock }, stream);
    AscendTensor<uint32_t, DIMS_3> opSize(mem, { blockNum, CORE_NUM, SIZE_ALIGN }, stream);
    AscendTensor<uint16_t, DIMS_3> opFlag(mem, { blockNum, flagNum, FLAG_SIZE }, stream);
    opFlag.zero();

    AscendTensor<float16_t, DIMS_2> topKThreshold(mem, { nq, numThresholds }, stream);
    APPERR_RETURN_IF_NOT_OK(getThreshold(topKThreshold, nq, k, pageId, minDistances));

    // attrs: [0]asc, [1]k, [2]burst_len, [3]block_num
    int pageNum = (int)(utils::divUp(this->ntotal, (size_t)this->pageSize));
    AscendTensor<int64_t, DIMS_1> attrsInput(mem, { aicpu::TOPK_FLAT_ATTR_IDX_COUNT }, stream);
    std::vector<int64_t> attrs(aicpu::TOPK_FLAT_ATTR_IDX_COUNT);
    attrs[aicpu::TOPK_FLAT_ATTR_ASC_IDX] = 0;
    attrs[aicpu::TOPK_FLAT_ATTR_K_IDX] = k;
    attrs[aicpu::TOPK_FLAT_ATTR_BURST_LEN_IDX] = BURST_LEN;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_NUM_IDX] = blockNum;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_IDX] = pageId;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_NUM_IDX] = pageNum;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_SIZE_IDX] = this->pageSize;
    attrs[aicpu::TOPK_FLAT_ATTR_QUICK_HEAP] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_SIZE] = this->codeBlockSize;

    auto ret = aclrtMemcpy(attrsInput.data(), attrsInput.getSizeInBytes(), attrs.data(), attrs.size() * sizeof(int64_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "aclrtMemcpy error %d", ret);

    // 1. run the topk operator to wait for distance result and csompute topk
    runTopkCompute(distResult, minDistResult, opSize, opFlag, attrsInput, minDistances, minIndices, streamAicpu);

    // 2. run the disance operator to compute the distance
    // opSize Host to Device,reduce communication
    uint32_t idxMaskLen = static_cast<uint32_t>(utils::divUp(this->ntotal, BINARY_BYTE_SIZE));
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
        opSizeHost[opSizeHostIdx + IDX_CODE_SIZE] = this->baseShaped[i]->size() / static_cast<size_t>(this->dims);
        opSizeHostIdx += opSizeHostOffset;
        offset += this->codeBlockSize;
    }
    ret = aclrtMemcpy(opSize.data(), opSize.getSizeInBytes(), opSizeHost.data(), opSizeHost.size() * sizeof(uint32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "aclrtMemcpy error %d", ret);

    const int dim1 = utils::divUp(this->codeBlockSize, CUBE_ALIGN);
    const int dim2 = utils::divUp(this->dims, CUBE_ALIGN_INT8);
    for (int i = 0; i < blockNum; ++i) {
        AscendTensor<int8_t, DIMS_4> shaped(baseShaped[blockOffset + (size_t)i]->data(),
            { dim1, dim2, CUBE_ALIGN, CUBE_ALIGN_INT8 });
        AscendTensor<float16_t, DIMS_1> codesNorm(normBase[blockOffset + i]->data(), { codeBlockSize });
        auto dist = distResult[i].view();
        auto minDist = minDistResult[i].view();
        auto flag = opFlag[i].view();
        auto actualSize = opSize[i].view();

        std::vector<const AscendTensorBase *> input {&queries, &mask, &shaped, &queriesNorm,
                                                     &codesNorm, &actualSize};
        if (deviceMemMng.GetStrategy() == DevMemStrategy::HETERO_MEM) {
            input.push_back(&topKThreshold);
        }
        std::vector<const AscendTensorBase *> output {&dist, &minDist, &flag};
        runDistCompute(nq, input, output, stream);
        deviceMemMng.PushDataAfterSearch(baseShaped, blockOffset, i, blockNum, stream);
    }

    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream default stream: %i\n",
        ret);

    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream aicpu stream failed: %i\n", ret);

    return APP_ERR_OK;
}

APP_ERROR IndexInt8FlatCosAicpu::resetDistCompOp(int codeNum) const
{
    std::string opTypeName = deviceMemMng.GetStrategy() == DevMemStrategy::HETERO_MEM ?
        "DistanceInt8CosMaxsFilter" : "DistanceInt8CosMaxs";
    if (faiss::ascend::SocUtils::GetInstance().IsAscend910B()) {
        opTypeName = "AscendcDistInt8FlatCos";
    }
    IndexTypeIdx indexType = deviceMemMng.GetStrategy() == DevMemStrategy::HETERO_MEM ?
        IndexTypeIdx::ITI_INT8_COS_FILTER : IndexTypeIdx::ITI_INT8_COS;
    for (auto batch : searchBatchSizes) {
        std::vector<int64_t> queryShape({ batch, dims });
        std::vector<int64_t> maskShape({ batch, utils::divUp(codeNum, 8) }); // divUp to 8
        std::vector<int64_t> codeShape({ codeNum / CUBE_ALIGN, dims / CUBE_ALIGN_INT8, CUBE_ALIGN, CUBE_ALIGN_INT8 });
        std::vector<int64_t> queriesNormShape({ (batch + FP16_ALGIN - 1) / FP16_ALGIN * FP16_ALGIN });
        std::vector<int64_t> codesNormShape({ codeNum });
        std::vector<int64_t> sizeShape({ CORE_NUM, SIZE_ALIGN });
        std::vector<int64_t> resultShape({ batch, codeNum });
        std::vector<int64_t> minResultShape({ batch, this->burstsOfBlock });
        std::vector<int64_t> flagShape({ flagNum, FLAG_SIZE });

        std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
            { ACL_INT8, queryShape },
            { ACL_UINT8, maskShape },
            { ACL_INT8, codeShape },
            { ACL_FLOAT16, queriesNormShape },
            { ACL_FLOAT16, codesNormShape },
            { ACL_UINT32, sizeShape },
        };
        if (opTypeName == "DistanceInt8CosMaxsFilter") {
            // number of vectors to deal with in each loop per core in dist op
            int numThresholds = calNumThresholds(batch, dims, BURST_LEN);
            std::vector<int64_t> thresholdShape({ batch, numThresholds });
            input.push_back({ACL_FLOAT16, thresholdShape});
        }

        std::vector<std::pair<aclDataType, std::vector<int64_t>>> output {
            { ACL_FLOAT16, resultShape },
            { ACL_FLOAT16, minResultShape },
            { ACL_UINT16, flagShape }
        };
        std::vector<int> keys({batch, dims, codeBlockSize});
        OpsMngKey opsKey(keys);
        auto ret = DistComputeOpsManager::getInstance().resetOp(opTypeName, indexType, opsKey, input, output);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "op init failed: %i", ret);
    }
    return APP_ERR_OK;
}

APP_ERROR IndexInt8FlatCosAicpu::resetAscendcDistCompOp(int codeNum) const
{
    std::string opTypeName = "AscendcDistanceInt8CosMaxsWithMask";
    IndexTypeIdx indexType = IndexTypeIdx::ASCENDC_ITI_INT8_COS_MASK;
    for (auto batch : searchBatchSizes) {
        std::vector<int64_t> queryShape({ batch, dims });
        std::vector<int64_t> maskShape({ batch, utils::divUp(codeNum, 8) }); // divUp to 8
        std::vector<int64_t> codeShape({ codeNum / CUBE_ALIGN, dims / CUBE_ALIGN_INT8, CUBE_ALIGN, CUBE_ALIGN_INT8 });
        std::vector<int64_t> queriesNormShape({ (batch + FP16_ALGIN - 1) / FP16_ALGIN * FP16_ALGIN });
        std::vector<int64_t> codesNormShape({ codeNum });
        std::vector<int64_t> sizeShape({ CORE_NUM, SIZE_ALIGN });
        std::vector<int64_t> resultShape({ batch, codeNum });
        std::vector<int64_t> minResultShape({ batch, this->burstsOfBlock });
        std::vector<int64_t> flagShape({ flagNum, FLAG_SIZE });

        std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
            { ACL_INT8, queryShape },
            { ACL_UINT8, maskShape },
            { ACL_INT8, codeShape },
            { ACL_FLOAT16, queriesNormShape },
            { ACL_FLOAT16, codesNormShape },
            { ACL_UINT32, sizeShape },
        };
        if (opTypeName == "DistanceInt8CosMaxsFilter") {
            // number of vectors to deal with in each loop per core in dist op
            int numThresholds = calNumThresholds(batch, dims, BURST_LEN);
            std::vector<int64_t> thresholdShape({ batch, numThresholds });
            input.push_back({ACL_FLOAT16, thresholdShape});
        }

        std::vector<std::pair<aclDataType, std::vector<int64_t>>> output {
            { ACL_FLOAT16, resultShape },
            { ACL_FLOAT16, minResultShape },
            { ACL_UINT16, flagShape }
        };
        std::vector<int> keys({batch, dims, codeBlockSize});
        OpsMngKey opsKey(keys);
        auto ret = DistComputeOpsManager::getInstance().resetOp(opTypeName, indexType, opsKey, input, output);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "op init failed: %i", ret);
    }
    return APP_ERR_OK;
}

void IndexInt8FlatCosAicpu::runDistCompute(int batch,
                                           const std::vector<const AscendTensorBase *> &input,
                                           const std::vector<const AscendTensorBase *> &output,
                                           aclrtStream stream, uint32_t actualNum) const
{
    VALUE_UNUSED(actualNum);
    IndexTypeIdx type = deviceMemMng.GetStrategy() == DevMemStrategy::HETERO_MEM ?
        IndexTypeIdx::ITI_INT8_COS_FILTER : IndexTypeIdx::ITI_INT8_COS;
    std::vector<int> keys({batch, dims, codeBlockSize});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().runOp(type, opsKey, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
}

APP_ERROR IndexInt8FlatCosAicpu::getThreshold(
    AscendTensor<float16_t, DIMS_2>& topKThreshold, int nq, int k, int pageId,
    const AscendTensor<float16_t, DIMS_2> &minDistances) const
{
    if (deviceMemMng.GetStrategy() == DevMemStrategy::PURE_DEVICE_MEM) {
        APP_LOG_DEBUG("Do nothing for PURE_DEVICE_MEM strategy.\n");
        return APP_ERR_OK;
    }
    APP_LOG_DEBUG("Need to prepare the threshold for HETERO_MEM strategy.\n");

    // 将minDistances从Device侧拷贝到Host侧(首页不拷贝，使用初始化的fp16的最小值)
    std::vector<float16_t> minDistancesHost(nq * k, Limits<float16_t>::getMin());

    if (pageId != 0) {
        auto ret = aclrtMemcpy(minDistancesHost.data(), nq * k * sizeof(float16_t),
            minDistances.data(), minDistances.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(
            ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);
    }
    // 在host侧制作 topKThresholdHost
    std::vector<float16_t> topKThresholdHost(nq * numThresholds);  // 保存host侧计算得到的阈值
    for (int i = 0; i < nq; ++i) {  // 遍历各个查询向量
        std::fill_n(topKThresholdHost.begin() + i * numThresholds, numThresholds,
                    minDistancesHost[i * k]);
    }
    // 把在host侧计算得到的阈值拷贝到device侧
    auto ret = aclrtMemcpy(topKThreshold.data(), topKThreshold.getSizeInBytes(), topKThresholdHost.data(),
        topKThresholdHost.size() * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(
        ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Mem operator error %d", ret);

    return APP_ERR_OK;
}


} // namespace ascend