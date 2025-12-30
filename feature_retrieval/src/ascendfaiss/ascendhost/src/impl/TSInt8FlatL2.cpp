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

#include "ascendhost/include/impl/TSInt8FlatL2.h"
#include <bitset>
#include <iostream>
#include <set>
#include "common/utils/DataType.h"
#include "faiss/impl/FaissAssert.h"
#include "fp16.h"

namespace ascend {

namespace {
const int BURST_LEN = 64;
// The value range of dim
const std::vector<int> DIMS_INT8L2 = {64, 128, 256, 384, 512, 768, 1024};
}

TSInt8FlatL2::TSInt8FlatL2(uint32_t deviceId, uint32_t dim, uint32_t tokenNum, uint64_t resources,
    uint32_t customAttrLen, uint32_t customAttrBlockSize)
    : TSBase(tokenNum, customAttrLen, customAttrBlockSize),
      IndexInt8FlatL2Aicpu(dim, resources)
{
    FAISS_THROW_IF_NOT_FMT(std::find(DIMS_INT8L2.begin(), DIMS_INT8L2.end(), dim) !=
        DIMS_INT8L2.end(), "Unsupported dims %u", dim);
    code_size = static_cast<int>(dim);
    this->deviceId = deviceId;
    auto ret = TSBase::initialize(deviceId);
    FAISS_THROW_IF_NOT_FMT(APP_ERR_OK == ret, "failed to init TSBase, ERRCODE:%d", ret);
    ret = IndexInt8FlatL2Aicpu::init();
    FAISS_THROW_IF_NOT_FMT(APP_ERR_OK == ret, "failed to init IndexInt8FlatL2Aicpu, ERRCODE:%d", ret);
    ret = resetInt8L2DistCompute(codeBlockSize, false);
    FAISS_THROW_IF_NOT_FMT(APP_ERR_OK == ret, "failed to resetDistCompOp, ERRCODE:%d", ret);
    ret = resetInt8L2DistCompute(codeBlockSize, true);
    FAISS_THROW_IF_NOT_FMT(APP_ERR_OK == ret, "failed to resetDistCompOp, ERRCODE:%d", ret);
}

APP_ERROR TSInt8FlatL2::addFeatureWithLabels(int64_t n, const void *features,
    const faiss::ascend::FeatureAttr *attrs, const int64_t *labels, const uint8_t *customAttr,
    const faiss::ascend::ExtraValAttr *)
{
    std::set<int64_t> uniqueLabels(labels, labels + n);
    APPERR_RETURN_IF_NOT_LOG(uniqueLabels.size() == static_cast<size_t>(n), APP_ERR_INVALID_PARAM,
        "the labels is not unique");
    for (int64_t i = 0; i < n; i++) {
        APPERR_RETURN_IF_NOT_FMT(label2Idx.find(*(labels + i)) == label2Idx.end(), APP_ERR_INVALID_PARAM,
            "the label[%ld] is already exists", *(labels + i));
    }

    addFeatureAttrs(n, attrs, customAttr);
    //  cast uint8 pointer to int8
    addWithIds(n, reinterpret_cast<const int8_t *>(features), reinterpret_cast<const idx_t *>(labels));
    for (int64_t i = static_cast<int64_t>(this->ntotal) - n; i < static_cast<int64_t>(this->ntotal); ++i) {
        label2Idx[ids[i]] = i;
    }
    return APP_ERR_OK;
}

APP_ERROR TSInt8FlatL2::getFeatureByLabel(int64_t n, const int64_t *labels, void *features) const
{
    APP_LOG_INFO("TSInt8FlatL2::getFeatureByLabel start");
    std::vector<int64_t> queryIds;
    for (int64_t i = 0; i < n; ++i) {
        auto it = label2Idx.find(*(labels + i));
        APPERR_RETURN_IF_NOT_FMT(it != label2Idx.end(), APP_ERR_INVALID_PARAM,
            "the label[%ld] does not exists", *(labels + i));
        queryIds.emplace_back(it->second);
    }
    const size_t idsNum = queryIds.size();
    for (size_t i = 0; i < idsNum; ++i) {
        queryVectorByIdx(queryIds[i], reinterpret_cast<uint8_t *>(features) + (i * code_size),
            (idsNum - i) * code_size);
    }
    APP_LOG_INFO("TSInt8FlatL2::getFeatureByLabel end");
    return APP_ERR_OK;
};

APP_ERROR TSInt8FlatL2::getBaseByRange(uint32_t offset, uint32_t num, int64_t *labels, void *features,
    faiss::ascend::FeatureAttr *attributes, faiss::ascend::ExtraValAttr *)
{
    APP_LOG_INFO("TSInt8FlatL2::getBaseByRange start");
    std::vector<int8_t> baseVectors(static_cast<int64_t>(num) * this->dims);
    getVectorsAiCpu(offset, num, baseVectors);
    auto ret =
        memcpy_s(features, num * this->dims * sizeof(int8_t), baseVectors.data(), baseVectors.size() * sizeof(int8_t));
    APPERR_RETURN_IF_NOT_FMT(ret == 0, APP_ERR_INNER_ERROR, "Memcpy_s features failed(%d).", ret);
    ret = memcpy_s(labels, num * sizeof(int64_t), this->ids.data() + offset, num * sizeof(int64_t));
    APPERR_RETURN_IF_NOT_FMT(ret == 0, APP_ERR_INNER_ERROR, "Memcpy_s labels failed(%d).", ret);
    ret = memcpy_s(attributes, num * sizeof(faiss::ascend::FeatureAttr), this->featureAttrs.data() + offset,
        num * sizeof(faiss::ascend::FeatureAttr));
    APPERR_RETURN_IF_NOT_FMT(ret == 0, APP_ERR_INNER_ERROR, "Memcpy_s attributes failed(%d).", ret);
    getBaseEnd();
    APP_LOG_INFO("TSInt8FlatL2::getBaseByRange end");
    return APP_ERR_OK;
}

void TSInt8FlatL2::removeIdsImpl(const std::vector<int64_t> &indices)
{
    APP_LOG_INFO("TSInt8FlatL2 removeIdsImpl operation start. \n");
    // move the end data to the locate of delete data
    int zRegionHeight = CUBE_ALIGN;
    int dimAlignSize = utils::divUp(this->code_size, CUBE_ALIGN_INT8);
    int removedCnt = static_cast<int>(indices.size());
    std::vector<uint64_t> srcAddr(removedCnt);
    std::vector<uint64_t> dstAddr(removedCnt);
    std::string opName = "RemovedataShaped";
    auto &mem = pResources->getMemoryManager();
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    for (int i = 0; i < removedCnt; i++) {
        int srcIdx = static_cast<int64_t>(this->ntotal) - i - 1;
        int srcIdx1 = srcIdx /this->codeBlockSize;
        int srcIdx2 = srcIdx % this->codeBlockSize;
        int dstIdx1 = indices[i] / this->codeBlockSize;
        int dstIdx2 = indices[i] % this->codeBlockSize;

        int8_t *srcDataPtr = baseShaped[srcIdx1]->data() + (srcIdx2 / zRegionHeight) *
            (dimAlignSize * zRegionHeight * CUBE_ALIGN_INT8) + (srcIdx2 % zRegionHeight) * CUBE_ALIGN_INT8;
        int8_t *dstDataPtr = baseShaped[dstIdx1]->data() + (dstIdx2 / zRegionHeight) *
            (dimAlignSize * zRegionHeight * CUBE_ALIGN_INT8) + (dstIdx2 % zRegionHeight) * CUBE_ALIGN_INT8;
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
    std::vector<int64_t> attrs(aicpu::REMOVEDATA_SHAPED_ATTR_IDX_COUNT);
    attrs[aicpu::REMOVEDATA_SHAPED_ATTR_DATA_TYPE] = faiss::ascend::INT8;
    attrs[aicpu::REMOVEDATA_SHAPED_ATTR_ZREGION_HEIGHT] = zRegionHeight;
    attrs[aicpu::REMOVEDATA_SHAPED_ATTR_DIM_ALIGN_NUM] = dimAlignSize;
    attrs[aicpu::REMOVEDATA_SHAPED_ATTR_CUBE_ALIGN] = CUBE_ALIGN_INT8;
    ret = aclrtMemcpy(attrsInput.data(), attrsInput.getSizeInBytes(), attrs.data(), attrs.size() * sizeof(int64_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to copy to device (error %d)", ret);
    LaunchOpTwoInOneOut<uint64_t, DIMS_1, ACL_UINT64, int64_t, DIMS_1, ACL_INT64, uint64_t, DIMS_1, ACL_UINT64>(opName,
        stream, srcInput, attrsInput, dstInput);

    ret = synchronizeStream(stream);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to acl Synchronize Streame (error %d)", ret);
    APP_LOG_INFO("TSInt8FlatL2 removeIdsImpl operation finished. \n");
}

void TSInt8FlatL2::removeFeatureByIds(const std::vector<int64_t> &ids)
{
    // removeIdsImpl使用算子对内存搬运进行优化，但是前提是内存都在device侧
    removeIdsImpl(ids);
    removeNormBase(ids, static_cast<size_t>(codeBlockSize), static_cast<size_t>(ntotal), faiss::ascend::INT32,
        normBase);
}

APP_ERROR TSInt8FlatL2::deleteFeatureByToken(int64_t count, const uint32_t *tokens)
{
    APPERR_RETURN_IF_NOT_LOG(!enableSaveHostMemory, APP_ERR_INNER_ERROR,
        "enableSaveHostMemory not support deletebytoken");
    std::vector<int64_t> removeIds;
    std::unordered_set<uint32_t> uniqueTokens(tokens, tokens + count);
    APP_LOG_INFO("TSInt8FlatL2::deleteFeatureByToken start count:%zu", removeIds.size());
    for (auto tokenId : uniqueTokens) {
        std::vector<int64_t> tmpIds;
        getIdsByToken(tokenId, tmpIds);
        std::move(tmpIds.begin(), tmpIds.end(), std::back_inserter(removeIds));
    }
    if (removeIds.empty()) {
        return APP_ERR_OK;
    }
    std::sort(removeIds.begin(), removeIds.end(), std::greater<::int64_t>());
    // use TSBase function to delete attr
    deleteAttrByIds(removeIds);
    //  remove labels and set new idx of last label
    removeLabels(removeIds);
    // remove baseShaped and base norm
    removeFeatureByIds(removeIds);
    // release the space  of baseShape and norm
    removeInvalidData(this->ntotal, removeIds.size());
    this->ntotal -= removeIds.size();
    APP_LOG_INFO("TSInt8FlatL2::deleteFeatureByToken delete count:%zu", removeIds.size());
    return APP_ERR_OK;
}

APP_ERROR TSInt8FlatL2::delFeatureWithLabels(int64_t n, const int64_t *labels)
{
    APP_LOG_INFO("TSInt8FlatL2::delFeatureWithLabels start cound:%ld", n);
    std::unordered_set<int64_t> removeIdSets;
    for (int64_t i = 0; i < n; ++i) {
        auto it = label2Idx.find(*(labels + i));
        if (it != label2Idx.end()) {
            removeIdSets.insert(it->second);
        }
    }
    if (removeIdSets.empty()) {
        return APP_ERR_OK;
    }
    std::vector<int64_t> removeIds(removeIdSets.begin(), removeIdSets.end());
    std::sort(removeIds.begin(), removeIds.end(), std::greater<::int64_t>());
    // use TSBase function to delete attr
    deleteAttrByIds(removeIds);
    //  remove labels and set new idx of last label
    removeLabels(removeIds);
    // remove baseShaped and base norm
    removeFeatureByIds(removeIds);
    // release the space  of baseShape and norm
    removeInvalidData(this->ntotal, removeIds.size());
    this->ntotal -= removeIds.size();
    APP_LOG_INFO("TSInt8FlatL2::delFeatureWithLabels delete count:%zu", removeIds.size());
    return APP_ERR_OK;
}

void TSInt8FlatL2::removeLabels(const std::vector<int64_t> &removeIds)
{
    FAISS_THROW_IF_NOT_MSG(this->ntotal > 0, "no data need to delete");
    int64_t lastIdx = static_cast<int64_t>(this->ntotal - 1);
    for (auto pos : removeIds) {
        uint64_t delLabel = ids[pos];
        ids[pos] = ids[lastIdx];
        label2Idx[ids[lastIdx]] = pos; // update new position label idx
        label2Idx.erase(delLabel); // delete original pos label idx
        --lastIdx;
    }
    ids.resize(lastIdx + 1);
}

APP_ERROR TSInt8FlatL2::searchPagedWithMasks(int pageIdx, int batch, const int8_t *features, int topK,
    AscendTensor<uint8_t, DIMS_3> &masks, AscendTensor<float16_t, DIMS_2> &outDistanceOnDevice,
    AscendTensor<int64_t, DIMS_2> &outIndicesOnDevice)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    int pageOffset = pageIdx * this->pageSize;
    int blockOffset = pageIdx * this->pageSize / codeBlockSize;
    int computeNum = std::min(static_cast<int>(this->ntotal) - pageOffset, this->pageSize);
    int blockNum = utils::divUp(computeNum, this->codeBlockSize);

    AscendTensor<int8_t, DIMS_2> queries(const_cast<int8_t *>(features), { batch, dims });

    auto ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream default stream: %i\n",
        ret);
    AscendTensor<float16_t, DIMS_3, size_t> distResult(mem,
        { static_cast<size_t>(blockNum), static_cast<size_t>(batch), static_cast<size_t>(codeBlockSize) }, stream);
    AscendTensor<float16_t, DIMS_3, size_t> minDistResult(mem,
        { static_cast<size_t>(blockNum), static_cast<size_t>(batch), static_cast<size_t>(this->burstsOfBlock) },
        stream);
    AscendTensor<uint32_t, DIMS_3> opSize(mem, { blockNum, CORE_NUM, SIZE_ALIGN }, stream);
    uint32_t opFlagSize = static_cast<uint64_t>((blockNum * FLAG_NUM * FLAG_SIZE)) * sizeof(uint16_t);
    uint32_t attrsSize = aicpu::TOPK_FLAT_ATTR_IDX_COUNT * sizeof(int64_t);
    uint32_t continuousMemSize = opFlagSize + attrsSize;
    AscendTensor<uint8_t, DIMS_1, uint32_t> continuousMem(mem, { continuousMemSize }, stream);
    std::vector<uint8_t> continuousValue(continuousMemSize, 0);
    uint8_t *data = continuousValue.data();
    int64_t *attrs = reinterpret_cast<int64_t *>(data + opFlagSize);

    int pageNum = static_cast<int>(utils::divUp(this->ntotal, static_cast<size_t>(this->pageSize)));

    attrs[aicpu::TOPK_FLAT_ATTR_ASC_IDX] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_K_IDX] = topK;
    attrs[aicpu::TOPK_FLAT_ATTR_BURST_LEN_IDX] = BURST_LEN;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_NUM_IDX] = blockNum;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_IDX] = pageIdx;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_NUM_IDX] = pageNum;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_SIZE_IDX] = this->pageSize;
    attrs[aicpu::TOPK_FLAT_ATTR_QUICK_HEAP] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_SIZE] = this->codeBlockSize;

    ret = aclrtMemcpy(continuousMem.data(), continuousMem.getSizeInBytes(), continuousValue.data(),
        continuousValue.size() * sizeof(uint8_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "aclrtMemcpy to device failed: %i\n", ret);

    uint16_t *opFlagMem = reinterpret_cast<uint16_t *>(continuousMem.data());
    AscendTensor<uint16_t, DIMS_3> opFlag(opFlagMem, { blockNum, FLAG_NUM, FLAG_SIZE });
    int64_t *attrMem = reinterpret_cast<int64_t *>(continuousMem.data() + opFlagSize);
    AscendTensor<int64_t, DIMS_1> attrsInput(attrMem, { aicpu::TOPK_FLAT_ATTR_IDX_COUNT });
    // 1. run the topk operator to wait for distance result and csompute topk
    runTopkCompute(distResult, minDistResult, opSize, opFlag, attrsInput, outDistanceOnDevice, outIndicesOnDevice,
        streamAicpu);

    // 2. run the disance operator to compute the distance
    // opSize Host to Device,reduce communication
    std::vector<uint32_t> opSizeHost(blockNum * CORE_NUM * SIZE_ALIGN);
    int opSizeHostOffset = CORE_NUM * SIZE_ALIGN;
    int opSizeHostIdx = 0;
    int offset = 0;
    for (int i = 0; i < blockNum; i++) {
        opSizeHost[opSizeHostIdx + IDX_ACTUAL_NUM] =
            std::min(static_cast<uint32_t>(computeNum - offset), static_cast<uint32_t>(this->codeBlockSize));
        opSizeHostIdx += opSizeHostOffset;
        offset += this->codeBlockSize;
    }
    ret = aclrtMemcpy(opSize.data(), opSize.getSizeInBytes(), opSizeHost.data(), opSizeHost.size() * sizeof(uint32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "aclrtMemcpy to device failed: %i\n", ret);

    opSizeHostIdx = 0;
    const int dim1 = utils::divUp(this->codeBlockSize, CUBE_ALIGN);
    const int dim2 = utils::divUp(this->dims, CUBE_ALIGN_INT8);
    for (int i = 0; i < blockNum; ++i) {
        AscendTensor<int8_t, DIMS_4> shaped(baseShaped[blockOffset + i]->data(),
            { dim1, dim2, CUBE_ALIGN, CUBE_ALIGN_INT8 });
        AscendTensor<int32_t, DIMS_1> norm(normBase[blockOffset + i]->data(), { codeBlockSize });
        auto dist = distResult[i].view();
        auto minDist = minDistResult[i].view();
        auto flag = opFlag[i].view();
        auto actualSize = opSize[i].view();
        uint32_t actualNum = opSizeHost[opSizeHostIdx + IDX_ACTUAL_NUM];
        auto mask = masks[i].view();

        std::vector<const AscendTensorBase *> input { &queries, &mask, &shaped, &norm, &actualSize };
        std::vector<const AscendTensorBase *> output { &dist, &minDist, &flag };

        runInt8L2DistCompute(batch, actualNum, this->shareAttrFilter, input, output, stream);
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

APP_ERROR TSInt8FlatL2::searchBatchWithShareMasks(int batch, const int8_t *features, int topK, float *distances,
    int64_t *labels, AscendTensor<uint8_t, DIMS_3> &masks)
{
    APP_LOG_INFO("TSInt8FlatL2 searchBatchWithShareMasks operation started.\n");
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();

    AscendTensor<float16_t, DIMS_2> outDistanceOnDevice(mem, { batch, topK }, stream);
    AscendTensor<int64_t, DIMS_2> outIndicesOnDevice(mem, { batch, topK }, stream);
    int pageNum = static_cast<int>(utils::divUp(static_cast<int>(this->ntotal), pageSize));
    int maskSize = utils::divUp(this->codeBlockSize, MASK_ALIGN);

    for (int pageIdx = 0; pageIdx < pageNum; ++pageIdx) {
        int pageOffset = pageIdx * pageSize;
        int blockOffset = pageOffset / this->codeBlockSize;
        int computeNum = std::min(static_cast<int>(this->ntotal) - pageOffset, pageSize);
        int blockNum = utils::divUp(computeNum, this->codeBlockSize);
        AscendTensor<uint8_t, DIMS_3> subMasks(masks[blockOffset].data(), { blockNum, 1, maskSize });
        auto ret =
            searchPagedWithMasks(pageIdx, batch, features, topK, subMasks, outDistanceOnDevice, outIndicesOnDevice);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "searchPagedWithMasks failed: %i.\n", ret);
    }
    postProcess(batch, topK, outDistanceOnDevice, outIndicesOnDevice, distances, labels);
    APP_LOG_INFO("TSInt8FlatL2 searchBatchWithShareMasks operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR TSInt8FlatL2::searchBatchWithNonshareMasks(int batch, const int8_t *features, int topK, float *distances,
    int64_t *labels, AscendTensor<int32_t, DIMS_2> &queryTimes, AscendTensor<uint8_t, DIMS_2> &tokenIds)
{
    APP_LOG_INFO("TSInt8FlatL2 searchBatchWithNonshareMasks operation started.\n");

    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();

    AscendTensor<float16_t, DIMS_2> outDistanceOnDevice(mem, { batch, topK }, stream);
    AscendTensor<int64_t, DIMS_2> outIndicesOnDevice(mem, { batch, topK }, stream);
    int pageNum = static_cast<int>(utils::divUp(static_cast<int>(this->ntotal), pageSize));
    int maskSize = utils::divUp(this->codeBlockSize, MASK_ALIGN);

    for (int pageIdx = 0; pageIdx < pageNum; ++pageIdx) {
        int pageOffset = pageIdx * pageSize;
        int blockOffset = pageOffset / this->codeBlockSize;
        int computeNum = std::min(static_cast<int>(this->ntotal) - pageOffset, pageSize);
        int blockNum = utils::divUp(computeNum, this->codeBlockSize);
        AscendTensor<uint8_t, DIMS_3> masks(mem, { blockNum, batch, maskSize }, stream);
        generateMask(batch, blockOffset, blockNum, queryTimes, tokenIds, masks);
        auto ret =
            searchPagedWithMasks(pageIdx, batch, features, topK, masks, outDistanceOnDevice, outIndicesOnDevice);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "searchPagedWithMasks failed: %i.\n", ret);
    }
    postProcess(batch, topK, outDistanceOnDevice, outIndicesOnDevice, distances, labels);
    APP_LOG_INFO("TSInt8FlatL2 searchBatchWithNonshareMasks operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR TSInt8FlatL2::searchBatchWithExtraNonshareMasks(int batch, const int8_t *features, int topK,
    float *distances, int64_t *labels, AscendTensor<int32_t, DIMS_2> &queryTimes,
    AscendTensor<uint8_t, DIMS_2> &tokenIds, const uint8_t *extraMask)
{
    APP_LOG_INFO("TSInt8FlatL2 searchBatchWithExtraNonshareMasks operation started.\n");
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();

    AscendTensor<float16_t, DIMS_2> outDistanceOnDevice(mem, { batch, topK }, stream);
    AscendTensor<int64_t, DIMS_2> outIndicesOnDevice(mem, { batch, topK }, stream);
    int pageNum = static_cast<int>(utils::divUp(static_cast<int>(this->ntotal), pageSize));
    int maskSize = utils::divUp(this->codeBlockSize, MASK_ALIGN);

    for (int pageIdx = 0; pageIdx < pageNum; ++pageIdx) {
        int pageOffset = pageIdx * pageSize;
        int blockOffset = pageOffset / this->codeBlockSize;
        int computeNum = std::min(static_cast<int>(this->ntotal) - pageOffset, pageSize);
        int blockNum = utils::divUp(computeNum, this->codeBlockSize);

        AscendTensor<uint8_t, DIMS_3> masks(mem, { blockNum, batch, maskSize }, stream);
        generateMaskWithExtra(batch, blockOffset, blockNum, queryTimes, tokenIds, extraMask, masks);
        auto ret = searchPagedWithMasks(pageIdx, batch, features, topK, masks, outDistanceOnDevice, outIndicesOnDevice);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "searchPagedWithMasks failed: %i.\n", ret);
    }

    postProcess(batch, topK, outDistanceOnDevice, outIndicesOnDevice, distances, labels);
    APP_LOG_INFO("TSInt8FlatL2 searchBatchWithExtraNonshareMasks operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR TSInt8FlatL2::searchInPureDev(uint32_t count, const int8_t *features,
    const faiss::ascend::AttrFilter *attrFilter, uint32_t topk, int64_t *labels, float *distances)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    uint32_t offset = 0;
    int32_t queryNum = static_cast<int32_t>(count);
    if (this->shareAttrFilter) {
        int totalBlock = static_cast<int>(utils::divUp(static_cast<int>(this->ntotal), this->codeBlockSize));
        AscendTensor<int32_t, DIMS_2> queryTimes(mem, { 1, OPS_DATA_TYPE_ALIGN }, stream);
        AscendTensor<uint8_t, DIMS_2> tokenIds(mem,
            { 1, utils::divUp(static_cast<int32_t>(tokenNum), OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES }, stream);
        AscendTensor<uint8_t, DIMS_3> masks(mem, { totalBlock, 1, utils::divUp(this->codeBlockSize, MASK_ALIGN) },
            stream);
        buildAttr(attrFilter, 1, queryTimes, tokenIds);
        generateMask(1, 0, totalBlock, queryTimes, tokenIds, masks);
        for (auto batch : this->searchBatchSizes) {
            while (queryNum >= batch) {
                auto ret = searchBatchWithShareMasks(batch,
                    features + offset * static_cast<uint32_t>(this->code_size),
                    static_cast<int>(topk), distances + offset * topk, labels + offset * topk, masks);
                APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                    "searchBatchWithShareMasks failed: %i.\n", ret);
                offset += static_cast<uint32_t>(batch);
                queryNum -= batch;
            }
        }
    } else {
        for (auto batch : this->searchBatchSizes) {
            while (queryNum >= batch) {
                AscendTensor<int32_t, DIMS_2> queryTimes(mem, { static_cast<int32_t>(batch), OPS_DATA_TYPE_ALIGN },
                    stream);
                AscendTensor<uint8_t, DIMS_2> tokenIds(mem, { static_cast<int32_t>(batch),
                    utils::divUp(static_cast<int32_t>(tokenNum), OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES }, stream);
                buildAttr(attrFilter + offset, batch, queryTimes, tokenIds);
                auto ret = searchBatchWithNonshareMasks(batch,
                    features + offset * static_cast<uint32_t>(this->code_size),
                    static_cast<int>(topk), distances + offset * topk, labels + offset * topk, queryTimes, tokenIds);
                APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                    "searchBatchWithNonshareMasks failed: %i.\n", ret);
                offset += static_cast<uint32_t>(batch);
                queryNum -= batch;
            }
        }
    }
    return APP_ERR_OK;
}

APP_ERROR TSInt8FlatL2::search(uint32_t count, const void *features, const faiss::ascend::AttrFilter *attrFilter,
    bool shareAttrFilter, uint32_t topk, int64_t *labels, float *distances, uint32_t *validNums,
    bool enableTimeFilter, const faiss::ascend::ExtraValFilter *)
{
    APP_LOG_INFO("TSInt8FlatL2 search operation started.\n");
    APPERR_RETURN_IF_NOT(aclrtSetDevice(this->deviceId) == ACL_ERROR_NONE, APP_ERR_ACL_BAD_ALLOC);
    AscendTensor<int8_t, DIMS_2> tensorDevQueries({ static_cast<int>(count), dims });
    auto ret = aclrtMemcpy(tensorDevQueries.data(), tensorDevQueries.getSizeInBytes(), features,
        count * dims * sizeof(int8_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "aclrtMemcpy error %d", ret);
    this->shareAttrFilter = shareAttrFilter;
    this->enableTimeFilter = enableTimeFilter;
    ret = searchInPureDev(count, tensorDevQueries.data(), attrFilter, topk, labels, distances);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "searchInPureDev failed: %i.\n", ret);
    getValidNum(count, topk, labels, validNums);

    APP_LOG_INFO("TSInt8FlatL2 search operation operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR TSInt8FlatL2::searchInPureDevWithExtraMask(uint32_t count, const int8_t *features,
    const faiss::ascend::AttrFilter *attrFilter, uint32_t topk, const uint8_t *extraMask, int64_t *labels,
    float *distances)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    uint32_t offset = 0;
    int32_t queryNum = static_cast<int32_t>(count);
    if (this->shareAttrFilter) {
        int blockNum = static_cast<int>(utils::divUp(static_cast<int>(this->ntotal), this->codeBlockSize));
        AscendTensor<int32_t, DIMS_2> queryTimes(mem, { 1, OPS_DATA_TYPE_ALIGN }, stream);
        AscendTensor<uint8_t, DIMS_2> tokenIds(mem,
            { 1, utils::divUp(static_cast<int32_t>(tokenNum), OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES }, stream);
        AscendTensor<uint8_t, DIMS_3> masks(mem, { blockNum, 1, utils::divUp(this->codeBlockSize, MASK_ALIGN) },
            stream);
        buildAttr(attrFilter, 1, queryTimes, tokenIds);
        generateMaskWithExtra(1, 0, blockNum, queryTimes, tokenIds, extraMask, masks);
        for (auto batch : this->searchBatchSizes) {
            while (queryNum >= batch) {
                auto ret = searchBatchWithShareMasks(batch,
                    features + offset * static_cast<uint32_t>(this->code_size),
                    static_cast<int>(topk), distances + offset * topk, labels + offset * topk, masks);
                APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                    "searchBatchWithShareMasks failed: %i.\n", ret);
                offset += static_cast<uint32_t>(batch);
                queryNum -= batch;
            }
        }
        return APP_ERR_OK;
    }

    for (auto batch : this->searchBatchSizes) {
        while (queryNum >= batch) {
            AscendTensor<int32_t, DIMS_2> queryTimes(mem,
                { static_cast<int32_t>(batch),
                OPS_DATA_TYPE_ALIGN },
                stream);
            AscendTensor<uint8_t, DIMS_2> tokenIds(mem,
                { static_cast<int32_t>(batch),
                utils::divUp(static_cast<int32_t>(tokenNum), OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES }, stream);
            buildAttr(attrFilter + offset, batch, queryTimes, tokenIds);
            auto ret = searchBatchWithExtraNonshareMasks(batch,
                features + offset * static_cast<uint32_t>(this->code_size),
                static_cast<int>(topk), distances + offset * topk, labels + offset * topk, queryTimes, tokenIds,
                extraMask + offset * extraMaskLen);
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                "searchBatchWithExtraNonshareMasks failed: %i.\n", ret);
            offset += static_cast<uint32_t>(batch);
            queryNum -= batch;
        }
    }
    return APP_ERR_OK;
}

APP_ERROR TSInt8FlatL2::searchWithExtraMask(uint32_t count, const void *features,
                                            const faiss::ascend::AttrFilter *attrFilter, bool shareAttrFilter,
                                            uint32_t topk, const uint8_t *extraMask, uint64_t extraMaskLen,
                                            bool extraMaskIsAtDevice, int64_t *labels, float *distances,
                                            uint32_t *validNums, bool enableTimeFilter, const float16_t *)
{
    APP_LOG_INFO("TSInt8FlatL2 searchWithExtraMask operation started.\n");
    APPERR_RETURN_IF_NOT(aclrtSetDevice(this->deviceId) == ACL_ERROR_NONE, APP_ERR_ACL_BAD_ALLOC);
    // no RPC local, need copy features to device firstly
    AscendTensor<int8_t, DIMS_2> tensorDevQueries({static_cast<int>(count), dims});
    auto ret = aclrtMemcpy(tensorDevQueries.data(), tensorDevQueries.getSizeInBytes(), features,
                           count * dims * sizeof(int8_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "aclrtMemcpy error %d", ret);
    this->shareAttrFilter = shareAttrFilter;
    this->enableTimeFilter = enableTimeFilter;
    this->extraMaskIsAtDevice = extraMaskIsAtDevice;
    this->extraMaskLen = extraMaskLen;
    ret = searchInPureDevWithExtraMask(count, tensorDevQueries.data(), attrFilter, topk, extraMask, labels,
        distances);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "searchInPureDevWithExtraMask failed: %i.\n",
        ret);
    getValidNum(count, topk, labels, validNums);
    APP_LOG_INFO("TSInt8FlatL2 searchWithExtraMask operation operation end.\n");
    return APP_ERR_OK;
}

void TSInt8FlatL2::getValidNum(uint64_t count, uint32_t topk, int64_t *labels, uint32_t *validNums) const
{
    APP_LOG_INFO("TSInt8FlatL2 getValidNum operation started.\n");

    for (uint64_t i = 0; i < count; i++) {
        *(validNums + i) = 0;
        for (uint32_t j = 0; j < topk; j++) {
            int64_t tmpLabel = labels[i * topk + j];
            if (tmpLabel != -1) {
                *(validNums + i) += 1;
            }
        }
    }
    APP_LOG_INFO("TSInt8FlatL2 getValidNum operation end.\n");
}

void TSInt8FlatL2::postProcess(uint64_t searchNum, int topK, AscendTensor<float16_t, DIMS_2> &outDistanceOnDevice,
    AscendTensor<int64_t, DIMS_2> &outIndicesOnDevice, float *distances, int64_t *labels)
{
    APP_LOG_INFO("TSInt8FlatL2 postProcess operation started.\n");
    std::vector<float16_t> outDistances(searchNum * static_cast<uint64_t>(topK));
    auto ret = aclrtMemcpy(outDistances.data(), outDistances.size() * sizeof(float16_t), outDistanceOnDevice.data(),
        outDistanceOnDevice.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to copy back to host, ret=%d", ret);
    ret = aclrtMemcpy(labels, outIndicesOnDevice.getSizeInBytes(), outIndicesOnDevice.data(),
        outIndicesOnDevice.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to copy back to host, ret=%d", ret);

    auto scaleFunctor = [](int inputDim) {
        // 0.01, 128, 4 is hyperparameter suitable for UB
        return 0.01 / std::min(inputDim / 64, std::max(inputDim / 128 + 1, 4));
    };
    float scale = scaleFunctor(this->dims);

    for (uint64_t i = 0; i < searchNum; ++i) {
        std::transform(outDistances.data() + i * static_cast<uint64_t>(topK),
            outDistances.data() + (i + 1) * static_cast<uint64_t>(topK), distances + i * static_cast<uint64_t>(topK),
            [&](const float16_t temp) -> float {
                return std::sqrt(static_cast<float>(faiss::ascend::fp16(temp)) / scale);
            });
        
        std::transform(labels + i * static_cast<uint64_t>(topK),
            labels + (i + 1) * static_cast<uint64_t>(topK), labels + i * static_cast<uint64_t>(topK),
            [&](const idx_t temp) -> int64_t {
                return static_cast<int64_t>(temp) == -1 ? -1 : static_cast<int64_t>(ids.at(temp));
            });
    }
    APP_LOG_INFO("TSInt8FlatL2 postProcess operation end.\n");
}

void TSInt8FlatL2::addWithIds(idx_t n, const int8_t *features, const idx_t *featureIds)
{
    APP_LOG_INFO("TSInt8FlatL2 addWithIds operation started.\n");
    ids.insert(ids.end(), featureIds, featureIds + n);
    size_t offset = 0;
    size_t addTotal = n;
    size_t singleAddMax =
        static_cast<size_t>(utils::divDown(UPPER_LIMIT_FOR_ADD, this->codeBlockSize) * this->codeBlockSize);
    while (addTotal > 0) {
        auto singleAdd = std::min(addTotal, singleAddMax);
        addWithIdsImpl(singleAdd, features + offset * static_cast<size_t>(this->code_size));
        offset += singleAdd;
        addTotal -= singleAdd;
    }
    APP_LOG_INFO("TSInt8FlatL2 addWithIds operation end.\n");
}

void TSInt8FlatL2::addWithIdsImpl(int n, const int8_t *x)
{
    APP_LOG_INFO("TSInt8FlatL2 addWithIdsImpl operation started.\n");

    AscendTensor<int8_t, DIMS_2> rawTensor(const_cast<int8_t *>(x), {n, dims});
    auto ret = addVectors(rawTensor);
    FAISS_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "add vector failed, error code:%d", ret);
    APP_LOG_INFO("TSInt8FlatL2 addWithIdsImpl operation end.\n");
}

void TSInt8FlatL2::queryVectorByIdx(int64_t idx, uint8_t *dis, int num) const
{
    int reshapeDim2 = utils::divUp(this->dims, CUBE_ALIGN_INT8);
    size_t total = static_cast<size_t>(idx);
    size_t offsetInBlock = total % static_cast<size_t>(this->codeBlockSize);
    size_t blockIdx = total / static_cast<size_t>(this->codeBlockSize);
    // we can make sure the size of offsetInblock is small
    int hoffset1 = static_cast<int>(offsetInBlock) / CUBE_ALIGN;
    int hoffset2 = static_cast<int>(offsetInBlock) % CUBE_ALIGN;
    int disOffset = 0;
    int srcOffset = hoffset1 * code_size * CUBE_ALIGN + hoffset2 * CUBE_ALIGN_INT8;
    for (int i = 0; i < reshapeDim2; ++i) {
        auto ret =
            aclrtMemcpy(dis + disOffset, (num - disOffset) * sizeof(int8_t), baseShaped[blockIdx]->data() + srcOffset,
                        CUBE_ALIGN_INT8 * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
        FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to copy back to host, ret=%d", ret);
        disOffset += CUBE_ALIGN_INT8;
        srcOffset += CUBE_ALIGN_INT8 * CUBE_ALIGN;
    }
}

void TSInt8FlatL2::runInt8L2DistCompute(int batch, uint32_t actualNum, bool shareMask,
    const std::vector<const AscendTensorBase *> &input, const std::vector<const AscendTensorBase *> &output,
    aclrtStream stream) const
{
    IndexTypeIdx indexType;
    if (actualNum == static_cast<uint32_t>(this->codeBlockSize)) {
        indexType = shareMask ? IndexTypeIdx::ITI_INT8_L2_FULL_SHARE_MASK : IndexTypeIdx::ITI_INT8_L2_FULL_MASK;
    } else {
        indexType = shareMask ? IndexTypeIdx::ITI_INT8_L2_SHARE_MASK : IndexTypeIdx::ITI_INT8_L2_MASK;
    }

    std::vector<int> keys({batch, dims, static_cast<int>(tokenNum)});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().runOp(indexType, opsKey, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
}

APP_ERROR TSInt8FlatL2::resetInt8L2DistCompute(int codeNum, bool shareMask) const
{
    std::vector<IndexTypeIdx> distCompOpsIdxs;
    if (shareMask) {
        distCompOpsIdxs = {IndexTypeIdx::ITI_INT8_L2_SHARE_MASK, IndexTypeIdx::ITI_INT8_L2_FULL_SHARE_MASK};
    } else {
        distCompOpsIdxs = {IndexTypeIdx::ITI_INT8_L2_MASK, IndexTypeIdx::ITI_INT8_L2_FULL_MASK};
    }
    std::vector<std::string> distCompOpsNames = {"DistanceInt8L2MinsWithMask", "DistanceInt8L2FullMinsWithMask"};

    for (size_t i = 0; i < distCompOpsIdxs.size(); i++) {
        std::string opTypeName = distCompOpsNames.at(i);
        IndexTypeIdx indexMaskType = distCompOpsIdxs.at(i);
        for (auto batch : searchBatchSizes) {
            std::vector<int64_t> queryShape({ batch, dims });
            std::vector<int64_t> maskShape({ shareMask ? 1 : batch, utils::divUp(codeNum, 8) }); // divUp to 8
            std::vector<int64_t> coarseCentroidsShape({
                utils::divUp(codeNum, CUBE_ALIGN), utils::divUp(dims, CUBE_ALIGN_INT8), CUBE_ALIGN, CUBE_ALIGN_INT8});
            std::vector<int64_t> preNormsShape({ codeNum });
            std::vector<int64_t> sizeShape({ CORE_NUM, SIZE_ALIGN });
            std::vector<int64_t> distResultShape({ batch, codeNum });
            std::vector<int64_t> minResultShape({ batch, this->burstsOfBlock });
            std::vector<int64_t> flagShape({ FLAG_NUM, FLAG_SIZE });

            std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
                { ACL_INT8, queryShape },
                { ACL_UINT8, maskShape },
                { ACL_INT8, coarseCentroidsShape },
                { ACL_INT32, preNormsShape },
                { ACL_UINT32, sizeShape }
            };
            std::vector<std::pair<aclDataType, std::vector<int64_t>>> output {
                { ACL_FLOAT16, distResultShape },
                { ACL_FLOAT16, minResultShape },
                { ACL_UINT16, flagShape }
            };
            std::vector<int> keys({batch, dims, static_cast<int>(tokenNum)});
            OpsMngKey opsKey(keys);
            auto ret = DistComputeOpsManager::getInstance().resetOp(opTypeName, indexMaskType, opsKey, input, output);
            APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "op init failed: %i", ret);
        }
    }
    return APP_ERR_OK;
}

} // namespace ascend