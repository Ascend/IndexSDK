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


#include "ascendhost/include/impl/TSBinaryFlat.h"
#include "AscendTensor.h"
#include "common/utils/CommonUtils.h"
#include "faiss/impl/FaissAssert.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"
#include "utils/OpLauncher.h"
namespace ascend {
TSBinaryFlat::TSBinaryFlat(int deviceId, uint32_t dim, uint32_t tokenNum, uint64_t resources, uint32_t customAttrLen,
    uint32_t customAttrBlockSize) : TSBase(tokenNum, customAttrLen, customAttrBlockSize),
    AscendIndexBinaryFlatImpl(dim, faiss::ascend::AscendIndexBinaryFlatConfig({ deviceId }, resources), false)
{
    Initialize();
    auto ret = TSBase::initialize(deviceId);
    FAISS_THROW_IF_NOT_FMT(APP_ERR_OK == ret, "failed to init TSBase, ERRCODE:%d", ret);
    FAISS_THROW_IF_NOT_MSG(AscendIndexBinaryFlatImpl::pResources->getResourceSize() == (size_t)resourceSize,
                           "Failed to set expected resource size.\n");
    resetMaskDistCompOp();
}

void TSBinaryFlat::resetMaskDistCompOp()
{
    using namespace faiss::ascend;
    APP_LOG_INFO("TSBinaryFlat resetDistMaskCompOp operation started.\n");
    std::string opTypeName = "DistanceFlatHammingWithMask";
    auto distCompMaskOpReset = [&](int batch, bool shareMask) {
        IndexTypeIdx indexMaskType = shareMask ? IndexTypeIdx::ITI_HAMMING_SHARE_MASK : IndexTypeIdx::ITI_HAMMING_MASK;
        std::vector<int64_t> queryShape { batch, code_size };
        std::vector<int64_t> coarseCentroidsShape { BLOCK_SIZE / zRegionHeight, d / CUBE_ALIGN, zRegionHeight,
            HAMMING_CUBE_ALIGN };
        std::vector<int64_t> sizeShape { ACTUAL_NUM_SIZE };
        std::vector<int64_t> maskShape { shareMask ? 1 : batch, utils::divUp(BLOCK_SIZE, BINARY_BYTE_SIZE) };
        std::vector<int64_t> resultShape { BLOCK_SIZE * batch };
        std::vector<int64_t> maxResultShape { BLOCK_SIZE * batch / burstLen * 2 }; // vmax and indice, so multiply by 2
        std::vector<int64_t> flagShape { FLAG_SIZE };

        std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
            { ACL_UINT8, queryShape },
            { ACL_UINT8, coarseCentroidsShape },
            { ACL_UINT32, sizeShape },
            { ACL_UINT8, maskShape },
        };
        std::vector<std::pair<aclDataType, std::vector<int64_t>>> output {
            { ACL_FLOAT16, resultShape },
            { ACL_FLOAT16, maxResultShape },
            { ACL_UINT16, flagShape }
        };
        std::vector<int> keys({batch, code_size, static_cast<int>(tokenNum)});
        OpsMngKey opsKey(keys);
        return DistComputeOpsManager::getInstance().resetOp(opTypeName, indexMaskType, opsKey, input, output);
    };

    for (auto batch : BATCH_SIZES) {
        FAISS_THROW_IF_NOT_MSG(!distCompMaskOpReset(batch, false), "no share op init failed");
        FAISS_THROW_IF_NOT_MSG(!distCompMaskOpReset(batch, true), "share op init failed");
    }

    APP_LOG_INFO("TSBinaryFlat resetDistMaskCompOp operation end.\n");
}

APP_ERROR TSBinaryFlat::addFeatureWithLabels(int64_t n, const void *features,
    const faiss::ascend::FeatureAttr *attrs, const int64_t *labels, const uint8_t *customAttr,
    const faiss::ascend::ExtraValAttr *extraVal)
{
    auto getAddModeFunctor = [this, &extraVal]() {
        this->isFirstUseExtraVal = (extraVal != nullptr);
    };
    std::call_once(firstAddOnceFlag, getAddModeFunctor);

    bool isUseExtraVal = (extraVal != nullptr);
    bool isSameAdd = (isUseExtraVal == isFirstUseExtraVal);

    APPERR_RETURN_IF_NOT_LOG(isSameAdd, APP_ERR_ILLEGAL_OPERATION, "AddFeature cannot be used with AddWithExtraVal");

    std::set<int64_t> uniqueLabels(labels, labels + n);
    APPERR_RETURN_IF_NOT_LOG(uniqueLabels.size() == (size_t)n, APP_ERR_INVALID_PARAM, "the labels is not unique");
    if (!enableSaveHostMemory) {
        for (int64_t i = 0; i < n; i++) {
            APPERR_RETURN_IF_NOT_FMT(label2Idx.find(*(labels + i)) == label2Idx.end(), APP_ERR_INVALID_PARAM,
                "the label[%ld] is already exists", *(labels + i));
        }
    }

    AddWithExtraValAttrs(n, attrs, customAttr, extraVal);
    add_with_ids(n, reinterpret_cast<const uint8_t *>(features), labels);
    if (!enableSaveHostMemory) {
        for (int64_t i = this->ntotal - n; i < this->ntotal; ++i) {
            label2Idx[ids[i]] = i;
        }
    }
    return APP_ERR_OK;
}

APP_ERROR TSBinaryFlat::deleteFeatureByToken(int64_t count, const uint32_t *tokens)
{
    APPERR_RETURN_IF_NOT_LOG(!enableSaveHostMemory, APP_ERR_ILLEGAL_OPERATION,
        "enableSaveHostMemory does not support deleteFeatureByToken");

    size_t delCount = 0;
    std::vector<int64_t> removeIds;
    std::unordered_set<uint32_t> uniqueTokens(tokens, tokens + count);
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
    // remove labels and set new idx of last label
    removeLabels(removeIds);
    delCount += removeIdsImpl(removeIds);
    APP_LOG_INFO("TSBinaryFlat::deleteFeatureByToken delete count:%d", delCount);
    return APP_ERR_OK;
}

APP_ERROR TSBinaryFlat::delFeatureWithLabels(int64_t n, const int64_t *labels)
{
    std::unordered_set<int64_t> removeIdSets;

    if (!enableSaveHostMemory) {
        for (int64_t i = 0; i < n; ++i) {
            auto it = label2Idx.find(*(labels + i));
            if (it != label2Idx.end()) {
                removeIdSets.insert(it->second);
            }
        }
    } else {
        std::unordered_set<int64_t> tmpLabel;
        for (int64_t i = 0; i < n; ++i) {
            tmpLabel.insert(*(labels + i));
        }

        for (size_t j = 0; j < ids.size(); j++) {
            auto it = tmpLabel.find(ids[j]);
            if (it != tmpLabel.end()) {
                removeIdSets.insert(j);
            }
        }
    }
    if (removeIdSets.empty()) {
        return APP_ERR_OK;
    }
    std::vector<int64_t> removeIds(removeIdSets.begin(), removeIdSets.end());
    std::sort(removeIds.begin(), removeIds.end(), std::greater<::int64_t>());
    // use TSBase function to delete attr
    deleteAttrByIds(removeIds);
    // remove labels and set new idx of last label
    removeLabels(removeIds);
    auto res = removeIdsImpl(removeIds);
    APP_LOG_INFO("TSBinaryFlat::delFeatureWithLabels delete count:%d", res);
    return APP_ERR_OK;
}

void TSBinaryFlat::removeLabels(const std::vector<int64_t> &removeIds)
{
    FAISS_THROW_IF_NOT_MSG(this->ntotal > 0, "no data need to delete");
    int lastIdx = this->ntotal - 1;
    if (enableSaveHostMemory) {
        for (auto pos : removeIds) {
            ids[pos] = ids[lastIdx];
            --lastIdx;
        }
    } else {
        for (auto pos : removeIds) {
            int64_t delLabel = ids[pos];
            ids[pos] = ids[lastIdx];
            label2Idx[ids[lastIdx]] = pos; // update new position label idx
            label2Idx.erase(delLabel);  // delete original pos label idx
            --lastIdx;
        }
    }
    ids.resize(lastIdx + 1);
}

int64_t TSBinaryFlat::getLabelsInIds(int64_t i, const int64_t *labels) const
{
    int64_t id = -1;
    for (size_t j = 0; j < ids.size(); j++) {
        if (*(labels + i) == ids[j]) {
            id = static_cast<int64_t>(j);
            break;
        }
    }
    return id;
}
APP_ERROR TSBinaryFlat::getFeatureByLabel(int64_t n, const int64_t *labels, void *features) const
{
    APP_LOG_INFO("TSBinaryFlat::getFeatureByLabel start");
    std::vector<int64_t> queryIds;
    if (enableSaveHostMemory) {
        for (int64_t i = 0; i < n; ++i) {
            int64_t tmpId = getLabelsInIds(i, labels);
            APPERR_RETURN_IF_NOT_FMT(tmpId != -1, APP_ERR_INVALID_PARAM,
                "the label[%ld] does not exists", *(labels + i));
            queryIds.emplace_back(tmpId);
        }
    } else {
        for (int64_t i = 0; i < n; ++i) {
            auto it = label2Idx.find(*(labels + i));
            APPERR_RETURN_IF_NOT_FMT(it != label2Idx.end(), APP_ERR_INVALID_PARAM,
                "the label[%ld] does not exists", *(labels + i));
            queryIds.emplace_back(it->second);
        }
    }
    for (size_t i = 0; i < queryIds.size(); ++i) {
        queryVectorByIdx(queryIds[i], reinterpret_cast<uint8_t *>(features) + i * code_size);
    }
    APP_LOG_INFO("TSBinaryFlat::getFeatureByLabel end");
    return APP_ERR_OK;
}

APP_ERROR TSBinaryFlat::getBaseByRange(uint32_t offset, uint32_t num, int64_t *labels, void *features,
    faiss::ascend::FeatureAttr *attributes, faiss::ascend::ExtraValAttr *extraVal)
{
    APP_LOG_INFO("TSBinaryFlat::getBaseByRange start");
    std::vector<uint8_t> baseVectors(static_cast<int64_t>(num) * code_size);
    getVectors(offset, num, baseVectors);
    auto ret = memcpy_s(features, num * this->code_size * sizeof(uint8_t), baseVectors.data(),
        baseVectors.size() * sizeof(uint8_t));
    APPERR_RETURN_IF_NOT_FMT(ret == 0, APP_ERR_INNER_ERROR, "Memcpy_s features failed(%d).", ret);
    ret = memcpy_s(labels, num * sizeof(int64_t), this->ids.data() + offset, num * sizeof(int64_t));
    APPERR_RETURN_IF_NOT_FMT(ret == 0, APP_ERR_INNER_ERROR, "Memcpy_s labels failed(%d).", ret);
    ret = memcpy_s(attributes, num * sizeof(faiss::ascend::FeatureAttr), this->featureAttrs.data() + offset,
        num * sizeof(faiss::ascend::FeatureAttr));
    APPERR_RETURN_IF_NOT_FMT(ret == 0, APP_ERR_INNER_ERROR, "Memcpy_s attributes failed(%d).", ret);

    if (extraVal != nullptr) {
        ret = memcpy_s(extraVal, num * sizeof(faiss::ascend::ExtraValAttr), this->extraValAttrs.data() + offset,
            num * sizeof(faiss::ascend::ExtraValAttr));
        APPERR_RETURN_IF_NOT_FMT(ret == 0, APP_ERR_INNER_ERROR, "Memcpy_s extraVal failed(%d).", ret);
    }
    APP_LOG_INFO("TSBinaryFlat::getBaseByRange end");
    return APP_ERR_OK;
}

void TSBinaryFlat::queryVectorByIdx(int64_t idx, uint8_t *dis) const
{
    int reshapeDim2 = d / faiss::ascend::CUBE_ALIGN;
    size_t total = static_cast<size_t>(idx);
    size_t offsetInBlock = total % faiss::ascend::BLOCK_SIZE;
    size_t blockIdx = total / faiss::ascend::BLOCK_SIZE;
    int hoffset1 = static_cast<int>(offsetInBlock) / zRegionHeight;
    int hoffset2 =
        static_cast<int>(offsetInBlock) % zRegionHeight; // we can make sure the size of offsetInblock is small
    int disOffset = 0;
    int srcOffset = hoffset1 * code_size * zRegionHeight + hoffset2 * faiss::ascend::HAMMING_CUBE_ALIGN;
    for (int i = 0; i < reshapeDim2; ++i) {
        auto ret =
            aclrtMemcpy(dis + disOffset, faiss::ascend::HAMMING_CUBE_ALIGN, baseShaped[blockIdx]->data() + srcOffset,
                        faiss::ascend::HAMMING_CUBE_ALIGN, ACL_MEMCPY_DEVICE_TO_HOST);
        FAISS_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Failed to copy to host");
        disOffset += faiss::ascend::HAMMING_CUBE_ALIGN;
        srcOffset += faiss::ascend::HAMMING_CUBE_ALIGN * zRegionHeight;
    }
}

void TSBinaryFlat::searchPagedWithMasks(int pageIdx, int batch, const uint8_t *x, int topK,
    AscendTensor<uint8_t, DIMS_3> &masks, AscendTensor<float16_t, DIMS_2> &outDistanceOnDevice,
    AscendTensor<int64_t, DIMS_2> &outIndicesOnDevice)
{
    using namespace faiss::ascend;
    APP_LOG_INFO("TSBinaryFlat searchPagedWithMasks operation started.\n");
    auto streamPtr = AscendIndexBinaryFlatImpl::pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = AscendIndexBinaryFlatImpl::pResources->getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    auto &mem = AscendIndexBinaryFlatImpl::pResources->getMemoryManager();

    int totalBlocks = utils::divUp(this->ntotal, BLOCK_SIZE);
    int pageBlocks = static_cast<int>((1.0 * resourceSize / BINARY_FLAT_DEFAULT_MEM) * PAGE_BLOCKS);
    int totalPages = utils::divUp(totalBlocks, pageBlocks);
    int lastPageBlocks = ((totalBlocks % pageBlocks == 0) ? pageBlocks : (totalBlocks % pageBlocks));
    int fullPageFeatures = BLOCK_SIZE * pageBlocks;
    int lastBlockFeatures = ((this->ntotal % BLOCK_SIZE) == 0 ? BLOCK_SIZE : (this->ntotal % BLOCK_SIZE));
    int blockNum = ((pageIdx == totalPages - 1) ? lastPageBlocks : pageBlocks);

    AscendTensor<uint8_t, DIMS_2> queries(mem, { batch, this->code_size }, stream);
    auto ret =
        aclrtMemcpy(queries.data(), queries.getSizeInBytes(), x, batch * this->code_size, ACL_MEMCPY_HOST_TO_DEVICE);
    FAISS_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Failed to copy to device");

    AscendTensor<uint8_t, DIMS_3> maskDeviceBlock(mem, { blockNum, batch, BLOCK_SIZE }, stream);
    AscendTensor<float16_t, DIMS_3> distResult(mem, { blockNum, batch, BLOCK_SIZE }, stream);
    AscendTensor<float16_t, DIMS_3> maxDistResult(mem, { blockNum, batch, BLOCK_SIZE / burstLen * 2 }, stream);

    AscendTensor<uint32_t, DIMS_3> opSize(mem, { blockNum, ACTUAL_NUM_SIZE, 1 }, stream);
    AscendTensor<uint16_t, DIMS_3> opFlag(mem, { blockNum, FLAG_SIZE, 1 }, stream);
    opFlag.zero();

    // attrs: [0]asc, [1]k, [2]burstLen, [3]block_num
    AscendTensor<int64_t, DIMS_1> attrsInput(mem, { aicpu::TOPK_FLAT_ATTR_IDX_COUNT }, stream);
    std::vector<int64_t> attrs(aicpu::TOPK_FLAT_ATTR_IDX_COUNT);
    attrs[aicpu::TOPK_FLAT_ATTR_ASC_IDX] = 0;
    attrs[aicpu::TOPK_FLAT_ATTR_K_IDX] = topK;
    attrs[aicpu::TOPK_FLAT_ATTR_BURST_LEN_IDX] = burstLen;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_NUM_IDX] = blockNum;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_IDX] = pageIdx;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_NUM_IDX] = totalPages;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_SIZE_IDX] = fullPageFeatures;
    attrs[aicpu::TOPK_FLAT_ATTR_QUICK_HEAP] = 1;
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
            { reshapeDim1, reshapeDim2, zRegionHeight, HAMMING_CUBE_ALIGN });
        auto mask = masks[i].view();
        auto dist = distResult[i].view();
        auto maxDist = maxDistResult[i].view();
        auto flag = opFlag[i].view();
        auto actualSize = opSize[i].view();
        std::vector<const AscendTensorBase *> input { &queries, &shaped, &actualSize, &mask };
        std::vector<const AscendTensorBase *> output { &dist, &maxDist, &flag };
        runDistMaskCompute(batch, this->shareAttrFilter, input, output, stream);
    }

    ret = synchronizeStream(stream);
    FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "synchronizeStream default stream: %i\n", ret);

    ret = synchronizeStream(streamAicpu);
    FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "synchronizeStream aicpu stream failed: %i\n", ret);

    APP_LOG_INFO("TSBinaryFlat searchPagedWithMasks operation end.\n");
}

void TSBinaryFlat::searchBatchWithNonshareMasks(int batch, const uint8_t *x, int topK, float *distances,
    int64_t *labels, AscendTensor<int32_t, DIMS_2> &queryTimes, AscendTensor<uint8_t, DIMS_2> &tokenIds,
    AscendTensor<int16_t, DIMS_2> &valFilter)
{
    using namespace faiss::ascend;
    APP_LOG_INFO("TSBinaryFlat searchBatchWithNonshareMasks operation started.\n");
    auto streamPtr = AscendIndexBinaryFlatImpl::pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = AscendIndexBinaryFlatImpl::pResources->getMemoryManager();

    int pageBlocks = static_cast<int>((1.0 * resourceSize / BINARY_FLAT_DEFAULT_MEM) * PAGE_BLOCKS);
    int pageSize = BLOCK_SIZE * pageBlocks;
    int pageNum = utils::divUp(this->ntotal, pageSize);
    int maskSize = utils::divUp(BLOCK_SIZE, MASK_ALIGN);

    AscendTensor<float16_t, DIMS_2> outDistanceOnDevice(mem, { batch, topK }, stream);
    AscendTensor<int64_t, DIMS_2> outIndicesOnDevice(mem, { batch, topK }, stream);

    for (int pageIdx = 0; pageIdx < pageNum; ++pageIdx) {
        int pageOffset = pageIdx * pageSize;
        int blockOffset = pageOffset / BLOCK_SIZE;
        int computeNum = std::min(static_cast<int>(this->ntotal - pageOffset), pageSize);
        int blockNum = utils::divUp(computeNum, BLOCK_SIZE);

        AscendTensor<uint8_t, DIMS_3> masks(mem, { blockNum, batch, maskSize },
            stream);
        if (this->enableValFilter) {
            generateMaskExtraVal(batch, blockOffset, blockNum, queryTimes, tokenIds, valFilter, masks);
        } else {
            generateMask(batch, blockOffset, blockNum, queryTimes, tokenIds, masks);
        }
        searchPagedWithMasks(pageIdx, batch, x, topK, masks, outDistanceOnDevice, outIndicesOnDevice);
    }

    postProcess(batch, topK, outDistanceOnDevice, outIndicesOnDevice, distances, labels);
    APP_LOG_INFO("TSBinaryFlat searchBatchWithNonshareMasks operation end.\n");
}

void TSBinaryFlat::generateMaskExtraVal(int batch, int blockOffset, int blockNum,
    AscendTensor<int32_t, DIMS_2> &queryTimes, AscendTensor<uint8_t, DIMS_2> &tokenIds,
    AscendTensor<int16_t, DIMS_2> &valFilter, AscendTensor<uint8_t, DIMS_3> &masks)
{
    auto streamPtr = AscendIndexBinaryFlatImpl::pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    for (int32_t blockId = 0; blockId < blockNum; blockId++) {
        AscendTensor<int32_t, DIMS_1> baseTimes(calcAttrStartAddress(attrTime, blockId + blockOffset),
            { static_cast<int32_t>(featureAttrBlockSize) });
        AscendTensor<int32_t, DIMS_1> baseTokenQs(calcAttrStartAddress(attrTokenQuotient, blockId + blockOffset),
            { static_cast<int32_t>(featureAttrBlockSize) });
        AscendTensor<uint8_t, DIMS_1> baseTokenRs(
            calcAttrStartAddress(attrTokenRemainder, blockId + blockOffset, OPS_DATA_TYPE_TIMES),
            { static_cast<int32_t>(featureAttrBlockSize * OPS_DATA_TYPE_TIMES) });
        AscendTensor<int16_t, DIMS_1> baseVals(calcAttrStartAddress(attrVal, blockId + blockOffset),
            { static_cast<int32_t>(featureAttrBlockSize) });

        auto subMask = masks[blockId].view();
        std::vector<const AscendTensorBase *> input {
            &queryTimes, &tokenIds, &baseTimes, &baseTokenQs, &baseTokenRs, &valFilter, &baseVals};
        std::vector<const AscendTensorBase *> output {&subMask};
        runBatchMaskValGenerateCompute(batch, input, output, stream);
    }
    auto ret = synchronizeStream(stream);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "synchronizeStream default stream: %i.\n", ret);
}

void TSBinaryFlat::searchBatchWithShareMasks(int batch, const uint8_t *x, int topK, float *distances, int64_t *labels,
    AscendTensor<uint8_t, DIMS_3> &masks)
{
    using namespace faiss::ascend;
    APP_LOG_INFO("TSBinaryFlat searchBatchWithShareMasks operation started.\n");
    auto streamPtr = AscendIndexBinaryFlatImpl::pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = AscendIndexBinaryFlatImpl::pResources->getMemoryManager();

    int pageBlocks = static_cast<int>((1.0 * resourceSize / BINARY_FLAT_DEFAULT_MEM) * PAGE_BLOCKS);
    int pageSize = BLOCK_SIZE * pageBlocks;
    int pageNum = utils::divUp(this->ntotal, pageSize);
    int maskSize = utils::divUp(BLOCK_SIZE, MASK_ALIGN);

    AscendTensor<int64_t, DIMS_2> outIndicesOnDevice(mem, { batch, topK }, stream);
    AscendTensor<float16_t, DIMS_2> outDistanceOnDevice(mem, { batch, topK }, stream);

    for (int pageIdx = 0; pageIdx < pageNum; ++pageIdx) {
        int pageOffset = pageIdx * pageSize;
        int blockOffset = pageOffset / BLOCK_SIZE;
        int computeNum = std::min(static_cast<int>(this->ntotal - pageOffset), pageSize);
        int blockNum = utils::divUp(computeNum, BLOCK_SIZE);
        AscendTensor<uint8_t, DIMS_3> subMasks(masks[blockOffset].data(), { blockNum, 1, maskSize });
        searchPagedWithMasks(pageIdx, batch, x, topK, subMasks, outDistanceOnDevice, outIndicesOnDevice);
    }

    postProcess(batch, topK, outDistanceOnDevice, outIndicesOnDevice, distances, labels);
    APP_LOG_INFO("TSBinaryFlat searchBatchWithShareMasks operation end.\n");
}

void TSBinaryFlat::searchBatchWithExtraNonshareMasks(int batch, const uint8_t *x, int topK, float *distances,
    int64_t *labels, AscendTensor<int32_t, DIMS_2> &queryTimes, AscendTensor<uint8_t, DIMS_2> &tokenIds,
    const uint8_t *extraMask)
{
    using namespace faiss::ascend;
    APP_LOG_INFO("TSBinaryFlat searchBatchWithExtraNonshareMasks operation started.\n");
    auto streamPtr = AscendIndexBinaryFlatImpl::pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = AscendIndexBinaryFlatImpl::pResources->getMemoryManager();

    AscendTensor<float16_t, DIMS_2> outDistanceOnDevice(mem, { batch, topK }, stream);
    AscendTensor<int64_t, DIMS_2> outIndicesOnDevice(mem, { batch, topK }, stream);

    int pageBlocks = static_cast<int>((1.0 * resourceSize / BINARY_FLAT_DEFAULT_MEM) * PAGE_BLOCKS);
    int pageSize = BLOCK_SIZE * pageBlocks;
    int pageNum = utils::divUp(this->ntotal, pageSize);
    int maskSize = utils::divUp(BLOCK_SIZE, MASK_ALIGN);

    for (int pageIdx = 0; pageIdx < pageNum; ++pageIdx) {
        int pageOffset = pageIdx * pageSize;
        int blockOffset = pageOffset / BLOCK_SIZE;
        int computeNum = std::min(static_cast<int>(this->ntotal - pageOffset), pageSize);
        int blockNum = utils::divUp(computeNum, BLOCK_SIZE);

        AscendTensor<uint8_t, DIMS_3> masks(mem, { blockNum, batch, maskSize }, stream);
        generateMaskWithExtra(batch, blockOffset, blockNum, queryTimes, tokenIds, extraMask, masks);
        searchPagedWithMasks(pageIdx, batch, x, topK, masks, outDistanceOnDevice, outIndicesOnDevice);
    }

    postProcess(batch, topK, outDistanceOnDevice, outIndicesOnDevice, distances, labels);
    APP_LOG_INFO("TSBinaryFlat searchBatchWithExtraNonshareMasks operation end.\n");
}

void TSBinaryFlat::runDistMaskCompute(int batch, bool shareMask, const std::vector<const AscendTensorBase *> &input,
    const std::vector<const AscendTensorBase *> &output, aclrtStream stream)
{
    APP_LOG_INFO("TSBinaryFlat runDistMaskCompute operation started.\n");
    IndexTypeIdx indexType = shareMask ? IndexTypeIdx::ITI_HAMMING_SHARE_MASK : IndexTypeIdx::ITI_HAMMING_MASK;
    std::vector<int> keys({batch, code_size, static_cast<int>(tokenNum)});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().runOp(indexType, opsKey, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
    APP_LOG_INFO("TSBinaryFlat runDistMaskCompute operation end.\n");
}

void TSBinaryFlat::postProcess(int64_t searchNum, int topK, AscendTensor<float16_t, DIMS_2> &outDistanceOnDevice,
    AscendTensor<int64_t, DIMS_2> &outIndicesOnDevice, float *distances, int64_t *labels)
{
    APP_LOG_INFO("TSBinaryFlat postProcess operation started.\n");
    std::vector<float16_t> outDistances(searchNum * topK);
    auto ret = aclrtMemcpy(outDistances.data(), outDistances.size() * sizeof(float16_t), outDistanceOnDevice.data(),
        outDistanceOnDevice.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    FAISS_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Failed to copy back to host");
    ret = aclrtMemcpy(labels, outIndicesOnDevice.getSizeInBytes(), outIndicesOnDevice.data(),
        outIndicesOnDevice.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    FAISS_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Failed to copy back to host");

    for (int64_t i = 0; i < searchNum; ++i) {
        std::transform(outDistances.data() + i * topK, outDistances.data() + (i + 1) * topK, distances + i * topK,
            [&](const float16_t temp) -> float {
                return (this->d - static_cast<float>(faiss::ascend::fp16(temp))) / 2.0; // 2.0 float to fp16
            });
        std::transform(labels + i * topK, labels + (i + 1) * topK, labels + i * topK,
            [&](const int64_t temp) -> int64_t { return temp == -1 ? -1 : ids.at(temp); });
    }

    APP_LOG_INFO("TSBinaryFlat postProcess operation end.\n");
}

void TSBinaryFlat::getValidNum(uint64_t count, uint32_t topk, int64_t *labels, uint32_t *validNums) const
{
    APP_LOG_INFO("TSBinaryFlat getValidNum operation started.\n");

    for (uint64_t i = 0; i < count; i++) {
        *(validNums + i) = 0;
        for (uint32_t j = 0; j < topk; j++) {
            int64_t tmpLabel = labels[i * topk + j];
            if (tmpLabel != -1) {
                *(validNums + i) += 1;
            }
        }
    }
    APP_LOG_INFO("TSBinaryFlat getValidNum operation end.\n");
}

APP_ERROR TSBinaryFlat::search(uint32_t count, const void *features, const faiss::ascend::AttrFilter *attrFilter,
                               bool shareAttrFilter, uint32_t topk, int64_t *labels, float *distances,
                               uint32_t *validNums, bool enableTimeFilter,
                               const faiss::ascend::ExtraValFilter *extraValFilter)
{
    using namespace faiss::ascend;
    APP_LOG_INFO("TSBinaryFlat search operation started.\n");

    auto streamPtr = AscendIndexBinaryFlatImpl::pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = AscendIndexBinaryFlatImpl::pResources->getMemoryManager();
    uint32_t queryNum = count;
    APPERR_RETURN_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE, APP_ERR_ACL_BAD_ALLOC);
    setFilter(shareAttrFilter, enableTimeFilter, extraValFilter);
    int64_t offset = 0;
    if (this->shareAttrFilter) {
        int totalBlock = utils::divUp(this->ntotal, BLOCK_SIZE);
        AscendTensor<int32_t, DIMS_2> queryTimes(mem, { 1, OPS_DATA_TYPE_ALIGN }, stream);
        AscendTensor<uint8_t, DIMS_2> tokenIds(mem,
            { 1, static_cast<int32_t>(utils::divUp(tokenNum, OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES) }, stream);
        AscendTensor<uint8_t, DIMS_3> masks(mem, { totalBlock, 1, utils::divUp(BLOCK_SIZE, MASK_ALIGN) }, stream);
        buildAttr(attrFilter, 1, queryTimes, tokenIds);
        generateMask(1, 0, totalBlock, queryTimes, tokenIds, masks);
        for (auto batch : BATCH_SIZES) {
            for (; count >= batch; count -= batch) {
                searchBatchWithShareMasks(batch, reinterpret_cast<const uint8_t *>(features) + offset * this->code_size,
                    static_cast<int>(topk), distances + offset * topk, labels + offset * topk, masks);
                offset += batch;
            }
        }
    } else {
        for (auto batch : BATCH_SIZES) {
            for (; count >= batch; count -= batch) {
                AscendTensor<int32_t, DIMS_2> queryTimes(mem, { static_cast<int32_t>(batch), OPS_DATA_TYPE_ALIGN },
                    stream);
                AscendTensor<uint8_t, DIMS_2> tokenIds(mem, { static_cast<int32_t>(batch),
                    static_cast<int32_t>(utils::divUp(tokenNum, OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES) }, stream);
                AscendTensor<int16_t, DIMS_2> ValFilter(mem, { static_cast<int32_t>(batch), EXTRA_VAL_ALIGN }, stream);
                if (this->enableValFilter) {
                    buildAttrWithExtraVal(attrFilter + offset, extraValFilter + offset,
                        batch, queryTimes, tokenIds, ValFilter);
                } else {
                    buildAttr(attrFilter + offset, batch, queryTimes, tokenIds);
                }
                searchBatchWithNonshareMasks(batch,
                    reinterpret_cast<const uint8_t *>(features) + offset * this->code_size, static_cast<int>(topk),
                    distances + offset * topk, labels + offset * topk, queryTimes, tokenIds, ValFilter);
                offset += batch;
            }
        }
    }

    getValidNum(queryNum, topk, labels, validNums);

    APP_LOG_INFO("TSBinaryFlat search operation operation end.\n");
    return APP_ERR_OK;
}

void TSBinaryFlat::buildAttrWithExtraVal(const faiss::ascend::AttrFilter *attrFilter,
    const faiss::ascend::ExtraValFilter *extraValFilter, int batch,
    AscendTensor<int32_t, DIMS_2> &queryTime, AscendTensor<uint8_t, DIMS_2> &tokenIds,
    AscendTensor<int16_t, DIMS_2> &valFilter)
{
    ASCEND_THROW_IF_NOT_MSG(attrFilter, "Invalid filter.\n");
    ASCEND_THROW_IF_NOT_MSG(extraValFilter, "Invalid valFilter.\n");

    int maxTokenValue = utils::divUp(static_cast<int32_t>(tokenNum), OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES;
    std::vector<int32_t> queryTimeVec(batch * OPS_DATA_TYPE_ALIGN, 0);
    std::vector<uint8_t> tokenIdsVec(batch * maxTokenValue, 0);
    std::vector<int16_t> valVec(batch * EXTRA_VAL_ALIGN, 0);

    for (int i = 0; i < batch; i++) {
        queryTimeVec[i * OPS_DATA_TYPE_ALIGN] = this->enableTimeFilter ? ((attrFilter + i)->timesStart * -1) : 0;
        queryTimeVec[i * OPS_DATA_TYPE_ALIGN + 1] =
            this->enableTimeFilter ? ((attrFilter + i)->timesEnd * -1) : (std::numeric_limits<int32_t>::max() * -1);
        for (uint32_t j = 0; j < (attrFilter + i)->tokenBitSetLen; j++) {
            tokenIdsVec[i * maxTokenValue + OPS_DATA_TYPE_TIMES * static_cast<int32_t>(j) + 1] = OPS_DATA_PADDING_VAL;
            tokenIdsVec[i * maxTokenValue + OPS_DATA_TYPE_TIMES * static_cast<int32_t>(j)] =
                *((attrFilter + i)->tokenBitSet + static_cast<int32_t>(j));
        }

        valVec[i * EXTRA_VAL_ALIGN] =
            this->enableValFilter ? (extraValFilter + i)->filterVal : std::numeric_limits<int16_t>::max();
        valVec[i * EXTRA_VAL_ALIGN + 1] = this->enableValFilter ? (extraValFilter + i)->matchVal : -1;
    }
    auto ret = aclrtMemcpy(queryTime.data(), queryTime.getSizeInBytes(), queryTimeVec.data(),
        batch * OPS_DATA_TYPE_ALIGN * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Copy querytime data to device failed.\n");

    ret = aclrtMemcpy(tokenIds.data(), tokenIds.getSizeInBytes(), tokenIdsVec.data(), batch * maxTokenValue,
        ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Copy token ids data to device failed.\n");

    ret = aclrtMemcpy(valFilter.data(), valFilter.getSizeInBytes(), valVec.data(),
        batch * EXTRA_VAL_ALIGN * sizeof(int16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Copy val data to device failed.\n");
}

void TSBinaryFlat::setSearchWithExtraMaskAttr(bool shareAttrFilter, bool extraMaskIsAtDevice,
                                              uint64_t extraMaskLen, bool enableTimeFilter)
{
    this->shareAttrFilter = shareAttrFilter;
    this->enableTimeFilter = enableTimeFilter;
    this->extraMaskIsAtDevice = extraMaskIsAtDevice;
    this->extraMaskLen = extraMaskLen;
}

APP_ERROR TSBinaryFlat::searchWithExtraMask(uint32_t count, const void *features,
                                            const faiss::ascend::AttrFilter *attrFilter, bool shareAttrFilter,
                                            uint32_t topk, const uint8_t *extraMask, uint64_t extraMaskLen,
                                            bool extraMaskIsAtDevice, int64_t *labels, float *distances,
                                            uint32_t *validNums, bool enableTimeFilter, const float16_t *)
{
    using namespace faiss::ascend;
    APP_LOG_INFO("TSBinaryFlat search with extra mask operation started.\n");
    auto streamPtr = AscendIndexBinaryFlatImpl::pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = AscendIndexBinaryFlatImpl::pResources->getMemoryManager();

    APPERR_RETURN_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE, APP_ERR_ACL_BAD_ALLOC);

    setSearchWithExtraMaskAttr(shareAttrFilter, extraMaskIsAtDevice, extraMaskLen, enableTimeFilter);
    uint32_t offset = 0;
    uint32_t queryNum = count;
    int blockNum = utils::divUp(this->ntotal, BLOCK_SIZE);

    if (shareAttrFilter) {
        int totalBlock = utils::divUp(this->ntotal, BLOCK_SIZE);
        AscendTensor<int32_t, DIMS_2> queryTimes(mem, { 1, OPS_DATA_TYPE_ALIGN }, stream);
        AscendTensor<uint8_t, DIMS_2> tokenIds(mem,
            { 1, static_cast<int32_t>(utils::divUp(tokenNum, OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES) }, stream);
        AscendTensor<uint8_t, DIMS_3> masks(mem, { totalBlock, 1, utils::divUp(BLOCK_SIZE, MASK_ALIGN) }, stream);
        buildAttr(attrFilter, 1, queryTimes, tokenIds);
        generateMaskWithExtra(1, 0, blockNum, queryTimes, tokenIds, extraMask, masks);
        for (auto batch : BATCH_SIZES) {
            while (count >= batch) {
                searchBatchWithShareMasks(batch, reinterpret_cast<const uint8_t *>(features) + offset * this->code_size,
                    static_cast<int>(topk), distances + offset * topk, labels + offset * topk, masks);
                offset += batch;
                count -= batch;
            }
        }
        getValidNum(queryNum, topk, labels, validNums);

        APP_LOG_INFO("TSBinaryFlat search shared with extra mask operation operation end.\n");
        return APP_ERR_OK;
    }

    for (auto batch : BATCH_SIZES) {
        while (count >= batch) {
            AscendTensor<int32_t, DIMS_2> queryTimes(mem, { static_cast<int32_t>(batch), OPS_DATA_TYPE_ALIGN }, stream);
            AscendTensor<uint8_t, DIMS_2> tokenIds(mem, { static_cast<int32_t>(batch),
                static_cast<int32_t>(utils::divUp(tokenNum, OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES) }, stream);
            buildAttr(attrFilter + offset, batch, queryTimes, tokenIds);
            searchBatchWithExtraNonshareMasks(batch,
                reinterpret_cast<const uint8_t *>(features) + offset * this->code_size, static_cast<int>(topk),
                distances + offset * topk, labels + offset * topk, queryTimes, tokenIds,
                extraMask + offset * extraMaskLen);
            offset += batch;
            count -= batch;
        }
    }

    getValidNum(queryNum, topk, labels, validNums);

    APP_LOG_INFO("TSBinaryFlat search with extra mask operation operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR TSBinaryFlat::getExtraValAttrsByLabel(int64_t n, const int64_t *labels,
    faiss::ascend::ExtraValAttr *extraVal) const
{
    APP_LOG_INFO("TSBinaryFlat getExtraValAttrsByLabel operation started.\n");
    if (enableSaveHostMemory) {
        for (int64_t i = 0; i < n; ++i) {
            int64_t tmpId = getLabelsInIds(i, labels);
            if (tmpId == -1) {
                extraVal[i].val = INT16_MIN;
            } else {
                extraVal[i] = extraValAttrs.at(tmpId);
            }
        }
    } else {
        for (int64_t i = 0; i < n; ++i) {
            auto it = label2Idx.find(*(labels + i));
            if (it != label2Idx.end()) {
                extraVal[i] = extraValAttrs.at(it->second);
            } else {
                extraVal[i].val = INT16_MIN;
            }
        }
    }
    APP_LOG_INFO("TSBinaryFlat getExtraValAttrsByLabel operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR TSBinaryFlat::getFeatureAttrsByLabel(int64_t n, const int64_t *labels,
    faiss::ascend::FeatureAttr *attrs) const
{
    APP_LOG_INFO("TSBase getFeatureAttrsByLabel operation started.\n");
    if (enableSaveHostMemory) {
        for (int64_t i = 0; i < n; ++i) {
            int64_t tmpId = getLabelsInIds(i, labels);
            if (tmpId == -1) {
                attrs[i].time = INT32_MIN;
                attrs[i].tokenId = UINT32_MAX;
            } else {
                attrs[i] = featureAttrs.at(tmpId);
            }
        }
    } else {
        for (int64_t i = 0; i < n; ++i) {
            auto it = label2Idx.find(*(labels + i));
            if (it != label2Idx.end()) {
                attrs[i] = featureAttrs.at(it->second);
            } else {
                attrs[i].time = INT32_MIN;
                attrs[i].tokenId = UINT32_MAX;
            }
        }
    }
    APP_LOG_INFO("TSBase getFeatureAttrsByLabel operation end.\n");
    return APP_ERR_OK;
}
} // namespace ascend