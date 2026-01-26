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

#include "ascendhost/include/impl/TSInt8FlatCos.h"
#include <bitset>
#include <iostream>
#include <set>
#include "common/utils/DataType.h"
#include "common/utils/SocUtils.h"
#include "faiss/impl/FaissAssert.h"
#include "fp16.h"

namespace ascend {

namespace {
const int L2NORM_COMPUTE_BATCH = 16384;
const int FP16_ALGIN = 16;
const int BURST_LEN = 64;
}

TSInt8FlatCos::TSInt8FlatCos(uint32_t deviceId, uint32_t dim, uint32_t tokenNum, uint64_t resources, bool useHmm,
    uint32_t customAttrLen, uint32_t customAttrBlockSize)
    : TSBase(tokenNum, customAttrLen, customAttrBlockSize), IndexInt8FlatCosAicpu(dim, resources)
{
    if (useHmm) {
        deviceMemMng.SetHeteroStrategy();
    }
    FAISS_THROW_IF_NOT_FMT(std::find(DIMS.begin(), DIMS.end(), dim) != DIMS.end(), "Unsupported dims %u", dim);
    code_size = static_cast<int>(dim);
    this->deviceId = deviceId;
    auto ret = TSBase::initialize(deviceId);
    FAISS_THROW_IF_NOT_FMT(APP_ERR_OK == ret, "failed to init TSBase, ERRCODE:%d", ret);
    this->searchBatchSizes = { 64, 48, 36, 32, 24, 18, 16, 12, 8, 6, 4, 2, 1 };
    ret = IndexInt8FlatCosAicpu::init();
    FAISS_THROW_IF_NOT_FMT(APP_ERR_OK == ret, "failed to init IndexInt8FlatCosAicpu, ERRCODE:%d", ret);
    if (faiss::ascend::SocUtils::GetInstance().IsAscend910B()) {
        ret = resetAscendcInt8CosDistCompute(codeBlockSize);
        FAISS_THROW_IF_NOT_FMT(APP_ERR_OK == ret, "failed to resetAscendcDistCompOp, ERRCODE:%d", ret);
    } else {
        ret = resetInt8CosDistCompute(codeBlockSize, false);
        FAISS_THROW_IF_NOT_FMT(APP_ERR_OK == ret, "failed to resetDistCompOp, ERRCODE:%d", ret);
        ret = resetInt8CosDistCompute(codeBlockSize, true);
        FAISS_THROW_IF_NOT_FMT(APP_ERR_OK == ret, "failed to resetDistCompOp, ERRCODE:%d", ret);
        ret = resetInt8CosExtraScore(codeBlockSize, false);
        FAISS_THROW_IF_NOT_FMT(APP_ERR_OK == ret, "failed to resetExtraScoreCompOp, ERRCODE:%d", ret);
        ret = resetInt8CosExtraScore(codeBlockSize, true);
        FAISS_THROW_IF_NOT_FMT(APP_ERR_OK == ret, "failed to resetExtraScoreCompOp, ERRCODE:%d", ret);
    }
}

APP_ERROR TSInt8FlatCos::addFeatureWithLabels(int64_t n, const void *features,
    const faiss::ascend::FeatureAttr *attrs, const int64_t *labels, const uint8_t *customAttr,
    const faiss::ascend::ExtraValAttr *extraVal)
{
    auto getAddModeFunctor = [this, &extraVal]() {
        this->isInt8FirstUseExtraVal = (extraVal != nullptr);
    };
    std::call_once(int8FirstAddOnceFlag, getAddModeFunctor);

    bool isUseExtraVal = (extraVal != nullptr);
    bool isSameAdd = (isUseExtraVal == isInt8FirstUseExtraVal);

    APPERR_RETURN_IF_NOT_LOG(isSameAdd, APP_ERR_ILLEGAL_OPERATION, "AddFeature cannot be used with AddWithExtraVal");
    std::set<int64_t> uniqueLabels(labels, labels + n);
    APPERR_RETURN_IF_NOT_LOG(uniqueLabels.size() == (size_t)n, APP_ERR_INVALID_PARAM, "the labels is not unique");

    if (!enableSaveHostMemory) {
        for (int64_t i = 0; i < n; i++) {
            APPERR_RETURN_IF_NOT_FMT(label2Idx.find(*(labels+i)) == label2Idx.end(), APP_ERR_INVALID_PARAM,
                "the label[%ld] is already exists", *(labels+i));
        }
    }

    AddWithExtraValAttrs(n, attrs, customAttr, extraVal);
    //  cast uint8 pointer to int8
    add_with_ids(n, reinterpret_cast<const int8_t *>(features), reinterpret_cast<const idx_t *>(labels));
    if (!enableSaveHostMemory) {
        for (int64_t i = static_cast<int64_t>(this->ntotal) - n; i < static_cast<int64_t>(this->ntotal); ++i) {
            label2Idx[ids[i]] = i;
        }
    }
    return APP_ERR_OK;
}

int64_t TSInt8FlatCos::getInt8LabelsInIds(int64_t offset, const int64_t *labels) const
{
    int64_t id = -1;
    for (size_t j = 0; j < ids.size(); j++) {
        if (*(labels + offset) == static_cast<int64_t>(ids[j])) {
            id = static_cast<int64_t>(j);
            break;
        }
    }
    return id;
}

APP_ERROR TSInt8FlatCos::getFeatureByLabel(int64_t n, const int64_t *labels, void *features) const
{
    if (deviceMemMng.GetStrategy() == DevMemStrategy::HETERO_MEM) {
        return getFeatureByLabelInOrder(n, labels, features);
    }

    APP_LOG_INFO("TSInt8FlatCos::getFeatureByLabel start");
    std::vector<int64_t> queryIds;
    if (enableSaveHostMemory) {
        for (int64_t i = 0; i < n; ++i) {
            int64_t tmpId = getInt8LabelsInIds(i, labels);
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
        queryVectorByIdx(queryIds[i], reinterpret_cast<uint8_t *>(features) + (i * code_size));
    }
    APP_LOG_INFO("TSInt8FlatCos::getFeatureByLabel end");
    return APP_ERR_OK;
};

APP_ERROR TSInt8FlatCos::getFeatureAttrsByLabel(int64_t n, const int64_t *labels,
    faiss::ascend::FeatureAttr *attrs) const
{
    APP_LOG_INFO("TSInt8FlatCos getFeatureAttrsByLabel operation started.\n");
    if (enableSaveHostMemory) {
        for (int64_t i = 0; i < n; ++i) {
            int64_t tmpId = getInt8LabelsInIds(i, labels);
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
    APP_LOG_INFO("TSInt8FlatCos getFeatureAttrsByLabel operation end.\n");
    return APP_ERR_OK;
}

void TSInt8FlatCos::removeIdsImpl(const std::vector<int64_t> &indices)
{
    APP_LOG_INFO("TSInt8FlatCos removeIdsImpl operation start. \n");
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
        int srcIdx = static_cast<int>(this->ntotal) - i - 1;
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
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to copy to device (error %d)", (int)ret);
    LaunchOpTwoInOneOut<uint64_t, DIMS_1, ACL_UINT64, int64_t, DIMS_1, ACL_INT64, uint64_t, DIMS_1, ACL_UINT64>(opName,
        stream, srcInput, attrsInput, dstInput);

    ret = synchronizeStream(stream);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to acl Synchronize Streame (error %d)", (int)ret);
    APP_LOG_INFO("TSInt8FlatCos removeIdsImpl operation finished. \n");
}

APP_ERROR TSInt8FlatCos::getBaseByRange(uint32_t offset, uint32_t num, int64_t *labels, void *features,
    faiss::ascend::FeatureAttr *attributes, faiss::ascend::ExtraValAttr *extraVal)
{
    APP_LOG_INFO("TSInt8FlatCos::getBaseByRange start");
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
    if (extraVal != nullptr) {
        ret = memcpy_s(extraVal, num * sizeof(faiss::ascend::ExtraValAttr), this->extraValAttrs.data() + offset,
            num * sizeof(faiss::ascend::ExtraValAttr));
        APPERR_RETURN_IF_NOT_FMT(ret == 0, APP_ERR_INNER_ERROR, "Memcpy_s extraVal failed(%d).", ret);
    }
    getBaseEnd();
    APP_LOG_INFO("TSInt8FlatCos::getBaseByRange end");
    return APP_ERR_OK;
}

APP_ERROR TSInt8FlatCos::getExtraValAttrsByLabel(int64_t n, const int64_t *labels,
    faiss::ascend::ExtraValAttr *extraVal) const
{
    APP_LOG_INFO("TSInt8FlatCos getExtraValAttrsByLabel operation started.\n");
    if (enableSaveHostMemory) {
        for (int64_t i = 0; i < n; ++i) {
            int64_t tmpId = getInt8LabelsInIds(i, labels);
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
    APP_LOG_INFO("TSInt8FlatCos getExtraValAttrsByLabel operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR TSInt8FlatCos::getFeatureByLabelInOrder(int64_t n, const int64_t *labels, void *features) const
{
    APP_LOG_INFO("TSInt8FlatCos::getFeatureByLabelInOrder start");
    std::map<size_t, uint8_t *> queryIdsFeatureMap;
    for (int64_t i = 0; i < n; i++) {
        auto it = label2Idx.find(labels[i]);
        APPERR_RETURN_IF_NOT_FMT(it != label2Idx.end(), APP_ERR_INVALID_PARAM,
            "the label[%ld] does not exists", labels[i]);
        queryIdsFeatureMap.insert({it->second, reinterpret_cast<uint8_t *>(features) + (i * code_size)});
    }

    size_t lastBlockIdx = 0;
    size_t blockIdx = 0;
    for (const auto &idsFeaturePair : queryIdsFeatureMap) {
        blockIdx = idsFeaturePair.first / static_cast<size_t>(codeBlockSize);
        queryVectorByIdx(idsFeaturePair.first, idsFeaturePair.second);
        if (blockIdx != lastBlockIdx) {
            baseShaped[lastBlockIdx]->pushData(false);
            lastBlockIdx = blockIdx;
        }
    }
    baseShaped[blockIdx]->pushData(false);

    APP_LOG_INFO("TSInt8FlatCos::getFeatureByLabelInOrder end");
    return APP_ERR_OK;
}

void TSInt8FlatCos::removeFeatureByIds(const std::vector<int64_t> &ids)
{
    removeNormBase(ids, static_cast<size_t>(codeBlockSize), static_cast<size_t>(ntotal), faiss::ascend::FLOAT16,
        normBase);

    // removeIdsImpl使用算子对内存搬运进行优化，但是前提是内存都在device侧；
    // 异构内存策略由于内存存在host侧，无法直接使用算子，因此继续沿用老方案。
    if (deviceMemMng.GetStrategy() != DevMemStrategy::HETERO_MEM) {
        return removeIdsImpl(ids);
    }

    idx_t lastValidId = this->ntotal - 1;
    for (const auto &id : ids) {
        moveShapedForward(lastValidId, id);
        lastValidId--;
    }
}

APP_ERROR TSInt8FlatCos::deleteFeatureByToken(int64_t count, const uint32_t *tokens)
{
    APPERR_RETURN_IF_NOT_LOG(!enableSaveHostMemory, APP_ERR_INNER_ERROR,
        "enableSaveHostMemory not support deletebytoken");
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
    APP_LOG_INFO("TSInt8FlatCos::deleteFeatureByToken start count:%u", removeIds.size());
    // use TSBase function to delete attr
    deleteAttrByIds(removeIds);
    //  remove labels and set new idx of last label
    removeLabels(removeIds);
    // remove baseShaped and base norm
    removeFeatureByIds(removeIds);
    // release the space  of baseShape and norm
    removeInvalidData(this->ntotal, removeIds.size());
    this->ntotal -= removeIds.size();
    APP_LOG_INFO("TSInt8FlatCos::deleteFeatureByToken delete count:%d", removeIds.size());
    return APP_ERR_OK;
}

APP_ERROR TSInt8FlatCos::delFeatureWithLabels(int64_t n, const int64_t *labels)
{
    APP_LOG_INFO("TSInt8FlatCos::delFeatureWithLabels start cound:%d", n);
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
    //  remove labels and set new idx of last label
    removeLabels(removeIds);
    // remove baseShaped and base norm
    removeFeatureByIds(removeIds);
    // release the space  of baseShape and norm
    removeInvalidData(this->ntotal, removeIds.size());
    this->ntotal -= removeIds.size();
    APP_LOG_INFO("TSInt8FlatCos::delFeatureWithLabels delete count:%d", removeIds.size());
    return APP_ERR_OK;
}

void TSInt8FlatCos::removeLabels(const std::vector<int64_t> &removeIds)
{
    FAISS_THROW_IF_NOT_MSG(this->ntotal > 0, "no data need to delete");
    int64_t lastIdx = static_cast<int64_t>(this->ntotal - 1);
    if (enableSaveHostMemory) {
        for (auto pos : removeIds) {
            ids[pos] = ids[lastIdx];
            --lastIdx;
        }
    } else {
        for (auto pos : removeIds) {
            uint64_t delLabel = ids[pos];
            ids[pos] = ids[lastIdx];
            label2Idx[ids[lastIdx]] = pos; // update new position label idx
            label2Idx.erase(delLabel);  // delete original pos label idx
            --lastIdx;
        }
    }
    ids.resize(lastIdx + 1);
}


APP_ERROR TSInt8FlatCos::AddFeatureByIndice(int64_t n, const void *features,
    const faiss::ascend::FeatureAttr *attrs, const int64_t *indices, const uint8_t *customAttr,
    const faiss::ascend::ExtraValAttr *extraValAttr)
{
    APP_LOG_INFO("TSInt8FlatCos::AddFeatureByIndice start");
    // 如有包含有新增的特征，新增的indice的位置
    int64_t replaceNum = -1;
    std::vector<std::pair<int64_t, int64_t>> segments;
    APPERR_RETURN_IF_NOT_LOG(CheckIndices(this->ntotal, n, indices, replaceNum, segments) == APP_ERR_OK,
        APP_ERR_INVALID_PARAM, "CheckIndices failed");
    APP_LOG_INFO("n:%ld, maxIndice:%ld, replaceNum:%ld, ntotal:%ld", n, indices[n - 1], replaceNum, this->ntotal);
    auto ret = AddFeatureAttrsByIndice(n, segments, indices, attrs, customAttr, extraValAttr);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "AddFeatureAttrsByIndice failed:%d\n", ret);
    ret = AddFeatureWithIndice(n, replaceNum, indices, segments, reinterpret_cast<const int8_t *>(features));
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "AddFeatureWithIndice failed:%d\n", ret);
    SetMaskValid(n, indices, this->ntotal);

    APP_LOG_INFO("TSInt8FlatCos::AddFeatureByIndice end");
    return APP_ERR_OK;
}

APP_ERROR TSInt8FlatCos::GetFeatureByIndice(int64_t count, const int64_t *indices, int64_t *labels,
    void *features, faiss::ascend::FeatureAttr *attributes, faiss::ascend::ExtraValAttr *extraVal) const
{
    APP_LOG_INFO("TSInt8FlatCos::GetFeatureByIndice start");
    // 获取特征向量
    if (features != nullptr) {
        for (int64_t i = 0; i < count; ++i) {
            queryVectorByIdx(indices[i], reinterpret_cast<uint8_t *>(features) + (i * code_size));
        }
    }
    // 获取label
    if (labels != nullptr) {
        for (int64_t i = 0; i < count; ++i) {
            labels[i] = ids[indices[i]];
        }
    }
    // 获取特征属性
    if (attributes != nullptr) {
        for (int64_t i = 0; i < count; ++i) {
            if (indices[i] < static_cast<int64_t>(featureAttrs.size())) {
                attributes[i] = featureAttrs.at(indices[i]);
            } else {
                attributes[i].time = INT32_MIN;
                attributes[i].tokenId = UINT32_MAX;
            }
        }
    }
    if (extraVal != nullptr) {
        for (int64_t i = 0; i < count; i++) {
            if (indices[i] < static_cast<int64_t>(extraValAttrs.size())) {
                extraVal[i] = extraValAttrs.at(indices[i]);
            } else {
                extraVal[i].val = INT16_MIN;
            }
        }
    }
    APP_LOG_INFO("TSInt8FlatCos::GetFeatureByIndice end");
    return APP_ERR_OK;
}


APP_ERROR TSInt8FlatCos::AddFeatureWithIndice(int64_t n, int64_t replaceNum, const int64_t *indices,
    const std::vector<std::pair<int64_t, int64_t>> &segments, const int8_t *features)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    APP_LOG_INFO("TSInt8FlatCos AddFeatureWithIndice operation started.\n");
    // 一次性最多申请singleAddMax条空间来更新或者添加
    int64_t singleAddMax = static_cast<int64_t>(utils::divDown(UPPER_LIMIT_FOR_ADD, this->codeBlockSize) * this->codeBlockSize);
    // 新增部分的空间申请
    int64_t addNum = n - replaceNum;
    if (addNum > 0) {
        resizeBaseShaped(addNum);
    }
    AscendTensor<float16_t, 1> precompData(mem, { static_cast<int>(utils::roundUp(n, CUBE_ALIGN)) }, stream);
    AscendTensor<int8_t, DIMS_2> rawTensor(const_cast<int8_t *>(features), {static_cast<int>(n), dims});
    auto ret = calL2norm(n, rawTensor, precompData);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, " calL2norm failed: %d", ret);
    // 每个连续段进行拷贝
    int64_t offset = 0;
    for (size_t segIdx = 0; segIdx < segments.size(); segIdx++) {
        int64_t start = segments[segIdx].first;  // 起始位置
        int64_t length = segments[segIdx].second; // 每段的长度
        int64_t startOffset = indices[start];
        while (length > 0) {
            auto singleAdd = std::min(length, singleAddMax);
            AscendTensor<int8_t, DIMS_2> rawTensor(const_cast<int8_t *>(features + offset * code_size),
                { static_cast<int>(singleAdd), this->code_size });
            copyAndSaveVectors(startOffset, rawTensor);
            // 拷贝norm
            ret = copyNormByIndice(startOffset, singleAdd, offset, precompData);
            offset += singleAdd;
            length -= singleAdd;
            startOffset += singleAdd;
        }
    }
    // 新增
    if (addNum > 0) {
        ids.insert(ids.end(), indices + replaceNum, indices + n);
        ntotal += static_cast<idx_t>(addNum);
    }
    APP_LOG_INFO("TSInt8FlatCos AddFeatureWithIndice operation end.\n");
    return APP_ERR_OK;
}

void TSInt8FlatCos::calOpSize(std::vector<uint32_t> &opSizeHost, int computeNum, int blockNum)
{
    int opSizeHostOffset = CORE_NUM * SIZE_ALIGN;
    int opSizeHostIdx = 0;
    int offset = 0;
    for (int i = 0; i < blockNum; i++) {
        opSizeHost[opSizeHostIdx + IDX_ACTUAL_NUM] =
            std::min(static_cast<uint32_t>(computeNum - offset), static_cast<uint32_t>(this->codeBlockSize));
        opSizeHost[opSizeHostIdx + IDX_ACTUAL_NUM + 1] = ntotalPad;
        opSizeHostIdx += opSizeHostOffset;
        offset += this->codeBlockSize;
    }
}

APP_ERROR TSInt8FlatCos::searchPagedWithMasks(int pageIdx, int batch, const int8_t *features, int topK,
    AscendTensor<uint8_t, DIMS_3> &masks, AscendTensor<float16_t, DIMS_2> &outDistanceOnDevice,
    AscendTensor<int64_t, DIMS_2> &outIndicesOnDevice, const float16_t *extraScore)
{
    const int FLAG_NUM = faiss::ascend::SocUtils::GetInstance().GetCoreNum();
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    int pageOffset = pageIdx * this->pageSize;
    int blockOffset = pageIdx * this->pageSize / codeBlockSize;
    int computeNum = std::min(this->ntotal - pageOffset, static_cast<idx_t>(this->pageSize));
    int blockNum = utils::divUp(computeNum, this->codeBlockSize);
    AscendTensor<int8_t, DIMS_2> queries(const_cast<int8_t *>(features), { batch, dims });

    AscendTensor<float16_t, DIMS_1> queriesNorm(mem, { utils::roundUp(batch, CUBE_ALIGN) }, stream);
    AscendTensor<uint32_t, DIMS_2> actualNum(mem, { utils::divUp(batch, L2NORM_COMPUTE_BATCH), SIZE_ALIGN }, stream);
    int8L2Norm->dispatchL2NormTask(queries, queriesNorm, actualNum, stream);

    auto ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream default stream: %i\n",
        ret);
    AscendTensor<float16_t, DIMS_3, size_t> distResult(mem,
        { static_cast<size_t>(blockNum), static_cast<size_t>(batch), static_cast<size_t>(codeBlockSize) }, stream);
    AscendTensor<float16_t, DIMS_3, size_t> minDistResult(mem,
        { static_cast<size_t>(blockNum), static_cast<size_t>(batch), static_cast<size_t>(this->burstsOfBlock) },
        stream);
    AscendTensor<uint32_t, DIMS_3> opSize(mem, { blockNum, CORE_NUM, SIZE_ALIGN }, stream);
    AscendTensor<uint16_t, DIMS_3> opFlag(mem, { blockNum, FLAG_NUM, FLAG_SIZE }, stream);
    opFlag.zero();

    int pageNum = static_cast<int>(utils::divUp(this->ntotal, static_cast<size_t>(this->pageSize)));
    AscendTensor<int64_t, DIMS_1> attrsInput(mem, { aicpu::TOPK_FLAT_ATTR_IDX_COUNT }, stream);
    std::vector<int64_t> attrs(aicpu::TOPK_FLAT_ATTR_IDX_COUNT);
    attrs[aicpu::TOPK_FLAT_ATTR_ASC_IDX] = 0;
    attrs[aicpu::TOPK_FLAT_ATTR_K_IDX] = topK;
    attrs[aicpu::TOPK_FLAT_ATTR_BURST_LEN_IDX] = BURST_LEN;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_NUM_IDX] = blockNum;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_IDX] = pageIdx;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_NUM_IDX] = pageNum;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_SIZE_IDX] = this->pageSize;
    attrs[aicpu::TOPK_FLAT_ATTR_QUICK_HEAP] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_SIZE] = this->codeBlockSize;

    ret = aclrtMemcpy(attrsInput.data(), attrsInput.getSizeInBytes(), attrs.data(), attrs.size() * sizeof(int64_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "aclrtMemcpy to device failed: %i\n", ret);
    // 1. run the topk operator to wait for distance result and csompute topk
    runTopkCompute(distResult, minDistResult, opSize, opFlag, attrsInput, outDistanceOnDevice, outIndicesOnDevice,
        streamAicpu);

    // 2. run the disance operator to compute the distance
    // opSize Host to Device,reduce communication
    std::vector<uint32_t> opSizeHost(blockNum * CORE_NUM * SIZE_ALIGN);
    calOpSize(opSizeHost, computeNum, blockNum);
    ret = aclrtMemcpy(opSize.data(), opSize.getSizeInBytes(), opSizeHost.data(), opSizeHost.size() * sizeof(uint32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "aclrtMemcpy to device failed: %i\n", ret);
    const int dim1 = utils::divUp(this->codeBlockSize, CUBE_ALIGN);
    const int dim2 = utils::divUp(this->dims, CUBE_ALIGN_INT8);
    for (int i = 0; i < blockNum; ++i) {
        AscendTensor<int8_t, DIMS_4> shaped(baseShaped[blockOffset + i]->data(),
            { dim1, dim2, CUBE_ALIGN, CUBE_ALIGN_INT8 });
        AscendTensor<float16_t, DIMS_1> codesNorm(normBase[blockOffset + i]->data(), { codeBlockSize });
        auto dist = distResult[i].view();
        auto minDist = minDistResult[i].view();
        auto flag = opFlag[i].view();
        auto actualSize = opSize[i].view();
        auto mask = masks[i].view();

        std::vector<const AscendTensorBase *> output { &dist, &minDist, &flag };
        
        if (extraScore != nullptr) {
            AscendTensor<float16_t, DIMS_3, size_t> extraScoreDev(const_cast<float16_t*>(extraScore),
            { static_cast<size_t>(blockNum), static_cast<size_t>(batch), static_cast<size_t>(codeBlockSize) });
            std::vector<const AscendTensorBase *> input { &queries, &mask, &shaped, &queriesNorm,
                                                          &codesNorm, &actualSize, &extraScoreDev};
            runInt8CosExtraScore(batch, this->shareAttrFilter, input, output, stream);
        } else {
            std::vector<const AscendTensorBase *> input { &queries, &mask,
                                                          &shaped, &queriesNorm, &codesNorm, &actualSize };
            if (faiss::ascend::SocUtils::GetInstance().IsAscend910B()) {
                runAscendcInt8CosDistCompute(batch, this->shareAttrFilter, input, output, stream);
            } else {
                runInt8CosDistCompute(batch, this->shareAttrFilter, input, output, stream);
            }
        }
    }
    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "synchronizeStream default stream: %i\n", ret);
    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream aicpu stream failed: %i\n", ret);
    return APP_ERR_OK;
}


APP_ERROR TSInt8FlatCos::searchBatchWithShareMasks(int batch, const int8_t *features, int topK, float *distances,
    int64_t *labels, AscendTensor<uint8_t, DIMS_3> &masks, const float16_t *extraScore)
{
    APP_LOG_INFO("TSInt8FlatCos searchBatchWithShareMasks operation started.\n");
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();

    AscendTensor<float16_t, DIMS_2> outDistanceOnDevice(mem, { batch, topK }, stream);
    AscendTensor<int64_t, DIMS_2> outIndicesOnDevice(mem, { batch, topK }, stream);
    int pageNum = static_cast<int>(utils::divUp(static_cast<int64_t>(this->ntotal), pageSize));
    int maskSize = utils::divUp(this->codeBlockSize, MASK_ALIGN);

    for (int pageIdx = 0; pageIdx < pageNum; ++pageIdx) {
        int pageOffset = pageIdx * pageSize;
        int blockOffset = pageOffset / this->codeBlockSize;
        int computeNum = std::min(static_cast<int>(this->ntotal - pageOffset), pageSize);
        int blockNum = utils::divUp(computeNum, this->codeBlockSize);
        AscendTensor<uint8_t, DIMS_3> subMasks(masks[blockOffset].data(), { blockNum, 1, maskSize });
        auto ret = searchPagedWithMasks(pageIdx, batch, features, topK, subMasks,
                                        outDistanceOnDevice, outIndicesOnDevice, extraScore);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "searchPagedWithMasks failed: %i.\n", ret);
    }
    postProcess(batch, topK, outDistanceOnDevice, outIndicesOnDevice, distances, labels);
    APP_LOG_INFO("TSInt8FlatCos searchBatchWithShareMasks operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR TSInt8FlatCos::searchBatchWithNonshareMasks(int batch, const int8_t *features, int topK, float *distances,
    int64_t *labels, AscendTensor<int32_t, DIMS_2> &queryTimes, AscendTensor<uint8_t, DIMS_2> &tokenIds,
    AscendTensor<int16_t, DIMS_2> &valFilter, const float16_t *extraScore)
{
    APP_LOG_INFO("TSInt8FlatCos searchBatchWithNonshareMasks operation started.\n");

    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();

    AscendTensor<float16_t, DIMS_2> outDistanceOnDevice(mem, { batch, topK }, stream);
    AscendTensor<int64_t, DIMS_2> outIndicesOnDevice(mem, { batch, topK }, stream);

    int pageNum = static_cast<int>(utils::divUp(static_cast<int64_t>(this->ntotal), pageSize));
    int maskSize = utils::divUp(this->codeBlockSize, MASK_ALIGN);
    for (int pageIdx = 0; pageIdx < pageNum; ++pageIdx) {
        int pageOffset = pageIdx * pageSize;
        int blockOffset = pageOffset / this->codeBlockSize;
        int computeNum = std::min(static_cast<int>(this->ntotal - pageOffset), pageSize);
        int blockNum = utils::divUp(computeNum, this->codeBlockSize);
        AscendTensor<uint8_t, DIMS_3> masks(mem, { blockNum, batch, maskSize }, stream);
        if (this->enableValFilter) {
            generateMaskExtraVal(batch, blockOffset, blockNum, queryTimes, tokenIds, valFilter, masks);
        } else {
            generateMask(batch, blockOffset, blockNum, queryTimes, tokenIds, masks);
        }
        auto ret = searchPagedWithMasks(pageIdx, batch, features, topK, masks,
                                        outDistanceOnDevice, outIndicesOnDevice, extraScore);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "searchPagedWithMasks failed: %i.\n", ret);
    }
    postProcess(batch, topK, outDistanceOnDevice, outIndicesOnDevice, distances, labels);
    APP_LOG_INFO("TSInt8FlatCos searchBatchWithNonshareMasks operation end.\n");
    return APP_ERR_OK;
}

void TSInt8FlatCos::generateMaskExtraVal(int batch, int blockOffset, int blockNum,
    AscendTensor<int32_t, DIMS_2> &queryTimes, AscendTensor<uint8_t, DIMS_2> &tokenIds,
    AscendTensor<int16_t, DIMS_2> &valFilter, AscendTensor<uint8_t, DIMS_3> &masks)
{
    auto streamPtr = resources.getDefaultStream();
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

APP_ERROR TSInt8FlatCos::searchBatchWithExtraNonshareMasks(int batch, const int8_t *features, int topK,
    float *distances, int64_t *labels, AscendTensor<int32_t, DIMS_2> &queryTimes,
    AscendTensor<uint8_t, DIMS_2> &tokenIds, const uint8_t *extraMask, const float16_t *extraScore)
{
    APP_LOG_INFO("TSInt8FlatCos searchBatchWithExtraNonshareMasks operation started.\n");
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();

    AscendTensor<int64_t, DIMS_2> outIndicesOnDevice(mem, { batch, topK }, stream);
    AscendTensor<float16_t, DIMS_2> outDistanceOnDevice(mem, { batch, topK }, stream);
    int pageNum = static_cast<int>(utils::divUp(static_cast<int64_t>(this->ntotal), pageSize));
    int maskSize = utils::divUp(this->codeBlockSize, MASK_ALIGN);

    for (int pageIdx = 0; pageIdx < pageNum; ++pageIdx) {
        int pageOffset = pageIdx * pageSize;
        int blockOffset = pageOffset / this->codeBlockSize;
        int computeNum = std::min(static_cast<int>(this->ntotal - pageOffset), pageSize);
        int blockNum = utils::divUp(computeNum, this->codeBlockSize);

        AscendTensor<uint8_t, DIMS_3> masks(mem, { blockNum, batch, maskSize }, stream);
        AscendTensor<uint8_t, DIMS_3> baseMaskDev;
        if (useBaseMask) {
            baseMaskDev = AscendTensor<uint8_t, DIMS_3>(mem, { blockNum, 1, maskSize}, stream);
            auto ret = aclrtMemcpy(baseMaskDev.data(), baseMaskDev.getSizeInBytes(),
                baseMask.data() + pageOffset / MASK_ALIGN, utils::divUp(computeNum, MASK_ALIGN),
                ACL_MEMCPY_HOST_TO_DEVICE);
            APPERR_RETURN_IF_NOT_FMT(ret == 0, APP_ERR_INNER_ERROR, "copy baseMask failed(%d).", ret);
        }
        generateMaskWithExtra(batch, blockOffset, blockNum, queryTimes, tokenIds, extraMask, masks, baseMaskDev);
        auto ret = searchPagedWithMasks(pageIdx, batch, features, topK, masks,
                                        outDistanceOnDevice, outIndicesOnDevice, extraScore);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "searchPagedWithMasks failed: %i.\n", ret);
    }

    postProcess(batch, topK, outDistanceOnDevice, outIndicesOnDevice, distances, labels);
    APP_LOG_INFO("TSInt8FlatCos searchBatchWithExtraNonshareMasks operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR TSInt8FlatCos::searchInPureDev(uint32_t count, const int8_t *features,
    const faiss::ascend::AttrFilter *attrFilter, uint32_t topk, int64_t *labels, float *distances,
    const faiss::ascend::ExtraValFilter *extraValFilter)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    uint64_t offset = 0;
    int32_t queryNum = static_cast<int32_t>(count);
    if (this->shareAttrFilter) {
        int totalBlock = static_cast<int>(utils::divUp(static_cast<int64_t>(this->ntotal), this->codeBlockSize));
        AscendTensor<int32_t, DIMS_2> queryTimes(mem, { 1, OPS_DATA_TYPE_ALIGN }, stream);
        AscendTensor<uint8_t, DIMS_2> tokenIds(mem,
            { 1, static_cast<int32_t>(utils::divUp(tokenNum, OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES) }, stream);
        AscendTensor<uint8_t, DIMS_3> masks(mem, { totalBlock, 1, utils::divUp(this->codeBlockSize, MASK_ALIGN) },
            stream);
        buildAttr(attrFilter, 1, queryTimes, tokenIds);
        generateMask(1, 0, totalBlock, queryTimes, tokenIds, masks);
        for (auto batch : this->searchBatchSizes) {
            for (; queryNum >= batch; queryNum -= batch) {
                auto ret = searchBatchWithShareMasks(batch, features + offset * this->code_size, static_cast<int>(topk),
                    distances + offset * topk, labels + offset * topk, masks);
                APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                    "searchBatchWithShareMasks failed: %i.\n", ret);
                offset += static_cast<uint64_t>(batch);
            }
        }
        return APP_ERR_OK;
    }
    for (auto batch : this->searchBatchSizes) {
        for (; queryNum >= batch; queryNum -= batch) {
            AscendTensor<int32_t, DIMS_2> queryTimes(mem, { static_cast<int32_t>(batch), OPS_DATA_TYPE_ALIGN }, stream);
            AscendTensor<uint8_t, DIMS_2> tokenIds(mem, { static_cast<int32_t>(batch),
                static_cast<int32_t>(utils::divUp(tokenNum, OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES) }, stream);
            AscendTensor<int16_t, DIMS_2> ValFilter(mem, { static_cast<int32_t>(batch), EXTRA_VAL_ALIGN }, stream);
            if (this->enableValFilter) {
                buildAttrWithExtraVal(attrFilter + offset, extraValFilter + offset,
                    batch, queryTimes, tokenIds, ValFilter);
            } else {
                buildAttr(attrFilter + offset, batch, queryTimes, tokenIds);
            }
            auto ret = searchBatchWithNonshareMasks(batch, features + offset * this->code_size,
                static_cast<int>(topk), distances + offset * topk, labels + offset * topk,
                queryTimes, tokenIds, ValFilter);
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                "searchBatchWithNonshareMasks failed: %i.\n", ret);
            offset += static_cast<uint64_t>(batch);
        }
    }
    return APP_ERR_OK;
}

void TSInt8FlatCos::buildAttrWithExtraVal(const faiss::ascend::AttrFilter *attrFilter,
    const faiss::ascend::ExtraValFilter *extraValFilter, int batch,
    AscendTensor<int32_t, DIMS_2> &queryTime, AscendTensor<uint8_t, DIMS_2> &tokenIds,
    AscendTensor<int16_t, DIMS_2> &valFilter)
{
    APP_LOG_INFO("TSInt8FlatCos buildAttrWithExtraVal operation started.\n");
    buildAttr(attrFilter, batch, queryTime, tokenIds);
    ASCEND_THROW_IF_NOT_MSG(extraValFilter, "Invalid valFilter.\n");

    std::vector<int16_t> valVec(batch * EXTRA_VAL_ALIGN, 0);
    for (int i = 0; i < batch; i++) {
        valVec[i * EXTRA_VAL_ALIGN] =
            this->enableValFilter ? (extraValFilter + i)->filterVal : std::numeric_limits<int16_t>::max();
        valVec[i * EXTRA_VAL_ALIGN + 1] = this->enableValFilter ? (extraValFilter + i)->matchVal : -1;
    }

    auto ret = aclrtMemcpy(valFilter.data(), valFilter.getSizeInBytes(), valVec.data(),
        batch * EXTRA_VAL_ALIGN * sizeof(int16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Copy val data to device failed.\n");
    APP_LOG_INFO("TSInt8FlatCos buildAttrWithExtraVal operation end.\n");
}

APP_ERROR TSInt8FlatCos::search(uint32_t count, const void *features, const faiss::ascend::AttrFilter *attrFilter,
    bool shareAttrFilter, uint32_t topk, int64_t *labels, float *distances, uint32_t *validNums,
    bool enableTimeFilter, const faiss::ascend::ExtraValFilter *extraValFilter)
{
    APP_LOG_INFO("TSInt8FlatCos search operation started.\n");
    APPERR_RETURN_IF_NOT(aclrtSetDevice(this->deviceId) == ACL_ERROR_NONE, APP_ERR_ACL_BAD_ALLOC);
    AscendTensor<int8_t, DIMS_2> tensorDevQueries({ static_cast<int>(count), dims });
    auto ret = aclrtMemcpy(tensorDevQueries.data(), tensorDevQueries.getSizeInBytes(), features,
        count * dims * sizeof(int8_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "aclrtMemcpy error %d", ret);
    this->shareAttrFilter = shareAttrFilter;
    this->enableTimeFilter = enableTimeFilter;
    this->enableValFilter = (extraValFilter != nullptr);
    if (deviceMemMng.GetStrategy() == DevMemStrategy::PURE_DEVICE_MEM) {
        ret = searchInPureDev(count, tensorDevQueries.data(), attrFilter, topk, labels, distances, extraValFilter);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "searchInPureDev failed: %i.\n", ret);
        getValidNum(count, topk, labels, validNums);
    } else {
        std::vector<float16_t> fp16Distance(static_cast<size_t>(count) * static_cast<size_t>(topk));
        std::vector<idx_t> fp16Indexs(static_cast<size_t>(count) * static_cast<size_t>(topk));
        ret = searchBatched(static_cast<idx_t>(count), tensorDevQueries.data(), attrFilter,
            static_cast<idx_t>(topk), fp16Distance.data(), fp16Indexs.data(), nullptr, 0, false);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "search error %d", ret);
        postProcess(count, topk, fp16Distance.data(), distances, fp16Indexs.data(), labels, validNums);
    }

    APP_LOG_INFO("TSInt8FlatCos search operation operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR TSInt8FlatCos::callSearchWithShareMaskByBatch(int32_t queryNum, const int8_t *features, uint32_t topk,
    int64_t *labels, float *distances, AscendTensor<uint8_t, DIMS_3> &masks)
{
    uint32_t offset = 0;
    for (auto batch : this->searchBatchSizes) {
        while (queryNum >= batch) {
            auto ret = searchBatchWithShareMasks(batch, features + offset * this->code_size, static_cast<int>(topk),
                distances + offset * topk, labels + offset * topk, masks, nullptr);
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                "searchBatchWithShareMasks failed: %i.\n", ret);
            offset += static_cast<uint32_t>(batch);
            queryNum -= batch;
        }
    }
    return APP_ERR_OK;
}

APP_ERROR TSInt8FlatCos::searchInPureDevWithExtraMask(uint32_t count, const int8_t *features,
    const faiss::ascend::AttrFilter *attrFilter, uint32_t topk, const uint8_t *extraMask, int64_t *labels,
    float *distances, const float16_t *extraScore)
{
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    uint32_t offset = 0;
    int32_t queryNum = static_cast<int32_t>(count);
    ntotalPad = utils::roundUp(ntotal, FP16_ALGIN);
    const float16_t *extraScoreAddr = nullptr;
    if (this->shareAttrFilter) {
        APPERR_RETURN_IF_NOT_LOG(extraScore == nullptr,
            APP_ERR_INVALID_PARAM, "TSInt8FlatCos is not suport Extra Score when shareAttrFilter is true!");
        int blockNum = static_cast<int>(utils::divUp(static_cast<int64_t>(this->ntotal), this->codeBlockSize));
        AscendTensor<int32_t, DIMS_2> queryTimes(mem, { 1, OPS_DATA_TYPE_ALIGN }, stream);
        AscendTensor<uint8_t, DIMS_2> tokenIds(mem,
            { 1, static_cast<int32_t>(utils::divUp(tokenNum, OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES) }, stream);
        AscendTensor<uint8_t, DIMS_3> masks(mem, {blockNum, 1, utils::divUp(this->codeBlockSize, MASK_ALIGN)}, stream);
        buildAttr(attrFilter, 1, queryTimes, tokenIds);
        generateMaskWithExtra(1, 0, blockNum, queryTimes, tokenIds, extraMask, masks);
        return callSearchWithShareMaskByBatch(queryNum, features, topk, labels, distances, masks);
    }
    for (auto batch : this->searchBatchSizes) {
        while (queryNum >= batch) {
            AscendTensor<int32_t, DIMS_2> queryTimes(mem, { static_cast<int32_t>(batch), OPS_DATA_TYPE_ALIGN }, stream);
            AscendTensor<uint8_t, DIMS_2> tokenIds(mem, { static_cast<int32_t>(batch),
                static_cast<int32_t>(utils::divUp(tokenNum, OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES) }, stream);
            buildAttr(attrFilter + offset, batch, queryTimes, tokenIds);
            AscendTensor<float16_t, DIMS_2, idx_t> extraScoreDev;
            if (extraScore != nullptr) {
                extraScoreDev = AscendTensor<float16_t, DIMS_2, idx_t>(mem,
                                                                       {static_cast<idx_t>(batch), ntotalPad}, stream);
                auto ret = aclrtMemcpy(extraScoreDev.data(), extraScoreDev.getSizeInBytes(),
                    extraScore + offset * ntotalPad, batch * ntotalPad * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
                APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "aclrtMemcpy error %d", ret);
                extraScoreAddr = static_cast<const float16_t *>(extraScoreDev.data());
            }
            auto ret = searchBatchWithExtraNonshareMasks(batch, features + offset * this->code_size,
                static_cast<int>(topk), distances + offset * topk, labels + offset * topk, queryTimes, tokenIds,
                extraMask + offset * extraMaskLen, extraScoreAddr);
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                "searchBatchWithExtraNonshareMasks failed: %i.\n", ret);
            offset += static_cast<uint32_t>(batch);
            queryNum -= batch;
        }
    }
    return APP_ERR_OK;
}

APP_ERROR TSInt8FlatCos::searchWithExtraMask(uint32_t count, const void *features,
                                             const faiss::ascend::AttrFilter *attrFilter, bool shareAttrFilter,
                                             uint32_t topk, const uint8_t *extraMask, uint64_t extraMaskLen,
                                             bool extraMaskIsAtDevice, int64_t *labels, float *distances,
                                             uint32_t *validNums, bool enableTimeFilter, const float16_t *extraScore)
{
    APP_LOG_INFO("TSInt8FlatCos searchWithExtraMask operation started.\n");
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
    if (deviceMemMng.GetStrategy() == DevMemStrategy::PURE_DEVICE_MEM) {
        ret = searchInPureDevWithExtraMask(count, tensorDevQueries.data(), attrFilter, topk, extraMask, labels,
            distances, extraScore);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "searchInPureDevWithExtraMask failed: %i.\n",
            ret);
        getValidNum(count, topk, labels, validNums);
    } else {
        std::vector<float16_t> fp16Distance(static_cast<size_t>(count) * static_cast<size_t>(topk));
        std::vector<idx_t> fp16Indexs(static_cast<size_t>(count) * static_cast<size_t>(topk));
        ret = searchBatched(static_cast<idx_t>(count), tensorDevQueries.data(), attrFilter, static_cast<idx_t>(topk),
            fp16Distance.data(), fp16Indexs.data(), extraMask, extraMaskLen, extraMaskIsAtDevice);
        APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "search error %d", ret);
        postProcess(count, topk, fp16Distance.data(), distances, fp16Indexs.data(), labels, validNums);
    }
    APP_LOG_INFO("TSInt8FlatCos searchWithExtraMask operation operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR TSInt8FlatCos::createMask(uint32_t count, const faiss::ascend::AttrFilter *attrFilter,
                                    AscendTensor<uint8_t, DIMS_1> &genMasks)
{
    int idxMaskLen = static_cast<int>(utils::divUp(this->ntotal, OPS_DATA_TYPE_ALIGN));
    this->maskOnDevice = true;
    this->maskData = genMasks.data();
    if (this->shareAttrFilter) {
        generateMask(attrFilter, genMasks.data());
        // copy to other query mask
        for (uint32_t i = 1; i < count; ++i) {
            auto ret = aclrtMemcpy(genMasks.data() + i * idxMaskLen, idxMaskLen, genMasks.data(), idxMaskLen,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE);
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                                     "aclrtMemcpy data to device failed: %i.\n", ret);
        }
        return APP_ERR_OK;
    }
    for (uint32_t i = 0; i < count; i++) {
        generateMask(attrFilter + i, genMasks.data() + i * idxMaskLen);
    }
    return APP_ERR_OK;
}

APP_ERROR TSInt8FlatCos::createMask(uint32_t count, const faiss::ascend::AttrFilter *attrFilter,
                                    const uint8_t *extraMask, uint64_t extraMaskLen, bool extraMaskIsAtDevice,
                                    AscendTensor<uint8_t, DIMS_1> &genMasks)
{
    int idxMaskLen = static_cast<int>(utils::divUp(this->ntotal, OPS_DATA_TYPE_ALIGN));
    this->maskOnDevice = true;
    this->maskData = genMasks.data();
    if (this->shareAttrFilter) {
        generateMaskWithExtra(attrFilter, extraMask, extraMaskLen, extraMaskIsAtDevice, genMasks.data());
        // copy to other query mask
        for (uint32_t i = 1; i < count; ++i) {
            auto ret = aclrtMemcpy(genMasks.data() + i * idxMaskLen, idxMaskLen, genMasks.data(), idxMaskLen,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE);
            APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                                     "aclrtMemcpy data to device failed: %i.\n", ret);
        }
        return APP_ERR_OK;
    }
    
    for (uint32_t i = 0; i < count; i++) {
        generateMaskWithExtra(attrFilter + i, extraMask + i * extraMaskLen, extraMaskLen, extraMaskIsAtDevice,
                              genMasks.data() + i * idxMaskLen);
    }
    return APP_ERR_OK;
}

APP_ERROR TSInt8FlatCos::searchBatched(int n, const int8_t *x, const faiss::ascend::AttrFilter *attrFilter, int k,
                                       float16_t *distance, idx_t *labels, const uint8_t *extraMask,
                                       uint64_t extraMaskLen, bool extraMaskIsAtDevice)
{
    int ntotalAlign = static_cast<int>(utils::roundUp(ntotal, multiFeaAttrBlkSize));
    int genLen = utils::divUp(ntotalAlign, OPS_DATA_TYPE_ALIGN);
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = pResources->getMemoryManager();
    int64_t offset = 0;
    for (auto batch : searchBatchSizes) {
        while (n >= batch) {
            AscendTensor<uint8_t, DIMS_1> genMasks(mem, {batch * genLen}, stream);
            if (extraMask != nullptr) {
                auto ret =
                    createMask(static_cast<uint32_t>(batch), this->shareAttrFilter ? attrFilter : attrFilter + offset,
                    extraMask + offset * extraMaskLen, extraMaskLen, extraMaskIsAtDevice, genMasks);
                APPERR_RETURN_IF(ret, ret);
            } else {
                auto ret = createMask(static_cast<uint32_t>(batch),
                    this->shareAttrFilter ? attrFilter : attrFilter + offset, genMasks);
                APPERR_RETURN_IF(ret, ret);
            }
            auto ret = searchImpl(batch, x + offset * this->dims, k, distance + offset * k, labels + offset * k);
            APPERR_RETURN_IF(ret, ret);
            offset += batch;
            n -= batch;
        }
    }
    return APP_ERR_OK;
}

void TSInt8FlatCos::postProcess(int64_t searchNum, int topK, float16_t *inDistances, float *outDistances,
                                idx_t *inLabels, int64_t *outLabels, uint32_t *validNums)
{
    APP_LOG_INFO("TSInt8FlatCos postProcess operation started.\n");
    for (int64_t i = 0; i < searchNum; ++i) {
        std::transform(
            inDistances + i * topK, inDistances + (i + 1) * topK, outDistances + i * topK,
            [&](const float16_t temp) -> float { return static_cast<float>(faiss::ascend::fp16(temp)); });
        std::transform(inLabels + i * topK, inLabels + (i + 1) * topK, outLabels + i * topK,
                       [&](const idx_t temp) -> int64_t {
                           return static_cast<int64_t>(temp) == -1 ? -1 : static_cast<int64_t>(ids.at(temp));
                       });
        validNums[i] = 0;
        for (int j = 0; j < topK; j++) {
            int64_t tmpLabel = outLabels[i * topK + j];
            if (tmpLabel != -1) {
                validNums[i] += 1;
            }
        }
    }
    APP_LOG_INFO("TSInt8FlatCos postProcess operation end.\n");
}

void TSInt8FlatCos::getValidNum(uint64_t count, uint32_t topk, int64_t *labels, uint32_t *validNums) const
{
    APP_LOG_INFO("TSInt8FlatCos getValidNum operation started.\n");

    for (uint64_t i = 0; i < count; i++) {
        *(validNums + i) = 0;
        for (uint32_t j = 0; j < topk; j++) {
            int64_t tmpLabel = labels[i * topk + j];
            if (tmpLabel != -1) {
                *(validNums + i) += 1;
            }
        }
    }
    APP_LOG_INFO("TSInt8FlatCos getValidNum operation end.\n");
}

void TSInt8FlatCos::postProcess(uint64_t searchNum, int topK, AscendTensor<float16_t, DIMS_2> &outDistanceOnDevice,
    AscendTensor<int64_t, DIMS_2> &outIndicesOnDevice, float *distances, int64_t *labels)
{
    APP_LOG_INFO("TSInt8FlatCos postProcess operation started.\n");
    std::vector<float16_t> outDistances(searchNum * static_cast<uint64_t>(topK));
    auto ret = aclrtMemcpy(outDistances.data(), outDistances.size() * sizeof(float16_t), outDistanceOnDevice.data(),
        outDistanceOnDevice.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to copy back to host, ret=%i", ret);
    ret = aclrtMemcpy(labels, outIndicesOnDevice.getSizeInBytes(), outIndicesOnDevice.data(),
        outIndicesOnDevice.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    FAISS_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to copy back to host, ret=%i", ret);

    for (uint64_t i = 0; i < searchNum; ++i) {
        std::transform(outDistances.data() + i * topK, outDistances.data() + (i + 1) * topK, distances + i * topK,
            [&](const float16_t temp) -> float { return static_cast<float>(faiss::ascend::fp16(temp)); });
        std::transform(labels + i * topK, labels + (i + 1) * topK, labels + i * topK,
            [&](const idx_t temp) -> int64_t {
                return static_cast<int64_t>(temp) == -1 ? -1 : static_cast<int64_t>(ids.at(temp));
            });
    }
    APP_LOG_INFO("TSInt8FlatCos postProcess operation end.\n");
}

void TSInt8FlatCos::add_with_ids(idx_t n, const int8_t *x, const idx_t *xids)
{
    APP_LOG_INFO("TSInt8FlatCos add_with_ids operation started.\n");
    ids.insert(ids.end(), xids, xids + n);
    size_t offset = 0;
    size_t addTotal = n;
    size_t singleAddMax =
        static_cast<size_t>(utils::divDown(UPPER_LIMIT_FOR_ADD, this->codeBlockSize) * this->codeBlockSize);
    while (addTotal > 0) {
        auto singleAdd = std::min(addTotal, singleAddMax);
        addWithIdsImpl(singleAdd, x + offset * this->code_size);
        offset += singleAdd;
        addTotal -= singleAdd;
    }
    deviceMemMng.AddFinshProc(baseShaped);
    APP_LOG_INFO("TSInt8FlatCos add_with_ids operation end.\n");
}

void TSInt8FlatCos::addWithIdsImpl(int n, const int8_t *x)
{
    APP_LOG_INFO("TSInt8FlatCos addWithIdsImpl operation started.\n");

    AscendTensor<int8_t, DIMS_2> rawTensor(const_cast<int8_t *>(x), {n, dims});
    auto res = addVectors(rawTensor);
    FAISS_THROW_IF_NOT_FMT(res == APP_ERR_OK, "add vector failed, error code:%d", res);
    APP_LOG_INFO("TSInt8FlatCos addWithIdsImpl operation end.\n");
}

void TSInt8FlatCos::queryVectorByIdx(int64_t idx, uint8_t *dis) const
{
    if (faiss::ascend::SocUtils::GetInstance().IsAscend910B()) {
        size_t total = static_cast<size_t>(idx);
        size_t offsetInBlock = total % static_cast<size_t>(this->blockSize);
        size_t blockIdx = total / static_cast<size_t>(this->blockSize);

        int disOffset = 0;
        int srcOffset = offsetInBlock * code_size;
        std::vector<uint8_t> xb(this->code_size);

        auto ret = aclrtMemcpy(xb.data(), this->code_size * sizeof(uint8_t),
        baseShaped[blockIdx]->data() + srcOffset, this->code_size * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
        FAISS_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Failed to copy to host");
    } else {
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
                aclrtMemcpy(dis + disOffset, CUBE_ALIGN_INT8 * sizeof(int8_t), baseShaped[blockIdx]->data() + srcOffset,
                            CUBE_ALIGN_INT8 * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
            FAISS_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Failed to copy to host");
            disOffset += CUBE_ALIGN_INT8;
            srcOffset += CUBE_ALIGN_INT8 * CUBE_ALIGN;
        }
    }
}

void TSInt8FlatCos::runInt8CosDistCompute(int batch, bool shareMask, const std::vector<const AscendTensorBase *> &input,
    const std::vector<const AscendTensorBase *> &output, aclrtStream stream) const
{
    IndexTypeIdx indexType = shareMask ? IndexTypeIdx::ITI_INT8_COS_SHARE_MASK : IndexTypeIdx::ITI_INT8_COS_MASK;
    std::vector<int> keys({batch, dims, static_cast<int>(tokenNum)});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().runOp(indexType, opsKey, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
}

void TSInt8FlatCos::runAscendcInt8CosDistCompute(int batch, bool shareMask,
    const std::vector<const AscendTensorBase*>& input,
    const std::vector<const AscendTensorBase*>& output, aclrtStream stream) const
{
    IndexTypeIdx indexType = IndexTypeIdx::ASCENDC_ITI_INT8_COS_MASK;
    std::vector<int> keys({ batch, dims, static_cast<int>(tokenNum) });
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().runOp(indexType, opsKey, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
}

void TSInt8FlatCos::runInt8CosExtraScore(int batch, bool shareMask, const std::vector<const AscendTensorBase *> &input,
    const std::vector<const AscendTensorBase *> &output, aclrtStream stream) const
{
    IndexTypeIdx indexType = shareMask ? \
        IndexTypeIdx::ITI_INT8_COS_SHARE_MASK_EXTRASCORE : IndexTypeIdx::ITI_INT8_COS_MASK_EXTRASCORE;
    std::vector<int> keys({batch, dims, static_cast<int>(tokenNum)});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().runOp(indexType, opsKey, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
}

APP_ERROR TSInt8FlatCos::resetInt8CosDistCompute(int codeNum, bool shareMask) const
{
    std::string opTypeName = "DistanceInt8CosMaxsWithMask";
    IndexTypeIdx indexMaskType = shareMask ? IndexTypeIdx::ITI_INT8_COS_SHARE_MASK : IndexTypeIdx::ITI_INT8_COS_MASK;

    for (auto batch : searchBatchSizes) {
        std::vector<int64_t> queryShape({ batch, dims });
        std::vector<int64_t> maskShape({ shareMask ? 1 : batch, utils::divUp(codeNum, 8) }); // divUp to 8
        std::vector<int64_t> codeShape({ codeNum / CUBE_ALIGN, dims / CUBE_ALIGN_INT8, CUBE_ALIGN, CUBE_ALIGN_INT8 });
        std::vector<int64_t> queriesNormShape({ (batch + FP16_ALGIN - 1) / FP16_ALGIN * FP16_ALGIN });
        std::vector<int64_t> codesNormShape({ codeNum });
        std::vector<int64_t> sizeShape({ CORE_NUM, SIZE_ALIGN });
        std::vector<int64_t> resultShape({ batch, codeNum });
        std::vector<int64_t> minResultShape({ batch, this->burstsOfBlock });
        std::vector<int64_t> flagShape({ FLAG_NUM, FLAG_SIZE });

        std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
            { ACL_INT8, queryShape },
            { ACL_UINT8, maskShape },
            { ACL_INT8, codeShape },
            { ACL_FLOAT16, queriesNormShape },
            { ACL_FLOAT16, codesNormShape },
            { ACL_UINT32, sizeShape },
        };
        std::vector<std::pair<aclDataType, std::vector<int64_t>>> output {
            { ACL_FLOAT16, resultShape },
            { ACL_FLOAT16, minResultShape },
            { ACL_UINT16, flagShape }
        };
        std::vector<int> keys({batch, dims, static_cast<int>(tokenNum)});
        OpsMngKey opsKey(keys);
        auto ret = DistComputeOpsManager::getInstance().resetOp(opTypeName, indexMaskType, opsKey, input, output);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "op init failed: %i", ret);
    }
    return APP_ERR_OK;
}

APP_ERROR TSInt8FlatCos::resetAscendcInt8CosDistCompute(int codeNum) const
{
    std::string opTypeName = "AscendcDistanceInt8CosMaxsWithMask";
    IndexTypeIdx indexMaskType = IndexTypeIdx::ASCENDC_ITI_INT8_COS_MASK;
    const int FLAG_NUM = faiss::ascend::SocUtils::GetInstance().GetCoreNum();
    for (auto batch : searchBatchSizes) {
        std::vector<int64_t> queryShape({ batch, dims });
        std::vector<int64_t> maskShape({ batch, utils::divUp(codeNum, 8) }); // divUp to 8
        std::vector<int64_t> codeShape({ codeNum / CUBE_ALIGN, dims / CUBE_ALIGN_INT8, CUBE_ALIGN, CUBE_ALIGN_INT8 });
        std::vector<int64_t> queriesNormShape({ (batch + FP16_ALGIN - 1) / FP16_ALGIN * FP16_ALGIN });
        std::vector<int64_t> codesNormShape({ codeNum });
        std::vector<int64_t> sizeShape({ CORE_NUM, SIZE_ALIGN });
        std::vector<int64_t> resultShape({ batch, codeNum });
        std::vector<int64_t> minResultShape({ batch, this->burstsOfBlock });
        std::vector<int64_t> flagShape({ FLAG_NUM, FLAG_SIZE });

        std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
            { ACL_INT8, queryShape },
            { ACL_UINT8, maskShape },
            { ACL_INT8, codeShape },
            { ACL_FLOAT16, queriesNormShape },
            { ACL_FLOAT16, codesNormShape },
            { ACL_UINT32, sizeShape },
        };
        std::vector<std::pair<aclDataType, std::vector<int64_t>>> output {
            { ACL_FLOAT16, resultShape },
            { ACL_FLOAT16, minResultShape },
            { ACL_UINT16, flagShape }
        };
        std::vector<int> keys({batch, dims, static_cast<int>(tokenNum)});
        OpsMngKey opsKey(keys);
        auto ret = DistComputeOpsManager::getInstance().resetOp(opTypeName, indexMaskType, opsKey, input, output);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "op init failed: %i", ret);
    }
    return APP_ERR_OK;
}

APP_ERROR TSInt8FlatCos::resetInt8CosExtraScore(int codeNum, bool shareMask) const
{
    std::string opTypeName = "DistanceInt8CosMaxsWithMaskExtraScore";
    IndexTypeIdx indexMaskType = shareMask ? \
        IndexTypeIdx::ITI_INT8_COS_SHARE_MASK_EXTRASCORE : IndexTypeIdx::ITI_INT8_COS_MASK_EXTRASCORE;
    for (auto batch : searchBatchSizes) {
        std::vector<int64_t> queryShape({ batch, dims });
        std::vector<int64_t> maskShape({ shareMask ? 1 : batch, utils::divUp(codeNum, 8) }); // divUp to 8
        std::vector<int64_t> codeShape({ codeNum / CUBE_ALIGN, dims / CUBE_ALIGN_INT8, CUBE_ALIGN, CUBE_ALIGN_INT8 });
        std::vector<int64_t> queriesNormShape({ (batch + FP16_ALGIN - 1) / FP16_ALGIN * FP16_ALGIN });
        std::vector<int64_t> codesNormShape({ codeNum });
        std::vector<int64_t> sizeShape({ CORE_NUM, SIZE_ALIGN });
        std::vector<int64_t> extraScoreShape({ batch, codeNum });
        std::vector<int64_t> resultShape({ batch, codeNum });
        std::vector<int64_t> minResultShape({ batch, this->burstsOfBlock });
        std::vector<int64_t> flagShape({ FLAG_NUM, FLAG_SIZE });

        std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
            { ACL_INT8, queryShape },
            { ACL_UINT8, maskShape },
            { ACL_INT8, codeShape },
            { ACL_FLOAT16, queriesNormShape },
            { ACL_FLOAT16, codesNormShape },
            { ACL_UINT32, sizeShape },
            { ACL_FLOAT16, extraScoreShape },
        };
        std::vector<std::pair<aclDataType, std::vector<int64_t>>> output {
            { ACL_FLOAT16, resultShape },
            { ACL_FLOAT16, minResultShape },
            { ACL_UINT16, flagShape }
        };
        std::vector<int> keys({batch, dims, static_cast<int>(tokenNum)});
        OpsMngKey opsKey(keys);
        auto ret = DistComputeOpsManager::getInstance().resetOp(opTypeName, indexMaskType, opsKey, input, output);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "op init failed: %i", ret);
    }
    return APP_ERR_OK;
}

} // namespace ascend