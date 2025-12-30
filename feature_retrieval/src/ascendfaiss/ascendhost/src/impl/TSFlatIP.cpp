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


#include "ascendhost/include/impl/TSFlatIP.h"
#include <bitset>
#include <set>
#include "common/utils/DataType.h"
#include "ascend/utils/fp16.h"
#include "faiss/impl/FaissAssert.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"

namespace ascend {
const int INT8_LOWER_BOUND = -128;
const int INT8_UPPER_BOUND = 127;

TSFlatIP::TSFlatIP(uint32_t deviceId, uint32_t dim, uint32_t tokenNum, uint64_t resources, uint32_t customAttrLen,
    uint32_t customAttrBlockSize, const std::vector<float> &scale)
    : TSBase(tokenNum, customAttrLen, customAttrBlockSize), IndexFlatIPAicpu(dim, resources)
{
    FAISS_THROW_IF_NOT_FMT(std::find(DIM_RANGE.begin(), DIM_RANGE.end(), dim) != DIM_RANGE.end(),
        "Unsupported dims %u", dim);
    code_size = static_cast<int>(dim);
    this->deviceId = deviceId;
    auto ret = TSBase::initialize(deviceId);
    FAISS_THROW_IF_NOT_FMT(APP_ERR_OK == ret, "failed to init TSBase, ERRCODE:%d", ret);

    ret = IndexFlatIPAicpu::init();
    FAISS_THROW_IF_NOT_FMT(APP_ERR_OK == ret, "failed to init IndexFlatIPAicpu, ERRCODE:%d", ret);
    searchBatchSizes = {48, 36, 32, 30, 24, 18, 16, 12, 8, 6, 4, 2, 1};
    resetDistMaskCompOp(this->blockSize);
    resetDistMaskExtraScoreCompOp(this->blockSize);
    resetDistMaskWithScaleCompOp(this->blockSize);

    // 配置scale说明要进行底库特征的量化
    if (!scale.empty()) {
        auto ret = SetScale(scale);
        ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "TSFlatIP SetScale failed: %d", ret);
    }
}

APP_ERROR TSFlatIP::addFeatureWithLabels(int64_t n, const void *features,
    const faiss::ascend::FeatureAttr *attrs, const int64_t *labels, const uint8_t *customAttr,
    const faiss::ascend::ExtraValAttr *)
{
    APP_LOG_INFO("TSFlatIP::addFeatureWithLabels start");
    APPERR_RETURN_IF_NOT_LOG(scale.empty(), APP_ERR_INVALID_PARAM,
        "InitWithQuantify not support AddFeature, please use AddFeatureByIndice");
    std::set<int64_t> uniqueLabels(labels, labels + n);
    APPERR_RETURN_IF_NOT_LOG(uniqueLabels.size() == static_cast<size_t>(n), APP_ERR_INVALID_PARAM,
        "the labels is not unique");
    for (int64_t i = 0; i < n; i++) {
        APPERR_RETURN_IF_NOT_FMT(label2Idx.find(*(labels + i)) == label2Idx.end(), APP_ERR_INVALID_PARAM,
            "the label[%ld] is already exists", *(labels + i));
    }

    addFeatureAttrs(n, attrs, customAttr);
    //  cast uint16 pointer to float16
    add_with_ids(n, reinterpret_cast<const float *>(features), reinterpret_cast<const idx_t *>(labels));
    for (int64_t i = static_cast<int64_t>(this->ntotal) - n; i < static_cast<int64_t>(this->ntotal); ++i) {
        label2Idx[ids[i]] = i;
    }
    APP_LOG_INFO("TSFlatIP::addFeatureWithLabels end");
    return APP_ERR_OK;
}

APP_ERROR TSFlatIP::AddFeatureByIndice(int64_t n, const void *features,
    const faiss::ascend::FeatureAttr *attrs, const int64_t *indices, const uint8_t *customAttr,
    const faiss::ascend::ExtraValAttr *)
{
    APP_LOG_INFO("TSFlatIP::AddFeatureByIndice start");
    // 如有包含有新增的特征，新增的indice的位置
    int64_t replaceNum = -1;
    std::vector<std::pair<int64_t, int64_t>> segments;
    APPERR_RETURN_IF_NOT_LOG(CheckIndices(this->ntotal, n, indices, replaceNum, segments) == APP_ERR_OK,
        APP_ERR_INVALID_PARAM, "CheckIndices failed");

    APP_LOG_INFO("n:%ld, maxIndice:%ld, replaceNum:%ld, ntotal:%ld", n, indices[n - 1], replaceNum, this->ntotal);

    auto ret = AddFeatureAttrsByIndice(n, segments, indices, attrs, customAttr, nullptr);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "AddFeatureAttrsByIndice failed:%d\n", ret);

    ret = AddFeatureWithIndice(n, replaceNum, indices, segments, reinterpret_cast<const float *>(features));
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "AddFeatureWithIndice failed:%d\n", ret);

    SetMaskValid(n, indices, this->ntotal);

    APP_LOG_INFO("TSFlatIP::AddFeatureByIndice end");
    return APP_ERR_OK;
}

APP_ERROR TSFlatIP::GetFeatureByIndice(int64_t count, const int64_t *indices, int64_t *labels,
    void *features, faiss::ascend::FeatureAttr *attributes, faiss::ascend::ExtraValAttr *) const
{
    APP_LOG_INFO("TSFlatIP::GetFeatureByIndice start");
    // 获取特征向量
    if (features != nullptr) {
        for (int64_t i = 0; i < count; ++i) {
            // 不进行量化时
            if (scale.empty()) {
                queryVectorByIdx(indices[i], reinterpret_cast<float *>(features) + (i * code_size));
            } else { // 量化时
                queryInt8VectorByIdx(indices[i], reinterpret_cast<float *>(features) + (i * code_size));
            }
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
    APP_LOG_INFO("TSFlatIP::GetFeatureByIndice end");
    return APP_ERR_OK;
}

APP_ERROR TSFlatIP::AddFeatureWithIndice(int64_t n, int64_t replaceNum, const int64_t *indices,
    const std::vector<std::pair<int64_t, int64_t>> &segments, const float *features)
{
    APP_LOG_INFO("TSFlatIP AddFeatureWithIndice operation started.\n");

    // 一次性最多申请singleAddMax条空间来更新或者添加
    int64_t singleAddMax = utils::divDown(UPPER_LIMIT_FOR_ADD, this->blockSize) * this->blockSize;

    // 新增部分的空间申请
    int64_t addNum = n - replaceNum;
    if (addNum > 0) {
        if (scale.empty()) {
            resizeBaseShaped(static_cast<size_t>(addNum));
        } else {
            resizeBaseShapedInt8(static_cast<size_t>(addNum));
        }
    }

    // 每个连续段进行拷贝
    int64_t offset = 0;
    for (size_t segIdx = 0; segIdx < segments.size(); segIdx++) {
        int64_t start = segments[segIdx].first;  // 起始位置
        int64_t length = segments[segIdx].second; // 每段的长度
        int64_t startOffset = indices[start];
        while (length > 0) {
            auto singleAdd = std::min(length, singleAddMax);
            auto ret = AddFeatureImpl(singleAdd, features, offset, startOffset);
            APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "add feature failed:%d\n", ret);
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

    APP_LOG_INFO("TSFlatIP AddFeatureWithIndice operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR TSFlatIP::AddFeatureImpl(int64_t singleAdd, const float *features, int64_t offset, int64_t startOffset)
{
    if (scale.empty()) { // 不需要量化
        auto featureFp16 = new (std::nothrow) float16_t[singleAdd * static_cast<int64_t>(this->code_size)];
        APPERR_RETURN_IF_NOT_LOG(featureFp16 != nullptr, APP_ERR_ACL_BAD_ALLOC,
            "Memory allocation fail for featureFp16");
        std::shared_ptr<float16_t> featureDeleter(featureFp16, std::default_delete<float16_t[]>());
        std::transform(features + offset * this->code_size, features + (offset + singleAdd) * this->code_size,
            featureFp16, [](float temp) { return faiss::ascend::fp16(temp).data; });

        AscendTensor<float16_t, DIMS_2> rawTensor(featureFp16, { static_cast<int>(singleAdd), this->code_size });
        auto ret = copyAndSaveVectors(startOffset, rawTensor);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "copy feature to device failed:%d\n", ret);
        return APP_ERR_OK;
    }

    // 需要量化
    auto featureInt8 = new (std::nothrow) int8_t[singleAdd * static_cast<int64_t>(this->code_size)];
    APPERR_RETURN_IF_NOT_LOG(featureInt8 != nullptr, APP_ERR_ACL_BAD_ALLOC,
        "Memory allocation fail for featureInt8");
    std::shared_ptr<int8_t> featureDeleter(featureInt8, std::default_delete<int8_t[]>());

    for (int64_t i = 0; i < singleAdd; i++) {
        auto featureOffset = offset * this->code_size;
        for (int64_t j = 0; j < this->code_size; j++) {
            auto dimOffset = i * this->code_size + j;
            float oriFeature = (*(features + featureOffset + dimOffset));
            int32_t data = static_cast<int32_t>(oriFeature * scale[j]);
            if (data < INT8_LOWER_BOUND) {
                featureInt8[dimOffset] = INT8_LOWER_BOUND;
            } else if (data > INT8_UPPER_BOUND) {
                featureInt8[dimOffset] = INT8_UPPER_BOUND;
            } else {
                featureInt8[dimOffset] = static_cast<int8_t>(data);
            }
        }
    }

    AscendTensor<int8_t, DIMS_2> rawTensor(featureInt8, { static_cast<int>(singleAdd), this->code_size });
    auto ret = copyAndSaveVectors(startOffset, rawTensor);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "copy feature to device failed:%d\n", ret);
    return APP_ERR_OK;
}

APP_ERROR TSFlatIP::getFeatureByLabel(int64_t n, const int64_t *labels, void *features) const
{
    APP_LOG_INFO("TSFlatIP::getFeatureByLabel start");
    std::vector<int64_t> queryIds;
    for (int64_t i = 0; i < n; ++i) {
        auto it = label2Idx.find(*(labels + i));
        APPERR_RETURN_IF_NOT_FMT(it != label2Idx.end(), APP_ERR_INVALID_PARAM,
            "the label[%ld] does not exists", *(labels+i));
        queryIds.emplace_back(it->second);
    }
    for (size_t i = 0; i < queryIds.size(); ++i) {
        queryVectorByIdx(queryIds[i], reinterpret_cast<float *>(features) + (i * code_size));
    }
    APP_LOG_INFO("TSFlatIP::getFeatureByLabel end");
    return APP_ERR_OK;
};

APP_ERROR TSFlatIP::getBaseByRange(uint32_t offset, uint32_t num, int64_t *labels, void *features,
    faiss::ascend::FeatureAttr *attributes, faiss::ascend::ExtraValAttr *)
{
    APP_LOG_INFO("TSFlatIP::getBaseByRange start");
    // 不需要量化
    if (scale.empty()) {
        std::vector<float16_t> baseVectors(static_cast<int64_t>(num) * this->code_size);
        getVectorsAiCpu(offset, num, baseVectors);
        std::transform(baseVectors.begin(), baseVectors.end(), reinterpret_cast<float *>(features),
            [&](const float16_t temp) -> float { return static_cast<float>(faiss::ascend::fp16(temp)); });
    } else { // 需要量化时
        std::vector<int8_t> baseVectors(static_cast<int64_t>(num) * this->code_size);
        getInt8VectorsAiCpu(offset, num, baseVectors);
        auto featuresFloat = reinterpret_cast<float *>(features);
        for (int64_t i = 0; i < num; i++) {
            for (int64_t j = 0; j < this->code_size; j++) {
                auto offset = i * this->code_size + j;
                *(featuresFloat + offset) = static_cast<float>(baseVectors[offset]) / scale[j];
            }
        }
    }

    auto ret = memcpy_s(labels, num * sizeof(int64_t), this->ids.data() + offset, num * sizeof(int64_t));
    APPERR_RETURN_IF_NOT_FMT(ret == 0, APP_ERR_INNER_ERROR, "Memcpy_s labels failed(%d).", ret);
    ret = memcpy_s(attributes, num * sizeof(faiss::ascend::FeatureAttr), this->featureAttrs.data() + offset,
        num * sizeof(faiss::ascend::FeatureAttr));
    APPERR_RETURN_IF_NOT_FMT(ret == 0, APP_ERR_INNER_ERROR, "Memcpy_s attributes failed(%d).", ret);
    getBaseEnd();
    APP_LOG_INFO("TSFlatIP::getBaseByRange end");
    return APP_ERR_OK;
};

void TSFlatIP::removeIdsImpl(const std::vector<int64_t> &indices)
{
    APP_LOG_INFO("TSFlatIP removeIdsImpl operation start. \n");
    // move the end data to the locate of delete data
    uint32_t zRegionHeight = CUBE_ALIGN;
    uint32_t dimAlignSize = static_cast<uint32_t>(utils::divUp(this->code_size, CUBE_ALIGN));
    int removedCnt = static_cast<int>(indices.size());
    std::vector<uint64_t> srcAddr(removedCnt);
    std::vector<uint64_t> dstAddr(removedCnt);
    std::string opName = "RemovedataShaped";
    auto &mem = pResources->getMemoryManager();
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    for (int i = 0; i < removedCnt; i++) {
        idx_t srcIdx = this->ntotal - static_cast<idx_t>(i) - 1;
        idx_t srcIdx1 = srcIdx / static_cast<idx_t>(this->blockSize);
        idx_t srcIdx2 = srcIdx % static_cast<idx_t>(this->blockSize);
        idx_t dstIdx = static_cast<idx_t>(indices[i]);
        idx_t dstIdx1 = dstIdx / static_cast<idx_t>(this->blockSize);
        idx_t dstIdx2 = dstIdx % static_cast<idx_t>(this->blockSize);

        float16_t *srcDataPtr = baseShaped[srcIdx1]->data() + (srcIdx2 / zRegionHeight) *
            (dimAlignSize * zRegionHeight * CUBE_ALIGN) + (srcIdx2 % zRegionHeight) * CUBE_ALIGN;
        float16_t *dstDataPtr = baseShaped[dstIdx1]->data() + (dstIdx2 / zRegionHeight) *
            (dimAlignSize * zRegionHeight * CUBE_ALIGN) + (dstIdx2 % zRegionHeight) * CUBE_ALIGN;

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
    attrs[aicpu::REMOVEDATA_SHAPED_ATTR_DATA_TYPE] = faiss::ascend::FLOAT16;
    attrs[aicpu::REMOVEDATA_SHAPED_ATTR_ZREGION_HEIGHT] = zRegionHeight;
    attrs[aicpu::REMOVEDATA_SHAPED_ATTR_DIM_ALIGN_NUM] = dimAlignSize;
    attrs[aicpu::REMOVEDATA_SHAPED_ATTR_CUBE_ALIGN] = CUBE_ALIGN;
    ret = aclrtMemcpy(attrsInput.data(), attrsInput.getSizeInBytes(), attrs.data(), attrs.size() * sizeof(int64_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to copy to device (error %d)", (int)ret);
    LaunchOpTwoInOneOut<uint64_t, DIMS_1, ACL_UINT64, int64_t, DIMS_1, ACL_INT64, uint64_t, DIMS_1, ACL_UINT64>(opName,
        stream, srcInput, attrsInput, dstInput);

    ret = synchronizeStream(stream);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to acl Synchronize Streame (error %d)", (int)ret);
    APP_LOG_INFO("TSFlatIP removeIdsImpl operation finished. \n");
}

APP_ERROR TSFlatIP::deleteFeatureByToken(int64_t count, const uint32_t *tokens)
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
    APP_LOG_INFO("TSFlatIP::deleteFeatureByToken start count:%u", removeIds.size());
    // use TSBase function to delete attr
    deleteAttrByIds(removeIds);
    //  remove labels and set new idx of last label
    removeLabels(removeIds);
    // remove baseShaped and base norm
    removeIdsImpl(removeIds);
    // release the space  of baseShape and norm
    releaseUnusageSpace(this->ntotal, removeIds.size());
    this->ntotal -= removeIds.size();
    APP_LOG_INFO("TSFlatIP::deleteFeatureByToken delete count:%d", removeIds.size());
    return APP_ERR_OK;
}

APP_ERROR TSFlatIP::delFeatureWithLabels(int64_t n, const int64_t *labels)
{
    APP_LOG_INFO("TSFlatIP::delFeatureWithLabels start cound:%d", n);
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
    removeIdsImpl(removeIds);
    // release the space  of baseShape and norm
    releaseUnusageSpace(this->ntotal, removeIds.size());
    this->ntotal -= removeIds.size();
    APP_LOG_INFO("TSFlatIP::delFeatureWithLabels delete count:%d", removeIds.size());
    return APP_ERR_OK;
}

void TSFlatIP::resetDistMaskCompOp(int numLists)
{
    APP_LOG_INFO("TSFlatIP resetDistMaskCompOp operation started.\n");
    std::string opTypeName = "DistanceFlatIPMaxsWithMask";
    auto distCompMaskOpReset = [&](int batch, bool shareMask) {
        IndexTypeIdx indexMaskType = shareMask ? IndexTypeIdx::ITI_FLAT_IP_SHARE_MASK : IndexTypeIdx::ITI_FLAT_IP_MASK;
        std::vector<int64_t> queryShape({ batch, dims });
        std::vector<int64_t> coarseCentroidsShape(
            { utils::divUp(numLists, CUBE_ALIGN), utils::divUp(dims, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN });
        std::vector<int64_t> sizeShape({ CORE_NUM, SIZE_ALIGN });
        std::vector<int64_t> maskShape({ shareMask ? 1 : batch, utils::divUp(numLists, 8) });
        std::vector<int64_t> distResultShape({ batch, numLists });
        std::vector<int64_t> maxResultShape({ batch, this->burstsOfBlock });
        std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });

        std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
            { ACL_FLOAT16, queryShape },
            { ACL_FLOAT16, coarseCentroidsShape },
            { ACL_UINT32, sizeShape },
            { ACL_UINT8, maskShape },
        };
        std::vector<std::pair<aclDataType, std::vector<int64_t>>> output {
            { ACL_FLOAT16, distResultShape },
            { ACL_FLOAT16, maxResultShape },
            { ACL_UINT16, flagShape }
        };
        std::vector<int> keys({batch, dims, static_cast<int>(tokenNum)});
        OpsMngKey opsKey(keys);
        return DistComputeOpsManager::getInstance().resetOp(opTypeName, indexMaskType, opsKey, input, output);
    };

    for (auto batch : searchBatchSizes) {
        FAISS_THROW_IF_NOT_MSG(!distCompMaskOpReset(batch, false), "no share op init failed");
        FAISS_THROW_IF_NOT_MSG(!distCompMaskOpReset(batch, true), "share op init failed");
    }

    APP_LOG_INFO("TSFlatIP resetDistMaskCompOp operation end.\n");
}

void TSFlatIP::resetDistMaskExtraScoreCompOp(int numLists)
{
    APP_LOG_INFO("TSFlatIP resetDistMaskExtraScoreCompOp operation started.\n");
    std::string opTypeName = "DistanceFlatIPMaxsWithExtraScore";
    auto distCompMaskOpReset = [&](int batch, bool shareMask) {
        IndexTypeIdx indexMaskType = shareMask ? \
            IndexTypeIdx::ITI_FLAT_IP_SHARE_MASK_EXTRA_SCORE : IndexTypeIdx::ITI_FLAT_IP_MASK_EXTRA_SCORE;
        std::vector<int64_t> queryShape({ batch, dims });
        std::vector<int64_t> coarseCentroidsShape(
            { utils::divUp(numLists, CUBE_ALIGN), utils::divUp(dims, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN });
        std::vector<int64_t> sizeShape({ CORE_NUM, SIZE_ALIGN });
        std::vector<int64_t> maskShape({ shareMask ? 1 : batch, utils::divUp(numLists, 8) });
        std::vector<int64_t> extraScoreShape({ batch, numLists });
        std::vector<int64_t> distResultShape({ batch, numLists });
        std::vector<int64_t> maxResultShape({ batch, this->burstsOfBlock });
        std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });

        std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
            { ACL_FLOAT16, queryShape },
            { ACL_FLOAT16, coarseCentroidsShape },
            { ACL_UINT32, sizeShape },
            { ACL_UINT8, maskShape },
            { ACL_FLOAT16, extraScoreShape },
        };
        std::vector<std::pair<aclDataType, std::vector<int64_t>>> output {
            { ACL_FLOAT16, distResultShape },
            { ACL_FLOAT16, maxResultShape },
            { ACL_UINT16, flagShape }
        };

        std::vector<int> keys({batch, dims, static_cast<int>(tokenNum)});
        OpsMngKey opsKey(keys);
        return DistComputeOpsManager::getInstance().resetOp(opTypeName, indexMaskType, opsKey, input, output);
    };

    for (auto batch : searchBatchSizes) {
        FAISS_THROW_IF_NOT_MSG(!distCompMaskOpReset(batch, false), "no share op init failed");
        FAISS_THROW_IF_NOT_MSG(!distCompMaskOpReset(batch, true), "share op init failed");
    }

    APP_LOG_INFO("TSFlatIP resetDistMaskExtraScoreCompOp operation end.\n");
}

void TSFlatIP::resetDistMaskWithScaleCompOp(int numLists)
{
    APP_LOG_INFO("TSFlatIP resetDistMaskWithScaleCompOp operation started.\n");
    auto distCompMaskWithScaleOpReset = [&](int batch, bool isUsedExtraScore) {
        IndexTypeIdx indexMaskType = isUsedExtraScore ?
            IndexTypeIdx::ITI_FLAT_IP_EXTRA_SCORE_AND_SCALE : IndexTypeIdx::ITI_FLAT_IP_NOSCORE_AND_SCALE;
        std::string opTypeName = isUsedExtraScore ?
            "DistanceFlatIPMaxsWithScale" : "DistanceFlatIPMaxsNoScoreWithScale";
        std::vector<int64_t> queryShape({ batch, dims });
        std::vector<int64_t> coarseCentroidsShape(
            { utils::divUp(numLists, CUBE_ALIGN), utils::divUp(dims, CUBE_ALIGN_INT8), CUBE_ALIGN, CUBE_ALIGN_INT8 });
        std::vector<int64_t> sizeShape({ CORE_NUM, SIZE_ALIGN });
        std::vector<int64_t> maskShape({ batch, utils::divUp(numLists, 8) });
        std::vector<int64_t> extraScoreShape({ batch, numLists });
        std::vector<int64_t> scaleShape({ dims });
        std::vector<int64_t> distResultShape({ batch, numLists });
        std::vector<int64_t> maxResultShape({ batch, this->burstsOfBlock });
        std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });

        std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
            { ACL_FLOAT16, queryShape },
            { ACL_INT8, coarseCentroidsShape },
            { ACL_UINT32, sizeShape },
            { ACL_UINT8, maskShape },
            { ACL_FLOAT16, scaleShape },
        };
        if (isUsedExtraScore) {
            input = {
                { ACL_FLOAT16, queryShape },
                { ACL_INT8, coarseCentroidsShape },
                { ACL_UINT32, sizeShape },
                { ACL_UINT8, maskShape },
                { ACL_FLOAT16, extraScoreShape },
                { ACL_FLOAT16, scaleShape },
            };
        }
        std::vector<std::pair<aclDataType, std::vector<int64_t>>> output {
            { ACL_FLOAT16, distResultShape },
            { ACL_FLOAT16, maxResultShape },
            { ACL_UINT16, flagShape }
        };
        std::vector<int> keys({batch, dims, static_cast<int>(tokenNum)});
        OpsMngKey opsKey(keys);
        return DistComputeOpsManager::getInstance().resetOp(opTypeName, indexMaskType, opsKey, input, output);
    };

    for (auto batch : searchBatchSizes) {
        FAISS_THROW_IF_NOT_MSG(!distCompMaskWithScaleOpReset(batch, false), "no extraScore op init failed");
        FAISS_THROW_IF_NOT_MSG(!distCompMaskWithScaleOpReset(batch, true), "extraScore op init failed");
    }

    APP_LOG_INFO("TSFlatIP resetDistMaskWithScaleCompOp operation end.\n");
}

void TSFlatIP::runDistMaskComputeWithScale(int batch, bool isUsedExtraScore,
    const std::vector<const AscendTensorBase *> &input,
    const std::vector<const AscendTensorBase *> &output, aclrtStream stream)
{
    APP_LOG_INFO("TSFlatIP runDistMaskComputeWithScale operation started.\n");
    IndexTypeIdx indexType = isUsedExtraScore ?
        IndexTypeIdx::ITI_FLAT_IP_EXTRA_SCORE_AND_SCALE : IndexTypeIdx::ITI_FLAT_IP_NOSCORE_AND_SCALE;
    std::vector<int> keys({batch, dims, static_cast<int>(tokenNum)});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().runOp(indexType, opsKey, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
    APP_LOG_INFO("TSFlatIP runDistMaskComputeWithScale operation end.\n");
}

void TSFlatIP::runDistMaskExtraScoreCompute(int batch, bool shareMask,
    const std::vector<const AscendTensorBase *> &input,
    const std::vector<const AscendTensorBase *> &output, aclrtStream stream)
{
    APP_LOG_INFO("TSFlatIP runDistMaskExtraScoreCompute operation started.\n");
    IndexTypeIdx indexType = shareMask ? \
        IndexTypeIdx::ITI_FLAT_IP_SHARE_MASK_EXTRA_SCORE : IndexTypeIdx::ITI_FLAT_IP_MASK_EXTRA_SCORE;
    std::vector<int> keys({batch, dims, static_cast<int>(tokenNum)});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().runOp(indexType, opsKey, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
    APP_LOG_INFO("TSFlatIP runDistMaskExtraScoreCompute operation end.\n");
}

void TSFlatIP::runDistMaskCompute(int batch, bool shareMask, const std::vector<const AscendTensorBase *> &input,
    const std::vector<const AscendTensorBase *> &output, aclrtStream stream)
{
    APP_LOG_INFO("TSFlatIP runDistMaskCompute operation started.\n");
    IndexTypeIdx indexType = shareMask ? IndexTypeIdx::ITI_FLAT_IP_SHARE_MASK : IndexTypeIdx::ITI_FLAT_IP_MASK;
    std::vector<int> keys({batch, dims, static_cast<int>(tokenNum)});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().runOp(indexType, opsKey, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
    APP_LOG_INFO("TSFlatIP runDistMaskCompute operation end.\n");
}

void TSFlatIP::getValidNum(uint64_t count, uint32_t topk, int64_t *labels, uint32_t *validNums) const
{
    APP_LOG_INFO("TSFlatIP getValidNum operation started.\n");

    for (uint64_t i = 0; i < count; i++) {
        *(validNums + i) = 0;
        for (uint32_t j = 0; j < topk; j++) {
            int64_t tmpLabel = labels[i * topk + j];
            if (tmpLabel != -1) {
                *(validNums + i) += 1;
            }
        }
    }
    APP_LOG_INFO("TSFlatIP getValidNum operation end.\n");
}

APP_ERROR TSFlatIP::search(uint32_t count, const void *features, const faiss::ascend::AttrFilter *attrFilter,
    bool shareAttrFilter, uint32_t topk, int64_t *labels, float *distances, uint32_t *validNums,
    bool enableTimeFilter, const faiss::ascend::ExtraValFilter *)
{
    using namespace faiss::ascend;
    APP_LOG_INFO("TSFlatIP search operation started.\n");

    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = pResources->getMemoryManager();

    APPERR_RETURN_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE, APP_ERR_ACL_BAD_ALLOC);
    this->shareAttrFilter = shareAttrFilter;
    this->enableTimeFilter = enableTimeFilter;
    int64_t offset = 0;
    int32_t queryNum = static_cast<int32_t>(count);

    const float *queryFeatures = reinterpret_cast<const float *>(features);
    int blockNum =  static_cast<int>(utils::divUp(this->ntotal, this->blockSize));
    if (this->shareAttrFilter) {
        APPERR_RETURN_IF_NOT_LOG(scale.empty(),
            APP_ERR_INVALID_PARAM, "TSFlatIP is not suport scale when shareAttrFilter is true!");

        AscendTensor<int32_t, DIMS_2> queryTimes(mem, { 1, OPS_DATA_TYPE_ALIGN }, stream);
        AscendTensor<uint8_t, DIMS_2> tokenIds(mem,
            { 1, static_cast<int32_t>(utils::divUp(tokenNum, OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES) }, stream);
        AscendTensor<uint8_t, DIMS_3> masks(mem, { blockNum, 1, utils::divUp(this->blockSize, MASK_ALIGN) }, stream);
        buildAttr(attrFilter, 1, queryTimes, tokenIds);
        generateMask(1, 0, blockNum, queryTimes, tokenIds, masks);
        for (auto batch : searchBatchSizes) {
            while (queryNum >= batch) {
                std::vector<uint16_t> query(batch * this->code_size, 0);
                transform(queryFeatures + offset * this->code_size, queryFeatures + (offset + batch) * this->code_size,
                    begin(query), [](float temp) { return fp16(temp).data; });
                APP_ERROR ret = searchBatchWithShareMasks(batch, query.data(), static_cast<int>(topk),
                    distances + offset * topk, labels + offset * topk, masks);
                APPERR_RETURN_IF_NOT_FMT(ret == 0, APP_ERR_INNER_ERROR, "searchBatchWithShareMasks failed(%d).", ret);
                offset += batch;
                queryNum -= batch;
            }
        }
    } else {
        for (auto batch : searchBatchSizes) {
            while (queryNum >= batch) {
                std::vector<uint16_t> query(batch * this->code_size, 0);
                transform(queryFeatures + offset * this->code_size, queryFeatures + (offset + batch) * this->code_size,
                    begin(query), [](float temp) { return fp16(temp).data; });

                AscendTensor<int32_t, DIMS_2> queryTimes(mem, { batch, OPS_DATA_TYPE_ALIGN }, stream);
                AscendTensor<uint8_t, DIMS_2> tokenIds(mem,
                    { batch, static_cast<int32_t>(utils::divUp(tokenNum, OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES) },
                    stream);
                buildAttr(attrFilter + offset, batch, queryTimes, tokenIds);
                APP_ERROR ret = searchBatchWithNonshareMasks(batch, query.data(), static_cast<int>(topk),
                    distances + offset * topk, labels + offset * topk, queryTimes, tokenIds);
                APPERR_RETURN_IF_NOT_FMT(ret == 0, APP_ERR_INNER_ERROR,
                    "searchBatchWithNonshareMasks failed(%d).", ret);
                offset += batch;
                queryNum -= batch;
            }
        }
    }

    getValidNum(count, topk, labels, validNums);

    APP_LOG_INFO("TSFlatIP search operation operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR TSFlatIP::CopyDataToHost(std::vector<float16_t>& outDistances, int64_t *labels,
    AscendTensor<float16_t, DIMS_2>& outDistanceOnDevice, AscendTensor<int64_t, DIMS_2>& outIndicesOnDevice)
{
    auto ret = aclrtMemcpy(outDistances.data(), outDistances.size() * sizeof(float16_t), outDistanceOnDevice.data(),
        outDistanceOnDevice.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy outDistances back to host");
    ret = aclrtMemcpy(labels, outDistances.size() * sizeof(int64_t), outIndicesOnDevice.data(),
        outIndicesOnDevice.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy outIndices back to host");
    return APP_ERR_OK;
}

APP_ERROR TSFlatIP::searchBatchWithNonshareMasks(int batch, const uint16_t *x, int topK, float *distances,
    int64_t *labels, AscendTensor<int32_t, DIMS_2> &queryTimes, AscendTensor<uint8_t, DIMS_2> &tokenIds)
{
    APP_LOG_INFO("TSFlatIP searchBatchWithNonshareMasks operation started.\n");
    // 1. generate result variable
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();

    AscendTensor<float16_t, DIMS_2> outDistanceOnDevice(mem, { batch, topK }, stream);
    AscendTensor<int64_t, DIMS_2> outIndicesOnDevice(mem, { batch, topK }, stream);

    // 2. compute distance by code page
    size_t pageNum = utils::divUp(this->ntotal, pageSize);
    for (size_t pageId = 0; pageId < pageNum; ++pageId) {
        size_t pageOffset = pageId * static_cast<size_t>(this->pageSize);
        size_t blockOffset = pageId * static_cast<size_t>(this->pageSize) / static_cast<size_t>(this->blockSize);
        int computeNum = std::min(this->ntotal - pageOffset, static_cast<idx_t>(this->pageSize));
        int blockNum = utils::divUp(computeNum, this->blockSize);
        AscendTensor<uint8_t, DIMS_3> masks(mem, { blockNum, batch, utils::divUp(this->blockSize, MASK_ALIGN) },
            stream);
        generateMask(batch, blockOffset, blockNum, queryTimes, tokenIds, masks);
        APP_ERROR ret = searchPagedWithMasks(pageId, pageNum, batch, x, masks, outDistanceOnDevice, outIndicesOnDevice);
        APPERR_RETURN_IF_NOT_FMT(ret == 0, APP_ERR_INNER_ERROR, "searchPagedWithMasks failed(%d).", ret);
    }
    // memcpy data back from dev to host
    std::vector<float16_t> outDistances(batch * topK);
    auto ret = CopyDataToHost(outDistances, labels, outDistanceOnDevice, outIndicesOnDevice);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "CopyDataToHost failed");

    postProcess(batch, topK, outDistances.data(), distances, labels);
    APP_LOG_INFO("TSFlatIP searchBatchWithNonshareMasks operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR TSFlatIP::searchBatchWithShareMasks(int batch, const uint16_t *x, int topK, float *distances, int64_t *labels,
    AscendTensor<uint8_t, DIMS_3> &masks)
{
    APP_LOG_INFO("TSFlatIP searchBatchWithShareMasks operation started.\n");
    // 1. generate result variable
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();

    AscendTensor<float16_t, DIMS_2> outDistanceOnDevice(mem, { batch, topK }, stream);
    AscendTensor<int64_t, DIMS_2> outIndicesOnDevice(mem, { batch, topK }, stream);

    // 2. compute distance by code page
    size_t pageNum = utils::divUp(this->ntotal, pageSize);
    for (size_t pageId = 0; pageId < pageNum; ++pageId) {
        size_t pageOffset = pageId * static_cast<size_t>(this->pageSize);
        size_t blockOffset = pageId * static_cast<size_t>(this->pageSize) / static_cast<size_t>(this->blockSize);
        int computeNum = std::min(this->ntotal - pageOffset, static_cast<idx_t>(this->pageSize));
        int blockNum = utils::divUp(computeNum, this->blockSize);
        AscendTensor<uint8_t, DIMS_3> subMasks(masks[blockOffset].data(),
            { blockNum, 1, utils::divUp(this->blockSize, MASK_ALIGN) });
        APP_ERROR ret =
            searchPagedWithMasks(pageId, pageNum, batch, x, subMasks, outDistanceOnDevice, outIndicesOnDevice);
        APPERR_RETURN_IF_NOT_FMT(ret == 0, APP_ERR_INNER_ERROR, "searchPagedWithMasks failed(%d).", ret);
    }
    // memcpy data back from dev to host
    std::vector<float16_t> outDistances(batch * topK);
    auto ret = CopyDataToHost(outDistances, labels, outDistanceOnDevice, outIndicesOnDevice);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "CopyDataToHost failed");

    postProcess(batch, topK, outDistances.data(), distances, labels);
    APP_LOG_INFO("TSFlatIP searchBatchWithShareMasks operation end.\n");
    return APP_ERR_OK;
}


APP_ERROR TSFlatIP::searchBatchWithExtraNonshareMasks(int batch, const uint16_t *x, int topK, float *distances,
    int64_t *labels, AscendTensor<int32_t, DIMS_2> &queryTimes, AscendTensor<uint8_t, DIMS_2> &tokenIds,
    const uint8_t *extraMask, float16_t *extraScore)
{
    APP_LOG_INFO("TSFlatIP searchBatchWithExtraNonshareMasks operation started.\n");
    // 1. generate result variable
    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = resources.getMemoryManager();

    AscendTensor<float16_t, DIMS_2> outDistanceOnDevice(mem, { batch, topK }, stream);
    AscendTensor<int64_t, DIMS_2> outIndicesOnDevice(mem, { batch, topK }, stream);

    // 2. compute distance by code page
    size_t pageNum = utils::divUp(this->ntotal, pageSize);
    for (size_t pageId = 0; pageId < pageNum; ++pageId) {
        size_t pageOffset = pageId * static_cast<size_t>(this->pageSize);
        size_t blockOffset = pageId * static_cast<size_t>(this->pageSize) / static_cast<size_t>(this->blockSize);
        int computeNum = std::min(this->ntotal - pageOffset, static_cast<idx_t>(this->pageSize));
        int blockNum = utils::divUp(computeNum, this->blockSize);
        AscendTensor<uint8_t, DIMS_3> baseMaskDev;
        if (useBaseMask) {
            baseMaskDev = AscendTensor<uint8_t, DIMS_3>(mem,
                { blockNum, 1, utils::divUp(this->blockSize, MASK_ALIGN)}, stream);
            auto ret = aclrtMemcpy(baseMaskDev.data(), baseMaskDev.getSizeInBytes(),
                baseMask.data() + pageOffset / MASK_ALIGN, utils::divUp(computeNum, MASK_ALIGN),
                ACL_MEMCPY_HOST_TO_DEVICE);
            APPERR_RETURN_IF_NOT_FMT(ret == 0, APP_ERR_INNER_ERROR, "copy baseMask failed(%d).", ret);
        }
        AscendTensor<uint8_t, DIMS_3> subMasks(mem,
            { blockNum, this->shareAttrFilter ? 1 : batch, utils::divUp(this->blockSize, MASK_ALIGN) }, stream);

        generateMaskWithExtra(batch, blockOffset, blockNum, queryTimes, tokenIds, extraMask, subMasks, baseMaskDev);
        APP_ERROR ret = searchPagedWithMasks(pageId, pageNum, batch, x, subMasks, outDistanceOnDevice,
            outIndicesOnDevice, extraScore);
        APPERR_RETURN_IF_NOT_FMT(ret == 0, APP_ERR_INNER_ERROR, "searchPagedWithMasks failed(%d).", ret);
    }
    // memcpy data back from dev to host
    std::vector<float16_t> outDistances(batch * topK);
    auto ret = CopyDataToHost(outDistances, labels, outDistanceOnDevice, outIndicesOnDevice);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "CopyDataToHost failed");

    postProcess(batch, topK, outDistances.data(), distances, labels);
    APP_LOG_INFO("TSFlatIP searchBatchWithExtraNonshareMasks operation end.\n");
    return APP_ERR_OK;
}


APP_ERROR TSFlatIP::searchPagedWithMasks(size_t pageId, size_t pageNum, int batch, const uint16_t *x,
    AscendTensor<uint8_t, DIMS_3> &masks, AscendTensor<float16_t, DIMS_2> &maxDistances,
    AscendTensor<int64_t, DIMS_2> &maxIndices, float16_t *extraScore)
{
    using namespace faiss::ascend;
    APP_LOG_INFO("TSFlatIP searchPagedWithMasks operation started.\n");

    auto streamPtr = resources.getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto streamAicpuPtr = resources.getAlternateStreams()[0];
    auto streamAicpu = streamAicpuPtr->GetStream();
    auto &mem = resources.getMemoryManager();
    int k = maxDistances.getSize(1);
    size_t pageOffset = pageId * static_cast<size_t>(this->pageSize);
    size_t blockOffset = pageId * static_cast<size_t>(this->pageSize) / static_cast<size_t>(this->blockSize);
    int computeNum = std::min(this->ntotal - pageOffset, static_cast<idx_t>(this->pageSize));
    int blockNum = utils::divUp(computeNum, this->blockSize);

    AscendTensor<float16_t, DIMS_2> queries(mem, { batch, this->code_size }, stream);
    auto ret = aclrtMemcpy(queries.data(), queries.getSizeInBytes(), x, batch * this->code_size * sizeof(float16_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    FAISS_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Failed to copy to device");
    AscendTensor<float16_t, DIMS_3> distResult(mem, { blockNum, batch, this->blockSize }, stream);
    AscendTensor<float16_t, DIMS_3> maxDistResult(mem, { blockNum, batch, this->burstsOfBlock }, stream);

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
    attrs[aicpu::TOPK_FLAT_ATTR_BURST_LEN_IDX] = BURST_LEN;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_NUM_IDX] = blockNum;
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_IDX] = static_cast<int64_t>(pageId);
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_NUM_IDX] = static_cast<int64_t>(pageNum);
    attrs[aicpu::TOPK_FLAT_ATTR_PAGE_SIZE_IDX] = this->pageSize;
    attrs[aicpu::TOPK_FLAT_ATTR_QUICK_HEAP] = 1;
    attrs[aicpu::TOPK_FLAT_ATTR_BLOCK_SIZE] = this->blockSize;
    uint32_t *opSizeData = reinterpret_cast<uint32_t *>(data + opFlagSize);
    idx_t ntotalPad = utils::divUp(this->ntotal, CUBE_ALIGN) * CUBE_ALIGN;
    for (int i = 0; i < blockNum; ++i) {
        int offset = i * this->blockSize;
        int opSizeHostIdx = i * CORE_NUM * SIZE_ALIGN;
        opSizeData[opSizeHostIdx] =
            std::min(static_cast<uint32_t>(computeNum - offset), static_cast<uint32_t>(this->blockSize));
        opSizeData[opSizeHostIdx + 1] =
            static_cast<uint32_t>(pageOffset) + static_cast<uint32_t>(offset);
        opSizeData[opSizeHostIdx + 2] = ntotalPad; // 2 for extra_score real length
    }
    ret = aclrtMemcpy(continuousMem.data(), continuousMem.getSizeInBytes(), continuousValue.data(),
        continuousValue.size() * sizeof(uint8_t), ACL_MEMCPY_HOST_TO_DEVICE);

    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy attr to device");
    uint16_t *opFlagMem = reinterpret_cast<uint16_t *>(continuousMem.data());
    AscendTensor<uint16_t, DIMS_3> opFlag(opFlagMem, { blockNum, CORE_NUM, FLAG_SIZE });
    uint32_t *opSizeMem = reinterpret_cast<uint32_t *>(continuousMem.data() + opFlagSize);
    AscendTensor<uint32_t, DIMS_3> opSize(opSizeMem, { blockNum, CORE_NUM, SIZE_ALIGN });
    int64_t *attrMem = reinterpret_cast<int64_t *>(continuousMem.data() + opFlagSize + opSizeLen);
    AscendTensor<int64_t, DIMS_1> attrsInput(attrMem, { aicpu::TOPK_FLAT_ATTR_IDX_COUNT });

    // 1. run the topk operator to wait for distance result and compute topk
    runTopkCompute(distResult, maxDistResult, opSize, opFlag, attrsInput, maxDistances, maxIndices, streamAicpu);
    // 2. run the disance operator to compute the distance

    const int dim1 = utils::divUp(this->blockSize, CUBE_ALIGN);
    const int dim2 = utils::divUp(this->code_size, CUBE_ALIGN);

    for (int i = 0; i < blockNum; ++i) {
        auto mask = masks[i].view();
        auto dist = distResult[i].view();
        auto maxDist = maxDistResult[i].view();
        auto actualSize = opSize[i].view();
        auto flag = opFlag[i].view();
        std::vector<const AscendTensorBase *> output { &dist, &maxDist, &flag };
        if (extraScore != nullptr) {
            AscendTensor<float16_t, DIMS_2> extraScoreDev(extraScore, { batch, this->blockSize });
            if (!scale.empty()) {
                int int8Dim2 = utils::divUp(this->code_size, CUBE_ALIGN_INT8);
                AscendTensor<int8_t, DIMS_4> shapedInt8(baseShapedInt8[blockOffset + static_cast<size_t>(i)]->data(),
                    { dim1, int8Dim2, CUBE_ALIGN, CUBE_ALIGN_INT8 });
                std::vector<const AscendTensorBase *> input = { &queries, &shapedInt8,
                    &actualSize, &mask, &extraScoreDev, &scaleReciprocal };
                runDistMaskComputeWithScale(batch, true, input, output, stream);
            } else {
                AscendTensor<float16_t, DIMS_4> shaped(baseShaped[blockOffset + static_cast<size_t>(i)]->data(),
                    { dim1, dim2, CUBE_ALIGN, CUBE_ALIGN });
                std::vector<const AscendTensorBase *> input = { &queries, &shaped, &actualSize, &mask, &extraScoreDev };
                runDistMaskExtraScoreCompute(batch, this->shareAttrFilter, input, output, stream);
            }
        } else {
            if (!scale.empty()) {
                int int8Dim2 = utils::divUp(this->code_size, CUBE_ALIGN_INT8);
                AscendTensor<int8_t, DIMS_4> shapedInt8(baseShapedInt8[blockOffset + static_cast<size_t>(i)]->data(),
                    { dim1, int8Dim2, CUBE_ALIGN, CUBE_ALIGN_INT8 });
                std::vector<const AscendTensorBase *> input = { &queries, &shapedInt8, &actualSize,
                    &mask, &scaleReciprocal };
                runDistMaskComputeWithScale(batch, false, input, output, stream);
            } else {
                AscendTensor<float16_t, DIMS_4> shaped(baseShaped[blockOffset + static_cast<size_t>(i)]->data(),
                    { dim1, dim2, CUBE_ALIGN, CUBE_ALIGN });
                std::vector<const AscendTensorBase *> input = { &queries, &shaped, &actualSize, &mask };
                runDistMaskCompute(batch, this->shareAttrFilter, input, output, stream);
            }
        }
    }
    ret = synchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream default stream failed: %i\n", ret);
    ret = synchronizeStream(streamAicpu);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
        "synchronizeStream aicpu stream failed: %i\n", ret);

    APP_LOG_INFO("TSFlatIP searchPagedWithMasks operation end.\n");
    return APP_ERR_OK;
}

void TSFlatIP::postProcess(int64_t searchNum, int topK, float16_t *outDistances, float *distances, int64_t *labels)
{
    APP_LOG_INFO("TSFlatIP postProcess operation started.\n");

    for (int64_t i = 0; i < searchNum; ++i) {
        std::transform(outDistances + i * topK, outDistances + (i + 1) * topK, distances + i * topK,
            [&](const float16_t temp) -> float { return static_cast<float>(faiss::ascend::fp16(temp)); });
        std::transform(labels + i * topK, labels + (i + 1) * topK, labels + i * topK,
            [&](const int64_t temp) -> int64_t { return temp == -1 ? -1 : static_cast<int64_t>(ids.at(temp)); });
    }
    APP_LOG_INFO("TSFlatIP postProcess operation end.\n");
}

APP_ERROR TSFlatIP::runSharedAttrFilter(int32_t queryNum, uint32_t topk, const faiss::ascend::AttrFilter *attrFilter,
                                        const uint8_t *extraMask, const float *queryFeatures, int64_t *labels,
                                        float *distances)
{
    using namespace faiss::ascend;
    int64_t offset = 0;
    int blockNum = static_cast<int>(utils::divUp(this->ntotal, this->blockSize));
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = pResources->getMemoryManager();
    AscendTensor<int32_t, DIMS_2> queryTimes(mem, { 1, OPS_DATA_TYPE_ALIGN }, stream);
    AscendTensor<uint8_t, DIMS_2> tokenIds(mem,
        { 1, static_cast<int32_t>(utils::divUp(tokenNum, OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES) }, stream);
    AscendTensor<uint8_t, DIMS_3> masks(mem, { blockNum, 1, utils::divUp(this->blockSize, MASK_ALIGN) }, stream);
    buildAttr(attrFilter, 1, queryTimes, tokenIds);
    generateMaskWithExtra(1, 0, blockNum, queryTimes, tokenIds, extraMask, masks);
    for (auto batch : searchBatchSizes) {
        while (queryNum >= batch) {
            std::vector<uint16_t> query(batch * this->code_size, 0);
            transform(queryFeatures + offset * this->code_size, queryFeatures + (offset + batch) * this->code_size,
                begin(query), [](float temp) { return fp16(temp).data; });
            APP_ERROR ret = searchBatchWithShareMasks(batch, query.data(), static_cast<int>(topk),
                distances + offset * topk, labels + offset * topk, masks);
            APPERR_RETURN_IF_NOT_FMT(ret == 0, APP_ERR_INNER_ERROR, "searchBatchWithShareMasks failed(%d).", ret);
            offset += batch;
            queryNum -= batch;
        }
    }
    return APP_ERR_OK;
}

APP_ERROR TSFlatIP::runNonSharedAttrFilter(int32_t queryNum, uint32_t topk, uint64_t extraMaskLen,
                                           const faiss::ascend::AttrFilter *attrFilter, const uint8_t *extraMask,
                                           const float *queryFeatures, const float16_t *extraScore, int64_t *labels,
                                           float *distances)
{
    using namespace faiss::ascend;
    int64_t offset = 0;
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = pResources->getMemoryManager();
    for (auto batch : searchBatchSizes) {
        while (queryNum >= batch) {
            std::vector<uint16_t> query(batch * this->code_size, 0);
            transform(queryFeatures + offset * this->code_size, queryFeatures + (offset + batch) * this->code_size,
                begin(query), [](float temp) { return fp16(temp).data; });
            AscendTensor<int32_t, DIMS_2> queryTimes(mem, { batch, OPS_DATA_TYPE_ALIGN }, stream);
            AscendTensor<uint8_t, DIMS_2> tokenIds(mem,
                { batch, static_cast<int32_t>(utils::divUp(tokenNum, OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES) },
                stream);
            buildAttr(attrFilter + offset, batch, queryTimes, tokenIds);
            AscendTensor<float16_t, DIMS_2> extraScoreDev;
            if (extraScore != nullptr) {
                idx_t ntotalPad = utils::divUp(this->ntotal, CUBE_ALIGN) * CUBE_ALIGN;
                extraScoreDev = AscendTensor<float16_t, DIMS_2>(mem, {batch, static_cast<int>(ntotalPad)}, stream);
                auto ret = aclrtMemcpy(extraScoreDev.data(), extraScoreDev.getSizeInBytes(),
                    extraScore + static_cast<idx_t>(offset) * ntotalPad,
                    static_cast<idx_t>(batch) * ntotalPad * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
                APPERR_RETURN_IF_NOT_FMT(ret == 0, APP_ERR_INNER_ERROR, "copy extra score to device failed %d", ret);
            }
            APP_ERROR ret = searchBatchWithExtraNonshareMasks(batch, query.data(), static_cast<int>(topk),
                distances + offset * topk, labels + offset * topk, queryTimes, tokenIds,
                extraMask + offset * extraMaskLen, extraScoreDev.data());
            APPERR_RETURN_IF_NOT_FMT(ret == 0, APP_ERR_INNER_ERROR,
                "searchBatchWithExtraNonshareMasks failed(%d).", ret);
            offset += batch;
            queryNum -= batch;
        }
    }
    return APP_ERR_OK;
}

APP_ERROR TSFlatIP::searchWithExtraMask(uint32_t count, const void *features,
    const faiss::ascend::AttrFilter *attrFilter, bool shareAttrFilter, uint32_t topk, const uint8_t *extraMask,
    uint64_t extraMaskLen, bool extraMaskIsAtDevice, int64_t *labels, float *distances, uint32_t *validNums,
    bool enableTimeFilter, const float16_t *extraScore)
{
    using namespace faiss::ascend;
    APP_LOG_INFO("TSFlatIP searchWithExtraMask operation started.\n");

    APPERR_RETURN_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE, APP_ERR_ACL_BAD_ALLOC);
    this->shareAttrFilter = shareAttrFilter;
    this->enableTimeFilter = enableTimeFilter;
    this->extraMaskIsAtDevice = extraMaskIsAtDevice;
    this->extraMaskLen = extraMaskLen;

    int32_t queryNum = static_cast<int32_t>(count);
    const float *queryFeatures = reinterpret_cast<const float *>(features);

    if (this->shareAttrFilter) {
        APPERR_RETURN_IF_NOT_LOG(extraScore == nullptr,
            APP_ERR_INVALID_PARAM, "TSFlatIP is not suport Extra Score when shareAttrFilter is true!");
        APPERR_RETURN_IF_NOT_LOG(scale.empty(),
            APP_ERR_INVALID_PARAM, "TSFlatIP is not suport scale when shareAttrFilter is true!");
        APP_ERROR ret = runSharedAttrFilter(queryNum, topk, attrFilter, extraMask, queryFeatures, labels, distances);
        APPERR_RETURN_IF_NOT_FMT(ret == 0, APP_ERR_INNER_ERROR, "runSharedAttrFilter failed(%d).", ret);
    } else {
        APP_ERROR ret = runNonSharedAttrFilter(queryNum, topk, extraMaskLen, attrFilter, extraMask,
                                               queryFeatures, extraScore, labels, distances);
        APPERR_RETURN_IF_NOT_FMT(ret == 0, APP_ERR_INNER_ERROR, "runNonSharedAttrFilter failed(%d).", ret);
    }

    getValidNum(count, topk, labels, validNums);

    APP_LOG_INFO("TSFlatIP searchWithExtraMask operation operation end.\n");
    return APP_ERR_OK;
}

void TSFlatIP::add_with_ids(idx_t n, const float *x, const idx_t *xids)
{
    APP_LOG_INFO("TSFlatIP add_with_ids operation started.\n");
    ids.insert(ids.end(), xids, xids + n);
    size_t offset = 0;
    size_t addTotal = n;
    size_t singleAddMax = static_cast<size_t>(utils::divDown(UPPER_LIMIT_FOR_ADD, this->blockSize) * this->blockSize);
    while (addTotal > 0) {
        auto singleAdd = std::min(addTotal, singleAddMax);
        std::vector<uint16_t> xb(singleAdd * static_cast<uint32_t>(this->code_size));
        transform(x + offset * this->code_size, x + (offset + singleAdd) * this->code_size, std::begin(xb),
            [](float temp) { return faiss::ascend::fp16(temp).data; });
        addWithIdsImpl(singleAdd, xb.data());
        offset += singleAdd;
        addTotal -= singleAdd;
    }
    APP_LOG_INFO("TSFlatIP add_with_ids operation end.\n");
}

void TSFlatIP::addWithIdsImpl(int n, uint16_t *x)
{
    APP_LOG_INFO("TSFlatIP addWithIdsImpl operation started.\n");

    AscendTensor<float16_t, DIMS_2> rawTensor(x, { n, this->code_size });
    auto res = addVectors(rawTensor);
    FAISS_THROW_IF_NOT_FMT(res == APP_ERR_OK, "add vector failed, error code:%d", res);
    APP_LOG_INFO("TSFlatIP addWithIdsImpl operation end.\n");
}

void TSFlatIP::queryVectorByIdx(int64_t idx, float *dis) const
{
    int reshapeDim2 = utils::divUp(this->code_size, CUBE_ALIGN);
    size_t total = static_cast<size_t>(idx);
    size_t offsetInBlock = total % static_cast<size_t>(this->blockSize);
    size_t blockIdx = total / static_cast<size_t>(this->blockSize);
    // we can make sure the size of offsetInblock is small
    int hoffset1 = static_cast<int>(offsetInBlock) / CUBE_ALIGN;
    int hoffset2 = static_cast<int>(offsetInBlock) % CUBE_ALIGN;
    int disOffset = 0;
    int srcOffset = hoffset1 * code_size * CUBE_ALIGN + hoffset2 * CUBE_ALIGN;
    std::vector<uint16_t> xb(this->code_size);
    for (int i = 0; i < reshapeDim2; ++i) {
        auto ret = aclrtMemcpy(xb.data() + disOffset, CUBE_ALIGN * sizeof(uint16_t),
            baseShaped[blockIdx]->data() + srcOffset, CUBE_ALIGN * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
        FAISS_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Failed to copy to host");
        disOffset += CUBE_ALIGN;
        srcOffset += CUBE_ALIGN * CUBE_ALIGN;
    }

    std::transform(xb.data(), xb.data() + this->code_size, dis,
        [&](const uint16_t temp) -> float { return (float)faiss::ascend::fp16(temp); });
}

// 获取单条的量化后的特征，需要注意按照CUBE_ALIGN_INT8对齐的
void TSFlatIP::queryInt8VectorByIdx(int64_t idx, float *dis) const
{
    int reshapeDim2 = utils::divUp(this->code_size, CUBE_ALIGN_INT8);
    size_t total = static_cast<size_t>(idx);
    size_t offsetInBlock = total % static_cast<size_t>(this->blockSize);
    size_t blockIdx = total / static_cast<size_t>(this->blockSize);
    // we can make sure the size of offsetInblock is small
    int hoffset1 = static_cast<int>(offsetInBlock) / CUBE_ALIGN;
    int hoffset2 = static_cast<int>(offsetInBlock) % CUBE_ALIGN;
    int disOffset = 0;
    int srcOffset = hoffset1 * code_size * CUBE_ALIGN + hoffset2 * CUBE_ALIGN_INT8;
    std::vector<int8_t> xb(this->code_size);
    for (int i = 0; i < reshapeDim2; ++i) {
        auto ret = aclrtMemcpy(xb.data() + disOffset, CUBE_ALIGN_INT8 * sizeof(int8_t),
            baseShapedInt8[blockIdx]->data() + srcOffset, CUBE_ALIGN_INT8 * sizeof(int8_t),
            ACL_MEMCPY_DEVICE_TO_HOST);
        FAISS_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Failed to copy to host");
        disOffset += CUBE_ALIGN_INT8;
        srcOffset += CUBE_ALIGN_INT8 * CUBE_ALIGN;
    }

    for (int i = 0; i < this->code_size; i++) {
        dis[i] = static_cast<float>(xb[i]) / scale[i];
    }
}

APP_ERROR TSFlatIP::SetScale(const std::vector<float> &scale)
{
    this->scale = scale;
    std::vector<float16_t> scaleReciprocalHost(dims);
    // 计算倒数，保存成fp16。算子不计算除，直接计算乘。
    std::transform(scale.begin(), scale.end(), scaleReciprocalHost.begin(),
        [](float temp) { return faiss::ascend::fp16(1.0f / temp).data; });

    // 数据量较小，直接保存到device侧
    AscendTensor<float16_t, DIMS_1> scaleReciprocalDevice({ dims });
    auto ret = aclrtMemcpy(scaleReciprocalDevice.data(), scaleReciprocalDevice.getSizeInBytes(),
        scaleReciprocalHost.data(), scaleReciprocalHost.size() * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_FMT(ret != ACL_SUCCESS, ret, "copy scale to device failed: %d", ret);
    scaleReciprocal = std::move(scaleReciprocalDevice);
    return APP_ERR_OK;
}

void TSFlatIP::removeLabels(const std::vector<int64_t> &removeIds)
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
} // namespace ascend