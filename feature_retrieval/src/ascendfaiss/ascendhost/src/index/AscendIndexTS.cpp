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


#include "ascendhost/include/index/AscendIndexTS.h"

#include "ascendhost/include/impl/TSBinaryFlat.h"
#include "ascendhost/include/impl/TSFlatIP.h"
#include "ascendhost/include/impl/TSInt8FlatCos.h"
#include "ascendhost/include/impl/TSInt8FlatL2.h"
#include "ascendhost/include/impl/TSInt8FlatHPPCosFactory.h"
using namespace ascend;

namespace faiss {
namespace ascend {

APP_ERROR AscendIndexTS::Init(uint32_t deviceId, uint32_t dim, uint32_t tokenNum, AlgorithmType algType,
    MemoryStrategy memoryStrategy, uint32_t customAttrLen, uint32_t customAttrBlockSize, uint64_t maxFeatureRowCount)
{
    return InitWithExtraVal(deviceId, dim, tokenNum, DEFAULT_RESOURCE_SIZE, algType, memoryStrategy, customAttrLen,
        customAttrBlockSize, maxFeatureRowCount);
}

static APP_ERROR CheckInitParams(uint32_t deviceId, uint32_t tokenNum, uint32_t customAttrLen, uint32_t customAttrBlockSize)
{
    // 1. deviceId 越界检查
    APPERR_RETURN_IF_NOT_FMT(deviceId <= UPPER_LIMIT_FOR_DEVICEID, APP_ERR_INVALID_PARAM,
                             "deviceId must be >= 0 and <= %ld", UPPER_LIMIT_FOR_DEVICEID);

    // 2. tokenNum 越界检查
    APPERR_RETURN_IF_NOT_FMT(tokenNum > 0 && tokenNum <= UPPER_LIMIT_FOR_TOKENNUM, APP_ERR_INVALID_PARAM,
                             "tokenNum must be > 0 and <= %ld", UPPER_LIMIT_FOR_TOKENNUM);

    // 3. customAttrLen 越界检查
    APPERR_RETURN_IF_NOT_FMT(customAttrLen <= UPPER_LIMIT_FOR_CUSTOM_ATTR_LEN,
        APP_ERR_INVALID_PARAM, "customAttrLen must be >= 0 and <= %lu", UPPER_LIMIT_FOR_CUSTOM_ATTR_LEN);

    // 4. customAttrBlockSize 越界检查
    APPERR_RETURN_IF_NOT_FMT(customAttrBlockSize <= UPPER_LIMIT_FOR_CUSTOM_ATTR_BLOCK_SIZE,
        APP_ERR_INVALID_PARAM, "customAttrBlockSize must be >= 0 and <= %lu", UPPER_LIMIT_FOR_CUSTOM_ATTR_BLOCK_SIZE);

    // 5. customAttrLen 和 customAttrBlockSize 对齐检查
    APPERR_RETURN_IF_NOT_FMT(
        (customAttrLen > 0 && customAttrBlockSize > 0 && customAttrBlockSize % DEFAULT_FEATURE_BLOCK_SIZE == 0) ||
        (customAttrLen == 0 && customAttrBlockSize == 0), APP_ERR_INVALID_PARAM,
        "customAttrLen[%u] and customAttrBlockSize[%u] must be >= 0 and customAttrBlockSize multiple of [%u]",
        customAttrLen, customAttrBlockSize, DEFAULT_FEATURE_BLOCK_SIZE);

    return APP_ERR_OK;
}

APP_ERROR AscendIndexTS::InitWithExtraVal(uint32_t deviceId, uint32_t dim, uint32_t tokenNum, uint64_t resources,
    AlgorithmType algType, MemoryStrategy memoryStrategy, uint32_t customAttrLen, uint32_t customAttrBlockSize,
    uint64_t maxFeatureRowCount)
{
    auto ret = CheckInitParams(deviceId, tokenNum, customAttrLen, customAttrBlockSize);
    APPERR_RETURN_IF(ret, ret);

    APPERR_RETURN_IF_FMT(memoryStrategy == MemoryStrategy::HETERO_MEMORY && algType != AlgorithmType::FLAT_COS_INT8,
        APP_ERR_INVALID_PARAM, "MemoryStrategy %d is not support AlgorithmType %d", memoryStrategy, algType);
    std::lock(AscendGlobalLock::GetInstance(deviceId), mtx);
    std::lock_guard<std::mutex> glck(AscendGlobalLock::GetInstance(deviceId), std::adopt_lock);
    std::lock_guard<std::mutex> lck(mtx, std::adopt_lock);
    APPERR_RETURN_IF_NOT_LOG(pImpl == nullptr, APP_ERR_ILLEGAL_OPERATION, "already initialized, cannot init twice");
    this->deviceId = deviceId;
    this->maxTokenNum = tokenNum;
    this->memoryStrategy = memoryStrategy;
    try {
        if (algType == AlgorithmType::FLAT_HAMMING) {
            pImpl =
                std::make_shared<TSBinaryFlat>(deviceId, dim, tokenNum, resources, customAttrLen, customAttrBlockSize);
        } else if (algType == AlgorithmType::FLAT_COS_INT8) {
            // IndexInt8 constructor need set device first
            ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(deviceId), APP_ERR_ACL_SET_DEVICE_FAILED);
            pImpl = std::make_shared<TSInt8FlatCos>(deviceId, dim, tokenNum, resources,
                memoryStrategy == MemoryStrategy::HETERO_MEMORY, customAttrLen, customAttrBlockSize);
        } else if (algType == AlgorithmType::FLAT_IP_FP16) {
            // IndexFlat constructor need set device first
            ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(deviceId), APP_ERR_ACL_SET_DEVICE_FAILED);
            pImpl = std::make_shared<TSFlatIP>(deviceId, dim, tokenNum, resources, customAttrLen, customAttrBlockSize);
        } else if (algType == AlgorithmType::FLAT_L2_INT8) {
            ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(deviceId), APP_ERR_ACL_SET_DEVICE_FAILED);
            pImpl = std::make_shared<TSInt8FlatL2>(deviceId, dim, tokenNum, resources,
                customAttrLen, customAttrBlockSize);
        } else if (algType == AlgorithmType::FLAT_HPP_COS_INT8) {
            pImpl = TSInt8FlatHPPCosFactory::Create(dim, deviceId, tokenNum, maxFeatureRowCount, customAttrLen,
                customAttrBlockSize);
        } else {
            return APP_ERR_NOT_IMPLEMENT;
        }
    } catch (std::exception &e) {
        APPERR_RETURN_IF_NOT_FMT(false, APP_ERR_INNER_ERROR, "AscendIndexTS init failed, error msg[%s]", e.what());
    }
    return APP_ERR_OK;
}

APP_ERROR AscendIndexTS::InitWithQuantify(uint32_t deviceId, uint32_t dim, uint32_t tokenNum, uint64_t resources,
    const float *scale, AlgorithmType algType, uint32_t customAttrLen, uint32_t customAttrBlockSize)
{
    auto ret = CheckInitParams(deviceId, tokenNum, customAttrLen, customAttrBlockSize);
    APPERR_RETURN_IF(ret, ret);

    APPERR_RETURN_IF_NOT_FMT(resources > 0 && resources <= MAX_MEM, APP_ERR_INVALID_PARAM,
        "resources must be > 0 and <= %ld", MAX_MEM);

    APPERR_RETURN_IF_NOT_LOG(scale != nullptr, APP_ERR_INVALID_PARAM, "scale can not be nullptr.");
    // 反量化时需要进行除运算，因此不能太小。
    for (uint32_t i = 0; i < dim; i++) {
        APPERR_RETURN_IF_FMT(std::fabs(scale[i]) < 1e-6f, APP_ERR_INVALID_PARAM,
            "scale[%u]:%f cannot be close to 0.", i, scale[i]);
    }

    std::lock(AscendGlobalLock::GetInstance(deviceId), mtx);
    std::lock_guard<std::mutex> glck(AscendGlobalLock::GetInstance(deviceId), std::adopt_lock);
    std::lock_guard<std::mutex> lck(mtx, std::adopt_lock);
    APPERR_RETURN_IF_NOT_LOG(pImpl == nullptr, APP_ERR_ILLEGAL_OPERATION, "already initialized, cannot init twice");
    this->maxTokenNum = tokenNum;
    this->deviceId = deviceId;
    try {
        if (algType == AlgorithmType::FLAT_IP_FP16) {
            // IndexFlat constructor need set device first
            ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(deviceId), APP_ERR_ACL_SET_DEVICE_FAILED);
            std::vector<float> scaleVector(scale, scale + dim);
            pImpl = std::make_shared<TSFlatIP>(deviceId, dim, tokenNum, resources, customAttrLen, customAttrBlockSize,
                scaleVector);
        } else {
            APPERR_RETURN_IF_NOT_LOG(false, APP_ERR_NOT_IMPLEMENT, "InitWithQuantify only support FLAT_IP_FP16");
        }
    } catch (std::exception &e) {
        APPERR_RETURN_IF_NOT_FMT(false, APP_ERR_INNER_ERROR, "AscendIndexTS init failed, error msg[%s]", e.what());
    }
    return APP_ERR_OK;
}

APP_ERROR AscendIndexTS::SetHeteroParam(size_t deviceCapacity, size_t deviceBuffer, size_t hostCapacity)
{
    std::lock(AscendGlobalLock::GetInstance(deviceId), mtx);
    std::lock_guard<std::mutex> glck(AscendGlobalLock::GetInstance(deviceId), std::adopt_lock);
    std::lock_guard<std::mutex> lck(mtx, std::adopt_lock);

    APPERR_RETURN_IF_NOT_FMT(memoryStrategy == MemoryStrategy::HETERO_MEMORY, APP_ERR_ILLEGAL_OPERATION,
        "memoryStrategy %d is not support this func.", memoryStrategy);
    APPERR_RETURN_IF_NOT_LOG(pImpl != nullptr, APP_ERR_ILLEGAL_OPERATION, "pImpl can not be nullptr.");
    TSInt8FlatCos *impl = dynamic_cast<TSInt8FlatCos *>(pImpl.get());
    APPERR_RETURN_IF_NOT_LOG(impl != nullptr, APP_ERR_ILLEGAL_OPERATION, "only FLAT_COS_INT8 support this func.");
    APPERR_RETURN_IF_NOT_LOG(!heteroParamSetFlag, APP_ERR_ILLEGAL_OPERATION, "can't set hetero param twice.");

    auto ret = impl->setHeteroParam(deviceId, deviceCapacity, deviceBuffer, hostCapacity);
    if (ret == APP_ERR_OK) {
        heteroParamSetFlag = true;
    }
    return ret;
}

APP_ERROR AscendIndexTS::AddFeature(int64_t count, const void *features, const FeatureAttr *attributes,
                                    const int64_t *labels, const uint8_t *customAttr)
{
    return AddWithExtraVal(count, features, attributes, labels, nullptr, customAttr);
}

APP_ERROR AscendIndexTS::AddWithExtraVal(int64_t count, const void *features, const FeatureAttr *attributes,
    const int64_t *labels, const ExtraValAttr *extraVal, const uint8_t *customAttr)
{
    APPERR_RETURN_IF_NOT_LOG(features != nullptr, APP_ERR_ILLEGAL_OPERATION, "features can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(attributes != nullptr, APP_ERR_ILLEGAL_OPERATION, "attributes can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(labels != nullptr, APP_ERR_ILLEGAL_OPERATION, "labels can not be nullptr.");
    std::lock(AscendGlobalLock::GetInstance(this->deviceId), mtx);
    std::lock_guard<std::mutex> glck(AscendGlobalLock::GetInstance(this->deviceId), std::adopt_lock);
    std::lock_guard<std::mutex> lck(mtx, std::adopt_lock);
    APPERR_RETURN_IF_NOT(pImpl != nullptr, APP_ERR_ILLEGAL_OPERATION);
    APPERR_RETURN_IF_NOT_FMT(count > 0 && count <= UPPER_LIMIT_FOR_ADD &&
                             count + pImpl->getAttrTotal() <= UPPER_LIMIT_FOR_NTOTAL,
                             APP_ERR_INVALID_PARAM, "count must be > 0 and <= %ld", UPPER_LIMIT_FOR_ADD);
    if (memoryStrategy == MemoryStrategy::HETERO_MEMORY) {
        APPERR_RETURN_IF_NOT_LOG(heteroParamSetFlag, APP_ERR_ILLEGAL_OPERATION, "should set hetero param first.");
    }
    APP_ERROR ret = APP_ERR_OK;
    ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(this->deviceId), APP_ERR_ACL_SET_DEVICE_FAILED);
    try {
        ret = pImpl->addFeatureWithLabels(count, features, attributes, labels, customAttr, extraVal);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_NOT_FMT(false, APP_ERR_INNER_ERROR,
            "AddFeature failed, please check parameters or obtain detail information from logs, error msg[%s]",
            e.what());
    }
    return ret;
}

APP_ERROR AscendIndexTS::AddFeatureByIndice(int64_t count, const void *features, const FeatureAttr *attributes,
    const int64_t *indices, const ExtraValAttr *extraVal, const uint8_t *customAttr)
{
    APPERR_RETURN_IF_NOT_LOG(features != nullptr, APP_ERR_INVALID_PARAM, "features can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(attributes != nullptr, APP_ERR_INVALID_PARAM, "attributes can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(indices != nullptr, APP_ERR_INVALID_PARAM, "indices can not be nullptr.");
    std::lock(AscendGlobalLock::GetInstance(this->deviceId), mtx);
    std::lock_guard<std::mutex> glck(AscendGlobalLock::GetInstance(this->deviceId), std::adopt_lock);
    std::lock_guard<std::mutex> lck(mtx, std::adopt_lock);
    APPERR_RETURN_IF_NOT(pImpl != nullptr, APP_ERR_ILLEGAL_OPERATION);
    APPERR_RETURN_IF_NOT_FMT(count > 0 && count <= UPPER_LIMIT_FOR_ADD &&
                             count + pImpl->getAttrTotal() <= UPPER_LIMIT_FOR_NTOTAL,
                             APP_ERR_INVALID_PARAM, "count must be > 0 and <= %ld", UPPER_LIMIT_FOR_ADD);
    APPERR_RETURN_IF_NOT_FMT(indices[count - 1] < UPPER_LIMIT_FOR_NTOTAL,
                             APP_ERR_INVALID_PARAM, "indice must be < %ld", UPPER_LIMIT_FOR_NTOTAL);
    if (memoryStrategy == MemoryStrategy::HETERO_MEMORY) {
        APPERR_RETURN_IF_NOT_LOG(heteroParamSetFlag, APP_ERR_ILLEGAL_OPERATION, "should set hetero param first.");
    }
    APP_ERROR ret = APP_ERR_OK;
    ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(this->deviceId), APP_ERR_ACL_SET_DEVICE_FAILED);
    try {
        ret = pImpl->AddFeatureByIndice(count, features, attributes, indices, customAttr, extraVal);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_NOT_FMT(false, APP_ERR_INNER_ERROR, "AddFeatureByIndice failed %s", e.what());
    }
    return ret;
}

APP_ERROR AscendIndexTS::GetFeatureByIndice(int64_t count, const int64_t *indices,
    int64_t *labels, void *features, FeatureAttr *attributes, ExtraValAttr *extraVal) const
{
    APPERR_RETURN_IF_NOT_LOG(indices != nullptr, APP_ERR_INVALID_PARAM, "indices can not be nullptr.");
    std::lock_guard<std::mutex> lck(mtx);
    APPERR_RETURN_IF_NOT(pImpl != nullptr, APP_ERR_ILLEGAL_OPERATION);
    APPERR_RETURN_IF_NOT_FMT(count > 0 && count <= UPPER_LIMIT_FOR_ADD && count <= pImpl->getAttrTotal(),
                             APP_ERR_INVALID_PARAM, "count must be > 0 and <= %ld",
                             std::min(UPPER_LIMIT_FOR_ADD, pImpl->getAttrTotal()));
    for (int64_t i = 0; i < count; i++) {
        APPERR_RETURN_IF_NOT_FMT(indices[i] >= 0 && indices[i] < pImpl->getAttrTotal(),
                                 APP_ERR_INVALID_PARAM, "indice must be >= 0 and < %ld", pImpl->getAttrTotal());
    }
    APP_ERROR ret = APP_ERR_OK;
    ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(this->deviceId), APP_ERR_ACL_SET_DEVICE_FAILED);
    try {
        ret = pImpl->GetFeatureByIndice(count, indices, labels, features, attributes, extraVal);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_NOT_FMT(false, APP_ERR_INNER_ERROR, "GetFeatureByIndice failed: %s", e.what());
    }
    return ret;
}

APP_ERROR AscendIndexTS::FastDeleteFeatureByIndice(int64_t count, const int64_t *indices)
{
    APPERR_RETURN_IF_NOT_LOG(indices != nullptr, APP_ERR_INVALID_PARAM, "indices can not be nullptr.");
    std::lock(AscendGlobalLock::GetInstance(this->deviceId), mtx);
    std::lock_guard<std::mutex> glck(AscendGlobalLock::GetInstance(this->deviceId), std::adopt_lock);
    std::lock_guard<std::mutex> lck(mtx, std::adopt_lock);
    APPERR_RETURN_IF_NOT(pImpl != nullptr, APP_ERR_ILLEGAL_OPERATION);
    APPERR_RETURN_IF_NOT_LOG(memoryStrategy == MemoryStrategy::PURE_DEVICE_MEMORY, APP_ERR_ILLEGAL_OPERATION,
        "FastDeleteFeatureByIndice failed !! only memoryStrategy PURE_DEVICE_MEMORY is support.");
    APPERR_RETURN_IF_NOT_FMT(count > 0 && count <= pImpl->getAttrTotal(), APP_ERR_INVALID_PARAM,
                             "count must be > 0 and <= %ld", pImpl->getAttrTotal());
    for (int64_t i = 0; i < count; i++) {
        APPERR_RETURN_IF_NOT_FMT(indices[i] >= 0 && indices[i] < pImpl->getAttrTotal(), APP_ERR_INVALID_PARAM,
                                 "indice must be >= 0 and < %ld, but indices[%ld] is %ld",
                                 pImpl->getAttrTotal(), i, indices[i]);
    }
    APP_ERROR ret = APP_ERR_OK;
    try {
        ret = pImpl->FastDeleteFeatureByIndice(count, indices);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_NOT_FMT(false, APP_ERR_INNER_ERROR,
            "FastDeleteFeatureByIndice failed, please check parameters or obtain detail information from logs, "
                "error msg[%s]", e.what());
    }
    return ret;
}

APP_ERROR AscendIndexTS::FastDeleteFeatureByRange(int64_t start, int64_t count)
{
    std::lock(AscendGlobalLock::GetInstance(this->deviceId), mtx);
    std::lock_guard<std::mutex> glck(AscendGlobalLock::GetInstance(this->deviceId), std::adopt_lock);
    std::lock_guard<std::mutex> lck(mtx, std::adopt_lock);
    APPERR_RETURN_IF_NOT(pImpl != nullptr, APP_ERR_ILLEGAL_OPERATION);
    APPERR_RETURN_IF_NOT_LOG(memoryStrategy == MemoryStrategy::PURE_DEVICE_MEMORY, APP_ERR_ILLEGAL_OPERATION,
        "FastDeleteFeatureByRange failed !! only memoryStrategy PURE_DEVICE_MEMORY is support.");
    APPERR_RETURN_IF_NOT_FMT(start >= 0 && start < pImpl->getAttrTotal(), APP_ERR_INVALID_PARAM,
                             "start must be >= 0 and < %ld, but start is %ld",
                             pImpl->getAttrTotal(), start);
    APPERR_RETURN_IF_NOT_FMT(count > 0 && count <= pImpl->getAttrTotal(), APP_ERR_INVALID_PARAM,
                             "count must be > 0 and <= %ld", pImpl->getAttrTotal());
    APPERR_RETURN_IF_NOT_FMT(start + count <= pImpl->getAttrTotal(), APP_ERR_INVALID_PARAM,
                             "start + count must be <= %ld, but start + count is %ld",
                             pImpl->getAttrTotal(), start + count);
    APP_ERROR ret = APP_ERR_OK;
    try {
        ret = pImpl->FastDeleteFeatureByRange(start, count);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_NOT_FMT(false, APP_ERR_INNER_ERROR,
            "FastDeleteFeatureByRange failed, please check parameters or obtain detail information from logs, "
                "error msg[%s]", e.what());
    }
    return ret;
}

APP_ERROR AscendIndexTS::GetBaseMask(int64_t count, uint8_t *mask) const
{
    APPERR_RETURN_IF_NOT_LOG(mask != nullptr, APP_ERR_INVALID_PARAM, "mask can not be nullptr.");
    std::lock(AscendGlobalLock::GetInstance(this->deviceId), mtx);
    std::lock_guard<std::mutex> glck(AscendGlobalLock::GetInstance(this->deviceId), std::adopt_lock);
    std::lock_guard<std::mutex> lck(mtx, std::adopt_lock);
    APPERR_RETURN_IF_NOT(pImpl != nullptr, APP_ERR_ILLEGAL_OPERATION);
    auto baseMask = pImpl->GetBaseMask();
    APPERR_RETURN_IF_NOT_FMT(count > 0 && count <= static_cast<int64_t>(baseMask.size()), APP_ERR_INVALID_PARAM,
                             "count must be > 0 and <= %ld", static_cast<int64_t>(baseMask.size()));
    for (int64_t i = 0; i < count; i++) {
        mask[i] = baseMask[i];
    }
    return APP_ERR_OK;
}

APP_ERROR AscendIndexTS::Search(uint32_t count, const void *features, const AttrFilter *attrFilter,
                                bool shareAttrFilter, uint32_t topk, int64_t *labels, float *distances,
                                uint32_t *validNums, bool enableTimeFilter)
{
    return SearchWithExtraVal(count, features, attrFilter, shareAttrFilter, topk, labels, distances, validNums,
        nullptr, enableTimeFilter);
}

APP_ERROR AscendIndexTS::SearchWithExtraVal(uint32_t count, const void *features, const AttrFilter *attrFilter,
    bool shareAttrFilter, uint32_t topk, int64_t *labels, float *distances,
    uint32_t *validNums, const ExtraValFilter *extraValFilter, bool enableTimeFilter)
{
    APPERR_RETURN_IF_NOT_FMT((count > 0) && (count <= TS_MAX_SEARCH), APP_ERR_INVALID_PARAM,
                             "count must be > 0 and <= %ld", TS_MAX_SEARCH);
    APPERR_RETURN_IF_NOT_FMT((topk > 0) && (topk <= TS_MAX_TOPK), APP_ERR_INVALID_PARAM, "topk must be > 0 and <= %ld",
                             TS_MAX_TOPK);
    APPERR_RETURN_IF_NOT_LOG(features != nullptr, APP_ERR_ILLEGAL_OPERATION, "features can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(attrFilter != nullptr, APP_ERR_ILLEGAL_OPERATION, "attrFilter can not be nullptr.");
    uint32_t attrFilterNum = shareAttrFilter ? 1 : count;  // shareAttrFilter为true时，共享filter，此时filter只有1个
    for (uint32_t i = 0; i < attrFilterNum; i++) {
        APPERR_RETURN_IF_NOT_LOG((attrFilter + i)->tokenBitSet != nullptr, APP_ERR_INVALID_PARAM,
            "tokenBitSet can not be nullptr");
        APPERR_RETURN_IF_NOT_FMT((attrFilter + i)->tokenBitSetLen > 0 &&
            (attrFilter + i)->tokenBitSetLen <= utils::divUp(maxTokenNum, TOKEN_SET_BIT),
            APP_ERR_INVALID_PARAM, "attrFilter[%u] tokenBitSetLen[%u] must be > 0 and <= %u", i,
            (attrFilter + i)->tokenBitSetLen, utils::divUp(maxTokenNum, TOKEN_SET_BIT));
        if (extraValFilter != nullptr) {
            ASCEND_THROW_IF_NOT_FMT((extraValFilter + i)->matchVal == 1 || (extraValFilter + i)->matchVal == 0,
                "Only mode 0 and mode 1 are supported for extra val. matchVal:%d\n", (extraValFilter + i)->matchVal);
        }
    }
    APPERR_RETURN_IF_NOT_LOG(labels != nullptr, APP_ERR_ILLEGAL_OPERATION, "labels can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(distances != nullptr, APP_ERR_ILLEGAL_OPERATION, "distances can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(validNums != nullptr, APP_ERR_ILLEGAL_OPERATION, "validNums can not be nullptr.");

    bool extraValWithSharedFilter = (extraValFilter != nullptr) && shareAttrFilter;
    APPERR_RETURN_IF_FMT(extraValWithSharedFilter, APP_ERR_ILLEGAL_OPERATION,
        "ExtraVal support only non-shared filter, please check shareAttrFilter:%d", shareAttrFilter);

    std::lock(AscendGlobalLock::GetInstance(this->deviceId), mtx);
    std::lock_guard<std::mutex> glck(AscendGlobalLock::GetInstance(this->deviceId), std::adopt_lock);
    std::lock_guard<std::mutex> lck(mtx, std::adopt_lock);
    APPERR_RETURN_IF_NOT_LOG(pImpl != nullptr, APP_ERR_ILLEGAL_OPERATION, "pImpl can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(pImpl->getAttrTotal() > 0, APP_ERR_INVALID_PARAM, "ntotal must be > 0");
    APP_ERROR ret = APP_ERR_OK;
    ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(this->deviceId), APP_ERR_ACL_SET_DEVICE_FAILED);
    try {
        ret = pImpl->search(count, features, attrFilter, shareAttrFilter, topk, labels, distances, validNums,
            enableTimeFilter, extraValFilter);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_NOT_FMT(false, APP_ERR_INNER_ERROR,
            "Search failed, please check parameters or obtain detail information from logs, error msg[%s]", e.what());
    }
    return ret;
}

static APP_ERROR CheckParameters(uint32_t count, uint32_t topk, uint32_t maxTokenNum, const void *features,
    const AttrFilter *attrFilter, bool shareAttrFilter, const uint8_t *extraMask, const int64_t *labels,
    const float *distances, const uint32_t *validNums)
{
    APPERR_RETURN_IF_NOT_FMT((count > 0) && (count <= TS_MAX_SEARCH), APP_ERR_INVALID_PARAM,
                             "count must be > 0 and <= %ld", TS_MAX_SEARCH);
    APPERR_RETURN_IF_NOT_FMT((topk > 0) && (topk <= TS_MAX_TOPK), APP_ERR_INVALID_PARAM,
                             "topk must be > 0 and <= %ld", TS_MAX_TOPK);
    APPERR_RETURN_IF_NOT_LOG(features != nullptr, APP_ERR_INVALID_PARAM, "features can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(attrFilter != nullptr, APP_ERR_INVALID_PARAM, "attrFilter can not be nullptr.");
    uint32_t attrFilterNum = shareAttrFilter ? 1 : count;  // shareAttrFilter为true时，共享filter，此时filter只有1个
    for (uint32_t i = 0; i < attrFilterNum; i++) {
        APPERR_RETURN_IF_NOT_LOG((attrFilter + i)->tokenBitSet != nullptr,
            APP_ERR_INVALID_PARAM, "tokenBitSet can not be nullptr");
        APPERR_RETURN_IF_NOT_FMT((attrFilter + i)->tokenBitSetLen > 0 &&
            (attrFilter + i)->tokenBitSetLen <= utils::divUp(maxTokenNum, TOKEN_SET_BIT),
            APP_ERR_INVALID_PARAM, "attrFilter[%u] tokenBitSetLen[%u] must be > 0 and <= %u", i,
            (attrFilter + i)->tokenBitSetLen, utils::divUp(maxTokenNum, TOKEN_SET_BIT));
    }
    APPERR_RETURN_IF_NOT_LOG(extraMask != nullptr, APP_ERR_INVALID_PARAM, "extraMask can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(labels != nullptr, APP_ERR_INVALID_PARAM, "labels can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(distances != nullptr, APP_ERR_INVALID_PARAM, "distances can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(validNums != nullptr, APP_ERR_INVALID_PARAM, "validNums can not be nullptr.");

    return APP_ERR_OK;
}

APP_ERROR AscendIndexTS::SearchWithExtraMask(uint32_t count, const void *features, const AttrFilter *attrFilter,
                                             bool shareAttrFilter, uint32_t topk, const uint8_t *extraMask,
                                             uint64_t extraMaskLenEachQuery, bool extraMaskIsAtDevice, int64_t *labels,
                                             float *distances, uint32_t *validNums, bool enableTimeFilter)
{
    auto ret = CheckParameters(count, topk, this->maxTokenNum, features, attrFilter, shareAttrFilter,
        extraMask, labels, distances, validNums);
    APPERR_RETURN_IF(ret, ret);

    std::lock(AscendGlobalLock::GetInstance(this->deviceId), mtx);
    std::lock_guard<std::mutex> glck(AscendGlobalLock::GetInstance(this->deviceId), std::adopt_lock);
    std::lock_guard<std::mutex> lck(mtx, std::adopt_lock);
    APPERR_RETURN_IF_NOT_LOG(pImpl != nullptr, APP_ERR_INVALID_PARAM, "pImpl can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(pImpl->getAttrTotal() > 0, APP_ERR_INVALID_PARAM, "ntotal must be > 0");
    uint64_t maxMaskLen = static_cast<uint64_t>(utils::divUp(pImpl->getAttrTotal(), OPS_DATA_TYPE_ALIGN));
    APPERR_RETURN_IF_NOT_LOG(extraMaskLenEachQuery <= maxMaskLen, APP_ERR_INVALID_PARAM, "extra mask len invalid.");

    ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(this->deviceId), APP_ERR_ACL_SET_DEVICE_FAILED);
    try {
        ret = pImpl->searchWithExtraMask(count, features, attrFilter, shareAttrFilter, topk, extraMask,
            extraMaskLenEachQuery, extraMaskIsAtDevice, labels, distances, validNums, enableTimeFilter, nullptr);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_NOT_FMT(false, APP_ERR_INNER_ERROR,
            "SearchWithExtraMask failed, please check parameters or obtain detail information from logs, error msg[%s]",
            e.what());
    }
    return ret;
}

APP_ERROR AscendIndexTS::SearchWithExtraMask(uint32_t count, const void *features, const AttrFilter *attrFilter,
                                             bool shareAttrFilter, uint32_t topk, const uint8_t *extraMask,
                                             uint64_t extraMaskLenEachQuery, bool extraMaskIsAtDevice,
                                             const uint16_t *extraScore, int64_t *labels, float *distances,
                                             uint32_t *validNums, bool enableTimeFilter)
{
    auto ret = CheckParameters(count, topk, this->maxTokenNum, features, attrFilter, shareAttrFilter,
        extraMask, labels, distances, validNums);
    APPERR_RETURN_IF(ret, ret);

    APPERR_RETURN_IF_NOT_LOG(extraScore != nullptr, APP_ERR_INVALID_PARAM, "extraScore can not be nullptr.");
    std::lock(AscendGlobalLock::GetInstance(this->deviceId), mtx);
    std::lock_guard<std::mutex> glck(AscendGlobalLock::GetInstance(this->deviceId), std::adopt_lock);
    std::lock_guard<std::mutex> lck(mtx, std::adopt_lock);
    APPERR_RETURN_IF_NOT_LOG(pImpl != nullptr, APP_ERR_INVALID_PARAM, "pImpl can not be nullptr.");
    APPERR_RETURN_IF_NOT_FMT(pImpl->getAttrTotal() > 0, APP_ERR_INVALID_PARAM,
        "attr total is %ld must be > 0", pImpl->getAttrTotal());
    uint64_t maxMaskLen = static_cast<uint64_t>(utils::divUp(pImpl->getAttrTotal(), OPS_DATA_TYPE_ALIGN));
    APPERR_RETURN_IF_NOT_FMT(extraMaskLenEachQuery <= maxMaskLen, APP_ERR_INVALID_PARAM,
        "extra mask len[%lu] must <= maxMaskLen[%lu]", extraMaskLenEachQuery, maxMaskLen);

    ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(this->deviceId), APP_ERR_ACL_SET_DEVICE_FAILED);
    try {
        ret = pImpl->searchWithExtraMask(count, features, attrFilter, shareAttrFilter, topk, extraMask,
            extraMaskLenEachQuery, extraMaskIsAtDevice, labels, distances, validNums, enableTimeFilter, extraScore);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_NOT_FMT(false, APP_ERR_INNER_ERROR,
            "SearchWithExtraMask failed, please check parameters or obtain detail information from logs, error msg[%s]",
            e.what());
    }
    return ret;
}

APP_ERROR AscendIndexTS::GetFeatureNum(int64_t *totalNum) const
{
    APPERR_RETURN_IF_NOT_LOG(totalNum != nullptr, APP_ERR_ILLEGAL_OPERATION, "totalNum can not be nullptr.");
    std::lock_guard<std::mutex> lck(mtx);
    APPERR_RETURN_IF_NOT_LOG(pImpl != nullptr, APP_ERR_INVALID_PARAM, "pImpl can not be nullptr.");
    APPERR_RETURN_IF_NOT(pImpl->getAttrTotal() >= 0, APP_ERR_INNER_ERROR);
    ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(this->deviceId), APP_ERR_ACL_SET_DEVICE_FAILED);
    *totalNum = pImpl->getAttrTotal();
    return APP_ERR_OK;
}

APP_ERROR AscendIndexTS::GetBaseByRange(uint32_t offset, uint32_t num, int64_t *labels, void *features,
    FeatureAttr *attributes)
{
    return GetBaseByRangeWithExtraVal(offset, num, labels, features, attributes, nullptr);
}

APP_ERROR AscendIndexTS::GetBaseByRangeWithExtraVal(uint32_t offset, uint32_t num, int64_t *labels, void *features,
    FeatureAttr *attributes, ExtraValAttr *extraVal) const
{
    APPERR_RETURN_IF_NOT_LOG(labels != nullptr, APP_ERR_ILLEGAL_OPERATION, "labels can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(features != nullptr, APP_ERR_ILLEGAL_OPERATION, "features can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(attributes != nullptr, APP_ERR_ILLEGAL_OPERATION, "attributes can not be nullptr.");

    std::lock_guard<std::mutex> lck(mtx);
    APPERR_RETURN_IF_NOT_LOG(pImpl != nullptr, APP_ERR_INVALID_PARAM, "pImpl can not be nullptr.");
    TSBase *impl = dynamic_cast<TSBase *>(pImpl.get());
    APPERR_RETURN_IF_NOT_LOG(impl != nullptr, APP_ERR_ILLEGAL_OPERATION, "TSBase pImpl get failed");
    APPERR_RETURN_IF_NOT_FMT(memoryStrategy == MemoryStrategy::PURE_DEVICE_MEMORY, APP_ERR_ILLEGAL_OPERATION,
        "memoryStrategy %d is not support this func.", memoryStrategy);
    ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(this->deviceId), APP_ERR_ACL_SET_DEVICE_FAILED);
    APPERR_RETURN_IF_NOT_FMT(offset < UPPER_LIMIT_FOR_BASE_SIZE && num > 0 &&
        num <= UPPER_LIMIT_FOR_BASE_SIZE, APP_ERR_ILLEGAL_OPERATION,
        "offset[%u] must be >= 0 and < [%u], num[%u] must be > 0 and <= [%u]", offset, UPPER_LIMIT_FOR_BASE_SIZE,
        num, UPPER_LIMIT_FOR_BASE_SIZE);
    APPERR_RETURN_IF_NOT_FMT((offset + num) <= impl->getAttrTotal(), APP_ERR_ILLEGAL_OPERATION,
        "offset[%u] + num[%u] must be <= ntotal[%ld]", offset, num, impl->getAttrTotal());
    APP_ERROR ret = APP_ERR_OK;
    try {
        ret = impl->getBaseByRange(offset, num, labels, features, attributes, extraVal);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_NOT_FMT(false, APP_ERR_INNER_ERROR,
            "getBaseByRange failed, please check parameters or obtain detail information from logs, error msg[%s]",
            e.what());
    }
    return ret;
}

APP_ERROR AscendIndexTS::GetFeatureByLabel(int64_t count, const int64_t *labels, void *features) const
{
    APPERR_RETURN_IF_NOT_LOG(labels != nullptr, APP_ERR_ILLEGAL_OPERATION, "labels can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(features != nullptr, APP_ERR_ILLEGAL_OPERATION, "features can not be nullptr.");
    std::lock_guard<std::mutex> lck(mtx);
    APPERR_RETURN_IF_NOT(pImpl != nullptr, APP_ERR_ILLEGAL_OPERATION);
    APPERR_RETURN_IF_NOT_FMT(count > 0 && count <= UPPER_LIMIT_FOR_ADD && count <= pImpl->getAttrTotal(),
                             APP_ERR_INVALID_PARAM, "count must be > 0 and <= %ld",
                             std::min(UPPER_LIMIT_FOR_ADD, pImpl->getAttrTotal()));
    APP_ERROR ret = APP_ERR_OK;
    ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(this->deviceId), APP_ERR_ACL_SET_DEVICE_FAILED);
    try {
        ret = pImpl->getFeatureByLabel(count, labels, features);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_NOT_FMT(false, APP_ERR_INNER_ERROR,
            "GetFeatureByLabel failed, please check parameters or obtain detail information from logs, error msg[%s]",
            e.what());
    }
    return ret;
}

APP_ERROR AscendIndexTS::GetFeatureAttrByLabel(int64_t count, const int64_t *labels, FeatureAttr *attributes) const
{
    APPERR_RETURN_IF_NOT_LOG(labels != nullptr, APP_ERR_INVALID_PARAM, "labels can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(attributes != nullptr, APP_ERR_INVALID_PARAM, "attributes can not be nullptr.");
    std::lock_guard<std::mutex> lck(mtx);
    APPERR_RETURN_IF_NOT(pImpl != nullptr, APP_ERR_ILLEGAL_OPERATION);
    APPERR_RETURN_IF_NOT_FMT(count > 0 && count <= UPPER_LIMIT_FOR_GET, APP_ERR_INVALID_PARAM,
        "count[%ld] must be > 0 and <= %ld", count, UPPER_LIMIT_FOR_GET);
    APP_ERROR ret = APP_ERR_OK;
    ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(this->deviceId), APP_ERR_ACL_SET_DEVICE_FAILED);
    try {
        ret = pImpl->getFeatureAttrsByLabel(count, labels, attributes);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_NOT_FMT(false, APP_ERR_INNER_ERROR,
            "GetFeatureAttrByLabel failed, please check parameters or obtain detail information from logs, "
                "error msg[%s]", e.what());
    }
    return ret;
}

APP_ERROR AscendIndexTS::GetExtraValAttrByLabel(int64_t count, const int64_t *labels, ExtraValAttr *extraVal) const
{
    APPERR_RETURN_IF_NOT_LOG(labels != nullptr, APP_ERR_INVALID_PARAM, "labels can not be nullptr.");
    APPERR_RETURN_IF_NOT_LOG(extraVal != nullptr, APP_ERR_INVALID_PARAM, "extraVal can not be nullptr.");
    std::lock_guard<std::mutex> lck(mtx);
    APPERR_RETURN_IF_NOT(pImpl != nullptr, APP_ERR_ILLEGAL_OPERATION);
    TSBinaryFlat *implBinaryFlat = dynamic_cast<TSBinaryFlat *>(pImpl.get());
    TSInt8FlatCos *implInt8Cos = dynamic_cast<TSInt8FlatCos *>(pImpl.get());
    APPERR_RETURN_IF_NOT_LOG(implInt8Cos != nullptr || implBinaryFlat != nullptr, APP_ERR_ILLEGAL_OPERATION,
        "GetExtraValAttrByLabel support Int8FlatCos and BinaryFlat");
    APPERR_RETURN_IF_NOT_FMT(count > 0 && count <= UPPER_LIMIT_FOR_GET, APP_ERR_INVALID_PARAM,
        "count[%ld] must be > 0 and <= %ld", count, UPPER_LIMIT_FOR_GET);
    APP_ERROR ret = APP_ERR_OK;
    ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(this->deviceId), APP_ERR_ACL_SET_DEVICE_FAILED);
    try {
        if (implBinaryFlat != nullptr) {
            ret = implBinaryFlat->getExtraValAttrsByLabel(count, labels, extraVal);
        } else {
            ret = implInt8Cos->getExtraValAttrsByLabel(count, labels, extraVal);
        }
    } catch (std::exception &e) {
        APPERR_RETURN_IF_NOT_FMT(false, APP_ERR_INNER_ERROR,
            "GetFeatureAttrByLabel failed, please check parameters or obtain detail information from logs, "
                "error msg[%s]", e.what());
    }
    return ret;
}

APP_ERROR AscendIndexTS::GetCustomAttrByBlockId(uint32_t blockId, uint8_t *&customAttr) const
{
    APPERR_RETURN_IF_NOT_LOG(pImpl != nullptr, APP_ERR_INVALID_PARAM, "pImpl can not be nullptr.");
    std::lock_guard<std::mutex> lck(mtx);
    APP_ERROR ret = APP_ERR_OK;
    ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(this->deviceId), APP_ERR_ACL_SET_DEVICE_FAILED);
    try {
        ret = pImpl->getCustomAttrByBlockId(blockId, customAttr);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_NOT_FMT(false, APP_ERR_INNER_ERROR,
            "GetCustomAttrByBlockId failed, please check parameters or obtain detail information from logs, "
                "error msg[%s]", e.what());
    }
    return ret;
}

APP_ERROR AscendIndexTS::DeleteFeatureByLabel(int64_t count, const int64_t *labels)
{
    APPERR_RETURN_IF_NOT_LOG(labels != nullptr, APP_ERR_ILLEGAL_OPERATION, "labels can not be nullptr.");
    std::lock(AscendGlobalLock::GetInstance(this->deviceId), mtx);
    std::lock_guard<std::mutex> glck(AscendGlobalLock::GetInstance(this->deviceId), std::adopt_lock);
    std::lock_guard<std::mutex> lck(mtx, std::adopt_lock);
    APPERR_RETURN_IF_NOT(pImpl != nullptr, APP_ERR_ILLEGAL_OPERATION);
    APPERR_RETURN_IF_NOT_FMT(count > 0 && count <= UPPER_LIMIT_FOR_ADD && count <= pImpl->getAttrTotal(),
                             APP_ERR_INVALID_PARAM, "count must be > 0 and <= %ld",
                             std::min(UPPER_LIMIT_FOR_ADD, pImpl->getAttrTotal()));
    APP_ERROR ret = APP_ERR_OK;
    ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(this->deviceId), APP_ERR_ACL_SET_DEVICE_FAILED);
    try {
        ret = pImpl->delFeatureWithLabels(count, labels);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_NOT_FMT(false, APP_ERR_INNER_ERROR,
            "DeleteFeatureByLabel failed, please check parameters or obtain detail information from logs, "
                "error msg[%s]", e.what());
    }
    return ret;
}

APP_ERROR AscendIndexTS::DeleteFeatureByToken(int64_t count, const uint32_t *tokens)
{
    APPERR_RETURN_IF_NOT_LOG(tokens != nullptr, APP_ERR_ILLEGAL_OPERATION, "tokens can not be nullptr.");
    std::lock(AscendGlobalLock::GetInstance(this->deviceId), mtx);
    std::lock_guard<std::mutex> glck(AscendGlobalLock::GetInstance(this->deviceId), std::adopt_lock);
    std::lock_guard<std::mutex> lck(mtx, std::adopt_lock);
    APPERR_RETURN_IF_NOT(pImpl != nullptr, APP_ERR_ILLEGAL_OPERATION);
    APPERR_RETURN_IF_NOT_FMT(count > 0 && count <= UPPER_LIMIT_FOR_ADD && count <= pImpl->getAttrTotal(),
                             APP_ERR_INVALID_PARAM, "count must be > 0 and <= %ld",
                             std::min(UPPER_LIMIT_FOR_ADD, pImpl->getAttrTotal()));
    APP_ERROR ret = APP_ERR_OK;
    ACL_REQUIRE_OK_RET_CODE(aclrtSetDevice(this->deviceId), APP_ERR_ACL_SET_DEVICE_FAILED);
    try {
        ret = pImpl->deleteFeatureByToken(count, tokens);
    } catch (std::exception &e) {
        APPERR_RETURN_IF_NOT_FMT(false, APP_ERR_INNER_ERROR,
            "DeleteFeatureByToken failed, please check parameters or obtain detail information from logs, "
                "error msg[%s]", e.what());
    }
    return ret;
}

APP_ERROR AscendIndexTS::SetSaveHostMemory()
{
    APPERR_RETURN_IF_NOT(pImpl != nullptr, APP_ERR_ILLEGAL_OPERATION);
    TSBinaryFlat *implBinaryFlat = dynamic_cast<TSBinaryFlat *>(pImpl.get());
    TSInt8FlatCos *implInt8Cos = dynamic_cast<TSInt8FlatCos *>(pImpl.get());
    APPERR_RETURN_IF_NOT_LOG(implInt8Cos != nullptr || implBinaryFlat != nullptr, APP_ERR_ILLEGAL_OPERATION,
        "SetSaveHostMemory support Int8FlatCos and BinaryFlat");
    APPERR_RETURN_IF_NOT_LOG(pImpl->getAttrTotal() == 0, APP_ERR_INVALID_PARAM, "ntotal must be == 0");
    if (implBinaryFlat != nullptr) {
        implBinaryFlat->setSaveHostMemory();
    } else {
        implInt8Cos->setSaveHostMemory();
    }
    return APP_ERR_OK;
}

} // namespace ascend
} // namespace faiss