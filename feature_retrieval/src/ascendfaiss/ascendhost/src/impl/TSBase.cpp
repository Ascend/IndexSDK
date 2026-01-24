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


#include "ascendhost/include/impl/TSBase.h"
#include "common/utils/SocUtils.h"


namespace ascend {
TSBase::TSBase(uint32_t tokenNum, uint32_t customAttrLen, uint32_t customAttrBlockSize)
{
    this->tokenNum = tokenNum;
    this->customAttrLen = customAttrLen;
    this->customAttrBlockSize = customAttrBlockSize;
}

APP_ERROR TSBase::initialize(int deviceId)
{
    APP_LOG_INFO("TSBase Initialize operation started.\n");
    if (is_initialized) {
        APP_LOG_INFO("TSBase has initialized.\n");
        return APP_ERR_OK;
    }
    auto aclErrorCode = aclrtSetDevice(deviceId);
    APPERR_RETURN_IF_NOT(aclErrorCode == ACL_SUCCESS, APP_ERR_INNER_ERROR);
    try {
        pResources = std::unique_ptr<AscendResourcesProxy>(new AscendResourcesProxy());
    } catch (AscendException &e) {
        return APP_ERR_INNER_ERROR;
    }
    // 由子类进行pResource内存管理，此处只使用
    pResources->initialize();
    if (faiss::ascend::SocUtils::GetInstance().IsAscend910B()) {
    } else {
        resetMaskGenerateComputeOp();
        resetExtraMaskGenerateComputeOp();
        auto ret = resetBatchMaskGenerateComputeOp();
        APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, ret, "failed to resetBatchMaskGenerateComputeOp");
        ret = resetBatchValMaskGenerateComputeOp();
        APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, ret, "failed to resetBatchValMaskGenerateComputeOp");
        ret = resetBatchExtraMaskGenerateComputeOp(false);
        APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, ret, "failed to resetBatchExtraMaskGenerateComputeOp");
        ret = resetBatchExtraMaskGenerateComputeOp(true);
        APPERR_RETURN_IF_NOT_LOG(ret == APP_ERR_OK, ret,
            "failed to resetBatchExtraMaskGenerateComputeOp with baseMask");
    }
    is_initialized = true;
    APP_LOG_INFO("TSBase Initialize operation end.\n");
    return APP_ERR_OK;
}
int64_t TSBase::getAttrTotal() const
{
    return attrTotal;
}
void TSBase::addFeatureAttrs(int64_t n, const faiss::ascend::FeatureAttr *attrs, const uint8_t *customAttr)
{
    AddWithExtraValAttrs(n, attrs, customAttr, nullptr);
}

void TSBase::AddWithExtraValAttrs(int64_t n, const faiss::ascend::FeatureAttr *attrs, const uint8_t *customAttr,
    const faiss::ascend::ExtraValAttr *extraVal)
{
    ASCEND_THROW_IF_NOT_FMT(n >= 0, "the count of attrs added is %d.\n", n);
    ASCEND_THROW_IF_NOT_MSG(attrs, "the attrs is nullptr.\n");
    if (customAttrLen > 0 && customAttrBlockSize > 0) {
        ASCEND_THROW_IF_NOT_MSG(customAttr != nullptr, "the customAttr is nullptr.\n");
    }
    int64_t oldBlockNum = utils::divUp(attrTotal, static_cast<int64_t>(multiFeaAttrBlkSize));
    int64_t addBlockNum = utils::divUp(attrTotal + n,  static_cast<int64_t>(multiFeaAttrBlkSize)) - oldBlockNum;
    for (int64_t i = 0; i < addBlockNum; ++i) {
        attrTime.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<int32_t>, MemorySpace::DEVICE_HUGEPAGE));
        attrTokenQuotient.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<int32_t>, MemorySpace::DEVICE_HUGEPAGE));
        attrTokenRemainder.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<uint8_t>, MemorySpace::DEVICE_HUGEPAGE));
 
        attrTime.at(oldBlockNum + i)->resize(multiFeaAttrBlkSize, true);
        attrTokenQuotient.at(oldBlockNum + i)->resize(multiFeaAttrBlkSize, true);
        attrTokenRemainder.at(oldBlockNum + i)->resize(multiFeaAttrBlkSize * OPS_DATA_TYPE_TIMES, true);
        if (extraVal != nullptr) {
            attrVal.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<int16_t>, MemorySpace::DEVICE_HUGEPAGE));
            attrVal.at(oldBlockNum + i)->resize(multiFeaAttrBlkSize, true);
        }
    }
    addAttrsImpl(n, attrs, extraVal);
    if (customAttrLen > 0 && customAttr != nullptr) {
        addCustomAttrsImpl(n, customAttr);
    }
    attrTotal += n;
}

APP_ERROR TSBase::AddFeatureByIndice(int64_t, const void *,
    const faiss::ascend::FeatureAttr *, const int64_t *, const uint8_t *,
    const faiss::ascend::ExtraValAttr *)
{
    APP_LOG_ERROR("Not support AddFeatureByIndice now");
    return APP_ERR_ILLEGAL_OPERATION;
}

void TSBase::ReshapeAttrsSpace(int64_t newAddCount, const uint8_t *customAttr,
    const faiss::ascend::ExtraValAttr *extraVal)
{
    // 根据位置的最大值计算需要新增的空间
    int64_t newAttrTotal = attrTotal + newAddCount;
    int64_t oldBlockNum = utils::divUp(attrTotal, static_cast<int64_t>(multiFeaAttrBlkSize));
    int64_t addBlockNum = utils::divUp(newAttrTotal, static_cast<int64_t>(multiFeaAttrBlkSize)) - oldBlockNum;
    for (int64_t i = 0; i < addBlockNum; ++i) {
        attrTime.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<int32_t>, MemorySpace::DEVICE_HUGEPAGE));
        attrTokenQuotient.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<int32_t>, MemorySpace::DEVICE_HUGEPAGE));
        attrTokenRemainder.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<uint8_t>, MemorySpace::DEVICE_HUGEPAGE));
 
        attrTime.at(oldBlockNum + i)->resize(multiFeaAttrBlkSize, true);
        attrTokenQuotient.at(oldBlockNum + i)->resize(multiFeaAttrBlkSize, true);
        attrTokenRemainder.at(oldBlockNum + i)->resize(multiFeaAttrBlkSize * OPS_DATA_TYPE_TIMES, true);
        if (extraVal != nullptr) {
            attrVal.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<int16_t>, MemorySpace::DEVICE_HUGEPAGE));
            attrVal.at(oldBlockNum + i)->resize(multiFeaAttrBlkSize, true);
        }
    }
    featureAttrs.resize(newAttrTotal);
    if (extraVal != nullptr) {
        extraValAttrs.resize(newAttrTotal);
    }

    if (customAttrLen > 0 && customAttr != nullptr) {
        oldBlockNum = utils::divUp(attrTotal, static_cast<int64_t>(customAttrBlockSize));
        addBlockNum = utils::divUp(newAttrTotal, static_cast<int64_t>(customAttrBlockSize)) - oldBlockNum;
        for (int64_t i = 0; i < addBlockNum; ++i) {
            customAttrBase.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<uint8_t>, MemorySpace::DEVICE_HUGEPAGE));
            customAttrBase.at(oldBlockNum + i)->resize(customAttrBlockSize * customAttrLen, true);
        }
    }
}

APP_ERROR TSBase::AddFeatureAttrsByIndice(int64_t n, const std::vector<std::pair<int64_t, int64_t>> &segments,
    const int64_t *indices, const faiss::ascend::FeatureAttr *attrs, const uint8_t *customAttr,
    const faiss::ascend::ExtraValAttr *extraVal)
{
    if (customAttrLen > 0 && customAttrBlockSize > 0) {
        APPERR_RETURN_IF_NOT_LOG(customAttr != nullptr, APP_ERR_INVALID_PARAM, "the customAttr is nullptr.");
    }
    int64_t newAddCount = (indices[n - 1] >= attrTotal) ? (indices[n - 1] - attrTotal + 1) : 0;
    
    ReshapeAttrsSpace(newAddCount, customAttr, extraVal);

    // 每个连续段进行拷贝
    for (size_t segIdx = 0; segIdx < segments.size(); segIdx++) {
        int64_t start = segments[segIdx].first;  // 起始位置
        int64_t length = segments[segIdx].second; // 每段的长度

        int64_t tmpAttrTotal = indices[start];
        // 拷贝时空属性和附加属性
        if (extraVal != nullptr) {
            CopyAndSaveTSAttrs(length, tmpAttrTotal, attrs + start, extraVal + start);
        } else {
            CopyAndSaveTSAttrs(length, tmpAttrTotal, attrs + start, nullptr);
        }

        // 拷贝自定义属性
        if (customAttrLen > 0 && customAttr != nullptr) {
            CopyAndSaveCustomAttrs(length, tmpAttrTotal,
                customAttr + start * static_cast<int32_t>(this->customAttrLen));
        }
    }
    attrTotal += newAddCount;
    return APP_ERR_OK;
}

APP_ERROR TSBase::GetFeatureByIndice(int64_t, const int64_t *, int64_t *,
    void *, faiss::ascend::FeatureAttr *, faiss::ascend::ExtraValAttr *) const
{
    APP_LOG_ERROR("Not support GetFeatureByIndice now");
    return APP_ERR_ILLEGAL_OPERATION;
}

void TSBase::SetMaskValid(int64_t n, const int64_t *indices, int64_t ntotal)
{
    int64_t newSize = utils::divUp(ntotal, MASK_ALIGN);
    if (newSize > static_cast<int64_t>(baseMask.size())) {
        baseMask.resize(newSize, 0);
    }
    for (int64_t i = 0; i < n; ++i) {
        int64_t byteIndex = indices[i] / MASK_ALIGN;
        int64_t bitIndex = indices[i] % MASK_ALIGN;
        baseMask[byteIndex] |= (1 << bitIndex);
    }
    for (int64_t idx = 0; idx < newSize; idx++) {
        if (baseMask[idx] != 0xff) {
            useBaseMask = true;
            return;
        }
    }
    int64_t last = static_cast<int64_t>(ntotal % MASK_ALIGN);
    if (last == 0) {
        useBaseMask = false;
        return;
    }
    uint8_t checkMask = (1 << last) - 1;
    if (checkMask == (baseMask[newSize] & checkMask)) {
        useBaseMask = false;
    } else {
        useBaseMask = true;
    }
}

void TSBase::SetMaskInvalid(int64_t start, int64_t end, const int64_t *indices, int64_t ntotal)
{
    APP_LOG_INFO("TSBase::SetMaskInvalid start");
    for (int64_t i = start; i < end; ++i) {
        int64_t indice = i;
        if (indices != nullptr) {
            indice = indices[i];
        }
        if (indice >= ntotal) {
            APP_LOG_ERROR("TSBase::SetMaskInvalid failed, indice must be < ntotal");
            continue;
        }
        int64_t byteIndex = indice / MASK_ALIGN;
        int64_t bitIndex = indice % MASK_ALIGN;
        if (static_cast<size_t>(byteIndex) >= baseMask.size()) {
            APP_LOG_ERROR("TSBase::SetMaskInvalid failed, bitIndex(%ld) must be < baseMask's size(%zu)",
                          byteIndex, baseMask.size());
            continue;
        }
        baseMask[byteIndex] &= (~(1 << bitIndex));
    }
    useBaseMask = true;
    APP_LOG_INFO("TSBase::SetMaskInvalid end");
}

std::vector<uint8_t> TSBase::GetBaseMask() const
{
    return baseMask;
}

APP_ERROR TSBase::FastDeleteFeatureByIndice(int64_t n, const int64_t *indices)
{
    APP_LOG_INFO("TSBase::FastDeleteFeatureByIndice start");
    SetMaskInvalid(0, n, indices, this->attrTotal);
    APP_LOG_INFO("TSBase::FastDeleteFeatureByIndice end");
    return APP_ERR_OK;
}

APP_ERROR TSBase::FastDeleteFeatureByRange(int64_t start, int n)
{
    APP_LOG_INFO("TSBase::FastDeleteFeatureByRange start:%ld, n:%ld", start, n);
    SetMaskInvalid(start, start + n, nullptr, this->attrTotal);
    APP_LOG_INFO("TSBase::FastDeleteFeatureByRange end");
    return APP_ERR_OK;
}

void TSBase::addCustomAttrsImpl(int64_t n, const uint8_t *customAttr)
{
    int64_t oldBlockNum = utils::divUp(attrTotal, static_cast<int64_t>(customAttrBlockSize));
    int64_t addBlockNum = utils::divUp(attrTotal + n, static_cast<int64_t>(customAttrBlockSize)) - oldBlockNum;
    for (int64_t i = 0; i < addBlockNum; ++i) {
        customAttrBase.emplace_back(CREATE_UNIQUE_PTR(DeviceVector<uint8_t>, MemorySpace::DEVICE_HUGEPAGE));
        customAttrBase.at(oldBlockNum + i)->resize(customAttrBlockSize * customAttrLen, true);
    }

    CopyAndSaveCustomAttrs(n, this->attrTotal, customAttr);
}

void TSBase::CopyAndSaveCustomAttrs(int64_t n, int64_t fakeTotal, const uint8_t *customAttr)
{
    std::string opName = "TransdataCustomAttr";
    auto &mem = pResources->getMemoryManager();
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();

    AscendTensor<uint8_t, DIMS_2> customAttrData(mem,
        { static_cast<int32_t>(n), static_cast<int32_t>(this->customAttrLen) }, stream);
    auto ret = aclrtMemcpy(customAttrData.data(), customAttrData.getSizeInBytes(), customAttr, n * this->customAttrLen,
        ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Mem operator error %d", ret);

    int64_t blockNum = utils::divUp(fakeTotal + n, static_cast<int64_t>(customAttrBlockSize));
    AscendTensor<int64_t, DIMS_2> attrs(mem,
        {  static_cast<int>(blockNum), aicpu::TRANSDATA_CUSTOM_ATTR_IDX_COUNT }, stream);
    for (size_t i = 0; i < static_cast<size_t>(n);) {
        size_t total = static_cast<size_t>(fakeTotal) + i;
        size_t offsetInBlock = total % customAttrBlockSize;
        size_t leftInBlock = customAttrBlockSize - offsetInBlock;
        size_t leftInData = static_cast<size_t>(n) - i;
        size_t copyCount = std::min(leftInBlock, leftInData);
        size_t blockIdx = total / customAttrBlockSize;
        AscendTensor<uint8_t, DIMS_2> src(customAttrData[i].data(),
            { static_cast<int>(copyCount), static_cast<int>(this->customAttrLen) });
        AscendTensor<uint8_t, DIMS_2> dst(customAttrBase[blockIdx]->data(),
            { static_cast<int>(customAttrLen), static_cast<int>(customAttrBlockSize) });
        AscendTensor<int64_t, DIMS_1> attr = attrs[blockIdx].view();
        attr[aicpu::TRANSDATA_CUSTOM_ATTR_NTOTAL_IDX] = offsetInBlock;
        i += copyCount;
        LaunchOpTwoInOneOut<uint8_t, DIMS_2, ACL_UINT8, int64_t, DIMS_1, ACL_INT64, uint8_t, DIMS_2, ACL_UINT8>(
            opName, stream, src, attr, dst);
    }
    ret = synchronizeStream(stream);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "transdata custom attr operator error %d.\n", ret);
}

void TSBase::addAttrsImpl(int64_t n, const faiss::ascend::FeatureAttr *attrs,
    const faiss::ascend::ExtraValAttr *extraVal)
{
    featureAttrs.resize(attrTotal + n);
    if (extraVal != nullptr) {
        extraValAttrs.resize(attrTotal + n);
    }
    CopyAndSaveTSAttrs(n, attrTotal, attrs, extraVal);
}

void TSBase::CopyAndSaveTSAttrs(int64_t n, int64_t fakeTotal, const faiss::ascend::FeatureAttr *attrs,
    const faiss::ascend::ExtraValAttr *extraVal)
{
    std::vector<int32_t> tmpTimes(n);
    std::vector<int32_t> tmpTokenQs(n);
    std::vector<uint8_t> tmpTokenRs(n * OPS_DATA_TYPE_TIMES);
    std::vector<int16_t> tmpVals(n);
    for (int64_t i = 0; i < n; i++) {
        faiss::ascend::FeatureAttr tmpAttr = *(attrs + i);
        uint32_t tmpTokenId = tmpAttr.tokenId; // tokenId < 1.0E5
        ASCEND_THROW_IF_NOT_FMT(tmpTokenId < tokenNum, "the token id  of attrs is more than %u.\n", tokenNum);
        tmpTimes[i] = tmpAttr.time;
        tmpTokenQs[i] = (static_cast<int>(tmpTokenId) / OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES;
        tmpTokenRs[OPS_DATA_TYPE_TIMES * i] = 1 << (tmpTokenId % OPS_DATA_TYPE_ALIGN);
        tmpTokenRs[OPS_DATA_TYPE_TIMES * i + 1] = OPS_DATA_PADDING_VAL;
        if (!enableSaveHostMemory) {
            token2Idx[tmpTokenId].emplace(static_cast<uint32_t>(fakeTotal + i));
        }
        featureAttrs[fakeTotal + i] = tmpAttr;
        if (extraVal != nullptr) {
            faiss::ascend::ExtraValAttr tmpVal = *(extraVal + i);
            tmpVals[i] = tmpVal.val;
            extraValAttrs[fakeTotal + i] = tmpVal;
        }
    }
    int64_t oldBlockNum = utils::divUp(fakeTotal, static_cast<int64_t>(featureAttrBlockSize));
    int64_t addBlockNum = utils::divUp(fakeTotal + n, static_cast<int64_t>(featureAttrBlockSize)) - oldBlockNum;

    int lastBlockOffset = fakeTotal % static_cast<int>(featureAttrBlockSize);
    int offset = 0;
    if (lastBlockOffset) {
        int64_t cpyNum = std::min(n, static_cast<int64_t>(featureAttrBlockSize) * oldBlockNum - fakeTotal);
        copyLastFeatureAttrs(cpyNum, oldBlockNum, lastBlockOffset, tmpTimes, tmpTokenQs, tmpTokenRs);
        if (extraVal != nullptr) {
            copyLastExtraValAttrs(cpyNum, oldBlockNum, lastBlockOffset, tmpVals);
        }
        offset += cpyNum;
    }
    for (int64_t i = oldBlockNum; i < addBlockNum + oldBlockNum; i++) {
        int64_t copyNum = std::min(n - offset, static_cast<int64_t>(featureAttrBlockSize));
        copyBlockFeatureAttrs(copyNum, offset, i, tmpTimes, tmpTokenQs, tmpTokenRs);

        if (extraVal != nullptr) {
            copyBlockExtraValAttrs(copyNum, offset, i, tmpVals);
        }

        offset += copyNum;
    }
}

void TSBase::copyBlockFeatureAttrs(int64_t copyNum, int offset, int64_t i,
    std::vector<int32_t> &tmpTimes, std::vector<int32_t> &tmpTokenQs, std::vector<uint8_t> &tmpTokenRs)
{
    auto ret = aclrtMemcpy(calcAttrStartAddress(attrTime, i),
        featureAttrBlockSize * sizeof(int32_t),
        tmpTimes.data() + offset,
        copyNum * sizeof(int32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "ACL copy time data operator error %d.\n", ret);

    ret = aclrtMemcpy(calcAttrStartAddress(attrTokenQuotient, i),
        featureAttrBlockSize * sizeof(int32_t),
        tmpTokenQs.data() + offset,
        copyNum * sizeof(int32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "ACL copy tokenq data operator error %d.\n", ret);

    ret = aclrtMemcpy(calcAttrStartAddress(attrTokenRemainder, i, OPS_DATA_TYPE_TIMES),
        featureAttrBlockSize * OPS_DATA_TYPE_TIMES,
        tmpTokenRs.data() + offset * OPS_DATA_TYPE_TIMES,
        copyNum * OPS_DATA_TYPE_TIMES,
        ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "ACL copy tokenr data operator error %d.\n", ret);
}

void TSBase::copyBlockExtraValAttrs(int64_t copyNum, int offset, int64_t i, std::vector<int16_t> &tmpVals)
{
    auto ret = aclrtMemcpy(calcAttrStartAddress(attrVal, i),
        featureAttrBlockSize * sizeof(int16_t),
        tmpVals.data() + offset,
        copyNum * sizeof(int16_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "ACL copy val data operator error %d.\n", ret);
}

void TSBase::copyLastFeatureAttrs(int64_t cpyNum, int64_t oldBlockNum, int lastBlockOffset,
    std::vector<int32_t> &tmpTimes, std::vector<int32_t> &tmpTokenQs, std::vector<uint8_t> &tmpTokenRs)
{
    int lastTokenRoffset =
        (OPS_DATA_TYPE_TIMES * lastBlockOffset) % (OPS_DATA_TYPE_TIMES * static_cast<int>(featureAttrBlockSize));
    int64_t lastBlockId = oldBlockNum - 1;
    auto ret = aclrtMemcpy(calcAttrStartAddress(attrTime, lastBlockId) + lastBlockOffset,
        cpyNum * sizeof(int32_t),
        tmpTimes.data(),
        cpyNum * sizeof(int32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "ACL copy time data operator error %d.\n", ret);

    ret = aclrtMemcpy(calcAttrStartAddress(attrTokenQuotient, lastBlockId) + lastBlockOffset,
        cpyNum * sizeof(int32_t),
        tmpTokenQs.data(),
        cpyNum * sizeof(int32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "ACL copy tokenq data operator error %d.\n", ret);

    ret = aclrtMemcpy(calcAttrStartAddress(attrTokenRemainder, lastBlockId, OPS_DATA_TYPE_TIMES) + lastTokenRoffset,
        cpyNum * OPS_DATA_TYPE_TIMES,
        tmpTokenRs.data(),
        cpyNum * OPS_DATA_TYPE_TIMES,
        ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "ACL copy tokenr data operator error %d.\n", ret);
}

void TSBase::copyLastExtraValAttrs(int64_t cpyNum, int64_t oldBlockNum, int lastBlockOffset,
    std::vector<int16_t> &tmpVals)
{
    int64_t lastBlockId = oldBlockNum - 1;
    auto ret = aclrtMemcpy(calcAttrStartAddress(attrVal, lastBlockId) + lastBlockOffset,
        featureAttrBlockSize * sizeof(int16_t),
        tmpVals.data(),
        cpyNum * sizeof(int16_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "ACL copy val data operator error %d.\n", ret);
}

void TSBase::runVectorRemoveOp(const std::vector<uint64_t> &src, const std::vector<uint64_t> &dst, int dataType,
    int copyNum)
{
    std::string opName = "RemovedataAttr";
    auto &mem = pResources->getMemoryManager();
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    int removeSize = static_cast<int>(dst.size());
    AscendTensor<int64_t, DIMS_1> attrsInput(mem, { aicpu::REMOVEDATA_ATTR_IDX_COUNT }, stream);
    std::vector<int64_t> attrs(aicpu::REMOVEDATA_ATTR_IDX_COUNT);
    AscendTensor<uint64_t, DIMS_1> srcAttr(mem, { removeSize }, stream);
    AscendTensor<uint64_t, DIMS_1> dstAttr(mem, { removeSize }, stream);
    auto ret = aclrtMemcpy(srcAttr.data(), srcAttr.getSizeInBytes(), src.data(), src.size() * sizeof(uint64_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to copy to device (error %d)", (int)ret);

    ret = aclrtMemcpy(dstAttr.data(), dstAttr.getSizeInBytes(), dst.data(), dst.size() * sizeof(uint64_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to copy to device (error %d)", (int)ret);

    attrs[aicpu::REMOVEDATA_ATTR_DATA_TYPE] = dataType;
    attrs[aicpu::REMOVEDATA_ATTR_COPY_NUM] = copyNum;
    ret = aclrtMemcpy(attrsInput.data(), attrsInput.getSizeInBytes(), attrs.data(), attrs.size() * sizeof(int64_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to copy to device (error %d)", (int)ret);

    LaunchOpTwoInOneOut<uint64_t, DIMS_1, ACL_UINT64, int64_t, DIMS_1, ACL_INT64, uint64_t, DIMS_1, ACL_UINT64>(
        opName, stream, srcAttr, attrsInput, dstAttr);

    ret = synchronizeStream(stream);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to acl Synchronize Streame (error %d)", (int)ret);
}

void TSBase::updataAttrsAfterDel(int64_t id, int lastIdx)
{
    featureAttrs[id] = featureAttrs[lastIdx];
    featureAttrs.pop_back();
    if (!extraValAttrs.empty()) {
        extraValAttrs[id] = extraValAttrs[lastIdx];
        extraValAttrs.pop_back();
    }
    --attrTotal;
}

void TSBase::updataMappingAfterDel(const std::vector<int64_t> &ids)
{
    if (enableSaveHostMemory) {
        for (const auto delId : ids) {
            int lastIdx = attrTotal - 1;
            updataAttrsAfterDel(delId, lastIdx);
        }
        return;
    }

    // update the mapping relationship
    for (const auto delId : ids) {
        int lastIdx = attrTotal - 1;
        uint32_t delIdMaptokenId = featureAttrs[delId].tokenId;
        token2Idx[delIdMaptokenId].erase(delId);
        if (token2Idx[delIdMaptokenId].size() == 0) {
            token2Idx.erase(delIdMaptokenId);
        }
        uint32_t lastTokenId = featureAttrs[lastIdx].tokenId;
        token2Idx[lastTokenId].insert(delId);
        token2Idx[lastTokenId].erase(lastIdx);
        updataAttrsAfterDel(delId, lastIdx);
    }
}

//  ids 由调用方保证去重和从大到小排序
void TSBase::deleteAttrByIds(const std::vector<int64_t> &ids)
{
    APP_LOG_INFO("TSBase DeleteAttrByIds operation started.\n");
    if (attrTotal <= 0) {
        APP_LOG_INFO("TSBase DeleteAttrByIds attrTotal invalid.\n");
        return;
    }

    int64_t originTotal = attrTotal;
    auto removeSize = ids.size();
    std::vector<uint64_t> attrTimeSrcAddr(removeSize);
    std::vector<uint64_t> attrTimeDstAddr(removeSize);
    std::vector<uint64_t> attrTokenQuotientSrcAddr(removeSize);
    std::vector<uint64_t> attrTokenQuotientDstAddr(removeSize);
    std::vector<uint64_t> attrTokenRemainderSrcAddr(removeSize);
    std::vector<uint64_t> attrTokenRemainderDstAddr(removeSize);
    std::vector<uint64_t> extraValSrcAddr(removeSize);
    std::vector<uint64_t> extraValDstAddr(removeSize);
    for (size_t i = 0; i < removeSize; i++) {
        size_t srcIdx = static_cast<size_t>(attrTotal) - i - 1;
        size_t srcIdx1 = srcIdx / featureAttrBlockSize;
        size_t srcIdx2 = srcIdx % featureAttrBlockSize;
        size_t srcIdxR2 = (OPS_DATA_TYPE_TIMES * srcIdx) % (OPS_DATA_TYPE_TIMES * featureAttrBlockSize);

        size_t dstIdx1 = ids[i] / featureAttrBlockSize;
        size_t dstIdx2 = ids[i] % featureAttrBlockSize;
        size_t dstIdxR2 = (OPS_DATA_TYPE_TIMES * ids[i]) % (OPS_DATA_TYPE_TIMES * featureAttrBlockSize);

        attrTimeSrcAddr[i] = reinterpret_cast<uint64_t>(calcAttrStartAddress(attrTime, srcIdx1) + srcIdx2);
        attrTimeDstAddr[i] = reinterpret_cast<uint64_t>(calcAttrStartAddress(attrTime, dstIdx1) + dstIdx2);
        attrTokenQuotientSrcAddr[i] =
            reinterpret_cast<uint64_t>(calcAttrStartAddress(attrTokenQuotient, srcIdx1) + srcIdx2);
        attrTokenQuotientDstAddr[i] =
            reinterpret_cast<uint64_t>(calcAttrStartAddress(attrTokenQuotient, dstIdx1) + dstIdx2);
        attrTokenRemainderSrcAddr[i] = reinterpret_cast<uint64_t>(
            calcAttrStartAddress(attrTokenRemainder, srcIdx1, OPS_DATA_TYPE_TIMES) + srcIdxR2);
        attrTokenRemainderDstAddr[i] = reinterpret_cast<uint64_t>(
            calcAttrStartAddress(attrTokenRemainder, dstIdx1, OPS_DATA_TYPE_TIMES) + dstIdxR2);
        
        if (!attrVal.empty()) {
            extraValSrcAddr[i] = reinterpret_cast<uint64_t>(calcAttrStartAddress(attrVal, srcIdx1) + srcIdx2);
            extraValDstAddr[i] = reinterpret_cast<uint64_t>(calcAttrStartAddress(attrVal, dstIdx1) + dstIdx2);
        }
    }

    runVectorRemoveOp(attrTimeSrcAddr, attrTimeDstAddr, faiss::ascend::INT32, 1);
    runVectorRemoveOp(attrTokenQuotientSrcAddr, attrTokenQuotientDstAddr, faiss::ascend::INT32, 1);
    runVectorRemoveOp(attrTokenRemainderSrcAddr, attrTokenRemainderDstAddr, faiss::ascend::UINT8, OPS_DATA_TYPE_TIMES);

    runExtraValOp(extraValSrcAddr, extraValDstAddr);

    if (customAttrLen > 0) {
        deleteCustomAttrByIds(ids);
    }
    // update the mapping relationship
    updataMappingAfterDel(ids);

    removeIndvalidAttr(static_cast<uint64_t>(originTotal), ids.size());
    APP_LOG_INFO("TSBase DeleteAttrByIds operation end.\n");
}

void TSBase::runExtraValOp(std::vector<uint64_t> &extraValSrcAddr, std::vector<uint64_t> &extraValDstAddr)
{
    if (!attrVal.empty()) {
        runVectorRemoveOp(extraValSrcAddr, extraValDstAddr, faiss::ascend::INT16, 1);
    }
}

void TSBase::deleteCustomAttrByIds(const std::vector<int64_t> &ids)
{
    APP_LOG_INFO("TSBase deleteCustomAttrByIds operation started.\n");
    std::string opName = "RemovedataCustomAttr";
    auto &mem = pResources->getMemoryManager();
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();

    auto removeSize = ids.size();
    std::vector<uint64_t> customAttrSrcAddr(removeSize);
    std::vector<uint64_t> customAttrDstAddr(removeSize);

    for (size_t i = 0; i < removeSize; i++) {
        size_t srcIdx = static_cast<size_t>(attrTotal) - i - 1;
        size_t srcIdx1 = srcIdx / customAttrBlockSize;
        size_t srcIdx2 = srcIdx % customAttrBlockSize;
        size_t dstIdx1 = ids[i] / customAttrBlockSize;
        size_t dstIdx2 = ids[i] % customAttrBlockSize;
        uint8_t *srcDataPtr = customAttrBase[srcIdx1]->data() + srcIdx2;
        uint8_t *dstDataPtr = customAttrBase[dstIdx1]->data() + dstIdx2;
        customAttrSrcAddr[i] = reinterpret_cast<uint64_t>(srcDataPtr);
        customAttrDstAddr[i] = reinterpret_cast<uint64_t>(dstDataPtr);
    }

    AscendTensor<uint64_t, DIMS_1> srcInput(mem, { static_cast<int32_t>(removeSize) }, stream);
    AscendTensor<uint64_t, DIMS_1> dstInput(mem, { static_cast<int32_t>(removeSize) }, stream);
    auto ret = aclrtMemcpy(srcInput.data(), srcInput.getSizeInBytes(), customAttrSrcAddr.data(),
        customAttrSrcAddr.size() * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to copy to device (error %d)", ret);
    ret = aclrtMemcpy(dstInput.data(), dstInput.getSizeInBytes(), customAttrDstAddr.data(),
        customAttrDstAddr.size() * sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to copy to device (error %d)", ret);

    AscendTensor<int64_t, DIMS_1> attrsInput(mem, { aicpu::REMOVEDATA_CUSTOM_ATTR_IDX_COUNT }, stream);
    std::vector<int64_t> attrs(aicpu::REMOVEDATA_CUSTOM_ATTR_IDX_COUNT);
    attrs[aicpu::REMOVEDATA_CUSTOM_ATTR_DATA_TYPE] = faiss::ascend::UINT8;
    attrs[aicpu::REMOVEDATA_CUSTOM_ATTR_LEN] = customAttrLen;
    attrs[aicpu::REMOVEDATA_CUSTOM_ATTR_BLOCKSIZE] = customAttrBlockSize;
    ret = aclrtMemcpy(attrsInput.data(), attrsInput.getSizeInBytes(), attrs.data(), attrs.size() * sizeof(int64_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to copy to device (error %d)", ret);
    LaunchOpTwoInOneOut<uint64_t, DIMS_1, ACL_UINT64, int64_t, DIMS_1, ACL_INT64, uint64_t, DIMS_1, ACL_UINT64>(
        opName, stream, srcInput, attrsInput, dstInput);

    ret = synchronizeStream(stream);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Failed to acl Synchronize Streame (error %d)", ret);

    removeIndvalidCustomAttr(this->attrTotal, removeSize);
    APP_LOG_INFO("TSBase deleteCustomAttrByIds operation end.\n");
}


void TSBase::removeIndvalidCustomAttr(uint64_t originTotal, uint64_t removeCnt)
{
    APP_LOG_INFO("TSBase removeIndvalidCustomAttr operation start. \n");
    size_t originVecNum = utils::divUp(originTotal, customAttrBlockSize);
    size_t newVecNum = utils::divUp(originTotal - removeCnt, customAttrBlockSize);
    for (size_t i = newVecNum; i < originVecNum; ++i) {
        customAttrBase.at(i)->clear();
    }
    customAttrBase.resize(newVecNum);
    APP_LOG_INFO("TSBase removeIndvalidCustomAttr operation finished. \n");
}

void TSBase::removeIndvalidAttr(uint64_t originTotal, uint64_t removeCnt)
{
    size_t originVecNum = utils::divUp(originTotal, multiFeaAttrBlkSize);
    size_t newVecNum = utils::divUp(originTotal - removeCnt, multiFeaAttrBlkSize);
    for (size_t i = newVecNum; i < originVecNum; ++i) {
        attrTime.at(i)->clear();
        attrTokenQuotient.at(i)->clear();
        attrTokenRemainder.at(i)->clear();
        if (!attrVal.empty()) {
            attrVal.at(i)->clear();
        }
    }
    attrTime.resize(newVecNum);
    attrTokenQuotient.resize(newVecNum);
    attrTokenRemainder.resize(newVecNum);
    if (!attrVal.empty()) {
        attrVal.resize(newVecNum);
    }
}

void TSBase::getIdsByToken(uint32_t tokenId, std::vector<int64_t> &ids)
{
    APP_LOG_INFO("TSBase GetIdsByToken operation started.\n");
    ASCEND_THROW_IF_NOT_FMT(token2Idx.count(tokenId), "tokenId: %u is not exist.\n", tokenId);
    ASCEND_THROW_IF_NOT_FMT(token2Idx[tokenId].size(), "tokenId: %u map ids size 0.\n", tokenId);
    ids.assign(token2Idx[tokenId].begin(), token2Idx[tokenId].end());
    APP_LOG_INFO("TSBase GetIdsByToken operation end.\n");
}

APP_ERROR TSBase::getCustomAttrByBlockId(uint32_t blockId, uint8_t *&customAttr) const
{
    APP_LOG_INFO("TSBase getCustomAttrByBlockId operation started.\n");
    APPERR_RETURN_IF_NOT_LOG(customAttrBlockSize > 0 && customAttrLen > 0, APP_ERR_ILLEGAL_OPERATION,
        "customAttrBlockSize & customAttrLen must be > 0");
    ASCEND_THROW_IF_NOT_FMT(blockId < customAttrBase.size(),
        "blockId[%u] must be < customAttrBase size[%zu]", blockId, customAttrBase.size());
    customAttr = customAttrBase[blockId]->data();
    APP_LOG_INFO("TSBase getCustomAttrByBlockId operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR TSBase::getFeatureAttrsByLabel(int64_t n, const int64_t *labels, faiss::ascend::FeatureAttr *attrs) const
{
    APP_LOG_INFO("TSBase getFeatureAttrsByLabel operation started.\n");
    for (int64_t i = 0; i < n; ++i) {
        auto it = label2Idx.find(*(labels + i));
        if (it != label2Idx.end()) {
            size_t idx = it->second;
            attrs[i] = featureAttrs.at(idx);
        } else {
            attrs[i].time = INT32_MIN;
            attrs[i].tokenId = UINT32_MAX;
        }
    }
    APP_LOG_INFO("TSBase getFeatureAttrsByLabel operation end.\n");
    return APP_ERR_OK;
}

void TSBase::InitTimeAndTokenId(AscendTensor<int32_t, DIMS_1>& queryTime, AscendTensor<uint8_t, DIMS_1>& tokenIds,
    const faiss::ascend::AttrFilter *attrFilter)
{
    std::vector<uint8_t> tmpTokenBitSet(attrFilter->tokenBitSetLen * OPS_DATA_TYPE_TIMES, OPS_DATA_PADDING_VAL);
    for (uint32_t i = 0; i < attrFilter->tokenBitSetLen; i++) {
        tmpTokenBitSet[OPS_DATA_TYPE_TIMES * i] = *(attrFilter->tokenBitSet + i);
    }
    std::vector<int32_t> queryTimeVec(OPS_DATA_TYPE_ALIGN, 0);
    queryTimeVec[0] = this->enableTimeFilter ? (attrFilter->timesStart * -1) : 0;
    queryTimeVec[1] =
        this->enableTimeFilter ? (attrFilter->timesEnd * -1) : (std::numeric_limits<int32_t>::max() * -1);

    auto ret = aclrtMemcpy(queryTime.data(), queryTime.getSizeInBytes(), queryTimeVec.data(),
                           OPS_DATA_TYPE_ALIGN * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Copy querytime data to device failed, ret: %d.\n", ret);

    ret = aclrtMemcpy(tokenIds.data(), tokenIds.getSizeInBytes(), tmpTokenBitSet.data(), tmpTokenBitSet.size(),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Copy token ids data to device failed, ret: %d.\n", ret);
}

void TSBase::generateMaskWithExtra(const faiss::ascend::AttrFilter *attrFilter, int batchIndex,
                                   const uint8_t *extraMask, const uint64_t extraMaskLen,
                                   const bool extraMaskIsAtDevice,
                                   std::vector<std::unique_ptr<DeviceVector<uint8_t>>> &masks)
{
    APP_LOG_INFO("TSBase generateMaskWithExtra operation started.\n");
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = pResources->getMemoryManager();

    AscendTensor<int32_t, DIMS_1> queryTime(mem, {OPS_DATA_TYPE_ALIGN}, stream);
    AscendTensor<uint8_t, DIMS_1> tokenIds(mem,
        {static_cast<int32_t>(utils::divUp(tokenNum, OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES)}, stream);
    InitTimeAndTokenId(queryTime, tokenIds, attrFilter);

    int32_t blockNum = static_cast<int32_t>(utils::divUp(attrTotal, static_cast<int32_t>(featureAttrBlockSize)));
    int32_t blockSize = static_cast<int32_t>(utils::divUp(featureAttrBlockSize, OPS_DATA_TYPE_ALIGN));
    AscendTensor<uint8_t, DIMS_2> tmpExtraMask(mem,
        {blockNum, static_cast<int32_t>(utils::divUp(featureAttrBlockSize, OPS_DATA_TYPE_ALIGN))}, stream);
    auto ret = aclrtMemset(tmpExtraMask.data(), tmpExtraMask.getSizeInBytes(), 0, tmpExtraMask.getSizeInBytes());
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Memset extra mask data to device failed, ret: %d.\n", ret);
    ret = aclrtMemcpy(tmpExtraMask.data(), tmpExtraMask.getSizeInBytes(), extraMask, extraMaskLen,
                      extraMaskIsAtDevice ? ACL_MEMCPY_DEVICE_TO_DEVICE : ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Copy extra mask data to device failed, ret: %d.\n", ret);

    for (int32_t blockId = 0; blockId < blockNum; blockId++) {
        AscendTensor<int32_t, DIMS_1> baseTimes(calcAttrStartAddress(attrTime, blockId),
            {static_cast<int32_t>(featureAttrBlockSize)});
        AscendTensor<int32_t, DIMS_1> baseTokenQs(calcAttrStartAddress(attrTokenQuotient, blockId),
            {static_cast<int32_t>(featureAttrBlockSize)});
        AscendTensor<uint8_t, DIMS_1> baseTokenRs(
            calcAttrStartAddress(attrTokenRemainder, blockId, OPS_DATA_TYPE_TIMES),
            {static_cast<int32_t>(featureAttrBlockSize * OPS_DATA_TYPE_TIMES)});
        auto subExtraMask = tmpExtraMask[blockId].view();
        AscendTensor<uint8_t, DIMS_1> subMask(masks.at(blockId)->data() + (batchIndex * blockSize), {blockSize});
        runExtraMaskGenerateCompute(queryTime, tokenIds, baseTimes, baseTokenQs, baseTokenRs, subExtraMask, subMask,
                                    DEFAULT_ATTR_MEM_BLOCK_COUNT, stream);
    }

    ret = synchronizeStream(stream);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "extra mask gen synchronizeStream default stream: %i.\n", ret);
    APP_LOG_INFO("TSBase generateMaskWithExtra operation end.\n");
}

void TSBase::generateMaskWithExtra(const faiss::ascend::AttrFilter *attrFilter, const uint8_t *extraMask,
                                   const uint64_t extraMaskLen, const bool extraMaskIsAtDevice, uint8_t *masks)
{
    APP_LOG_INFO("TSBase generateMaskWithExtra operation started.\n");
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = pResources->getMemoryManager();
    AscendTensor<int32_t, DIMS_1> queryTime(mem, {OPS_DATA_TYPE_ALIGN}, stream);
    AscendTensor<uint8_t, DIMS_1> tokenIds(mem,
        {static_cast<int32_t>(utils::divUp(tokenNum, OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES)}, stream);
    InitTimeAndTokenId(queryTime, tokenIds, attrFilter);

    int32_t blockNum = static_cast<int32_t>(utils::divUp(attrTotal,
        static_cast<int32_t>(multiFeaAttrBlkSize)));
    int32_t blockSize = static_cast<int32_t>(utils::divUp(multiFeaAttrBlkSize,
        OPS_DATA_TYPE_ALIGN));
    AscendTensor<uint8_t, DIMS_2> tmpExtraMask(mem, {blockNum, blockSize}, stream);
    auto ret = aclrtMemset(tmpExtraMask.data(), tmpExtraMask.getSizeInBytes(), 0, tmpExtraMask.getSizeInBytes());
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Memset extra mask data to device failed, ret: %d.\n", ret);
    ret = aclrtMemcpy(tmpExtraMask.data(), tmpExtraMask.getSizeInBytes(), extraMask, extraMaskLen,
                      extraMaskIsAtDevice ? ACL_MEMCPY_DEVICE_TO_DEVICE : ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Copy extra mask data to device failed, ret: %d.\n", ret);

    for (int32_t blockId = 0; blockId < blockNum; blockId++) {
        AscendTensor<int32_t, DIMS_1> baseTimes(attrTime[blockId]->data(),
            {static_cast<int32_t>(multiFeaAttrBlkSize)});
        AscendTensor<int32_t, DIMS_1> baseTokenQs(attrTokenQuotient[blockId]->data(),
            {static_cast<int32_t>(multiFeaAttrBlkSize)});
        AscendTensor<uint8_t, DIMS_1> baseTokenRs(
            attrTokenRemainder[blockId]->data(),
            {static_cast<int32_t>(multiFeaAttrBlkSize * OPS_DATA_TYPE_TIMES)});
        auto subExtraMask = tmpExtraMask[blockId].view();
        AscendTensor<uint8_t, DIMS_1> subMask(masks + blockId * blockSize, {static_cast<int32_t>(blockSize)});
        runExtraMaskGenerateCompute(queryTime, tokenIds, baseTimes, baseTokenQs, baseTokenRs, subExtraMask, subMask,
                                    ATTR_MEM_BLOCK_COUNT, stream);
    }

    ret = synchronizeStream(stream);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "extra mask gen synchronizeStream default stream: %i.\n", ret);
    APP_LOG_INFO("TSBase generateMaskWithExtra operation end.\n");
}

void TSBase::generateMask(const faiss::ascend::AttrFilter *attrFilter, uint8_t *masks)
{
    ASCEND_THROW_IF_NOT_MSG(attrFilter, "Invalid filter.\n");
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    auto &mem = pResources->getMemoryManager();
    AscendTensor<int32_t, DIMS_1> queryTime(mem, {OPS_DATA_TYPE_ALIGN}, stream);
    AscendTensor<uint8_t, DIMS_1> tokenIds(mem,
        {static_cast<int32_t>(utils::divUp(tokenNum, OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES)}, stream);
    InitTimeAndTokenId(queryTime, tokenIds, attrFilter);
    int32_t blockNum = static_cast<int32_t>(utils::divUp(attrTotal, static_cast<int32_t>(featureAttrBlockSize)));
    int blockMaskLen = static_cast<int32_t>(utils::divUp(featureAttrBlockSize, OPS_DATA_TYPE_ALIGN));
    for (int32_t blockId = 0; blockId < blockNum; blockId += ATTR_MEM_BLOCK_COUNT) {
        AscendTensor<int32_t, DIMS_1> baseTimes(calcAttrStartAddress(attrTime, blockId),
            {static_cast<int32_t>(multiFeaAttrBlkSize)});
        AscendTensor<int32_t, DIMS_1> baseTokenQs(calcAttrStartAddress(attrTokenQuotient, blockId),
            {static_cast<int32_t>(multiFeaAttrBlkSize)});
        AscendTensor<uint8_t, DIMS_1> baseTokenRs(
            calcAttrStartAddress(attrTokenRemainder, blockId, OPS_DATA_TYPE_TIMES),
            {static_cast<int32_t>(multiFeaAttrBlkSize * OPS_DATA_TYPE_TIMES)});
        AscendTensor<uint8_t, DIMS_1> subMask(masks + blockId * blockMaskLen,
            {static_cast<int32_t>(blockMaskLen * ATTR_MEM_BLOCK_COUNT)});
        runMaskGenerateCompute(queryTime, tokenIds, baseTimes, baseTokenQs, baseTokenRs, subMask,
                               ATTR_MEM_BLOCK_COUNT, stream);
    }
    auto ret = synchronizeStream(stream);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "synchronizeStream default stream: %i.\n", ret);
}

void TSBase::buildAttr(const faiss::ascend::AttrFilter *attrFilter, int batch,
    AscendTensor<int32_t, DIMS_2> &queryTime, AscendTensor<uint8_t, DIMS_2> &tokenIds)
{
    ASCEND_THROW_IF_NOT_MSG(attrFilter, "Invalid filter.\n");

    int maxTokenValue = utils::divUp(static_cast<int32_t>(tokenNum), OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES;
    std::vector<int32_t> queryTimeVec(batch * OPS_DATA_TYPE_ALIGN, 0);
    std::vector<uint8_t> tokenIdsVec(batch * maxTokenValue, 0);

    for (int i = 0; i < batch; i++) {
        queryTimeVec[i * OPS_DATA_TYPE_ALIGN] = this->enableTimeFilter ? ((attrFilter + i)->timesStart * -1) : 0;
        queryTimeVec[i * OPS_DATA_TYPE_ALIGN + 1] =
            this->enableTimeFilter ? ((attrFilter + i)->timesEnd * -1) : (std::numeric_limits<int32_t>::max() * -1);
        for (uint32_t j = 0; j < (attrFilter + i)->tokenBitSetLen; j++) {
            tokenIdsVec[i * maxTokenValue + OPS_DATA_TYPE_TIMES * static_cast<int32_t>(j) + 1] = OPS_DATA_PADDING_VAL;
            tokenIdsVec[i * maxTokenValue + OPS_DATA_TYPE_TIMES * static_cast<int32_t>(j)] =
                *((attrFilter + i)->tokenBitSet + static_cast<int32_t>(j));
        }
    }

    auto ret = aclrtMemcpy(queryTime.data(), queryTime.getSizeInBytes(), queryTimeVec.data(),
        batch * OPS_DATA_TYPE_ALIGN * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Copy querytime data to device failed.\n");

    ret = aclrtMemcpy(tokenIds.data(), tokenIds.getSizeInBytes(), tokenIdsVec.data(), batch * maxTokenValue,
        ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "Copy token ids data to device failed.\n");
}

void TSBase::generateMask(int batch, int blockOffset, int blockNum,
    AscendTensor<int32_t, DIMS_2> &queryTimes, AscendTensor<uint8_t, DIMS_2> &tokenIds,
    AscendTensor<uint8_t, DIMS_3> &masks)
{
    auto streamPtr = pResources->getDefaultStream();
    auto stream = streamPtr->GetStream();
    for (int32_t blockId = 0; blockId < blockNum; blockId++) {
            AscendTensor<int32_t, DIMS_1> baseTimes(calcAttrStartAddress(attrTime, blockId + blockOffset),
                { static_cast<int32_t>(featureAttrBlockSize) });
            AscendTensor<int32_t, DIMS_1> baseTokenQs(calcAttrStartAddress(attrTokenQuotient, blockId + blockOffset),
                { static_cast<int32_t>(featureAttrBlockSize) });
            AscendTensor<uint8_t, DIMS_1> baseTokenRs(
                calcAttrStartAddress(attrTokenRemainder, blockId + blockOffset, OPS_DATA_TYPE_TIMES),
                { static_cast<int32_t>(featureAttrBlockSize * OPS_DATA_TYPE_TIMES) });

        auto subMask = masks[blockId].view();
        std::vector<const AscendTensorBase *> input {
            &queryTimes, &tokenIds, &baseTimes, &baseTokenQs, &baseTokenRs};
        std::vector<const AscendTensorBase *> output {&subMask};
        runBatchMaskGenerateCompute(batch, input, output, stream);
    }
    auto ret = synchronizeStream(stream);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "synchronizeStream default stream: %i.\n", ret);
}

void TSBase::generateMaskWithExtra(int batch, int blockOffset, int blockNum, AscendTensor<int32_t, DIMS_2> &queryTimes,
    AscendTensor<uint8_t, DIMS_2> &tokenIds, const uint8_t *extraMask, AscendTensor<uint8_t, DIMS_3> &masks,
    const AscendTensor<uint8_t, DIMS_3> &baseMaskDev)
{
    auto streamPtr = pResources->getDefaultStream();
    auto &mem = pResources->getMemoryManager();
    uint64_t blockSize = utils::divUp(featureAttrBlockSize, OPS_DATA_TYPE_ALIGN);
    AscendTensor<uint8_t, DIMS_2> extraMaskInDev;
    int32_t maskLenInBlock = std::max(extraMaskLen, blockSize);
    if (extraMaskIsAtDevice) {
        AscendTensor<uint8_t, DIMS_2> tmpExtraMaskInDev(const_cast<uint8_t *>(extraMask), { batch, maskLenInBlock });
        extraMaskInDev = std::move(tmpExtraMaskInDev);
    } else {
        AscendTensor<uint8_t, DIMS_2> tmpExtraMaskInDev(mem, { batch, maskLenInBlock }, streamPtr->GetStream());
        auto ret = aclrtMemcpy(tmpExtraMaskInDev.data(), tmpExtraMaskInDev.getSizeInBytes(), extraMask,
            batch * extraMaskLen, ACL_MEMCPY_HOST_TO_DEVICE);
        ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy extra mask data to device failed, ret: %d.\n", ret);
        extraMaskInDev = std::move(tmpExtraMaskInDev);
    }

    std::vector<int32_t> opSizeHost(blockNum * MASK_ALIGN);
    int opSizeHostIdx = 0;
    for (int32_t blockId = 0; blockId < blockNum; blockId++) {
        int32_t curBlockOffset = (blockOffset + blockId) * static_cast<int32_t>(blockSize);
        int32_t idxUseExtraMask = (this->extraMaskLen > static_cast<uint32_t>(curBlockOffset)) ? 1 : 0;
        int32_t idxActualMaskLen = static_cast<int32_t>((this->extraMaskLen > (static_cast<uint32_t>(
            curBlockOffset) + blockSize)) ? blockSize : (this->extraMaskLen - static_cast<uint32_t>(curBlockOffset)));
        opSizeHost[opSizeHostIdx + IDX_BLOCK_OFFSET] = curBlockOffset;
        opSizeHost[opSizeHostIdx + IDX_EXTRA_MASK_LEN] = static_cast<int32_t>(this->extraMaskLen);
        opSizeHost[opSizeHostIdx + IDX_USE_EXTRA_MASK] = idxUseExtraMask;
        opSizeHost[opSizeHostIdx + IDX_ACTUAL_MASK_LEN] = idxActualMaskLen;
        opSizeHostIdx += MASK_ALIGN;
    }

    AscendTensor<int32_t, DIMS_2> opSize(mem, { blockNum, MASK_ALIGN }, streamPtr->GetStream());
    auto ret = aclrtMemcpy(opSize.data(), opSize.getSizeInBytes(), opSizeHost.data(),
        opSizeHost.size() * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy opSize data to device failed, ret: %d.\n", ret);

    for (int32_t blockId = 0; blockId < blockNum; blockId++) {
        int32_t blockIdx = blockId + blockOffset;
        AscendTensor<int32_t, DIMS_1> baseTimes(calcAttrStartAddress(attrTime, blockIdx),
            { static_cast<int32_t>(featureAttrBlockSize) });
        AscendTensor<int32_t, DIMS_1> baseTokenQs(calcAttrStartAddress(attrTokenQuotient, blockIdx),
            { static_cast<int32_t>(featureAttrBlockSize) });
        AscendTensor<uint8_t, DIMS_1> baseTokenRs(
            calcAttrStartAddress(attrTokenRemainder, blockIdx, OPS_DATA_TYPE_TIMES),
            { static_cast<int32_t>(featureAttrBlockSize * OPS_DATA_TYPE_TIMES) });
        auto extreaMaskAttr = opSize[blockId].view();
        auto subMask = masks[blockId].view();
        std::vector<const AscendTensorBase *> input { &queryTimes, &tokenIds, &baseTimes, &baseTokenQs,
            &baseTokenRs, &extraMaskInDev, &extreaMaskAttr };
        std::vector<const AscendTensorBase *> output { &subMask };
        if (baseMaskDev.getSize(0) > 0) {
            AscendTensor<uint8_t, DIMS_2> subBaseMask(baseMaskDev[blockId].data(), {1, baseMaskDev.getSize(2)});
            input.emplace_back(&subBaseMask);
            runBatchExtraAndBaseMaskGenerateCompute(batch, input, output, streamPtr->GetStream());
        } else {
            runBatchExtraMaskGenerateCompute(batch, input, output, streamPtr->GetStream());
        }
    }

    ret = synchronizeStream(streamPtr->GetStream());
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "synchronizeStream default stream: %i.\n", ret);
}

void TSBase::setSaveHostMemory()
{
    enableSaveHostMemory = true;
}

void TSBase::resetMaskGenerateComputeOp()
{
    APP_LOG_INFO("TSBase resetMaskGenerateComputeOp operation started.\n");
    auto maskCompOpReset = [&](std::unique_ptr<AscendOperator> &op, uint32_t blockCount) {
        AscendOpDesc desc("DistanceMaskGenerator");

        const uint32_t featureAttrNum = featureAttrBlockSize * blockCount;
        std::vector<int64_t> input_shape0({OPS_DATA_TYPE_ALIGN});
        std::vector<int64_t> input_shape1({utils::divUp(tokenNum, OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES});
        std::vector<int64_t> input_shape2({featureAttrNum});
        std::vector<int64_t> input_shape3({featureAttrNum});
        std::vector<int64_t> input_shape4({featureAttrNum * OPS_DATA_TYPE_TIMES});
        std::vector<int64_t> output_shape0({utils::divUp(featureAttrNum,
                                                         OPS_DATA_TYPE_ALIGN)});
        desc.addInputTensorDesc(ACL_INT32, input_shape0.size(), input_shape0.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT8, input_shape1.size(), input_shape1.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, input_shape2.size(), input_shape2.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, input_shape3.size(), input_shape3.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT8, input_shape4.size(), input_shape4.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_UINT8, output_shape0.size(), output_shape0.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    maskGenerateComputeOpMap[DEFAULT_ATTR_MEM_BLOCK_COUNT] = std::unique_ptr<AscendOperator>(nullptr);
    maskGenerateComputeOpMap[ATTR_MEM_BLOCK_COUNT] = std::unique_ptr<AscendOperator>(nullptr);
    ASCEND_THROW_IF_NOT_MSG(maskCompOpReset(maskGenerateComputeOpMap[DEFAULT_ATTR_MEM_BLOCK_COUNT],
        DEFAULT_ATTR_MEM_BLOCK_COUNT), "DistanceMaskGenerator op init failed.\n");
    ASCEND_THROW_IF_NOT_MSG(maskCompOpReset(maskGenerateComputeOpMap[ATTR_MEM_BLOCK_COUNT], ATTR_MEM_BLOCK_COUNT),
        "DistanceMaskGenerator op init failed.\n");
    APP_LOG_INFO("TSBase resetMaskGenerateComputeOp resetDistCompOp operation end.\n");
}

void TSBase::resetExtraMaskGenerateComputeOp()
{
    APP_LOG_INFO("TSBase resetExtraMaskGenerateComputeOp operation started.\n");
    auto extraMaskCompOpReset = [&](std::unique_ptr<AscendOperator> &op, uint32_t blockCount) {
        AscendOpDesc desc("DistanceMaskGeneratorWithExtra");

        const uint32_t featureAttrNum = featureAttrBlockSize * blockCount;
        std::vector<int64_t> input_shape0({OPS_DATA_TYPE_ALIGN});
        std::vector<int64_t> input_shape1({utils::divUp(tokenNum, OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES});
        std::vector<int64_t> input_shape2({featureAttrNum});
        std::vector<int64_t> input_shape3({featureAttrNum});
        std::vector<int64_t> input_shape4({featureAttrNum * OPS_DATA_TYPE_TIMES});
        std::vector<int64_t> input_shape5({utils::divUp(featureAttrNum, OPS_DATA_TYPE_ALIGN)});
        std::vector<int64_t> output_shape0({utils::divUp(featureAttrNum, OPS_DATA_TYPE_ALIGN)});
        desc.addInputTensorDesc(ACL_INT32, input_shape0.size(), input_shape0.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT8, input_shape1.size(), input_shape1.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, input_shape2.size(), input_shape2.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, input_shape3.size(), input_shape3.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT8, input_shape4.size(), input_shape4.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT8, input_shape5.size(), input_shape5.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_UINT8, output_shape0.size(), output_shape0.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    extraMaskGenerateComputeOpMap[DEFAULT_ATTR_MEM_BLOCK_COUNT] = std::unique_ptr<AscendOperator>(nullptr);
    extraMaskGenerateComputeOpMap[ATTR_MEM_BLOCK_COUNT] = std::unique_ptr<AscendOperator>(nullptr);
    ASCEND_THROW_IF_NOT_MSG(extraMaskCompOpReset(extraMaskGenerateComputeOpMap[DEFAULT_ATTR_MEM_BLOCK_COUNT],
        DEFAULT_ATTR_MEM_BLOCK_COUNT), "DistanceMaskGeneratorWithExtra op init failed.\n");
    ASCEND_THROW_IF_NOT_MSG(extraMaskCompOpReset(extraMaskGenerateComputeOpMap[ATTR_MEM_BLOCK_COUNT],
        ATTR_MEM_BLOCK_COUNT), "DistanceMaskGeneratorWithExtra op init failed.\n");
    APP_LOG_INFO("TSBase resetExtraMaskGenerateComputeOp operation end.\n");
}

void TSBase::runMaskGenerateCompute(const AscendTensor<int32_t, DIMS_1> &queryTime,
                                    const AscendTensor<uint8_t, DIMS_1> &tokenBitSet,
                                    const AscendTensor<int32_t, DIMS_1> &attrTimes,
                                    const AscendTensor<int32_t, DIMS_1> &attrTokenQs,
                                    const AscendTensor<uint8_t, DIMS_1> &attrTokenRs,
                                    AscendTensor<uint8_t, DIMS_1> &outMask,
                                    uint32_t blockCount,
                                    aclrtStream stream)
{
    APP_LOG_INFO("TSBase runMaskGenerateCompute operation started.\n");

    AscendOperator *op = maskGenerateComputeOpMap[blockCount].get();
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    distOpInput->emplace_back(aclCreateDataBuffer(queryTime.data(), queryTime.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(tokenBitSet.data(), tokenBitSet.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(attrTimes.data(), attrTimes.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(attrTokenQs.data(), attrTokenQs.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(attrTokenRs.data(), attrTokenRs.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(outMask.data(), outMask.getSizeInBytes()));

    op->exec(*distOpInput, *distOpOutput, stream);
    APP_LOG_INFO("TSBase runMaskGenerateCompute operation end.\n");
}

void TSBase::runExtraMaskGenerateCompute(const AscendTensor<int32_t, DIMS_1> &queryTime,
                                         const AscendTensor<uint8_t, DIMS_1> &tokenBitSet,
                                         const AscendTensor<int32_t, DIMS_1> &attrTimes,
                                         const AscendTensor<int32_t, DIMS_1> &attrTokenQs,
                                         const AscendTensor<uint8_t, DIMS_1> &attrTokenRs,
                                         const AscendTensor<uint8_t, DIMS_1> &extraMask,
                                         AscendTensor<uint8_t, DIMS_1> &outMask,
                                         uint32_t blockCount,
                                         aclrtStream stream)
{
    APP_LOG_INFO("TSBase runExtraMaskGenerateCompute operation started.\n");

    AscendOperator *op = extraMaskGenerateComputeOpMap[blockCount].get();
    ASCEND_THROW_IF_NOT(op);

    std::shared_ptr<std::vector<const aclDataBuffer *>> distOpInput(
        new std::vector<const aclDataBuffer *>(), CommonUtils::AclInputBufferDelete);
    distOpInput->emplace_back(aclCreateDataBuffer(queryTime.data(), queryTime.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(tokenBitSet.data(), tokenBitSet.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(attrTimes.data(), attrTimes.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(attrTokenQs.data(), attrTokenQs.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(attrTokenRs.data(), attrTokenRs.getSizeInBytes()));
    distOpInput->emplace_back(aclCreateDataBuffer(extraMask.data(), extraMask.getSizeInBytes()));

    std::shared_ptr<std::vector<aclDataBuffer *>> distOpOutput(
        new std::vector<aclDataBuffer *>(), CommonUtils::AclOutputBufferDelete);
    distOpOutput->emplace_back(aclCreateDataBuffer(outMask.data(), outMask.getSizeInBytes()));

    op->exec(*distOpInput, *distOpOutput, stream);
    APP_LOG_INFO("TSBase runExtraMaskGenerateCompute operation end.\n");
}

APP_ERROR TSBase::resetBatchMaskGenerateComputeOp() const
{
    APP_LOG_INFO("TSBase resetBatchMaskGenerateComputeOp operation started.\n");
    std::string opTypeName = "DistanceBatchMaskGenerator";
    IndexTypeIdx indexMaskType = IndexTypeIdx::ITI_MASK_GENERATOR;
    
    if (faiss::ascend::SocUtils::GetInstance().IsAscend910B()) {
        opTypeName = "AscendcDistanceBatchMaskGenerator";
        indexMaskType = IndexTypeIdx::ASCENDC_ITI_MASK_GENERATOR;
    }

    for (auto batch : MASK_BATCH) {
        std::vector<int64_t> queryTime({ batch, OPS_DATA_TYPE_ALIGN });
        std::vector<int64_t> tokenBitSet({ batch, utils::divUp(tokenNum, OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES });
        std::vector<int64_t> attrTimes({ featureAttrBlockSize });
        std::vector<int64_t> attrTokenQs({ featureAttrBlockSize });
        std::vector<int64_t> attrTokenRs({ featureAttrBlockSize * OPS_DATA_TYPE_TIMES });
        std::vector<int64_t> outMask({ batch, utils::divUp(featureAttrBlockSize, OPS_DATA_TYPE_ALIGN) });

        std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
            { ACL_INT32, queryTime },
            { ACL_UINT8, tokenBitSet },
            { ACL_INT32, attrTimes },
            { ACL_INT32, attrTokenQs },
            { ACL_UINT8, attrTokenRs }
        };
        std::vector<std::pair<aclDataType, std::vector<int64_t>>> output {
            { ACL_UINT8, outMask }
        };
        std::vector<int> keys({static_cast<int>(batch), static_cast<int>(tokenNum)});
        OpsMngKey opsKey(keys);
        auto ret = DistComputeOpsManager::getInstance().resetOp(opTypeName, indexMaskType, opsKey, input, output);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "op init failed: %i", ret);
    }
    APP_LOG_INFO("TSBase resetBatchMaskGenerateComputeOp resetDistCompOp operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR TSBase::resetBatchValMaskGenerateComputeOp() const
{
    APP_LOG_INFO("TSBase resetBatchValMaskGenerateComputeOp operation started.\n");
    std::string opTypeName = "DistanceBatchValMaskGenerator";
    IndexTypeIdx indexMaskType = IndexTypeIdx::ITI_MASK_WITH_VAL_GENERATOR;

    for (auto batch : MASK_BATCH) {
        std::vector<int64_t> queryTime({ batch, OPS_DATA_TYPE_ALIGN });
        std::vector<int64_t> tokenBitSet({ batch, utils::divUp(tokenNum, OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES });
        std::vector<int64_t> extraValFilter({ batch, EXTRA_VAL_ALIGN });
        std::vector<int64_t> attrTimes({ featureAttrBlockSize });
        std::vector<int64_t> attrVals({ featureAttrBlockSize });
        std::vector<int64_t> attrTokenQs({ featureAttrBlockSize });
        std::vector<int64_t> attrTokenRs({ featureAttrBlockSize * OPS_DATA_TYPE_TIMES });
        std::vector<int64_t> outMask({ batch, utils::divUp(featureAttrBlockSize, OPS_DATA_TYPE_ALIGN) });

        std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
            { ACL_INT32, queryTime },
            { ACL_UINT8, tokenBitSet },
            { ACL_INT32, attrTimes },
            { ACL_INT32, attrTokenQs },
            { ACL_UINT8, attrTokenRs },
            { ACL_INT16, extraValFilter },
            { ACL_INT16, attrVals }
        };
        std::vector<std::pair<aclDataType, std::vector<int64_t>>> output {
            { ACL_UINT8, outMask }
        };
        std::vector<int> keys({static_cast<int>(batch), static_cast<int>(tokenNum)});
        OpsMngKey opsKey(keys);
        auto ret = DistComputeOpsManager::getInstance().resetOp(opTypeName, indexMaskType, opsKey, input, output);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "op init failed: %i", ret);
    }
    APP_LOG_INFO("TSBase resetBatchValMaskGenerateComputeOp resetDistCompOp operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR TSBase::resetBatchExtraMaskGenerateComputeOp(bool withBaseMask) const
{
    APP_LOG_INFO("TSBase resetBatchExtraMaskGenerateComputeOp operation started.\n");

    std::string opTypeName = withBaseMask ? "DistanceBatchMaskGeneratorWithExtraAndBaseMask":
                                            "DistanceBatchMaskGeneratorWithExtra";
    IndexTypeIdx indexMaskType = withBaseMask ? IndexTypeIdx::ITI_MASK_WITH_EXTRA_AND_BASEMASK_GENERATOR :
                                 IndexTypeIdx::ITI_MASK_WITH_EXTRA_GENERATOR;

    if (faiss::ascend::SocUtils::GetInstance().IsAscend910B()) {
        opTypeName = "AscendcDistanceBatchMaskGeneratorWithExtra";
        indexMaskType = IndexTypeIdx::ASCENDC_ITI_MASK_WITH_EXTRA_GENERATOR;
    }

    for (auto batch : MASK_BATCH) {
        std::vector<int64_t> queryTime({ batch, OPS_DATA_TYPE_ALIGN });
        std::vector<int64_t> tokenBitSet({ batch, utils::divUp(tokenNum, OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES });
        std::vector<int64_t> attrTimes({ featureAttrBlockSize });
        std::vector<int64_t> attrTokenQs({ featureAttrBlockSize });
        std::vector<int64_t> attrTokenRs({ featureAttrBlockSize * OPS_DATA_TYPE_TIMES });
        std::vector<int64_t> extraMask({ batch, utils::divUp(featureAttrBlockSize, OPS_DATA_TYPE_ALIGN) });
        std::vector<int64_t> extraMaskAttr({ OPS_DATA_TYPE_ALIGN });
        std::vector<int64_t> baseMask;
        std::vector<int64_t> outMask({ batch, utils::divUp(featureAttrBlockSize, OPS_DATA_TYPE_ALIGN) });
        std::vector<std::pair<aclDataType, std::vector<int64_t>>> input {
            { ACL_INT32, queryTime },
            { ACL_UINT8, tokenBitSet },
            { ACL_INT32, attrTimes },
            { ACL_INT32, attrTokenQs },
            { ACL_UINT8, attrTokenRs },
            { ACL_UINT8, extraMask },
            { ACL_INT32, extraMaskAttr }
        };
        if (withBaseMask) {
            baseMask = std::vector<int64_t>({ 1, utils::divUp(featureAttrBlockSize, OPS_DATA_TYPE_ALIGN) });
            input.emplace_back(ACL_UINT8, baseMask);
        }
        std::vector<std::pair<aclDataType, std::vector<int64_t>>> output {
            { ACL_UINT8, outMask }
        };
        std::vector<int> keys({static_cast<int>(batch), static_cast<int>(tokenNum)});
        OpsMngKey opsKey(keys);
        auto ret = DistComputeOpsManager::getInstance().resetOp(opTypeName, indexMaskType, opsKey, input, output);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, ret, "op init failed: %i", ret);
    }

    APP_LOG_INFO("TSBase resetBatchExtraMaskGenerateComputeOp operation end.\n");
    return APP_ERR_OK;
}

void TSBase::runBatchMaskGenerateCompute(int batch, const std::vector<const AscendTensorBase *> &input,
    const std::vector<const AscendTensorBase *> &output, aclrtStream stream) const
{
    APP_LOG_INFO("TSBase runBatchMaskGenerateCompute operation started.\n");
    IndexTypeIdx indexType = IndexTypeIdx::ITI_MASK_GENERATOR;
    if (faiss::ascend::SocUtils::GetInstance().IsAscend910B()) {
        indexType = IndexTypeIdx::ASCENDC_ITI_MASK_GENERATOR;
    }
    std::vector<int> keys({batch, static_cast<int>(tokenNum)});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().runOp(indexType, opsKey, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
    APP_LOG_INFO("TSBase runBatchMaskGenerateCompute operation end.\n");
}

void TSBase::runBatchMaskValGenerateCompute(int batch, const std::vector<const AscendTensorBase *> &input,
    const std::vector<const AscendTensorBase *> &output, aclrtStream stream) const
{
    APP_LOG_INFO("TSBase runBatchMaskValGenerateCompute operation started.\n");
    IndexTypeIdx indexType = IndexTypeIdx::ITI_MASK_WITH_VAL_GENERATOR;
    std::vector<int> keys({batch, static_cast<int>(tokenNum)});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().runOp(indexType, opsKey, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
    APP_LOG_INFO("TSBase runBatchMaskValGenerateCompute operation end.\n");
}

void TSBase::runBatchExtraMaskGenerateCompute(int batch, const std::vector<const AscendTensorBase *> &input,
    const std::vector<const AscendTensorBase *> &output, aclrtStream stream) const
{
    APP_LOG_INFO("TSBase runBatchExtraMaskGenerateCompute operation started.\n");
    IndexTypeIdx indexType = IndexTypeIdx::ITI_MASK_WITH_EXTRA_GENERATOR;
    if (faiss::ascend::SocUtils::GetInstance().IsAscend910B()) {
        indexType = IndexTypeIdx::ASCENDC_ITI_MASK_WITH_EXTRA_GENERATOR;
    }
    std::vector<int> keys({batch, static_cast<int>(tokenNum)});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().runOp(indexType, opsKey, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
    APP_LOG_INFO("TSBase runBatchExtraMaskGenerateCompute operation end.\n");
}

void TSBase::runBatchExtraAndBaseMaskGenerateCompute(int batch, const std::vector<const AscendTensorBase *> &input,
    const std::vector<const AscendTensorBase *> &output, aclrtStream stream) const
{
    APP_LOG_INFO("TSBase runBatchExtraAndBaseMaskGenerateCompute operation started.\n");
    IndexTypeIdx indexType = IndexTypeIdx::ITI_MASK_WITH_EXTRA_AND_BASEMASK_GENERATOR;
    std::vector<int> keys({batch, static_cast<int>(tokenNum)});
    OpsMngKey opsKey(keys);
    auto ret = DistComputeOpsManager::getInstance().runOp(indexType, opsKey, input, output, stream);
    ASCEND_THROW_IF_NOT_FMT(ret == APP_ERR_OK, "run operator failed: %i\n", ret);
    APP_LOG_INFO("TSBase runBatchExtraAndBaseMaskGenerateCompute operation end.\n");
}

APP_ERROR TSBase::CheckIndices(int64_t ntotal, int64_t n, const int64_t* indices, int64_t &replaceNum,
    std::vector<std::pair<int64_t, int64_t>> &segments)
{
    APPERR_RETURN_IF_NOT_FMT(n > 0, APP_ERR_INVALID_PARAM, "n(%ld) must > 0", n);
    APPERR_RETURN_IF_NOT_FMT(indices[0] >= 0, APP_ERR_INVALID_PARAM, "indices[0] (%ld) must >= 0", indices[0]);
    replaceNum = n;
    bool found = false;  // 第一次大于等于ntotal时记录位置
    int64_t start = 0;   // 当前连续段的起始索引
    int64_t length = 1;  // 当前连续段的长度
    segments.clear();
    for (int64_t i = 0; i < n; i++) {
        // indice必须递增
        if (i > 0) {
            APPERR_RETURN_IF_NOT_FMT(indices[i] > indices[i - 1], APP_ERR_INVALID_PARAM,
                "indices must be in ascending order, indices[%ld]:%ld, indices[%ld]:%ld",
                i, indices[i], i - 1, indices[i - 1]);
            // 检查是否连续（当前元素 = 前一个元素 + 1）
            if (indices[i] == indices[i - 1] + 1) {
                length++;
            } else {
                // 保存当前段信息（起始索引，长度）
                segments.emplace_back(start, length);
                // 重置下一段的起始位置和长度
                start = i;
                length = 1;
            }
        }
        // 当前已经是新增的，则要求是连续的
        if (found) {
            APPERR_RETURN_IF_NOT_FMT(indices[i] == indices[i - 1] + 1, APP_ERR_INVALID_PARAM,
                "The newly added indice must be continuous, but ntotal(%ld), indices[%ld] is %ld, indices[%ld] is %ld",
                ntotal, i, indices[i], i - 1, indices[i - 1]);
        }
        // 如果比ntotal大或等于时，表示有新增
        if (!found && indices[i] >= ntotal) {
            // 新增起始位置必须等于ntotal
            APPERR_RETURN_IF_NOT_FMT(indices[i] == ntotal, APP_ERR_INVALID_PARAM,
                "The newly added indice(%ld) must be starting from the ntotal(%ld)",
                indices[i], ntotal);
            replaceNum = i; // 需要进行替换的数量
            found = true;
        }
    }
    // 添加最后一段
    segments.emplace_back(start, length);

    int64_t newAddN = (indices[n - 1] >= ntotal) ? (indices[n - 1] - ntotal + 1) : 0;
    APPERR_RETURN_IF_NOT_FMT(replaceNum + newAddN == n, APP_ERR_INVALID_PARAM,
        "replaceNum(%ld) + newAddN(%ld) must be == n(%ld)", replaceNum, newAddN, n);
    return APP_ERR_OK;
}
} // namespace ascend
