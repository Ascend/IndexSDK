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


#ifndef OCK_VSA_NPU_ANN_INDEX_ADD_FEATURE_H
#define OCK_VSA_NPU_ANN_INDEX_ADD_FEATURE_H

#include <set>
#include <stdlib.h>
#include "ock/vsa/neighbor/npu/impl/OckVsaAnnNpuIndexSystem.h"
#include "ock/acladapter/utils/OckSyncUtils.h"
#include "ock/hcps/nop/OckOpConst.h"
#include "ock/hcps/nop/l2_norm_op/OckL2NormOpRun.h"
#include "ock/utils/OckSafeUtils.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace npu {
namespace impl {
inline uint64_t CalcAddNum(
    uint64_t needAddCount, uint64_t lastBlockRowCount, uint64_t curRowCount, const OckVsaAnnCreateParam &param)
{
    if (curRowCount >= param.MaxFeatureRowCount()) {
        return 0ULL;
    }
    uint64_t leftRowSpace = param.MaxFeatureRowCount() - curRowCount;
    uint64_t leftBlockRowCount = param.BlockRowCount() - lastBlockRowCount;
    return std::min(std::min(needAddCount, leftRowSpace), leftBlockRowCount);
}
}  // namespace impl
template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::AddFeature(
    const OckVsaAnnAddFeatureParam<DataTemp, KeyTraitTemp> &featureParam)
{
    if (featureParam.features == nullptr || featureParam.attributes == nullptr || featureParam.labels == nullptr) {
        OCK_HCPS_LOG_ERROR("Input param cannot be nullptr");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    OckVsaAnnAddFeatureParam<DataTemp, KeyTraitTemp> leftFeatureParam = featureParam;

    while (leftFeatureParam.count > 0) {
        OCK_CHECK_RETURN_ERRORCODE(PrepareOneFeatureBlock());

        auto curGrp = blockGroups.back();
        if (curGrp == nullptr) {
            return VSA_ERROR_EMPTY_BASE;
        }
        uint64_t addNum = impl::CalcAddNum(leftFeatureParam.count, curGrp->lastBlockRowCount, GetFeatureNum(), *param);
        if (addNum == 0) {
            OCK_HCPS_LOG_ERROR(
                "GetFeatureNum: " << GetFeatureNum() << " Exceed maxFeatureBlockCount:" << param->MaxFeatureRowCount()
                                  << " leftFeatureParam.count:" << leftFeatureParam.count
                                  << ", rawFeatureParam.count:" << featureParam.count
                                  << " lastBlockRowCount:" << curGrp->lastBlockRowCount << " param:" << *param);
            return VSA_ERROR_EXCEED_NPU_INDEX_MAX_FEATURE_NUMBER;
        }

        OCK_CHECK_RETURN_ERRORCODE(AddTimeSpaceAttr(addNum, leftFeatureParam.attributes, curGrp));
        OCK_CHECK_RETURN_ERRORCODE(AddBaseFeature(addNum, leftFeatureParam.features, curGrp));
        if (param->ExtKeyAttrByteSize() > 0 && param->ExtKeyAttrBlockSize() > 0) {
            if (leftFeatureParam.customAttr == nullptr) {
                OCK_HCPS_LOG_ERROR("the customAttr is nullptr!");
                return hcps::HCPS_ERROR_CUSTOM_ATTR_EMPTY;
            }
            OCK_CHECK_RETURN_ERRORCODE(AddCustomAttr(addNum, leftFeatureParam.customAttr, curGrp));
        }

        AddIds(addNum, leftFeatureParam, curGrp);
        curGrp->lastBlockRowCount += static_cast<uint32_t>(addNum);
        curGrp->rowCount += static_cast<uint32_t>(addNum);
        leftFeatureParam.Shift(addNum, DimSizeTemp, param->ExtKeyAttrByteSize());
    }
    return VSA_SUCCESS;
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::AddTimeSpaceAttr(
    uint64_t addNum, const KeyTypeTupleT *attributes, std::shared_ptr<OckVsaAnnNpuBlockGroup> curGrp)
{
    std::vector<int32_t> attrTimes(addNum);
    std::vector<int32_t> attrTokenQuotients(addNum);
    std::vector<int8_t> attrTokenRemainders(addNum * OPS_DATA_TYPE_TIMES);
    if (curGrp == nullptr) {
        OCK_HCPS_LOG_ERROR("Input param cannot be nullptr");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    uint32_t mapStartPos = curGrp->rowCount;
    auto tokenIdxMap = tokenIdxVectorMap.back();

    for (uint64_t i = 0; i < addNum; ++i) {
        // attr结构体需包含time、tokenId
        auto keyAttr = std::get<0>(*(attributes + i));
        attrTimes[i] = keyAttr.time;
        if (keyAttr.tokenId >= param->TokenNum()) {
            OCK_HCPS_LOG_ERROR("Invalid tokenId, should be smaller than " << param->TokenNum() << ", input is " <<
                keyAttr.tokenId);
            return VSA_ERROR_INVALID_INPUT_PARAM;
        }
        attrTokenQuotients[i] =
            (static_cast<int>(keyAttr.tokenId) / __CHAR_BIT__) * OPS_DATA_TYPE_TIMES;  // tokenId < maxTokennum
        attrTokenRemainders[i * OPS_DATA_TYPE_TIMES] = static_cast<int8_t>(1 << (keyAttr.tokenId % __CHAR_BIT__));
        attrTokenRemainders[i * OPS_DATA_TYPE_TIMES + 1] = OPS_DATA_PADDING_VAL;
        tokenIdxMap.back()->AddData(keyAttr.tokenId, static_cast<uint32_t>(mapStartPos + i));
    }

    std::shared_ptr<hmm::OckHmmHMObject> attrTimesHmo = curGrp->keyAttrsTime.back();
    std::shared_ptr<hmm::OckHmmHMObject> attrTokenQuotientHmo = curGrp->keyAttrsQuotient.back();
    std::shared_ptr<hmm::OckHmmHMObject> attrTokenRemainderHmo = curGrp->keyAttrsRemainder.back();

    uint32_t startPos = curGrp->lastBlockRowCount + (curGrp->BlockCount() - 1U) * param->BlockRowCount();
    auto ret = CopyDataToHmo(attrTimesHmo, attrTimes.data(), startPos * sizeof(int32_t), addNum * sizeof(int32_t));
    OCK_CHECK_RETURN_ERRORCODE(ret);

    ret = CopyDataToHmo(
        attrTokenQuotientHmo, attrTokenQuotients.data(), startPos * sizeof(int32_t), addNum * sizeof(int32_t));
    OCK_CHECK_RETURN_ERRORCODE(ret);
    ret = CopyDataToHmo(attrTokenRemainderHmo,
        attrTokenRemainders.data(),
        startPos * sizeof(int8_t) * OPS_DATA_TYPE_TIMES,
        addNum * sizeof(int8_t) * OPS_DATA_TYPE_TIMES);
    OCK_CHECK_RETURN_ERRORCODE(ret);
    return ret;
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::AddBaseFeature(
    const uint64_t addNum, const DataTemp *features, std::shared_ptr<OckVsaAnnNpuBlockGroup> curGrp)
{
    // 分配预留norm数据的hmo，临时变量，使用后会free
    hmm::OckHmmErrorCode errorCode = hmm::HMM_SUCCESS;

    auto featCompHmo =
        hcps::handler::helper::MakeDeviceHmo(*handler, addNum * sizeof(DataTemp) * DimSizeTemp, errorCode);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);

    auto cpyRet = ockSyncUtils->Copy(reinterpret_cast<void *>(featCompHmo->Addr()),
        addNum * sizeof(DataTemp) * DimSizeTemp,
        features,
        addNum * sizeof(DataTemp) * DimSizeTemp,
        acladapter::OckMemoryCopyKind::HOST_TO_DEVICE);
    OCK_CHECK_RETURN_ERRORCODE(cpyRet);

    auto hmoNormGroup = hcps::nop::OckL2NormOpRun::BuildNormHmoBlock(featCompHmo, *handler, DimSizeTemp, addNum,
        errorCode);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    auto streamBase = hcps::handler::helper::MakeStream(*handler, errorCode, hcps::OckDevStreamType::AI_CORE);
    errorCode = hcps::nop::OckL2NormOpRun::ComputeNormSync(hmoNormGroup, *handler, streamBase);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);

    // 把norm结果拷贝到blockGroups里
    std::shared_ptr<hmm::OckHmmHMObject> normHmo = curGrp->norms.back();
    errorCode = handler->HmmMgrPtr()->CopyHMO(*normHmo,
        curGrp->lastBlockRowCount * sizeof(OckFloat16),
        *(hmoNormGroup->normResult),
        0,
        addNum * sizeof(OckFloat16));
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    handler->HmmMgrPtr()->Free(featCompHmo);

    errorCode = AddVectorsImpl(features, addNum, curGrp);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    return errorCode;
}

// addVectorsAicpu
template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::AddVectorsImpl(
    const DataTemp *features, const uint64_t addNum, std::shared_ptr<OckVsaAnnNpuBlockGroup> curGrp)
{
    hmm::OckHmmErrorCode errorCode = hmm::HMM_SUCCESS;

    auto dataHmo = hcps::handler::helper::MakeDeviceHmo(*handler, addNum * DimSizeTemp * sizeof(DataTemp), errorCode);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    if (curGrp == nullptr) {
        OCK_HCPS_LOG_ERROR("Input param cannot be nullptr");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    auto cpyRet = ockSyncUtils->Copy(reinterpret_cast<void *>(dataHmo->Addr()),
        addNum * DimSizeTemp * sizeof(DataTemp),
        features,
        addNum * DimSizeTemp * sizeof(DataTemp),
        acladapter::OckMemoryCopyKind::HOST_TO_DEVICE);
    OCK_CHECK_RETURN_ERRORCODE(cpyRet);

    auto dstHmo = curGrp->features.back();

    std::shared_ptr<hcps::nop::OckTransDataShapedOpHmoBlock> hmoBlock =
        std::make_shared<hcps::nop::OckTransDataShapedOpHmoBlock>();
    hmoBlock->srcHmo = dataHmo;
    hmoBlock->dstHmo = dstHmo;
    hmoBlock->dims = DimSizeTemp;
    hmoBlock->codeBlockSize = param->BlockRowCount();
    hmoBlock->addNum = static_cast<uint32_t>(addNum);
    hmoBlock->offsetInDstHmo = curGrp->lastBlockRowCount;

    auto streamBase = hcps::handler::helper::MakeStream(*handler, errorCode, hcps::OckDevStreamType::AI_CPU);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);

    hcps::nop::OckTransDataShapedOpRun::AddTransShapedOp(hmoBlock, *handler, streamBase);
    errorCode = streamBase->WaitExecComplete();
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    handler->HmmMgrPtr()->Free(dataHmo);
    return errorCode;
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::AddCustomAttr(
    const uint64_t addNum, const uint8_t *customAttr, std::shared_ptr<OckVsaAnnNpuBlockGroup> curGrp)
{
    if (customAttr == nullptr || curGrp == nullptr) {
        OCK_HCPS_LOG_ERROR("Input param cannot be nullptr");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    hmm::OckHmmErrorCode errorCode = hmm::HMM_SUCCESS;
    auto streamBase = hcps::handler::helper::MakeStream(*handler, errorCode, hcps::OckDevStreamType::AI_CPU);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    for (uint64_t i = 0; i < addNum;) {
        if (curGrp->lastCustomBlockRowCount == param->ExtKeyAttrBlockSize() || curGrp->extKeyAttrs.empty()) {
            auto customAttrHmo = hcps::handler::helper::MakeDeviceHmo(*handler,
                param->ExtKeyAttrBlockSize() * param->ExtKeyAttrByteSize(), errorCode);
            OCK_CHECK_RETURN_ERRORCODE(errorCode);
            curGrp->extKeyAttrs.emplace_back(customAttrHmo);
            curGrp->lastCustomBlockRowCount = 0;
        }
        uint64_t leftInBlock = param->ExtKeyAttrBlockSize() - curGrp->lastCustomBlockRowCount;
        uint64_t leftInAddData = addNum - i;
        uint64_t copyCount = std::min(leftInBlock, leftInAddData);
        auto hmoBlock = std::make_shared<hcps::nop::OckTransDataCustomAttrOpHmoBlock>();
        hmoBlock->customAttrLen = param->ExtKeyAttrByteSize();
        hmoBlock->customAttrBlockSize = param->ExtKeyAttrBlockSize();
        hmoBlock->copyCount = static_cast<uint32_t>(copyCount);
        hmoBlock->offsetInBlock = curGrp->lastCustomBlockRowCount;
        hmoBlock->dstHmo = curGrp->extKeyAttrs.back();
        hmoBlock->srcHmo =
            hcps::handler::helper::MakeDeviceHmo(*handler, copyCount * param->ExtKeyAttrByteSize(), errorCode);
        OCK_CHECK_RETURN_ERRORCODE(errorCode);
        const uint8_t *customAttrPos = customAttr + i * param->ExtKeyAttrByteSize();
        errorCode = ockSyncUtils->Copy(reinterpret_cast<void *>(hmoBlock->srcHmo->Addr()),
                                       hmoBlock->srcHmo->GetByteSize(),
                                       customAttrPos,
                                       copyCount * param->ExtKeyAttrByteSize(),
                                       acladapter::OckMemoryCopyKind::HOST_TO_DEVICE);
        OCK_CHECK_RETURN_ERRORCODE(errorCode);
        hcps::nop::OckTransDataCustomAttrOpRun::AddTransCustomAttrOp(hmoBlock, *handler, streamBase);
        errorCode = streamBase->WaitExecComplete();
        OCK_CHECK_RETURN_ERRORCODE(errorCode);
        i += copyCount;
        curGrp->lastCustomBlockRowCount += static_cast<uint32_t>(copyCount);
    }
    return errorCode;
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::PrepareOneFeatureBlock(void)
{
    OckVsaErrorCode errorCode = VSA_SUCCESS;
    if (blockGroups.empty() || IsGroupFull(blockGroups.back())) {
        uint32_t grpId = hisGroupCount % param->MaxGroupCount();
        auto newGroup = std::make_shared<OckVsaAnnNpuBlockGroup>();
        auto keyAttrsTimeHmo = hcps::handler::helper::MakeDeviceHmo(*handler,
            param->BlockRowCount() * param->GroupBlockCount() * sizeof(int32_t), errorCode);
        auto keyAttrsQuotientHmo = hcps::handler::helper::MakeDeviceHmo(*handler,
            param->BlockRowCount() * param->GroupBlockCount() * sizeof(int32_t), errorCode);
        auto keyAttrsRemainderHmo = hcps::handler::helper::MakeDeviceHmo(*handler,
            param->BlockRowCount() * param->GroupBlockCount() * sizeof(uint8_t) * OPS_DATA_TYPE_TIMES, errorCode);
        OCK_CHECK_RETURN_ERRORCODE(errorCode);
        newGroup->keyAttrsTime.emplace_back(keyAttrsTimeHmo);
        newGroup->keyAttrsQuotient.emplace_back(keyAttrsQuotientHmo);
        newGroup->keyAttrsRemainder.emplace_back(keyAttrsRemainderHmo);
        blockGroups.emplace_back(newGroup);
        grpPosMap[grpId] = static_cast<uint32_t>(groupIdDeque.size());
        groupIdDeque.push_back(grpId);
        auto newTokenIdMapGroup = std::deque<std::shared_ptr<hcps::hfo::OckTokenIdxMap>>();
        tokenIdxVectorMap.emplace_back(newTokenIdMapGroup);
        hisGroupCount++;
    }

    auto lastGroup = blockGroups.back();
    if (lastGroup->lastBlockRowCount < param->BlockRowCount() && lastGroup->lastBlockRowCount > 0) {
        return VSA_SUCCESS;
    }

    auto featureHmo = hcps::handler::helper::MakeDeviceHmo(*handler,
        param->BlockRowCount() * sizeof(DataTemp) * DimSizeTemp, errorCode);
    auto normHmo =
        hcps::handler::helper::MakeDeviceHmo(*handler, param->BlockRowCount() * sizeof(OckFloat16), errorCode);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);

    auto tokenIdMap = hcps::hfo::OckTokenIdxMap::Create(param->TokenNum(), *handler->HmmMgrPtr(), groupIdDeque.back());
    tokenIdxVectorMap.back().emplace_back(tokenIdMap);

    lastGroup->features.emplace_back(featureHmo);
    lastGroup->norms.emplace_back(normHmo);
    lastGroup->lastBlockRowCount = 0;
    return errorCode;
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::CopyDataToHmo(
    std::shared_ptr<hmm::OckHmmHMObject> dstHmo, const void *srcData, uint64_t startByteSize, uint64_t byteSize)
{
    if (dstHmo == nullptr || srcData == nullptr) {
        OCK_HCPS_LOG_ERROR("Input param cannot be nullptr");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    auto buffer = dstHmo->GetBuffer(hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR, startByteSize, byteSize);
    if (buffer == nullptr) {
        OCK_HCPS_LOG_ERROR("hmo getBuffer failed");
        return hmm::HMM_ERROR_HMO_BUFFER_NOT_ALLOCED;
    }
    auto cpyRet = ockSyncUtils->Copy(reinterpret_cast<void *>(buffer->Address()),
        byteSize,
        srcData,
        byteSize,
        acladapter::OckMemoryCopyKind::HOST_TO_DEVICE);
    OCK_CHECK_RETURN_ERRORCODE(cpyRet);
    buffer->FlushData();
    return cpyRet;
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
bool OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::IsGroupFull(
    std::shared_ptr<OckVsaAnnNpuBlockGroup> group)
{
    if (group == nullptr) {
        OCK_HCPS_LOG_ERROR("Input param cannot be nullptr");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    if (group->BlockCount() == param->GroupBlockCount() && group->lastBlockRowCount == param->BlockRowCount()) {
        return true;
    }
    return false;
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
uint64_t OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::GetByteSize(void) const
{
    uint64_t totalByteSize = 0;
    for (size_t grpId = 0; grpId < blockGroups.size(); ++grpId) {
        totalByteSize += blockGroups[grpId]->GetByteSize();
    }
    return totalByteSize;
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
void OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::AddIds(uint64_t addNum,
    OckVsaAnnAddFeatureParam<DataTemp, KeyTraitTemp> leftFeatureParam, std::shared_ptr<OckVsaAnnNpuBlockGroup> curGrp)
{
    uint32_t grpId = groupIdDeque.back();
    for (uint32_t i = 0; i < addNum; ++i) {
        uint64_t innerIdx = innerIdCvt.ToIdx(grpId, curGrp->rowCount + i);
        idMapMgr->SetIdxMap(innerIdx, *(leftFeatureParam.labels + i));
    }
}
} // namespace npu
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif
