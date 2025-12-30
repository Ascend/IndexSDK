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


#ifndef OCK_VSA_NPU_ANN_INDEX_DELETE_FEATURE_H
#define OCK_VSA_NPU_ANN_INDEX_DELETE_FEATURE_H
#include "ock/log/OckHcpsLogger.h"
#include "ock/utils/OckSafeUtils.h"
#include "ock/acladapter/data/OckTaskResourceType.h"
#include "ock/vsa/neighbor/npu/OckVsaAnnNpuBlockGroup.h"
#include "ock/hcps/hop/OckExternalQuicklySortOp.h"
#include "ock/vsa/neighbor/npu/impl/OckVsaAnnNpuIndexSystem.h"
#include "ock/hcps/nop/remove_data_custom_attr_op/OckRemoveDataCustomAttrMeta.h"
#include "ock/hcps/nop/remove_data_custom_attr_op/OckRemoveDataCustomAttrOpRun.h"
#include "ock/hcps/nop/remove_data_attr_op/OckRemoveDataAttrMeta.h"
#include "ock/hcps/nop/remove_data_attr_op/OckRemoveDataAttrOpRun.h"
#include "ock/hcps/nop/remove_data_shaped_op/OckRemoveDataShapedMeta.h"
#include "ock/hcps/nop/remove_data_shaped_op/OckRemoveDataShapedOpRun.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace npu {

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
std::shared_ptr<hcps::OckHeteroOperatorBase> OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::BuildDeleteAttrOp(
    const std::vector<uint64_t> &src, const std::vector<uint64_t> &dst,
    uint64_t deleteSize, uint32_t dataType, uint8_t copyNum)
{
    hmm::OckHmmErrorCode errorCode = hmm::HMM_SUCCESS;
    auto srcAttrHmo = hcps::handler::helper::MakeDeviceHmo(*handler, deleteSize * sizeof(uint64_t), errorCode);
    if (errorCode != hmm::HMM_SUCCESS) {
        OCK_HCPS_LOG_ERROR("Make src hmo failed");
        return std::shared_ptr<hcps::OckHeteroOperatorBase>();
    }

    auto dstAttrHmo = hcps::handler::helper::MakeDeviceHmo(*handler, deleteSize * sizeof(uint64_t), errorCode);
    if (errorCode != hmm::HMM_SUCCESS) {
        OCK_HCPS_LOG_ERROR("Make dst hmo failed");
        return std::shared_ptr<hcps::OckHeteroOperatorBase>();
    }

    errorCode = ockSyncUtils->Copy(reinterpret_cast<void *>(srcAttrHmo->Addr()), deleteSize * sizeof(uint64_t),
        src.data(), deleteSize * sizeof(uint64_t), acladapter::OckMemoryCopyKind::HOST_TO_DEVICE);
    if (errorCode != hmm::HMM_SUCCESS) {
        OCK_HCPS_LOG_ERROR("Copy src hmo failed");
        return std::shared_ptr<hcps::OckHeteroOperatorBase>();
    }

    errorCode = ockSyncUtils->Copy(reinterpret_cast<void *>(dstAttrHmo->Addr()), deleteSize * sizeof(uint64_t),
        dst.data(), deleteSize * sizeof(uint64_t), acladapter::OckMemoryCopyKind::HOST_TO_DEVICE);
    if (errorCode != hmm::HMM_SUCCESS) {
        OCK_HCPS_LOG_ERROR("Copy dst hmo failed");
        return std::shared_ptr<hcps::OckHeteroOperatorBase>();
    }

    auto hmoBlock = std::make_shared<hcps::nop::OckRemoveDataAttrOpHmoBlock>();

    hmoBlock->srcHmo = srcAttrHmo;
    hmoBlock->dstHmo = dstAttrHmo;
    hmoBlock->removeSize = static_cast<uint32_t>(deleteSize);
    hmoBlock->dataType = dataType;
    hmoBlock->copyNum = copyNum;

    return hcps::nop::OckRemoveDataAttrOpRun::CreateOp(hmoBlock, *handler);
}
template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::DeleteAttrByIds(
    const std::vector<uint64_t> &innerIds, const std::vector<uint64_t> &copyIds,
    std::shared_ptr<hcps::OckHeteroOperatorGroup> attrGroupOp)
{
    if (attrGroupOp == nullptr) {
        OCK_HCPS_LOG_ERROR("Input param cannot be nullptr");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    uint64_t deleteSize = innerIds.size();

    std::vector<uint64_t> attrTimeSrcAddr(deleteSize);
    std::vector<uint64_t> attrTimeDstAddr(deleteSize);
    std::vector<uint64_t> attrTokenQuotientSrcAddr(deleteSize);
    std::vector<uint64_t> attrTokenQuotientDstAddr(deleteSize);
    std::vector<uint64_t> attrTokenRemainderSrcAddr(deleteSize);
    std::vector<uint64_t> attrTokenRemainderDstAddr(deleteSize);
    for (uint64_t i = 0; i < deleteSize; i++) {
        auto OckSrcIndex = innerIdCvt.ToGroupOffset(copyIds[i]);
        std::shared_ptr<OckVsaAnnNpuBlockGroup> srcGroup{ blockGroups[grpPosMap[OckSrcIndex.grpId]] };
        auto OckDstIndex = innerIdCvt.ToGroupOffset(innerIds[i]);
        std::shared_ptr<OckVsaAnnNpuBlockGroup> dstGroup{ blockGroups[grpPosMap[OckDstIndex.grpId]] };

        attrTimeSrcAddr[i] =
            reinterpret_cast<uint64_t>(srcGroup->keyAttrsTime[0UL]->Addr() + OckSrcIndex.offset * sizeof(uint32_t));
        attrTimeDstAddr[i] =
            reinterpret_cast<uint64_t>(dstGroup->keyAttrsTime[0UL]->Addr() + OckDstIndex.offset * sizeof(uint32_t));
        attrTokenQuotientSrcAddr[i] = reinterpret_cast<uint64_t>(srcGroup->keyAttrsQuotient[0UL]->Addr() +
            OckSrcIndex.offset * sizeof(int32_t));
        attrTokenQuotientDstAddr[i] = reinterpret_cast<uint64_t>(dstGroup->keyAttrsQuotient[0UL]->Addr() +
            OckDstIndex.offset * sizeof(int32_t));
        attrTokenRemainderSrcAddr[i] = reinterpret_cast<uint64_t>(srcGroup->keyAttrsRemainder[0UL]->Addr() +
            OckSrcIndex.offset * sizeof(int8_t) * OPS_DATA_TYPE_TIMES);
        attrTokenRemainderDstAddr[i] = reinterpret_cast<uint64_t>(dstGroup->keyAttrsRemainder[0UL]->Addr() +
            OckDstIndex.offset * sizeof(int8_t) * OPS_DATA_TYPE_TIMES);
    }

    auto deleteAttrOp = BuildDeleteAttrOp(attrTimeSrcAddr, attrTimeDstAddr, deleteSize, hcps::nop::Type::INT32, 1UL);
    if (deleteAttrOp.get() == nullptr) {
        OCK_HCPS_LOG_ERROR("build delete time attr op failed.");
        return VSA_ERROR_BUILD_DELETE_ATTR_OP_FAILED;
    }
    attrGroupOp->push_back(deleteAttrOp);

    deleteAttrOp = BuildDeleteAttrOp(attrTokenQuotientSrcAddr, attrTokenQuotientDstAddr, deleteSize,
        hcps::nop::Type::INT32, 1UL);
    if (deleteAttrOp.get() == nullptr) {
        OCK_HCPS_LOG_ERROR("build delete token quotient attr op failed.");
        return VSA_ERROR_BUILD_DELETE_ATTR_OP_FAILED;
    }
    attrGroupOp->push_back(deleteAttrOp);

    deleteAttrOp = BuildDeleteAttrOp(attrTokenRemainderSrcAddr, attrTokenRemainderDstAddr, deleteSize,
        hcps::nop::Type::UINT8, OPS_DATA_TYPE_TIMES);
    if (deleteAttrOp.get() == nullptr) {
        OCK_HCPS_LOG_ERROR("build delete token reminder attr op failed.");
        return VSA_ERROR_BUILD_DELETE_ATTR_OP_FAILED;
    }
    attrGroupOp->push_back(deleteAttrOp);

    return VSA_SUCCESS;
}
template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::DeleteCustomAttrByIds(
    const std::vector<uint64_t> &innerIds, const std::vector<uint64_t> &copyIds,
    std::shared_ptr<hcps::OckHeteroOperatorGroup> customAttrGroupOp)
{
    if (customAttrGroupOp == nullptr) {
        OCK_HCPS_LOG_ERROR("Input param cannot be nullptr");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    uint64_t deleteSize = innerIds.size();
    uint32_t blockRowCount = param->ExtKeyAttrBlockSize();

    std::vector<uint64_t> customAttrSrcAddr(deleteSize);
    std::vector<uint64_t> customAttrDstAddr(deleteSize);
    for (uint64_t i = 0; i < deleteSize; i++) {
        auto OckSrcIndex = innerIdCvt.ToGroupOffset(copyIds[i]);
        std::shared_ptr<OckVsaAnnNpuBlockGroup> srcGroup{ blockGroups[grpPosMap[OckSrcIndex.grpId]] };
        uint32_t srcBlockIndex = OckSrcIndex.offset / blockRowCount;
        uint32_t srcFeatureIndex = OckSrcIndex.offset % blockRowCount;

        auto OckDstIndex = innerIdCvt.ToGroupOffset(innerIds[i]);
        std::shared_ptr<OckVsaAnnNpuBlockGroup> dstGroup{ blockGroups[grpPosMap[OckDstIndex.grpId]] };
        uint32_t dstBlockIndex = OckDstIndex.offset / blockRowCount;
        uint32_t dstFeatureIndex = OckDstIndex.offset % blockRowCount;

        customAttrSrcAddr[i] =
            reinterpret_cast<uint64_t>(srcGroup->extKeyAttrs[srcBlockIndex]->Addr() + srcFeatureIndex);
        customAttrDstAddr[i] =
            reinterpret_cast<uint64_t>(dstGroup->extKeyAttrs[dstBlockIndex]->Addr() + dstFeatureIndex);
    }

    OckVsaErrorCode errorCode = VSA_SUCCESS;
    auto srcAttrHmo = hcps::handler::helper::MakeDeviceHmo(*handler, deleteSize * sizeof(uint64_t), errorCode);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    auto dstAttrHmo = hcps::handler::helper::MakeDeviceHmo(*handler, deleteSize * sizeof(uint64_t), errorCode);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);

    errorCode = ockSyncUtils->Copy(reinterpret_cast<void *>(srcAttrHmo->Addr()), deleteSize * sizeof(uint64_t),
        customAttrSrcAddr.data(), deleteSize * sizeof(uint64_t), acladapter::OckMemoryCopyKind::HOST_TO_DEVICE);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    errorCode = ockSyncUtils->Copy(reinterpret_cast<void *>(dstAttrHmo->Addr()), deleteSize * sizeof(uint64_t),
        customAttrDstAddr.data(), deleteSize * sizeof(uint64_t), acladapter::OckMemoryCopyKind::HOST_TO_DEVICE);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);

    auto hmoBlock = std::make_shared<hcps::nop::OckRemoveDataCustomAttrOpHmoBlock>();
    hmoBlock->srcHmo = srcAttrHmo;
    hmoBlock->dstHmo = dstAttrHmo;
    hmoBlock->removeSize = static_cast<uint32_t>(deleteSize);
    hmoBlock->customAttrLen = param->ExtKeyAttrByteSize();
    hmoBlock->customAttrBlockSize = param->ExtKeyAttrBlockSize();
    auto op = hcps::nop::OckRemoveDataCustomAttrOpRun::CreateOp(hmoBlock, *handler);
    if (op.get() == nullptr) {
        OCK_HCPS_LOG_ERROR("build delete custom data op failed");
        return VSA_ERROR_BUILD_DELETE_OP_FAILED;
    }
    customAttrGroupOp->push_back(op);
    return errorCode;
}
template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::DeleteShapedByIds(
    const std::vector<uint64_t> &innerIds, const std::vector<uint64_t> &copyIds,
    std::shared_ptr<hcps::OckHeteroOperatorGroup> shapedGroupOp)
{
    if (shapedGroupOp == nullptr) {
        OCK_HCPS_LOG_ERROR("Input param cannot be nullptr");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    uint64_t deleteSize = innerIds.size();
    uint32_t blockRowCount = param->BlockRowCount();
    const uint32_t cubeAlign = hcps::nop::CUBE_ALIGN;
    const uint32_t cubeBandWidthBytes = hcps::nop::CUBE_ALIGN_INT8;
    uint32_t subDim = utils::SafeDiv(cubeBandWidthBytes, sizeof(DataTemp));
    uint32_t subNum = utils::SafeDiv(static_cast<uint32_t>(DimSizeTemp), subDim);

    std::vector<uint64_t> shapedSrcAddr(deleteSize);
    std::vector<uint64_t> shapedDstAddr(deleteSize);
    for (uint64_t i = 0; i < deleteSize; i++) {
        auto OckSrcIndex = innerIdCvt.ToGroupOffset(copyIds[i]);
        std::shared_ptr<OckVsaAnnNpuBlockGroup> srcGroupAddr{ blockGroups[grpPosMap[OckSrcIndex.grpId]] };
        uint32_t srcBlockIndex = OckSrcIndex.offset / blockRowCount;
        uint32_t srcFeatureIndex = OckSrcIndex.offset % blockRowCount;

        auto OckDstIndex = innerIdCvt.ToGroupOffset(innerIds[i]);
        std::shared_ptr<OckVsaAnnNpuBlockGroup> dstGroupAddr{ blockGroups[grpPosMap[OckDstIndex.grpId]] };
        uint32_t dstBlockIndex = OckDstIndex.offset / blockRowCount;
        uint32_t dstFeatureIndex = OckDstIndex.offset % blockRowCount;

        shapedSrcAddr[i] = reinterpret_cast<uint64_t>(srcGroupAddr->features[srcBlockIndex]->Addr() +
            ((srcFeatureIndex / cubeAlign) * cubeAlign * subNum + (srcFeatureIndex % cubeAlign)) * cubeBandWidthBytes);
        shapedDstAddr[i] = reinterpret_cast<uint64_t>(dstGroupAddr->features[dstBlockIndex]->Addr() +
            ((dstFeatureIndex / cubeAlign) * cubeAlign * subNum + (dstFeatureIndex % cubeAlign)) * cubeBandWidthBytes);
    }
    OckVsaErrorCode errorCode = VSA_SUCCESS;
    auto srcShapedHmo = hcps::handler::helper::MakeDeviceHmo(*handler, deleteSize * sizeof(uint64_t), errorCode);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    auto dstShapedHmo = hcps::handler::helper::MakeDeviceHmo(*handler, deleteSize * sizeof(uint64_t), errorCode);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);

    errorCode = ockSyncUtils->Copy(reinterpret_cast<void *>(srcShapedHmo->Addr()), deleteSize * sizeof(uint64_t),
        shapedSrcAddr.data(), deleteSize * sizeof(uint64_t), acladapter::OckMemoryCopyKind::HOST_TO_DEVICE);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    errorCode = ockSyncUtils->Copy(reinterpret_cast<void *>(dstShapedHmo->Addr()), deleteSize * sizeof(uint64_t),
        shapedDstAddr.data(), deleteSize * sizeof(uint64_t), acladapter::OckMemoryCopyKind::HOST_TO_DEVICE);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    auto hmoBlock = std::make_shared<hcps::nop::OckRemoveDataShapedOpHmoBlock>();
    hmoBlock->srcPosHmo = srcShapedHmo;
    hmoBlock->dstPosHmo = dstShapedHmo;
    hmoBlock->dims = DimSizeTemp;
    hmoBlock->removeCount = static_cast<uint32_t>(deleteSize);
    shapedGroupOp->push_back(hcps::nop::OckRemoveDataShapedOpRun::GenRemoveShapedOp(hmoBlock, *handler));
    return errorCode;
}
template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::DeleteNormByIds(
    const std::vector<uint64_t> &innerIds, const std::vector<uint64_t> &copyIds)
{
    if (innerIds.size() != copyIds.size()) {
        OCK_HCPS_LOG_ERROR("innerIds and copyIds have different sizes!");
        return VSA_ERROR_INPUT_PARAM_WRONG;
    }
    uint64_t deleteSize = innerIds.size();
    uint32_t blockRowCount = param->BlockRowCount();
    OckVsaErrorCode errorCode = VSA_SUCCESS;
    for (uint64_t i = 0; i < deleteSize; i++) {
        auto OckSrcIndex = innerIdCvt.ToGroupOffset(copyIds[i]);
        uint32_t srcBlockIndex = OckSrcIndex.offset / blockRowCount;
        uint32_t srcFeatureIndex = OckSrcIndex.offset % blockRowCount;
        uint64_t srcAddr = blockGroups[grpPosMap[OckSrcIndex.grpId]]->norms[srcBlockIndex]->Addr() +
            srcFeatureIndex * sizeof(OckFloat16);
        auto OckDstIndex = innerIdCvt.ToGroupOffset(innerIds[i]);
        uint32_t dstBlockIndex = OckDstIndex.offset / blockRowCount;
        uint32_t dstFeatureIndex = OckDstIndex.offset % blockRowCount;
        uint64_t dstAddr = blockGroups[grpPosMap[OckDstIndex.grpId]]->norms[dstBlockIndex]->Addr() +
            dstFeatureIndex * sizeof(OckFloat16);
        errorCode = ockSyncUtils->Copy(reinterpret_cast<void *>(dstAddr), sizeof(OckFloat16),
                                       reinterpret_cast<void *>(srcAddr), sizeof(OckFloat16),
                                       acladapter::OckMemoryCopyKind::DEVICE_TO_DEVICE);
        OCK_CHECK_RETURN_ERRORCODE(errorCode);
    }
    return errorCode;
}
template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
void OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::UpdateOckTokenIdxMap(
    const std::vector<uint64_t> &innerIds, const std::vector<uint64_t> &copyIds,
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> &quotientQueue,
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> &reminderQueue,
    uint64_t count)
{
    if (innerIds.size() != copyIds.size()) {
        OCK_HCPS_LOG_ERROR("deleteSize(" << innerIds.size() << ") is not equal to copySize(" << copyIds.size() << ").");
        return;
    }
    uint64_t totalNum = GetFeatureNum();
    uint64_t groupRowCount = param->GroupRowCount();
    auto CountBit = [](uint8_t x) { return x == 0UL ? 0UL : (31UL - __builtin_clz(x)); };
    for (uint64_t i = 0; i < innerIds.size(); i++) {
        auto lastIndexs = innerIdCvt.ToGroupOffset(copyIds[i]);
        auto curIndexs = innerIdCvt.ToGroupOffset(innerIds[i]);
        uintptr_t srcQuotientPtr =
            quotientQueue.at(grpPosMap[lastIndexs.grpId])->Addr() + lastIndexs.offset * sizeof(int32_t);
        uintptr_t srcRemainderPtr = reminderQueue.at(grpPosMap[lastIndexs.grpId])->Addr() +
            lastIndexs.offset * OPS_DATA_TYPE_TIMES * sizeof(uint8_t);
        uintptr_t dstQuotientPtr =
            quotientQueue.at(grpPosMap[curIndexs.grpId])->Addr() + curIndexs.offset * sizeof(int32_t);
        uintptr_t dstRemainderPtr = reminderQueue.at(grpPosMap[curIndexs.grpId])->Addr() +
            curIndexs.offset * OPS_DATA_TYPE_TIMES * sizeof(uint8_t);

        int32_t lastQuotient = *(reinterpret_cast<int32_t *>(srcQuotientPtr));
        uint8_t lastReminder = *(reinterpret_cast<uint8_t *>(srcRemainderPtr));
        uint32_t lastTokenId =
            static_cast<uint32_t>(lastQuotient / OPS_DATA_TYPE_TIMES * __CHAR_BIT__ + CountBit(lastReminder));

        int32_t curQuotient = *(reinterpret_cast<int32_t *>(dstQuotientPtr));
        uint8_t curReminder = *(reinterpret_cast<uint8_t *>(dstRemainderPtr));
        uint32_t curTokenId =
            static_cast<uint32_t>(curQuotient / OPS_DATA_TYPE_TIMES * __CHAR_BIT__ + CountBit(curReminder));

        uint32_t lastBlockOffset = lastIndexs.offset / param->BlockRowCount();
        uint32_t curBlockOffset = curIndexs.offset / param->BlockRowCount();
        tokenIdxVectorMap.at(grpPosMap[lastIndexs.grpId])[lastBlockOffset]->DeleteData(lastTokenId,
            static_cast<uint32_t>(copyIds[i] % groupRowCount));
        tokenIdxVectorMap.at(grpPosMap[curIndexs.grpId])[curBlockOffset]->DeleteData(curTokenId,
            static_cast<uint32_t>(innerIds[i] % groupRowCount));
        tokenIdxVectorMap.at(grpPosMap[curIndexs.grpId])[curBlockOffset]->InsertData(lastTokenId,
            static_cast<uint32_t>(innerIds[i] % groupRowCount));

        // 更新拷贝到host侧的商和余数，以免下一个循环计算错误
        *(reinterpret_cast<int32_t *>(dstQuotientPtr)) = *(reinterpret_cast<int32_t *>(srcQuotientPtr));
        *(reinterpret_cast<uint8_t *>(dstRemainderPtr)) = *(reinterpret_cast<uint8_t *>(srcRemainderPtr));
    }
    // 二次删除
    DeleteBackTokenIdxMap(quotientQueue, reminderQueue, count);
    OCK_HCPS_LOG_INFO("count is " << count << ", copyNum is " << innerIds.size() << ", totalNum is " << totalNum);
}
template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
void OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::DeleteBackTokenIdxMap(
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> &quotientQueue,
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> &reminderQueue,
    uint64_t count)
{
    if (quotientQueue.empty() || reminderQueue.empty()) {
        OCK_HCPS_LOG_ERROR("quotientQueue and reminderQueue should not be empty!");
        return;
    }
    if (quotientQueue.size() != reminderQueue.size()) {
        OCK_HCPS_LOG_ERROR("quotientQueue and reminderQueue have different sizes!");
        return;
    }
    uint64_t totalNum = GetFeatureNum();
    uint64_t groupRowCount = param->GroupRowCount();
    auto CountBit = [](uint8_t x) { return x == 0UL ? 0UL : (31UL - __builtin_clz(x)); };

    // 尾部删除
    uint64_t num = 0;
    for (uint64_t i = totalNum - count; i < totalNum; i++) {
        uint64_t innerId = 0;
        auto errorCode = TransOffsetInBaseToInnerIdx(i, innerId);
        if (errorCode != VSA_SUCCESS) {
            OCK_HCPS_LOG_ERROR("trans offset(" << i << ") to innerId failed.");
            return;
        }
        auto indexs = innerIdCvt.ToGroupOffset(innerId);
        uintptr_t quotientPtr = quotientQueue.at(grpPosMap[indexs.grpId])->Addr() + indexs.offset * sizeof(int32_t);
        uintptr_t remainderPtr =
            reminderQueue.at(grpPosMap[indexs.grpId])->Addr() + indexs.offset * OPS_DATA_TYPE_TIMES * sizeof(uint8_t);
        int32_t quotient = *(reinterpret_cast<int32_t *>(quotientPtr));
        uint8_t reminder = *(reinterpret_cast<uint8_t *>(remainderPtr));
        uint32_t tokenId = static_cast<uint32_t>(quotient / OPS_DATA_TYPE_TIMES * __CHAR_BIT__ + CountBit(reminder));
        uint32_t blockOffset = indexs.offset / param->BlockRowCount();
        if (tokenIdxVectorMap.at(grpPosMap[indexs.grpId])[blockOffset]->InToken(tokenId,
            static_cast<uint32_t>(i % groupRowCount))) {
            tokenIdxVectorMap.at(grpPosMap[indexs.grpId])[blockOffset]->DeleteData(tokenId,
                static_cast<uint32_t>(i % groupRowCount));
            num++;
        }
    }
    OCK_HCPS_LOG_INFO("num is " << num << ", actually is " << count);
}
template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::DeleteInvalidBlock(
    uint64_t oldFeatureNum, uint64_t removeCount)
{
    if (oldFeatureNum < removeCount) {
        OCK_HCPS_LOG_ERROR("oldFeatureNum(" << oldFeatureNum << ") must be greater than or equal to removeCount(" <<
                           removeCount << ")!");
        return VSA_ERROR_INPUT_PARAM_WRONG;
    }
    uint64_t oldGroupNum = utils::SafeDivUp(oldFeatureNum, static_cast<uint64_t>(param->GroupRowCount()));
    uint64_t newFeatureNum = oldFeatureNum - removeCount;
    uint64_t newGroupNum = utils::SafeDivUp(newFeatureNum, static_cast<uint64_t>(param->GroupRowCount()));
    for (uint64_t i = 0; i < oldGroupNum - newGroupNum; ++i) {
        blockGroups.pop_back();
        groupIdDeque.pop_back();
        tokenIdxVectorMap.pop_back();
        hisGroupCount--;
    }
    UpdateGroupPosMap();
    if (blockGroups.size() != 0) {
        auto curGroup = blockGroups.back();
        auto curTokenIdGroup = tokenIdxVectorMap.back();
        uint64_t newBlockNumInLastGroup =
            utils::SafeDivUp(newFeatureNum, param->BlockRowCount()) % param->GroupBlockCount();
        if (newBlockNumInLastGroup == 0) {
            newBlockNumInLastGroup = param->GroupBlockCount();
        }
        uint64_t curBlockNumInLastGroup = curGroup->features.size();
        for (uint64_t i = 0; i < curBlockNumInLastGroup - newBlockNumInLastGroup; ++i) {
            curGroup->features.pop_back();
            curGroup->norms.pop_back();
            curTokenIdGroup.pop_back();
        }
        uint32_t lastBlockRowCount =
            static_cast<uint32_t>((newFeatureNum % param->GroupRowCount()) % param->BlockRowCount());
        curGroup->lastBlockRowCount = lastBlockRowCount == 0 ? param->BlockRowCount() : lastBlockRowCount;
        if (param->ExtKeyAttrBlockSize() != 0 && param->ExtKeyAttrByteSize() != 0) {
            uint64_t groupCustomBlockCount = param->GroupRowCount() / param->ExtKeyAttrBlockSize();
            uint64_t newCustomBlockNumInLastGroup =
                utils::SafeDivUp(newFeatureNum, param->ExtKeyAttrBlockSize()) % groupCustomBlockCount;
            if (newCustomBlockNumInLastGroup == 0) {
                newCustomBlockNumInLastGroup = groupCustomBlockCount;
            }
            uint64_t curCustomBlockNumInLastGroup = curGroup->extKeyAttrs.size();
            for (uint64_t i = 0; i < curCustomBlockNumInLastGroup - newCustomBlockNumInLastGroup; ++i) {
                curGroup->extKeyAttrs.pop_back();
            }
            uint32_t lastCustomBlockRowCount =
                static_cast<uint32_t>((newFeatureNum % param->GroupRowCount()) % param->ExtKeyAttrBlockSize());
            curGroup->lastCustomBlockRowCount =
                lastCustomBlockRowCount == 0 ? param->ExtKeyAttrBlockSize() : lastCustomBlockRowCount;
        }
        curGroup->rowCount = static_cast<uint32_t>((curGroup->features.size() - 1) * param->BlockRowCount() +
            curGroup->lastBlockRowCount);
    }
    return vsa::VSA_SUCCESS;
}
template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::LoadTokenDataToHost(
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> &quotientQueue,
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> &reminderQueue)
{
    for (auto groupIter = blockGroups.begin(); groupIter != blockGroups.end(); ++groupIter) {
        OckVsaErrorCode quotientErrorCode = VSA_SUCCESS;
        OckVsaErrorCode reminderErrorCode = VSA_SUCCESS;

        auto tmpAttrsQuotient =
            hcps::handler::helper::CopyToHostHmo(*handler, *((*groupIter)->keyAttrsQuotient[0UL]), quotientErrorCode);
        OCK_CHECK_RETURN_ERRORCODE(quotientErrorCode);

        auto tmpAttrsRemainder =
            hcps::handler::helper::CopyToHostHmo(*handler, *((*groupIter)->keyAttrsRemainder[0UL]), reminderErrorCode);
        OCK_CHECK_RETURN_ERRORCODE(reminderErrorCode);

        quotientQueue.push_back(tmpAttrsQuotient);
        reminderQueue.push_back(tmpAttrsRemainder);
    }
    return VSA_SUCCESS;
}
template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
std::shared_ptr<hcps::OckHeteroOperatorGroupQueue> OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::CreateDeleteFeatureOp(
    const std::vector<uint64_t> &deleteIds, const std::vector<uint64_t> &copyIds)
{
    OckVsaErrorCode errorCode = VSA_SUCCESS;
    auto deleteOperatorGroupQueue = std::make_shared<hcps::OckHeteroOperatorGroupQueue>();
    if (deleteIds.size() != copyIds.size()) {
        OCK_HCPS_LOG_ERROR("deleteIds(" << deleteIds.size() << ") and copyIds(" << copyIds.size() <<
                           ") have different sizes!");
        return std::shared_ptr<hcps::OckHeteroOperatorGroupQueue>();
    }
    // Attr, CustomAttr, Shaped
    auto attrGroupOp = std::make_shared<hcps::OckHeteroOperatorGroup>();
    auto customAttrGroupOp = std::make_shared<hcps::OckHeteroOperatorGroup>();
    auto shapedGroupOp = std::make_shared<hcps::OckHeteroOperatorGroup>();

    // 时空属性
    errorCode = DeleteAttrByIds(deleteIds, copyIds, attrGroupOp);
    if (errorCode != VSA_SUCCESS) {
        OCK_HCPS_LOG_ERROR("Delete time space attribute failed, the errorCode is " << errorCode);
        return std::shared_ptr<hcps::OckHeteroOperatorGroupQueue>();
    }
    deleteOperatorGroupQueue->push(attrGroupOp);

    // 自定义属性
    if (param->ExtKeyAttrBlockSize() != 0 && param->ExtKeyAttrByteSize() != 0) {
        errorCode = DeleteCustomAttrByIds(deleteIds, copyIds, customAttrGroupOp);
        if (errorCode != VSA_SUCCESS) {
            OCK_HCPS_LOG_ERROR("Delete custom attribute failed, the errorCode is " << errorCode);
            return std::shared_ptr<hcps::OckHeteroOperatorGroupQueue>();
        }
        deleteOperatorGroupQueue->push(customAttrGroupOp);
    }

    // 拷贝底库分形数据
    errorCode = DeleteShapedByIds(deleteIds, copyIds, shapedGroupOp);
    if (errorCode != VSA_SUCCESS) {
        OCK_HCPS_LOG_ERROR("Delete shaped feature failed, the errorCode is " << errorCode);
        return std::shared_ptr<hcps::OckHeteroOperatorGroupQueue>();
    }
    deleteOperatorGroupQueue->push(shapedGroupOp);
    return deleteOperatorGroupQueue;
}
template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::DeleteRun(
    std::vector<uint64_t> &deleteChaosIds, std::vector<uint64_t> &copyIds, uint64_t totalNum, uint64_t count)
{
    if (deleteChaosIds.size() != copyIds.size()) {
        OCK_HCPS_LOG_ERROR("deleteChaosIds(" << deleteChaosIds.size() << ") and copyIds(" << copyIds.size() <<
                           ") have different sizes!");
        return VSA_ERROR_INPUT_PARAM_WRONG;
    }
    if (totalNum < count) {
        OCK_HCPS_LOG_ERROR("totalNum(" << totalNum << ") must be greater than or equal to count(" << count << ")!");
        return VSA_ERROR_INPUT_PARAM_WRONG;
    }
    OckVsaErrorCode errorCode = VSA_SUCCESS;
    auto stream = hcps::handler::helper::MakeStream(*handler, errorCode, hcps::OckDevStreamType::AI_DEFAULT);
    // 递减排序
    std::vector<uint64_t> deleteIds(deleteChaosIds.size(), 0);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    uint64_t splitThreshold = 10240UL;
    hcps::hop::ExternalQuicklySort(*stream, deleteChaosIds, deleteIds, std::greater<uint64_t>(), splitThreshold);

    // 加载token数据到host侧
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> keyAttrsQuotientQueue;
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> keyAttrsRemainderQueue;
    errorCode = LoadTokenDataToHost(keyAttrsQuotientQueue, keyAttrsRemainderQueue);
    if (errorCode != VSA_SUCCESS) {
        OCK_HCPS_LOG_ERROR("Load token data to host failed");
        return errorCode;
    }
    // 拷贝norm数据
    errorCode = DeleteNormByIds(deleteIds, copyIds);
    if (errorCode != VSA_SUCCESS) {
        OCK_HCPS_LOG_ERROR("Delete norm feature failed");
        return errorCode;
    }
    // 调用device删除算子
    std::shared_ptr<hcps::OckHeteroOperatorGroupQueue> deleteOperatorGroupQueue =
        CreateDeleteFeatureOp(deleteIds, copyIds);
    if (deleteOperatorGroupQueue.get() == nullptr) {
        return VSA_ERROR_BUILD_DELETE_OP_FAILED;
    }
    errorCode = stream->RunOps(*deleteOperatorGroupQueue, hcps::OckStreamExecPolicy::STOP_IF_ERROR);
    if (errorCode != VSA_SUCCESS) {
        OCK_HCPS_LOG_ERROR("Delete ts_attr, shaped and customAttr feature failed");
        return errorCode;
    }

    // 修改label表和index表的映射关系
    idMapMgr->BatchDeleteByInnerId(deleteIds, copyIds, static_cast<uint32_t>(deleteIds.size()));
    for (uint64_t i = totalNum - count; i < totalNum; i++) {
        uint64_t innerId = 0ULL;
        errorCode = TransOffsetInBaseToInnerIdx(i, innerId);
        OCK_CHECK_RETURN_ERRORCODE(errorCode);
        idMapMgr->DeleteByInnerId(innerId);
    }
    UpdateOckTokenIdxMap(deleteIds, copyIds, keyAttrsQuotientQueue, keyAttrsRemainderQueue, count);

    // 释放内存
    errorCode = DeleteInvalidBlock(totalNum, count);

    return errorCode;
}
template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::TransOffsetInBaseToInnerIdx(
    uint64_t offset, uint64_t &innerIdx)
{
    uint64_t totalNum = this->GetFeatureNum();
    if (offset >= totalNum) {
        OCK_HCPS_LOG_ERROR("the offset(" << offset << ") is out of range [0, " << totalNum << ").");
        return hcps::HCPS_ERROR_OFFSET_EXCEED_SCOPE;
    }
    uint32_t grpOffset = static_cast<uint32_t>(offset / param->GroupRowCount());
    uint32_t offsetInGrp = static_cast<uint32_t>(offset % param->GroupRowCount());
    innerIdx = innerIdCvt.ToIdx(groupIdDeque[grpOffset], offsetInGrp);
    return VSA_SUCCESS;
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::TransInnerIdxToOffsetInBase(
    uint64_t innerIdx, uint64_t &offset)
{
    adapter::OckVsaHPPIdx idx = innerIdCvt.ToGroupOffset(innerIdx);
    if (std::find(groupIdDeque.begin(), groupIdDeque.end(), idx.grpId) == groupIdDeque.end()) {
        return hcps::HCPS_ERROR_NPU_GROUP_NOT_EXIST;
    }
    uint64_t grpOffset = grpPosMap[idx.grpId];
    uint64_t offsetInBase = grpOffset * param->GroupRowCount() + idx.offset;
    uint64_t totalNum = this->GetFeatureNum();
    if (offsetInBase >= totalNum) {
        OCK_HCPS_LOG_ERROR("the offsetInBase(" << offsetInBase << ") is out of range [0, " << totalNum << ").");
        return hcps::HCPS_ERROR_OFFSET_EXCEED_SCOPE;
    }
    offset = offsetInBase;
    return VSA_SUCCESS;
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::DeleteFeatureByLabel(
    uint64_t count, const int64_t *labels)
{
    if (labels == nullptr || count > MAX_GET_NUMBER) {
        OCK_HCPS_LOG_ERROR("Input nullptr or delete count exceeds threshold(" << MAX_GET_NUMBER << ").");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    if (count == 0) {
        return VSA_SUCCESS;
    }
    OCK_HCPS_LOG_INFO("Delete feature by label start, label count: " << count);
    OckVsaErrorCode errorCode = VSA_SUCCESS;
    // deleteIdSets是实际在释放块的id，removeIdSets是用户请求删除的id
    std::unordered_set<uint64_t> removeIdSets;
    uint64_t totalNum = GetFeatureNum();

    for (uint64_t i = 0; i < count; i++) {
        uint64_t innerId = idMapMgr->GetInnerIdx(*(labels + i));
        if (innerId != hcps::hfo::INVALID_IDX_VALUE) {
            removeIdSets.insert(innerId);
        } else {
            OCK_HCPS_LOG_WARN("DeleteFeatureByLabel(" << *(labels + i) << ") does not exist");
        }
    }
    if (removeIdSets.empty()) {
        OCK_HCPS_LOG_WARN("the labels input is not exists in npu base.");
        return VSA_SUCCESS;
    }

    std::unordered_set<uint64_t> deleteIdSets(removeIdSets.begin(), removeIdSets.end());
    for (uint64_t i = totalNum - removeIdSets.size(); i < totalNum; ++i) {
        uint64_t innerId = 0;
        errorCode = TransOffsetInBaseToInnerIdx(i, innerId);
        OCK_CHECK_RETURN_ERRORCODE(errorCode);
        auto it = deleteIdSets.find(innerId);
        if (it != deleteIdSets.end()) {
            deleteIdSets.erase(it);
        }
    }

    uint64_t removeSize = removeIdSets.size();
    OCK_HCPS_LOG_INFO("Delete by label: removeSize(" << removeSize << "), totalNum(" << totalNum << ")");

    // 为空则删除数据全在尾部且连续
    if (deleteIdSets.empty()) {
        // 加载token数据到host侧
        std::deque<std::shared_ptr<hmm::OckHmmHMObject>> keyAttrsQuotientQueue;
        std::deque<std::shared_ptr<hmm::OckHmmHMObject>> keyAttrsRemainderQueue;
        errorCode = LoadTokenDataToHost(keyAttrsQuotientQueue, keyAttrsRemainderQueue);
        if (errorCode != VSA_SUCCESS) {
            OCK_HCPS_LOG_ERROR("Load token data to host failed");
            return errorCode;
        }
        // 修改label表和index表的映射关系
        for (uint64_t i = totalNum - removeSize; i < totalNum; i++) {
            uint64_t innerId = 0;
            errorCode = TransOffsetInBaseToInnerIdx(i, innerId);
            OCK_CHECK_RETURN_ERRORCODE(errorCode);
            idMapMgr->DeleteByInnerId(innerId);
        }
        OCK_HCPS_LOG_INFO("Update idMap success");
        DeleteBackTokenIdxMap(keyAttrsQuotientQueue, keyAttrsRemainderQueue, removeSize);
        OCK_HCPS_LOG_INFO("Update tokenMap success");
        // 释放内存
        errorCode = DeleteInvalidBlock(totalNum, removeSize);
        if (errorCode == VSA_SUCCESS) {
            OCK_HCPS_LOG_INFO("Free memory success");
            OCK_HCPS_LOG_INFO("Delete feature by label success");
        }
        return errorCode;
    }
    // 否则离散拷贝
    uint64_t copyNum = deleteIdSets.size();
    std::vector<uint64_t> copyIds;
    for (uint64_t i = 0; i < totalNum; i++) {
        uint64_t innerId = 0;
        errorCode = TransOffsetInBaseToInnerIdx(totalNum - i - 1UL, innerId);
        OCK_CHECK_RETURN_ERRORCODE(errorCode);
        if (removeIdSets.find(innerId) == removeIdSets.end()) {
            copyIds.push_back(innerId);
            if (copyIds.size() == copyNum) {
                break;
            }
        }
    }
    if (copyIds.size() != copyNum) {
        OCK_HCPS_LOG_ERROR("Size of copyIds(" << copyIds.size() << ") is not equal to copyNum(" << copyNum << ")");
    }
    std::vector<uint64_t> deleteChaosIds(deleteIdSets.begin(), deleteIdSets.end());

    errorCode = DeleteRun(deleteChaosIds, copyIds, totalNum, removeSize);
    if (errorCode == VSA_SUCCESS) {
        OCK_HCPS_LOG_INFO("Delete feature by label success");
    }
    return errorCode;
}
template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::DeleteFeatureByToken(
    uint64_t count, const uint32_t *tokens)
{
    if (tokens == nullptr || count > MAX_GET_NUMBER) {
        OCK_HCPS_LOG_ERROR("Input nullptr or delete count exceeds threshold[" << MAX_GET_NUMBER << "].");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    if (count == 0) {
        return VSA_SUCCESS;
    }
    OCK_HCPS_LOG_INFO("Delete feature by token start, token count: " << count);
    OckVsaErrorCode errorCode = VSA_SUCCESS;
    uint64_t totalNum = GetFeatureNum();
    uint64_t groupSize = blockGroups.size();
    std::unordered_set<uint64_t> removeInnerIds;
    bool tokenFlag;
    for (uint32_t i = 0; i < count; ++i) {
        tokenFlag = false;
        if (tokens[i] >= param->TokenNum()) {
            OCK_HCPS_LOG_ERROR("Token(" << tokens[i] << ") out of range!");
            continue;
        }
        for (uint64_t j = 0; j < groupSize; ++j) {
            for (uint64_t k = 0; k < tokenIdxVectorMap[j].size(); ++k) {
                hcps::hfo::OckHmmUint32Vector tmpVec = tokenIdxVectorMap[j][k]->RowIds(tokens[i]);
                uint64_t rowOffset = tokenIdxVectorMap[j][k]->GroupId() * param->GroupRowCount();
                if (!tmpVec.empty()) {
                    tokenFlag = true;
                }
                for (auto &id : tmpVec) {
                    uint64_t tmpId = (uint64_t)id + rowOffset;
                    removeInnerIds.insert(tmpId);
                }
            }
        }
        if (tokenFlag == false) {
            OCK_HCPS_LOG_WARN("Token(" << tokens[i] << ") does not exist on device");
        }
    }

    uint64_t removeSize = removeInnerIds.size();
    if (removeSize == 0) {
        OCK_HCPS_LOG_WARN("the token ids input is not exists in npu base.");
        return VSA_SUCCESS;
    }
    OCK_HCPS_LOG_INFO("Delete by token: removeSize(" << removeSize << "), totalNum(" << totalNum << ")");

    std::unordered_set<uint64_t> deleteIdsSet(removeInnerIds.begin(), removeInnerIds.end());
    for (uint64_t i = totalNum - removeSize; i < totalNum; ++i) {
        uint64_t innerId = 0;
        errorCode = TransOffsetInBaseToInnerIdx(i, innerId);
        OCK_CHECK_RETURN_ERRORCODE(errorCode);
        auto it = deleteIdsSet.find(innerId);
        if (it != deleteIdsSet.end()) {
            deleteIdsSet.erase(it);
        }
    }

    if (deleteIdsSet.empty()) {
        // 修改label表和index表的映射关系
        for (uint64_t i = totalNum - removeSize; i < totalNum; i++) {
            uint64_t innerId = 0;
            errorCode = TransOffsetInBaseToInnerIdx(i, innerId);
            OCK_CHECK_RETURN_ERRORCODE(errorCode);
            idMapMgr->DeleteByInnerId(innerId);
        }
        OCK_HCPS_LOG_INFO("Update idMap success");
        for (uint64_t i = 0; i < groupSize; ++i) {
            for (uint64_t j = 0; j < tokenIdxVectorMap[i].size(); ++j) {
                for (uint64_t k = 0; k < count; ++k) {
                    if (tokens[k] >= param->TokenNum()) {
                        OCK_HCPS_LOG_ERROR("Token(" << tokens[k] << ") out of range!");
                        continue;
                    }
                    tokenIdxVectorMap[i][j]->DeleteToken(tokens[k]);
                }
            }
        }
        OCK_HCPS_LOG_INFO("Update tokenMap success");
        // 释放内存
        errorCode = DeleteInvalidBlock(totalNum, removeSize);
        if (errorCode == hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_INFO("Free memory success");
            OCK_HCPS_LOG_INFO("Delete feature by token success");
        }
        return errorCode;
    }

    // 否侧离散拷贝
    std::vector<uint64_t> copyIds;
    uint64_t copyNum = deleteIdsSet.size();
    for (uint64_t i = 0; i < totalNum; i++) {
        uint64_t innerId = 0;
        errorCode = TransOffsetInBaseToInnerIdx(totalNum - i - 1ULL, innerId);
        OCK_CHECK_RETURN_ERRORCODE(errorCode);
        if (removeInnerIds.find(innerId) == removeInnerIds.end()) {
            copyIds.push_back(innerId);
            if (copyIds.size() == copyNum) {
                break;
            }
        }
    }
    if (copyIds.size() != copyNum) {
        OCK_HCPS_LOG_ERROR("Size of copyIds(" << copyIds.size() << ") is not equal to copyNum(" << copyNum << ")");
    }
    std::vector<uint64_t> deleteChaosIds(deleteIdsSet.begin(), deleteIdsSet.end());
    errorCode = DeleteRun(deleteChaosIds, copyIds, totalNum, removeSize);
    if (errorCode == hmm::HMM_SUCCESS) {
        OCK_HCPS_LOG_INFO("Delete feature by token success");
    }
    return errorCode;
}
template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
std::shared_ptr<OckVsaAnnRawBlockInfo> OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::PopFrontBlockGroup(
    std::shared_ptr<OckVsaAnnNpuBlockGroup>& outBlockGroup,
    std::vector<uint64_t> &outterLabels, std::deque<std::shared_ptr<hcps::hfo::OckTokenIdxMap>> &tokenToRowIdsMap,
    OckVsaErrorCode &errorCode)
{
    outBlockGroup = blockGroups.front();
    blockGroups.pop_front();

    uint32_t grpId = groupIdDeque[0];
    groupIdDeque.pop_front();

    tokenToRowIdsMap = tokenIdxVectorMap.front();
    tokenIdxVectorMap.pop_front();

    outterLabels = std::vector<uint64_t>(param->GroupRowCount());
    idMapMgr->GetOutterIdxs(innerIdCvt.ToIdx(grpId, 0), param->GroupRowCount(), outterLabels.data());

    auto stream = hcps::handler::helper::MakeStream(*handler, errorCode, hcps::OckDevStreamType::AI_DEFAULT);
    if (errorCode != VSA_SUCCESS) {
        OCK_HCPS_LOG_ERROR("MakeStream failed! errorCode=" << errorCode);
        return std::shared_ptr<OckVsaAnnRawBlockInfo>();
    }

    auto ops = idMapMgr->CreateSetRemovedOps(param->GroupRowCount(), innerIdCvt.ToIdx(grpId, 0));
    stream->RunOps(*ops, hcps::OckStreamExecPolicy::STOP_IF_ERROR);

    this->UpdateGroupPosMap();

    auto rawBlockInfo = npu::LoadGroupBlocksIntoHost<DataTemp, DimSizeTemp>(*handler, *outBlockGroup, errorCode);
    if (errorCode != VSA_SUCCESS) {
        OCK_HCPS_LOG_ERROR("LoadGroupBlocksIntoHost failed! errorCode=" << errorCode);
        return rawBlockInfo;
    }

    return rawBlockInfo;
}
template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
void OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::UpdateGroupPosMap(void)
{
    for (uint32_t pos = 0; pos < groupIdDeque.size(); ++pos) {
        grpPosMap[groupIdDeque[pos]] = pos;
    }
}
} // namespace npu
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif