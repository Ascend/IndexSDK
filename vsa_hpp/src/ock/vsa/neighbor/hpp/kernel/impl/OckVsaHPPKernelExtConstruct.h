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


#ifndef OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_KERNEL_EXT_CONSTRUCT_H
#define OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_KERNEL_EXT_CONSTRUCT_H
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPKernelSystem.h"
#include "ock/vsa/neighbor/hpp/impl/OckVsaAnnHPPIndexSystem.h"
#include "ock/hcps/hop/OckSplitGroupOp.h"
#include "ock/utils/OckContainerBuilder.h"
namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
namespace impl {
inline void BuildBitSetDeque(std::deque<std::shared_ptr<hcps::algo::OckElasticBitSet>> &outData, uint64_t groupRowCount,
    uint32_t groupCount)
{
    for (uint64_t i = 0; i < groupCount; ++i) {
        outData.push_back(std::make_shared<hcps::algo::OckElasticBitSet>(groupRowCount));
    }
}
} // namespace impl
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait, uint64_t BitPerDimT>
OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait, BitPerDimT>::OckVsaHPPKernelExt(
    std::shared_ptr<hcps::handler::OckHeteroHandler> heteroHandler, const KeyTrait &defaultTrait,
    std::shared_ptr<OckVsaAnnCreateParam> parameter)
    : param(parameter),
      dftTrait(defaultTrait),
      innerIdConvertor(adapter::OckVsaHPPInnerIdConvertor::CalcBitCount(param->GroupRowCount())),
      tmpOutterLables(param->GroupRowCount()),
      grpPosMap(param->MaxGroupCount() + 1, 0),
      handler(heteroHandler)
{}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait, uint64_t BitPerDimT>
OckVsaErrorCode OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait, BitPerDimT>::Init(void)
{
    OCK_VSA_HPP_LOG_INFO("Init param " << *param);
    OckVsaErrorCode ret = hmm::HMM_SUCCESS;
    unusedFeatures = hcps::handler::helper::MakeHostHmoDeque(handler->HmmMgr(),
        DimSizeT * sizeof(DataT) * param->GroupRowCount(), param->MaxGroupCount() + 1, ret); // 100G
    unusedNorms = hcps::handler::helper::MakeHostHmoDeque(handler->HmmMgr(), NormTypeByteSizeT * param->GroupRowCount(),
        param->MaxGroupCount() + 1, ret); // 780M
    unusedAttrTimeFilters = hcps::handler::helper::MakeDeviceHmoDeque(handler->HmmMgr(),
        npu::OckVsaAnnRawBlockInfo::KeyAttrTimeBytes() * param->GroupRowCount(), param->MaxGroupCount() + 1, ret);
    unusedAttrQuotientFilters = hcps::handler::helper::MakeDeviceHmoDeque(handler->HmmMgr(),
        npu::OckVsaAnnRawBlockInfo::KeyAttrQuotientBytes() * param->GroupRowCount(), param->MaxGroupCount() + 1, ret);
    unusedAttrRemainderFilters = hcps::handler::helper::MakeDeviceHmoDeque(handler->HmmMgr(),
        npu::OckVsaAnnRawBlockInfo::KeyAttrRemainderBytes() * param->GroupRowCount(), param->MaxGroupCount() + 1, ret);

    if (param->ExtKeyAttrByteSize() != 0) {
        unusedCustomerAttrs = hcps::handler::helper::MakeDeviceHmoDeque(handler->HmmMgr(),
            param->ExtKeyAttrByteSize() * param->GroupRowCount(), param->MaxGroupCount() + 1, ret);
    }

    impl::BuildBitSetDeque(unusedValidTags, param->GroupRowCount(), param->MaxGroupCount() + 1);
    unusedDistHMOs = hcps::handler::helper::MakeHostHmoDeque(handler->HmmMgr(),
        DimSizeT * sizeof(DataT) * param->BlockRowCount(), param->MaxGroupCount() + 1, ret); // 1.6G

    idMapMgr = hcps::hfo::OckLightIdxMap::Create((param->MaxGroupCount() + 1) * param->GroupRowCount(),
        handler->HmmMgr()); // 约10G

    sampleFeatureMgr = std::make_shared<relation::OckVsaSampleFeatureMgr<DataT, DimSizeT>>(handler,
        param->MaxFeatureRowCount(), param->BlockRowCount()); // 约1.6G
    return ret;
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait, uint64_t BitPerDimT>
OckVsaErrorCode OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait, BitPerDimT>::GetFeatureByLabel(
    uint64_t label, DataT *feature)
{
    auto innerIdx = idMapMgr->GetInnerIdx(label);
    if (innerIdx == hcps::hfo::INVALID_IDX_VALUE) {
        return VSA_ERROR_INVALID_OUTTER_LABEL;
    }
    auto grpOffset = innerIdConvertor.ToGroupOffset(innerIdx);
    hcps::algo::OckShape<DataT, DimSizeT> shape(usedFeatures.at(grpPosMap[grpOffset.grpId])->Addr(),
        usedFeatures.at(grpPosMap[grpOffset.grpId])->GetByteSize(), param->GroupRowCount());
    shape.GetData(grpOffset.offset, feature);
    return hmm::HMM_SUCCESS;
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait, uint64_t BitPerDimT>
std::deque<npu::OckVsaAnnKeyAttrInfo> OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait, BitPerDimT>::GetAllFeatureAttrs(void) const
{
    std::deque<npu::OckVsaAnnKeyAttrInfo> ret;
    for (uint32_t i = 0; i < usedFeatures.size(); ++i) {
        npu::OckVsaAnnKeyAttrInfo blockInfo;
        blockInfo.keyAttrTime = usedAttrTimeFilters[i];
        blockInfo.keyAttrQuotient = usedAttrQuotientFilters[i];
        blockInfo.keyAttrRemainder = usedAttrRemainderFilters[i];
        ret.push_back(blockInfo);
    }
    return ret;
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait, uint64_t BitPerDimT>
OckVsaErrorCode OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait, BitPerDimT>::GetFeatureAttrByLabel(
    uint64_t label, typename KeyTrait::KeyTypeTuple *feature)
{
    auto innerIdx = idMapMgr->GetInnerIdx(label);
    if (innerIdx == hcps::hfo::INVALID_IDX_VALUE) {
        return VSA_ERROR_INVALID_OUTTER_LABEL;
    }
    auto grpOffset = innerIdConvertor.ToGroupOffset(innerIdx);

    auto timeBuffer = usedAttrTimeFilters.at(grpPosMap[grpOffset.grpId])
                          ->GetBuffer(hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY,
        grpOffset.offset * npu::OckVsaAnnRawBlockInfo::KeyAttrTimeBytes(),
        npu::OckVsaAnnRawBlockInfo::KeyAttrTimeBytes());
    if (timeBuffer == nullptr) {
        OCK_VSA_HPP_LOG_ERROR("usedAttrTimeFilters GetBuffer failed, group id is : " << grpOffset.grpId);
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    OCK_CHECK_RETURN_ERRORCODE(timeBuffer->ErrorCode());
    uintptr_t timeAddr = timeBuffer->Address();

    auto quotientBuffer = usedAttrQuotientFilters.at(grpPosMap[grpOffset.grpId])
                              ->GetBuffer(hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY,
        grpOffset.offset * npu::OckVsaAnnRawBlockInfo::KeyAttrQuotientBytes(),
        npu::OckVsaAnnRawBlockInfo::KeyAttrQuotientBytes());
    if (quotientBuffer == nullptr) {
        OCK_VSA_HPP_LOG_ERROR("usedAttrQuotientFilters GetBuffer failed, group id is : " << grpOffset.grpId);
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    OCK_CHECK_RETURN_ERRORCODE(quotientBuffer->ErrorCode());
    uintptr_t quotientAddr = quotientBuffer->Address();

    auto remainderBuffer = usedAttrRemainderFilters.at(grpPosMap[grpOffset.grpId])
                               ->GetBuffer(hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY,
        grpOffset.offset * npu::OckVsaAnnRawBlockInfo::KeyAttrRemainderBytes(),
        npu::OckVsaAnnRawBlockInfo::KeyAttrRemainderBytes() / 2);
    if (remainderBuffer == nullptr) {
        OCK_VSA_HPP_LOG_ERROR("usedAttrRemainderFilters GetBuffer failed, group id is : " << grpOffset.grpId);
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    OCK_CHECK_RETURN_ERRORCODE(remainderBuffer->ErrorCode());
    uintptr_t remainderAddr = remainderBuffer->Address();

    npu::OckVsaAnnRawBlockInfo::AttrCvt(std::get<0>(*feature), timeAddr, quotientAddr, remainderAddr);
    return hmm::HMM_SUCCESS;
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait, uint64_t BitPerDimT>
uintptr_t OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait, BitPerDimT>::GetCustomAttrByBlockId(
    uint32_t blockId, OckVsaErrorCode &errorCode) const
{
    if (errorCode != VSA_SUCCESS) {
        return 0UL;
    }
    if (param->ExtKeyAttrBlockSize() == 0UL || param->GroupRowCount() % param->ExtKeyAttrBlockSize() != 0) {
        OCK_VSA_HPP_LOG_ERROR("param error." << *param);
        errorCode = VSA_ERROR_INVALID_INPUT_PARAM;
        return 0UL;
    }

    uint64_t grpId = ((uint64_t)blockId * (uint64_t)param->ExtKeyAttrBlockSize()) / param->GroupRowCount();
    uint64_t grpOffset = (((uint64_t)blockId * (uint64_t)param->ExtKeyAttrBlockSize()) % param->GroupRowCount()) *
        param->ExtKeyAttrByteSize();

    if (usedCustomerAttrs[grpId]->Location() == hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR) {
        return usedCustomerAttrs[grpId]->Addr() + grpOffset;
    } else {
        auto deviceBuffer = usedCustomerAttrs[grpId]->GetBuffer(hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR, grpOffset,
            (uint64_t)param->ExtKeyAttrByteSize() * (uint64_t)param->ExtKeyAttrBlockSize());
        if (deviceBuffer == nullptr) {
            errorCode = VSA_ERROR_INVALID_INPUT_PARAM;
            return 0UL;
        }
        return deviceBuffer->Address();
    }
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait, uint64_t BitPerDimT>
uint32_t OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait, BitPerDimT>::GetCustomAttrBlockCount(
    void) const
{
    if (param->ExtKeyAttrBlockSize() == 0) {
        return 0;
    }
    return GroupCount() * param->GroupRowCount() / param->ExtKeyAttrBlockSize();
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait, uint64_t BitPerDimT>
uint32_t OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait, BitPerDimT>::ValidRowCount(
    uint32_t groupId) const
{
    if (groupId >= usedValidTags.size()) {
        return 0;
    }
    return static_cast<uint32_t>(usedValidTags.at(groupId)->Count());
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait, uint64_t BitPerDimT>
uint64_t OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait, BitPerDimT>::ValidRowCount(void) const
{
    uint64_t ret = 0ULL;
    for (uint32_t i = 0; i < this->GroupCount(); ++i) {
        ret += this->ValidRowCount(i);
    }
    return ret;
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait, uint64_t BitPerDimT>
uint32_t OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait, BitPerDimT>::GroupCount(void) const
{
    return static_cast<uint32_t>(usedFeatures.size());
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait, uint64_t BitPerDimT>
uint32_t OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait, BitPerDimT>::MaxGroupCount(void) const
{
    return param->MaxGroupCount();
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait, uint64_t BitPerDimT>
uint64_t OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait, BitPerDimT>::ValidRowCountInOldestGroup(
    void) const
{
    if (usedValidTags.empty()) {
        return 0;
    }
    return usedValidTags.front()->Count();
}
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif