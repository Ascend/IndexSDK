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


#ifndef OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_KERNEL_EXT_DEL_FEATURE_H
#define OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_KERNEL_EXT_DEL_FEATURE_H
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPKernelSystem.h"
#include "ock/acladapter/utils/OckSyncUtils.h"
namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTraitT, uint64_t BitPerDimT>
OckVsaErrorCode OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTraitT, BitPerDimT>::DeleteEmptyGroup(
    uint64_t index)
{
    unusedFeatures.push_back(usedFeatures[index]);
    usedFeatures.erase(usedFeatures.begin() + index);

    unusedNorms.push_back(usedNorms[index]);
    usedNorms.erase(usedNorms.begin() + index);

    unusedAttrTimeFilters.push_back(usedAttrTimeFilters[index]);
    usedAttrTimeFilters.erase(usedAttrTimeFilters.begin() + index);

    unusedAttrQuotientFilters.push_back(usedAttrQuotientFilters[index]);
    usedAttrQuotientFilters.erase(usedAttrQuotientFilters.begin() + index);

    unusedAttrRemainderFilters.push_back(usedAttrRemainderFilters[index]);
    usedAttrRemainderFilters.erase(usedAttrRemainderFilters.begin() + index);

    if (param->ExtKeyAttrByteSize() != 0) {
        unusedCustomerAttrs.push_back(usedCustomerAttrs[index]);
        usedCustomerAttrs.erase(usedCustomerAttrs.begin() + index);
    }
    unusedValidTags.push_back(usedValidTags[index]);
    usedValidTags.erase(usedValidTags.begin() + index);

    usedNeighborRelationGroups.erase(usedNeighborRelationGroups.begin() + index);
    tokenIdxVectorMap.erase(tokenIdxVectorMap.begin() + index);
    OCK_CHECK_RETURN_ERRORCODE(sampleFeatureMgr->DropEmptyGroup(index));

    groupIdDeque.erase(groupIdDeque.begin() + index);
    UpdateGroupPosMap();
    return VSA_SUCCESS;
}

template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait, uint64_t BitPerDimT>
void OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait, BitPerDimT>::UpdateGroupPosMap(void)
{
    for (uint32_t pos = 0; pos < groupIdDeque.size(); ++pos) {
        grpPosMap[groupIdDeque[pos]] = pos;
    }
}
/*
@brief 将数据打上删除标记，数据查询的时候，mask数据应该与usedValidTags合并
*/
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait, uint64_t BitPerDimT>
void OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait, BitPerDimT>::SetDroppedByLabel(uint64_t outterIdx)
{
    uint64_t innerId = idMapMgr->GetInnerIdx(outterIdx);
    if (innerId == hcps::hfo::INVALID_IDX_VALUE) {
        return;
    }
    auto innerInfo = innerIdConvertor.ToGroupOffset(innerId);
    usedValidTags[grpPosMap[innerInfo.grpId]]->Set(innerInfo.offset, false);
    idMapMgr->SetRemoved(outterIdx);
}

template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait, uint64_t BitPerDimT>
OckVsaErrorCode OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait, BitPerDimT>::SetDroppedByLabel(
    uint64_t count, const uint64_t *labels)
{
    for (uint64_t i = 0; i < count; ++i) {
        this->SetDroppedByLabel(labels[i]);
    }
    // 删除完成后校验是否存在全是无效数据的group，若存在则硬删除该group
    uint64_t emptyGroupCount = 0ULL;
    for (uint64_t i = 0; i < usedValidTags.size(); ++i) {
        if (usedValidTags[i]->Count() == 0ULL) {
            emptyGroupCount++;
        }
    }
    while (emptyGroupCount > 0) {
        for (uint64_t i = 0; i < usedValidTags.size(); ++i) {
            if (usedValidTags[i]->Count() == 0ULL) {
                OCK_CHECK_RETURN_ERRORCODE(this->DeleteEmptyGroup(i));
                emptyGroupCount--;
                break;
            }
        }
    }
    return VSA_SUCCESS;
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait, uint64_t BitPerDimT>
std::shared_ptr<hcps::OckHeteroOperatorBase> OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait, BitPerDimT>::CreateSetDroppedByTokenOp(
    uint32_t grpId, uint32_t token, const hcps::hfo::OckTokenIdxMap &idxMap)
{
    auto &rows = idxMap.RowIds(token);
    uint32_t dequeId = grpPosMap[grpId];
    return hcps::OckSimpleHeteroOperator<acladapter::OckTaskResourceType::HOST_CPU>::Create(
        [grpId, dequeId, &rows, this](hcps::OckHeteroStreamContext &) {
            for (auto rowId : rows) {
                auto innerId = innerIdConvertor.ToIdx(grpId, rowId);
                idMapMgr->SetRemovedByInnerId(innerId);
                usedValidTags[dequeId]->Set(rowId, false);
            }
            return VSA_SUCCESS;
        });
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait, uint64_t BitPerDimT>
void OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait, BitPerDimT>::AddSetDroppedByTokenOp(
    hcps::OckHeteroOperatorGroup &ops, uint32_t token)
{
    for (uint32_t i = 0; i < tokenIdxVectorMap.size(); ++i) {
        for (uint32_t j = 0; j < tokenIdxVectorMap[i].size(); ++j) {
            ops.push_back(this->CreateSetDroppedByTokenOp(groupIdDeque[i], token, *tokenIdxVectorMap[i][j]));
        }
    }
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait, uint64_t BitPerDimT>
OckVsaErrorCode OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait, BitPerDimT>::SetDroppedByToken(
    uint64_t count, const uint32_t *tokens)
{
    OckVsaErrorCode errorCode = VSA_SUCCESS;
    auto stream = hcps::handler::helper::MakeStream(*handler, errorCode, hcps::OckDevStreamType::AI_CPU);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    hcps::OckHeteroOperatorGroup ops;
    for (uint64_t i = 0; i < count; ++i) {
        if (tokens[i] >= param->TokenNum()) {
            continue;
        }
        this->AddSetDroppedByTokenOp(ops, tokens[i]);
    }
    for (auto &op : ops) {
        stream->AddOp(op);
        OCK_CHECK_RETURN_ERRORCODE(stream->WaitExecComplete());
    }
    // 删除完成后校验是否存在全是无效数据的group，若存在则硬删除该group
    uint64_t emptyGroupCount = 0ULL;
    for (uint64_t i = 0; i < usedValidTags.size(); ++i) {
        if (usedValidTags[i]->Count() == 0ULL) {
            emptyGroupCount++;
        }
    }
    while (emptyGroupCount > 0) {
        for (uint64_t i = 0; i < usedValidTags.size(); ++i) {
            if (usedValidTags[i]->Count() == 0ULL) {
                OCK_CHECK_RETURN_ERRORCODE(this->DeleteEmptyGroup(i));
                emptyGroupCount--;
                break;
            }
        }
    }
    return errorCode;
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait, uint64_t BitPerDimT>
OckVsaErrorCode OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait, BitPerDimT>::DeleteInvalidFeature(
    AddFeatureParamMeta<int8_t, attr::OckTimeSpaceAttrTrait> &paramStruct)
{
    OckVsaErrorCode errorCode = VSA_SUCCESS;
    // 获取要清理的group中的有效数据
    errorCode = this->GetValidData(paramStruct);
    // 删除最老的group
    unusedFeatures.push_back(usedFeatures.front());
    usedFeatures.pop_front();
    unusedValidTags.push_back(usedValidTags.front());
    usedValidTags.pop_front();
    unusedNorms.push_back(usedNorms.front());
    usedNorms.pop_front();
    unusedAttrTimeFilters.push_back(usedAttrTimeFilters.front());
    usedAttrTimeFilters.pop_front();
    unusedAttrQuotientFilters.push_back(usedAttrQuotientFilters.front());
    usedAttrQuotientFilters.pop_front();
    unusedAttrRemainderFilters.push_back(usedAttrRemainderFilters.front());
    usedAttrRemainderFilters.pop_front();
    if (param->ExtKeyAttrByteSize() != 0) {
        unusedCustomerAttrs.push_back(usedCustomerAttrs.front());
        usedCustomerAttrs.pop_front();
    }
    usedNeighborRelationGroups.pop_front();
    tokenIdxVectorMap.pop_front();
    idMapMgr->BatchRemoveByInner(innerIdConvertor.ToIdx(groupIdDeque.front(), 0ULL), param->GroupRowCount());
    OCK_CHECK_RETURN_ERRORCODE(sampleFeatureMgr->PopFrontGroup());
    groupIdDeque.pop_front();
    UpdateGroupPosMap();
    return errorCode;
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait, uint64_t BitPerDimT>
OckVsaErrorCode OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait, BitPerDimT>::RestoreCustomAttr(
    std::shared_ptr<acladapter::OckSyncUtils> syncUtils, uint32_t featureOffset, std::vector<uint8_t> &customAttr,
    uint32_t addFeatureNum)
{
    if (param->ExtKeyAttrByteSize() != 0) {
        uint32_t blockIndex = static_cast<uint32_t>(featureOffset / param->ExtKeyAttrBlockSize());
        uint32_t blockOffset = static_cast<uint32_t>(featureOffset % param->ExtKeyAttrBlockSize());
        OckVsaErrorCode errorCode = VSA_SUCCESS;
        for (uint32_t j = 0; j < param->ExtKeyAttrByteSize(); ++j) {
            auto srcCustomAddr = usedCustomerAttrs.front()->Addr() +
                blockIndex * param->ExtKeyAttrBlockSize() * param->ExtKeyAttrByteSize() +
                j * param->ExtKeyAttrBlockSize() + blockOffset;
            errorCode = syncUtils->Copy(customAttr.data() + addFeatureNum * param->ExtKeyAttrByteSize() + j,
                sizeof(uint8_t), reinterpret_cast<void *>(srcCustomAddr), sizeof(uint8_t),
                acladapter::OckMemoryCopyKind::DEVICE_TO_HOST);
            if (errorCode != hmm::HMM_SUCCESS) {
                OCK_VSA_HPP_LOG_ERROR("copy data from srcCustomAddr failed, the errorCode is " << errorCode);
                return errorCode;
            }
        }
    }
    return VSA_SUCCESS;
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait, uint64_t BitPerDimT>
OckVsaErrorCode OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait, BitPerDimT>::GetValidData(
    AddFeatureParamMeta<int8_t, attr::OckTimeSpaceAttrTrait> &paramStruct)
{
    std::shared_ptr<acladapter::OckSyncUtils> syncUtils =
        std::make_shared<acladapter::OckSyncUtils>(*(handler->Service()));
    uint32_t addFeatureNum = 0ULL;
    auto localAttrTime = usedAttrTimeFilters.front()->GetBuffer(hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY,
        0ULL, sizeof(int32_t) * param->GroupRowCount());
    if (localAttrTime == nullptr) {
        OCK_VSA_HPP_LOG_ERROR("usedAttrTimeFilters front HMO GetBuffer to host failed");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    auto localAttrQuot = usedAttrQuotientFilters.front()->GetBuffer(hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY,
        0ULL, sizeof(int32_t) * param->GroupRowCount());
    if (localAttrQuot == nullptr) {
        OCK_VSA_HPP_LOG_ERROR("usedAttrQuotientFilters front HMO GetBuffer to host failed");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    auto localAttrRemain =
        usedAttrRemainderFilters.front()->GetBuffer(hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0ULL,
        sizeof(int8_t) * param->GroupRowCount() * npu::OPS_DATA_TYPE_TIMES);
    if (localAttrRemain == nullptr) {
        OCK_VSA_HPP_LOG_ERROR("usedAttrRemainderFilters front HMO GetBuffer to host failed");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    // 遍历usedValidTags，拿到最老的group中未被删除的数据
    OckVsaErrorCode errorCode = VSA_SUCCESS;
    for (uint32_t i = 0; i < param->GroupRowCount(); ++i) {
        if (usedValidTags.front()->At(i)) {
            auto srcAddr = usedFeatures.front()->Addr() + i * DimSizeT * sizeof(DataT);
            errorCode = syncUtils->Copy(paramStruct.validateFeatures.data() + addFeatureNum * DimSizeT,
                DimSizeT * sizeof(DataT), reinterpret_cast<void *>(srcAddr), DimSizeT * sizeof(DataT),
                acladapter::OckMemoryCopyKind::HOST_TO_HOST);
            if (errorCode != hmm::HMM_SUCCESS) {
                OCK_VSA_HPP_LOG_ERROR("copy data from usedFeatures Addr failed, the errorCode is " << errorCode);
                return errorCode;
            }
            attr::OckTimeSpaceAttr toData;
            npu::OckVsaAnnKeyAttrInfo info;
            info.AttrCvt(toData, localAttrTime->Address() + i * sizeof(int32_t),
                localAttrQuot->Address() + i * sizeof(int32_t),
                localAttrRemain->Address() + i * sizeof(int8_t) * npu::OPS_DATA_TYPE_TIMES);
            memcpy_s(paramStruct.attributes.data() + addFeatureNum, sizeof(KeyTypeTupleT), &toData,
                sizeof(KeyTypeTupleT));
            // 还原customAttr
            errorCode = this->RestoreCustomAttr(syncUtils, i, paramStruct.customAttr, addFeatureNum);
            // 计算label
            uint32_t groupId = groupIdDeque.front();
            uint64_t innerIdx = innerIdConvertor.ToIdx(groupId, i);
            paramStruct.validLabels.push_back(idMapMgr->GetOutterIdx(innerIdx));
            addFeatureNum++;
        }
    }
    return errorCode;
}
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif