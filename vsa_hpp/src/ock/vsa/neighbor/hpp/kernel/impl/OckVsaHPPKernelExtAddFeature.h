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


#ifndef OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_KERNEL_EXT_ADD_FEATURE_H
#define OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_KERNEL_EXT_ADD_FEATURE_H
#include "ock/hcps/hop/OckSplitGroupOp.h"
#include "ock/hcps/algo/OckShape.h"
#include "ock/hcps/algo/OckCustomerAttrShape.h"
#include "ock/hcps/hop/OckExternalQuicklySortOp.h"
#include "ock/vsa/neighbor/OckVsaAnnDetailPrinter.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPKernelSystem.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPHmoFeatureDataRef.h"
namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
namespace impl {
template <typename DataT, uint64_t DimSizeT>
void ParallelReArrangeShapeFeature(hcps::OckHeteroStreamBase &stream, OckVsaHPPHmoFeatureDataRef relFeatureRef,
    const OckVsaAnnCreateParam &param)
{
    DataT *pSrcData = reinterpret_cast<DataT *>(relFeatureRef.srcData.Addr());
    auto dstData = relFeatureRef.PickUnused();
    if (dstData == nullptr || pSrcData == nullptr) {
        return;
    }
    auto ops =
        hcps::hop::MakeOckSplitGroupOps<uint32_t, acladapter::OckTaskResourceType::HOST_CPU>(0UL, param.GroupRowCount(),
        param.BlockRowCount(), [relFeatureRef, dstData, &param, pSrcData](uint32_t startPos, uint32_t endPos) {
            auto dstBuffer = dstData->GetBuffer(hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY,
                sizeof(DataT) * DimSizeT * startPos, sizeof(DataT) * DimSizeT * param.BlockRowCount());
            if (dstBuffer == nullptr) {
                return hmm::HMM_ERROR_HMO_OBJECT_INVALID;
            }
            hcps::algo::OckShape<DataT, DimSizeT> shape(dstBuffer->Address(), dstBuffer->Size());
            shape.AddData(pSrcData + startPos * DimSizeT, param.BlockRowCount());
            return dstBuffer->FlushData();
        });
    if (ops == nullptr) {
        return;
    }
    stream.AddOps(*ops);
    relFeatureRef.PopUnusedToUsed();
}
inline void ParallelCopyFeature(hcps::handler::OckHeteroHandler &handler, hcps::OckHeteroStreamBase &stream,
    OckVsaHPPHmoFeatureDataRef relFeatureRef)
{
    auto dstData = relFeatureRef.PickUnused();
    if (dstData == nullptr) {
        return;
    }
    auto op = hcps::OckSimpleHeteroOperator<acladapter::OckTaskResourceType::HOST_CPU>::Create(
        [dstData, relFeatureRef, &handler](hcps::OckHeteroStreamContext &) {
            return handler.HmmMgr().CopyHMO(*dstData, 0ULL, relFeatureRef.srcData, 0ULL,
                relFeatureRef.srcData.GetByteSize());
        });
    stream.AddOp(op);
    relFeatureRef.PopUnusedToUsed();
}

inline void ParallelCopyCustomerFeature(hcps::handler::OckHeteroHandler &handler, hcps::OckHeteroStreamBase &stream,
    OckVsaHPPHmoFeatureDataRef relFeatureRef)
{
    auto dstData = relFeatureRef.PickUnused();
    if (dstData == nullptr) {
        return;
    }
    auto op = hcps::OckSimpleHeteroOperator<acladapter::OckTaskResourceType::HOST_CPU>::Create(
        [relFeatureRef, dstData, &handler](hcps::OckHeteroStreamContext &) {
            return handler.HmmMgr().CopyHMO(*dstData, 0ULL, relFeatureRef.srcData, 0ULL,
                relFeatureRef.srcData.GetByteSize());
        });
    stream.AddOp(op);
    relFeatureRef.PopUnusedToUsed();
}
inline void ParallelReArrangeOutterLabel(hcps::OckHeteroStreamBase &stream, const std::vector<uint64_t> &rawLables,
    std::vector<uint64_t> &newLables)
{
    stream.AddOp(hcps::OckSimpleHeteroOperator<acladapter::OckTaskResourceType::HOST_CPU>::Create(
        [&rawLables, &newLables](hcps::OckHeteroStreamContext &) {
            return memcpy_s(newLables.data(), newLables.size() * sizeof(uint64_t), rawLables.data(),
                rawLables.size() * sizeof(uint64_t));
        }));
}
inline std::shared_ptr<hcps::OckHeteroOperatorGroup> CreateIdxMapAddOp(uint32_t grpId,
    hcps::hfo::OckLightIdxMap &idMapMgr, const std::vector<uint64_t> &outterLabels,
    adapter::OckVsaHPPInnerIdConvertor &idxCvt)
{
    return idMapMgr.CreateAddDatasOps(outterLabels.size(), outterLabels.data(), idxCvt.ToIdx(grpId, 0UL));
}
inline void TokenRowsMapAdd(hcps::handler::OckHeteroHandler &handler, const hcps::hfo::OckTokenIdxMap &srcDatas,
    hcps::hfo::OckTokenIdxMap &dstDatas)
{
    for (uint32_t token = 0; token < srcDatas.TokenNum(); ++token) {
        const hmm::helper::OckHmmUint32Vector &srcRows = srcDatas.RowIds(token);
        hmm::helper::OckHmmUint32Vector &dstRows = dstDatas.RowIds(token);
        if (srcRows.size() == 0 || dstRows.size() == 0) {
            continue;
        }
        auto ret = memcpy_s(dstRows.data(), sizeof(uint32_t) * dstRows.size(), srcRows.data(),
            srcRows.size() * sizeof(uint32_t));
        if (ret != 0) {
            OCK_VSA_HPP_LOG_ERROR("memcpy_s from count(" << srcRows.size() * sizeof(uint32_t) << ") to count(" <<
                sizeof(uint32_t) * dstRows.size() << ") failed, errorCode is " << ret);
            return;
        }
    }
}
inline std::shared_ptr<hcps::OckHeteroOperatorBase> CreateTokenRowsMapAddOp(hcps::handler::OckHeteroHandler &handler,
    const hcps::hfo::OckTokenIdxMap &srcDatas, hcps::hfo::OckTokenIdxMap &dstDatas)
{
    return hcps::OckSimpleHeteroOperator<acladapter::OckTaskResourceType::HOST_CPU>::Create(
        [&handler, &srcDatas, &dstDatas](hcps::OckHeteroStreamContext &) {
            TokenRowsMapAdd(handler, srcDatas, dstDatas);
            return hmm::HMM_SUCCESS;
        });
}
} // namespace impl
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait, uint64_t BitPerDimT>
OckVsaErrorCode OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait, BitPerDimT>::AddFeature(
    npu::OckVsaAnnRawBlockInfo &blockInfo, const std::vector<uint64_t> &outterLabels,
    std::deque<std::shared_ptr<hcps::hfo::OckTokenIdxMap>> tokenToRowIdsMap,
    std::shared_ptr<relation::OckVsaNeighborRelationHmoGroup> neighborRelationGroup)
{
    if (outterLabels.size() != param->GroupRowCount()) {
        OCK_VSA_HPP_LOG_ERROR("Invalid outterLabels.size(" << outterLabels.size() << ") != param->GroupRowCount(" <<
            param->GroupRowCount() << ")");
    }
    uint32_t curGroupId = hisGroupCount % (param->MaxGroupCount() + 1);
    OckVsaErrorCode errorCode = hmm::HMM_SUCCESS;

    auto stream = hcps::handler::helper::MakeStream(*handler, errorCode);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    usedValidTags.push_back(unusedValidTags.front());
    unusedValidTags.pop_front();

    stream->AddOp(hcps::OckSimpleHeteroOperator<acladapter::OckTaskResourceType::HOST_CPU>::Create(
        [this](hcps::OckHeteroStreamContext &) {
            usedValidTags.back()->SetAll();
            return hmm::HMM_SUCCESS;
        }));

    impl::ParallelReArrangeShapeFeature<DataT, DimSizeT>(*stream,
        impl::OckVsaHPPHmoFeatureDataRef(*blockInfo.feature, unusedFeatures, usedFeatures), *param);
    if (param->ExtKeyAttrByteSize() != 0) {
        impl::ParallelCopyCustomerFeature(*handler, *stream,
            impl::OckVsaHPPHmoFeatureDataRef(*blockInfo.extKeyAttr, unusedCustomerAttrs, usedCustomerAttrs));
    }

    impl::ParallelCopyFeature(*handler, *stream,
        impl::OckVsaHPPHmoFeatureDataRef(*blockInfo.norm, unusedNorms, usedNorms));
    impl::ParallelCopyFeature(*handler, *stream,
        impl::OckVsaHPPHmoFeatureDataRef(*blockInfo.keyAttrTime, unusedAttrTimeFilters, usedAttrTimeFilters));
    impl::ParallelCopyFeature(*handler, *stream,
        impl::OckVsaHPPHmoFeatureDataRef(*blockInfo.keyAttrQuotient, unusedAttrQuotientFilters,
        usedAttrQuotientFilters));
    impl::ParallelCopyFeature(*handler, *stream,
        impl::OckVsaHPPHmoFeatureDataRef(*blockInfo.keyAttrRemainder, unusedAttrRemainderFilters,
        usedAttrRemainderFilters));
    impl::ParallelReArrangeOutterLabel(*stream, outterLabels, tmpOutterLables);
    uint64_t needIncHostByteSizes = blockInfo.GetByteSize();
    errorCode = hcps::handler::helper::UseIncBindMemory(*handler, needIncHostByteSizes, "MakeShapedSample");
    if (errorCode != hmm::HMM_SUCCESS) {
        return errorCode;
    }
    auto usedInfo = handler->HmmMgrPtr()->GetUsedInfo(64ULL * 1024ULL * 1024ULL);
    OCK_VSA_HPP_LOG_INFO("MakeShapedSample need: " << needIncHostByteSizes << "usedInfo: " << *usedInfo);
    auto shapedFeature = relation::MakeShapedSampleFeature<DataT, DimSizeT>(*handler, *neighborRelationGroup,
        blockInfo.feature, errorCode);
    auto shapedNorm = relation::MakeShapedSampleNorm(*handler, *neighborRelationGroup, blockInfo.norm, errorCode);

    if (errorCode == hmm::HMM_SUCCESS) {
        sampleFeatureMgr->AddSampleFeature(shapedFeature, shapedNorm, neighborRelationGroup->validateRowCount);
    }
    auto waitErrorCode = stream->WaitExecComplete();
    OCK_VSA_HPP_LOG_INFO("Rearrange data complete. curGroupId:" << curGroupId << " memInfo:" <<
        *(handler->HmmMgr().GetUsedInfo(64ULL * 1024ULL * 1024ULL)));
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    OCK_CHECK_RETURN_ERRORCODE(waitErrorCode);

    // 一个 group 的 tokenToRowIdsMap 所需的最大空间 = tokenNum * 二次分配最小单位 * block数 * 2
    needIncHostByteSizes = param->TokenNum() * 256ULL * tokenToRowIdsMap.size() * 2ULL;
    errorCode = hcps::handler::helper::UseIncBindMemory(*handler, needIncHostByteSizes, "tokenToRowIdsMap");
    if (errorCode != hmm::HMM_SUCCESS) {
        return errorCode;
    }
    usedInfo = handler->HmmMgrPtr()->GetUsedInfo(64ULL * 1024ULL * 1024ULL);
    OCK_VSA_HPP_LOG_INFO("tokenToRowIdsMap need: " << needIncHostByteSizes << "usedInfo: " << *usedInfo);

    auto flushIdxOps = impl::CreateIdxMapAddOp(curGroupId, *idMapMgr, tmpOutterLables, innerIdConvertor);
    stream->AddOps(*flushIdxOps);
    auto newTokenToRowIdsMap = std::deque<std::shared_ptr<hcps::hfo::OckTokenIdxMap>>();
    for (uint32_t i = 0; i < tokenToRowIdsMap.size(); ++i) {
        auto newTokenIdsMap = tokenToRowIdsMap[i]->CopySpec(handler->HmmMgr());
        newTokenToRowIdsMap.push_back(newTokenIdsMap);
    }
    OCK_VSA_HPP_LOG_INFO("Flush idxMap CopySpec. curGroupId:" << curGroupId << " memInfo:" <<
        *(handler->HmmMgr().GetUsedInfo(64ULL * 1024ULL * 1024ULL)));
    for (uint32_t i = 0; i < tokenToRowIdsMap.size(); ++i) {
        std::shared_ptr<hcps::hfo::OckTokenIdxMap> tokenIdsMap = tokenToRowIdsMap[i];
        std::shared_ptr<hcps::hfo::OckTokenIdxMap> newTokenIdsMap = newTokenToRowIdsMap[i];
        stream->AddOp(impl::CreateTokenRowsMapAddOp(*handler, *tokenIdsMap, *newTokenIdsMap));
    }
    tokenIdxVectorMap.emplace_back(newTokenToRowIdsMap);
    newTokenToRowIdsMap.clear();
    OCK_VSA_HPP_LOG_INFO("Flush idxMap complete. curGroupId:" << curGroupId << " memInfo:" <<
        *(handler->HmmMgr().GetUsedInfo(64ULL * 1024ULL * 1024ULL)));

    usedNeighborRelationGroups.push_back(neighborRelationGroup);

    grpPosMap[curGroupId] = static_cast<uint32_t>(groupIdDeque.size());
    groupIdDeque.push_back(curGroupId);
    hisGroupCount++;
    errorCode = stream->WaitExecComplete();

    hcps::algo::OckShape<DataT, DimSizeT> shape(usedFeatures.back()->Addr(), usedFeatures.back()->GetByteSize(),
        param->GroupRowCount());

    return errorCode;
}
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif