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


#ifndef OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_KERNEL_EXT_QUERY_FEATURE_H
#define OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_KERNEL_EXT_QUERY_FEATURE_H
#include <chrono>
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPKernelSystem.h"
#include "ock/vsa/neighbor/base/OckVsaHPPInnerIdConvertor.h"
#include "ock/vsa/neighbor/npu/OckVsaAnnCollectResultNpuProcessor.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPMaskQuery.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPSelectRate.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPSliceIdMgr.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPTopNSliceSelector.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPSliceAssemble.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPSearchStaticInfo.h"
#include "ock/vsa/neighbor/base/OckVsaHPPInnerIdConvertor.h"
#include "ock/vsa/neighbor/base/OckVsaAnnCollectResultProcessor.h"
namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
namespace impl {
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait, uint64_t BitPerDimT>
class OckVsaHPPFeatureGenerator {
public:
    OckVsaHPPFeatureGenerator(hmm::OckHmmHeteroMemoryMgrBase &hmmManager, uint32_t rowNum)
        : rowCount(rowNum), hmmMgr(hmmManager)
    {}
    std::shared_ptr<adapter::OckVsaAnnFeature> Alloc(void)
    {
        OckVsaErrorCode errorCode = hmm::HMM_SUCCESS;
        uint64_t featureByteSize = rowCount * DimSizeT * sizeof(DataT);
        uint64_t normByteSize = rowCount * NormTypeByteSizeT;
        uint64_t maskByteSize = utils::SafeDivUp(rowCount, __CHAR_BIT__);
        uint64_t needIncHostByteSizes = featureByteSize + normByteSize + maskByteSize;
        errorCode = hcps::handler::helper::UseIncBindMemory(std::shared_ptr<hmm::OckHmmHeteroMemoryMgrBase>(hmmMgr),
            needIncHostByteSizes, "FeatureGenerator");
        if (errorCode != hmm::HMM_SUCCESS) {
            return std::shared_ptr<adapter::OckVsaAnnFeature>();
        }
        auto usedInfo = hmmMgr.GetUsedInfo(64ULL * 1024ULL * 1024ULL);
        OCK_VSA_HPP_LOG_INFO("FeatureGenerator need: " << needIncHostByteSizes << "usedInfo: " << *usedInfo);
        return std::make_shared<adapter::OckVsaAnnFeature>(
            hcps::handler::helper::MakeHostHmo(hmmMgr, featureByteSize, errorCode),
            hcps::handler::helper::MakeHostHmo(hmmMgr, normByteSize, errorCode),
            hcps::handler::helper::MakeHostHmo(hmmMgr, maskByteSize, errorCode), 0UL);
    }

private:
    uint32_t rowCount{ 0 };
    hmm::OckHmmHeteroMemoryMgrBase &hmmMgr;
};
} // namespace impl

template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait, uint64_t BitPerDimT>
OckVsaErrorCode OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait, BitPerDimT>::SearchByFullFilter(
    const OckVsaAnnSingleBatchQueryCondition<DataT, DimSizeT, KeyTrait> &singleBatchQueryCond,
    const impl::OckVsaHPPMaskQuery &maskQuery, std::shared_ptr<hcps::OckHeteroOperatorBase> rawSearchOp,
    OckFloatTopNQueue &outResult, std::shared_ptr<adapter::OckVsaAnnCollectResultProcessor<DataT, DimSizeT>> processor,
    OckVsaHPPSearchStaticInfo &stInfo)
{
    OckVsaErrorCode errorCode = hmm::HMM_SUCCESS;
    auto stream = hcps::handler::helper::MakeStream(*handler, errorCode);

    impl::OckVsaHPPAssembleDataContextSimple<DataT, DimSizeT> simpleContext(usedFeatures, usedNorms, maskQuery,
        *idMapMgr, innerIdConvertor, *param, groupIdDeque);
    errorCode = impl::AssembleDataByFullFilter<DataT, DimSizeT, NormTypeByteSizeT>(*handler, simpleContext, *processor,
        rawSearchOp, stInfo);

    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    auto notifyEndStartTime = std::chrono::steady_clock::now();
    auto topNResult = processor->NotifyResultEnd(errorCode);

    outResult.AddNodes(topNResult);
    stInfo.notifyEndTime = ElapsedMicroSeconds(notifyEndStartTime);
    stInfo.filterType = OckVsaHPPFilterType::FULL_FILTER;
    return errorCode;
}
template <typename DataT, uint64_t DimSizeT>
void LogIsolateCellResultIntoTopNResult(const ::ock::vsa::neighbor::relation::OckVsaNeighborRelationTopNResult &result,
    OckFloatTopNQueue &outResult,
    const std::deque<std::shared_ptr<relation::OckVsaNeighborRelationHmoGroup>> &relationGrp,
    const hcps::hfo::OckLightIdxMap &idxMap, const impl::OckVsaHPPMaskQuery &maskQuery,
    const adapter::OckVsaHPPInnerIdConvertor &innerConvertor, const std::deque<uint32_t> &groupIdDeque)
{
    for (uint32_t grpId = 0; grpId < result.grpTopIds.size(); ++grpId) {
        auto &ids = result.grpTopIds[grpId];
        auto &dis = result.grpTopDistances[grpId];
        auto &rel = relationGrp[grpId];
        auto &mask = maskQuery.GroupQuery(grpId);
        for (uint32_t i = 0; i < ids.size(); ++i) {
            if (rel->At(ids[i]).Isolate() && mask.BitSet().At(relationGrp[grpId]->relationTable[ids[i]]->primary)) {
                outResult.AddData(
                    innerConvertor.ToIdx(groupIdDeque[grpId], relationGrp[grpId]->relationTable[ids[i]]->primary),
                    dis[i]);
            }
        }
    }
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait, uint64_t BitPerDimT>
OckVsaErrorCode OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait, BitPerDimT>::Search(
    const OckVsaAnnSingleBatchQueryCondition<DataT, DimSizeT, KeyTrait> &singleBatchQueryCond,
    const std::vector<std::shared_ptr<hmm::OckHmmSubHMObject>> &maskDatas,
    std::shared_ptr<hcps::OckHeteroOperatorBase> rawSearchOp, OckFloatTopNQueue &outResult,
    OckVsaHPPSearchStaticInfo &stInfo)
{
    OckVsaErrorCode errorCode = hmm::HMM_SUCCESS;
    auto queryFeature = singleBatchQueryCond.BuildQueryFeatureVec();
    uint32_t topN = singleBatchQueryCond.topk;
    auto processor =
        adapter::OckVsaAnnCollectResultProcessor<DataT, DimSizeT>::CreateNPUProcessor(handler, queryFeature, topN);
    auto processorInitStartTime = std::chrono::steady_clock::now();
    errorCode = processor->Init();
    stInfo.processorInitTime = ElapsedMicroSeconds(processorInitStartTime);

    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    auto maskQuery =
        std::make_shared<impl::OckVsaHPPMaskQuery>(maskDatas, param->GroupRowCount(), param->SliceRowCount());
    maskQuery->MergeValidTags(usedValidTags);
    double coverageRate = static_cast<double>(maskQuery->UsedCount()) / (param->GroupRowCount() * 1.0 * GroupCount());
    uint32_t curGroupCount = static_cast<uint32_t>(usedFeatures.size());
    uint32_t sampleTopN = impl::CalcTopNInSampleGroup(topN, curGroupCount, coverageRate, param->GroupRowCount(),
        usedNeighborRelationGroups) * 2U;
    OCK_VSA_HPP_LOG_INFO("Search topN=" << topN << " coverageRate=" << coverageRate << " curGroupCount=" <<
        curGroupCount << " sampleTopN:" << sampleTopN << " OckVsaSampleFeatureMgr:" << *sampleFeatureMgr);
    if (impl::UsingPureFilter(coverageRate)) {
        return SearchByFullFilter(singleBatchQueryCond, *maskQuery, rawSearchOp, outResult, processor, stInfo);
    }

    adapter::OckVsaNeighborSampleInfo sampleInfo(sampleFeatureMgr->blockRowCount, sampleFeatureMgr->lastBlockRowCount,
        sampleFeatureMgr->shapedFeatureBlockListInNpu, sampleFeatureMgr->normBlockListInNpu,
        sampleFeatureMgr->groupRowCountInfo);
    auto primaryTopNResult = processor->GetSampleCellTopNResult(sampleInfo, sampleTopN, errorCode);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);

    auto stream = hcps::handler::helper::MakeStream(*handler, errorCode);
    // 先添加rawSearchOp, 与HOST端的刷选过程并行。
    stream->AddOp(rawSearchOp);
    auto selectSlice = impl::OckVsaHPPSliceIdMgr::Create(static_cast<uint32_t>(usedFeatures.size()));
    impl::OckVsaHPPAssembleDataContext<DataT, DimSizeT> context(*selectSlice, usedFeatures, usedNorms, *maskQuery,
        *idMapMgr, innerIdConvertor, *param, groupIdDeque);

    impl::AddSelectMayBeRowsByPrimaryResultOp<DataT, DimSizeT>(*stream, *primaryTopNResult, context, topN,
        usedNeighborRelationGroups);

    adapter::OckVsaHPPInnerIdConvertor innerIdCvt(
        adapter::OckVsaHPPInnerIdConvertor::CalcBitCount(param->GroupRowCount()));

    errorCode = stream->WaitExecComplete();
    OCK_CHECK_RETURN_ERRORCODE(errorCode);

    OCK_VSA_HPP_LOG_INFO("SelectSlice Result count:" << selectSlice->SliceCount());

    errorCode =
        impl::CalcAssembleDataTopKByRowSet<DataT, DimSizeT, NormTypeByteSizeT>(*handler, context, *processor, stInfo);
    stInfo.filterType = OckVsaHPPFilterType::SLICE_FILTER;
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    // 将processor结果添加到对外结果中
    auto notifyEndStartTime = std::chrono::steady_clock::now();
    auto topNResult = processor->NotifyResultEnd(errorCode);
    outResult.AddNodes(topNResult);
    stInfo.notifyEndTime = ElapsedMicroSeconds(notifyEndStartTime);
    return errorCode;
}
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif