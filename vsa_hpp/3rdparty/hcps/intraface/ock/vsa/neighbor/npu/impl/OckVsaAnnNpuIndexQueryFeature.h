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


#ifndef OCK_VSA_NPU_ANN_INDEX_QUERY_FEATURE_H
#define OCK_VSA_NPU_ANN_INDEX_QUERY_FEATURE_H
#include "ock/vsa/neighbor/npu/impl/OckVsaAnnNpuIndexSystem.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace npu {
inline void CheckTopNDataValid(const std::vector<hcps::algo::FloatNode> &nodes, std::string const & fName,
    uint64_t lineNo)
{
    for (uint32_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i].idx == hcps::hfo::INVALID_IDX_VALUE) {
            OCK_HCPS_LOG_WARN(fName << "[" << lineNo << "] Invalid data: " << nodes[i]);
        }
    }
}
template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::Search(
    const OckVsaAnnQueryCondition<DataTemp, DimSizeTemp, KeyTraitTemp> &queryCond,
    OckVsaAnnQueryResult<DataTemp, KeyTraitTemp> &outResult)
{
    if (queryCond.queryFeature == nullptr || queryCond.attrFilter == nullptr ||
        queryCond.queryBatchCount > MAX_SEARCH_BATCH_SIZE || queryCond.topk > MAX_SEARCH_TOPK || queryCond.topk == 0) {
        OCK_HCPS_LOG_ERROR("Input param cannot be nullptr, batchSize cannot exceed " << MAX_SEARCH_BATCH_SIZE <<
            ", topK cannot exceed " << MAX_SEARCH_TOPK << "and cannot be smaller than 0.");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    if (outResult.labels == nullptr || outResult.distances == nullptr || outResult.validNums == nullptr ||
        outResult.queryCount != queryCond.queryBatchCount || outResult.topk != queryCond.topk) {
        OCK_HCPS_LOG_ERROR("Input OckVsaAnnQueryResult cannot have nullptr, batchSize should be equal to " <<
            queryCond.queryBatchCount << ", topK should be equal to " << queryCond.topk);
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    if (queryCond.extraMask != nullptr &&
        queryCond.extraMaskLenEachQuery != utils::SafeDivUp(GetFeatureNum(), __CHAR_BIT__)) {
        OCK_HCPS_LOG_ERROR("queryCond.extraMaskLenEachQuery should be equal to " <<
            utils::SafeDivUp(GetFeatureNum(), __CHAR_BIT__) << ", input is " << queryCond.extraMaskLenEachQuery);
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    if (GetFeatureNum() == 0) {
        OCK_HCPS_LOG_WARN("NPU blockGroup is null, return");
        return VSA_ERROR_EMPTY_BASE;
    }

    if (GetFeatureNum() < queryCond.topk) {
        OCK_HCPS_LOG_WARN("topK value(" << queryCond.topk << ") is bigger than feature num(" << GetFeatureNum() << ")");
    }
    auto topNQueues = OckFloatTopNQueue::CreateMany(queryCond.queryBatchCount, queryCond.topk);
    auto ret = this->Search(queryCond, topNQueues);
    for (uint32_t i = 0; i < topNQueues.size(); ++i) {
        auto nodes = topNQueues[i]->PopAll();
        std::reverse(nodes->begin(), nodes->end());
        CheckTopNDataValid(*nodes, OCK_HMM_LOG_FILENAME, __LINE__);
        outResult.AddResult(i, *nodes);
    }
    return ret;
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::Search(
    const OckVsaAnnSingleBatchQueryCondition<DataT, DimSizeTemp, KeyTraitTemp> &queryCond, OckFloatTopNQueue &outResult)
{
    if (queryCond.queryFeature == nullptr || queryCond.attrFilter == nullptr || queryCond.topk > MAX_SEARCH_TOPK ||
        queryCond.topk == 0) {
        OCK_HCPS_LOG_ERROR("Input param cannot be nullptr, batchSize cannot exceed " << MAX_SEARCH_BATCH_SIZE <<
            ", topK cannot exceed " << MAX_SEARCH_TOPK << "and cannot be smaller than 0.");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    if (GetFeatureNum() == 0) {
        OCK_HCPS_LOG_WARN("NPU blockGroup is null, return");
        return VSA_ERROR_EMPTY_BASE;
    }

    if (GetFeatureNum() < queryCond.topk) {
        OCK_HCPS_LOG_ERROR("topK value(" << queryCond.topk << ") is bigger than feature num(" << GetFeatureNum() <<
            ")");
    }
    // makedevicehmo
    OckVsaErrorCode errorCode = VSA_SUCCESS;
    auto topkDistHmo = hcps::handler::helper::MakeDeviceHmo(*handler, sizeof(OckFloat16) * queryCond.topk, errorCode);
    auto topkLabelsHmo = hcps::handler::helper::MakeDeviceHmo(*handler, sizeof(uint64_t) * queryCond.topk, errorCode);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);

    // prepare search input
    errorCode = DoSearchProcess(1UL, queryCond, topkDistHmo, topkLabelsHmo);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    // put result into outResult
    auto topkDistHmoHost = hcps::handler::helper::CopyToHostHmo(*handler, topkDistHmo, errorCode);
    auto topkLabelHmoHost = hcps::handler::helper::CopyToHostHmo(*handler, topkLabelsHmo, errorCode);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    std::vector<uint64_t> rawLabel(queryCond.topk);
    std::vector<uint64_t> innerIdx(queryCond.topk);
    CalculateInnerIdx(reinterpret_cast<uint64_t *>(topkLabelHmoHost->Addr()), queryCond.topk, innerIdx.data());

    idMapMgr->GetOutterIdxs(innerIdx.data(), queryCond.topk, rawLabel.data());

    AddDatas(std::min(queryCond.topk, static_cast<uint32_t>(GetFeatureNum())), outResult, rawLabel.data(),
        reinterpret_cast<OckFloat16 *>(topkDistHmoHost->Addr()));
    return VSA_SUCCESS;
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::GenerateQueryNorm(
    std::shared_ptr<hcps::nop::OckL2NormOpHmoBlock> hmoNormGroup)
{
    if (hmoNormGroup == nullptr) {
        OCK_HCPS_LOG_ERROR("Input param cannot be nullptr");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    OckVsaErrorCode streamRet = VSA_SUCCESS;
    auto streamBase = hcps::handler::helper::MakeStream(*handler, streamRet, hcps::OckDevStreamType::AI_CORE);
    streamRet = hcps::nop::OckL2NormOpRun::ComputeNormSync(hmoNormGroup, *handler, streamBase);
    OCK_CHECK_RETURN_ERRORCODE(streamRet);
    return streamRet;
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::GenerateQueryAttr(
    uint32_t batch, const OckVsaAnnSingleBatchQueryCondition<DataT, DimSizeTemp, KeyTraitTemp> &queryCond,
    std::vector<std::shared_ptr<hmm::OckHmmHMObject>> &queryTimeHmoVec,
    std::vector<std::shared_ptr<hmm::OckHmmHMObject>> &queryTokenIdHmoVec)
{
    auto tokenNumber = param->TokenNum();
    if (queryCond.shareAttrFilter) {
        auto ret = GenerateSingleQueryAttr(queryCond.attrFilter, queryTimeHmoVec, queryTokenIdHmoVec, tokenNumber,
            queryCond.enableTimeFilter);
        OCK_CHECK_RETURN_ERRORCODE(ret);
    } else {
        for (uint32_t innerBatchId = 0; innerBatchId < batch; ++innerBatchId) {
            auto ret = GenerateSingleQueryAttr(queryCond.attrFilter + innerBatchId, queryTimeHmoVec, queryTokenIdHmoVec,
                tokenNumber, queryCond.enableTimeFilter);
            OCK_CHECK_RETURN_ERRORCODE(ret);
        }
    }
    return VSA_SUCCESS;
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::GenerateSingleQueryAttr(
    const KeyTraitTemp *attrFilter, std::vector<std::shared_ptr<hmm::OckHmmHMObject>> &queryTimeHmoVec,
    std::vector<std::shared_ptr<hmm::OckHmmHMObject>> &queryTokenIdHmoVec, uint32_t maxTokenNumber,
    bool enableTimeFilter)
{
    if (attrFilter == nullptr) {
        OCK_HCPS_LOG_ERROR("Input param cannot be nullptr");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    // 生成mask算子需要的query attr
    // get filter time info from attrFilter
    auto timeStart = enableTimeFilter ? (attrFilter->minTime * (-1)) : 0;
    auto timeEnd = enableTimeFilter ? (attrFilter->maxTime * (-1)) : (std::numeric_limits<int32_t>::max() * (-1));
    std::vector<int32_t> timeVec(__CHAR_BIT__, 0);
    timeVec[0] = timeStart;
    timeVec[1] = timeEnd;

    // alloc query time hmo and copy time info to hmo
    OckVsaErrorCode errorCode = VSA_SUCCESS;
    auto queryTimeHmo = hcps::handler::helper::MakeDeviceHmo(*handler, sizeof(int32_t) * __CHAR_BIT__, errorCode);
    auto queryTokenHmo = hcps::handler::helper::MakeDeviceHmo(*handler,
        utils::SafeDivUp(maxTokenNumber, __CHAR_BIT__) * OPS_DATA_TYPE_TIMES * sizeof(uint8_t), errorCode);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);

    auto cpyRet = ockSyncUtils->Copy(reinterpret_cast<void *>(queryTimeHmo->Addr()), sizeof(int32_t) * __CHAR_BIT__,
        timeVec.data(), sizeof(int32_t) * __CHAR_BIT__, acladapter::OckMemoryCopyKind::HOST_TO_DEVICE);
    OCK_CHECK_RETURN_ERRORCODE(cpyRet);
    queryTimeHmoVec.emplace_back(queryTimeHmo);

    // get tokenId from attrFilter
    std::vector<uint8_t> tokenIdVec(utils::SafeDivUp(maxTokenNumber, __CHAR_BIT__) * OPS_DATA_TYPE_TIMES,
        OPS_DATA_PADDING_VAL);
    ChangeBitSetToVector(attrFilter, maxTokenNumber, tokenIdVec);

    // alloc token id hmo and copy token data to hmo
    cpyRet = ockSyncUtils->Copy(reinterpret_cast<void *>(queryTokenHmo->Addr()),
        utils::SafeDivUp(maxTokenNumber, __CHAR_BIT__) * OPS_DATA_TYPE_TIMES, tokenIdVec.data(),
        utils::SafeDivUp(maxTokenNumber, __CHAR_BIT__) * OPS_DATA_TYPE_TIMES,
        acladapter::OckMemoryCopyKind::HOST_TO_DEVICE);
    OCK_CHECK_RETURN_ERRORCODE(cpyRet);

    queryTokenIdHmoVec.emplace_back(queryTokenHmo);
    return VSA_SUCCESS;
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
void OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::ChangeBitSetToVector(
    const KeyTraitTemp *attrFilter, uint32_t maxTokenNumber, std::vector<uint8_t> &tokenIdVec)
{
    auto bitSet = reinterpret_cast<uint8_t *>((attrFilter)->bitSet.dataHolder);
    for (size_t tokenPos = 0; tokenPos < utils::SafeDivUp(maxTokenNumber, __CHAR_BIT__); ++tokenPos) {
        tokenIdVec[tokenPos * OPS_DATA_TYPE_TIMES] = *(bitSet + tokenPos);
    }
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
std::shared_ptr<hcps::nop::OckDistMaskGenOpHmoGroups> OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::PrepareMaskData(
    const std::vector<std::shared_ptr<hmm::OckHmmHMObject>> &queryTimeHmoVec,
    const std::vector<std::shared_ptr<hmm::OckHmmHMObject>> &queryTokenIdHmoVec,
    std::shared_ptr<hmm::OckHmmSubHMObject> maskHmo, const std::deque<OckVsaAnnKeyAttrInfo> &attrFeature,
    uint64_t maskLen, uint32_t blockNumPerGroup)
{
    if (maskHmo == nullptr) {
        OCK_HCPS_LOG_ERROR("Input param cannot be nullptr");
    }
    std::shared_ptr<hcps::nop::OckDistMaskGenOpHmoGroups> hmoMaskGroups =
        std::make_shared<hcps::nop::OckDistMaskGenOpHmoGroups>();
    auto tokenNumber = param->TokenNum();
    hmoMaskGroups->queryTimes = queryTimeHmoVec;
    hmoMaskGroups->queryTokenIds = queryTokenIdHmoVec;
    hmoMaskGroups->mask = maskHmo;
    hmoMaskGroups->tokenNum = tokenNumber;
    hmoMaskGroups->featureAttrBlockSize = hcps::nop::DEFAULT_CODE_BLOCK_SIZE;
    hmoMaskGroups->blockCount = blockNumPerGroup;
    hmoMaskGroups->maskLen = maskLen;
    for (size_t i = 0; i < attrFeature.size(); ++i) {
        hmoMaskGroups->attrTimes.emplace_back(hmm::OckHmmHMObject::CreateSubHmoList(attrFeature[i].keyAttrTime,
            hcps::nop::DEFAULT_CODE_BLOCK_SIZE * blockNumPerGroup * sizeof(int32_t)));
        hmoMaskGroups->attrTokenQuotients.emplace_back(hmm::OckHmmHMObject::CreateSubHmoList(
            attrFeature[i].keyAttrQuotient, hcps::nop::DEFAULT_CODE_BLOCK_SIZE * blockNumPerGroup * sizeof(int32_t)));
        hmoMaskGroups->attrTokenRemainders.emplace_back(
            hmm::OckHmmHMObject::CreateSubHmoList(attrFeature[i].keyAttrRemainder,
            hcps::nop::DEFAULT_CODE_BLOCK_SIZE * blockNumPerGroup * hcps::nop::OPS_DATA_TYPE_TIMES * sizeof(uint8_t)));
    }
    return hmoMaskGroups;
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
std::shared_ptr<hcps::nop::OckDistMaskWithExtraGenOpHmoGroups> OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::PrepareMaskDataWithExtra(
    const std::vector<std::shared_ptr<hmm::OckHmmHMObject>> &queryTimeHmoVec,
    const std::vector<std::shared_ptr<hmm::OckHmmHMObject>> &queryTokenIdHmoVec,
    std::shared_ptr<hmm::OckHmmSubHMObject> maskHmo, std::shared_ptr<hmm::OckHmmSubHMObject> extraMaskHmo,
    const std::deque<OckVsaAnnKeyAttrInfo> &attrFeature, uint64_t maskLen, uint32_t blockNumPerGroup)
{
    std::shared_ptr<hcps::nop::OckDistMaskWithExtraGenOpHmoGroups> extraMaskHmoGroups =
        std::make_shared<hcps::nop::OckDistMaskWithExtraGenOpHmoGroups>();
    auto tokenNumber = param->TokenNum();
    extraMaskHmoGroups->queryTimes = queryTimeHmoVec;
    extraMaskHmoGroups->queryTokenIds = queryTokenIdHmoVec;
    extraMaskHmoGroups->mask = maskHmo;
    extraMaskHmoGroups->extraMask = extraMaskHmo;
    extraMaskHmoGroups->tokenNum = tokenNumber;
    extraMaskHmoGroups->featureAttrBlockSize = hcps::nop::DEFAULT_CODE_BLOCK_SIZE;
    extraMaskHmoGroups->blockCount = blockNumPerGroup;
    extraMaskHmoGroups->maskLen = static_cast<uint32_t>(maskLen);
    for (size_t i = 0; i < attrFeature.size(); ++i) {
        extraMaskHmoGroups->attrTimes.emplace_back(hmm::OckHmmHMObject::CreateSubHmoList(attrFeature[i].keyAttrTime,
            hcps::nop::DEFAULT_CODE_BLOCK_SIZE * blockNumPerGroup * sizeof(int32_t)));
        extraMaskHmoGroups->attrTokenQuotients.emplace_back(hmm::OckHmmHMObject::CreateSubHmoList(
            attrFeature[i].keyAttrQuotient, hcps::nop::DEFAULT_CODE_BLOCK_SIZE * blockNumPerGroup * sizeof(int32_t)));
        extraMaskHmoGroups->attrTokenRemainders.emplace_back(
            hmm::OckHmmHMObject::CreateSubHmoList(attrFeature[i].keyAttrRemainder,
            hcps::nop::DEFAULT_CODE_BLOCK_SIZE * blockNumPerGroup * hcps::nop::OPS_DATA_TYPE_TIMES * sizeof(uint8_t)));
    }
    return extraMaskHmoGroups;
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::Search(
    const OckVsaAnnQueryCondition<DataT, DimSizeTemp, KeyTraitTemp> &queryCond,
    std::vector<std::shared_ptr<OckFloatTopNQueue>> &outResult)
{
    if (queryCond.queryFeature == nullptr || queryCond.attrFilter == nullptr ||
        queryCond.queryBatchCount > MAX_SEARCH_BATCH_SIZE || queryCond.topk > MAX_SEARCH_TOPK || queryCond.topk == 0) {
        OCK_HCPS_LOG_ERROR("Input param cannot be nullptr, batchSize cannot exceed " << MAX_SEARCH_BATCH_SIZE <<
            ", topK cannot exceed " << MAX_SEARCH_TOPK << "and cannot be smaller than 0.");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    if (GetFeatureNum() == 0) {
        OCK_HCPS_LOG_WARN("NPU blockGroup is null, return");
        return VSA_ERROR_EMPTY_BASE;
    }

    if (GetFeatureNum() < queryCond.topk) {
        OCK_HCPS_LOG_ERROR("topK value(" << queryCond.topk << ") is bigger than feature num(" << GetFeatureNum() <<
            ")");
    }
    std::vector<int> searchBatchSizes = { 4, 2, 1 };
    uint32_t offset = 0;
    int searchSize = queryCond.queryBatchCount;
    for (auto batch : searchBatchSizes) {
        while (searchSize >= batch) {
            OCK_CHECK_RETURN_ERRORCODE(SearchImpl(batch, queryCond.QueryCondAt(offset), outResult, offset));
            offset += batch;
            searchSize -= batch;
        }
    }
    return VSA_SUCCESS;
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::SearchImpl(int batch,
    const OckVsaAnnSingleBatchQueryCondition<DataT, DimSizeTemp, KeyTraitTemp> &queryCond,
    std::vector<std::shared_ptr<OckFloatTopNQueue>> &outResult, uint32_t offset)
{
    // prepare search compose op
    // create empty topk dist hmo to store the output of search op
    OckVsaErrorCode errorCode = VSA_SUCCESS;
    auto topkDistHmo =
        hcps::handler::helper::MakeDeviceHmo(*handler, sizeof(OckFloat16) * queryCond.topk * batch, errorCode);
    auto topkLabelsHmo =
        hcps::handler::helper::MakeDeviceHmo(*handler, sizeof(uint64_t) * queryCond.topk * batch, errorCode);

    OCK_CHECK_RETURN_ERRORCODE(errorCode);

    // comp result error
    OCK_CHECK_RETURN_ERRORCODE(DoSearchProcess(batch, queryCond, topkDistHmo, topkLabelsHmo));
    // put result into outResult
    auto topkDistHmoHost = hcps::handler::helper::CopyToHostHmo(*handler, topkDistHmo, errorCode);
    auto topkLabelHmoHost = hcps::handler::helper::CopyToHostHmo(*handler, topkLabelsHmo, errorCode);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    std::vector<uint64_t> innerIdx(queryCond.topk * batch, hcps::hfo::INVALID_IDX_VALUE);
    CalculateInnerIdx(reinterpret_cast<uint64_t *>(topkLabelHmoHost->Addr()), queryCond.topk * batch, innerIdx.data());

    std::vector<uint64_t> rawLabel(queryCond.topk * batch);
    idMapMgr->GetOutterIdxs(innerIdx.data(), queryCond.topk * batch, rawLabel.data());

    PushDataToResult(batch, outResult.begin() + offset, queryCond, rawLabel.data(),
        reinterpret_cast<OckFloat16 *>(topkDistHmoHost->Addr()));
    return VSA_SUCCESS;
}
template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
void OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::CalculateInnerIdx(uint64_t *topkLabel,
    uint64_t count, uint64_t *innerIdx)
{
    for (uint64_t i = 0; i < count; ++i) {
        if (topkLabel[i] == hcps::hfo::INVALID_IDX_VALUE) {
            continue;
        }
        uint32_t curGrpPos = static_cast<uint32_t>(topkLabel[i]) / param->GroupRowCount();
        uint32_t grpId = groupIdDeque[curGrpPos];
        uint64_t innerId = innerIdCvt.ToIdx(grpId, static_cast<uint32_t>(topkLabel[i] % param->GroupRowCount()));
        innerIdx[i] = innerId;
    }
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
void OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::PushDataToResult(int batch,
    std::vector<std::shared_ptr<OckFloatTopNQueue>>::iterator outResult,
    const OckVsaAnnSingleBatchQueryCondition<DataT, DimSizeTemp, KeyTraitTemp> &queryCond, uint64_t *topkLabel,
    OckFloat16 *topkDist)
{
    for (int startBatchPos = 0; startBatchPos < batch; ++startBatchPos) {
        AddDatas(std::min(queryCond.topk, static_cast<uint32_t>(GetFeatureNum())), *(outResult[startBatchPos]),
            (topkLabel + queryCond.topk * startBatchPos), (topkDist + queryCond.topk * startBatchPos));
    }
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
void OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::AddDatas(uint32_t count,
    OckFloatTopNQueue &outResult, uint64_t *idx, OckFloat16 *distance)
{
    if (idx == nullptr || distance == nullptr) {
        OCK_HCPS_LOG_ERROR("Input param cannot be nullptr");
        return;
    }
    for (uint32_t i = 0; i < count; ++i) {
        if (idx[i] == hcps::hfo::INVALID_IDX_VALUE ||
            (acladapter::OckAscendFp16::Fp16ToFloat(distance[i]) > distanceThreshold)) {
            OCK_HCPS_LOG_ERROR("there is not a label in idxMap at position " << idx[i] << ", which is top " << i << ", "
                                                                             << " dis=" <<
                acladapter::OckAscendFp16::Fp16ToFloat(distance[i]));
            continue;
        }
        outResult.AddData(idx[i], acladapter::OckAscendFp16::Fp16ToFloat((distance[i])));
    }
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::DoSearchProcess(
    uint32_t batch, const OckVsaAnnSingleBatchQueryCondition<DataT, DimSizeTemp, KeyTraitTemp> &queryCond,
    std::shared_ptr<hmm::OckHmmHMObject> topkDistHmo, std::shared_ptr<hmm::OckHmmHMObject> topkLabelsHmo)
{
    // prepare and do mask generation
    OckVsaErrorCode errorCode = VSA_SUCCESS;
    // create query hmo and copy query features
    uint64_t featureNum = GetFeatureNum();
    int64_t maskLenAligned = utils::SafeDivUp(
        utils::SafeRoundUp(featureNum, param->BlockRowCount() * param->GroupBlockCount()), __CHAR_BIT__);
    auto maskHmo = hcps::handler::helper::MakeDeviceHmo(*handler, sizeof(uint8_t) * maskLenAligned * batch, errorCode);
    auto queryHmo = hcps::handler::helper::MakeDeviceHmo(*handler, sizeof(uint8_t) * DimSizeTemp * batch, errorCode);
    // create empty query norm hmo for run norm op to store the output

    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    // run mask op
    std::deque<OckVsaAnnKeyAttrInfo> attrFeatures;
    for (size_t i = 0; i < blockGroups.size(); ++i) {
        auto curGrp = blockGroups[i];
        for (uint32_t j = 0; j < curGrp->keyAttrsTime.size(); ++j) {
            OckVsaAnnKeyAttrInfo attrFeat;
            attrFeat.keyAttrTime = curGrp->keyAttrsTime[j];
            attrFeat.keyAttrQuotient = curGrp->keyAttrsQuotient[j];
            attrFeat.keyAttrRemainder = curGrp->keyAttrsRemainder[j];
            attrFeatures.emplace_back(attrFeat);
        }
    }

    if (queryCond.extraMask == nullptr) {
        errorCode = GenerateMask(queryCond, attrFeatures, maskHmo, batch, maskLenAligned);
    } else {
        errorCode = GenerateMaskWithExtra(queryCond, attrFeatures, maskHmo, batch, maskLenAligned);
    }
    OCK_CHECK_RETURN_ERRORCODE(errorCode);

    // prepare and do query norm generation
    // create query hmo and copy query features
    errorCode = ockSyncUtils->Copy(reinterpret_cast<void *>(queryHmo->Addr()), DimSizeTemp * sizeof(uint8_t) * batch,
        reinterpret_cast<const uint8_t *>(queryCond.queryFeature), DimSizeTemp * sizeof(uint8_t) * batch,
        acladapter::OckMemoryCopyKind::HOST_TO_DEVICE);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);

    // run norm op
    auto hmoNormGroup = hcps::nop::OckL2NormOpRun::BuildNormHmoBlock(queryHmo, *handler, DimSizeTemp, batch,
        errorCode);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    OCK_CHECK_RETURN_ERRORCODE(GenerateQueryNorm(hmoNormGroup));

    auto queryNormHmo = hmm::OckHmmHMObject::CreateSubHmo(hmoNormGroup->normResult, 0,
        sizeof(OckFloat16) * utils::SafeRoundUp(batch, hcps::nop::FP16_ALIGN));

    auto subMaskHmo = hmm::OckHmmHMObject::CreateSubHmo(maskHmo, 0,
        utils::SafeDivUp(std::max(featureNum, static_cast<uint64_t>(param->BlockRowCount())), __CHAR_BIT__) * batch);
    std::vector<std::shared_ptr<hcps::hcop::OckTopkDistCompOpHmoGroup>> compOpGroupVec;
    for (size_t grpId = 0; grpId < blockGroups.size(); ++grpId) {
        std::shared_ptr<hcps::hcop::OckTopkDistCompOpHmoGroup> compOpGroup =
            std::make_shared<hcps::hcop::OckTopkDistCompOpHmoGroup>(true, batch, DimSizeTemp, queryCond.topk,
            param->BlockRowCount(), param->GroupBlockCount(), featureNum, grpId);
        compOpGroup->featuresHmo = blockGroups[grpId]->features;
        compOpGroup->normsHmo = blockGroups[grpId]->norms;
        compOpGroup->SetQueryHmos(queryHmo, queryNormHmo, subMaskHmo);
        compOpGroup->SetOutputHmos(topkDistHmo, topkLabelsHmo);
        compOpGroupVec.emplace_back(compOpGroup);
    }

    return hcps::hcop::OckTopkDistCompOpRun::RunMultiGroupsSync(compOpGroupVec, handler);
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
uint64_t OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::KeyAttrByteSize(void)
{
    // 返回每条时空属性在NPU中存储的字节数
    return 10ULL;
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::GetFeatureByLabel(
    uint64_t count, const int64_t *labels, DataTemp *features) const
{
    if (count > MAX_GET_NUMBER || labels == nullptr || features == nullptr) {
        OCK_HCPS_LOG_ERROR("Input nullptr or count exceeds theshold[" << MAX_GET_NUMBER << "].");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    int64_t label;
    std::vector<uint64_t> ids;
    for (uint64_t i = 0; i < count; ++i) {
        label = static_cast<uint64_t>(labels[i]);
        auto id = idMapMgr->GetInnerIdx(label);
        if (id == hcps::hfo::INVALID_IDX_VALUE) {
            return VSA_ERROR_LABEL_NOT_EXIST;
        }
        auto innerIdxInfo = innerIdCvt.ToGroupOffset(id);

        OCK_CHECK_RETURN_ERRORCODE(CopySingleFeature(innerIdxInfo, features + i * DimSizeTemp));
    }
    return VSA_SUCCESS;
}
template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::GetFeatureAttrByLabel(
    uint64_t count, const int64_t *labels, KeyTypeTupleT *attributes) const
{
    if (count > MAX_GET_NUMBER || labels == nullptr || attributes == nullptr) {
        OCK_HCPS_LOG_ERROR("Input nullptr or count exceeds theshold[" << MAX_GET_NUMBER << "].");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    int64_t label;
    std::vector<uint64_t> ids;
    for (uint64_t i = 0; i < count; ++i) {
        label = static_cast<uint64_t>(labels[i]);
        uint64_t id = idMapMgr->GetInnerIdx(label);
        if (id == hcps::hfo::INVALID_IDX_VALUE) {
            return VSA_ERROR_LABEL_NOT_EXIST;
        }
        auto innerIdxInfo = innerIdCvt.ToGroupOffset(id);
        OCK_CHECK_RETURN_ERRORCODE(CopySingleAttr(innerIdxInfo, attributes + i));
    }
    return VSA_SUCCESS;
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
uintptr_t OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::GetCustomAttrByBlockId(
    uint32_t blockId, OckVsaErrorCode &errorCode) const
{
    uintptr_t address = 0ULL;
    if (errorCode != VSA_SUCCESS) {
        return address;
    }
    if (param->ExtKeyAttrBlockSize() == 0UL || param->GroupRowCount() % param->ExtKeyAttrBlockSize() != 0) {
        OCK_HCPS_LOG_ERROR("param error." << *param);
        errorCode = VSA_ERROR_GROUP_ROW_COUNT_DIVISIBLE;
        return address;
    }
    uint32_t customBlockCountInNpu = 0;
    for (uint32_t i = 0; i < blockGroups.size(); ++i) {
        customBlockCountInNpu += static_cast<uint32_t>(blockGroups[i]->extKeyAttrs.size());
    }
    if (blockId >= customBlockCountInNpu) {
        OCK_HCPS_LOG_ERROR("blockId exceeds threshold(" << customBlockCountInNpu << ")");
        errorCode = VSA_ERROR_INVALID_INPUT_PARAM;
        return address;
    }
    uint32_t customBlockNumInOneGroup = param->GroupRowCount() / param->ExtKeyAttrBlockSize();
    uint32_t grpOffset = blockId / customBlockNumInOneGroup;
    uint32_t offsetInGroup = blockId % customBlockNumInOneGroup;
    address = blockGroups[grpOffset]->extKeyAttrs[offsetInGroup]->Addr();
    return address;
}
template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
uint32_t OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::GetCustomAttrBlockCount(void) const
{
    return utils::SafeDivUp(static_cast<uint32_t>(this->GetFeatureNum()), this->param->ExtKeyAttrBlockSize());
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::CopySingleFeature(
    adapter::OckVsaHPPIdx innerIdxInfo, DataTemp *features) const
{
    auto errorCode = VSA_SUCCESS;
    uint32_t blockIdx = innerIdxInfo.offset / param->BlockRowCount();
    uint32_t blockOffset = innerIdxInfo.offset % param->BlockRowCount();
    uint32_t cubeOffset = blockOffset / static_cast<uint32_t>(hcps::nop::CUBE_ALIGN);
    uint32_t offsetInCube = blockOffset % static_cast<uint32_t>(hcps::nop::CUBE_ALIGN);

    std::shared_ptr<hmm::OckHmmHMObject> featHmo = blockGroups[grpPosMap[innerIdxInfo.grpId]]->features[blockIdx];
    uintptr_t startPos = featHmo->Addr() + cubeOffset * DimSizeTemp * hcps::nop::CUBE_ALIGN +
        offsetInCube * hcps::nop::CUBE_ALIGN_INT8;
    for (uint32_t i = 0; i < __CHAR_BIT__; ++i) {
        errorCode = ockSyncUtils->Copy(features + i * hcps::nop::CUBE_ALIGN_INT8,
            hcps::nop::CUBE_ALIGN_INT8 * sizeof(int8_t),
            reinterpret_cast<void *>(startPos + hcps::nop::CUBE_ALIGN * hcps::nop::CUBE_ALIGN_INT8 * i),
            hcps::nop::CUBE_ALIGN_INT8 * sizeof(int8_t), acladapter::OckMemoryCopyKind::DEVICE_TO_HOST);
        OCK_CHECK_RETURN_ERRORCODE(errorCode);
    }
    return errorCode;
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::CopySingleAttr(
    adapter::OckVsaHPPIdx innerIdxInfo, KeyTypeTupleT *attributes) const
{
    uint32_t blockOffset = innerIdxInfo.offset;
    std::shared_ptr<hmm::OckHmmHMObject> attrTimeHmo = blockGroups[grpPosMap[innerIdxInfo.grpId]]->keyAttrsTime[0UL];
    auto attrTimeBuffer = attrTimeHmo->GetBuffer(hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY,
        blockOffset * sizeof(int32_t), sizeof(int32_t));

    std::shared_ptr<hmm::OckHmmHMObject> attrQuotHmo =
        blockGroups[grpPosMap[innerIdxInfo.grpId]]->keyAttrsQuotient[0UL];
    auto attrQuotBuffer = attrQuotHmo->GetBuffer(hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY,
        blockOffset * sizeof(int32_t), sizeof(int32_t));
    std::shared_ptr<hmm::OckHmmHMObject> attrRemainHmo =
        blockGroups[grpPosMap[innerIdxInfo.grpId]]->keyAttrsRemainder[0UL];
    auto attrRemainBuffer = attrRemainHmo->GetBuffer(hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY,
        blockOffset * sizeof(int8_t) * OPS_DATA_TYPE_TIMES, sizeof(int8_t));
    if (attrTimeBuffer == nullptr || attrQuotBuffer == nullptr || attrRemainBuffer == nullptr) {
        OCK_HCPS_LOG_ERROR("With grpId is " << grpPosMap[innerIdxInfo.grpId] << ", blockOffset is " << blockOffset <<
            ". Nullptr is " << ((attrTimeBuffer == nullptr) ? "attrTimeBuffer; " : "") <<
            ((attrQuotBuffer == nullptr) ? "attrQuotBuffer; " : "") <<
            ((attrRemainBuffer == nullptr) ? "attrRemainBuffer" : ""));
        return hmm::HMM_ERROR_HMO_BUFFER_NOT_ALLOCED;
    }
    attr::OckTimeSpaceAttr toData;
    OckVsaAnnKeyAttrInfo info;
    info.AttrCvt(toData, attrTimeBuffer->Address(), attrQuotBuffer->Address(), attrRemainBuffer->Address());
    auto ret = memcpy_s(attributes, sizeof(KeyTypeTupleT), &toData, sizeof(KeyTypeTupleT));
    if (ret != EOK) {
        return ret;
    }
    return VSA_SUCCESS;
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::GetMaskResult(
    const OckVsaAnnSingleBatchQueryCondition<DataTemp, DimSizeTemp, KeyTraitTemp> &queryCond,
    const std::deque<OckVsaAnnKeyAttrInfo> &attrFeatureGroups, uint32_t rowCountPerGroup,
    std::shared_ptr<hmm::OckHmmHMObject> maskHmo)
{
    std::vector<std::shared_ptr<hmm::OckHmmHMObject>> queryTimeHmoVec;
    std::vector<std::shared_ptr<hmm::OckHmmHMObject>> queryTokenIdHmoVec;
    auto ret = GenerateQueryAttr(1U, queryCond, queryTimeHmoVec, queryTokenIdHmoVec);
    OCK_CHECK_RETURN_ERRORCODE(ret);

    OckVsaErrorCode errorCode = VSA_SUCCESS;
    auto streamBase = hcps::handler::helper::MakeStream(*handler, errorCode, hcps::OckDevStreamType::AI_CORE);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    for (uint32_t i = 0; i < attrFeatureGroups.size(); ++i) {
        uint32_t subMaskLen = utils::SafeDivUp(rowCountPerGroup, __CHAR_BIT__);
        auto subMaskHmo = hmm::OckHmmHMObject::CreateSubHmo(maskHmo, subMaskLen * i, subMaskLen);
        auto extraMaskHmo = hcps::handler::helper::MakeDeviceHmo(*handler, subMaskLen * sizeof(uint8_t), errorCode);
        OCK_CHECK_RETURN_ERRORCODE(errorCode);
        if (queryCond.extraMask == nullptr) {
            auto hmoMaskGroups = PrepareMaskData(queryTimeHmoVec, queryTokenIdHmoVec, subMaskHmo,
                { attrFeatureGroups[i] }, subMaskLen, param->GroupBlockCount());
            hcps::nop::OckDistMaskGenOpRun::AddMaskOpsSingleBatch(hmoMaskGroups, streamBase);
            errorCode = streamBase->WaitExecComplete();
            OCK_CHECK_RETURN_ERRORCODE(errorCode);
        } else {
            errorCode = ockSyncUtils->Copy(reinterpret_cast<void *>(extraMaskHmo->Addr()), extraMaskHmo->GetByteSize(),
                queryCond.extraMask + i * subMaskLen, subMaskLen * sizeof(uint8_t),
                queryCond.extraMaskIsAtDevice ? acladapter::OckMemoryCopyKind::DEVICE_TO_DEVICE :
                                                acladapter::OckMemoryCopyKind::HOST_TO_DEVICE);
            OCK_CHECK_RETURN_ERRORCODE(errorCode);
            auto extraMaskHmoGroups = PrepareMaskDataWithExtra(queryTimeHmoVec, queryTokenIdHmoVec, subMaskHmo,
                extraMaskHmo, { attrFeatureGroups[i] }, subMaskLen, param->GroupBlockCount());
            hcps::nop::OckDistMaskWithExtraGenOpRun::AddMaskWithExtraOpsSingleBatch(extraMaskHmoGroups, streamBase);
            errorCode = streamBase->WaitExecComplete();
            OCK_CHECK_RETURN_ERRORCODE(errorCode);
        }
    }
    return VSA_SUCCESS;
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::GenerateMask(
    const OckVsaAnnSingleBatchQueryCondition<DataT, DimSizeTemp, KeyTraitT> &queryCond,
    const std::deque<OckVsaAnnKeyAttrInfo> &attrFeatureGroups, std::shared_ptr<hmm::OckHmmHMObject> maskHmo,
    uint32_t queryBatch, uint64_t maskLenAligned)
{
    std::vector<std::shared_ptr<hmm::OckHmmHMObject>> queryTimeHmoVec;
    std::vector<std::shared_ptr<hmm::OckHmmHMObject>> queryTokenIdHmoVec;
    auto ret = GenerateQueryAttr(queryBatch, queryCond, queryTimeHmoVec, queryTokenIdHmoVec);
    OCK_CHECK_RETURN_ERRORCODE(ret);

    OckVsaErrorCode errorCode = VSA_SUCCESS;
    auto streamBase = hcps::handler::helper::MakeStream(*handler, errorCode, hcps::OckDevStreamType::AI_CORE);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);

    uint64_t maskLen = utils::SafeDivUp(GetFeatureNum(), __CHAR_BIT__);
    if (queryCond.shareAttrFilter) {
        auto singleMaskHmo =
            hcps::handler::helper::MakeDeviceHmo(*handler, maskLenAligned * sizeof(uint8_t), errorCode);
        OCK_CHECK_RETURN_ERRORCODE(errorCode);
        auto hmoMaskGroups = PrepareMaskData(queryTimeHmoVec, queryTokenIdHmoVec, singleMaskHmo, attrFeatureGroups,
            maskLen, param->GroupBlockCount());
        hcps::nop::OckDistMaskGenOpRun::AddMaskOpsSingleBatch(hmoMaskGroups, streamBase);
        errorCode = streamBase->WaitExecComplete();
        OCK_CHECK_RETURN_ERRORCODE(errorCode);
        for (uint32_t i = 0; i < queryBatch; ++i) {
            errorCode = handler->HmmMgrPtr()->CopyHMO(*maskHmo, maskLen * sizeof(uint8_t) * i, *singleMaskHmo, 0,
                maskLen * sizeof(uint8_t));
            OCK_CHECK_RETURN_ERRORCODE(errorCode);
        }
        return VSA_SUCCESS;
    }
    auto hmoMaskGroups = PrepareMaskData(queryTimeHmoVec, queryTokenIdHmoVec, maskHmo, attrFeatureGroups, maskLen,
        param->GroupBlockCount());
    errorCode = hcps::nop::OckDistMaskGenOpRun::AddMaskOpsMultiBatches(hmoMaskGroups, streamBase);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);

    return VSA_SUCCESS;
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
OckVsaErrorCode OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::GenerateMaskWithExtra(
    const OckVsaAnnSingleBatchQueryCondition<DataT, DimSizeTemp, KeyTraitT> &queryCond,
    const std::deque<OckVsaAnnKeyAttrInfo> &attrFeatureGroups, std::shared_ptr<hmm::OckHmmHMObject> maskHmo,
    uint32_t queryBatch, uint64_t maskLenAligned)
{
    std::vector<std::shared_ptr<hmm::OckHmmHMObject>> queryTimeHmoVec;
    std::vector<std::shared_ptr<hmm::OckHmmHMObject>> queryTokenIdHmoVec;
    auto ret = GenerateQueryAttr(queryBatch, queryCond, queryTimeHmoVec, queryTokenIdHmoVec);
    OCK_CHECK_RETURN_ERRORCODE(ret);

    OckVsaErrorCode errorCode = VSA_SUCCESS;
    auto streamBase = hcps::handler::helper::MakeStream(*handler, errorCode, hcps::OckDevStreamType::AI_CORE);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);

    uint64_t maskLen = utils::SafeDivUp(GetFeatureNum(), __CHAR_BIT__);
    if (queryCond.shareAttrFilter) {
        auto singleMaskHmo =
            hcps::handler::helper::MakeDeviceHmo(*handler, maskLenAligned * sizeof(uint8_t), errorCode);
        OCK_CHECK_RETURN_ERRORCODE(errorCode);
        auto extraMaskHmo = hcps::handler::helper::MakeDeviceHmo(*handler, maskLenAligned * sizeof(uint8_t), errorCode);
        OCK_CHECK_RETURN_ERRORCODE(errorCode);
        errorCode = ockSyncUtils->Copy(reinterpret_cast<void *>(extraMaskHmo->Addr()), extraMaskHmo->GetByteSize(),
            queryCond.extraMask, maskLen * sizeof(uint8_t),
            queryCond.extraMaskIsAtDevice ? acladapter::OckMemoryCopyKind::DEVICE_TO_DEVICE :
                                            acladapter::OckMemoryCopyKind::HOST_TO_DEVICE);
        OCK_CHECK_RETURN_ERRORCODE(errorCode);
        auto extraMaskGroups = PrepareMaskDataWithExtra(queryTimeHmoVec, queryTokenIdHmoVec, singleMaskHmo,
            extraMaskHmo, attrFeatureGroups, maskLen, param->GroupBlockCount());
        hcps::nop::OckDistMaskWithExtraGenOpRun::AddMaskWithExtraOpsSingleBatch(extraMaskGroups, streamBase);
        errorCode = streamBase->WaitExecComplete();
        OCK_CHECK_RETURN_ERRORCODE(errorCode);
        for (uint32_t i = 0; i < queryBatch; ++i) {
            errorCode = handler->HmmMgrPtr()->CopyHMO(*maskHmo, maskLen * sizeof(uint8_t) * i, *singleMaskHmo, 0,
                maskLen * sizeof(uint8_t));
            OCK_CHECK_RETURN_ERRORCODE(errorCode);
        }
        return VSA_SUCCESS;
    }
    auto extraMaskHmo = hcps::handler::helper::MakeDeviceHmo(*handler, maskHmo->GetByteSize(), errorCode);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    errorCode = ockSyncUtils->Copy(reinterpret_cast<void *>(extraMaskHmo->Addr()), extraMaskHmo->GetByteSize(),
        queryCond.extraMask, maskLen * queryBatch * sizeof(uint8_t),
        queryCond.extraMaskIsAtDevice ? acladapter::OckMemoryCopyKind::DEVICE_TO_DEVICE :
                                        acladapter::OckMemoryCopyKind::HOST_TO_DEVICE);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    auto extraMaskGroups = PrepareMaskDataWithExtra(queryTimeHmoVec, queryTokenIdHmoVec, maskHmo, extraMaskHmo,
        attrFeatureGroups, maskLen, param->GroupBlockCount());
    errorCode = hcps::nop::OckDistMaskWithExtraGenOpRun::AddMaskWithExtraOpsMultiBatches(extraMaskGroups, streamBase);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);

    return VSA_SUCCESS;
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
uint64_t OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::GetFeatureNum(void) const
{
    if (blockGroups.empty()) {
        return 0;
    }
    return (blockGroups.size() - 1) * param->GroupRowCount() + blockGroups.back()->rowCount;
}

template <typename DataTemp, uint64_t DimSizeTemp, uint64_t NormTypeByteSize, typename KeyTraitTemp>
uint64_t OckVsaAnnNpuIndex<DataTemp, DimSizeTemp, NormTypeByteSize, KeyTraitTemp>::MaxFeatureRowCount(void) const
{
    return param->MaxFeatureRowCount();
}
} // namespace npu
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif