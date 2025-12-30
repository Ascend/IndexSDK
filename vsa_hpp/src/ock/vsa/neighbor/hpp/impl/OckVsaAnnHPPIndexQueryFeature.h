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


#ifndef OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_ANN_INDEX_QUERY_FEATURE_H
#define OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_ANN_INDEX_QUERY_FEATURE_H
#include "ock/vsa/neighbor/hpp/impl/OckVsaAnnHPPIndexSystem.h"
#include "ock/hcps/algo/OckTopNQueue.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
namespace impl {
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait>
std::shared_ptr<hcps::OckHeteroOperatorBase> CreateNpuRawDataSearch(
    npu::OckVsaAnnNpuIndex<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait> &npuIndex,
    const OckVsaAnnSingleBatchQueryCondition<DataT, DimSizeT, KeyTrait> &queryCond, OckFloatTopNQueue &outResult,
    OckVsaHPPSearchStaticInfo &stInfo)
{
    return hcps::OckSimpleHeteroOperator<acladapter::OckTaskResourceType::OP_TASK>::Create(
        [&npuIndex, &queryCond, &outResult, &stInfo](hcps::OckHeteroStreamContext &) {
            auto startSearchTime = std::chrono::steady_clock::now();
            auto errorCode = npuIndex.Search(queryCond, outResult);
            stInfo.npuSearchTime = ElapsedMicroSeconds(startSearchTime);
            return errorCode;
        });
}
} // namespace impl
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait>
OckVsaErrorCode OckVsaAnnHPPIndex<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>::Search(
    const OckVsaAnnQueryCondition<DataT, DimSizeT, KeyTrait> &queryCond,
    OckVsaAnnQueryResult<DataT, KeyTrait> &outResult)
{
    if (queryCond.queryFeature == nullptr || queryCond.attrFilter == nullptr ||
        queryCond.queryBatchCount > MAX_SEARCH_BATCH_SIZE || queryCond.topk > MAX_SEARCH_TOPK || queryCond.topk == 0) {
        OCK_VSA_HPP_LOG_ERROR("Input OckVsaAnnQueryCondition cannot have nullptr, batchSize cannot exceed " <<
            MAX_SEARCH_BATCH_SIZE << ", topK cannot exceed " << MAX_SEARCH_TOPK << " and cannot be smaller than 0.");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    if (outResult.labels == nullptr || outResult.distances == nullptr || outResult.validNums == nullptr ||
        outResult.queryCount != queryCond.queryBatchCount || outResult.topk != queryCond.topk) {
        OCK_VSA_HPP_LOG_ERROR("Input OckVsaAnnQueryResult cannot have nullptr, batchSize should be equal to " <<
            queryCond.queryBatchCount << ", topK should be equal to " << queryCond.topk);
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    if (queryCond.extraMask != nullptr &&
        queryCond.extraMaskLenEachQuery != utils::SafeDivUp(GetFeatureNum(), __CHAR_BIT__)) {
        OCK_VSA_HPP_LOG_ERROR("queryCond.extraMaskLenEachQuery should be equal to " <<
            utils::SafeDivUp(GetFeatureNum(), __CHAR_BIT__));
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }

    if (GetFeatureNum() == 0) {
        OCK_VSA_HPP_LOG_ERROR("Base is empty");
        return VSA_ERROR_EMPTY_BASE;
    }

    if (hppKernel->GroupCount() == 0UL) {
        return npuIndex->Search(queryCond, outResult);
    }
    for (uint32_t batch = 0; batch < queryCond.queryBatchCount; ++batch) {
        OckFloatTopNQueue batchResult(queryCond.topk);
        auto tmpRet = this->SearchSingle(queryCond.QueryCondAt(batch), batchResult);
        if (tmpRet != hmm::HMM_SUCCESS) {
            return tmpRet;
        }
        auto node = batchResult.PopAll();
        std::reverse(node->begin(), node->end());
        outResult.AddResult(batch, *node);
    }
    return hmm::HMM_SUCCESS;
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait>
OckVsaErrorCode OckVsaAnnHPPIndex<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>::SearchSingle(
    const OckVsaAnnSingleBatchQueryCondition<DataT, DimSizeT, KeyTrait> &queryCond, OckFloatTopNQueue &outResult) const
{
    auto startSearchTime = std::chrono::steady_clock::now();
    OckVsaHPPSearchStaticInfo stInfo;
    OckVsaErrorCode errorCode = hmm::HMM_SUCCESS;
    // 创建纯NPU上的数据查询条件
    auto npuQueryCond = queryCond.SelectData((uint64_t)hppKernel->GroupCount() * (uint64_t)param->GroupRowCount());
    // 创建纯NPU上的数据查询算子
    auto rawSearchOp = impl::CreateNpuRawDataSearch<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>(*npuIndex,
        npuQueryCond, outResult, stInfo);

    auto startGeneraterMaskTime = std::chrono::steady_clock::now();
    std::shared_ptr<hmm::OckHmmHMObject> maskResult = hcps::handler::helper::MakeDeviceHmo(handler->HmmMgr(),
        utils::SafeDivUp((uint64_t)param->GroupRowCount() * (uint64_t)hppKernel->GroupCount(), __CHAR_BIT__),
        errorCode);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);

    errorCode = npuIndex->GetMaskResult(queryCond, hppKernel->GetAllFeatureAttrs(), param->GroupRowCount(), maskResult);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    stInfo.maskGenerateTime = ElapsedMicroSeconds(startGeneraterMaskTime);

    std::shared_ptr<hmm::OckHmmHMObject> hostMaskResult =
        hcps::handler::helper::CopyToHostHmo(*handler, maskResult, errorCode);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);

    auto maskHmoVec =
        hmm::OckHmmHMObject::CreateSubHmoList(hostMaskResult, utils::SafeDivUp(param->GroupRowCount(), __CHAR_BIT__));
    // 生成Host数据的TopN
    errorCode = hppKernel->Search(queryCond, *maskHmoVec, rawSearchOp, outResult, stInfo);
    stInfo.searchTime = ElapsedMicroSeconds(startSearchTime);
    OCK_VSA_HPP_LOG_INFO("StaticInfo " << stInfo);
    return errorCode;
}

template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait>
uint64_t OckVsaAnnHPPIndex<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>::GetFeatureNum(void) const
{
    return npuIndex->GetFeatureNum() + hppKernel->ValidRowCount();
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait>
uint64_t OckVsaAnnHPPIndex<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>::MaxFeatureRowCount(void) const
{
    return param->MaxFeatureRowCount();
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait>
OckVsaErrorCode OckVsaAnnHPPIndex<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>::GetFeatureByLabel(uint64_t count,
    const int64_t *labels, DataT *features) const
{
    if (count > MAX_GET_NUMBER || labels == nullptr || features == nullptr) {
        OCK_VSA_HPP_LOG_ERROR("Input nullptr or count exceeds threshold[" << MAX_GET_NUMBER << "].");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    OckVsaErrorCode errorCode = hmm::HMM_SUCCESS;
    for (uint64_t i = 0; i < count; ++i) {
        if (npuIndex->GetFeatureByLabel(1ULL, labels + i, features + i * DimSizeT) != hmm::HMM_SUCCESS) {
            if (hppKernel->GetFeatureByLabel(static_cast<uint64_t>(*(labels + i)), features + i * DimSizeT)) {
                OCK_VSA_HPP_LOG_ERROR("Can not find feature by label: " << *(labels + i) << "(" <<
                    static_cast<uint64_t>(*(labels + i)) << ")");
                return VSA_ERROR_INVALID_OUTTER_LABEL;
            }
        }
    }
    return errorCode;
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait>
OckVsaErrorCode OckVsaAnnHPPIndex<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>::GetFeatureAttrByLabel(uint64_t count,
    const int64_t *labels, KeyTypeTupleT *attributes) const
{
    if (count > MAX_GET_NUMBER || labels == nullptr || attributes == nullptr) {
        OCK_VSA_HPP_LOG_ERROR("Input nullptr or count exceeds threshold[" << MAX_GET_NUMBER << "].");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    OckVsaErrorCode errorCode = hmm::HMM_SUCCESS;
    for (uint64_t i = 0; i < count; ++i) {
        if (npuIndex->GetFeatureAttrByLabel(1ULL, labels + i, attributes + i) != hmm::HMM_SUCCESS) {
            if (hppKernel->GetFeatureAttrByLabel(static_cast<uint64_t>(*(labels + i)), attributes + i)) {
                OCK_VSA_HPP_LOG_ERROR("Can not find attr by label: " << *(labels + i) << "(" <<
                    static_cast<uint64_t>(*(labels + i)) << ")");
                return VSA_ERROR_INVALID_OUTTER_LABEL;
            }
        }
    }
    return errorCode;
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait>
uintptr_t OckVsaAnnHPPIndex<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>::GetCustomAttrByBlockId(uint32_t blockId,
    OckVsaErrorCode &errorCode) const
{
    if (blockId >= GetCustomAttrBlockCount()) {
        OCK_VSA_HPP_LOG_ERROR("Input nullptr or blockId exceeds threshold[" << GetCustomAttrBlockCount() << "]");
        errorCode = VSA_ERROR_INVALID_INPUT_PARAM;
        return 0UL;
    }
    if (param->ExtKeyAttrBlockSize() == 0UL || param->GroupRowCount() % param->ExtKeyAttrBlockSize() != 0) {
        OCK_VSA_HPP_LOG_ERROR("param error." << *param);
        errorCode = VSA_ERROR_INVALID_INPUT_PARAM;
        return 0UL;
    }

    if (param->ExtKeyAttrByteSize() == 0UL) {
        errorCode = VSA_ERROR_INVALID_INPUT_PARAM;
        return 0UL;
    }
    uint64_t grpId = ((uint64_t)blockId * (uint64_t)param->ExtKeyAttrBlockSize()) / param->GroupRowCount();
    if (grpId < hppKernel->GroupCount()) {
        return hppKernel->GetCustomAttrByBlockId(blockId, errorCode);
    } else {
        return npuIndex->GetCustomAttrByBlockId(blockId -
            hppKernel->GroupCount() * param->GroupRowCount() / param->ExtKeyAttrBlockSize(), errorCode);
    }
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait>
uint32_t OckVsaAnnHPPIndex<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>::GetCustomAttrBlockCount(void) const
{
    return hppKernel->GetCustomAttrBlockCount() + npuIndex->GetCustomAttrBlockCount();
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait>
std::shared_ptr<OckVsaHPPKernelExt<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>> OckVsaAnnHPPIndex<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>::Kernel(void)
{
    return hppKernel;
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait>
std::shared_ptr<hcps::handler::OckHeteroHandler> OckVsaAnnHPPIndex<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>::Handler(void)
{
    return handler;
}
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif