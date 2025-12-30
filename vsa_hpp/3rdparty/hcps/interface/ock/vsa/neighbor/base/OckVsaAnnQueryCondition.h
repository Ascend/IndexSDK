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


#ifndef OCK_VSA_ANN_INDEX_QUERY_CONDITION_H
#define OCK_VSA_ANN_INDEX_QUERY_CONDITION_H
#include <cstdint>
#include <memory>
#include <securec.h>
#include "ock/vsa/OckVsaErrorCode.h"
#include "ock/utils/OckSafeUtils.h"
#include "ock/hcps/algo/OckTopNQueue.h"
#include "ock/log/OckLogger.h"

namespace ock {
namespace vsa {
namespace neighbor {

template <typename DataTemp, uint64_t DimSizeTemp, typename KeyTraitTemp>
struct OckVsaAnnSingleBatchQueryCondition {
    using DataT = DataTemp;
    using KeyTraitT = KeyTraitTemp;
    using KeyTypeTupleT = typename KeyTraitTemp::KeyTypeTuple;
    OckVsaAnnSingleBatchQueryCondition(uint32_t batchPosition, const DataTemp *queryData, const KeyTraitTemp *filter,
        bool shareFilter, uint32_t topK, const uint8_t *extraMaskData, uint64_t extraMaskLenEachQuery,
        bool extraMaskIsOnDevice, bool whetherFilterTime = true);

    bool UsingMask(void) const; // extraMask == nullptr
    std::vector<DataTemp> BuildQueryFeatureVec(void) const;
    OckVsaAnnSingleBatchQueryCondition<DataTemp, DimSizeTemp, KeyTraitTemp> SelectData(uint64_t startPos) const;

    template <typename _DT, uint64_t _Dim, typename _KT>
    friend std::ostream &operator<<(std::ostream &os, const OckVsaAnnSingleBatchQueryCondition<_DT, _Dim, _KT> &data);

    const uint32_t batchPos;
    const DataTemp *queryFeature;
    const KeyTraitTemp *attrFilter;
    bool shareAttrFilter;
    uint32_t topk;
    const uint8_t *extraMask;
    uint64_t extraMaskLenEachQuery;
    bool extraMaskIsAtDevice;
    bool enableTimeFilter;
};
template <typename DataTemp, uint64_t DimSizeTemp, typename KeyTraitTemp>
struct OckVsaAnnQueryCondition : public OckVsaAnnSingleBatchQueryCondition<DataTemp, DimSizeTemp, KeyTraitTemp> {
    using DataT = DataTemp;
    using KeyTraitT = KeyTraitTemp;
    using KeyTypeTupleT = typename KeyTraitTemp::KeyTypeTuple;

    OckVsaAnnQueryCondition(uint32_t queryCount, const DataTemp *queryData, const KeyTraitTemp *filter,
        bool shareFilter, uint32_t topK, const uint8_t *extraMaskData, uint64_t extraMaskLengthEachQuery,
        bool extraMaskIsOnDevice, bool whetherFilterTime = true);

    uint32_t QueryBatchCount(void) const;
    OckVsaAnnSingleBatchQueryCondition<DataTemp, DimSizeTemp, KeyTraitTemp> QueryCondAt(uint32_t queryBatchPos) const;

    template <typename _DT, uint64_t _Dim, typename _KT>
    friend std::ostream &operator<<(std::ostream &os, const OckVsaAnnQueryCondition<_DT, _Dim, _KT> &data);

    uint32_t queryBatchCount;
};
template <typename DataTemp, uint64_t DimSizeTemp, typename KeyTraitTemp>
OckVsaAnnSingleBatchQueryCondition<DataTemp, DimSizeTemp, KeyTraitTemp>::OckVsaAnnSingleBatchQueryCondition(
    const uint32_t batchPosition, const DataTemp *queryData, const KeyTraitTemp *filter, bool shareFilter,
    uint32_t topK, const uint8_t *extraMaskData, uint64_t extraMaskLengthEachQuery, bool extraMaskIsOnDevice,
    bool whetherFilterTime)
    : batchPos(batchPosition),
      queryFeature(queryData),
      attrFilter(filter),
      shareAttrFilter(shareFilter),
      topk(topK),
      extraMask(extraMaskData),
      extraMaskLenEachQuery(extraMaskLengthEachQuery),
      extraMaskIsAtDevice(extraMaskIsOnDevice),
      enableTimeFilter(whetherFilterTime)
{}
template <typename DataTemp, uint64_t DimSizeTemp, typename KeyTraitTemp>
std::vector<DataTemp> OckVsaAnnSingleBatchQueryCondition<DataTemp, DimSizeTemp, KeyTraitTemp>::BuildQueryFeatureVec(
    void) const
{
    std::vector<DataTemp> ret(DimSizeTemp);
    if (sizeof(DataTemp) * ret.size() != 0 && sizeof(DataTemp) * DimSizeTemp != 0) {
        auto errorcode =
                memcpy_s(ret.data(), sizeof(DataTemp) * ret.size(), queryFeature, sizeof(DataTemp) * DimSizeTemp);
        if (errorcode != EOK) {
            OCK_HMM_LOG_ERROR("memcpy_s failed, the errorCode is " << errorcode);
            return ret;
        }
    }
    return ret;
}
template <typename DataTemp, uint64_t DimSizeTemp, typename KeyTraitTemp>
OckVsaAnnSingleBatchQueryCondition<DataTemp, DimSizeTemp, KeyTraitTemp> OckVsaAnnSingleBatchQueryCondition<DataTemp, DimSizeTemp, KeyTraitTemp>::SelectData(
    uint64_t startPos) const
{
    return OckVsaAnnSingleBatchQueryCondition<DataTemp, DimSizeTemp, KeyTraitTemp>(batchPos,
        queryFeature,
        attrFilter,
        shareAttrFilter,
        topk,
        (extraMask == nullptr) ? extraMask : extraMask + utils::SafeDiv(startPos, __CHAR_BIT__),
        extraMaskLenEachQuery,
        extraMaskIsAtDevice,
        enableTimeFilter);
}
template <typename _DT, uint64_t _Dim, typename _KT>
std::ostream &operator << (std::ostream &os, const OckVsaAnnSingleBatchQueryCondition<_DT, _Dim, _KT> &data)
{
    os << "[batchPos is " << data.batchPos << "]=={'query features size is: " << _Dim << ", shareAttrFilter is: " <<
        data.shareAttrFilter << ", topk is: " << data.topk << ", extraMaskLenEachQuery is: " <<
        data.extraMaskLenEachQuery << ", extraMaskIsAtDevice: " << data.extraMaskIsAtDevice << ", enableTimeFilter: " <<
        data.enableTimeFilter << "}";
    return os;
}

template <typename DataTemp, uint64_t DimSizeTemp, typename KeyTraitTemp>
OckVsaAnnQueryCondition<DataTemp, DimSizeTemp, KeyTraitTemp>::OckVsaAnnQueryCondition(uint32_t queryCount,
    const DataTemp *queryData, const KeyTraitTemp *filter, bool shareFilter, uint32_t topK,
    const uint8_t *extraMaskData, uint64_t extraMaskLengthEachQuery, bool extraMaskIsOnDevice, bool whetherFilterTime)
    : OckVsaAnnSingleBatchQueryCondition<DataTemp, DimSizeTemp, KeyTraitTemp>(0UL, queryData, filter,
    shareFilter, topK, extraMaskData, extraMaskLengthEachQuery, extraMaskIsOnDevice, whetherFilterTime),
      queryBatchCount(queryCount)
{}

template <typename DataTemp, uint64_t DimSizeTemp, typename KeyTraitTemp>
uint32_t OckVsaAnnQueryCondition<DataTemp, DimSizeTemp, KeyTraitTemp>::QueryBatchCount(void) const
{
    return queryBatchCount;
}

template <typename DataTemp, uint64_t DimSizeTemp, typename KeyTraitTemp>
OckVsaAnnSingleBatchQueryCondition<DataTemp, DimSizeTemp, KeyTraitTemp> OckVsaAnnQueryCondition<DataTemp, DimSizeTemp, KeyTraitTemp>::QueryCondAt(
    uint32_t queryBatchPos) const
{
    auto attrFilterCopy = this->attrFilter;
    if (!this->shareAttrFilter) {
        attrFilterCopy = this->attrFilter + queryBatchPos;
    }
    const uint8_t *maskCopy = nullptr;
    if (this->UsingMask()) {
        maskCopy = this->extraMask + queryBatchPos * this->extraMaskLenEachQuery;
    }
    return OckVsaAnnSingleBatchQueryCondition<DataTemp, DimSizeTemp, KeyTraitTemp>(queryBatchPos,
        this->queryFeature + queryBatchPos * DimSizeTemp, attrFilterCopy, this->shareAttrFilter,
        this->topk, maskCopy, this->extraMaskLenEachQuery, this->extraMaskIsAtDevice, this->enableTimeFilter);
}

template <typename DataTemp, uint64_t DimSizeTemp, typename KeyTraitTemp>
bool OckVsaAnnSingleBatchQueryCondition<DataTemp, DimSizeTemp, KeyTraitTemp>::UsingMask(void) const
{
    return !(this->extraMask == nullptr);
}

template <typename _DT, uint64_t _Dim, typename _KT>
std::ostream &operator << (std::ostream &os, const OckVsaAnnQueryCondition<_DT, _Dim, _KT> &data)
{
    os << "{queryBatchCount is: " << data.queryBatchCount << ", 'query features size is: " << _Dim <<
        ", shareAttrFilter is: " << data.shareAttrFilter << ", topk is: " << data.topk <<
        ", extraMaskLenEachQuery is: " << data.extraMaskLenEachQuery << ", extraMaskIsAtDevice: " <<
        data.extraMaskIsAtDevice << ", enableTimeFilter: " << data.enableTimeFilter << "}";
    return os;
}
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif