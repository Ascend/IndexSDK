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


#ifndef OCK_HCPS_PIER_QUICKLY_SORT_IMPL_OP_H
#define OCK_HCPS_PIER_QUICKLY_SORT_IMPL_OP_H
#include "ock/utils/OckSafeUtils.h"
#include "ock/log/OckHcpsLogger.h"
#include "ock/hcps/stream/OckHeteroOperatorBase.h"
#include "ock/hcps/hop/OckInplaceMergeSortOp.h"
#include "ock/hcps/hop/OckMergeSortOp.h"
#include "ock/utils/OckTypeTraits.h"
namespace ock {
namespace hcps {
namespace hop {
template <typename _Iterator, typename _Compare>
OckQuicklySortOp<_Iterator, _Compare>::OckQuicklySortOp(_Iterator beginPos, _Iterator endPos,
    const _Compare &compareFunc)
    : dataInfo(beginPos, endPos), compare(compareFunc)
{}

template <typename _Iterator, typename _Compare>
hmm::OckHmmErrorCode OckQuicklySortOp<_Iterator, _Compare>::Run(OckHeteroStreamContext &context)
{
    std::sort(dataInfo.begin, dataInfo.end, compare);
    return hmm::HMM_SUCCESS;
}

template <typename _Iterator, typename _Compare>
std::shared_ptr<OckHeteroOperatorBase> MakeOckQuicklySortOp(_Iterator begin, _Iterator end, const _Compare &compare)
{
    return std::make_shared<OckQuicklySortOp<_Iterator, _Compare>>(begin, end, compare);
}

namespace quick_sort_impl {
template <typename _Iterator, typename _Compare>
void SplitIntoMultiQuicklySortOp(_Iterator begin, _Iterator end, const _Compare &compare,
    std::vector<utils::OckContainerInfo<_Iterator>> &outDatas, uint64_t splitThreshold)
{
    uint64_t realThreshold = std::distance(begin, end) / utils::SafeDivUp(std::distance(begin, end), splitThreshold);
    for (auto iterBegin = begin; iterBegin < end; iterBegin += realThreshold) {
        auto iterEnd = iterBegin + std::min(realThreshold, (uint64_t)std::distance(iterBegin, end));
        outDatas.push_back(utils::MakeContainerInfo(iterBegin, iterEnd));
    }
}
template <typename _Iterator, typename _Compare>
void GenMergeSegmentOpByInterval(std::vector<utils::OckContainerInfo<_Iterator>> &segments,
    OckHeteroOperatorGroup &outDatas, const _Compare &compare, uint64_t mergeInterval, uint64_t splitThreshold)
{
    for (uint64_t beginPos = 0; beginPos < segments.size(); beginPos += mergeInterval) {
        auto begin = segments[beginPos].begin;
        auto middle = segments[beginPos + mergeInterval / 2ULL].begin;
        uint64_t endPos = beginPos + mergeInterval - 1ULL;
        if (endPos + 1ULL >= segments.size()) {
            endPos = segments.size() - 1ULL;
        }
        auto end = segments[endPos].end;
        outDatas.push_back(MakeOckInplaceMergeSortOp(begin, middle, end, compare));
    }
}
template <typename _Iterator, typename _Compare>
uint64_t GenMergeOpIntervalByInterval(std::vector<utils::OckContainerInfo<_Iterator>> &segments,
    OckHeteroOperatorGroupQueue &outDatas, const _Compare &compare, uint64_t splitThreshold)
{
    uint64_t mergeInterval = 0ULL;
    for (mergeInterval = 2ULL; mergeInterval < segments.size(); mergeInterval *= 2ULL) {
        auto group = std::make_shared<OckHeteroOperatorGroup>();
        GenMergeSegmentOpByInterval(segments, *group, compare, mergeInterval, splitThreshold);
        outDatas.push(group);
    }
    return mergeInterval;
}
template <typename _Iterator, typename _Compare>
void BuildParallelQuicklySortOp(std::vector<utils::OckContainerInfo<_Iterator>> &segments,
    OckHeteroOperatorGroupQueue &outDatas, const _Compare &compare)
{
    auto group = std::make_shared<OckHeteroOperatorGroup>();
    for (auto &segment : segments) {
        group->push_back(MakeOckQuicklySortOp(segment.begin, segment.end, compare));
    }
    outDatas.push(group);
}
template <typename _Iterator, typename _Compare>
void CalcMergeMultiSegmentOp(std::vector<utils::OckContainerInfo<_Iterator>> &segments,
    OckHeteroOperatorGroupQueue &outDatas, const _Compare &compare, uint64_t splitThreshold)
{
    if (segments.empty()) {
        return;
    }
    BuildParallelQuicklySortOp(segments, outDatas, compare);
    if (segments.size() == 1ULL) {
        return;
    }
    uint64_t mergeInterval = GenMergeOpIntervalByInterval(segments, outDatas, compare, splitThreshold);

    auto begin = segments.front().begin;
    auto middle = segments[mergeInterval / 2ULL].begin;
    auto end = segments.back().end;
    outDatas.push(OckHeteroOperatorBase::CreateGroup(MakeOckInplaceMergeSortOp(begin, middle, end, compare)));
}
template <typename _Iterator, typename _Compare>
inline void SplitQuicklySortOpQueue(_Iterator begin, _Iterator end, const _Compare &compare,
    OckHeteroOperatorGroupQueue &outDatas, uint64_t splitThreshold)
{
    std::vector<utils::OckContainerInfo<_Iterator>> parallelSortSegments;
    SplitIntoMultiQuicklySortOp(begin, end, compare, parallelSortSegments, splitThreshold);
    CalcMergeMultiSegmentOp<_Iterator>(parallelSortSegments, outDatas, compare, splitThreshold);
}
}  // namespace quick_sort_impl
}  // namespace hop
}  // namespace hcps
}  // namespace ock
#endif