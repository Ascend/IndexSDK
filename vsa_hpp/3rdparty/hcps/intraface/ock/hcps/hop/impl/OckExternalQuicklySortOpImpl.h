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


#ifndef OCK_HCPS_PIER_EXTERNAL_QUICKLY_SORT_OP_IMPL_H
#define OCK_HCPS_PIER_EXTERNAL_QUICKLY_SORT_OP_IMPL_H
#include <cstdint>
#include <memory>
#include <algorithm>
#include <vector>
#include "ock/hcps/stream/OckHeteroOperatorBase.h"
#include "ock/hcps/stream/OckHeteroStreamBase.h"
#include "ock/hcps/algo/OckQuicklyExternSortSpliter.h"
#include "ock/hcps/algo/OckQuicklyExternMergeSpliter.h"
#include "ock/hcps/hop/OckQuicklySortOp.h"
namespace ock {
namespace hcps {
namespace hop {
namespace ext_quick_sort_impl {
template <typename _Tp, typename _CompareT>
std::shared_ptr<OckHeteroOperatorGroup> CreateQuicklySortOps(
    algo::spliter::VectorSortSpliterResult<_Tp> &spliterResult, const _CompareT &compare)
{
    auto sortGrp = std::make_shared<OckHeteroOperatorGroup>();
    for (auto &segment : spliterResult.sortSeg) {
        sortGrp->push_back(MakeOckQuicklySortOp(segment.begin, segment.end, compare));
    }
    return sortGrp;
}
template <typename _Tp, typename _CompareT>
void ExternalMerge(OckHeteroStreamBase &stream, std::vector<algo::spliter::VectorExternMergeSegment<_Tp>> &mergeSeg,
    const _CompareT &compare, uint64_t splitThreshold)
{
    auto troupe = OckHeteroOperatorTroupe();
    for (auto &segment : mergeSeg) {
        troupe.push_back(MakeOckMergeSortOpList(utils::MakeContainerInfo(segment.aBegin, segment.aEnd),
            utils::MakeContainerInfo(segment.bBegin, segment.bEnd),
            utils::MakeContainerInfo(segment.result,
                segment.result +
                    (std::distance(segment.aBegin, segment.aEnd) + std::distance(segment.bBegin, segment.bEnd))),
            compare,
            splitThreshold));
    }
    stream.AddOps(troupe);
    stream.WaitExecComplete();
}
template <typename _Tp, typename _CompareT>
void ExternalMergeQueue(OckHeteroStreamBase &stream,
    std::vector<std::vector<algo::spliter::VectorExternMergeSegment<_Tp>>> &mergeSeg, const _CompareT &compare,
    uint64_t splitThreshold)
{
    for (auto &segment : mergeSeg) {
        ExternalMerge(stream, segment, compare, splitThreshold);
    }
}
}  // namespace ext_quick_sort_impl
template <typename _Tp, typename _CompareT>
void ExternalQuicklySort(OckHeteroStreamBase &stream, std::vector<_Tp> &inputDatas, std::vector<_Tp> &outputDatas,
    const _CompareT &compare, uint64_t splitThreshold)
{
    auto spliterResult = algo::CalcQuicklyExternSortSpliterSegments(inputDatas, outputDatas, splitThreshold);
    auto ops = ext_quick_sort_impl::CreateQuicklySortOps(*spliterResult, compare);
    stream.AddOps(*ops);
    stream.WaitExecComplete();

    ext_quick_sort_impl::ExternalMergeQueue(stream, spliterResult->mergeSeg, compare, splitThreshold);

    if (spliterResult->isSwap) {
        outputDatas.swap(inputDatas);
    }
}
template <typename _Tp, typename _CompareT>
void ExternalQuicklyMerge(OckHeteroStreamBase &stream, std::vector<std::vector<_Tp>> &sortedGroups,
    std::vector<_Tp> &outSortedData, std::vector<_Tp> &tmpData, const _CompareT &compare, uint64_t splitThreshold)
{
    auto spliterResult = algo::CalcQuicklyExternMergeSpliterSegments(sortedGroups, outSortedData, tmpData);

    ext_quick_sort_impl::ExternalMergeQueue(stream, spliterResult->mergeSeg, compare, splitThreshold);

    if (spliterResult->isSwap) {
        outSortedData.swap(tmpData);
    }
}
}  // namespace hop
}  // namespace hcps
}  // namespace ock
#endif