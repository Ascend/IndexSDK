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


#ifndef OCK_HCPS_OCK_MERGE_SORT_IMPL_OP_H
#define OCK_HCPS_OCK_MERGE_SORT_IMPL_OP_H
#include "ock/log/OckHcpsLogger.h"
namespace ock {
namespace hcps {
namespace hop {
template <typename _ContainerInfoA, typename _ContainerInfoB, typename _ContainerInfoOut, typename _Compare>
OckMergeSortOp<_ContainerInfoA, _ContainerInfoB, _ContainerInfoOut, _Compare>::OckMergeSortOp(
    _ContainerInfoA compareDataA, _ContainerInfoB compareDataB, _ContainerInfoOut outResult,
    const _Compare &compareFunc)
    : dataA(compareDataA), dataB(compareDataB), result(outResult), compare(compareFunc)
{}

template <typename _ContainerInfoA, typename _ContainerInfoB, typename _ContainerInfoOut, typename _Compare>
hmm::OckHmmErrorCode OckMergeSortOp<_ContainerInfoA, _ContainerInfoB, _ContainerInfoOut, _Compare>::Run(
    OckHeteroStreamContext &context)
{
    if (dataA.Size() + dataB.Size() > result.Size()) {
        return hmm::HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE;
    }
    std::merge(dataA.begin, dataA.end, dataB.begin, dataB.end, result.begin, compare);
    return hmm::HMM_SUCCESS;
}

namespace merge_sort_impl {
template <typename _ContainerInfoA, typename _ContainerInfoB, typename _ContainerInfoOut, typename _Compare,
    typename _InputIteratorA = typename _ContainerInfoA::IteratorT,
    typename _InputIteratorB = typename _ContainerInfoB::IteratorT>
inline void SplitMergeTaskToMultiSegment(_ContainerInfoA dataA, _ContainerInfoB dataB, _ContainerInfoOut result,
    OckHeteroOperatorGroup &outOps, const _Compare &compare, uint64_t splitThreshold)
{
    if (dataA.Size() + dataB.Size() < splitThreshold || dataA.Empty() || dataB.Empty()) {
        outOps.push_back(MakeOckMergeSortOp(dataA, dataB, result, compare));
    } else {
        auto midA = dataA.begin;
        auto midB = dataB.begin;
        if (dataA.Size() > dataB.Size()) {
            midA = dataA.Middle();
            midB = std::lower_bound(dataB.begin, dataB.end, *midA, compare);
        } else {
            midB = dataB.Middle();
            midA = std::lower_bound(dataA.begin, dataA.end, *midB, compare);
        }
        SplitMergeTaskToMultiSegment(utils::MakeContainerInfo(dataA.begin, midA),
            utils::MakeContainerInfo(dataB.begin, midB),
            result,
            outOps,
            compare,
            splitThreshold);
        auto resultBegin = result.begin;
        std::advance(resultBegin, std::distance(dataA.begin, midA) + std::distance(dataB.begin, midB));
        SplitMergeTaskToMultiSegment(utils::MakeContainerInfo(midA, dataA.end),
            utils::MakeContainerInfo(midB, dataB.end),
            utils::MakeContainerInfo(resultBegin, result.end),
            outOps,
            compare,
            splitThreshold);
    }
}
}  // namespace merge_sort_impl
template <typename _ContainerInfoA, typename _ContainerInfoB, typename _ContainerInfoOut, typename _Compare>
inline std::shared_ptr<OckHeteroOperatorBase> MakeOckMergeSortOp(_ContainerInfoA dataA, _ContainerInfoB dataB,
    _ContainerInfoOut result, const _Compare &compare)
{
    return std::make_shared<OckMergeSortOp<_ContainerInfoA, _ContainerInfoB, _ContainerInfoOut, _Compare>>(dataA, dataB,
        result, compare);
}
template <typename _ContainerInfoA, typename _ContainerInfoB, typename _ContainerInfoOut, typename _Compare>
inline std::shared_ptr<OckHeteroOperatorGroup> MakeOckMergeSortOpList(_ContainerInfoA dataA, _ContainerInfoB dataB,
    _ContainerInfoOut result, const _Compare &compare, uint64_t splitThreshold)
{
    auto retOps = std::make_shared<OckHeteroOperatorGroup>();
    merge_sort_impl::SplitMergeTaskToMultiSegment(dataA, dataB, result, *retOps, compare, splitThreshold);
    return retOps;
}

}  // namespace hop
}  // namespace hcps
}  // namespace ock
#endif