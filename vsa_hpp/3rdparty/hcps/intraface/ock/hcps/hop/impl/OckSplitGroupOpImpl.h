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


#ifndef OCK_HCPS_PIER_SPLIT_GROUP_OP_IMPL_H
#define OCK_HCPS_PIER_SPLIT_GROUP_OP_IMPL_H
#include <cstdint>
#include <memory>
#include <algorithm>
#include <vector>
#include <type_traits>
#include "ock/utils/OckTypeTraits.h"
#include "ock/hmm/mgr/OckHmmErrorCode.h"
#include "ock/hcps/stream/OckHeteroOperatorBase.h"
#include "ock/hcps/hop/OckSplitGroupOp.h"
namespace ock {
namespace hcps {
namespace hop {

namespace impl {

template <typename _Iterator, acladapter::OckTaskResourceType resourceType, typename _ReturnT>
std::shared_ptr<_ReturnT> MakeOckSplitGroupOpsImpl(_Iterator begin, _Iterator end, uint64_t stepInterval,
    std::function<hmm::OckHmmErrorCode(_Iterator begin, _Iterator end)> opFun)
{
    static_assert(utils::traits::IsIterator<_Iterator>::value || std::is_integral<_Iterator>::value,
        "begin, end must be a iterator type or a integral");
    auto ret = std::make_shared<_ReturnT>();
    while (utils::traits::Distance(begin, end) > (int64_t)stepInterval) {
        ret->push_back(std::make_shared<OckSplitGroupOp<_Iterator, resourceType>>(begin, begin + stepInterval, opFun));
        begin += static_cast<uint32_t>(stepInterval);
    }
    if (begin < end) {
        ret->push_back(std::make_shared<OckSplitGroupOp<_Iterator, resourceType>>(begin, end, opFun));
    }
    return ret;
}
}  // namespace impl
template <typename _Iterator, acladapter::OckTaskResourceType resourceType>
OckSplitGroupOp<_Iterator, resourceType>::OckSplitGroupOp(
    _Iterator beginPos, _Iterator endPos, std::function<hmm::OckHmmErrorCode(_Iterator begin, _Iterator end)> func)
    : begin(beginPos), end(endPos), opFun(func)
{}
template <typename _Iterator, acladapter::OckTaskResourceType resourceType>
hmm::OckHmmErrorCode OckSplitGroupOp<_Iterator, resourceType>::Run(OckHeteroStreamContext &context)
{
    return opFun(begin, end);
}
template <typename _Iterator, acladapter::OckTaskResourceType resourceType>
std::shared_ptr<OckHeteroOperatorGroup> MakeOckSplitGroupOps(_Iterator begin, _Iterator end, uint64_t stepInterval,
    std::function<hmm::OckHmmErrorCode(_Iterator begin, _Iterator end)> opFun)
{
    return impl::MakeOckSplitGroupOpsImpl<_Iterator, resourceType, OckHeteroOperatorGroup>(
        begin, end, stepInterval, opFun);
}
template <typename _Iterator, acladapter::OckTaskResourceType resourceType>
std::shared_ptr<OckHeteroOperatorGroup> MakeOckSplitGroupAtmoicOps(
    _Iterator begin, _Iterator end, uint64_t stepInterval, std::function<hmm::OckHmmErrorCode(_Iterator)> opFun)
{
    return impl::MakeOckSplitGroupOpsImpl<_Iterator, resourceType, OckHeteroOperatorGroup>(
        begin, end, stepInterval, [opFun](_Iterator beginPos, _Iterator endPos) {
            for (auto iter = beginPos; iter != endPos; ++iter) {
                auto ret = opFun(iter);
                if (ret != hmm::HMM_SUCCESS) {
                    return ret;
                }
            }
            return hmm::HMM_SUCCESS;
        });
}
template <typename _Iterator, acladapter::OckTaskResourceType resourceType>
std::shared_ptr<OckHeteroOperatorGroup> MakeOckSplitGroupAtmoicOpsNoReturn(
    _Iterator begin, _Iterator end, uint64_t stepInterval, std::function<void(_Iterator)> opFun)
{
    return impl::MakeOckSplitGroupOpsImpl<_Iterator, resourceType, OckHeteroOperatorGroup>(
        begin, end, stepInterval, [opFun](_Iterator beginPos, _Iterator endPos) {
            for (auto iter = beginPos; iter != endPos; ++iter) {
                opFun(iter);
            }
            return hmm::HMM_SUCCESS;
        });
}
}  // namespace hop
}  // namespace hcps
}  // namespace ock
#endif