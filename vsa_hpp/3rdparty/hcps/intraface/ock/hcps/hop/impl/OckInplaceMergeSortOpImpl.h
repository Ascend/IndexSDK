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


#ifndef OCK_HCPS_OCK_INPLACE_MERGE_SORT_IMPL_OP_H
#define OCK_HCPS_OCK_INPLACE_MERGE_SORT_IMPL_OP_H
#include "ock/log/OckHcpsLogger.h"
namespace ock {
namespace hcps {
namespace hop {

template <typename _Iterator, typename _Compare>
OckInplaceMergeSortOp<_Iterator, _Compare>::OckInplaceMergeSortOp(
    _Iterator beginPos, _Iterator middlePos, _Iterator endPos, const _Compare &compareFunc)
    : begin(beginPos), middle(middlePos), end(endPos), compare(compareFunc)
{}

template <typename _Iterator, typename _Compare>
inline hmm::OckHmmErrorCode OckInplaceMergeSortOp<_Iterator, _Compare>::Run(OckHeteroStreamContext &context)
{
    std::inplace_merge(begin, middle, end, compare);
    return hmm::HMM_SUCCESS;
}

template <typename _Iterator, typename _Compare>
inline std::shared_ptr<OckHeteroOperatorBase> MakeOckInplaceMergeSortOp(
    _Iterator begin, _Iterator middle, _Iterator end, const _Compare &compare)
{
    return std::make_shared<OckInplaceMergeSortOp<_Iterator, _Compare>>(begin, middle, end, compare);
}
}  // namespace hop
}  // namespace hcps
}  // namespace ock
#endif