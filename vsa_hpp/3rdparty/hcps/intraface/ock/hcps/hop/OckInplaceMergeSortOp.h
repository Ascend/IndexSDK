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


#ifndef OCK_HCPS_PIER_INPLACE_MERGE_SORT_OP_H
#define OCK_HCPS_PIER_INPLACE_MERGE_SORT_OP_H
#include <cstdint>
#include <memory>
#include <algorithm>
#include <type_traits>
#include "ock/utils/OckContainerInfo.h"
#include "ock/hmm/mgr/OckHmmErrorCode.h"
#include "ock/hcps/stream/OckHeteroOperatorBase.h"
namespace ock {
namespace hcps {
namespace hop {

/*
@brief 约束：类同与std::inplace_merge
*/
template <typename _Iterator, typename _Compare>
class OckInplaceMergeSortOp : public OckHeteroOperatorGen<acladapter::OckTaskResourceType::HOST_CPU> {
public:
    virtual ~OckInplaceMergeSortOp() noexcept = default;
    OckInplaceMergeSortOp(_Iterator beginPos, _Iterator middlePos, _Iterator endPos, const _Compare &compareFunc);

    hmm::OckHmmErrorCode Run(OckHeteroStreamContext &context) override;

    _Iterator begin;
    _Iterator middle;
    _Iterator end;
    _Compare compare;
};

template <typename _Iterator, typename _Compare>
std::shared_ptr<OckHeteroOperatorBase> MakeOckInplaceMergeSortOp(
    _Iterator begin, _Iterator middle, _Iterator end, const _Compare &compare);

}  // namespace hop
}  // namespace hcps
}  // namespace ock
#include "ock/hcps/hop/impl/OckInplaceMergeSortOpImpl.h"
#endif