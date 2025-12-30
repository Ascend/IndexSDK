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


#ifndef OCK_HCPS_PIER_QUICKLY_SORT_OP_H
#define OCK_HCPS_PIER_QUICKLY_SORT_OP_H
#include <cstdint>
#include <memory>
#include <algorithm>
#include <vector>
#include <type_traits>
#include "ock/utils/OckContainerInfo.h"
#include "ock/utils/OckCompareUtils.h"
#include "ock/hmm/mgr/OckHmmErrorCode.h"
#include "ock/hcps/stream/OckHeteroOperatorBase.h"
namespace ock {
namespace hcps {
namespace hop {

template <typename _Iterator, typename _Compare>
class OckQuicklySortOp : public OckHeteroOperatorGen<acladapter::OckTaskResourceType::HOST_CPU> {
public:
    using IteratorT = _Iterator;
    virtual ~OckQuicklySortOp() noexcept = default;
    OckQuicklySortOp(_Iterator beginPos, _Iterator endPos, const _Compare &compareFunc);

    hmm::OckHmmErrorCode Run(OckHeteroStreamContext &context) override;

    utils::OckContainerInfo<_Iterator> dataInfo;
    _Compare compare;
};

template <typename _Iterator, typename _Compare>
std::shared_ptr<OckHeteroOperatorBase> MakeOckQuicklySortOp(_Iterator begin, _Iterator end, const _Compare &compare);


}  // namespace hop
}  // namespace hcps
}  // namespace ock
#include "ock/hcps/hop/impl/OckQuicklySortOpImpl.h"
#endif