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


#ifndef OCK_HCPS_PIER_SPLIT_GROUP_OP_H
#define OCK_HCPS_PIER_SPLIT_GROUP_OP_H
#include <cstdint>
#include <memory>
#include <algorithm>
#include <vector>
#include <type_traits>
#include <iterator>
#include "ock/hmm/mgr/OckHmmErrorCode.h"
#include "ock/hcps/stream/OckHeteroOperatorBase.h"
namespace ock {
namespace hcps {
namespace hop {
template <typename _Iterator, acladapter::OckTaskResourceType resourceType = acladapter::OckTaskResourceType::HOST_CPU>
class OckSplitGroupOp : public OckHeteroOperatorGen<resourceType> {
public:
    using IteratorT = _Iterator;
    virtual ~OckSplitGroupOp() noexcept = default;
    OckSplitGroupOp(_Iterator beginPos, _Iterator endPos,
        std::function<hmm::OckHmmErrorCode(_Iterator begin, _Iterator end)> func);

    hmm::OckHmmErrorCode Run(OckHeteroStreamContext &context) override;

private:
    _Iterator begin;
    _Iterator end;
    std::function<hmm::OckHmmErrorCode(_Iterator, _Iterator end)> opFun;
};
/*
@brief  特别地： _Iterator 支持整数、指针、迭代器
*/
template <typename _Iterator, acladapter::OckTaskResourceType resourceType = acladapter::OckTaskResourceType::HOST_CPU>
std::shared_ptr<OckHeteroOperatorGroup> MakeOckSplitGroupOps(_Iterator begin, _Iterator end, uint64_t stepInterval,
    std::function<hmm::OckHmmErrorCode(_Iterator, _Iterator end)> opFun);
template <typename _Iterator, acladapter::OckTaskResourceType resourceType = acladapter::OckTaskResourceType::HOST_CPU>
std::shared_ptr<OckHeteroOperatorGroup> MakeOckSplitGroupAtmoicOps(_Iterator begin, _Iterator end,
    uint64_t stepInterval, std::function<hmm::OckHmmErrorCode(_Iterator)> opFun);
template <typename _Iterator, acladapter::OckTaskResourceType resourceType = acladapter::OckTaskResourceType::HOST_CPU>
std::shared_ptr<OckHeteroOperatorGroup> MakeOckSplitGroupAtmoicOpsNoReturn(_Iterator begin, _Iterator end,
    uint64_t stepInterval, std::function<void(_Iterator)> opFun);
}  // namespace hop
}  // namespace hcps
}  // namespace ock
#include "ock/hcps/hop/impl/OckSplitGroupOpImpl.h"
#endif