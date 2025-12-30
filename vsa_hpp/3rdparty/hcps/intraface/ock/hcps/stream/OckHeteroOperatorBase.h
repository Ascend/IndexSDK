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

#ifndef OCK_HCPS_OCK_HETERO_OPERATOR_BASE_H
#define OCK_HCPS_OCK_HETERO_OPERATOR_BASE_H
#include <vector>
#include <queue>
#include <functional>
#include "ock/acladapter/data/OckTaskResourceType.h"
#include "ock/hcps/stream/OckHeteroStreamContext.h"
#include "ock/hmm/mgr/OckHmmHMObject.h"
namespace ock {
namespace hcps {

class OckHeteroOperatorBase;
/*
一个算子组(Group)包含多个算子(Op)。这些OP可以并发执行
一个算子纵队(Queue)包含多个算子(Op)， 这些组(Op)间只能串行行。
一个算子纵队(GroupQueue)包含多个组(Group)， 这些组(Group)间只能串行行。
一个算子团(Troupe)包含多组(Group)算子, 这些组(Group)可以并行执行
*/
using OckHeteroOperatorGroup = std::vector<std::shared_ptr<OckHeteroOperatorBase>>;
using OckHeteroOperatorQueue = std::queue<std::shared_ptr<OckHeteroOperatorBase>>;
using OckHeteroOperatorGroupQueue = std::queue<std::shared_ptr<OckHeteroOperatorGroup>>;
using OckHeteroOperatorTroupe = std::vector<std::shared_ptr<OckHeteroOperatorGroup>>;

class OckHeteroOperatorBase {
public:
    virtual ~OckHeteroOperatorBase() noexcept = default;
    virtual acladapter::OckTaskResourceType ResourceType(void) const = 0;
    virtual hmm::OckHmmErrorCode Run(OckHeteroStreamContext &context) = 0;

    static std::shared_ptr<OckHeteroOperatorGroup> CreateGroup(std::shared_ptr<OckHeteroOperatorBase> op);
    static std::shared_ptr<OckHeteroOperatorGroup> CreateGroup(
        std::shared_ptr<OckHeteroOperatorBase> op1, std::shared_ptr<OckHeteroOperatorBase> op2);
    static std::shared_ptr<OckHeteroOperatorGroup> CreateGroup(std::shared_ptr<OckHeteroOperatorBase> op1,
        std::shared_ptr<OckHeteroOperatorBase> op2, std::shared_ptr<OckHeteroOperatorBase> op3);
};

template <acladapter::OckTaskResourceType RTemp>
class OckHeteroOperatorGen : public OckHeteroOperatorBase {
public:
    virtual ~OckHeteroOperatorGen() noexcept = default;
    acladapter::OckTaskResourceType ResourceType(void) const override
    {
        return RTemp;
    }
    hmm::OckHmmErrorCode Enable(void)
    {
        return hmm::HMM_SUCCESS;
    }
    hmm::OckHmmErrorCode Disable(void)
    {
        return hmm::HMM_SUCCESS;
    }
    hmm::OckHmmErrorCode Run(OckHeteroStreamContext &context) override
    {
        return hmm::HMM_SUCCESS;
    }
};

template <acladapter::OckTaskResourceType RTemp>
class OckSimpleHeteroOperator : public OckHeteroOperatorGen<RTemp> {
public:
    virtual ~OckSimpleHeteroOperator() noexcept = default;
    OckSimpleHeteroOperator(std::function<hmm::OckHmmErrorCode(OckHeteroStreamContext &)> func) : opFun(func)
    {}
    hmm::OckHmmErrorCode Run(OckHeteroStreamContext &context) override
    {
        return opFun(context);
    }
    static std::shared_ptr<OckHeteroOperatorBase> Create(
        std::function<hmm::OckHmmErrorCode(OckHeteroStreamContext &)> opFun)
    {
        return std::make_shared<OckSimpleHeteroOperator>(opFun);
    }

private:
    std::function<hmm::OckHmmErrorCode(OckHeteroStreamContext &)> opFun;
};

std::ostream &operator<<(std::ostream &os, const OckHeteroOperatorBase &heteroOp);
}  // namespace hcps
}  // namespace ock
#endif