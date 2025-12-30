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

#include "ock/utils/OstreamUtils.h"
#include "ock/hcps/stream/OckHeteroOperatorBase.h"
namespace ock {
namespace hcps {

std::ostream &operator<<(std::ostream &os, const OckHeteroOperatorBase &heteroOp)
{
    return os << "{'type:'" << heteroOp.ResourceType() << "}";
}
std::shared_ptr<OckHeteroOperatorGroup> OckHeteroOperatorBase::CreateGroup(std::shared_ptr<OckHeteroOperatorBase> op)
{
    auto group = std::make_shared<OckHeteroOperatorGroup>();
    group->push_back(op);
    return group;
}
std::shared_ptr<OckHeteroOperatorGroup> OckHeteroOperatorBase::CreateGroup(
    std::shared_ptr<OckHeteroOperatorBase> op1, std::shared_ptr<OckHeteroOperatorBase> op2)
{
    auto group = std::make_shared<OckHeteroOperatorGroup>();
    group->push_back(op1);
    group->push_back(op2);
    return group;
}
std::shared_ptr<OckHeteroOperatorGroup> OckHeteroOperatorBase::CreateGroup(std::shared_ptr<OckHeteroOperatorBase> op1,
    std::shared_ptr<OckHeteroOperatorBase> op2, std::shared_ptr<OckHeteroOperatorBase> op3)
{
    auto group = std::make_shared<OckHeteroOperatorGroup>();
    group->push_back(op1);
    group->push_back(op2);
    group->push_back(op3);
    return group;
}
}  // namespace hcps
}  // namespace ock