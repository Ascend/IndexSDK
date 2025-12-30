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

#include <map>
#include <chrono>
#include "acl/acl.h"
#include "ock/acladapter/task/OckDftFunOpTask.h"
namespace ock {
namespace acladapter {

OckDftFunOp::OckDftFunOp(
    std::function<hmm::OckHmmErrorCode()> funcOp, OckTaskResourceType taskResourceType)
    : funOp(funcOp), resourceType(taskResourceType)
{}
OckTaskResourceType OckDftFunOp::ResourceType(void) const
{
    return resourceType;
}
bool OckDftFunOp::PreConditionMet(void)
{
    return true;
}
std::shared_ptr<OckDefaultResult> OckDftFunOp::Run(
    OckAsyncTaskContext &context, OckDftAsyncTaskParam &param, OckUserWaitInfoBase &waitInfo)
{
    if (waitInfo.WaitTimeOut()) {
        return std::make_shared<OckDefaultResult>(hmm::HMM_ERROR_WAIT_TIME_OUT);
    }
    return std::make_shared<OckDefaultResult>(funOp());
}
}  // namespace acladapter
}  // namespace ock