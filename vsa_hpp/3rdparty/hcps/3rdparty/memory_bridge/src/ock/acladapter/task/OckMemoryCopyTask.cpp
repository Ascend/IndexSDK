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
#include "ock/acladapter/task/OckMemoryCopyTask.h"
namespace ock {
namespace acladapter {

OckMemoryCopyFunOp::OckMemoryCopyFunOp(void)
{}
bool OckMemoryCopyFunOp::PreConditionMet(void)
{
    return true;
}
OckTaskResourceType OckMemoryCopyFunOp::ResourceType(void) const
{
    return OckTaskResourceType::MEMORY_TRANSFER;
}
std::shared_ptr<OckDefaultResult> OckMemoryCopyFunOp::Run(
    OckAsyncTaskContext &context, OckMemoryCopyParam &param, OckUserWaitInfoBase &waitInfo)
{
    if (waitInfo.WaitTimeOut()) {
        return std::make_shared<OckDefaultResult>(hmm::HMM_ERROR_WAIT_TIME_OUT);
    }
    OCK_HMM_LOG_DEBUG("Copy " << param << " " << context);
    const std::map<OckMemoryCopyKind, aclrtMemcpyKind> kindMap{
        {OckMemoryCopyKind::HOST_TO_HOST, ACL_MEMCPY_HOST_TO_HOST},
        {OckMemoryCopyKind::HOST_TO_DEVICE, ACL_MEMCPY_HOST_TO_DEVICE},
        {OckMemoryCopyKind::DEVICE_TO_HOST, ACL_MEMCPY_DEVICE_TO_HOST},
        {OckMemoryCopyKind::DEVICE_TO_DEVICE, ACL_MEMCPY_DEVICE_TO_DEVICE},
    };

    auto iter = kindMap.find(param.Kind());
    if (iter == kindMap.end()) {
        return std::make_shared<OckDefaultResult>(hmm::HMM_ERROR_INPUT_PARAM_NOT_SUPPORT_SUCH_OP);
    }
    auto startTime = std::chrono::steady_clock::now();
    aclrtMemcpyKind kind = iter->second;
    hmm::OckHmmErrorCode retCode =
        aclrtMemcpy(param.DestAddr(), param.DestMax(), param.SrcAddr(), param.SrcCount(), kind);
    if (retCode != ACL_SUCCESS) {
        OCK_HMM_LOG_ERROR("aclrtMemcpy failed. param=" << param << " " << context);
    } else {
        context.StatisticsMgr()->AddTrafficInfo(
            std::make_shared<OckHmmTrafficInfo>(context.GetDeviceId(), param.SrcCount(), param.Kind(), startTime));
    }
    return std::make_shared<OckDefaultResult>(retCode);
}
}  // namespace acladapter
}  // namespace ock