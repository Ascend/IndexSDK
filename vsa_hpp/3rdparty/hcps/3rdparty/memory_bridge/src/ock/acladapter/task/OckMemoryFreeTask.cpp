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

#include "acl/acl.h"
#include "ock/acladapter/task/OckMemoryFreeTask.h"
namespace ock {
namespace acladapter {

OckMemoryFreeFunOp::OckMemoryFreeFunOp(void)
{}
OckTaskResourceType OckMemoryFreeFunOp::ResourceType(void) const
{
    return OckTaskResourceType::DEVICE_MEMORY_OP | OckTaskResourceType::HOST_MEMORY_OP;
}
bool OckMemoryFreeFunOp::PreConditionMet(void)
{
    return true;
}
std::shared_ptr<OckDefaultResult> OckMemoryFreeFunOp::Run(
    OckAsyncTaskContext &context, OckMemoryFreeParam &param, OckUserWaitInfoBase &waitInfo)
{
    if (waitInfo.WaitTimeOut()) {
        return std::make_shared<OckDefaultResult>(hmm::HMM_ERROR_WAIT_TIME_OUT);
    }
    OCK_HMM_LOG_DEBUG("Free " << param << " " << context);
    if (param.Addr() == nullptr) {
        return std::make_shared<OckDefaultResult>(hmm::HMM_SUCCESS);
    }
    hmm::OckHmmErrorCode errorCode = ACL_SUCCESS;
    if (param.Location() == hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR ||
        param.Location() == hmm::OckHmmHeteroMemoryLocation::DEVICE_HBM) {
        errorCode = aclrtFree(param.Addr());
    } else if (param.Location() == hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY) {
        errorCode = aclrtFreeHost(param.Addr());
    } else {
        errorCode = hmm::HMM_ERROR_INPUT_PARAM_NOT_SUPPORT_SUCH_OP;
    }
    if (errorCode != ACL_SUCCESS) {
        return std::make_shared<OckDefaultResult>(errorCode);
    }
    return std::make_shared<OckDefaultResult>(hmm::HMM_SUCCESS);
}
}  // namespace acladapter
}  // namespace ock