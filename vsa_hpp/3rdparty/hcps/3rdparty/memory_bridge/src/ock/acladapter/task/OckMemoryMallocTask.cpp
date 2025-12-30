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
#include "ock/acladapter/task/OckMemoryMallocTask.h"
namespace ock {
namespace acladapter {
OckMemoryMallocFunOp::OckMemoryMallocFunOp(void)
{}
bool OckMemoryMallocFunOp::PreConditionMet(void)
{
    return true;
}

OckTaskResourceType OckMemoryMallocFunOp::ResourceType(void) const
{
    return OckTaskResourceType::DEVICE_MEMORY_OP | OckTaskResourceType::HOST_MEMORY_OP;
}
std::shared_ptr<OckDefaultResult> OckMemoryMallocFunOp::Run(
    OckAsyncTaskContext &context, OckMemoryMallocParam &param, OckUserWaitInfoBase &waitInfo)
{
    if (waitInfo.WaitTimeOut()) {
        return std::make_shared<OckDefaultResult>(hmm::HMM_ERROR_WAIT_TIME_OUT);
    }
    OCK_HMM_LOG_DEBUG("Alloc " << param << " " << context);
    if (param.Size() == 0) {
        return std::make_shared<OckDefaultResult>(hmm::HMM_ERROR_INPUT_PARAM_ZERO_MALLOC);
    }
    hmm::OckHmmErrorCode errorCode = ACL_SUCCESS;
    if (param.Location() == hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR) {
        void *addr = nullptr;
        errorCode = aclrtMalloc(&addr, param.Size(), ACL_MEM_MALLOC_HUGE_FIRST);
        if (waitInfo.WaitTimeOut()) {
            auto ret = aclrtFree(addr);
            if (ret != ACL_SUCCESS) {
                OCK_HMM_LOG_ERROR("aclrtFree failed." << " ret=" << ret);
            }
        } else {
            *param.PtrAddr() = addr;
        }
    } else if (param.Location() == hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY) {
        void *addr = nullptr;
        errorCode = aclrtMallocHost(&addr, param.Size());
        if (waitInfo.WaitTimeOut()) {
            auto ret = aclrtFreeHost(addr);
            if (ret != ACL_SUCCESS) {
                OCK_HMM_LOG_ERROR("aclrtFreeHost failed." << " ret=" << ret);
            }
        } else {
            *param.PtrAddr() = addr;
        }
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