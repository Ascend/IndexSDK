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

#include <memory>
#include "acl/acl.h"
#include "ock/acladapter/param/OckMemoryCopyParam.h"
#include "ock/acladapter/param/OckMemoryFreeParam.h"
#include "ock/acladapter/param/OckMemoryMallocParam.h"
#include "ock/acladapter/task/OckMemoryCopyTask.h"
#include "ock/acladapter/task/OckMemoryFreeTask.h"
#include "ock/acladapter/task/OckMemoryMallocTask.h"
#include "ock/acladapter/task/OckDftFunOpTask.h"
#include "ock/acladapter/utils/OckSyncUtils.h"
namespace ock {
namespace acladapter {

OckSyncUtils::OckSyncUtils(OckAsyncTaskExecuteService &taskService) : service(taskService)
{}
OckAsyncTaskExecuteService &OckSyncUtils::GetService(void)
{
    return service;
}
std::pair<hmm::OckHmmErrorCode, std::unique_ptr<OckAdapterMemoryGuard>> OckSyncUtils::Malloc(
    size_t byteSize, hmm::OckHmmHeteroMemoryLocation location, uint32_t timeout)
{
    uint8_t *retAddr = nullptr;
    auto bridge = std::make_shared<OckAsyncResultInnerBridge<OckDefaultResult>>();
    service.AddTask(OckMemoryMallocTask::Create(
        std::make_shared<OckMemoryMallocParam>((void **)&retAddr, byteSize, location), bridge));
    auto result = bridge->WaitResult(timeout);
    if (result.get() == nullptr) {
        return std::make_pair(
            hmm::HMM_ERROR_WAIT_TIME_OUT, OckAdapterMemoryGuard::Create(service, retAddr, byteSize, location));
    }
    return std::make_pair(result->ErrorCode(), OckAdapterMemoryGuard::Create(service, retAddr, byteSize, location));
}
hmm::OckHmmErrorCode OckSyncUtils::Free(void *addr, hmm::OckHmmHeteroMemoryLocation location, uint32_t timeout)
{
    auto bridge = std::make_shared<OckAsyncResultInnerBridge<OckDefaultResult>>();
    service.AddTask(OckMemoryFreeTask::Create(std::make_shared<OckMemoryFreeParam>(addr, location), bridge));
    auto result = bridge->WaitResult(timeout);
    if (result.get() == nullptr) {
        return hmm::HMM_ERROR_WAIT_TIME_OUT;
    }
    return result->ErrorCode();
}
hmm::OckHmmErrorCode OckSyncUtils::Copy(
    void *dst, size_t destMax, const void *src, size_t count, OckMemoryCopyKind kind, uint32_t timeout)
{
    auto bridge = std::make_shared<OckAsyncResultInnerBridge<OckDefaultResult>>();
    service.AddTask(
        OckMemoryCopyTask::Create(std::make_shared<OckMemoryCopyParam>(dst, destMax, src, count, kind), bridge));
    auto result = bridge->WaitResult(timeout);
    if (result.get() == nullptr) {
        return hmm::HMM_ERROR_WAIT_TIME_OUT;
    }
    return result->ErrorCode();
}
hmm::OckHmmErrorCode OckSyncUtils::CreateStream(OckDevRtStream &stream)
{
    return OckSyncUtils::ExecFun(OckTaskResourceType::DEVICE_STREAM, [&stream]() {
        auto ret = aclrtCreateStream(&stream);
        OCK_HMM_LOG_DEBUG("CreateStream");
        return ret;
    });
}
hmm::OckHmmErrorCode OckSyncUtils::DestroyStream(OckDevRtStream stream)
{
    return OckSyncUtils::ExecFun(OckTaskResourceType::DEVICE_STREAM, [stream]() {
        OCK_HMM_LOG_DEBUG("DestroyStream");
        return aclrtDestroyStream(stream);
    });
}
hmm::OckHmmErrorCode OckSyncUtils::SynchronizeStream(OckDevRtStream stream, uint32_t timeout)
{
    return OckSyncUtils::ExecFun(
        OckTaskResourceType::DEVICE_STREAM,
        [stream]() {
            OCK_HMM_LOG_DEBUG("SynchronizeStream");
            return aclrtSynchronizeStream(stream);
        },
        timeout);
}
hmm::OckHmmErrorCode OckSyncUtils::ExecFun(
    OckTaskResourceType resourceType, std::function<hmm::OckHmmErrorCode()> opFun, uint32_t timeout)
{
    auto bridge = OckSyncUtils::ExecFunAsync(resourceType, opFun);
    auto result = bridge->WaitResult(timeout);
    if (result.get() == nullptr) {
        return hmm::HMM_ERROR_WAIT_TIME_OUT;
    }
    return result->ErrorCode();
}
std::shared_ptr<OckAsyncResultInnerBridge<OckDefaultResult>> OckSyncUtils::ExecFunAsync(
    OckTaskResourceType resourceType, std::function<hmm::OckHmmErrorCode()> opFun)
{
    auto bridge = std::make_shared<OckAsyncResultInnerBridge<OckDefaultResult>>();
    service.AddTask(OckDftFunOpTask::Create(std::make_shared<OckDftFunOpTask::ParamT>(),
        std::make_shared<OckDftFunOpTask::FunOpT>(opFun, resourceType),
        bridge));
    return bridge;
}

}  // namespace acladapter
}  // namespace ock