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

#include <limits>
#include "ock/log/OckHcpsLogger.h"
#include "ock/hcps/handler/OckHeteroHandler.h"
#include "ock/hmm/mgr/OckHmmMgrCreator.h"
#include "ock/acladapter/executor/OckAsyncTaskExecuteService.h"
#include "ock/hcps/stream/OckDevStreamMgr.h"

namespace ock {
namespace hcps {
namespace handler {
const uint32_t TRANSFER_THREAD_NUMBER = 2UL;
const uint32_t STREAM_THREAD_NUMBER = 1UL;
const uint32_t TASKOP_THREAD_NUMBER = 25UL;
const uint32_t HMM_CREATE_TIMEOUT_MILLISECONDS = 300000UL;  // 300s

class OckHeteroHandlerImpl : public OckHeteroHandler {
public:
    virtual ~OckHeteroHandlerImpl() noexcept
    {
        OckDevStreamMgr::Instance().DestroyStreams(*service);
    }
    OckHeteroHandlerImpl(std::shared_ptr<hmm::OckHmmHeteroMemoryMgrBase> hmmManager,
        std::shared_ptr<acladapter::OckAsyncTaskExecuteService> taskService)
        : hmmMgr(hmmManager), service(taskService)
    {}
    hmm::OckHmmHeteroMemoryMgrBase &HmmMgr(void) override
    {
        return *hmmMgr;
    }
    std::shared_ptr<hmm::OckHmmHeteroMemoryMgrBase> HmmMgrPtr(void) override
    {
        return hmmMgr;
    }
    hmm::OckHmmDeviceId GetDeviceId() override
    {
        hmm::OckHmmSingleDeviceMgr *singleMgr = dynamic_cast<hmm::OckHmmSingleDeviceMgr *>(hmmMgr.get());
        if (singleMgr == nullptr) {
            OCK_HCPS_LOG_ERROR("hmmMgr dynamic cast to singleMgr failed.");
            return std::numeric_limits<uint16_t>::max();
        }
        return singleMgr->GetDeviceId();
    }
    std::shared_ptr<acladapter::OckAsyncTaskExecuteService> Service(void) override
    {
        return service;
    }
    std::shared_ptr<hmm::OckHmmHeteroMemoryMgrBase> hmmMgr;
    std::shared_ptr<acladapter::OckAsyncTaskExecuteService> service;
};

std::shared_ptr<OckHeteroHandler> OckHeteroHandler::CreateSingleDeviceHandler(hmm::OckHmmDeviceId deviceId,
    const cpu_set_t &cpuSet, const hmm::OckHmmMemorySpecification &memorySpec, hmm::OckHmmErrorCode &errorCode)
{
    if (errorCode != hmm::HMM_SUCCESS) {
        return std::shared_ptr<OckHeteroHandler>();
    }
    auto ret = OckHeteroHandler::CreateSingleDeviceHandler(deviceId, cpuSet, memorySpec);
    errorCode = ret.first;
    return ret.second;
}

std::pair<hmm::OckHmmErrorCode, std::shared_ptr<OckHeteroHandler>> OckHeteroHandler::CreateSingleDeviceHandler(
    hmm::OckHmmDeviceId deviceId, const cpu_set_t &cpuSet, const hmm::OckHmmMemorySpecification &memorySpec)
{
    hmm::OckHmmDeviceInfo deviceInfo;
    deviceInfo.deviceId = deviceId;
    deviceInfo.cpuSet = cpuSet;
    deviceInfo.transferThreadNum = TRANSFER_THREAD_NUMBER;
    deviceInfo.memorySpec = memorySpec;
    std::shared_ptr<acladapter::OckAsyncTaskExecuteService> service =
        acladapter::OckAsyncTaskExecuteService::Create(deviceId,
            cpuSet,
            {{acladapter::OckTaskResourceType::HMM, deviceInfo.transferThreadNum},
                {acladapter::OckTaskResourceType::DEVICE_STREAM, STREAM_THREAD_NUMBER},
                {acladapter::OckTaskResourceType::OP_TASK, TASKOP_THREAD_NUMBER}});
    auto hmmRet = hmm::OckHmmMgrCreator::Create(deviceInfo, service, HMM_CREATE_TIMEOUT_MILLISECONDS);
    return std::make_pair(hmmRet.first, std::make_shared<OckHeteroHandlerImpl>(hmmRet.second, service));
}
}  // namespace handler
}  // namespace hcps
}  // namespace ock