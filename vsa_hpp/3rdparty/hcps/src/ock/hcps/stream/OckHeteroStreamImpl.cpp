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

#include <list>
#include <mutex>
#include "ock/log/OckLogger.h"
#include "ock/hmm/mgr/OckHmmErrorCode.h"
#include "ock/acladapter/task/OckDftFunOpTask.h"
#include "ock/acladapter/utils/OckSyncUtils.h"
#include "ock/hcps/stream/OckHeteroStreamBase.h"
#include "ock/hcps/stream/OckHeteroStreamContext.h"
#include "ock/hcps/stream/OckDevStreamMgr.h"
namespace ock {
namespace hcps {

class OckHeteroStreamImpl : public OckHeteroStreamBase, public OckHeteroStreamContext {
public:
    using OckHeteroOperatorTaskBridgeContainer = std::list<std::shared_ptr<acladapter::OckDftFunOpTask::BridgeT>>;
    virtual ~OckHeteroStreamImpl() noexcept
    {
        WaitExecComplete(0UL);
    }
    explicit OckHeteroStreamImpl(
        const OckDevStreamInfo &streamInfo, std::shared_ptr<acladapter::OckAsyncTaskExecuteService> taskService)
        : stream(streamInfo), service(taskService), complete(false)
    {}
    acladapter::OckDevRtStream DevRtStream(void) const override
    {
        return stream.stream;
    }
    void AddOp(std::shared_ptr<OckHeteroOperatorBase> op) override
    {
        if (op.get() == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> guard(thrMutex);
        complete.store(false);
        auto bridge = std::make_shared<acladapter::OckDftFunOpTask::BridgeT>();
        service->AddTask(acladapter::OckDftFunOpTask::Create(std::make_shared<acladapter::OckDftFunOpTask::ParamT>(),
            std::make_shared<acladapter::OckDftFunOpTask::FunOpT>(
                [op, this]() { return op->Run(*this); }, op->ResourceType()),
            bridge));
        bridgeContainer.push_back(bridge);
    }
    void AddOps(OckHeteroOperatorGroup &ops) override
    {
        for (auto &op : ops) {
            AddOp(op);
        }
    }
    void AddOps(OckHeteroOperatorTroupe &troupes) override
    {
        for (auto &ops : troupes) {
            if (ops.get() == nullptr) {
                continue;
            }
            AddOps(*ops);
        }
    }
    OckHcpsErrorCode RunOps(OckHeteroOperatorGroupQueue &ops, OckStreamExecPolicy policy, uint32_t timeout) override
    {
        OckHcpsErrorCode retCode = hmm::HMM_SUCCESS;
        while (!ops.empty()) {
            auto data = ops.front();
            if (data.get() == nullptr) {
                continue;
            }
            this->AddOps(*data);
            ops.pop();
            auto errorCode = this->WaitExecComplete(timeout);
            if (errorCode != hmm::HMM_SUCCESS) {
                retCode = errorCode;
                if (policy == OckStreamExecPolicy::STOP_IF_ERROR) {
                    break;
                }
            }
        }
        return retCode;
    }
    OckHcpsErrorCode WaitExecComplete(uint32_t timeout) override
    {
        std::lock_guard<std::mutex> guard(thrMutex);
        if (complete.load()) {
            return hmm::HMM_SUCCESS;
        }
        auto executedRetCode = WaitTaskExecuted(timeout);
        OckHcpsErrorCode syncRetCode = hmm::HMM_SUCCESS;
        if (stream.streamType != OckDevStreamType::AI_NULL) {
            acladapter::OckSyncUtils syncUtils(*service);
            syncRetCode = syncUtils.SynchronizeStream(stream.stream, timeout);
        }

        bridgeContainer.clear();  // 清理任务，避免下次等待的时候遍历之前的任务
        complete.store(true);
        if (executedRetCode != hmm::HMM_SUCCESS) {
            return executedRetCode;
        }
        return syncRetCode;
    }

private:
    OckHcpsErrorCode WaitTaskExecuted(uint32_t timeout)
    {
        auto retCode = hmm::HMM_SUCCESS;
        for (auto &bridge : bridgeContainer) {
            auto tmpRet = bridge->WaitResult(timeout);
            if (tmpRet.get() == nullptr) {
                return hmm::HMM_ERROR_WAIT_TIME_OUT;
            }
            if (tmpRet->ErrorCode() != hmm::HMM_SUCCESS) {
                retCode = tmpRet->ErrorCode();
            }
        }
        return retCode;
    }
    OckDevStreamInfo stream;
    OckHeteroOperatorTaskBridgeContainer bridgeContainer{};
    std::shared_ptr<acladapter::OckAsyncTaskExecuteService> service;
    std::atomic<bool> complete;
    mutable std::mutex thrMutex{};
};
std::pair<hmm::OckHmmErrorCode, std::shared_ptr<OckHeteroStreamBase>> OckHeteroStreamBase::Create(
    std::shared_ptr<acladapter::OckAsyncTaskExecuteService> service, OckDevStreamType streamType)
{
    if (service.get() == nullptr) {
        OCK_HMM_LOG_ERROR("Invalid service, nullptr");
        return std::make_pair(hmm::HMM_ERROR_INPUT_PARAM_EMPTY, std::shared_ptr<OckHeteroStreamBase>());
    }
    OckHcpsErrorCode errorCode = hmm::HMM_SUCCESS;
    auto stream = OckDevStreamMgr::Instance().CreateStream(*service, streamType, errorCode);
    return std::make_pair(errorCode, std::make_shared<OckHeteroStreamImpl>(stream, service));
}
std::ostream &operator<<(std::ostream &os, const OckHeteroStreamBase &data)
{
    return os << "'devStream':" << (uintptr_t)data.DevRtStream();
}
std::ostream &operator<<(std::ostream &os, OckDevStreamType data)
{
    switch (data) {
        case OckDevStreamType::AI_DEFAULT:
            os << "AI_DEFAULT";
            break;
        case OckDevStreamType::AI_CPU:
            os << "AI_CPU";
            break;
        case OckDevStreamType::AI_CORE:
            os << "AI_CORE";
            break;
        case OckDevStreamType::AI_NULL:
            os << "AI_NULL";
            break;
        default:
            os << "UNKNOWN(" << static_cast<uint32_t>(data) << ")";
            break;
    }
    return os;
}
}  // namespace hcps
}  // namespace ock