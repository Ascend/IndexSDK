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

#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <unistd.h>
#include <functional>
#include <sys/syscall.h>
#include "acl/acl.h"
#include "ock/log/OckLogger.h"
#include "ock/conf/OckSysConf.h"
#include "ock/utils/FunGuard.h"
#include "ock/acladapter/executor/OckTaskRunThreadExt.h"
namespace ock {
namespace acladapter {

std::ostream &operator<<(std::ostream &os, const cpu_set_t &data)
{
    return hmm::operator<<(os, data);
}

class OckTaskRunThreadExt : public OckTaskRunThreadGen, public OckAsyncTaskContext {
public:
    virtual ~OckTaskRunThreadExt() noexcept
    {
        Stop();
    }

    OckTaskRunThreadExt(hmm::OckHmmDeviceId deviceIdx, const cpu_set_t &cpuSets,
        OckAsyncTaskExecuteServiceExt &tasksQueue, OckTaskResourceType taskResourceType)
        : OckTaskRunThreadGen(deviceIdx, cpuSets),
          taskQueue(tasksQueue),
          currentTask(nullptr),
          resourceType(taskResourceType)
    {}

    hmm::OckHmmDeviceId GetDeviceId(void) const override
    {
        return deviceId;
    }

    std::shared_ptr<OckTaskStatisticsMgr> StatisticsMgr(void) override
    {
        return taskQueue.TaskStatisticsMgr();
    }

    hmm::OckHmmErrorCode Start(void) override
    {
        auto ret = StartRun();
        if (ret != hmm::HMM_SUCCESS) {
            return ret;
        }
        return WaitStart();
    }

    hmm::OckHmmErrorCode Stop(void) override
    {
        if (IsNullThread()) {
            return hmm::HMM_SUCCESS;
        }
        NotifyStop();
        return WaitStop();
    }

    void NotifyStop(void) override
    {
        OCK_HMM_LOG_INFO("NotifyStop deviceId=" << deviceId << " thread=" << threadId.load());
        canceled.store(true);
        std::lock_guard<std::mutex> lock(thrMutex);
        if (currentTask.get() != nullptr) {
            currentTask->Cancel();
        }
        condVar.notify_all();
        OCK_HMM_LOG_INFO("NotifyStop deviceId=" << deviceId << " thread=" << threadId.load() << " complete");
    }

private:
    hmm::OckHmmErrorCode StartRun(void)
    {
        std::lock_guard<std::mutex> lock(thrMutex);
        if (thr.get() != nullptr) {
            return hmm::HMM_ERROR_TASK_ALREADY_RUNNING;
        }
        canceled.store(false);
        started.store(false);
        thr = std::make_shared<std::thread>(&OckTaskRunThreadExt::Run, this);
        return hmm::HMM_SUCCESS;
    }

    hmm::OckHmmErrorCode WaitStart(void)
    {
        uint32_t tryTimes = 0;
        while (tryTimes++ < (conf::OckSysConf::AclAdapterConf().taskThreadMaxQueryStartTimes)) {
            if (started.load()) {
                return hmm::HMM_SUCCESS;
            }
            std::this_thread::sleep_for(
                std::chrono::milliseconds(conf::OckSysConf::AclAdapterConf().taskThreadQueryStartInterval));
        }
        return lastErrorCode.load();
    }

    void Run(void)
    {
        threadId.store(static_cast<int>(syscall(SYS_gettid)));
        utils::FunGuardWithRet<std::function<hmm::OckHmmErrorCode()>, std::function<hmm::OckHmmErrorCode()>>
            deviceGuard(
                [this](void) { return aclrtSetDevice(deviceId); }, [this](void) { return aclrtResetDevice(deviceId); });
        if (deviceGuard.AllocRet() != ACL_SUCCESS) {
            lastErrorCode.store(deviceGuard.AllocRet());
            OCK_HMM_LOG_ERROR("aclrtSetDevice failed. deviceId=" << GetDeviceId() << " ret=" << deviceGuard.AllocRet());
            return;
        }
        (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuSet);
        lastErrorCode.store(hmm::HMM_SUCCESS);
        started.store(true);
        OCK_HMM_LOG_INFO("Start thread succeed. thread=" << syscall(SYS_gettid) << " deviceId=" << GetDeviceId()
                                                         << " cpuSet=" << cpuSet);
        while (!canceled.load()) {
            currentTask = taskQueue.PickUp(resourceType);
            if (currentTask.get() == nullptr) {
                std::unique_lock<std::mutex> lock(thrMutex);
                condVar.wait_for(lock, std::chrono::microseconds(5UL));
                continue;
            }
            OCK_HMM_LOG_DEBUG("run task type: " << *currentTask);
            currentTask->Run(*this);
            currentTask.reset();
        }
        OCK_HMM_LOG_INFO("Thread run complete. thread=" << syscall(SYS_gettid) << " deviceId=" << GetDeviceId()
                                                        << " cpuSet=" << cpuSet);
    }

    OckAsyncTaskExecuteServiceExt &taskQueue;
    std::shared_ptr<OckAsyncTaskBase> currentTask;
    OckTaskResourceType resourceType;
};

std::shared_ptr<OckTaskRunThreadBase> CreateRunThread(hmm::OckHmmDeviceId deviceId, const cpu_set_t &cpuSet,
    OckAsyncTaskExecuteServiceExt &taskQueue, OckTaskResourceType resourceType)
{
    return std::make_shared<OckTaskRunThreadExt>(deviceId, cpuSet, taskQueue, resourceType);
}
}  // namespace acladapter
}  // namespace ock