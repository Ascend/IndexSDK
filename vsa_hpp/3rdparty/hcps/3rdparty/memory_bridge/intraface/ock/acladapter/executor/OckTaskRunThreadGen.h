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


#ifndef OCK_MEMORY_BRIDGE_ACL_ADAPTER_TASK_RUN_THREAD_GEN_H
#define OCK_MEMORY_BRIDGE_ACL_ADAPTER_TASK_RUN_THREAD_GEN_H
#include <atomic>
#include <mutex>
#include <thread>
#include <condition_variable>
#include "ock/log/OckLogger.h"
#include "ock/acladapter/executor/OckTaskRunThreadBase.h"

namespace ock {
namespace acladapter {
class OckTaskRunThreadGen : public OckTaskRunThreadBase {
public:
    virtual ~OckTaskRunThreadGen() noexcept override = default;

    OckTaskRunThreadGen(hmm::OckHmmDeviceId deviceIdx, const cpu_set_t &cpuSets)
        : deviceId(deviceIdx), cpuSet(cpuSets), lastErrorCode(hmm::HMM_ERROR_WAIT_TIME_OUT), canceled(true),
          started(false), threadId(0)
    {}

    hmm::OckHmmDeviceId DeviceId(void) const override
    {
        return deviceId;
    }

    cpu_set_t CpuSet(void) const override
    {
        return cpuSet;
    }

    void NotifyAdded(void) override
    {
        OCK_HMM_LOG_DEBUG("NotifyAdded deviceId=" << DeviceId() << " thread=" << threadId.load());
        std::unique_lock<std::mutex> lock(thrMutex);
        condVar.notify_all();
        OCK_HMM_LOG_DEBUG("NotifyAdded deviceId=" << DeviceId() << " thread=" << threadId.load() << " complete");
    }

    bool Active(void) const override
    {
        std::unique_lock<std::mutex> lock(thrMutex);
        return thr.get() != nullptr && !canceled.load() && lastErrorCode.load() == hmm::HMM_SUCCESS;
    }

    bool IsNullThread(void) const
    {
        std::unique_lock<std::mutex> lock(thrMutex);
        if (thr.get() == nullptr) {
            return true;
        }
        return false;
    }

    hmm::OckHmmErrorCode WaitStop(void)
    {
        if (thr.get() != nullptr) {
            thr->join();
            started.store(false);
        }
        thr.reset();
        return hmm::HMM_SUCCESS;
    }

    hmm::OckHmmDeviceId deviceId;
    cpu_set_t cpuSet;
    std::atomic<hmm::OckHmmErrorCode> lastErrorCode;
    std::atomic_bool canceled;
    std::atomic_bool started;
    std::atomic_int threadId;
    mutable std::mutex thrMutex{};
    std::condition_variable condVar{};
    std::shared_ptr<std::thread> thr{ nullptr };
};
}
}
#endif // OCK_MEMORY_BRIDGE_ACL_ADAPTER_TASK_RUN_THREAD_GEN_H
