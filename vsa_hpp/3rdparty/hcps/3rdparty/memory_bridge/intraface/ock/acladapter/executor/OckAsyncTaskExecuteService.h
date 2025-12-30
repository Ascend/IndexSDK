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


#ifndef OCK_MEMORY_BRIDGE_OCK_ADAPTER_TASK_ASYNC_TASK_EXECUTE_SERVICE_H
#define OCK_MEMORY_BRIDGE_OCK_ADAPTER_TASK_ASYNC_TASK_EXECUTE_SERVICE_H
#include <memory>
#include <thread>
#include <vector>
#include "ock/hmm/mgr/OckHmmErrorCode.h"
#include "ock/acladapter/executor/OckTaskStatisticsMgr.h"
#include "ock/acladapter/executor/OckAsyncTaskBase.h"
#include "ock/acladapter/executor/OckTaskRunThreadBase.h"
namespace ock {
namespace acladapter {

using OckDevRtStream = void *;

class OckAsyncTaskExecuteService {
public:
    virtual ~OckAsyncTaskExecuteService() noexcept = default;

    virtual void AddTask(std::shared_ptr<OckAsyncTaskBase> task) = 0;

    virtual void CancelAll(void) = 0;

    virtual void Stop(void) = 0;
    virtual bool AllStarted(void) const = 0;

    virtual uint32_t ThreadCount(void) const = 0;
    virtual uint32_t ActiveThreadCount(void) const = 0;
    virtual uint32_t TaskCount(void) const = 0;
    virtual hmm::OckHmmErrorCode StartErrorCode(void) const = 0;
    virtual std::shared_ptr<OckTaskStatisticsMgr> TaskStatisticsMgr(void) = 0;
    virtual const OckTaskThreadNumberMap &TaskThreadNumberMap(void) const = 0;

    static std::shared_ptr<OckAsyncTaskExecuteService> Create(hmm::OckHmmDeviceId deviceId, const cpu_set_t &cpuSet,
        const OckTaskThreadNumberMap &taskThreadMap);

    static std::shared_ptr<OckAsyncTaskExecuteService> Create(hmm::OckHmmDeviceId deviceId,
        std::vector<cpu_set_t> &cpuSets, const OckTaskThreadNumberMap &taskThreadMap);

    static std::shared_ptr<OckAsyncTaskExecuteService> Create(
        std::vector<std::shared_ptr<OckTaskRunThreadBase>> threadList);
};
std::ostream &operator<<(std::ostream &os, const OckAsyncTaskExecuteService &service);
}  // namespace acladapter
}  // namespace ock
#endif