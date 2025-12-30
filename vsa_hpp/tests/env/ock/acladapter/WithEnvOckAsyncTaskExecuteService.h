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


#ifndef OCK_MEMORY_BRIDGE_OCK_ADAPTER_TASK_WITH_ENV_ASYNC_TASK_EXECUTE_SERVICE_H
#define OCK_MEMORY_BRIDGE_OCK_ADAPTER_TASK_WITH_ENV_ASYNC_TASK_EXECUTE_SERVICE_H
#include <thread>
#include <chrono>
#include "acl/acl.h"
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include "ock/acladapter/executor/OckAsyncTaskExecuteService.h"
namespace ock {
namespace acladapter {
template <typename BaseT>
class WithEnvOckAsyncTaskExecuteService : public BaseT {
public:
    void SetUp(void) override
    {
        BaseT::SetUp();
        MOCKER(aclInit).stubs().will(returnValue(ACL_SUCCESS));
        MOCKER(aclFinalize).stubs().will(returnValue(ACL_SUCCESS));
        uint32_t cpuId = 1;
        CPU_ZERO(&cpuSet);
        CPU_SET(cpuId, &cpuSet);
        deviceId = 0;
        transferThreadNum = {2};
    }
    void TearDown(void) override
    {
        if (taskService.get() != nullptr) {
            taskService->Stop();
        }
        taskService.reset();
        BaseT::TearDown();
        GlobalMockObject::verify();
    }
    void WaitServiceStarted(uint32_t waitTimeOut = 20)
    {
        if (!taskService->AllStarted()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(waitTimeOut));
        }
    }
    cpu_set_t cpuSet;
    hmm::OckHmmDeviceId deviceId;
    uint32_t transferThreadNum;
    std::shared_ptr<OckAsyncTaskExecuteService> taskService;
};
}  // namespace acladapter
}  // namespace ock
#endif