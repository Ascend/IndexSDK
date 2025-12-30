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
#include <chrono>
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "acl/acl.h"
#include "ock/utils/StrUtils.h"
#include "ock/acladapter/executor/OckAsyncTaskExecuteServiceExt.h"
#include "ock/acladapter/executor/OckAsyncTaskExecuteService.h"
#include "ock/acladapter/executor/MockOckAsyncTaskBase.h"
#include "ock/acladapter/WithEnvOckAsyncTaskExecuteService.h"
namespace ock {
namespace acladapter {
namespace test {
class TestOckAsyncTaskExecuteService : public WithEnvOckAsyncTaskExecuteService<testing::Test> {
protected:
    void SetUp(void) override
    {
        WithEnvOckAsyncTaskExecuteService<testing::Test>::SetUp();
        aclInit(nullptr);
        taskMockPtr = new MockOckAsyncTaskBase();
        task = std::shared_ptr<OckAsyncTaskBase>(taskMockPtr);
    }
    void TearDown(void) override
    {
        aclFinalize();
        WithEnvOckAsyncTaskExecuteService<testing::Test>::TearDown();
    }
    std::shared_ptr<OckAsyncTaskBase> task;
    MockOckAsyncTaskBase *taskMockPtr;
};
TEST_F(TestOckAsyncTaskExecuteService, toString)
{
    MOCKER(aclrtSetDevice).stubs().will(returnValue(ACL_SUCCESS));
    taskService = OckAsyncTaskExecuteService::Create(
        deviceId, cpuSet, {{acladapter::OckTaskResourceType::HMM, transferThreadNum}});
    EXPECT_EQ("{'TaskCount':0,'ActiveThreadCount':2,'AllStarted':1,'StartErrorCode':0}", utils::ToString(*taskService));
    EXPECT_EQ("{'TaskCount':0,'ActiveThreadCount':2,'AllStarted':1,'StartErrorCode':0,'DeviceId':0,'cpuSet':[1]}",
        utils::ToString(static_cast<const OckAsyncTaskExecuteServiceExt &>(*taskService)));
    taskService->Stop();
}
TEST_F(TestOckAsyncTaskExecuteService, stop_succeed_while_empty_task_not_started)
{
    MOCKER(aclrtSetDevice).stubs().will(returnValue(ACL_SUCCESS));
    taskService = OckAsyncTaskExecuteService::Create(
        deviceId, cpuSet, {{acladapter::OckTaskResourceType::HMM, transferThreadNum}});
    EXPECT_CALL(*taskMockPtr, PreConditionMet()).WillRepeatedly(testing::Return(true));
    taskService->Stop();
    EXPECT_EQ(transferThreadNum, taskService->ThreadCount());
    EXPECT_EQ(0UL, taskService->ActiveThreadCount());
    EXPECT_EQ(0UL, taskService->TaskCount());
}
TEST_F(TestOckAsyncTaskExecuteService, stop_succeed_while_one_task_added)
{
    MOCKER(aclrtSetDevice).stubs().will(returnValue(ACL_SUCCESS));
    taskService = OckAsyncTaskExecuteService::Create(
        deviceId, cpuSet, {{acladapter::OckTaskResourceType::HMM, transferThreadNum}});
    EXPECT_CALL(*taskMockPtr, PreConditionMet()).WillRepeatedly(testing::Return(false));
    EXPECT_CALL(*taskMockPtr, Run(testing::_)).Times(0);
    taskService->AddTask(task);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    taskService->Stop();
    EXPECT_EQ(transferThreadNum, taskService->ThreadCount());
    EXPECT_EQ(0UL, taskService->ActiveThreadCount());
    EXPECT_EQ(0UL, taskService->TaskCount());
}
TEST_F(TestOckAsyncTaskExecuteService, stop_succeed_while_one_task_added_but_no_wait)
{
    MOCKER(aclrtSetDevice).stubs().will(returnValue(ACL_SUCCESS));
    taskService = OckAsyncTaskExecuteService::Create(
        deviceId, cpuSet, {{acladapter::OckTaskResourceType::HMM, transferThreadNum}});
    EXPECT_CALL(*taskMockPtr, PreConditionMet()).WillRepeatedly(testing::Return(false));
    EXPECT_CALL(*taskMockPtr, Run(testing::_)).WillRepeatedly(testing::Return());
    std::shared_ptr<OckAsyncTaskBase> task(new MockOckAsyncTaskBase());
    taskService->AddTask(task);
    taskService->Stop();
    EXPECT_EQ(transferThreadNum, taskService->ThreadCount());
    EXPECT_EQ(0UL, taskService->ActiveThreadCount());
    EXPECT_EQ(0UL, taskService->TaskCount());
}
}  // namespace test
}  // namespace acladapter
}  // namespace ock
