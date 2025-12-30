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
#include <thread>
#include <chrono>
#include "gtest/gtest.h"
#include "securec.h"
#include "acl/acl.h"
#include "mockcpp/mockcpp.hpp"
#include "ock/acladapter/executor/OckAsyncTaskExecuteService.h"
#include "ock/log/OckLogger.h"
#include "ock/acladapter/WithEnvOckAsyncTaskExecuteService.h"
#include "ock/acladapter/task/OckMemoryCopyTask.h"
#include "ock/acladapter/WithEnvAclMock.h"
namespace ock {
namespace acladapter {
namespace test {
template <typename BaseT>
class TestOckMemoryCopyTask : public WithEnvOckAsyncTaskExecuteService<BaseT> {
public:
    void SetUp(void) override
    {
        WithEnvOckAsyncTaskExecuteService<BaseT>::SetUp();
        aclInit(nullptr);
        MOCKER(aclrtSetDevice).stubs().will(returnValue(ACL_SUCCESS));
        this->taskService = OckAsyncTaskExecuteService::Create(
            this->deviceId, this->cpuSet, {{acladapter::OckTaskResourceType::HMM, this->transferThreadNum}});
    }
    void TearDown(void) override
    {
        aclFinalize();
        WithEnvOckAsyncTaskExecuteService<BaseT>::TearDown();
    }
};
using TestOckMemoryCopyTaskIns = TestOckMemoryCopyTask<testing::Test>;

class BatchTestOckMemoryCopyTaskArgs {
public:
    BatchTestOckMemoryCopyTaskArgs(OckMemoryCopyKind ockKind, aclrtMemcpyKind aclKind)
        : ockKind(ockKind), aclKind(aclKind)
    {}
    OckMemoryCopyKind ockKind;
    aclrtMemcpyKind aclKind;
};

class BatchTestOckMemoryCopyTask
    : public TestOckMemoryCopyTask<testing::TestWithParam<BatchTestOckMemoryCopyTaskArgs>> {};

INSTANTIATE_TEST_SUITE_P(TestMemoryCopyTaskParameterized, BatchTestOckMemoryCopyTask,
    testing::Values(BatchTestOckMemoryCopyTaskArgs(OckMemoryCopyKind::HOST_TO_DEVICE, ACL_MEMCPY_HOST_TO_DEVICE),
        BatchTestOckMemoryCopyTaskArgs(OckMemoryCopyKind::DEVICE_TO_HOST, ACL_MEMCPY_DEVICE_TO_HOST),
        BatchTestOckMemoryCopyTaskArgs(OckMemoryCopyKind::DEVICE_TO_DEVICE, ACL_MEMCPY_DEVICE_TO_DEVICE)));

TEST_P(BatchTestOckMemoryCopyTask, call_acl_correct)
{
    auto args = GetParam();
    MOCKER(aclrtMemcpy)
        .expects(atLeast(1))
        .with(any(), any(), any(), any(), eq(args.aclKind))
        .will(returnValue(ACL_SUCCESS));
    auto bridge = std::make_shared<OckMemoryCopyTask::BridgeT>();
    taskService->AddTask(OckMemoryCopyTask::Create(
        std::make_shared<OckMemoryCopyTask::ParamT>(nullptr, 0, nullptr, 0, args.ockKind), bridge));
    auto result = bridge->WaitResult();
    ASSERT_TRUE(result.get() != nullptr);
    EXPECT_EQ(hmm::HMM_SUCCESS, result->ErrorCode());
}
TEST_F(TestOckMemoryCopyTaskIns, call_acl_failed_while_acl_return_failed)
{
    MOCKER(aclrtMemcpy).expects(exactly(1)).with().will(returnValue(ACL_ERROR_RT_MEMORY_FREE));
    auto bridge = std::make_shared<OckMemoryCopyTask::BridgeT>();
    taskService->AddTask(OckMemoryCopyTask::Create(
        std::make_shared<OckMemoryCopyTask::ParamT>(nullptr, 0, nullptr, 0, OckMemoryCopyKind::HOST_TO_DEVICE),
        bridge));
    auto result = bridge->WaitResult();
    ASSERT_TRUE(result.get() != nullptr);
    EXPECT_EQ(ACL_ERROR_RT_MEMORY_FREE, result->ErrorCode());
}
TEST_F(TestOckMemoryCopyTaskIns, call_acl_failed_while_unkown_ock_kind)
{
    MOCKER(aclrtMemcpy).expects(exactly(0));
    auto bridge = std::make_shared<OckMemoryCopyTask::BridgeT>();
    taskService->AddTask(OckMemoryCopyTask::Create(
        std::make_shared<OckMemoryCopyTask::ParamT>(nullptr, 0, nullptr, 0, static_cast<OckMemoryCopyKind>(100UL)),
        bridge));
    auto result = bridge->WaitResult();
    ASSERT_TRUE(result.get() != nullptr);
    EXPECT_EQ(hmm::HMM_ERROR_INPUT_PARAM_NOT_SUPPORT_SUCH_OP, result->ErrorCode());
}

}  // namespace test
}  // namespace acladapter
}  // namespace ock
