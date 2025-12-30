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
#include "acl/acl.h"
#include "mockcpp/mockcpp.hpp"
#include "ock/hmm/mgr/OckHmmErrorCode.h"
#include "ock/acladapter/executor/OckAsyncTaskExecuteService.h"
#include "ock/log/OckLogger.h"
#include "ock/acladapter/WithEnvOckAsyncTaskExecuteService.h"
#include "ock/acladapter/task/OckMemoryMallocTask.h"
#include "ock/acladapter/task/OckMemoryFreeTask.h"
namespace ock {
namespace acladapter {
namespace test {

class TestOckMemoryMallocFreeTask : public WithEnvOckAsyncTaskExecuteService<testing::Test> {
public:
    void SetUp(void) override
    {
        ptrAddr = nullptr;
        byteSize = {8192U};
        location = hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR;
        MOCKER(aclrtSetDevice).stubs().will(returnValue(ACL_SUCCESS));
        WithEnvOckAsyncTaskExecuteService<testing::Test>::SetUp();
        this->taskService = OckAsyncTaskExecuteService::Create(
            this->deviceId, this->cpuSet, {{acladapter::OckTaskResourceType::HMM, this->transferThreadNum}});
    }
    void TearDown(void) override
    {
        taskService->Stop();
        WithEnvOckAsyncTaskExecuteService<testing::Test>::TearDown();
    }
    void CheckFree(hmm::OckHmmErrorCode erroCode)
    {
        auto bridge = std::make_shared<OckAsyncResultInnerBridge<OckDefaultResult>>();
        taskService->AddTask(
            OckMemoryFreeTask::Create(std::make_shared<OckMemoryFreeParam>((void *)ptrAddr, location), bridge));
        auto result = bridge->WaitResult();
        ASSERT_TRUE(result.get() != nullptr);
        EXPECT_EQ(erroCode, result->ErrorCode());
    }
    void CheckMalloc(hmm::OckHmmErrorCode erroCode)
    {
        auto bridge = std::make_shared<OckAsyncResultInnerBridge<OckDefaultResult>>();
        taskService->AddTask(OckMemoryMallocTask::Create(
            std::make_shared<OckMemoryMallocParam>((void **)(&ptrAddr), byteSize, location), bridge));
        auto result = bridge->WaitResult();
        ASSERT_TRUE(result.get() != nullptr);
        EXPECT_EQ(erroCode, result->ErrorCode());
    }
    uint8_t demoDataA;
    uint8_t demoDataB;
    uint8_t *ptrAddr;
    size_t byteSize;
    hmm::OckHmmHeteroMemoryLocation location;
};

TEST_F(TestOckMemoryMallocFreeTask, malloc_free_on_device)
{
    // 随便去个地址，避免ptrAddr为空
    ptrAddr = {&demoDataA};
    MOCKER(aclrtMalloc).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtFree).stubs().will(returnValue(ACL_SUCCESS));
    CheckMalloc(hmm::HMM_SUCCESS);
    CheckFree(hmm::HMM_SUCCESS);
    location = hmm::OckHmmHeteroMemoryLocation::DEVICE_HBM;
    CheckMalloc(hmm::HMM_ERROR_INPUT_PARAM_NOT_SUPPORT_SUCH_OP);
    CheckFree(hmm::HMM_SUCCESS);
}
TEST_F(TestOckMemoryMallocFreeTask, malloc_free_on_device_return_error)
{
    // 随便去个地址，避免ptrAddr为空
    ptrAddr = {&demoDataA};
    MOCKER(aclrtMalloc).stubs().will(returnValue(ACL_ERROR_RT_MEMORY_ALLOCATION));
    MOCKER(aclrtFree).stubs().will(returnValue(ACL_ERROR_RT_MEMORY_FREE));
    CheckMalloc(ACL_ERROR_RT_MEMORY_ALLOCATION);
    ptrAddr = {&demoDataB};
    CheckFree(ACL_ERROR_RT_MEMORY_FREE);
}

TEST_F(TestOckMemoryMallocFreeTask, malloc_free_on_host_while_return_success)
{
    location = hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY;
    MOCKER(aclrtMallocHost).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtFreeHost).stubs().will(returnValue(ACL_SUCCESS));
    CheckMalloc(hmm::HMM_SUCCESS);
    CheckFree(hmm::HMM_SUCCESS);
}

TEST_F(TestOckMemoryMallocFreeTask, free_null)
{
    location = hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY;
    ptrAddr = nullptr;
    CheckFree(hmm::HMM_SUCCESS);
    location = hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR;
    CheckFree(hmm::HMM_SUCCESS);
}
TEST_F(TestOckMemoryMallocFreeTask, malloc_zero)
{
    // 随便去个地址，避免ptrAddr为空
    ptrAddr = {&demoDataA};
    byteSize = 0U;
    CheckMalloc(hmm::HMM_ERROR_INPUT_PARAM_ZERO_MALLOC);
}
TEST_F(TestOckMemoryMallocFreeTask, unkown_location)
{
    // 随便去个地址，避免ptrAddr为空
    ptrAddr = {&demoDataA};
    location = static_cast<hmm::OckHmmHeteroMemoryLocation>(100U);
    CheckMalloc(hmm::HMM_ERROR_INPUT_PARAM_NOT_SUPPORT_SUCH_OP);
    CheckFree(hmm::HMM_ERROR_INPUT_PARAM_NOT_SUPPORT_SUCH_OP);
}

}  // namespace test
}  // namespace acladapter
}  // namespace ock
