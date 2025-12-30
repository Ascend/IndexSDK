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
#include "secodeFuzz.h"
#include "mockcpp/mockcpp.hpp"
#include "ock/hmm/mgr/OckHmmErrorCode.h"
#include "ock/acladapter/executor/OckAsyncTaskExecuteService.h"
#include "ock/log/OckLogger.h"
#include "ock/acladapter/WithEnvOckAsyncTaskExecuteService.h"
#include "ock/acladapter/task/OckMemoryMallocTask.h"
namespace ock {
namespace acladapter {
namespace test {

class FuzzTestOckMemoryMallocTask : public WithEnvOckAsyncTaskExecuteService<testing::Test> {
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
    void CheckMalloc(hmm::OckHmmErrorCode erroCode)
    {
        auto bridge = std::make_shared<OckAsyncResultInnerBridge<OckDefaultResult>>();
        taskService->AddTask(OckMemoryMallocTask::Create(
            std::make_shared<OckMemoryMallocParam>((void **)(&ptrAddr), byteSize, location), bridge));
        auto result = bridge->WaitResult();
        ASSERT_TRUE(result.get() != nullptr);
        EXPECT_EQ(erroCode, result->ErrorCode());
    }
    uint8_t demoData;
    uint8_t *ptrAddr;
    size_t byteSize;
    hmm::OckHmmHeteroMemoryLocation location;
};
TEST_F(FuzzTestOckMemoryMallocTask, malloc_on_device)
{
    // 随便去个地址，避免ptrAddr为空
    ptrAddr = {&demoData};
    WaitServiceStarted();
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    uint32_t testCount = 10000;
    DT_FUZZ_START(seed, testCount, "malloc_on_device", 0)
    {
        int number = *(int *)DT_SetGetS32(&g_Element[0], 0x123456);
        MOCKER(aclrtMalloc).stubs().will(returnValue(number));
        CheckMalloc(number);
        GlobalMockObject::verify();
    }
    DT_FUZZ_END()
}

}  // namespace test
}  // namespace acladapter
}  // namespace ock
