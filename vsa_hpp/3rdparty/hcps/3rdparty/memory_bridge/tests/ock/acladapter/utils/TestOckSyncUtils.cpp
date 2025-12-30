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
#include "ock/acladapter/utils/OckSyncUtils.h"
#include "ock/acladapter/WithEnvOckAsyncTaskExecuteService.h"
#include "ock/acladapter/WithEnvAclMock.h"
namespace ock {
namespace acladapter {
namespace test {
namespace {
const size_t PACKAGE_BYTES_PER_TRANSFER = 1024U * 1024U * 64U;
}
class TestOckSyncUtils : public WithEnvOckAsyncTaskExecuteService<WithEnvAclMock<testing::Test>> {
public:
    using BaseT = WithEnvOckAsyncTaskExecuteService<WithEnvAclMock<testing::Test>>;
    TestOckSyncUtils(void) : BaseT(), byteSize(PACKAGE_BYTES_PER_TRANSFER)
    {}
    void SetUp(void) override
    {
        BaseT::SetUp();
        aclInit(nullptr);
    }
    void TearDown(void) override
    {
        if (taskService.get() != nullptr) {
            taskService->Stop();
        }
        aclFinalize();
        BaseT::TearDown();
    }
    void WithCorrectTaskService(void)
    {
        if (this->taskService.get() != nullptr) {
            this->taskService->Stop();
        }
        MOCKER(aclrtSetDevice).stubs().will(returnValue(ACL_SUCCESS));
        this->taskService = OckAsyncTaskExecuteService::Create(
            this->deviceId, this->cpuSet, {{acladapter::OckTaskResourceType::HMM, this->transferThreadNum}});
        syncUtils = std::make_shared<OckSyncUtils>(*(this->taskService));
    }
    void WithInCorrectTaskService(void)
    {
        if (this->taskService.get() != nullptr) {
            this->taskService->Stop();
        }
        MOCKER(aclrtSetDevice).stubs().will(returnValue(ACL_ERROR_INTERNAL_ERROR));
        this->taskService = OckAsyncTaskExecuteService::Create(
            this->deviceId, this->cpuSet, {{acladapter::OckTaskResourceType::HMM, this->transferThreadNum}});
        syncUtils = std::make_shared<OckSyncUtils>(*(this->taskService));
    }
    size_t byteSize;
    std::shared_ptr<OckSyncUtils> syncUtils;
};
TEST_F(TestOckSyncUtils, auto_free)
{
    WithCorrectTaskService();
    auto mallocRet = syncUtils->Malloc(byteSize, hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY);
    EXPECT_EQ(mallocRet.first, hmm::HMM_SUCCESS);
    ASSERT_TRUE(mallocRet.second->GetAddr() != nullptr);
    EXPECT_EQ(mallocRet.second->Location(), hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY);
    EXPECT_EQ((uintptr_t)mallocRet.second->GetAddr(), mallocRet.second->Addr());
}
TEST_F(TestOckSyncUtils, malloc_free)
{
    WithCorrectTaskService();
    auto mallocRet = syncUtils->Malloc(byteSize, hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY);
    EXPECT_EQ(mallocRet.first, hmm::HMM_SUCCESS);
    ASSERT_TRUE(mallocRet.second->GetAddr() != nullptr);
    auto freeRet =
        syncUtils->Free(mallocRet.second->ReleaseGuard(), hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY);
    EXPECT_EQ(freeRet, hmm::HMM_SUCCESS);
}
TEST_F(TestOckSyncUtils, copy)
{
    WithCorrectTaskService();
    auto mallocSrc = syncUtils->Malloc(byteSize, hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY);
    auto mallocDst = syncUtils->Malloc(byteSize, hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY);
    ASSERT_EQ(mallocSrc.first, hmm::HMM_SUCCESS);
    ASSERT_EQ(mallocDst.first, hmm::HMM_SUCCESS);
    auto copyRet = syncUtils->Copy(
        mallocDst.second->GetAddr(), byteSize, mallocSrc.second->GetAddr(), byteSize, OckMemoryCopyKind::HOST_TO_HOST);
    EXPECT_EQ(copyRet, hmm::HMM_SUCCESS);
    syncUtils->Free(mallocSrc.second->ReleaseGuard(), hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY);
    syncUtils->Free(mallocDst.second->ReleaseGuard(), hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY);
}
TEST_F(TestOckSyncUtils, malloc_null)
{
    WithInCorrectTaskService();
    auto mallocRet = syncUtils->Malloc(byteSize, hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 1U);
    EXPECT_EQ(mallocRet.first, hmm::HMM_ERROR_WAIT_TIME_OUT);
    aclmock::UnDoAclMockAsan();
}
}  // namespace test
}  // namespace acladapter
}  // namespace ock
