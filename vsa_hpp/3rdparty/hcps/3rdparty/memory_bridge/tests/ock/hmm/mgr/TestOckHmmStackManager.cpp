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
#include "ock/hmm/mgr/OckHmmStackManager.h"
#include "ock/hmm/mgr/MockOckHmmHMObject.h"
namespace ock {
namespace hmm {
namespace test {
class TestOckHmmStackManager : public testing::Test {
public:
    std::shared_ptr<OckHmmStackManager> BuildHmmStackManager(void)
    {
        MockOckHmmHMObject *hmoMock = new MockOckHmmHMObject();
        EXPECT_CALL(*hmoMock, Addr).WillRepeatedly(testing::Return(hmoAddr));
        EXPECT_CALL(*hmoMock, GetByteSize).WillRepeatedly(testing::Return(hmoLength));
        std::shared_ptr<OckHmmHMObject> hmo(hmoMock);
        return OckHmmStackManager::Create(hmo);
    }
    uintptr_t hmoAddr{0x123456ULL};
    uintptr_t nullAddr{0ULL};
    uint64_t hmoLength{1024ULL};
    OckHmmErrorCode errorCode{HMM_SUCCESS};
};
TEST_F(TestOckHmmStackManager, alloc_zero)
{
    auto stackMgr = BuildHmmStackManager();
    EXPECT_EQ(stackMgr->GetBuffer(0, errorCode), OckHmmStackBuffer());
    EXPECT_EQ(errorCode, HMM_SUCCESS);
}
TEST_F(TestOckHmmStackManager, alloc_too_large)
{
    auto stackMgr = BuildHmmStackManager();
    EXPECT_EQ(stackMgr->GetBuffer(hmoLength + 1ULL, errorCode), OckHmmStackBuffer());
    EXPECT_EQ(errorCode, HMM_ERROR_STACK_MANAGE_SPACE_NOT_ENOUGH);
}
TEST_F(TestOckHmmStackManager, alloc_free_min_value)
{
    auto stackMgr = BuildHmmStackManager();
    OckHmmStackBufferGuard guard(stackMgr, stackMgr->GetBuffer(1ULL, errorCode));
    EXPECT_EQ(errorCode, HMM_SUCCESS);
    EXPECT_EQ(guard.Address(), hmoAddr);
    EXPECT_EQ(guard.Size(), 1ULL);
}
TEST_F(TestOckHmmStackManager, alloc_free_max_value)
{
    auto stackMgr = BuildHmmStackManager();
    OckHmmStackBufferGuard guard(stackMgr, stackMgr->GetBuffer(hmoLength, errorCode));
    EXPECT_EQ(errorCode, HMM_SUCCESS);
    EXPECT_EQ(guard.Address(), hmoAddr);
    EXPECT_EQ(guard.Size(), hmoLength);
}
TEST_F(TestOckHmmStackManager, alloc_free_multi)
{
    // 场景1： 连续申请3个
    auto stackMgr = BuildHmmStackManager();
    auto guardA = std::make_shared<OckHmmStackBufferGuard>(stackMgr, stackMgr->GetBuffer(3UL, errorCode));
    auto guardB = std::make_shared<OckHmmStackBufferGuard>(stackMgr, stackMgr->GetBuffer(3UL, errorCode));
    auto guardC = std::make_shared<OckHmmStackBufferGuard>(stackMgr, stackMgr->GetBuffer(3UL, errorCode));
    EXPECT_EQ(guardC->Address(), hmoAddr + guardA->Size() + guardB->Size());
    EXPECT_EQ(guardB->Address(), hmoAddr + guardA->Size());

    // 场景2： 释放最后1个后再申请
    guardC.reset();
    auto guardD = std::make_shared<OckHmmStackBufferGuard>(stackMgr, stackMgr->GetBuffer(3UL, errorCode));
    EXPECT_EQ(guardD->Address(), hmoAddr + guardA->Size() + guardB->Size());

    // 场景3： 释放中间的，后再申请 A,-B,-C|D,E
    guardB.reset();
    auto guardE = std::make_shared<OckHmmStackBufferGuard>(stackMgr, stackMgr->GetBuffer(3UL, errorCode));
    EXPECT_EQ(guardE->Address(), hmoAddr + 9ULL);

    // 场景4：在场景3基础上，释放最后的两个，剩下1个，后再申请
    guardD.reset();
    guardE.reset();
    auto guardF = std::make_shared<OckHmmStackBufferGuard>(stackMgr, stackMgr->GetBuffer(3UL, errorCode));
    EXPECT_EQ(guardF->Address(), hmoAddr + guardA->Size());
}
}  // namespace test
}  // namespace hmm
}  // namespace ock
