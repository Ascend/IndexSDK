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
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "mockcpp/mockcpp.hpp"
#include "ock/acladapter/executor/OckAsyncResultInnerBridge.h"
namespace ock {
namespace acladapter {
namespace test {

TEST(TestOckAsyncResultInnerBridge, waitResult_timeout)
{
    const uint32_t timeOutMilliSeconds = 1;
    OckAsyncResultInnerBridge<OckDefaultResult> bridge;
    auto result = bridge.WaitResult(timeOutMilliSeconds);
    ASSERT_TRUE(result.get() == nullptr || result->ErrorCode() == hmm::HMM_ERROR_WAIT_TIME_OUT);
}

TEST(TestOckAsyncResultInnerBridge, waitResult_succeed_while_setresult_firstly_error)
{
    OckAsyncResultInnerBridge<OckDefaultResult> bridge;
    bridge.SetResult(std::make_shared<OckDefaultResult>(hmm::HMM_ERROR_UNKOWN_INNER_ERROR));
    auto result = bridge.WaitResult();
    ASSERT_TRUE(result.get() != nullptr);
    EXPECT_EQ(result->ErrorCode(), hmm::HMM_ERROR_UNKOWN_INNER_ERROR);
}

TEST(TestOckAsyncResultInnerBridge, waitResult_succeed_while_setresult_firstly_success)
{
    OckAsyncResultInnerBridge<OckDefaultResult> bridge;
    bridge.SetResult(std::make_shared<OckDefaultResult>(hmm::HMM_SUCCESS));
    auto result = bridge.WaitResult();
    ASSERT_TRUE(result.get() != nullptr);
    EXPECT_EQ(result->ErrorCode(), hmm::HMM_SUCCESS);
}

}  // namespace test
}  // namespace acladapter
}  // namespace ock
