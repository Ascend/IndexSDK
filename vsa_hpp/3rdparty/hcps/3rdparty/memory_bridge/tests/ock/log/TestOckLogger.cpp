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


#include <sstream>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "ock/log/OckLogger.h"
#include "ock/log/MockOckHmmLogHandler.h"
namespace ock {
namespace test {
using testing::_;
class TestOckLogger : public testing::Test {
public:
    void SetUp(void) override
    {
        logHandler = new MockOckHmmLogHandler();
        OckHmmSetLogHandler(std::shared_ptr<OckHmmLogHandler>(logHandler));
    }
    void TearDown(void) override
    {
        OckHmmSetLogHandler(std::shared_ptr<OckHmmLogHandler>(new DftOckHmmLogHandler()));
    }
    MockOckHmmLogHandler &GetHandler(void)
    {
        return *logHandler;
    }
    MockOckHmmLogHandler *logHandler;
};
TEST_F(TestOckLogger, SetHandler)
{
    EXPECT_CALL(GetHandler(), Write(_, _, _, _, _)).WillOnce(testing::Return());
    OCK_HMM_LOG_ERROR("test log");
}
TEST_F(TestOckLogger, log_info_while_setlevel_invalid_input)
{
    EXPECT_CALL(GetHandler(), Write(_, _, _, _, _)).Times(2U).WillRepeatedly(testing::Return());
    OckHmmSetLogLevel(-1);
    OCK_HMM_LOG_INFO("test log");
    OCK_HMM_LOG_DEBUG("test log");
}
TEST_F(TestOckLogger, log_info_while_setlevel_info)
{
    EXPECT_CALL(GetHandler(), Write(_, _, _, _, _)).WillOnce(testing::Return());
    OckHmmSetLogLevel(OCK_LOG_LEVEL_INFO);
    OCK_HMM_LOG_INFO("test log");
    OCK_HMM_LOG_DEBUG("test log");
}
TEST_F(TestOckLogger, log_info_while_setlevel_debug)
{
    EXPECT_CALL(GetHandler(), Write(_, _, _, _, _)).Times(2U).WillRepeatedly(testing::Return());
    OckHmmSetLogLevel(OCK_LOG_LEVEL_DEBUG);
    OCK_HMM_LOG_INFO("test log");
    OCK_HMM_LOG_DEBUG("test log");
}
TEST_F(TestOckLogger, log_info_while_setlevel_error)
{
    EXPECT_CALL(GetHandler(), Write(_, _, _, _, _)).Times(1U).WillRepeatedly(testing::Return());
    OckHmmSetLogLevel(OCK_LOG_LEVEL_ERROR);
    OCK_HMM_LOG_INFO("test log");
    OCK_HMM_LOG_DEBUG("test log");
    OCK_HMM_LOG_ERROR("test log");
}
}  // namespace test
}  // namespace ock
