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


#include "gtest/gtest.h"
#include "secodeFuzz.h"
#include "ock/log/MockOckHmmLogHandler.h"

namespace ock {
static constexpr size_t DF_FUZZ_EXEC_COUNT = 1000000;
static constexpr size_t DF_FUZZ_EXEC_SECOND = 10800;

class FuzzTestOckHmmLog : public testing::Test {
public:
    void SetUpForFuzz(void)
    {
        DT_Set_Running_Time_Second(DF_FUZZ_EXEC_SECOND);
    }

    void SetUp(void) override
    {
        logHandler = std::make_shared<MockOckHmmLogHandler>();
    }

    void TearDown(void) override
    {
        logHandler.reset();
    }
    std::shared_ptr<MockOckHmmLogHandler> logHandler;
};
TEST_F(FuzzTestOckHmmLog, set_log_level_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;

    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "set_log_level", 0)
    {
        OckHmmSetLogLevel(OCK_LOG_LEVEL_WARN);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmLog, set_log_handler_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "set_log_handler", 0)
    {
        OckHmmSetLogHandler(logHandler);
    }
    DT_FUZZ_END()
}
}
