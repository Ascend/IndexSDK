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
#include "acl_base.h"
#include "simu/AscendSimuLog.h"


class testLogger : public ::testing::Test {
protected:
    static void SetUpTestCase()
    {
        // 设置模拟环境
        LOGGER().Reset();
        LOGGER().SetLogLevel(ACL_INFO); // 日志等级设为INFO
        LOGGER().SetLogFile("./log.txt"); // 日志目录为log.txt
    }

    static void TearDownTestCase()
    {
        LOGGER().Reset();
    }
};

TEST_F(testLogger, logwrite)
{
    ACL_APP_LOG(ACL_DEBUG, "hello %s %d debug\n", "world", 123);
    ACL_APP_LOG(ACL_INFO, "hello %s %d info\n", "world", 123);
    ACL_APP_LOG(ACL_WARNING, "hello %s %d warn\n", "world", 123);
}