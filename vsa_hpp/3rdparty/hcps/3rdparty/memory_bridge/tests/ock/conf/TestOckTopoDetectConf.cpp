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

#include <gtest/gtest.h>
#include "ock/utils/StrUtils.h"
#include "ock/tools/topo/TopoDetectParam.h"
#include "ock/conf/OckTopoDetectConf.h"
#include "ock/conf/OckSysConf.h"

namespace ock {
namespace conf {

TEST(TestOckTopoDetectConf, operator)
{
    OckTopoDetectConf confA;
    OckTopoDetectConf confB;
    EXPECT_EQ(confA, confB);
    EXPECT_EQ(utils::ToString(confA), utils::ToString(confB));
    confA.toolParrallQueryIntervalMicroSecond++;
    EXPECT_NE(confA, confB);
    EXPECT_NE(utils::ToString(confA), utils::ToString(confB));
    confA.toolParrallQueryIntervalMicroSecond = confB.toolParrallQueryIntervalMicroSecond;
    confA.toolParrallMaxQueryTimes++;
    EXPECT_NE(confA, confB);
    EXPECT_NE(utils::ToString(confA), utils::ToString(confB));
}
TEST(TestOckTopoDetectConf, relation)
{
    EXPECT_LT(OckSysConf::ToolConf().testTime.maxValue,
        (OckSysConf::ToolConf().toolParrallQueryIntervalMicroSecond * OckSysConf::ToolConf().toolParrallMaxQueryTimes) /
            (1000000U));
}
}  // namespace conf
}  // namespace ock