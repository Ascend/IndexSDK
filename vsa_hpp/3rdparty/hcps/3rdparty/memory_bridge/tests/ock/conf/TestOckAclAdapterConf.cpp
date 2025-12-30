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
#include "ock/conf/OckAclAdapterConf.h"

namespace ock {
namespace conf {

TEST(TestOckAclAdapterConf, operator)
{
    OckAclAdapterConf confA;
    OckAclAdapterConf confB;
    EXPECT_EQ(confA, confB);
    EXPECT_EQ(utils::ToString(confA), utils::ToString(confB));
    confA.maxFreeWaitMilliSecondThreshold++;
    EXPECT_NE(confA, confB);
    EXPECT_NE(utils::ToString(confA), utils::ToString(confB));
    confA.maxFreeWaitMilliSecondThreshold = confB.maxFreeWaitMilliSecondThreshold;
    confA.taskThreadMaxCyclePickUpInterval++;
    EXPECT_NE(confA, confB);
    EXPECT_NE(utils::ToString(confA), utils::ToString(confB));
    confA.taskThreadMaxCyclePickUpInterval = confB.taskThreadMaxCyclePickUpInterval;
    confB.taskThreadQueryStartInterval++;
    EXPECT_NE(confA, confB);
    EXPECT_NE(utils::ToString(confA), utils::ToString(confB));
    confA.taskThreadQueryStartInterval = confB.taskThreadQueryStartInterval;
    confB.taskThreadMaxQueryStartTimes++;
    EXPECT_NE(confA, confB);
    EXPECT_NE(utils::ToString(confA), utils::ToString(confB));
}
}  // namespace conf
}  // namespace ock