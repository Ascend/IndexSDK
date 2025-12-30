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
#include "ock/conf/OckHmmConf.h"

namespace ock {
namespace conf {

TEST(TestOckHmmConf, checkSubMemoryBaseSize)
{
    const uint64_t SubMemoryBaseSize = 64U; // 二次分配的内存块最小为64字节且以64字节对齐
    OckHmmConf hmmConf;
    bool checkMinSubMemoryBaseSize = hmmConf.subMemoryBaseSize >= SubMemoryBaseSize;
    bool checkModSubMemoryBaseSize = hmmConf.subMemoryBaseSize % SubMemoryBaseSize == 0;
    EXPECT_TRUE(checkMinSubMemoryBaseSize);
    EXPECT_TRUE(checkModSubMemoryBaseSize);
}

TEST(TestOckHmmConf, operator)
{
    OckHmmConf hmmConfA;
    OckHmmConf hmmConfB;
    EXPECT_EQ(hmmConfA, hmmConfB);
    EXPECT_EQ(utils::ToString(hmmConfA), utils::ToString(hmmConfB));
    hmmConfA.defaultFragThreshold++;
    EXPECT_NE(hmmConfA, hmmConfB);
    EXPECT_NE(utils::ToString(hmmConfA), utils::ToString(hmmConfB));
    hmmConfA.defaultFragThreshold = hmmConfB.defaultFragThreshold;
    hmmConfA.maxWaitTimeMilliSecond++;
    EXPECT_NE(hmmConfA, hmmConfB);
    EXPECT_NE(utils::ToString(hmmConfA), utils::ToString(hmmConfB));
    hmmConfA.maxWaitTimeMilliSecond = hmmConfB.maxWaitTimeMilliSecond;
    hmmConfB.maxHMOCountPerDevice++;
    EXPECT_NE(hmmConfA, hmmConfB);
    EXPECT_NE(utils::ToString(hmmConfA), utils::ToString(hmmConfB));
}
}  // namespace conf
}  // namespace ock