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

#include <ostream>
#include <cstdint>
#include <memory>
#include <chrono>
#include <gtest/gtest.h>
#include "ock/utils/OckIdGenerator.h"

namespace ock {
namespace utils {

TEST(TestOckIdGenerator, del_newId)
{
    uint32_t const maxIdCount = 100;
    OckIdGenerator<maxIdCount> generator;
    auto genInfo = generator.NewId();
    EXPECT_TRUE(genInfo.first);
    EXPECT_EQ(genInfo.second, 0U);
    EXPECT_EQ(generator.UsedCount(), 1U);
    generator.DelId(genInfo.second);
    EXPECT_EQ(generator.UsedCount(), 0U);
}
TEST(TestOckIdGenerator, full_new)
{
    uint32_t const maxIdCount = 100;
    OckIdGenerator<maxIdCount> generator;
    for (uint32_t i = 0; i < maxIdCount; ++i) {
        auto genInfo = generator.NewId();
        EXPECT_TRUE(genInfo.first);
        EXPECT_EQ(generator.UsedCount(), i + 1U);
    }
    auto genInfo = generator.NewId();
    EXPECT_FALSE(genInfo.first);
    uint32_t anyId = 3U;
    generator.DelId(anyId);  // 随便释放一个ID
    EXPECT_EQ(generator.UsedCount(), maxIdCount - 1U);
    genInfo = generator.NewId();
    EXPECT_TRUE(genInfo.first);
    EXPECT_EQ(genInfo.second, anyId);
    EXPECT_EQ(generator.UsedCount(), maxIdCount);
}
}  // namespace utils
}  // namespace ock