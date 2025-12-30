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
#include <string>
#include <gtest/gtest.h>
#include "ock/utils/OckSafeUtils.h"

namespace ock {
namespace utils {

TEST(TestSafeEqual, safe_equal_basic)
{
    EXPECT_TRUE(SafeEqual(int8_t{1}, int8_t{1}));
    EXPECT_TRUE(SafeEqual(int16_t{1}, int16_t{1}));
    EXPECT_TRUE(SafeEqual(int32_t{1}, int32_t{1}));
    EXPECT_TRUE(SafeEqual(int64_t{1}, int64_t{1}));
    EXPECT_TRUE(SafeEqual(uint8_t{1}, uint8_t{1}));
    EXPECT_TRUE(SafeEqual(uint16_t{1}, uint16_t{1}));
    EXPECT_TRUE(SafeEqual(uint32_t{1}, uint32_t{1}));
    EXPECT_TRUE(SafeEqual(uint64_t{1}, uint64_t{1}));
    EXPECT_TRUE(SafeFloatEqual(double{0.000000001}, double{0.000000002}, double{0.00000001}));
    EXPECT_TRUE(SafeFloatEqual(double{0.000000001}, double{0.000000001}));
    EXPECT_FALSE(SafeEqual(int8_t{1}, int8_t{2}));
    EXPECT_FALSE(SafeEqual(int16_t{1}, int16_t{2}));
    EXPECT_FALSE(SafeEqual(int32_t{1}, int32_t{2}));
    EXPECT_FALSE(SafeEqual(int64_t{1}, int64_t{2}));
    EXPECT_FALSE(SafeEqual(uint8_t{1}, uint8_t{2}));
    EXPECT_FALSE(SafeEqual(uint16_t{1}, uint16_t{2}));
    EXPECT_FALSE(SafeEqual(uint32_t{1}, uint32_t{2}));
    EXPECT_FALSE(SafeEqual(uint64_t{1}, uint64_t{2}));
    EXPECT_FALSE(SafeFloatEqual(double{0.000000001}, double{0.000000002}, double{0.0000000001}));
}

TEST(TestSafeEqual, signed_with_unsigned)
{
    EXPECT_TRUE(SafeEqual(int8_t{1}, uint8_t{1}));
    EXPECT_TRUE(SafeEqual(int16_t{1}, uint16_t{1}));
    EXPECT_TRUE(SafeEqual(int32_t{1}, uint32_t{1}));
    EXPECT_TRUE(SafeEqual(int64_t{1}, uint64_t{1}));
    EXPECT_TRUE(SafeEqual(double{1.0}, uint64_t{1}));
    EXPECT_TRUE(SafeEqual(int64_t{1}, double{1.0}));
    EXPECT_FALSE(SafeEqual(double{1.1}, uint64_t{1}));
    EXPECT_FALSE(SafeEqual(int64_t{1}, double{1.1}));
    EXPECT_FALSE(SafeEqual(int8_t{-1}, std::numeric_limits<uint8_t>::max()));
    EXPECT_FALSE(SafeEqual(int16_t{-1}, std::numeric_limits<uint16_t>::max()));
    EXPECT_FALSE(SafeEqual(int32_t{-1}, std::numeric_limits<uint32_t>::max()));
    EXPECT_FALSE(SafeEqual(int64_t{-1}, std::numeric_limits<uint64_t>::max()));
}
TEST(TestSafeLessThan, signed_with_unsigned)
{
    EXPECT_TRUE(SafeLessThan(int8_t{1}, uint8_t{2}));
    EXPECT_TRUE(SafeLessThan(int16_t{1}, uint16_t{2}));
    EXPECT_TRUE(SafeLessThan(int32_t{1}, uint32_t{2}));
    EXPECT_TRUE(SafeLessThan(int64_t{1}, uint64_t{2}));
    EXPECT_TRUE(SafeLessThan(double{1.0}, uint64_t{2}));
    EXPECT_TRUE(SafeLessThan(int64_t{1}, double{2.0}));
    EXPECT_FALSE(SafeLessThan(double{1.1}, uint64_t{1}));
    EXPECT_FALSE(SafeLessThan(int64_t{2}, double{1.1}));
    EXPECT_TRUE(SafeLessThan(int8_t{-1}, std::numeric_limits<uint8_t>::max()));
    EXPECT_TRUE(SafeLessThan(int16_t{-1}, std::numeric_limits<uint16_t>::max()));
    EXPECT_TRUE(SafeLessThan(int32_t{-1}, std::numeric_limits<uint32_t>::max()));
    EXPECT_TRUE(SafeLessThan(int64_t{-1}, std::numeric_limits<uint64_t>::max()));
}
TEST(TestDivDown, divdown)
{
    EXPECT_EQ(SafeDivDown(int64_t{10}, uint64_t{10}), 1U);
    EXPECT_EQ(SafeDivDown(int64_t{15}, uint64_t{10}), 1U);
    EXPECT_EQ(SafeDivDown(int64_t{10}, uint64_t{0}), std::numeric_limits<uint64_t>::max());
}
TEST(TestDivUp, divUp)
{
    EXPECT_EQ(SafeDivUp(int64_t{10}, uint64_t{10}), 1U);
    EXPECT_EQ(SafeDivUp(int64_t{15}, uint64_t{10}), 2U);
    EXPECT_EQ(SafeDivUp(int64_t{10}, uint64_t{0}), std::numeric_limits<uint64_t>::max());
}
TEST(TestDiv, div)
{
    EXPECT_EQ(SafeDiv(double{10}, double{10}), double{1.0});
    EXPECT_EQ(SafeDiv(double{15}, double{10}), double{1.5});
    EXPECT_EQ(SafeDiv(double{10}, double {0}), std::numeric_limits<double>::max());
}
TEST(TestMod, mod)
{
    EXPECT_EQ(SafeMod(int64_t{10}, uint64_t{10}), 0U);
    EXPECT_EQ(SafeMod(int64_t{15}, uint64_t{10}), 5U);
    EXPECT_EQ(SafeMod(int64_t{10}, uint64_t{0}), uint64_t{10});
}
}  // namespace utils
}  // namespace ock