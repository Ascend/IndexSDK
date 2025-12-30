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
#include <limits>
#include "ock/utils/StrUtils.h"

namespace ock {
namespace utils {
namespace test {
namespace {
template <typename T>
void ExpectFromStringValue(const std::string &data, bool expectRet, T expectValue)
{
    T result;
    bool ret = FromString(result, data);
    EXPECT_EQ(expectRet, ret);
    if (ret) {
        EXPECT_EQ(expectValue, result);
    }
}
template <typename T>
void TestInvalidNumberCaseSet(void)
{
    ExpectFromStringValue("!", false, T{0});
    ExpectFromStringValue("false", false, T{0});
    ExpectFromStringValue("12ab", false, T{0});
    ExpectFromStringValue(" ", false, T{0});
    ExpectFromStringValue("", false, T{0});
    ExpectFromStringValue("12 ", false, T{0});
    ExpectFromStringValue("-12 ", false, T{0});
    ExpectFromStringValue("Y", false, T{0});
    ExpectFromStringValue("y", false, T{0});
    ExpectFromStringValue("t", false, T{0});
    ExpectFromStringValue("T", false, T{0});
    ExpectFromStringValue("N", false, T{0});
    ExpectFromStringValue("n", false, T{0});
    ExpectFromStringValue("true", false, T{0});
    ExpectFromStringValue("false", false, T{0});
    ExpectFromStringValue("True", false, T{0});
    ExpectFromStringValue("False", false, T{0});
    ExpectFromStringValue("a\b1", false, T{0});
    ExpectFromStringValue("\n", false, T{0});
}
template <typename T>
void TestMinMaxNumberCaseSet(void)
{
    ExpectFromStringValue(utils::ToString(std::numeric_limits<T>::min()), true, std::numeric_limits<T>::min());
    ExpectFromStringValue(utils::ToString(std::numeric_limits<T>::max()), true, std::numeric_limits<T>::max());
}
template <typename T>
void TestUnsignedCaseSet(void)
{
    ExpectFromStringValue("0x01", true, T{1});
    ExpectFromStringValue("+1", true, T{1});
    ExpectFromStringValue("-1", false, T{0});
    ExpectFromStringValue("-1 ", false, T{0});
    ExpectFromStringValue(" -1", false, T{0});
    ExpectFromStringValue("    -1", false, T{0});
    ExpectFromStringValue(" +1", true, T{1});
    ExpectFromStringValue(" -1", false, T{0});
    ExpectFromStringValue(" 12", true, T{12});
    ExpectFromStringValue("\n1", true, T{1});
    ExpectFromStringValue("\n-1", false, T{0});
    TestInvalidNumberCaseSet<T>();
}
template <typename T>
void TestSignedCaseSet(void)
{
    ExpectFromStringValue("0x01", true, T{1});
    ExpectFromStringValue("+0x01", true, T{1});
    ExpectFromStringValue("-0x01", true, T{-1});
    ExpectFromStringValue(" 12", true, T{12});
    ExpectFromStringValue(" -12", true, T{-12});
    ExpectFromStringValue(" +12", true, T{12});
    ExpectFromStringValue("+1", true, T{1});
    ExpectFromStringValue("-1", true, T{-1});
    ExpectFromStringValue("\n1", true, T{1});
    ExpectFromStringValue("\n-1", true, T{-1});
    ExpectFromStringValue(" \n1", true, T{1});
    ExpectFromStringValue(" \n-1", true, T{-1});
    TestInvalidNumberCaseSet<T>();
}
}  // namespace
TEST(TestFromString, to_bool)
{
    ExpectFromStringValue("1", true, true);
    ExpectFromStringValue("Y", true, true);
    ExpectFromStringValue("Yes", true, true);
    ExpectFromStringValue("T", true, true);
    ExpectFromStringValue("true", true, true);
    ExpectFromStringValue("TRUE", true, true);
    ExpectFromStringValue("True", true, true);
    ExpectFromStringValue("YES", true, true);
    ExpectFromStringValue("0", true, false);
    ExpectFromStringValue("N", true, false);
    ExpectFromStringValue("No", true, false);
    ExpectFromStringValue("F", true, false);
    ExpectFromStringValue("false", true, false);
    ExpectFromStringValue("False", true, false);
    ExpectFromStringValue("FALSE", true, false);
    ExpectFromStringValue("No", true, false);
    ExpectFromStringValue("11", false, false);
    ExpectFromStringValue("Y1", false, false);
    ExpectFromStringValue("N1", false, false);
    ExpectFromStringValue("A", false, false);
    ExpectFromStringValue("a", false, false);
    ExpectFromStringValue("", false, false);
    ExpectFromStringValue(" ", false, false);
}
TEST(TestFromString, signed_number)
{
    TestSignedCaseSet<int8_t>();
    ExpectFromStringValue(utils::ToString(std::numeric_limits<int8_t>::max()), false, int8_t{0});
    ExpectFromStringValue(utils::ToString(int32_t{INT8_MIN}), true, std::numeric_limits<int8_t>::min());
    ExpectFromStringValue(utils::ToString(int32_t{INT8_MAX}), true, std::numeric_limits<int8_t>::max());
    TestSignedCaseSet<int16_t>();
    TestMinMaxNumberCaseSet<int16_t>();
    ExpectFromStringValue(utils::ToString(std::numeric_limits<int32_t>::max()), false, int16_t{0});
    TestSignedCaseSet<int32_t>();
    TestMinMaxNumberCaseSet<int32_t>();
    ExpectFromStringValue(utils::ToString(std::numeric_limits<int64_t>::max()), false, int32_t{0});
    TestSignedCaseSet<int64_t>();
    TestMinMaxNumberCaseSet<int64_t>();
}
TEST(TestFromString, unsigned_number)
{
    TestUnsignedCaseSet<uint8_t>();
    ExpectFromStringValue(utils::ToString(uint32_t{0}), true, std::numeric_limits<uint8_t>::min());
    ExpectFromStringValue(utils::ToString(uint32_t{UINT8_MAX}), true, std::numeric_limits<uint8_t>::max());
    TestUnsignedCaseSet<uint16_t>();
    TestMinMaxNumberCaseSet<uint16_t>();
    ExpectFromStringValue(utils::ToString(std::numeric_limits<uint32_t>::max()), false, uint16_t{0});
    TestUnsignedCaseSet<uint32_t>();
    TestMinMaxNumberCaseSet<uint32_t>();
    ExpectFromStringValue(utils::ToString(std::numeric_limits<uint64_t>::max()), false, uint32_t{0});
    TestUnsignedCaseSet<uint64_t>();
    TestMinMaxNumberCaseSet<uint64_t>();
}
}  // namespace test
}  // namespace utils
}  // namespace ock