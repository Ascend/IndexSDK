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
#include "ock/hmm/mgr/OckHmmMemorySpecification.h"
#include "ock/utils/StrUtils.h"
namespace ock {
namespace hmm {
namespace test {

template <typename T, typename FunA, typename FunB>
void CompareSpecificationStruct(const T &spec, FunA incFunA, FunB incFunB)
{
    EXPECT_EQ(spec, spec);
    T other = spec;
    EXPECT_EQ(utils::ToString(other), utils::ToString(spec));
    incFunA(other);
    EXPECT_NE(spec, other);
    EXPECT_NE(utils::ToString(other), utils::ToString(spec));
    other = spec;
    incFunB(other);
    EXPECT_NE(spec, other);
    EXPECT_NE(utils::ToString(other), utils::ToString(spec));
}
TEST(TestOckHmmMemoryCapacitySpecification, operator_equal)
{
    CompareSpecificationStruct(
        OckHmmMemoryCapacitySpecification(),
        [](OckHmmMemoryCapacitySpecification &data) { data.maxDataCapacity++; },
        [](OckHmmMemoryCapacitySpecification &data) { data.maxSwapCapacity++; });
}
TEST(TestOckHmmMemorySpecification, operator_equal)
{
    CompareSpecificationStruct(
        OckHmmMemorySpecification(),
        [](OckHmmMemorySpecification &data) { data.devSpec.maxDataCapacity++; },
        [](OckHmmMemorySpecification &data) { data.hostSpec.maxSwapCapacity++; });
}
TEST(TestOckHmmMemoryUsedInfo, operator_equal)
{
    CompareSpecificationStruct(
        OckHmmMemoryUsedInfo(),
        [](OckHmmMemoryUsedInfo &data) {
            data.usedBytes++;
            data.leftBytes++;
            data.swapLeftBytes++;
        },
        [](OckHmmMemoryUsedInfo &data) {
            data.unusedFragBytes++;
            data.swapUsedBytes++;
        });
}
TEST(TestOckHmmResourceUsedInfo, operator_equal)
{
    CompareSpecificationStruct(
        OckHmmResourceUsedInfo(),
        [](OckHmmResourceUsedInfo &data) { data.devUsedInfo.swapUsedBytes++; },
        [](OckHmmResourceUsedInfo &data) { data.hostUsedInfo.usedBytes++; });
}
}  // namespace test
}  // namespace hmm
}  // namespace ock
