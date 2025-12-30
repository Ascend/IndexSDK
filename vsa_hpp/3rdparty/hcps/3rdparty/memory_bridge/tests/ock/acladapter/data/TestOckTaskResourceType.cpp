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


#include <memory>
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "mockcpp/mockcpp.hpp"
#include "ock/utils/StrUtils.h"
#include "ock/acladapter/data/OckTaskResourceType.h"
namespace ock {
namespace acladapter {
namespace test {
namespace {
bool IsInAll(OckTaskResourceType type)
{
    return RhsInLhs(OckTaskResourceType::ALL, type);
}
uint32_t Or(OckTaskResourceType lhs, OckTaskResourceType rhs)
{
    return static_cast<uint32_t>(lhs) + static_cast<uint32_t>(rhs);
}
}  // namespace
TEST(TestOckTaskResourceType, print_single_enum_type)
{
    EXPECT_EQ(utils::ToString(OckTaskResourceType::MEMORY_TRANSFER), "MEMORY_TRANSFER");
    EXPECT_EQ(utils::ToString(OckTaskResourceType::MEMORY_OP), "DEVICE_MEMORY_OP|HOST_MEMORY_OP");
    EXPECT_EQ(utils::ToString(OckTaskResourceType::DEVICE_MEMORY_OP), "DEVICE_MEMORY_OP");
    EXPECT_EQ(utils::ToString(OckTaskResourceType::HOST_MEMORY_OP), "HOST_MEMORY_OP");
    EXPECT_EQ(utils::ToString(OckTaskResourceType::DEVICE_AI_CUBE), "DEVICE_AI_CUBE");
    EXPECT_EQ(utils::ToString(OckTaskResourceType::DEVICE_AI_VECTOR), "DEVICE_AI_VECTOR");
    EXPECT_EQ(utils::ToString(OckTaskResourceType::DEVICE_AI_CPU), "DEVICE_AI_CPU");
    EXPECT_EQ(utils::ToString(OckTaskResourceType::HOST_CPU), "HOST_CPU");
    EXPECT_EQ(utils::ToString(OckTaskResourceType::CPU), "DEVICE_AI_CPU|HOST_CPU");
    EXPECT_EQ(utils::ToString(OckTaskResourceType::DEVICE_STREAM), "DEVICE_STREAM");
    EXPECT_EQ(utils::ToString(OckTaskResourceType::OP_TASK),
        "MEMORY_TRANSFER|DEVICE_MEMORY_OP|HOST_MEMORY_OP|DEVICE_AI_CUBE|DEVICE_AI_VECTOR|DEVICE_AI_CPU|HOST_CPU");
    EXPECT_EQ(utils::ToString(OckTaskResourceType::OP_SYNC_TASK),
        "MEMORY_TRANSFER|DEVICE_MEMORY_OP|HOST_MEMORY_OP|DEVICE_AI_CUBE|DEVICE_AI_VECTOR|DEVICE_AI_CPU|HOST_CPU|"
        "DEVICE_STREAM");

    // 此处选择当前OckTaskResourceType的最大枚举的两倍作为非法值测试。
    uint32_t unknownResourceType = static_cast<uint32_t>(OckTaskResourceType::DEVICE_STREAM) << 1U;
    std::ostringstream osStr;
    osStr << "UnknownResourceType(" << unknownResourceType << ")";
    EXPECT_EQ(utils::ToString(static_cast<OckTaskResourceType>(unknownResourceType)), osStr.str());
}
TEST(TestOckTaskResourceType, print_compose_enum_type)
{
    EXPECT_EQ(utils::ToString(static_cast<OckTaskResourceType>(
                  Or(OckTaskResourceType::MEMORY_TRANSFER, OckTaskResourceType::MEMORY_OP))),
        "MEMORY_TRANSFER|DEVICE_MEMORY_OP|HOST_MEMORY_OP");
    EXPECT_EQ(utils::ToString(static_cast<OckTaskResourceType>(
                  Or(OckTaskResourceType::DEVICE_MEMORY_OP, OckTaskResourceType::HOST_MEMORY_OP))),
        "DEVICE_MEMORY_OP|HOST_MEMORY_OP");
    EXPECT_EQ(utils::ToString(static_cast<OckTaskResourceType>(
                  Or(OckTaskResourceType::DEVICE_AI_CUBE, OckTaskResourceType::DEVICE_AI_VECTOR))),
        "DEVICE_AI_CUBE|DEVICE_AI_VECTOR");
    EXPECT_EQ(utils::ToString(static_cast<OckTaskResourceType>(
                  Or(OckTaskResourceType::HOST_CPU, OckTaskResourceType::DEVICE_STREAM))),
        "HOST_CPU|DEVICE_STREAM");
    EXPECT_EQ(utils::ToString(
                  static_cast<OckTaskResourceType>(Or(OckTaskResourceType::CPU, OckTaskResourceType::MEMORY_TRANSFER))),
        "MEMORY_TRANSFER|DEVICE_AI_CPU|HOST_CPU");
    EXPECT_EQ(utils::ToString(OckTaskResourceType::ALL), "ALL");
}

TEST(TestOckTaskResourceType, enum_bit_in_memory_op)
{
    EXPECT_TRUE(RhsInLhs(OckTaskResourceType::MEMORY_OP, OckTaskResourceType::HOST_MEMORY_OP));
    EXPECT_TRUE(RhsInLhs(OckTaskResourceType::MEMORY_OP, OckTaskResourceType::DEVICE_MEMORY_OP));
    EXPECT_FALSE(RhsInLhs(OckTaskResourceType::MEMORY_OP, OckTaskResourceType::MEMORY_TRANSFER));
    EXPECT_FALSE(RhsInLhs(OckTaskResourceType::MEMORY_OP, OckTaskResourceType::DEVICE_AI_CUBE));
    EXPECT_FALSE(RhsInLhs(OckTaskResourceType::MEMORY_OP, OckTaskResourceType::DEVICE_STREAM));
    EXPECT_FALSE(RhsInLhs(OckTaskResourceType::MEMORY_OP, OckTaskResourceType::HOST_CPU));
}
TEST(TestOckTaskResourceType, enum_bit_in_hmm)
{
    EXPECT_TRUE(RhsInLhs(OckTaskResourceType::HMM, OckTaskResourceType::HOST_MEMORY_OP));
    EXPECT_TRUE(RhsInLhs(OckTaskResourceType::HMM, OckTaskResourceType::DEVICE_MEMORY_OP));
    EXPECT_TRUE(RhsInLhs(OckTaskResourceType::HMM, OckTaskResourceType::MEMORY_TRANSFER));
    EXPECT_FALSE(RhsInLhs(OckTaskResourceType::HMM, OckTaskResourceType::DEVICE_AI_CUBE));
    EXPECT_FALSE(RhsInLhs(OckTaskResourceType::HMM, OckTaskResourceType::DEVICE_STREAM));
    EXPECT_FALSE(RhsInLhs(OckTaskResourceType::HMM, OckTaskResourceType::HOST_CPU));
}
TEST(TestOckTaskResourceType, enum_bit_in_all)
{
    EXPECT_TRUE(IsInAll(OckTaskResourceType::MEMORY_TRANSFER));
    EXPECT_TRUE(IsInAll(OckTaskResourceType::MEMORY_OP));
    EXPECT_TRUE(IsInAll(OckTaskResourceType::DEVICE_MEMORY_OP));
    EXPECT_TRUE(IsInAll(OckTaskResourceType::HOST_MEMORY_OP));
    EXPECT_TRUE(IsInAll(OckTaskResourceType::DEVICE_AI_CUBE));
    EXPECT_TRUE(IsInAll(OckTaskResourceType::MEMORY_TRANSFER));
    EXPECT_TRUE(IsInAll(OckTaskResourceType::DEVICE_AI_VECTOR));
    EXPECT_TRUE(IsInAll(OckTaskResourceType::DEVICE_AI_CPU));
    EXPECT_TRUE(IsInAll(OckTaskResourceType::HOST_CPU));
    EXPECT_TRUE(IsInAll(OckTaskResourceType::CPU));
    EXPECT_TRUE(IsInAll(OckTaskResourceType::DEVICE_STREAM));
}

TEST(TestOckTaskThreadNumberMap, print)
{
    EXPECT_EQ("[]", utils::ToString(OckTaskThreadNumberMap{}));
    EXPECT_EQ("[{'resourceType':MEMORY_TRANSFER, 'threadCount':1}]",
        utils::ToString(OckTaskThreadNumberMap{{OckTaskResourceType::MEMORY_TRANSFER, 1U}}));
}
}  // namespace test
}  // namespace acladapter
}  // namespace ock
