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


#include <sstream>
#include "gtest/gtest.h"
#include "ock/utils/StrUtils.h"
#include "ock/hmm/mgr/OckHmmHeteroMemoryLocation.h"
namespace ock {
namespace hmm {
namespace test {
TEST(TestOckHmmHeteroMemoryLocation, toString)
{
    EXPECT_EQ(utils::ToString(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY), "LOCAL_HOST_MEMORY");
    EXPECT_EQ(utils::ToString(OckHmmHeteroMemoryLocation::DEVICE_HBM), "DEVICE_HBM");
    EXPECT_EQ(utils::ToString(OckHmmHeteroMemoryLocation::DEVICE_DDR), "DEVICE_DDR");
    EXPECT_EQ(utils::ToString(static_cast<OckHmmHeteroMemoryLocation>(100UL)), "UnknownLocation(value=100)");
}

TEST(TestOckHmmMemoryAllocatePolicy, toString)
{
    EXPECT_EQ(utils::ToString(OckHmmMemoryAllocatePolicy::DEVICE_DDR_FIRST), "DEVICE_DDR_FIRST");
    EXPECT_EQ(utils::ToString(OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY), "DEVICE_DDR_ONLY");
    EXPECT_EQ(utils::ToString(OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY), "LOCAL_HOST_ONLY");
    EXPECT_EQ(utils::ToString(static_cast<OckHmmMemoryAllocatePolicy>(100UL)), "UnknownPolicy(value=100)");
}
}  // namespace test
}  // namespace hmm
}  // namespace ock
