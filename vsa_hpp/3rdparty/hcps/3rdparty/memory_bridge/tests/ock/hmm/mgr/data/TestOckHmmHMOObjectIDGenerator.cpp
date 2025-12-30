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
#include "ock/hmm/mgr/data/OckHmmHMOObjectIDGenerator.h"
namespace ock {
namespace hmm {
namespace test {
TEST(TestOckHmmHMOObjectIDGenerator, generator_correct)
{
    hmm::OckHmmDeviceId deviceId = 1313;
    uintptr_t addr = uintptr_t(&deviceId);
    uint32_t csnId = 2959;
    auto objectId = OckHmmHMOObjectIDGenerator::Gen(deviceId, addr, csnId);
    EXPECT_TRUE(OckHmmHMOObjectIDGenerator::Valid(objectId, deviceId, addr));
    EXPECT_EQ(OckHmmHMOObjectIDGenerator::ParseDeviceId(objectId), deviceId);
    EXPECT_EQ(OckHmmHMOObjectIDGenerator::ParseCsnId(objectId), csnId);
    EXPECT_FALSE(OckHmmHMOObjectIDGenerator::Valid(objectId, deviceId + 1, addr));
    EXPECT_TRUE(OckHmmHMOObjectIDGenerator::Valid(objectId + 1, deviceId, addr));
    EXPECT_FALSE(OckHmmHMOObjectIDGenerator::Valid(objectId, deviceId, (uintptr_t)&csnId));
}

TEST(TestOckHmmHMOObjectIDGenerator, generator_boundary)
{
    hmm::OckHmmDeviceId deviceId = 0xF1;
    uintptr_t addr = 0xF3F5F7FA;
    uint32_t csnId = 0x8111;
    auto objectId = OckHmmHMOObjectIDGenerator::Gen(deviceId, addr, csnId);
    EXPECT_TRUE(OckHmmHMOObjectIDGenerator::Valid(objectId, deviceId, addr));
    EXPECT_EQ(OckHmmHMOObjectIDGenerator::ParseCsnId(objectId), csnId);
    EXPECT_EQ(OckHmmHMOObjectIDGenerator::ParseDeviceId(objectId), deviceId);
}

}  // namespace test
}  // namespace hmm
}  // namespace ock
