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
#include "ock/utils/StrUtils.h"
#include "ock/acladapter/data/OckHmmTrafficInfo.h"
namespace ock {
namespace acladapter {
namespace test {
TEST(TestOckHmmTrafficInfo, toString)
{
    hmm::OckHmmDeviceId deviceId = 1;
    uint64_t movedBytes = 100;
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    OckHmmTrafficInfo trafficInfoA(deviceId, movedBytes, OckMemoryCopyKind::HOST_TO_DEVICE, startTime);
    OckHmmTrafficInfo trafficInfoB(deviceId, movedBytes, OckMemoryCopyKind::DEVICE_TO_HOST, startTime);
    OckHmmTrafficInfo trafficInfoC(deviceId, movedBytes, OckMemoryCopyKind::HOST_TO_DEVICE, startTime);

    EXPECT_EQ(trafficInfoA, trafficInfoA);
    EXPECT_NE(trafficInfoA, trafficInfoB);
    EXPECT_NE(utils::ToString(trafficInfoA), utils::ToString(trafficInfoB));
    EXPECT_NE(utils::ToString(trafficInfoA), utils::ToString(trafficInfoC));
    EXPECT_EQ(trafficInfoA, trafficInfoC);
}
}  // namespace test
}  // namespace acladapter
}  // namespace ock
