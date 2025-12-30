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
#include "ock/acladapter/executor/OckTaskStatisticsMgr.h"
namespace ock {
namespace acladapter {
namespace test {

class TestOckTaskStatisticsMgr : public testing::Test {
public:
    void SetUp() override
    {
        statMgr = OckTaskStatisticsMgr::Create();
        bs = std::chrono::steady_clock::now();
    }

    void AddData(hmm::OckHmmDeviceId deviceId, OckMemoryCopyKind copyKind, uint64_t startOffset, uint64_t endOffset)
    {
        OckHmmTrafficInfo trafficInfo(deviceId, 100U, copyKind, bs + std::chrono::microseconds(startOffset));
        trafficInfo.endTime = bs + std::chrono::microseconds(endOffset);
        statMgr->AddTrafficInfo(std::make_shared<OckHmmTrafficInfo>(trafficInfo));
    }

    std::shared_ptr<OckTaskStatisticsMgr> statMgr;
    std::chrono::steady_clock::time_point bs;
    uint32_t maxGapMilliSeconds = 1U;
};

TEST_F(TestOckTaskStatisticsMgr, single_device_continuous)
{
    AddData(0, OckMemoryCopyKind::HOST_TO_DEVICE, 0U, 60U);
    AddData(0, OckMemoryCopyKind::DEVICE_TO_HOST, 40U, 50U);
    AddData(0, OckMemoryCopyKind::HOST_TO_DEVICE, 60U, 150U);
    // 200: h2dBytes; 100:d2hBytes; 150:h2dTimeDuration; 10:d2hTimeDuration
    hmm::OckHmmTrafficStatisticsInfo correctInfo(200U, 100U, double(200) / double(150), double(100) / double(10));
    EXPECT_EQ(*(statMgr->PickUp(maxGapMilliSeconds)), correctInfo);
}

TEST_F(TestOckTaskStatisticsMgr, multi_device_continuous)
{
    AddData(0, OckMemoryCopyKind::HOST_TO_DEVICE, 0U, 60U);
    AddData(0, OckMemoryCopyKind::DEVICE_TO_HOST, 40U, 50U);
    AddData(1, OckMemoryCopyKind::DEVICE_TO_HOST, 150U, 170U);
    AddData(0, OckMemoryCopyKind::HOST_TO_DEVICE, 60U, 150U);
    AddData(1, OckMemoryCopyKind::HOST_TO_DEVICE, 260U, 300U);
    AddData(1, OckMemoryCopyKind::HOST_TO_DEVICE, 300U, 330U);
    // 200: h2dBytes; 100:d2hBytes; 150:h2dTimeDuration; 10:d2hTimeDuration
    hmm::OckHmmTrafficStatisticsInfo correctInfo0(200U, 100U, double(200) / double(150), double(100) / double(10));
    // 200: h2dBytes; 100:d2hBytes; 70:h2dTimeDuration; 20:d2hTimeDuration
    hmm::OckHmmTrafficStatisticsInfo correctInfo1(200U, 100U, double(200) / double(70), double(100) / double(20));
    EXPECT_EQ(*(statMgr->PickUp(0, maxGapMilliSeconds)), correctInfo0);
    EXPECT_EQ(*(statMgr->PickUp(1, maxGapMilliSeconds)), correctInfo1);
}

TEST_F(TestOckTaskStatisticsMgr, multi_device_gap_inside_same_device_and_kind)
{
    AddData(0, OckMemoryCopyKind::HOST_TO_DEVICE, 0U, 600U);
    AddData(0, OckMemoryCopyKind::DEVICE_TO_HOST, 400U, 500U);
    AddData(1, OckMemoryCopyKind::HOST_TO_DEVICE, 1500U, 1700U);
    AddData(0, OckMemoryCopyKind::HOST_TO_DEVICE, 600U, 1500U);
    AddData(1, OckMemoryCopyKind::DEVICE_TO_HOST, 2600U, 3000U);
    AddData(1, OckMemoryCopyKind::HOST_TO_DEVICE, 3000U, 3300U);
    // 200: h2dBytes; 100:d2hBytes; 1500:h2dTimeDuration; 100:d2hTimeDuration
    hmm::OckHmmTrafficStatisticsInfo correctInfo0(200U, 100U, double(200) / double(1500), double(100) / double(100));
    // 100: h2dBytes; 100:d2hBytes; 300:h2dTimeDuration; 400:d2hTimeDuration
    hmm::OckHmmTrafficStatisticsInfo correctInfo1(100U, 100U, double(100) / double(300), double(100) / double(400));
    EXPECT_EQ(*(statMgr->PickUp(0, maxGapMilliSeconds)), correctInfo0);
    EXPECT_EQ(*(statMgr->PickUp(1, maxGapMilliSeconds)), correctInfo1);
}

TEST_F(TestOckTaskStatisticsMgr, multi_device_gap_inside_same_device)
{
    AddData(0, OckMemoryCopyKind::HOST_TO_DEVICE, 0U, 600U);
    AddData(0, OckMemoryCopyKind::DEVICE_TO_HOST, 400U, 500U);
    AddData(1, OckMemoryCopyKind::HOST_TO_DEVICE, 1400U, 1500U);
    AddData(0, OckMemoryCopyKind::HOST_TO_DEVICE, 600U, 1500U);
    AddData(1, OckMemoryCopyKind::DEVICE_TO_HOST, 2600U, 3100U);
    AddData(1, OckMemoryCopyKind::HOST_TO_DEVICE, 3000U, 3300U);
    // 200: h2dBytes; 100:d2hBytes; 1500:h2dTimeDuration; 100:d2hTimeDuration
    hmm::OckHmmTrafficStatisticsInfo correctInfo0(200U, 100U, double(200) / double(1500), double(100) / double(100));
    // 100: h2dBytes; 100:d2hBytes; 300:h2dTimeDuration; 500:d2hTimeDuration
    hmm::OckHmmTrafficStatisticsInfo correctInfo1(100U, 100U, double(100) / double(300), double(100) / double(500));
    EXPECT_EQ(*(statMgr->PickUp(0, maxGapMilliSeconds)), correctInfo0);
    EXPECT_EQ(*(statMgr->PickUp(1, maxGapMilliSeconds)), correctInfo1);
}

}  // namespace test
}  // namespace acladapter
}  // namespace ock