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
#include "OckTestUtils.h"
namespace ock {
namespace vsa {
namespace neighbor {
namespace test {
const float ERRORRANGE = 1e-4;
class TestOckTestUtils : public testing::Test {
public:
    std::vector<int64_t> labelsHCP;
    std::vector<int64_t> labelsMindX;
    std::vector<float> distancesHCP;
    std::vector<float> distancesMindX;

    std::unordered_map<int64_t, uint32_t> labelsHCPSet;

    float minNNCosValue = 0.0f;
    float recallRate;
};

TEST_F(TestOckTestUtils, recall_all_unorder)
{
    labelsHCP = { 1, 2, 3, 100, 101, 102, 103 };
    labelsMindX = { 1, 3, 2, 102, 101, 100, 103 };
    distancesHCP = { 1.0f, 0.9f, 0.9f, 0.8f, 0.8f, 0.8f, 0.0f };
    distancesMindX = { 1.0f, 0.9f, 0.9f, 0.8f, 0.8f, 0.8f, 0.0f };
    labelsHCPSet = OckTestUtils::VecToMap<int64_t, uint32_t>(labelsHCP);
    recallRate = OckTestUtils::CalcRecallWithDist(labelsHCPSet, labelsMindX, distancesHCP, distancesMindX, ERRORRANGE,
        minNNCosValue);
    EXPECT_EQ(recallRate, 1.0f);
}

TEST_F(TestOckTestUtils, recall_all_min_distance)
{
    labelsHCP = { 1, 2, 3, 100, 101, 102 };
    labelsMindX = { 1, 3, 2, 102, 103, 104 };
    distancesHCP = { 1.0f, 0.9f, 0.9f, 0.8f, 0.8f, 0.8f };
    distancesMindX = { 1.0f, 0.9f, 0.9f, 0.8f, 0.8f, 0.8f };
    labelsHCPSet = OckTestUtils::VecToMap<int64_t, uint32_t>(labelsHCP);
    recallRate = OckTestUtils::CalcRecallWithDist(labelsHCPSet, labelsMindX, distancesHCP, distancesMindX, ERRORRANGE,
        minNNCosValue);

    EXPECT_EQ(recallRate, 1.0f);
}

TEST_F(TestOckTestUtils, recall_all_minNNCosValue)
{
    labelsHCP = { 1, 2, 3, 100, 101, 102 };
    labelsMindX = { 1, 3, 2, 100, 103, 104 };
    distancesHCP = { 1.0f, 0.9f, 0.9f, 0.8f, -0.5f, -0.6f };
    distancesMindX = { 1.0f, 0.9f, 0.9f, 0.8f, -0.8f, -0.8f };
    labelsHCPSet = OckTestUtils::VecToMap<int64_t, uint32_t>(labelsHCP);
    recallRate = OckTestUtils::CalcRecallWithDist(labelsHCPSet, labelsMindX, distancesHCP, distancesMindX, ERRORRANGE,
        minNNCosValue);

    EXPECT_EQ(recallRate, 1.0f);
}

TEST_F(TestOckTestUtils, recall_part)
{
    labelsHCP = { 1, 2, 3, 101, 102, 1000, 1001 };
    labelsMindX = { 1, 3, 2, 100, 101, 1002, 1003 };
    distancesHCP = { 1.0f, 0.9f, 0.9f, 0.5f, 0.4f, -0.5f, -0.6f };
    distancesMindX = { 1.0f, 0.9f, 0.9f, 0.8f, 0.4f, -0.8f, -0.8f };
    labelsHCPSet = OckTestUtils::VecToMap<int64_t, uint32_t>(labelsHCP);
    recallRate = OckTestUtils::CalcRecallWithDist(labelsHCPSet, labelsMindX, distancesHCP, distancesMindX, ERRORRANGE,
        minNNCosValue);

    EXPECT_EQ(recallRate, 1.0f - 1.0f / labelsMindX.size());
}
} // namespace test
} // namespace neighbor
} // namespace vsa
} // namespace ock