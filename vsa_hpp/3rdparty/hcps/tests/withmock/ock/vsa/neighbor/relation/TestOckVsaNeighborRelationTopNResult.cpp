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

#include <random>
#include <vector>
#include <algorithm>
#include <fstream>
#include <gtest/gtest.h>
#include "ock/vsa/neighbor/base/OckVsaNeighborRelationTopNResult.h"
#include "ock/acladapter/utils/OckAscendFp16.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace relation {
namespace test {
class TestOckVsaNeighborRelationTopNResult : public testing::Test {
public:
    explicit TestOckVsaNeighborRelationTopNResult()
    {}

    void SetUp() override
    {}

    void TearDown() override
    {}
};

TEST_F(TestOckVsaNeighborRelationTopNResult, make_neighbor_relation_topn_result)
{
    // 测试多个组且大小不同的场景
    uint32_t topN = 6;
    uint32_t topNlabels[] = {1, 2, 3, 4, 5, 6};
    OckFloat16 topNdistance[topN];
    for (uint32_t i = 0; i < topN; ++i) {
        topNdistance[i] = acladapter::OckAscendFp16::FloatToFp16(static_cast<float>(i + 1));
    }
    std::deque<uint32_t> validateRowCountVec = { 2, 3, 4 };
    size_t n = validateRowCountVec.size();
    // 调用被测试函数
    std::shared_ptr<relation::OckVsaNeighborRelationTopNResult> result =
        MakeOckVsaNeighborRelationTopNResult(topNlabels, topNdistance, topN, validateRowCountVec);

    // 验证结果
    EXPECT_EQ(result->grpTopIds.size(), validateRowCountVec.size());
    EXPECT_EQ(result->grpTopDistances.size(), validateRowCountVec.size());

    // 检查每个组的结果
    std::vector<uint32_t> expectedGroup0Ids = { 1 };
    std::vector<float> expectedGroup0Distances = { 1.0f };
    EXPECT_EQ(result->grpTopIds[0], expectedGroup0Ids);
    EXPECT_EQ(result->grpTopDistances[0], expectedGroup0Distances);

    std::vector<uint32_t> expectedGroup1Ids = { 0, 1, 2 };
    std::vector<float> expectedGroup1Distances = { 2.0f, 3.0f, 4.0f };
    EXPECT_EQ(result->grpTopIds[1], expectedGroup1Ids);
    EXPECT_EQ(result->grpTopDistances[1], expectedGroup1Distances);

    std::vector<uint32_t> expectedGroup2Ids = { 0, 1 };
    std::vector<float> expectedGroup2Distances = { 5.0f, 6.0f };
    EXPECT_EQ(result->grpTopIds[n - 1], expectedGroup2Ids);
    EXPECT_EQ(result->grpTopDistances[n - 1], expectedGroup2Distances);
}

TEST_F(TestOckVsaNeighborRelationTopNResult, make_neighbor_relation_topn_result2)
{
    // 测试多个组且大小相同的场景
    std::deque<uint32_t> validateRowCountVec = { 3, 3, 3 };
    uint32_t topN = 6;
    uint32_t topNlabels[] = {1, 2, 3, 4, 5, 6};
    OckFloat16 topNdistance[topN];
    for (uint32_t i = 0; i < topN; ++i) {
        topNdistance[i] = acladapter::OckAscendFp16::FloatToFp16(static_cast<float>(i + 1));
    }
    size_t n = validateRowCountVec.size();
    // 调用被测试函数
    std::shared_ptr<relation::OckVsaNeighborRelationTopNResult> result =
            MakeOckVsaNeighborRelationTopNResult(topNlabels, topNdistance, topN, validateRowCountVec);
    // 验证结果
    std::vector<uint32_t> expectedGroup0Ids = { 1, 2 };
    std::vector<float> expectedGroup0Distances = { 1.0f, 2.0f };
    EXPECT_EQ(result->grpTopIds[0], expectedGroup0Ids);
    EXPECT_EQ(result->grpTopDistances[0], expectedGroup0Distances);

    std::vector<uint32_t> expectedGroup1Ids = { 0, 1, 2 };
    std::vector<float> expectedGroup1Distances = { 3.0f, 4.0f, 5.0f };
    EXPECT_EQ(result->grpTopIds[1], expectedGroup1Ids);
    EXPECT_EQ(result->grpTopDistances[1], expectedGroup1Distances);

    std::vector<uint32_t> expectedGroup2Ids = { 0 };
    std::vector<float> expectedGroup2Distances = { 6.0f };
    EXPECT_EQ(result->grpTopIds[n - 1], expectedGroup2Ids);
    EXPECT_EQ(result->grpTopDistances[n - 1], expectedGroup2Distances);
}

TEST_F(TestOckVsaNeighborRelationTopNResult, make_neighbor_relation_topn_result3)
{
    // 测试单个组的场景
    std::deque<uint32_t> validateRowCountVec = { 5 };
    uint32_t topN = 4;
    uint32_t topNlabels[] = {1, 2, 3, 4 };
    OckFloat16 topNdistance[topN];
    for (uint32_t i = 0; i < topN; ++i) {
        topNdistance[i] = acladapter::OckAscendFp16::FloatToFp16(static_cast<float>(i + 1));
    }
    // 调用被测试函数
    std::shared_ptr<relation::OckVsaNeighborRelationTopNResult> result =
            MakeOckVsaNeighborRelationTopNResult(topNlabels, topNdistance, topN, validateRowCountVec);
    // 验证结果
    std::vector<uint32_t> expectedGroupIds = { 1, 2, 3, 4 };
    std::vector<float> expectedGroupDistances = { 1.0f, 2.0f, 3.0f, 4.0f };
    EXPECT_EQ(result->grpTopIds[0], expectedGroupIds);
    EXPECT_EQ(result->grpTopDistances[0], expectedGroupDistances);
}
} // namespace test
} // namespace relation
} // namespace neighbor
} // namespace vsa
} // namespace ock