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
#include <cassert>
#include <string>
#include "ock/utils/StrUtils.h"
#include "ock/hcps/algo/OckTopNQueue.h"
#include "ock/acladapter/utils/OckAscendFp16.h"

namespace ock {
namespace hcps {
namespace algo {
class TestOckTopNQueue : public testing::Test {
public:
    void InitSortedData(uint32_t vecNum, std::vector<uint64_t> &idxVec, std::vector<OckFloat16> &distVec)
    {
        assert(idxVec.size() == 0);
        assert(distVec.size() == 0);
        for (int64_t i = vecNum - 1; i >= 0; i--) {
            idxVec.push_back(uint64_t(i));
            distVec.push_back(acladapter::OckAscendFp16::FloatToFp16(float(i) / 10UL));
        }
    }
    uint32_t topN{2U};

    uint64_t idx10 = 10UL;
    uint64_t idx1 = 1UL;
    uint64_t idx3 = 3UL;
    uint64_t idx4 = 4UL;
    uint64_t idx6 = 6UL;

    float dist01 = 0.1;
    float dist02 = 0.2;
    float dist005 = 0.05;
    float dist03 = 0.3;
    float dist04 = 0.4;
    float dist012 = 0.12;
    float dist019 = 0.19;
};
TEST_F(TestOckTopNQueue, top_one)
{
    topN = 1U;
    auto topMgr = OckTopNQueue<double, uint32_t, OckCompareAscAdapter<double, uint32_t>>::Create(topN);
    topMgr->AddData(idx10, dist01);  // id + distance
    topMgr->AddData(idx1, dist02);
    EXPECT_EQ(topMgr->Pop(), idx10);
    EXPECT_TRUE(topMgr->Empty());
    topMgr->AddData(idx1, dist02);
    topMgr->AddData(idx10, dist005);
    EXPECT_EQ(topMgr->Pop(), idx10);
    EXPECT_TRUE(topMgr->Empty());
}
TEST_F(TestOckTopNQueue, top_two)
{
    topN = 2U;
    auto topMgr = OckTopNQueue<double, uint32_t, OckCompareAscAdapter<double, uint32_t>>::Create(topN);
    topMgr->AddData(idx10, dist01);
    topMgr->AddData(idx1, dist02);
    EXPECT_EQ(topMgr->Pop(), idx1);
    EXPECT_EQ(topMgr->Pop(), idx10);
    EXPECT_TRUE(topMgr->Empty());
    topMgr->AddData(idx1, dist02);
    topMgr->AddData(idx10, dist005);
    EXPECT_EQ(topMgr->Pop(), idx1);
    EXPECT_EQ(topMgr->Pop(), idx10);
    EXPECT_TRUE(topMgr->Empty());
}
TEST_F(TestOckTopNQueue, top_three)
{
    topN = 3U;
    auto topMgr = OckTopNQueue<double, uint32_t, OckCompareAscAdapter<double, uint32_t>>::Create(topN);
    topMgr->AddData(idx10, dist01);
    topMgr->AddData(idx1, dist02);
    EXPECT_EQ(topMgr->Pop(), idx1);
    EXPECT_EQ(topMgr->Pop(), idx10);
    EXPECT_TRUE(topMgr->Empty());
    topMgr->AddData(idx1, dist02);
    topMgr->AddData(idx3, dist03);
    topMgr->AddData(idx10, dist005);
    topMgr->AddData(idx4, dist02);
    EXPECT_EQ(topMgr->Pop(), idx1);
    EXPECT_EQ(topMgr->Pop(), idx4);
    EXPECT_EQ(topMgr->Pop(), idx10);
    EXPECT_TRUE(topMgr->Empty());
}
TEST_F(TestOckTopNQueue, multi_data)
{
    topN = 5U;
    auto topMgr = OckTopNQueue<double, uint32_t, OckCompareDescAdapter<double, uint32_t>>::Create(topN);
    topMgr->AddData(idx10, dist01);
    topMgr->AddData(idx1, dist02);
    topMgr->AddData(idx1, dist03);
    topMgr->AddData(idx3, dist04);
    topMgr->AddData(idx10, dist005);
    topMgr->AddData(idx4, dist02);
    topMgr->AddData(idx3, dist012);
    topMgr->AddData(idx6, dist019);
    auto ret = topMgr->PopAll();
    EXPECT_EQ("[{'idx':6,'dis':0.19},{'idx':1,'dis':0.2},{'idx':4,'dis':0.2},{'idx':1,'dis':0.3},{'idx':3,'dis':0.4}]",
        utils::ToString(*ret));
}
TEST_F(TestOckTopNQueue, multi_data_des)
{
    topN = 5U;
    auto topMgr = OckTopNQueue<double, uint32_t, OckCompareDescAdapter<double, uint32_t>>::Create(topN);
    topMgr->AddData(idx10, 0.9f);
    topMgr->AddData(idx1, 0.88f);
    topMgr->AddData(idx1, 0.7f);
    topMgr->AddData(idx3, 0.6f);
    topMgr->AddData(idx10, 0.5f);
    topMgr->AddData(idx4, 0.4f);
    topMgr->AddData(idx3, 0.3f);
    topMgr->AddData(idx6, 0.1f);
    topMgr->AddData(idx1, 0.88f);
    topMgr->AddData(idx1, 0.7f);
    topMgr->AddData(idx3, 0.6f);
    topMgr->AddData(idx10, 0.5f);
    topMgr->AddData(idx4, 0.4f);
    topMgr->AddData(idx3, 0.3f);
    topMgr->AddData(idx6, 0.1f);
    auto ret = topMgr->PopAll();
    EXPECT_EQ(
        "[{'idx':1,'dis':0.7},{'idx':1,'dis':0.7},{'idx':1,'dis':0.88},{'idx':1,'dis':0.88},{'idx':10,'dis':0.9}]",
        utils::ToString(*ret));
}
TEST_F(TestOckTopNQueue, multi_data_des_big)
{
    topN = 5U;
    auto topMgr = OckTopNQueue<double, uint32_t, OckCompareDescAdapter<double, uint32_t>>::Create(topN);
    topMgr->AddData(idx10, 1.0f);
    topMgr->AddData(idx1, 2.0f);
    topMgr->AddData(idx1, 3.0f);
    topMgr->AddData(idx3, 4.0f);
    topMgr->AddData(idx10, 5.0f);
    topMgr->AddData(idx4, 6.0f);
    topMgr->AddData(idx3, 7.0f);
    topMgr->AddData(idx6, 8.0f);
    topMgr->AddData(idx1, 1.0f);
    topMgr->AddData(idx1, 2.0f);
    topMgr->AddData(idx3, 3.0f);
    topMgr->AddData(idx10, 4.0f);
    topMgr->AddData(idx4, 5.0f);
    topMgr->AddData(idx3, 6.0f);
    topMgr->AddData(idx6, 7.0f);
    topMgr->AddData(idx6, 8.0f);
    auto ret = topMgr->PopAll();
    EXPECT_EQ("[{'idx':3,'dis':6},{'idx':3,'dis':7},{'idx':6,'dis':7},{'idx':6,'dis':8},{'idx':6,'dis':8}]",
        utils::ToString(*ret));
}
TEST_F(TestOckTopNQueue, multi_nodes)
{
    topN = 5U;
    uint topNumMulti = 2;
    using FloatNode = OckTopNNode<double, uint32_t>;
    std::vector<FloatNode> testNodes;
    for (uint i = 0; i < topN * topNumMulti; ++i) {
        testNodes.push_back(FloatNode(i, static_cast<double>(i)));
    }
    auto topMgr = OckTopNQueue<double, uint32_t, OckCompareDescAdapter<double, uint32_t>>::Create(topN);
    topMgr->AddNodes(testNodes);
    auto ret = topMgr->PopAll();
    EXPECT_EQ("[{'idx':5,'dis':5},{'idx':6,'dis':6},{'idx':7,'dis':7},{'idx':8,'dis':8},{'idx':9,'dis':9}]",
        utils::ToString(*ret));
}
TEST_F(TestOckTopNQueue, multi_data_des_getall)
{
    topN = 5U;
    auto topMgr = OckTopNQueue<double, uint32_t, OckCompareDescAdapter<double, uint32_t>>::Create(topN);
    topMgr->AddData(idx10, 1.0f);
    topMgr->AddData(idx1, 2.0f);
    topMgr->AddData(idx1, 3.0f);
    topMgr->AddData(idx3, 4.0f);
    topMgr->AddData(idx10, 5.0f);
    topMgr->AddData(idx4, 6.0f);
    topMgr->AddData(idx3, 7.0f);
    topMgr->AddData(idx6, 8.0f);
    topMgr->AddData(idx1, 1.0f);
    topMgr->AddData(idx1, 2.0f);
    topMgr->AddData(idx3, 3.0f);
    topMgr->AddData(idx10, 4.0f);
    topMgr->AddData(idx4, 5.0f);
    topMgr->AddData(idx3, 6.0f);
    topMgr->AddData(idx6, 7.0f);
    topMgr->AddData(idx6, 8.0f);
    auto ret = topMgr->GetAll();
    EXPECT_EQ("[{'idx':3,'dis':6},{'idx':6,'dis':7},{'idx':3,'dis':7},{'idx':6,'dis':8},{'idx':6,'dis':8}]",
              utils::ToString(ret));
}
}  // namespace algo
}  // namespace hcps
}  // namespace ock