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
#include "ock/vsa/attr/OckKeyTrait.h"
#include "ock/vsa/attr/OckTimeSpaceAttrTrait.h"

namespace ock {
namespace vsa {
namespace attr {
namespace test {
class TestOckTimeSpaceAttrTrait : public testing::Test {
public:
    explicit TestOckTimeSpaceAttrTrait(uint32_t attrNum = 256UL, uint32_t maxTokenNumber = 1024UL)
        : attrNum(attrNum), maxTokenNumber(maxTokenNumber), tsTraits(OckTimeSpaceAttrTrait(maxTokenNumber))
    {}

    void SetUp() override
    {
        attrData.reserve(attrNum);
        std::vector<int32_t> timeAttrs;
        std::vector<uint32_t> tokenAttrs;
        InitRandVec<int32_t>(-1000L, 1000L, timeAttrs, attrNum);
        InitRandVec<uint32_t>(100UL, 500UL, tokenAttrs, attrNum);
        InitTimeSpaceAttrs(timeAttrs, tokenAttrs);
        InitTimeSpaceTraits(tsTraits);
    }

    void TearDown() override
    {
        attrData.clear();
    }

    // 左闭右闭区间[minValue, maxValue]
    template <typename DataTemp>
    void InitRandVec(DataTemp minValue, DataTemp maxValue, std::vector<DataTemp> &dataVec, uint32_t num)
    {
        // 确保数据在类型范围内
        EXPECT_TRUE(minValue >= std::numeric_limits<DataTemp>::min());
        EXPECT_TRUE(maxValue <= std::numeric_limits<DataTemp>::max());
        std::random_device seed;
        std::ranlux48 engine(seed());
        std::uniform_int_distribution<> distrib(minValue, maxValue);

        for (uint32_t i = 0; i < num; ++i) {
            dataVec.emplace_back(distrib(engine));
        }
    }

    template <typename DataTemp>
    void InitRangeVec(DataTemp minValue, DataTemp maxValue, std::vector<DataTemp> &dataVec, DataTemp num)
    {
        DataTemp rangeLen = maxValue - minValue + 1;
        EXPECT_TRUE(minValue >= std::numeric_limits<DataTemp>::min());
        EXPECT_TRUE(maxValue <= std::numeric_limits<DataTemp>::max());
        EXPECT_TRUE(rangeLen <= num);

        for (DataTemp i = 0; i < num; ++i) {
            dataVec.emplace_back((i % rangeLen) + minValue);
        }
    }

    void InitTimeSpaceAttrs(std::vector<int32_t> timeAttrs, std::vector<uint32_t> tokenAttrs)
    {
        attrData.clear();
        for (uint32_t i = 0; i < timeAttrs.size(); i++) {
            attrData.push_back(OckTimeSpaceAttr(timeAttrs[i], tokenAttrs[i]));
        }
    }

    void InitTimeSpaceTraits(OckTimeSpaceAttrTrait &traits)
    {
        for (uint32_t i = 0; i < attrData.size(); i++) {
            traits.Add(attrData[i]);
        }
    }

    uint32_t attrNum;
    uint32_t maxTokenNumber{ 1024 };
    std::vector<OckTimeSpaceAttr> attrData;
    OckTimeSpaceAttrTrait tsTraits{ maxTokenNumber };
};

TEST_F(TestOckTimeSpaceAttrTrait, add_time_space_attr_correctly)
{
    int32_t upTime = 1001;
    int32_t belowTime = -1001;
    uint32_t upTokenId = 501UL;
    uint32_t belowTokenId = 0UL;

    tsTraits.Add(OckTimeSpaceAttr(upTime, upTokenId));
    EXPECT_EQ(tsTraits.maxTime, upTime);
    EXPECT_EQ(tsTraits.maxTokenId, upTokenId);
    EXPECT_EQ(tsTraits.bitSet[upTokenId], 1);

    tsTraits.Add(OckTimeSpaceAttr(belowTime, belowTokenId));
    EXPECT_EQ(tsTraits.minTime, belowTime);
    EXPECT_EQ(tsTraits.minTokenId, belowTokenId);
    EXPECT_EQ(tsTraits.bitSet[belowTokenId], 1);
}

TEST_F(TestOckTimeSpaceAttrTrait, add_time_space_trait_correctly)
{
    OckTimeSpaceAttrTrait otherTraitsA(maxTokenNumber);
    OckTimeSpaceAttrTrait otherTraitsB(maxTokenNumber);
    int32_t curMinTime = tsTraits.minTime;
    uint32_t curMinTokenId = tsTraits.minTokenId;
    int32_t upTime = 2000;
    int32_t belowTime = -1001;
    uint32_t upTokenId = 600UL;
    uint32_t belowTokenId = 0UL;

    otherTraitsA.Add(OckTimeSpaceAttr(upTime, upTokenId));
    tsTraits.Add(otherTraitsA);
    EXPECT_EQ(tsTraits.maxTime, upTime);
    EXPECT_EQ(tsTraits.maxTokenId, upTokenId);
    EXPECT_EQ(tsTraits.minTime, curMinTime);
    EXPECT_EQ(tsTraits.minTokenId, curMinTokenId);
    EXPECT_EQ(tsTraits.bitSet[upTokenId], 1);

    otherTraitsB.Add(OckTimeSpaceAttr(belowTime, belowTokenId));
    tsTraits.Add(otherTraitsB);
    EXPECT_EQ(tsTraits.maxTime, upTime);
    EXPECT_EQ(tsTraits.maxTokenId, upTokenId);
    EXPECT_EQ(tsTraits.minTime, belowTime);
    EXPECT_EQ(tsTraits.minTokenId, belowTokenId);
    EXPECT_EQ(tsTraits.bitSet[belowTokenId], 1);
}

TEST_F(TestOckTimeSpaceAttrTrait, initFrom_successfully)
{
    OckTimeSpaceAttrTrait otherTraits = OckTimeSpaceAttrTrait::InitFrom(tsTraits);
    EXPECT_EQ(otherTraits.maxTokenNumber, maxTokenNumber);
}

TEST_F(TestOckTimeSpaceAttrTrait, in_successfully)
{
    std::vector<int32_t> timeVec;
    std::vector<uint32_t> tokenVec;
    OckTimeSpaceAttrTrait specialTraits(maxTokenNumber);
    InitRangeVec<int32_t>(-100L, 200L, timeVec, 500L);
    InitRangeVec<uint32_t>(100UL, 500UL, tokenVec, 500UL);
    InitTimeSpaceAttrs(timeVec, tokenVec);
    InitTimeSpaceTraits(specialTraits);

    EXPECT_EQ(specialTraits.In(OckTimeSpaceAttr(200L, 200UL)), true);
    EXPECT_EQ(specialTraits.In(OckTimeSpaceAttr(-100L, 200UL)), true);
    EXPECT_EQ(specialTraits.In(OckTimeSpaceAttr(500L, 800UL)), false);
    EXPECT_EQ(specialTraits.In(OckTimeSpaceAttr(300L, 200UL)), false);
}

TEST_F(TestOckTimeSpaceAttrTrait, intersect_successfully)
{
    std::vector<int32_t> timeVec;
    std::vector<uint32_t> tokenVec;
    OckTimeSpaceAttrTrait specialTraits(maxTokenNumber);
    InitRangeVec<int32_t>(-100L, 200L, timeVec, 500L);
    InitRangeVec<uint32_t>(100UL, 500UL, tokenVec, 500UL);
    InitTimeSpaceAttrs(timeVec, tokenVec);
    InitTimeSpaceTraits(specialTraits);

    OckTimeSpaceAttrTrait traitsA(maxTokenNumber);
    traitsA.Add(OckTimeSpaceAttr(201L, 501UL));
    EXPECT_EQ(specialTraits.Intersect(traitsA), false);
    traitsA.Add(OckTimeSpaceAttr(100L, 80UL));
    EXPECT_EQ(specialTraits.Intersect(traitsA), false);

    traitsA.Add(OckTimeSpaceAttr(0L, 100UL));
    EXPECT_EQ(specialTraits.Intersect(traitsA), true);
    traitsA.Add(OckTimeSpaceAttr(100L, 500UL));
    EXPECT_EQ(specialTraits.Intersect(traitsA), true);
}

TEST_F(TestOckTimeSpaceAttrTrait, calc_coverage_rate_correctly)
{
    std::vector<int32_t> timeVec;
    std::vector<uint32_t> tokenVec;
    OckTimeSpaceAttrTrait traitsA(maxTokenNumber);
    InitRangeVec<int32_t>(-200L, 200L, timeVec, 500UL);
    InitRangeVec<uint32_t>(101UL, 200UL, tokenVec, 500UL);
    InitTimeSpaceAttrs(timeVec, tokenVec);
    InitTimeSpaceTraits(traitsA);
    EXPECT_EQ(traitsA.CoverageRate(traitsA), 1.0L);

    double aCoverB = 0.0025;
    double bCoverA = 0.5;
    OckTimeSpaceAttrTrait traitsB(maxTokenNumber);
    traitsB.Add(OckTimeSpaceAttr(0L, 0UL));
    traitsB.Add(OckTimeSpaceAttr(100L, 200UL));
    EXPECT_EQ(traitsA.CoverageRate(traitsB), aCoverB);
    EXPECT_EQ(traitsB.CoverageRate(traitsA), bCoverA);
}

TEST_F(TestOckTimeSpaceAttrTrait, copy_construct_successfully)
{
    auto traitsA(tsTraits);
    auto traitsB = tsTraits;
    auto traitsC = OckTimeSpaceAttrTrait{ 10UL };
    auto traitsD = OckTimeSpaceAttrTrait(10UL);
    auto traitsE(OckTimeSpaceAttrTrait(10UL));

    EXPECT_EQ(traitsA, tsTraits);
    EXPECT_EQ(traitsB, tsTraits);
    EXPECT_EQ(traitsC, traitsD);
    EXPECT_EQ(traitsE, OckTimeSpaceAttrTrait(10UL));
}
} // namespace test
} // namespace attr
} // namespace vsa
} // namespace ock