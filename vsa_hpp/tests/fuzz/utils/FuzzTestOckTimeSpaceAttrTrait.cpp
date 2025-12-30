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
#include "secodeFuzz.h"
#include "ock/vsa/attr/OckTimeSpaceAttrTrait.h"

namespace ock {
namespace vsa {
namespace attr {
namespace test {
static constexpr size_t DF_FUZZ_EXEC_COUNT = 300000000;
static constexpr size_t DF_FUZZ_EXEC_SECOND = 10800;
class FuzzTestOckTimeSpaceAttrTrait : public testing::Test {
public:
    explicit FuzzTestOckTimeSpaceAttrTrait(uint32_t attrNum = 256UL, uint32_t maxTokenNumber = 1024UL)
        : attrNum(attrNum), maxTokenNumber(maxTokenNumber), tsTraits(OckTimeSpaceAttrTrait(maxTokenNumber))
    {}

    void SetUp() override
    {
        DT_Set_Running_Time_Second(DF_FUZZ_EXEC_SECOND);
        DT_Enable_Leak_Check(0, 0);
        DT_Set_Report_Path("/home/pjy/vsa/tests/fuzz/build/report");
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

TEST_F(FuzzTestOckTimeSpaceAttrTrait, add_time_space_attr_correctly)
{
    std::string name = "add_time_space_attr_correctly";
    uint32_t seed = 0;
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, const_cast<char *>(name.c_str()), 0)
    {
        s32 maxTime =
                *(s32 *)DT_SetGetNumberRange(&g_Element[0], 200U, 200U, 300U);
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
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckTimeSpaceAttrTrait, add_time_space_trait_correctly)
{
    std::string name = "add_time_space_trait_correctly";
    uint32_t seed = 0;
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, const_cast<char *>(name.c_str()), 0)
    {
        s32 maxTime =
                *(s32 *)DT_SetGetNumberRange(&g_Element[0], 200U, 200U, 300U);
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
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckTimeSpaceAttrTrait, initFrom_successfully)
{
    std::string name = "initFrom_successfully";
    uint32_t seed = 0;
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, const_cast<char *>(name.c_str()), 0)
    {
        OckTimeSpaceAttrTrait otherTraits = OckTimeSpaceAttrTrait::InitFrom(tsTraits);
        EXPECT_EQ(otherTraits.maxTokenNumber, maxTokenNumber);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckTimeSpaceAttrTrait, in_successfully)
{
    std::string name = "initFrom_successfully";
    uint32_t seed = 0;
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, const_cast<char *>(name.c_str()), 0)
    {
        s32 maxTime =
                *(s32 *)DT_SetGetNumberRange(&g_Element[0], 200U, 200U, 300U);
        s32 maxToken =
                *(s32 *)DT_SetGetNumberRange(&g_Element[1], 200U, 200U, 500U);
        std::vector<int32_t> timeVec;
        std::vector<uint32_t> tokenVec;
        OckTimeSpaceAttrTrait specialTraits(maxTokenNumber);
        InitRangeVec<int32_t>(-100L, maxTime, timeVec, 500L);
        InitRangeVec<uint32_t>(100UL, maxToken, tokenVec, 500UL);
        InitTimeSpaceAttrs(timeVec, tokenVec);
        InitTimeSpaceTraits(specialTraits);

        EXPECT_EQ(specialTraits.In(OckTimeSpaceAttr(200L, 200UL)), true);
        EXPECT_EQ(specialTraits.In(OckTimeSpaceAttr(-100L, 200UL)), true);
        EXPECT_EQ(specialTraits.In(OckTimeSpaceAttr(maxTime, maxToken + 1UL)), false);
        EXPECT_EQ(specialTraits.In(OckTimeSpaceAttr(maxTime + 1UL, maxToken)), false);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckTimeSpaceAttrTrait, intersect_successfully)
{
    std::string name = "intersect_successfully";
    uint32_t seed = 0;
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, const_cast<char *>(name.c_str()), 0)
    {
        s32 maxTime =
                *(s32 *)DT_SetGetNumberRange(&g_Element[0], 200U, 200U, 300U);
        s32 maxToken =
                *(s32 *)DT_SetGetNumberRange(&g_Element[1], 300U, 500U, 500U);
        std::vector<int32_t> timeVec;
        std::vector<uint32_t> tokenVec;
        OckTimeSpaceAttrTrait specialTraits(maxTokenNumber);
        InitRangeVec<int32_t>(-100L, maxTime, timeVec, 500L);
        InitRangeVec<uint32_t>(100UL, maxToken, tokenVec, 500UL);
        InitTimeSpaceAttrs(timeVec, tokenVec);
        InitTimeSpaceTraits(specialTraits);

        OckTimeSpaceAttrTrait traitsA(maxTokenNumber);
        traitsA.Add(OckTimeSpaceAttr(maxTime + 1L, maxToken + 1L));
        EXPECT_EQ(specialTraits.Intersect(traitsA), false);
        traitsA.Add(OckTimeSpaceAttr(100L, 80UL));
        EXPECT_EQ(specialTraits.Intersect(traitsA), false);

        traitsA.Add(OckTimeSpaceAttr(0L, 100UL));
        EXPECT_EQ(specialTraits.Intersect(traitsA), true);
        traitsA.Add(OckTimeSpaceAttr(100L, 500UL));
        EXPECT_EQ(specialTraits.Intersect(traitsA), true);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckTimeSpaceAttrTrait, calc_coverage_rate_correctly)
{
    std::string name = "calc_coverage_rate_correctly";
    uint32_t seed = 0;

    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, const_cast<char *>(name.c_str()), 0)
    {
        s32 maxTime =
                *(s32 *)DT_SetGetNumberRange(&g_Element[0], 200U, 200U, 300U);
        std::vector<int32_t> timeVec;
        std::vector<uint32_t> tokenVec;
        OckTimeSpaceAttrTrait traitsA(maxTokenNumber);
        InitRangeVec<int32_t>(-200L, 200L, timeVec, 500U);
        InitRangeVec<uint32_t>(101UL, 200UL, tokenVec, 500U);
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
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckTimeSpaceAttrTrait, copy_construct_successfully)
{
    std::string name = "copy_construct_successfully";
    uint32_t seed = 0;
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, const_cast<char *>(name.c_str()), 0)
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
    DT_FUZZ_END()
}
} // namespace test
} // namespace attr
} // namespace vsa
} // namespace ock