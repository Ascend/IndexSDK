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


#include <cstring>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include <chrono>
#include <bitset>
#include "secodeFuzz.h"
#include "ock/hcps/algo/OckBitSet.h"
#include "ock/utils/StrUtils.h"

namespace ock {
namespace hcps {
namespace algo {
static constexpr size_t DF_FUZZ_EXEC_COUNT = 300000000;
static constexpr size_t DF_FUZZ_EXEC_SECOND = 10800;
class FuzzTestOckBitSet : public testing::Test {
public:
    void SetUp(void) override
    {
        DT_Set_Running_Time_Second(DF_FUZZ_EXEC_SECOND);
        DT_Enable_Leak_Check(0, 0);
        DT_Set_Report_Path("/home/pjy/vsa/tests/fuzz/build/report");
    }

    template <typename _StdBitSetT, typename _OckBitSetT>
    void ExpectSame(const _StdBitSetT &lhs, const _OckBitSetT &rhs)
    {
        ASSERT_EQ(lhs.size(), rhs.Size());
        for (uint64_t i = 0; i < lhs.size(); ++i) {
            EXPECT_EQ(lhs[i], rhs[i]);
        }
        EXPECT_EQ(lhs.count(), rhs.Count());
    }
    template <typename DataTemp, uint64_t DimSizeData> void TestBasicOperator(void)
    {
        std::bitset<DimSizeData * sizeof(DataTemp) * __CHAR_BIT__> bitSet;
        OckBitSet<DimSizeData * sizeof(DataTemp) * __CHAR_BIT__, DimSizeData> ockBitSet;
        ExpectSame(bitSet, ockBitSet);
        uint64_t hashValue = 0;
        for (uint64_t i = 0; i < bitSet.size(); ++i) {
            bitSet.set(i);
            ockBitSet.SetExt(i % DimSizeData, i / DimSizeData);
            ExpectSame(bitSet, ockBitSet);
            EXPECT_EQ(i % DimSizeData + 1U, ockBitSet.CountExt(i / DimSizeData));
            bitSet.set(i, false);
            ockBitSet.Set(i, false);
            ExpectSame(bitSet, ockBitSet);
            EXPECT_EQ(i % DimSizeData, ockBitSet.CountExt(i / DimSizeData));
            bitSet.set(i);
            ockBitSet.Set(i);
            ExpectSame(bitSet, ockBitSet);
            EXPECT_EQ(i % DimSizeData + 1U, ockBitSet.CountExt(i / DimSizeData));
            hashValue += (1UL << (sizeof(DataTemp) * __CHAR_BIT__ - 1UL - i / DimSizeData));
            EXPECT_EQ(hashValue, ockBitSet.HashValue());
        }
        bitSet.reset();
        ockBitSet.UnSetAll();
        ExpectSame(bitSet, ockBitSet);
    }
};
TEST_F(FuzzTestOckBitSet, set_unset)
{
    std::string name = "set_unset";
    uint32_t seed = 0;
    DT_FUZZ_START(seed, 1000U, const_cast<char *>(name.c_str()), 0)
    {
        TestBasicOperator<uint8_t, 2U>();
        TestBasicOperator<uint8_t, 128U>();
        TestBasicOperator<uint16_t, 256U>();
        TestBasicOperator<uint32_t, 17U>();
        TestBasicOperator<uint32_t, 128U>();
    }
    DT_FUZZ_END()
}
TEST_F(FuzzTestOckBitSet, hashValue_and_compare)
{
    std::string name = "hashValue_and_compare";
    uint32_t seed = 0;
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, const_cast<char *>(name.c_str()), 0)
    {
        s32 bitPos =
                *(s32 *)DT_SetGetNumberRange(&g_Element[0], 3U, 2047U, 2047U);
        OckBitSet<2048ULL, 256UL> queryData;
        OckBitSet<2048ULL, 256UL> leftData;
        queryData.Set(0UL);
        queryData.Set(1UL);
        queryData.Set(2UL);
        queryData.Set(bitPos);

        leftData.Set(1UL);
        EXPECT_EQ(queryData, queryData);
        EXPECT_EQ(0, queryData.Compare(queryData));
        EXPECT_LT(0, queryData.Compare(leftData));
        EXPECT_GT(0, leftData.Compare(queryData));
        EXPECT_EQ(leftData, leftData);
        EXPECT_GT(queryData, leftData);
        EXPECT_LT(leftData, queryData);
        EXPECT_GE(queryData, leftData);
        EXPECT_LE(leftData, queryData);
    }
    DT_FUZZ_END()
}
TEST_F(FuzzTestOckBitSet, to_string)
{
    std::string name = "to_string";
    uint32_t seed = 0;
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, const_cast<char *>(name.c_str()), 0)
    {
        s32 bitPos =
                *(s32 *)DT_SetGetNumberRange(&g_Element[0], 200U, 200U, 300U);
        OckBaseBitSet<12U, 3U> dataA("100011001111");
        EXPECT_EQ(utils::ToString(dataA), "100011001111");
        OckBaseBitSet<12U, 3U> dataB("010011001111");
        EXPECT_EQ(utils::ToString(dataB), "010011001111");
        OckBaseBitSet<12U, 3U> dataC("100000000000");
        EXPECT_EQ(utils::ToString(dataC), "100000000000");
        EXPECT_TRUE(dataA.At(0ULL));
        EXPECT_FALSE(dataB.At(0ULL));
        EXPECT_TRUE(dataC.At(0ULL));
        EXPECT_TRUE(dataA.At(11ULL));
        EXPECT_TRUE(dataB.At(11ULL));
        EXPECT_FALSE(dataC.At(11ULL));
        OckBaseBitSet<12U, 3U> middle("101010101010");
        EXPECT_FALSE(middle.At(1ULL));
        EXPECT_FALSE(middle.At(3ULL));
        EXPECT_FALSE(middle.At(5ULL));
        EXPECT_FALSE(middle.At(7ULL));
        EXPECT_FALSE(middle.At(9ULL));
        EXPECT_FALSE(middle.At(11ULL));
        EXPECT_TRUE(middle.At(0ULL));
        EXPECT_TRUE(middle.At(2ULL));
        EXPECT_TRUE(middle.At(4ULL));
        EXPECT_TRUE(middle.At(6ULL));
        EXPECT_TRUE(middle.At(8ULL));
        EXPECT_TRUE(middle.At(10ULL));
    }
    DT_FUZZ_END()
}
} // namespace algo
} // namespace hcps
} // namespace ock
