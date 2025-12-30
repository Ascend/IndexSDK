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

#include <numeric>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <sys/time.h>
#include <gtest/gtest.h>
#include <limits>
#include <chrono>
#include <bitset>
#include "ock/hcps/algo/OckBitSet.h"
#include "ock/utils/StrUtils.h"

namespace ock {
namespace hcps {
namespace algo {
class TestOckBitSet : public testing::Test {
public:
    template <typename _StdBitSetT, typename _OckBitSetT>
    void ExpectSame(const _StdBitSetT &lhs, const _OckBitSetT &rhs)
    {
        ASSERT_EQ(lhs.size(), rhs.Size());
        for (uint64_t i = 0; i < lhs.size(); ++i) {
            EXPECT_EQ(lhs[i], rhs[i]);
        }
        EXPECT_EQ(lhs.count(), rhs.Count());
    }
    template <typename DataTemp, uint64_t DimSizeData>
    void TestBasicOperator(void)
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
TEST_F(TestOckBitSet, set_unset)
{
    TestBasicOperator<uint8_t, 2U>();
    TestBasicOperator<uint8_t, 4U>();
    TestBasicOperator<uint8_t, 7U>();
    TestBasicOperator<uint8_t, 15U>();
    TestBasicOperator<uint8_t, 100U>();
    TestBasicOperator<uint8_t, 128U>();
    TestBasicOperator<uint8_t, 256U>();
    TestBasicOperator<uint16_t, 4U>();
    TestBasicOperator<uint16_t, 10U>();
    TestBasicOperator<uint16_t, 256U>();
    TestBasicOperator<uint32_t, 17U>();
    TestBasicOperator<uint32_t, 128U>();
}
TEST_F(TestOckBitSet, hashValue_and_compare)
{
    OckBitSet<2048ULL, 256UL> queryData;
    OckBitSet<2048ULL, 256UL> leftData;
    queryData.Set(0UL);
    queryData.Set(1UL);
    queryData.Set(2UL);
    queryData.Set(2047UL);
    queryData.Set(2046UL);

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
TEST_F(TestOckBitSet, to_string)
{
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
}  // namespace algo
}  // namespace hcps
}  // namespace ock
