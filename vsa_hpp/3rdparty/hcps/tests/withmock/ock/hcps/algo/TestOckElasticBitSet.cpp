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
#include "ock/hcps/algo/OckElasticBitSet.h"
#include "ock/utils/StrUtils.h"
#include "ock/hcps/algo/OckBitSet.h"

namespace ock {
namespace hcps {
namespace algo {
namespace {
const uint64_t DIM_SIZE = 256UL;
} // namespace
class TestOckElasticBitSet : public testing::Test {
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
    template <uint64_t _BitSizeT> void TestBasicOperator(void)
    {
        std::bitset<_BitSizeT> bitSet;
        OckElasticBitSet ockBitSet(_BitSizeT);
        ExpectSame(bitSet, ockBitSet);
        for (uint64_t i = 0; i < bitSet.size(); ++i) {
            bitSet.set(i);
            ockBitSet.Set(i);
            ExpectSame(bitSet, ockBitSet);
            EXPECT_EQ(i + 1UL, ockBitSet.Count());
            bitSet.set(i, false);
            ockBitSet.Set(i, false);
            ExpectSame(bitSet, ockBitSet);
            EXPECT_EQ(i, ockBitSet.Count());
            bitSet.set(i);
            ockBitSet.Set(i);
            ExpectSame(bitSet, ockBitSet);
        }
        bitSet.reset();
        ockBitSet.UnSetAll();
        ExpectSame(bitSet, ockBitSet);
    }
};
TEST_F(TestOckElasticBitSet, set_unset)
{
    TestBasicOperator<2U>();
    TestBasicOperator<4U>();
    TestBasicOperator<7U>();
    TestBasicOperator<15U>();
    TestBasicOperator<100U>();
    TestBasicOperator<128U>();
    TestBasicOperator<256U>();
}
TEST_F(TestOckElasticBitSet, toString)
{
    EXPECT_EQ(utils::ToString(OckElasticBitSet(3ULL)), "000");
    EXPECT_EQ(utils::ToString(OckElasticBitSet(13ULL)), "0000000000000");
    OckElasticBitSet bitSet(15ULL);
    bitSet.Set(3UL);
    bitSet.Set(10UL);
    EXPECT_EQ(utils::ToString(bitSet), "000100000010000");
}
TEST_F(TestOckElasticBitSet, HasSetBit)
{
    OckElasticBitSet bitSet(256ULL);
    bitSet.Set(3UL);
    bitSet.Set(129UL);
    EXPECT_FALSE(bitSet.HasSetBit(0ULL, 2ULL));     // 首段未覆盖场景
    EXPECT_TRUE(bitSet.HasSetBit(0ULL, 64ULL));     // 首段有覆盖场景
    EXPECT_FALSE(bitSet.HasSetBit(4ULL, 67ULL));    // 跨越首段场景
    EXPECT_TRUE(bitSet.HasSetBit(4ULL, 128ULL));    // 两段场景
    EXPECT_FALSE(bitSet.HasSetBit(64ULL, 64ULL));   // 完整1段场景
    EXPECT_TRUE(bitSet.HasSetBit(64ULL, 128ULL));   // 完整2段场景
    EXPECT_FALSE(bitSet.HasSetBit(130ULL, 125ULL)); // 包含尾段的多段场景
    EXPECT_FALSE(bitSet.HasSetBit(253ULL, 2ULL));   // 包含尾段单段场景
}
TEST_F(TestOckElasticBitSet, has_set_word_correctlly)
{
    OckElasticBitSet bitSet(256UL);
    bitSet.Set(23UL);
    bitSet.Set(189UL);
    EXPECT_TRUE(bitSet.HasSetWord(0UL, 1UL));
    EXPECT_TRUE(bitSet.HasSetWord(2UL, 3UL));
    EXPECT_FALSE(bitSet.HasSetWord(4UL, 3UL));
    EXPECT_FALSE(bitSet.HasSetWord(3UL, 4UL));
}
TEST_F(TestOckElasticBitSet, and_other_bitset_correctlly)
{
    OckElasticBitSet bitSetA(256UL);
    OckElasticBitSet bitSetB(256UL);
    bitSetA.Set(11UL);
    bitSetB.Set(22UL);
    bitSetA.AndWith(bitSetB);
    EXPECT_EQ(bitSetA.Count(), 0UL);

    bitSetA.SetAll();
    bitSetB.Set(11UL);
    bitSetA.AndWith(bitSetB);
    EXPECT_TRUE(bitSetA.At(11UL));
}
TEST_F(TestOckElasticBitSet, or_other_bitset_correctlly)
{
    OckElasticBitSet bitSetA(256UL);
    OckElasticBitSet bitSetB(256UL);
    OckElasticBitSet bitSetC(220UL);
    bitSetA.Set(11UL);
    bitSetB.Set(222UL);

    bitSetA.OrWith(bitSetB);
    bitSetC.OrWith(bitSetB);

    EXPECT_TRUE(bitSetA.At(11UL));
    EXPECT_TRUE(bitSetA.At(222UL));
    EXPECT_FALSE(bitSetA.At(0UL));
    EXPECT_FALSE(bitSetB.At(11UL));
    EXPECT_FALSE(bitSetC[122UL]);
    EXPECT_EQ(bitSetA.Count(), 2UL);
    EXPECT_EQ(bitSetC.Count(), 0UL);
}
TEST_F(TestOckElasticBitSet, intersect_set_count_correctlly)
{
    OckElasticBitSet bitSetA(256UL);
    OckElasticBitSet bitSetB(256UL);
    bitSetA.Set(1UL);
    bitSetA.Set(255UL);
    EXPECT_EQ(bitSetA.IntersectCount(bitSetB), 0UL);
    bitSetB.Set(64UL);
    EXPECT_EQ(bitSetA.IntersectCount(bitSetB), 0UL);

    bitSetB.SetAll();
    EXPECT_EQ(bitSetA.IntersectCount(bitSetB), bitSetB.IntersectCount(bitSetA));
    EXPECT_EQ(bitSetA.IntersectCount(bitSetB), 2UL);
}
TEST_F(TestOckElasticBitSet, intersect_correctlly)
{
    OckElasticBitSet bitSetA(256UL);
    OckElasticBitSet bitSetB(256UL);
    bitSetA.Set(25UL);
    bitSetA.Set(168UL);
    EXPECT_EQ(bitSetA.Intersect(bitSetB), bitSetA.IntersectCount(bitSetB) > 0);
    bitSetB.Set(37UL);
    EXPECT_EQ(bitSetA.Intersect(bitSetB), bitSetA.IntersectCount(bitSetB) > 0);
    bitSetB.SetAll();
    EXPECT_EQ(bitSetA.Intersect(bitSetB), bitSetA.IntersectCount(bitSetB) > 0);
}
TEST_F(TestOckElasticBitSet, set_all_correctlly)
{
    OckElasticBitSet bitSetA(256UL);
    OckElasticBitSet bitSetB(210UL);
    OckElasticBitSet bitSetC(100UL);
    bitSetA.SetAll();
    bitSetB.SetAll();
    bitSetC.SetAll();
 
    EXPECT_EQ(bitSetA.Count(), 256UL);
    EXPECT_EQ(bitSetB.Count(), 210UL);
    EXPECT_EQ(bitSetC.Count(), 100UL);
}
TEST_F(TestOckElasticBitSet, set_range_correctlly)
{
    OckElasticBitSet bitSetA(1024UL);
    // 左对齐
    bitSetA.SetRange(0UL, 137UL);
    EXPECT_EQ(bitSetA.Count(), 137UL);
    // 右对齐
    bitSetA.SetRange(177UL, 143UL);
    EXPECT_EQ(bitSetA.Count(), 280UL);

    // 无对齐
    bitSetA.SetRange(577UL, 190UL);
    EXPECT_EQ(bitSetA.Count(), 470UL);
    // 对齐
    bitSetA.SetRange(800UL, 128UL);
    EXPECT_EQ(bitSetA.Count(), 598UL);

    // 交叉
    bitSetA.SetRange(20UL, 300UL);
    EXPECT_EQ(bitSetA.Count(), 638UL);
}
TEST_F(TestOckElasticBitSet, unset_range_correctlly)
{
    OckElasticBitSet bitSetA(1024UL);
    bitSetA.SetAll();
    // 左对齐
    bitSetA.UnSetRange(0UL, 137UL);
    EXPECT_EQ(bitSetA.Count(), 887UL);
    // 右对齐
    bitSetA.UnSetRange(177UL, 143UL);
    EXPECT_EQ(bitSetA.Count(), 744UL);

    // 无对齐
    bitSetA.UnSetRange(577UL, 190UL);
    EXPECT_EQ(bitSetA.Count(), 554UL);
    // 对齐
    bitSetA.UnSetRange(800UL, 128UL);
    EXPECT_EQ(bitSetA.Count(), 426UL);

    // 交叉
    bitSetA.UnSetRange(20UL, 300UL);
    EXPECT_EQ(bitSetA.Count(), 386UL);
}
TEST_F(TestOckElasticBitSet, hash_value_zero)
{
    uint64_t zeroResult = 0;
    OckElasticBitSet inputData(DIM_SIZE * 16UL);
    inputData.UnSetAll();
    EXPECT_EQ(inputData.HashValue(), zeroResult);
}
TEST_F(TestOckElasticBitSet, hash_value_one)
{
    OckElasticBitSet inputData(DIM_SIZE * 16UL);
    inputData.SetAll();
    // 全为 1
    uint64_t oneResult = 33553920ULL;
    EXPECT_EQ(inputData.HashValue(), oneResult);
}
TEST_F(TestOckElasticBitSet, hash_value_local_one)
{
    OckElasticBitSet inputData(DIM_SIZE * 16UL);
    // Data[WordCount], 值单独赋为1
    for (uint64_t i = 0; i < inputData.WordCount(); ++i) {
        inputData.Data()[i] = 1;
    }
    // 每组中有 4 个 1
    uint64_t localOneResult = 524280ULL;
    EXPECT_EQ(inputData.HashValue(), localOneResult);
}
TEST_F(TestOckElasticBitSet, hash_value_space_one)
{
    OckElasticBitSet inputData(DIM_SIZE * 16UL);
    uint64_t space = 2ULL;
    for (uint64_t i = 0; i < inputData.WordCount(); i+=space) {
        inputData.Data()[i] = 1;
    }
    // 每组中有 2 个 1
    uint64_t spaceOneResult = 262140ULL;
    EXPECT_EQ(inputData.HashValue(), spaceOneResult);
}
TEST_F(TestOckElasticBitSet, hash_value_half_one)
{
    OckElasticBitSet inputData(DIM_SIZE * 16UL);
    uint64_t space = 2ULL;
    for (uint64_t i = 0; i < inputData.Size(); i+=space) {
        inputData.Set(i, 1);
    }
    // 每组中有 2 个 1
    uint64_t halfOneResult = 16776960ULL;
    EXPECT_EQ(inputData.HashValue(), halfOneResult);
}
TEST_F(TestOckElasticBitSet, hash_value_rand_one)
{
    OckElasticBitSet inputData(DIM_SIZE * 16UL);
    inputData.Set(1ULL, 1);
    inputData.Set(550ULL, 1);
    inputData.Set(1660ULL, 1);
    inputData.Set(2330ULL, 1);
    inputData.Set(4095ULL, 1);
    uint64_t randOneResult = 83074ULL;
    EXPECT_EQ(inputData.HashValue(), randOneResult);
}
} // namespace algo
} // namespace hcps
} // namespace ock
