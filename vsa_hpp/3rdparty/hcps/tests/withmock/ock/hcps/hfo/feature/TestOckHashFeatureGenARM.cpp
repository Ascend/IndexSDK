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


#include <ostream>
#include <cstdint>
#include <random>
#include <gtest/gtest.h>
#include "ock/hcps/algo/OckBitSet.h"
#include "ock/hcps/hfo/feature/OckHashFeatureGen.h"
#include "ptest/ptest.h"

namespace ock {
namespace hcps {
namespace hfo {
namespace {
const uint64_t DIM_SIZE = 256UL;
const uint64_t PRECISIONT = (1U << 16ULL) - 1U;
} // namespace
class TestOckHashFeatureGenARM : public testing::Test {
public:
    using FromDataT = int8_t;
    using BitDataT = algo::OckBitSet<DIM_SIZE * 16UL, DIM_SIZE>;
    using OckHashFeatureGenT = OckHashFeatureGen<int8_t, DIM_SIZE, 16ULL, PRECISIONT, BitDataT>;
    TestOckHashFeatureGenARM(void) {}
    void InitRandAData(void)
    {
        for (uint64_t i = 0; i < DIM_SIZE; ++i) {
            fromData[i] = static_cast<FromDataT>(rand() % (std::numeric_limits<FromDataT>::max() + 1U));
        }
    }
    int CompareARMwithx86Result(void)
    {
        OckHashFeatureGenT::Gen(fromData, toDataARM);
        OckHashFeatureGenT::GenX86(fromData, toDataBaseline);
        if (toDataARM.Compare(toDataBaseline) == 0) {
            return 1;
        } else {
            return 0;
        }
    }

    FromDataT fromData[DIM_SIZE];
    BitDataT toDataARM;
    BitDataT toDataBaseline;
};
TEST_F(TestOckHashFeatureGenARM, compare_ARM_and_x86_results)
{
    InitRandAData();
    int result = CompareARMwithx86Result();
    EXPECT_EQ(result, 1);
}
TEST_F(TestOckHashFeatureGenARM, compare_all1_results)
{
    InitRandAData();
    memset_s(&fromData, sizeof(FromDataT) * DIM_SIZE, 1, sizeof(FromDataT) * DIM_SIZE);
    int result = CompareARMwithx86Result();
    EXPECT_EQ(result, 1);
}
TEST_F(TestOckHashFeatureGenARM, compare_all0_results)
{
    InitRandAData();
    memset_s(&fromData, sizeof(FromDataT) * DIM_SIZE, 0, sizeof(FromDataT) * DIM_SIZE);
    int result = CompareARMwithx86Result();
    EXPECT_EQ(result, 1);
}
} // namespace hfo
} // namespace hcps
} // namespace ock
