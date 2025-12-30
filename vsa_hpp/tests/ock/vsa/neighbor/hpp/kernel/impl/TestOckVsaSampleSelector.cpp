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
#include <cassert>
#include <gtest/gtest.h>
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaSampleSelector.h"
#include "ock/vsa/neighbor/hpp/OckVsaAnnHppSetup.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
namespace relation {
namespace test {
class TestOckVsaSampleSelector : public testing::Test {
public:
    explicit TestOckVsaSampleSelector() {}

    void SetUp() override
    {
        SetUpHPPTsFactory();
    }

    void TearDown() override {}

    uint32_t rowCount = 100UL;
    OckVsaSampleSelector<int8_t, 256U> ockSelector{ rowCount };
};

TEST_F(TestOckVsaSampleSelector, select_not_overflow_correctly)
{
    EXPECT_EQ(rowCount, 100UL);
    EXPECT_EQ(ockSelector.relatedBitSet.Count(), 0UL);
    EXPECT_EQ(ockSelector.primaryBitSet.Count(), 0UL);

    uint32_t needCount = 16UL;
    std::vector<uint32_t> dataVec(needCount);
    ockSelector.SelectUnusedRow(needCount, dataVec.data());

    EXPECT_EQ(ockSelector.relatedBitSet.Count(), 16UL);
    EXPECT_EQ(ockSelector.primaryBitSet.Count(), 16UL);

    ockSelector.SetUsed(16UL);
    ockSelector.SetUsed(20UL);
    EXPECT_EQ(ockSelector.relatedBitSet.Count(), 18UL);

    needCount = 48UL;
    dataVec.clear();
    dataVec.reserve(needCount);
    ockSelector.SelectUnusedRow(needCount, dataVec.data());
    EXPECT_EQ(dataVec[0UL], 17UL);
    EXPECT_EQ(dataVec[1UL], 18UL);
    EXPECT_EQ(dataVec[2UL], 19UL);
    EXPECT_EQ(dataVec[3UL], 21UL);
    EXPECT_EQ(ockSelector.relatedBitSet.Count(), 66UL);
    EXPECT_EQ(ockSelector.primaryBitSet.Count(), 64UL);
}

TEST_F(TestOckVsaSampleSelector, select_overflow_correctly)
{
    uint32_t needCount = 48UL;
    std::vector<uint32_t> dataVec(needCount);
    ockSelector.SelectUnusedRow(needCount, dataVec.data());
    for (size_t i = 0; i < 30UL; i++) {
        ockSelector.SetUsed(i + 40UL);
    }
    EXPECT_EQ(ockSelector.relatedBitSet.Count(), 70UL);
    EXPECT_EQ(ockSelector.primaryBitSet.Count(), 48UL);

    needCount = 32UL;
    dataVec.clear();
    dataVec.reserve(needCount);
    bool res = ockSelector.SelectUnusedRow(needCount, dataVec.data());
    EXPECT_EQ(ockSelector.relatedBitSet.Count(), 100UL);
    EXPECT_EQ(ockSelector.primaryBitSet.Count(), 80UL);
    EXPECT_EQ(res, true);

    res = ockSelector.SelectUnusedRow(needCount, dataVec.data());
    EXPECT_EQ(res, false);
}
}
} // namespace test
} // namespace relation
} // namespace neighbor
} // namespace vsa
} // namespace ock