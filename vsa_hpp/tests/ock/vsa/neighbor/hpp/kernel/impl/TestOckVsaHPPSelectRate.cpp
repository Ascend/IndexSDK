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
#include <deque>
#include <memory>
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPSelectRate.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
namespace impl {
class TestOckVsaHPPSelectRate : public testing::Test {
public:
    uint32_t topLow{10UL};
    uint32_t top{100UL};
    uint32_t topHigh{1000UL};
    uint32_t groupCountLow{1UL};
    uint32_t groupCount{20UL};
    uint32_t groupCountHigh{50UL};
    uint32_t groupSliceCountLow{8192UL};
    uint32_t groupSliceCount{65536UL};
    uint32_t groupSliceCountHigh{655360UL};
    double keyCovRateLow{0.0001};
    double keyCovRate{0.5};
    double keyCovRateHigh{1.0};
    double selectRateLow{0.01};
    double selectRate{0.5};
    double selectRateHigh{1.0};
    uint32_t externTopNLow{1000UL};
    uint32_t externTopN{5000UL};
    uint32_t externTopNHigh{10000UL};
    const uint64_t sliceByteSize{65536ULL};
};
TEST_F(TestOckVsaHPPSelectRate, CalcTopNInSampleGroup)
{
    uint32_t validateRowCount = 20UL;
    uint32_t validateRowCount1 = 23UL;
    uint32_t validateRowCount2 = 46UL;
    std::deque<std::shared_ptr<relation::OckVsaNeighborRelationHmoGroup>> relTable;
    std::shared_ptr<relation::OckVsaNeighborRelationHmoGroup> relationHmoGroup =
        std::make_shared<relation::OckVsaNeighborRelationHmoGroup>(validateRowCount);
    std::shared_ptr<relation::OckVsaNeighborRelationHmoGroup> relationHmoGroup1 =
        std::make_shared<relation::OckVsaNeighborRelationHmoGroup>(validateRowCount1);
    std::shared_ptr<relation::OckVsaNeighborRelationHmoGroup> relationHmoGroup2 =
        std::make_shared<relation::OckVsaNeighborRelationHmoGroup>(validateRowCount2);
    relTable.emplace_back(relationHmoGroup);
    relTable.emplace_back(relationHmoGroup1);
    relTable.emplace_back(relationHmoGroup2);

    EXPECT_EQ(CalcTopNInSampleGroup(top, groupCount, keyCovRate, 262144UL, relTable), 575UL);
}
}  // namespace impl
}  // namespace hpp
}  // namespace neighbor
}  // namespace vsa
}  // namespace ock