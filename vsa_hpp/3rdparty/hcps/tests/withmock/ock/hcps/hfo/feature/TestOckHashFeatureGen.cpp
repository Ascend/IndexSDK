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
#include <memory>
#include <chrono>
#include <random>
#include <gtest/gtest.h>
#include "ock/utils/StrUtils.h"
#include "ock/utils/OstreamUtils.h"
#include "ock/hcps/hfo/feature/OckHashFeatureGen.h"
#include "ptest/ptest.h"

namespace ock {
namespace hcps {
namespace hfo {
namespace {
const uint64_t DIM_SIZE = 256UL;
const uint64_t ROW_COUNT = 10240UL;
const double DISTANCE_PRECESION = 0.0001;
}  // namespace
class TestOckHashFeatureGen : public testing::Test {
public:
    using FromDataT = uint8_t;
    using ToDataT = uint32_t;
    const uint64_t maxTraitsValue = (1ULL << 18UL) - 1ULL;
    using OckHashFeatureGenT = OckHashFeatureGen<int8_t, DIM_SIZE, 16ULL>;
    using OckCosLenT = OckCosLen<ToDataT, (1ULL << 18UL) - 1ULL>;
    using HashDataT = OckHashFeatureGenT::BitDataT;
    TestOckHashFeatureGen(void)
        : inputVecA(DIM_SIZE * ROW_COUNT), inputVecB(DIM_SIZE * ROW_COUNT), outputVecA(DIM_SIZE * ROW_COUNT),
          outputVecB(DIM_SIZE * ROW_COUNT)
    {}
    void SetUp(void) override
    {
        memset_s(&fromData, sizeof(FromDataT) * DIM_SIZE, 0, sizeof(FromDataT) * DIM_SIZE);
        memset_s(&toData, sizeof(ToDataT) * DIM_SIZE, 0, sizeof(ToDataT) * DIM_SIZE);
    }
    void TestLenDistribute(void)
    {
        double maxValue = 0;
        double minValue = 999999999.0;
        double sumValue = 0;
        for (uint64_t i = 0; i < ROW_COUNT; ++i) {
            double tmpLen = OckCosLenT::LenImpl(&outputVecA[i * DIM_SIZE], DIM_SIZE);
            maxValue = std::max(tmpLen, maxValue);
            minValue = std::min(tmpLen, minValue);
            sumValue += tmpLen;
        }
        double meanValue = sumValue / ROW_COUNT;
        EXPECT_LT((maxValue - minValue) / meanValue, DISTANCE_PRECESION);
    }
    std::vector<FromDataT> inputVecA;
    std::vector<FromDataT> inputVecB;
    std::vector<ToDataT> outputVecA;
    std::vector<ToDataT> outputVecB;
    FromDataT fromData[DIM_SIZE];
    ToDataT toData[DIM_SIZE];
};
}  // namespace hfo
}  // namespace hcps
}  // namespace ock