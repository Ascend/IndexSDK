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
#include <securec.h>
#include <memory>
#include <chrono>
#include <random>
#include <gtest/gtest.h>
#include "ock/utils/StrUtils.h"
#include "ock/hcps/hfo/feature/OckCosFeature.h"

namespace ock {
namespace hcps {
namespace hfo {
namespace {
const uint64_t DIM_SIZE = 256UL;
const uint64_t ROW_COUNT = 1024UL;
const uint64_t PRECISIONT = (1U << 16ULL) - 1U;
} // namespace
class TestOckCosFeature : public testing::Test {
public:
    void SetUp(void) override
    {
        memset_s(dataA, DIM_SIZE, 0, DIM_SIZE);
        memset_s(dataB, DIM_SIZE, 0, DIM_SIZE);
    }

    uint8_t dataA[DIM_SIZE];
    uint8_t dataB[DIM_SIZE];
};
TEST_F(TestOckCosFeature, norm_factor_ARM_and_x86)
{
    using OckCosLenT = OckCosLen<uint8_t, PRECISIONT>;
    for (uint64_t i = 0; i < DIM_SIZE; ++i) {
        dataA[i] = static_cast<uint8_t>(rand() % (std::numeric_limits<uint8_t>::max()));
    }
    double factorARM = OckCosLenT::NormFactorImplARM(dataA, DIM_SIZE);
    double factorX86 = OckCosLenT::NormFactorImpl(dataA, DIM_SIZE);
    EXPECT_DOUBLE_EQ(factorARM, factorX86);
}
} // namespace hfo
} // namespace hcps
} // namespace ock