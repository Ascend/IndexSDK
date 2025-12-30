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
#include <cstring>
#include <cassert>
#include <fstream>
#include <gtest/gtest.h>
#include "ptest/ptest.h"
#include "ock/log/OckHcpsLogger.h"
#include "ock/hcps/algo/OckShape.h"

namespace ock {
namespace hcps {
namespace algo {
namespace {
constexpr uint32_t DIM = 256UL;
constexpr uint32_t CUBE_ALIGN = 16UL;
} // namespace

class PTestOckShape : public testing::Test {
public:
    // Default: Test 64MB data, when dimSize = 256, dataType = int8, dataNum = 64 * 1024 * 1024 / 256 = 262144
    explicit PTestOckShape(uint32_t dataNum = 262144UL) : dataNum(dataNum), dimSize(DIM), byteSize(dataNum * DIM) {}

    void SetUp() override
    {
        dataBase.reserve(byteSize);
        data_ = static_cast<int8_t *>(dataBase.data());
    }

    // 随机生成num条int8向量
    void InitRandData(uint32_t num, int8_t minValue = -20, int8_t maxValue = 64)
    {
        if (num > dataNum) {
            OCK_HCPS_LOG_ERROR("The size is " << num * dimSize << ", larger than byteSize, which is " << byteSize);
            return;
        }
        assert(minValue >= std::numeric_limits<int8_t>::min());
        assert(maxValue <= std::numeric_limits<int8_t>::max());
        std::random_device seed;
        std::ranlux48 engine(seed());
        std::uniform_int_distribution<> distrib(minValue, maxValue);

        for (uint32_t i = 0; i < num; ++i) {
            for (uint32_t j = 0; j < dimSize; ++j) {
                dataBase.emplace_back(distrib(engine));
            }
        }
    }

    void InitSameData(uint32_t num = 100)
    {
        int8_t tmpNum;
        int maxInt8 = std::numeric_limits<int8_t>::max();
        for (uint32_t i = 0; i < num; ++i) {
            tmpNum = i % maxInt8;
            for (uint32_t j = 0; j < dimSize; ++j) {
                dataBase.emplace_back(tmpNum);
            }
        }
    }

    void InitRandVec()
    {
        if (dataBase.size() + dimSize * sizeof(int8_t) > byteSize) {
            OCK_HCPS_LOG_ERROR("The size is " << dataBase.size() << ", larger than byteSize, which is " << byteSize);
            return;
        }
        for (uint32_t j = 0; j < dimSize; ++j) {
            dataBase.push_back((int8_t)(rand() % std::numeric_limits<int8_t>::max()));
        }
    }

    uint32_t dataNum;
    uint32_t dimSize;
    std::vector<int8_t> dataBase;
    int8_t *data_;
    uint64_t byteSize;
};

TEST_F(PTestOckShape, performance_test_1_add_data)
{
    const uint32_t initVecNum = 262144UL;
    InitRandData(initVecNum);
    std::vector<int8_t> blockData(byteSize);
    OckShape<int8_t, DIM, CUBE_ALIGN> ockShape(reinterpret_cast<uintptr_t>(blockData.data()), byteSize);

    auto timeGuard = fast::hdt::TestTimeGuard();
    ockShape.AddData(data_, initVecNum);
    EXPECT_TRUE(FAST_PTEST().Test("OCK.HCPS.ALGO.Shape.Add", "UsedTime", timeGuard.ElapsedMicroSeconds()));
}

TEST_F(PTestOckShape, performance_test_2_get_data)
{
    const uint32_t initVecNum = 50000UL;
    InitRandData(initVecNum);
    std::vector<int8_t> blockData(byteSize);
    std::vector<int8_t> fetchData(dimSize);
    OckShape<int8_t, DIM, CUBE_ALIGN> ockShape(reinterpret_cast<uintptr_t>(blockData.data()), byteSize);
    ockShape.AddData(data_, initVecNum);

    auto timeGuard = fast::hdt::TestTimeGuard();
    for (uint64_t i = 0; i < initVecNum; i++) {
        ockShape.GetData(i, fetchData.data());
    }
    EXPECT_TRUE(FAST_PTEST().Test("OCK.HCPS.ALGO.Shape.PickUp", "UsedTime", timeGuard.ElapsedMicroSeconds()));
}

TEST_F(PTestOckShape, performance_test_3_copy_from_block)
{
    const uint32_t initVecNum = 262144UL;
    InitRandData(initVecNum);
    std::vector<int8_t> blockA(byteSize);
    std::vector<int8_t> blockB(byteSize);
    int8_t *fetchPtr = new int8_t[dimSize]{};
    OckShape<int8_t, DIM, CUBE_ALIGN> ockShapeA(reinterpret_cast<uintptr_t>(blockA.data()), byteSize);
    OckShape<int8_t, DIM, CUBE_ALIGN> ockShapeB(reinterpret_cast<uintptr_t>(blockB.data()), byteSize);
    ockShapeA.AddData(data_, initVecNum);
    ockShapeB.AddData(data_, initVecNum);

    auto timeGuard = fast::hdt::TestTimeGuard();
    for (uint64_t i = 0; i < initVecNum; i++) {
        ockShapeB.CopyFrom(i, ockShapeA, i);
    }
    EXPECT_TRUE(FAST_PTEST().Test("OCK.HCPS.ALGO.Shape.CopyFrom", "UsedTime", timeGuard.ElapsedMicroSeconds()));
    delete[] fetchPtr;
}

TEST_F(PTestOckShape, performance_test_4_restore_block)
{
    const uint32_t initVecNum = 262144UL;
    InitRandData(initVecNum);
    std::vector<int8_t> blockData(byteSize);
    std::vector<int8_t> fetchData(byteSize);
    OckShape<int8_t, DIM, CUBE_ALIGN> ockShape(reinterpret_cast<uintptr_t>(blockData.data()), byteSize);
    ockShape.AddData(data_, initVecNum);

    auto timeGuard = fast::hdt::TestTimeGuard();
    ockShape.Restore(fetchData.data());
    EXPECT_TRUE(FAST_PTEST().Test("OCK.HCPS.ALGO.Shape.Restore", "UsedTime", timeGuard.ElapsedMicroSeconds()));
}
} // namespace algo
} // namespace hcps
} // namespace ock
