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
#include <fstream>
#include <cassert>
#include <gtest/gtest.h>
#include "ock/log/OckHcpsLogger.h"
#include "ock/hcps/error/OckHcpsErrorCode.h"
#include "ock/hcps/handler/OckHandlerHmmHelper.h"
#include "ock/hmm/mgr/MockOckHmmSingleDeviceMgr.h"
#include "ock/hcps/WithEnvOckHeteroHandler.h"
#include "ock/hcps/algo/OckCustomerAttrShape.h"

namespace ock {
namespace hcps {
namespace algo {
namespace test {
namespace {
// Default: 22个属性，262144条数据
const uint32_t ATTR_COUNT = 22UL;
const uint32_t BLOCK_ROW_COUNT = 262144UL;
const uint32_t BLOCK_COUNT = 3UL;
} // namespace

struct IdxClass {
    explicit IdxClass(uint64_t pos) : pos(pos) {}
    uint64_t pos;
};

class TestOckCustomerAttrShape : public handler::WithEnvOckHeteroHandler<testing::Test> {
public:
    using BaseT = handler::WithEnvOckHeteroHandler<testing::Test>;
    using DataT = uint8_t;

    explicit TestOckCustomerAttrShape()
    {
        groupData.reserve(BLOCK_COUNT * ATTR_COUNT * BLOCK_ROW_COUNT);
        otherData.reserve(BLOCK_COUNT * ATTR_COUNT * BLOCK_ROW_COUNT);
    }

    void SetUp() override
    {
        BaseT::SetUp();
        InitRandBlock(otherData);
        InitShuffleMap(idxMap);
        data_ = static_cast<DataT *>(groupData.data());
    }

    void TearDown(void) override
    {
        BaseT::TearDown();
    }

    // 随机生成uint8矩阵
    void InitRandBlock(std::vector<DataT> &dataBase, DataT minValue = 0, DataT maxValue = 255)
    {
        // 确保数据在类型范围内
        assert(minValue >= std::numeric_limits<DataT>::min());
        assert(maxValue <= std::numeric_limits<DataT>::max());
        std::random_device seed;
        std::ranlux48 engine(seed());
        std::uniform_int_distribution<> distrib(minValue, maxValue);
        uint64_t groupSize = ATTR_COUNT * BLOCK_ROW_COUNT * BLOCK_COUNT;
        for (uint64_t i = 0; i < groupSize; ++i) {
            dataBase[i] = DataT(distrib(engine));
        }
    }

    void InitShuffleMap(std::vector<std::shared_ptr<IdxClass>> &tmpMap)
    {
        assert(tmpMap.size() == 0);
        std::shared_ptr<IdxClass> mapPtr;
        std::vector<DataT> randVec(BLOCK_ROW_COUNT);
        for (uint32_t i = 0; i < BLOCK_ROW_COUNT; ++i) {
            randVec[i] = i;
        }
        std::random_shuffle(randVec.begin(), randVec.end());
        for (uint32_t i = 0; i < BLOCK_ROW_COUNT; ++i) {
            mapPtr = std::make_shared<IdxClass>(randVec[i]);
            tmpMap.push_back(mapPtr);
        }
    }

    bool ComparePtrValue(DataT *inputPtr, DataT *outputPtr, uint32_t vecLen)
    {
        for (uint32_t i = 0; i < vecLen; i++) {
            if (*(inputPtr + i) != *(outputPtr + i)) {
                return false;
            }
        }
        return true;
    }

    std::vector<DataT> groupData;
    std::vector<DataT> otherData;
    std::vector<std::shared_ptr<IdxClass>> idxMap;
    DataT *data_;
    hmm::OckHmmErrorCode errorCode{ hmm::HMM_SUCCESS };
    std::shared_ptr<handler::OckHeteroHandler> handler;
};

TEST_F(TestOckCustomerAttrShape, shuffle_oneRow_correctly)
{
    uint64_t blocks = 1UL;
    uint64_t attrs = 1UL;
    uint64_t rowCount = 5UL;
    uint64_t groupSize = blocks * attrs * rowCount;
    std::vector<DataT> dataBaseA{ 1UL, 2UL, 3UL, 4UL, 5UL };
    std::vector<std::shared_ptr<IdxClass>> idxMapA{ std::make_shared<IdxClass>(4UL), std::make_shared<IdxClass>(3UL),
        std::make_shared<IdxClass>(2UL), std::make_shared<IdxClass>(1UL), std::make_shared<IdxClass>(0UL) };
    std::vector<DataT> dataBaseB(groupSize, 0);

    OckCustomerAttrShape<std::shared_ptr<IdxClass>> shapeA(dataBaseA.data(), attrs, blocks, rowCount);
    OckCustomerAttrShape<std::shared_ptr<IdxClass>> shapeB(dataBaseB.data(), attrs, blocks, rowCount);

    const std::vector<std::shared_ptr<IdxClass>> noModifiedMap = idxMapA;
    shapeB.CopyFromOneAttr(shapeA, noModifiedMap.data(), uint32_t(0UL));

    EXPECT_EQ(dataBaseB[0UL], 5UL);
    EXPECT_EQ(dataBaseB[1UL], 4UL);
    EXPECT_EQ(dataBaseB[2UL], 3UL);
    EXPECT_EQ(dataBaseB[3UL], 2UL);
    EXPECT_EQ(dataBaseB[4UL], 1UL);
}

TEST_F(TestOckCustomerAttrShape, shuffle_multiRows_correctly)
{
    uint64_t groupSize = BLOCK_COUNT * ATTR_COUNT * BLOCK_ROW_COUNT;
    std::vector<DataT> dataBaseA(groupSize, 0);
    std::vector<DataT> dataBaseB(groupSize, 0);
    std::vector<DataT> dataBaseC(groupSize, 0);
    OckCustomerAttrShape<std::shared_ptr<IdxClass>> otherShape(otherData.data(), ATTR_COUNT, BLOCK_COUNT,
        BLOCK_ROW_COUNT);
    OckCustomerAttrShape<std::shared_ptr<IdxClass>> shapeA(dataBaseA.data(), ATTR_COUNT, BLOCK_COUNT, BLOCK_ROW_COUNT);
    OckCustomerAttrShape<std::shared_ptr<IdxClass>> shapeB(dataBaseB.data(), ATTR_COUNT, BLOCK_COUNT, BLOCK_ROW_COUNT);
    OckCustomerAttrShape<std::shared_ptr<IdxClass>> shapeC(dataBaseC.data(), ATTR_COUNT, BLOCK_COUNT, BLOCK_ROW_COUNT);

    const std::vector<std::shared_ptr<IdxClass>> noModifiedMap = idxMap;
    shapeA.CopyFromGeneral(otherShape, noModifiedMap.data(), uint32_t(0UL));
    shapeB.CopyFromBoost(otherShape, noModifiedMap.data(), uint32_t(0UL));
    shapeC.CopyFrom(otherShape, noModifiedMap.data(), uint32_t(0UL));
    EXPECT_TRUE(ComparePtrValue(dataBaseA.data(), dataBaseB.data(), groupSize));
    EXPECT_TRUE(ComparePtrValue(dataBaseA.data(), dataBaseC.data(), groupSize));

    shapeA.CopyFromGeneral(otherShape, noModifiedMap.data(), uint32_t(2UL));
    shapeB.CopyFromBoost(otherShape, noModifiedMap.data(), uint32_t(2UL));
    shapeC.CopyFrom(otherShape, noModifiedMap.data(), uint32_t(2UL));
    EXPECT_TRUE(ComparePtrValue(dataBaseA.data(), dataBaseB.data(), groupSize));
    EXPECT_TRUE(ComparePtrValue(dataBaseA.data(), dataBaseC.data(), groupSize));
}

} // namespace test
} // namespace algo
} // namespace hcps
} // namespace ock
