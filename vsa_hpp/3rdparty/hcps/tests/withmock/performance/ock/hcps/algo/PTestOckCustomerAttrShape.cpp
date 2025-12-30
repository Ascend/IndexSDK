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
#include "ptest/ptest.h"
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
// Default: 100个属性，262144条数据
const uint32_t ATTR_COUNT = 100UL;
const uint32_t BLOCK_ROW_COUNT = 262144UL;
const uint32_t BLOCK_COUNT = 1UL;
} // namespace

struct IdxClass {
    explicit IdxClass(uint64_t pos) : pos(pos) {}
    uint64_t pos;
};

class PTestOckCustomerAttrShape : public handler::WithEnvOckHeteroHandler<testing::Test> {
public:
    using BaseT = handler::WithEnvOckHeteroHandler<testing::Test>;
    using DataT = uint8_t;

    explicit PTestOckCustomerAttrShape()
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

    std::vector<DataT> groupData;
    std::vector<DataT> otherData;
    std::vector<std::shared_ptr<IdxClass>> idxMap;
    DataT *data_;
};

TEST_F(PTestOckCustomerAttrShape, customerAttr_reArrange)
{
    OckCustomerAttrShape<std::shared_ptr<IdxClass>> attrShape(data_, ATTR_COUNT, BLOCK_COUNT, BLOCK_ROW_COUNT);
    OckCustomerAttrShape<std::shared_ptr<IdxClass>> otherShape(otherData.data(), ATTR_COUNT, BLOCK_COUNT,
        BLOCK_ROW_COUNT);
    const std::vector<std::shared_ptr<IdxClass>> noModifiedMap = idxMap;

    auto timeGuard = fast::hdt::TestTimeGuard();
    attrShape.CopyFromBoost(otherShape, noModifiedMap.data(), uint32_t(0UL));
    EXPECT_TRUE(FAST_PTEST().Test("OCK.HCPS.Algo.CustomerAttr.ReArrange", "UsedTime", timeGuard.ElapsedMicroSeconds()));
}
} // namespace test
} // namespace algo
} // namespace hcps
} // namespace ock
