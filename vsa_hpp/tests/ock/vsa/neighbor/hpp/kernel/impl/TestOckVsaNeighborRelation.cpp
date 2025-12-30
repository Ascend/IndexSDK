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
#include "ock/hcps/algo/OckShape.h"
#include "ock/log/OckHcpsLogger.h"
#include "ock/hcps/error/OckHcpsErrorCode.h"
#include "ock/hcps/WithEnvOckHeteroHandler.h"
#include "ock/hcps/handler/OckHeteroHandler.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaNeighborRelation.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
namespace relation {
namespace test {
constexpr uint32_t DIM_SIZE = 256UL;

class TestOckVsaNeighborRelation : public hcps::handler::WithEnvOckHeteroHandler<testing::Test> {
public:
    using BaseT = WithEnvOckHeteroHandler<testing::Test>;
    using DataT = uint8_t;

    explicit TestOckVsaNeighborRelation() {}

    void SetUp() override
    {
        BaseT::SetUp();
        handler = CreateSingleDeviceHandler(errorCode);
    }

    void TearDown() override
    {
        handler.reset();
        BaseT::TearDown();
    }

    std::vector<DataT> InitSameData(uint32_t num = 100)
    {
        DataT tmpNum;
        std::vector<DataT> dataBase;
        int maxInt8 = std::numeric_limits<int8_t>::max();
        for (uint32_t i = 0; i < num; ++i) {
            tmpNum = i % maxInt8;
            for (uint32_t j = 0; j < DIM_SIZE; ++j) {
                dataBase.emplace_back(tmpNum);
            }
        }
        return dataBase;
    }

    bool CheckGetVec(int64_t num, int8_t *fetchPtr)
    {
        for (uint32_t i = 0; i < DIM_SIZE; i++) {
            if (*(fetchPtr + i) != num) {
                return false;
            }
        }
        return true;
    }

    uint32_t rowCount = 100UL;
    OckVsaErrorCode errorCode = hmm::HMM_SUCCESS;
    std::shared_ptr<hcps::handler::OckHeteroHandler> handler;
};

TEST_F(TestOckVsaNeighborRelation, make_shaped_correctly)
{
    uint64_t byteSize = rowCount * DIM_SIZE * sizeof(DataT);
    std::shared_ptr<hmm::OckHmmHMObject> featureHMO = hcps::handler::helper::MakeHostHmo(*handler, byteSize, errorCode);
    if (errorCode != hmm::HMM_SUCCESS) {
        OCK_HCPS_LOG_ERROR("Malloc featureHMO fail, the errorCode is " << errorCode);
    }
    std::vector<hpp::relation::OckVsaNeighborRelation> neighborRelationVec(rowCount);

    std::vector<uint32_t> constVec(NEIGHBOR_RELATION_COUNT_PER_CELL, 0UL);
    // 生成底库数据
    auto feature = InitSameData();

    // 生成groupInfo
    OckVsaNeighborRelationHmoGroup groupInfo(0UL);
    for (size_t i = 0; i < rowCount; i++) {
        groupInfo.AddData(i, constVec);
    }

    DataT *pFeature = reinterpret_cast<DataT *>(featureHMO->Addr());
    memcpy_s(pFeature, byteSize, feature.data(), byteSize);

    auto hostShapedHMO = MakeShapedSampleFeature<int8_t, DIM_SIZE>(*handler, groupInfo, featureHMO, errorCode);
    EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);

    std::vector<int8_t> fetchPtr(DIM_SIZE);
    hcps::algo::OckShape<> ockShape(hostShapedHMO->Addr(), byteSize, rowCount);
    ockShape.GetData(0UL, fetchPtr.data());
    EXPECT_EQ(CheckGetVec(0UL, fetchPtr.data()), true);
    ockShape.GetData(10UL, fetchPtr.data());
    EXPECT_EQ(CheckGetVec(10UL, fetchPtr.data()), true);
    ockShape.GetData(90UL, fetchPtr.data());
    EXPECT_EQ(CheckGetVec(90UL, fetchPtr.data()), true);
}

TEST_F(TestOckVsaNeighborRelation, make_norm_correctly)
{
    uint64_t byteSize = rowCount * sizeof(uint16_t);
    std::shared_ptr<hmm::OckHmmHMObject> normHMO = hcps::handler::helper::MakeHostHmo(*handler, byteSize, errorCode);
    if (errorCode != hmm::HMM_SUCCESS) {
        OCK_HCPS_LOG_ERROR("Malloc normHMO fail, the errorCode is " << errorCode);
    }
    std::vector<OckVsaNeighborRelation> neighborRelationVec(rowCount);

    std::vector<uint16_t> normVec(rowCount);
    for (size_t i = 0; i < rowCount; i++) {
        normVec[i] = i;
    }
    std::vector<uint32_t> constVec(NEIGHBOR_RELATION_COUNT_PER_CELL, 0UL);
    OckVsaNeighborRelationHmoGroup groupInfo(0UL);
    for (size_t i = 0; i < rowCount; i++) {
        groupInfo.AddData(i, constVec);
    }

    uint16_t *pNorm = reinterpret_cast<uint16_t *>(normHMO->Addr());
    memcpy_s(pNorm, byteSize, normVec.data(), byteSize);

    auto hostNormHMO = MakeShapedSampleNorm(*handler, groupInfo, normHMO, errorCode);
    uint16_t *pData = reinterpret_cast<uint16_t *>(hostNormHMO->Addr());

    EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
    EXPECT_EQ(pData[0UL], 0UL);
    EXPECT_EQ(pData[10UL], 10UL);
    EXPECT_EQ(pData[90UL], 90UL);
}
}
} // namespace test
} // namespace relation
} // namespace neighbor
} // namespace vsa
} // namespace ock