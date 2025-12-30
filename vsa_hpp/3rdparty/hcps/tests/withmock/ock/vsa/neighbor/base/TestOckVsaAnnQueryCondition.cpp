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
#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"
#include "ock/vsa/neighbor/base/OckVsaAnnQueryCondition.h"
#include "ock/vsa/attr/OckTimeSpaceAttrTrait.h"
#include "ock/vsa/attr/impl/OckTimeSpaceAttrTraitImpl.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace test {
const int DIMS = 8;
class TestOckVsaAnnQueryCondition : public testing::Test {
public:
    void BuildSingleQuery(uint8_t startValue = 0)
    {
        InitQueryFeature(DIMS, 1, startValue);
        InitAttrFilter();
        singleQueryCondition =
            std::make_shared<OckVsaAnnSingleBatchQueryCondition<uint8_t, DIMS, attr::OckTimeSpaceAttrTrait>>(batchPos,
            queryFeature, attrFilter, shareAttrFilter, topk, extraMask, extraMaskLenEachQuery, extraMaskIsAtDevice,
            enableTimeFilter);
    }

    void TearDown()
    {
        free(queryFeature);
        delete attrFilter;
    }

    void BuildQueryCondition(uint8_t startValue = 0)
    {
        InitQueryFeature(DIMS, queryBatchCount, startValue);
        InitAttrFilter();
        queryCondition = std::make_shared<OckVsaAnnQueryCondition<uint8_t, DIMS, attr::OckTimeSpaceAttrTrait>>(
            queryBatchCount, queryFeature, attrFilter, shareAttrFilter, topk, extraMask, extraMaskLenEachQuery,
            extraMaskIsAtDevice, enableTimeFilter);
    }

    void InitQueryFeature(uint8_t dim, uint8_t batchSize, uint8_t startValue = 1)
    {
        queryFeature = (uint8_t *)malloc(sizeof(uint8_t) * dim * batchSize);
        int pos = 0;
        for (uint8_t i = 0; i < batchSize; ++i) {
            for (uint8_t j = 0; j < dim; ++j) {
                queryFeature[pos] = i + startValue;
                pos++;
            }
        }
    }

    void InitAttrFilter(uint32_t maxTokenNumber = 1024, int32_t time = 5, uint32_t tokenId = 5)
    {
        attrFilter = new attr::OckTimeSpaceAttrTrait(maxTokenNumber);
        attrFilter->Add(attr::OckTimeSpaceAttr(time, tokenId));
    }

    void CompareQueryFeature(const uint8_t *ptrA, const uint8_t *ptrB)
    {
        for (uint32_t i = 0; i < DIMS; i++) {
            EXPECT_EQ(ptrA[i], ptrB[i]);
        }
    }

    std::shared_ptr<OckVsaAnnSingleBatchQueryCondition<uint8_t, DIMS, attr::OckTimeSpaceAttrTrait>>
        singleQueryCondition;
    std::shared_ptr<OckVsaAnnQueryCondition<uint8_t, DIMS, attr::OckTimeSpaceAttrTrait>> queryCondition;

    const uint32_t batchPos{ 0 };
    const uint32_t queryBatchCount{ 4 };
    uint8_t *queryFeature;
    attr::OckTimeSpaceAttrTrait *attrFilter;
    bool shareAttrFilter{ true };
    uint32_t topk{ 16 };
    uint8_t *extraMask = nullptr;
    uint64_t extraMaskLenEachQuery{ 0 };
    bool extraMaskIsAtDevice{ false };
    bool enableTimeFilter{ true };
    uint32_t maxTokenNumber{1024};
};

TEST_F(TestOckVsaAnnQueryCondition, CreateSingleQueryCond)
{
    uint8_t startValue = 50;
    BuildSingleQuery(startValue);
    EXPECT_EQ(singleQueryCondition->extraMask, nullptr);

    EXPECT_EQ((*(singleQueryCondition->attrFilter)).maxTime, (*(attrFilter)).maxTime);
    EXPECT_EQ((*(singleQueryCondition->attrFilter)).maxTokenId, (*(attrFilter)).maxTokenId);
    EXPECT_EQ((*(singleQueryCondition->attrFilter)).maxTokenNumber, (*(attrFilter)).maxTokenNumber);

    CompareQueryFeature(singleQueryCondition->queryFeature, queryFeature);
    EXPECT_EQ(singleQueryCondition->enableTimeFilter, enableTimeFilter);
    EXPECT_EQ(singleQueryCondition->extraMaskIsAtDevice, extraMaskIsAtDevice);
    EXPECT_EQ(singleQueryCondition->extraMaskLenEachQuery, extraMaskLenEachQuery);
    EXPECT_EQ(singleQueryCondition->topk, topk);
    EXPECT_EQ(singleQueryCondition->shareAttrFilter, shareAttrFilter);
}

TEST_F(TestOckVsaAnnQueryCondition, CreateQueryCond)
{
    BuildQueryCondition();
    EXPECT_EQ(queryCondition->extraMask, nullptr);
    for (uint32_t i = 0; i < queryBatchCount; ++i) {
        CompareQueryFeature((queryCondition->queryFeature) + i, queryFeature + i);
    }
    EXPECT_EQ((*(queryCondition->attrFilter)).maxTime, (*(attrFilter)).maxTime);
    EXPECT_EQ((*(queryCondition->attrFilter)).maxTokenId, (*(attrFilter)).maxTokenId);
    EXPECT_EQ((*(queryCondition->attrFilter)).maxTokenNumber, (*(attrFilter)).maxTokenNumber);
    EXPECT_EQ(queryCondition->enableTimeFilter, enableTimeFilter);
    EXPECT_EQ(queryCondition->extraMaskIsAtDevice, extraMaskIsAtDevice);
    EXPECT_EQ(queryCondition->extraMaskLenEachQuery, extraMaskLenEachQuery);
    EXPECT_EQ(queryCondition->topk, topk);
    EXPECT_EQ(queryCondition->shareAttrFilter, shareAttrFilter);
}

TEST_F(TestOckVsaAnnQueryCondition, BuildQueryFeatureVec)
{
    uint8_t startValue = 31;
    BuildSingleQuery(startValue);
    auto ret = singleQueryCondition->BuildQueryFeatureVec();
    CompareQueryFeature(ret.data(), queryFeature);
}

TEST_F(TestOckVsaAnnQueryCondition, QueryBatchCount)
{
    uint8_t startValue = 31;
    BuildQueryCondition(startValue);
    auto ret = queryCondition->QueryBatchCount();
    EXPECT_EQ(ret, queryBatchCount);
}

TEST_F(TestOckVsaAnnQueryCondition, UsingMask)
{
    uint8_t startValue = 31;
    BuildQueryCondition(startValue);
    auto ret = queryCondition->UsingMask();
    EXPECT_EQ(ret, false);
}

TEST_F(TestOckVsaAnnQueryCondition, QueryCondAt)
{
    uint8_t startValue = 31;
    uint32_t queryBatchPos = 2;
    BuildQueryCondition(startValue);
    auto ret = queryCondition->QueryCondAt(queryBatchPos);
    EXPECT_EQ(ret.batchPos, queryBatchPos);
    EXPECT_EQ(ret.queryFeature, queryFeature + queryBatchPos * DIMS * sizeof(uint8_t));
}

TEST_F(TestOckVsaAnnQueryCondition, SingleQueryCondPrint)
{
    BuildSingleQuery();
    std::string printInfo = "[batchPos is 0]=={'query features size is: 8, shareAttrFilter is: 1, topk is: 16, "
                            "extraMaskLenEachQuery is: 0, extraMaskIsAtDevice: 0, enableTimeFilter: 1}";
    std::stringstream ss;
    ss << *singleQueryCondition;
    EXPECT_EQ(ss.str(), printInfo);
}

TEST_F(TestOckVsaAnnQueryCondition, QueryCondPrint)
{
    BuildQueryCondition();
    std::string printInfo = "{queryBatchCount is: 4, 'query features size is: 8, shareAttrFilter is: 1, topk is: 16, "
                            "extraMaskLenEachQuery is: 0, extraMaskIsAtDevice: 0, enableTimeFilter: 1}";
    std::stringstream ss;
    ss << *queryCondition;
    EXPECT_EQ(ss.str(), printInfo);
}
}
}
}
}