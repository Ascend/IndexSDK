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
#include <random>
#include <iostream>
#include "ptest/ptest.h"
#include "mockcpp/mockcpp.hpp"
#include "acl/acl.h"
#include "ock/log/OckHcpsLogger.h"
#include "ock/vsa/neighbor/base/OckVsaAnnFactory.h"
#include "ock/vsa/neighbor/base/OckVsaAnnCreateParam.h"
#include "ock/vsa/neighbor/base/OckVsaAnnAddFeatureParam.h"
#include "ock/vsa/attr/OckTimeSpaceAttrTrait.h"
#include "ock/log/OckVsaHppLogger.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace test {
const uint64_t DIM_SIZE = 256;  // 256: dim
const uint32_t MAX_TIME_SCOPE = 256;
class TestOckVsaHPPIndex : public testing::Test {
public:
    void SetUp(void) override
    {
        aclInit(nullptr);
        aclrtSetDevice(deviceId);
        InitCpuSetVec();
    }
    void TearDown(void) override
    {
        aclrtResetDevice(deviceId);
        aclFinalize();
    }
    void InitCpuSetVec()
    {
        CPU_ZERO(&cpuSet);
        for (uint32_t i = 64U; i < 80U; ++i) {
            CPU_SET(i, &cpuSet);
        }
    }
    void BuildIndexBase()
    {
        OckVsaErrorCode errorCode = hmm::HMM_SUCCESS;
        auto factoryRegister =
            OckVsaAnnIndexFactoryRegister<int8_t, DIM_SIZE, 2U, attr::OckTimeSpaceAttrTrait>::Instance();
        auto factory = factoryRegister.GetFactory("HPPTS");
        auto param = OckVsaAnnCreateParam::Create(
            deviceId, maxFeatureRowCount, tokenNum, cpuSet, extKeyAttrsByteSize);
        indexBase = factory->Create(param, dftTrait, errorCode);
        ASSERT_TRUE(indexBase.get() != nullptr);
    }
    void GenerateFeature(uint64_t inputCount)
    {
        features = std::vector<int8_t>(DIM_SIZE * inputCount);
        uint64_t count = inputCount / 2ULL;
        auto timeGuard = fast::hdt::TestTimeGuard();
        uint64_t sampleCount = 262144UL;
        std::vector<int8_t> sampleFeature(sampleCount);
        GenerateRandFeatureSample(sampleFeature.data(), sampleCount);

        uint64_t stepCopyCount = sampleCount - 256UL;
        for (uint64_t i = 0; i < DIM_SIZE * count; i += stepCopyCount) {
            uint64_t copyCount = std::min(stepCopyCount, DIM_SIZE * count - i);
            uint64_t srcOffsetPos = rand() % (sampleCount - stepCopyCount);
            // 注意， memcpy_s这里第二个参数不能使用 DIM_SIZE * count, 因为很容易超过 memcpy_s的
            // size_t的限制，导致copy失败
            memcpy_s(features.data() + i, copyCount, sampleFeature.data() + srcOffsetPos, copyCount);
            memcpy_s(features.data() + i + count * DIM_SIZE, copyCount, sampleFeature.data() + srcOffsetPos, copyCount);
        }
        OCK_VSA_HPP_LOG_INFO("GenerateFeature Used Time=" << timeGuard.ElapsedMicroSeconds() << "us");
    }
    void GenerateRandFeatureSample(int8_t *sampleFeatures, uint32_t sampleSize)
    {
        for (uint64_t i = 0; i < sampleSize; ++i) {
            uint8_t value = rand() % std::numeric_limits<uint8_t>::max();
            sampleFeatures[i] = static_cast<int8_t>(value);
            // 0,1,...,127,0,1,...,127,
            // 1,2,...,128,1,2,...,128,...
            // sampleFeatures[i] = static_cast<int8_t>((i / (DIM_SIZE))%(DIM_SIZE/2UL) + i % (DIM_SIZE/2UL));
        }
    }
    void GenerateLable(uint64_t count)
    {
        auto timeGuard = fast::hdt::TestTimeGuard();
        for (uint64_t i = 0; i < count; ++i) {
            labels.push_back(i);
        }
        OCK_VSA_HPP_LOG_INFO("GenerateLable Used Time=" << timeGuard.ElapsedMicroSeconds() << "us");
    }
    void GenerateTimeSpaceAttr(uint64_t count)
    {
        auto timeGuard = fast::hdt::TestTimeGuard();
        for (uint64_t i = 0; i < count; ++i) {
            attrs.push_back((attr::OckTimeSpaceAttr(int32_t(i % MAX_TIME_SCOPE), uint32_t(i % 4U))));
        }
        OCK_VSA_HPP_LOG_INFO("GenerateLable Used Time=" << timeGuard.ElapsedMicroSeconds() << "us");
    }
    void GenerateCustormerAttr(uint64_t count)
    {
        auto timeGuard = fast::hdt::TestTimeGuard();
        for (uint64_t i = 0; i < count; ++i) {
            for (uint32_t j = 0; j < extKeyAttrsByteSize; ++j) {
                customAttr.push_back(j % std::numeric_limits<uint8_t>::max());
            }
        }
        OCK_VSA_HPP_LOG_INFO("GenerateLable Used Time=" << timeGuard.ElapsedMicroSeconds() << "us");
    }
    void GenerateData(uint64_t count)
    {
        GenerateFeature(count);
        GenerateLable(count);
        GenerateTimeSpaceAttr(count);
        GenerateCustormerAttr(count);
    }
    void BuildQueryFeatureVec(uint64_t count)
    {
        queryFeature = std::vector<int8_t>(count * DIM_SIZE);
        memcpy_s(queryFeature.data(), count * DIM_SIZE, features.data(), count * DIM_SIZE);
    }
    void GenerateWholeFilterQueryData(uint64_t count)
    {
        // 生成时空过滤条件
        int32_t minTime = 0;
        int32_t maxTime = MAX_TIME_SCOPE;
        attrFilter = std::make_shared<attr::OckTimeSpaceAttrTrait>(tokenNum);
        attrFilter->minTime = minTime;
        attrFilter->maxTime = maxTime;
        attrFilter->bitSet.Set(0U);
        attrFilter->bitSet.Set(1U);
        attrFilter->bitSet.Set(2U);
        attrFilter->bitSet.Set(3U);
        attrFilter->bitSet.Set(4U);
    }
    void GenerateSliceFilterQueryDataMiddle(uint64_t count)
    {
        // 生成时空过滤条件
        int32_t minTime = 0;
        int32_t maxTime = 50UL;
        attrFilter = std::make_shared<attr::OckTimeSpaceAttrTrait>(tokenNum);
        attrFilter->minTime = minTime;
        attrFilter->maxTime = maxTime;
        attrFilter->bitSet.Set(0U);
        attrFilter->bitSet.Set(1U);
    }
    void GenerateSliceFilterQueryDataLow(uint64_t count)
    {
        // 生成时空过滤条件
        int32_t minTime = 0;
        int32_t maxTime = 20UL;
        attrFilter = std::make_shared<attr::OckTimeSpaceAttrTrait>(tokenNum);
        attrFilter->minTime = minTime;
        attrFilter->maxTime = maxTime;
        attrFilter->bitSet.Set(0U);
        attrFilter->bitSet.Set(1U);
    }
    void GenerateLittleQueryData(uint64_t count)
    {
        // 生成时空过滤条件
        int32_t minTime = 0;
        int32_t maxTime = 2;
        attrFilter = std::make_shared<attr::OckTimeSpaceAttrTrait>(tokenNum);
        attrFilter->minTime = minTime;
        attrFilter->maxTime = maxTime;
        attrFilter->bitSet.Set(0U);
        attrFilter->bitSet.Set(1U);
    }

    void PrepareOutdata()
    {
        outLabels.resize(batch * topk, -1);
        outDistances.resize(batch * topk, -1);
        validNums.resize(batch, 1);
    }
    void TestAddFeature(void)
    {
        auto addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(ntotal,
            features.data(),
            reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()),
            labels.data(),
            customAttr.data());
        EXPECT_EQ(std::get<0>(*(addFeatureParam.attributes + 2U)).tokenId, 2U);
        EXPECT_EQ(addFeatureParam.count, ntotal);
        EXPECT_EQ(indexBase->AddFeature(addFeatureParam), hmm::HMM_SUCCESS);
    }
    void QueryFeature(void)
    {
        auto queryCondition = OckVsaAnnQueryCondition<int8_t, DIM_SIZE, attr::OckTimeSpaceAttrTrait>(batch,
            queryFeature.data(),
            attrFilter.get(),
            shareAttrFilter,
            topk,
            extraMask,
            extraMaskLenEachQuery,
            extraMaskIsAtDevice,
            enableTimeFilter);
        auto outResult = OckVsaAnnQueryResult<int8_t, attr::OckTimeSpaceAttrTrait>(
            batch, topk, outLabels.data(), outDistances.data(), validNums.data());

        auto timeGuard = fast::hdt::TestTimeGuard();
        EXPECT_EQ(indexBase->Search(queryCondition, outResult), hmm::HMM_SUCCESS);
        OCK_VSA_HPP_LOG_INFO("Search TopK=" << topk << " QPS= " << (1000000.0f / timeGuard.ElapsedMicroSeconds())
                                            << " feature count=" << indexBase->GetFeatureNum());

        for (uint32_t i = 0; i < topk; ++i) {
            std::cout << "lable:" << outLabels[i] << " dis:" << outDistances[i] << std::endl;
        }
    }
    void TestQueryWholeFilterFeature(void)
    {
        OCK_VSA_HPP_LOG_INFO("No data filter by TS condition");
        GenerateWholeFilterQueryData(batch);
        QueryFeature();
    }
    void TestQuerySliceFilterFeature(void)
    {
        OCK_VSA_HPP_LOG_INFO("Partiial data filter by TS condition");
        GenerateSliceFilterQueryDataMiddle(batch);  // 生成查询数据
        QueryFeature();
        GenerateSliceFilterQueryDataLow(batch);  // 生成查询数据
        QueryFeature();
    }
    void TestQueryFullFilterFeature(void)
    {
        OCK_VSA_HPP_LOG_INFO("Little data left, filter by TS condition");
        GenerateLittleQueryData(batch);  // 生成查询数据
        QueryFeature();
    }
    void TestQueryFeature(void)
    {
        BuildQueryFeatureVec(batch);
        TestQueryWholeFilterFeature();
        TestQueryWholeFilterFeature();
        TestQueryWholeFilterFeature();
        TestQuerySliceFilterFeature();
        TestQueryFullFilterFeature();
    }
    void TestCustormerAttrGet(void)
    {
        auto blockCount = indexBase->GetCustomAttrBlockCount();
        for (uint32_t i = 0; i < blockCount; ++i) {
            EXPECT_NE(indexBase->GetCustomAttrByBlockId(i), nullptr);
        }
    }

    uint64_t ntotal{262144ULL * 64ULL * 26ULL};
    uint32_t batch{1U};
    uint64_t maxFeatureRowCount{262144ULL * 64ULL * 26ULL};
    uint32_t deviceId{2U};
    cpu_set_t cpuSet;
    uint32_t tokenNum{2500};
    uint32_t extKeyAttrsByteSize{1};
    attr::OckTimeSpaceAttrTrait dftTrait{tokenNum};
    std::shared_ptr<OckVsaAnnIndexBase<int8_t, DIM_SIZE, 2U, attr::OckTimeSpaceAttrTrait>> indexBase;
    // 底库数据
    std::vector<int8_t> features;
    std::vector<int64_t> labels;
    std::vector<attr::OckTimeSpaceAttr> attrs;
    std::vector<uint8_t> customAttr;
    // 查询数据
    std::vector<int8_t> queryFeature;
    std::shared_ptr<attr::OckTimeSpaceAttrTrait> attrFilter;
    bool shareAttrFilter{true};
    uint32_t topk{200};
    uint8_t *extraMask = nullptr;
    uint64_t extraMaskLenEachQuery{0};
    bool extraMaskIsAtDevice{false};
    bool enableTimeFilter{true};
    // 结果数据
    std::vector<int64_t> outLabels;
    std::vector<float> outDistances;
    std::vector<uint32_t> validNums;
};  // namespace test

TEST_F(TestOckVsaHPPIndex, add_query)
{
    OckHmmSetLogLevel(OCK_LOG_LEVEL_INFO);
    BuildIndexBase();

    GenerateData(ntotal);  // 生成底库数据
    PrepareOutdata();
    TestAddFeature();
    TestQueryFeature();
    OckHmmSetLogLevel(OCK_LOG_LEVEL_ERROR);
}
}  // namespace test
}  // namespace neighbor
}  // namespace vsa
}  // namespace ock
