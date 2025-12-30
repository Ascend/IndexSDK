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
#include <iostream>
#include <fstream>
#include <securec.h>
#include "secodeFuzz.h"
#include "acl/acl.h"
#include "ock/vsa/neighbor/base/OckVsaAnnFactory.h"
#include "ock/vsa/neighbor/base/OckVsaAnnCreateParam.h"
#include "ock/vsa/neighbor/base/OckVsaAnnAddFeatureParam.h"
#include "ock/vsa/neighbor/base/OckVsaAnnIndexBase.h"
#include "ock/vsa/attr/OckTimeSpaceAttrTrait.h"
#include "ock/vsa/neighbor/base/OckVsaAnnQueryCondition.h"
#include "ock/vsa/neighbor/base/OckVsaAnnQueryResult.h"

namespace ock {
static constexpr size_t DF_FUZZ_EXEC_COUNT = 30;
static constexpr size_t DF_FUZZ_EXEC_SECOND = 10800;
const uint64_t DIM = 256; // 256: dim
const uint32_t MAX_TIME_SCOPE = 256;
const uint32_t MAX_TOKEN_SCOPE = 2500U;
const uint64_t MAX_FEATURE_ROW_COUNT = 262144ULL * 64ULL * 10ULL;
uint32_t g_extKeyAttrByteSize = 22U;
uint32_t g_deviceId = 0U;

std::shared_ptr<vsa::neighbor::OckVsaAnnIndexBase<int8_t, DIM, 2U, vsa::attr::OckTimeSpaceAttrTrait>> BuildIndexBase(
    std::string factoryName)
{
    vsa::OckVsaErrorCode errorCode = vsa::VSA_SUCCESS;
    auto facReg =
        vsa::neighbor::OckVsaAnnIndexFactoryRegister<int8_t, DIM, 2U, vsa::attr::OckTimeSpaceAttrTrait>::Instance();
    auto fac = facReg.GetFactory(factoryName);
    if (fac == nullptr) {
        std::cout << "nullptr!" << std::endl;
    }
    auto deviceInfo = std::make_shared<hmm::OckHmmDeviceInfo>();
    deviceInfo->deviceId = g_deviceId;
    CPU_SET(1U, &deviceInfo->cpuSet); // 设置1号CPU核
    CPU_SET(2U, &deviceInfo->cpuSet);
    CPU_SET(3U, &deviceInfo->cpuSet);
    CPU_SET(4U, &deviceInfo->cpuSet);
    auto param = vsa::neighbor::OckVsaAnnCreateParam::Create(deviceInfo->cpuSet, deviceInfo->deviceId,
        MAX_FEATURE_ROW_COUNT, 2500U, g_extKeyAttrByteSize);
    vsa::attr::OckTimeSpaceAttrTrait dftTrait{ 2500U };
    return fac->Create(param, dftTrait, errorCode);
}

std::shared_ptr<vsa::neighbor::OckVsaAnnIndexBase<int8_t, DIM, 2U, vsa::attr::OckTimeSpaceAttrTrait>> GetHppTsInstance()
{
    static auto hppTs = BuildIndexBase("HPPTS");
    return hppTs;
}
class FuzzTestOckVsaHPPIndex : public testing::Test {
public:
    void SetUp(void) override
    {
        DT_Set_Running_Time_Second(DF_FUZZ_EXEC_SECOND);
        DT_Enable_Leak_Check(0, 0);
        DT_Set_Report_Path("/home/pjy/vsa/tests/fuzz/build/report");
    }

    static void SetUpTestCase(void)
    {
        DT_Set_Running_Time_Second(DF_FUZZ_EXEC_SECOND);
    }
    void InitCpuSetVec()
    {
        CPU_ZERO(&cpuSet);
        for (uint32_t i = 64U; i < 80U; ++i) {
            CPU_SET(i, &cpuSet);
        }
    }

    std::vector<int> RandomPosGen(int maxData = 255, int minData = 0, int posNum = 20)
    {
        int seed = 333;
        srand(seed);
        // 生成 posNum 个随机整数并输出
        std::vector<int> randomResult;
        randomResult.resize(posNum);
        std::cout << "random position: ";
        for (int i = 0; i < posNum; i++) {
            int num = rand() % (maxData - minData + 1) + minData;
            randomResult[i] = num;
            std::cout << num << " ";
        }
        std::cout << std::endl;
        return randomResult;
    }

    std::vector<int8_t> TwoDimRandomInt8Gen(int lines, int columns)
    {
        int seed = 333;
        srand(seed);
        std::cout << "Start Random base data!" << std::endl;
        std::vector<int8_t> randomData;
        randomData.resize(lines * columns);
        for (int i = 0; i < lines * columns; i++) {
            randomData[i] = static_cast<int8_t>(rand() % (std::numeric_limits<int8_t>::max()));
        }
        std::cout << "Random base data generated finished!" << std::endl;
        return randomData;
    }

    void GenerateFeature(uint64_t inputCount)
    {
        features.resize(DIM * inputCount);
        uint64_t count = inputCount / 2ULL;
        uint64_t sampleCount = 262144UL;
        std::vector<int8_t> sampleFeature(sampleCount);
        GenerateRandFeatureSample(sampleFeature.data(), sampleCount);

        uint64_t stepCopyCount = sampleCount - 256UL;
        for (uint64_t i = 0; i < DIM * count; i += stepCopyCount) {
            uint64_t copyCount = std::min(stepCopyCount, DIM * count - i);
            uint64_t srcOffsetPos = rand() % (sampleCount - stepCopyCount);
            // 注意， memcpy_s这里第二个参数不能使用 DIM_SIZE * count, 因为很容易超过 memcpy_s的
            // size_t的限制，导致copy失败
            memcpy_s(features.data() + i, copyCount, sampleFeature.data() + srcOffsetPos, copyCount);
            memcpy_s(features.data() + i + count * DIM, copyCount, sampleFeature.data() + srcOffsetPos, copyCount);
        }
    }
    void GenerateRandFeatureSample(int8_t *sampleFeatures, uint32_t sampleSize)
    {
        for (uint64_t i = 0; i < sampleSize; ++i) {
            uint8_t value = rand() % std::numeric_limits<uint8_t>::max();
            sampleFeatures[i] = static_cast<int8_t>(value);
        }
    }

    void GenerateLabel(uint64_t count)
    {
        labels.resize(count);
        for (uint64_t i = 0; i < count; ++i) {
            labels[i] = i;
        }
    }
    void GenerateTimeSpaceAttr(uint64_t count)
    {
        attrs.resize(count);
        for (uint64_t i = 0; i < count; ++i) {
            attrs[i] = vsa::attr::OckTimeSpaceAttr(int32_t(i % MAX_TIME_SCOPE), uint32_t(i % MAX_TOKEN_SCOPE));
        }
    }
    void GenerateCustormerAttr(uint64_t count)
    {
        customAttr.resize(count * g_extKeyAttrByteSize);
        for (uint64_t i = 0; i < count; ++i) {
            for (uint32_t j = 0; j < g_extKeyAttrByteSize; ++j) {
                customAttr[i * g_extKeyAttrByteSize + j] = j % std::numeric_limits<uint8_t>::max();
            }
        }
    }
    void GenerateData(uint32_t addCount, uint32_t baseNum)
    {
        GenerateFeature(baseNum);
        GenerateLabel(addCount * baseNum);
        GenerateTimeSpaceAttr(addCount * baseNum);
        GenerateCustormerAttr(addCount * baseNum);
    }

    void PrepareOutdata()
    {
        outLabels.resize(batch * topk, -1);
        outDistances.resize(batch * topk, -1);
        validNums.resize(batch, 1);
    }

    void BuildQueryFeatureVec(uint64_t count)
    {
        queryFeature = std::vector<int8_t>(count * DIM);
        memcpy_s(queryFeature.data(), count * DIM, features.data(), count * DIM);
    }

    void GenerateWholeFilterQueryData()
    {
        // 生成时空过滤条件
        int32_t minTime = 0;
        int32_t maxTime = MAX_TIME_SCOPE;
        attrFilter = std::make_shared<vsa::attr::OckTimeSpaceAttrTrait>(tokenNum);
        attrFilter->minTime = minTime;
        attrFilter->maxTime = maxTime;
        attrFilter->bitSet.Set(0U);
        attrFilter->bitSet.Set(1U);
        attrFilter->bitSet.Set(2U);
        attrFilter->bitSet.Set(3U);
        attrFilter->bitSet.Set(4U);
    }
    void BuildIndexBaseInner()
    {
        InitCpuSetVec();
        vsa::OckVsaErrorCode errorCode = hmm::HMM_SUCCESS;
        auto factoryRegister =
            vsa::neighbor::OckVsaAnnIndexFactoryRegister<int8_t, DIM, 2U, vsa::attr::OckTimeSpaceAttrTrait>::Instance();
        auto factory = factoryRegister.GetFactory("HPPTS");
        auto param = vsa::neighbor::OckVsaAnnCreateParam::Create(cpuSet, deviceId, maxFeatureRowCount, tokenNum,
            g_extKeyAttrByteSize);
        indexBase = factory->Create(param, dftTrait, errorCode);
        ASSERT_TRUE(indexBase.get() != nullptr);
    }
    void TestAddFeature(uint32_t addCount, uint32_t baseNum)
    {
        std::cout << "features.size is " << features.size() << ", attrs.size is " << attrs.size() <<
            ", labels.size is " << labels.size() << ", customAttr.size is " << customAttr.size() << std::endl;
        for (int i = 0; i < addCount; ++i) {
            std::cout << "add feature batch " << i << std::endl;
            auto addFeatureParam = vsa::neighbor::OckVsaAnnAddFeatureParam<int8_t, vsa::attr::OckTimeSpaceAttrTrait>(
                baseNum, features.data(),
                reinterpret_cast<vsa::attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data() + i * baseNum),
                labels.data() + i * baseNum, customAttr.data() + i * baseNum);

            EXPECT_EQ(indexBase->AddFeature(addFeatureParam), hmm::HMM_SUCCESS);
            addFeatureParam.Shift(0, DIM, g_extKeyAttrByteSize);
        }
    }

    void TestQueryFeature(void)
    {
        PrepareOutdata();
        GenerateWholeFilterQueryData();
        BuildQueryFeatureVec(batch);
        auto queryCondition = vsa::neighbor::OckVsaAnnQueryCondition<int8_t, DIM, vsa::attr::OckTimeSpaceAttrTrait>(
            batch, queryFeature.data(), attrFilter.get(), shareAttrFilter, topk, extraMask, extraMaskLenEachQuery,
            extraMaskIsAtDevice, enableTimeFilter);
        auto outResult = vsa::neighbor::OckVsaAnnQueryResult<int8_t, vsa::attr::OckTimeSpaceAttrTrait>(batch, topk,
            outLabels.data(), outDistances.data(), validNums.data());

        EXPECT_EQ(indexBase->Search(queryCondition, outResult), hmm::HMM_SUCCESS);
    }

    void TestGetFeatureNum(uint64_t expectedValue)
    {
        uint64_t baseNum = indexBase->GetFeatureNum();
        EXPECT_EQ(baseNum, expectedValue);
    }

    void TestGetFeatureByLabel(const int64_t *labels, uint64_t count = 1)
    {
        std::vector<int8_t> outFeatures(count * DIM);
        EXPECT_EQ(indexBase->GetFeatureByLabel(count, labels, outFeatures.data()), hmm::HMM_SUCCESS);
    }

    void TestGetFeatureAttrByLabel(const int64_t *labels, uint64_t count = 1)
    {
        std::vector<vsa::attr::OckTimeSpaceAttr> outAttrs(count);
        EXPECT_EQ(indexBase->GetFeatureAttrByLabel(count, labels,
            reinterpret_cast<vsa::attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(outAttrs.data())),
            hmm::HMM_SUCCESS);
        for (uint32_t i = 0; i < count; ++i) {
            EXPECT_EQ(outAttrs[i].time, labels[i] % MAX_TIME_SCOPE);
            EXPECT_EQ(outAttrs[i].tokenId, labels[i] % MAX_TOKEN_SCOPE);
        }
    }

    void TestGetCustomAttrByBlockId(uint32_t blockId)
    {
        auto errorCode = vsa::VSA_SUCCESS;
        uintptr_t address = indexBase->GetCustomAttrByBlockId(0, errorCode);
        EXPECT_EQ(errorCode, vsa::VSA_SUCCESS);
    }

    void TestGetCustomAttrBlockCount(uint32_t expectedBlockNum)
    {
        EXPECT_EQ(indexBase->GetCustomAttrBlockCount(), expectedBlockNum);
    }

    void TestDeleteFeatureByLabel(uint64_t count, const int64_t *labels)
    {
        EXPECT_EQ(indexBase->DeleteFeatureByLabel(count, labels), hmm::HMM_SUCCESS);
    }

    void TestDeleteFeatureByToken(uint64_t count, const uint32_t *tokens)
    {
        EXPECT_EQ(indexBase->DeleteFeatureByToken(count, tokens), hmm::HMM_SUCCESS);
    }

    uint64_t initialNumber{ 1000000 };
    uint32_t batch{ 1U };
    uint64_t minFeatureRowCount{ 10U };
    uint64_t maxFeatureRowCount = 1000000;
    uint64_t minAddCount{ 10U };
    uint64_t maxAddCount{ 60U };
    uint32_t deviceId{ 0U };
    int32_t baseNum{ 1000000 };
    int32_t addCount{ 50 };
    cpu_set_t cpuSet;
    uint32_t tokenNum{ 2500 };
    vsa::attr::OckTimeSpaceAttrTrait dftTrait{ tokenNum };
    std::shared_ptr<vsa::neighbor::OckVsaAnnIndexBase<int8_t, DIM, 2U, vsa::attr::OckTimeSpaceAttrTrait>> indexBase;
    uint32_t blockBase = 262144UL;
    // 底库数据
    std::vector<int8_t> features;
    std::vector<int64_t> labels;
    std::vector<vsa::attr::OckTimeSpaceAttr> attrs;
    std::vector<uint8_t> customAttr;
    // 查询数据
    std::vector<int8_t> queryFeature;
    std::shared_ptr<vsa::attr::OckTimeSpaceAttrTrait> attrFilter;
    bool shareAttrFilter{ true };
    uint32_t topk{ 200 };
    uint8_t *extraMask = nullptr;
    uint64_t extraMaskLenEachQuery{ 0 };
    bool extraMaskIsAtDevice{ false };
    bool enableTimeFilter{ true };
    // 结果数据
    std::vector<int64_t> outLabels;
    std::vector<float> outDistances;
    std::vector<uint32_t> validNums;
};

TEST_F(FuzzTestOckVsaHPPIndex, add)
{
    std::string name = "add";
    OckHmmSetLogLevel(OCK_LOG_LEVEL_INFO);
    indexBase = GetHppTsInstance();
    ASSERT_NE(indexBase, nullptr);

    uint32_t seed = 0;
    DT_FUZZ_START(seed, 1U, const_cast<char *>(name.c_str()), 0)
    {
        std::cout << "====================add====================" << std::endl;
        GenerateData(addCount, baseNum);

        TestAddFeature(addCount, baseNum);
        TestGetFeatureNum(addCount * baseNum);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckVsaHPPIndex, search)
{
    std::string name = "search";
    OckHmmSetLogLevel(OCK_LOG_LEVEL_INFO);
    indexBase = GetHppTsInstance();
    ASSERT_NE(indexBase, nullptr);

    uint32_t seed = 0;
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, const_cast<char *>(name.c_str()), 0)
    {
        indexBase = GetHppTsInstance();
        ASSERT_NE(indexBase, nullptr);
        TestQueryFeature();
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckVsaHPPIndex, getFeatureNum)
{
    std::string name = "get_feature_num";
    OckHmmSetLogLevel(OCK_LOG_LEVEL_INFO);
    indexBase = GetHppTsInstance();
    ASSERT_NE(indexBase, nullptr);

    uint32_t seed = 0;
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, const_cast<char *>(name.c_str()), 0)
    {
        TestGetFeatureNum(baseNum * addCount);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckVsaHPPIndex, getFeatureByLabel)
{
    std::string name = "get_feature_by_label";
    OckHmmSetLogLevel(OCK_LOG_LEVEL_INFO);
    indexBase = GetHppTsInstance();
    ASSERT_NE(indexBase, nullptr);

    uint32_t seed = 0;
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, const_cast<char *>(name.c_str()), 0)
    {
        s32 label = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 0U, 0U, baseNum * addCount - 1U);
        std::vector<int64_t> labels{ label, baseNum * addCount - 1U};
        TestGetFeatureByLabel(labels.data(), labels.size());
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckVsaHPPIndex, GetFeatureAttrByLabel)
{
    std::string name = "get_feature_attr_by_label";

    OckHmmSetLogLevel(OCK_LOG_LEVEL_INFO);
    indexBase = GetHppTsInstance();
    ASSERT_NE(indexBase, nullptr);

    uint32_t seed = 0;
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, const_cast<char *>(name.c_str()), 0)
    {
        s32 label = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 0U, 0U, baseNum * addCount - 1U);
        std::vector<int64_t> labels{ label,  baseNum * addCount - 1U};
        TestGetFeatureAttrByLabel(labels.data(), labels.size());
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckVsaHPPIndex, GetCustomAttrByBlockId)
{
    std::string name = "get_custom_attr_by_blockId";

    OckHmmSetLogLevel(OCK_LOG_LEVEL_INFO);
    indexBase = GetHppTsInstance();
    ASSERT_NE(indexBase, nullptr);

    uint32_t seed = 0;
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, const_cast<char *>(name.c_str()), 0)
    {
        uint32_t blockNum = (addCount * baseNum + blockBase - 1) / blockBase;
        s32 blockId = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 0U, 0U, blockNum - 1);
        TestGetCustomAttrByBlockId(blockId);
        TestGetCustomAttrByBlockId(blockNum - 1);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckVsaHPPIndex, GetCustomAttrBlockCount)
{
    std::string name = "get_custom_attr_block_count";

    OckHmmSetLogLevel(OCK_LOG_LEVEL_INFO);
    indexBase = GetHppTsInstance();
    ASSERT_NE(indexBase, nullptr);

    uint32_t seed = 0;
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, const_cast<char *>(name.c_str()), 0)
    {
        uint32_t blockNum = (addCount * baseNum + blockBase - 1) / blockBase;
        TestGetCustomAttrBlockCount(blockNum);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckVsaHPPIndex, DeleteFeatureByLabel)
{
    std::string name = "delete_feature_by_label";

    OckHmmSetLogLevel(OCK_LOG_LEVEL_INFO);
    indexBase = GetHppTsInstance();
    ASSERT_NE(indexBase, nullptr);

    uint32_t seed = 0;
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, const_cast<char *>(name.c_str()), 0)
    {
        s32 label = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 0U, 0U, baseNum * addCount - 1U);
        std::vector<int64_t> labels{ label };
        TestDeleteFeatureByLabel(labels.size(), labels.data());
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckVsaHPPIndex, DeleteFeatureByToken)
{
    std::string name = "delete_feature_by_token";

    OckHmmSetLogLevel(OCK_LOG_LEVEL_INFO);
    indexBase = GetHppTsInstance();
    ASSERT_NE(indexBase, nullptr);

    uint32_t seed = 0;
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, const_cast<char *>(name.c_str()), 0)
    {
        s32 tokenId = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 0U, 0U, MAX_TOKEN_SCOPE - 1U);
        std::vector<uint32_t> tokens{ tokenId };
        TestDeleteFeatureByToken(tokens.size(), tokens.data());
    }
    DT_FUZZ_END()
}
}