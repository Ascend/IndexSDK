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
#include <numeric>
#include <iostream>
#include <sys/time.h>
#include <sys/stat.h>
#include <thread>
#include <chrono>
#include "acl/acl.h"
#include "ock/vsa/neighbor/base/OckVsaAnnFactory.h"
#include "ock/vsa/neighbor/base/OckVsaAnnCreateParam.h"
#include "ock/vsa/neighbor/base/OckVsaAnnAddFeatureParam.h"
#include "ock/vsa/attr/OckTimeSpaceAttrTrait.h"


namespace ock {
namespace vsa {
namespace neighbor {
namespace test1 {
const int DIMS = 256; // 256: dim
const uint64_t MAX_FEATURE_ROW_COUNT = 262144ULL * 64ULL * 3ULL;
const uint32_t EXTKEY_ATTR_BYTE_SIZE = 22U;
const uint32_t EXTKEY_ATTR_BLOCK_SIZE = 262144U;
const uint32_t DEVICE_ID = 1U;

inline std::shared_ptr<OckVsaAnnIndexBase<int8_t, DIMS, 2U, attr::OckTimeSpaceAttrTrait>> BuildIndexBase(
    std::string factoryName)
{
    OckVsaErrorCode errorCode = VSA_SUCCESS;
    auto facReg = OckVsaAnnIndexFactoryRegister<int8_t, DIMS, 2U, attr::OckTimeSpaceAttrTrait>::Instance();
    auto fac = facReg.GetFactory(factoryName);
    if (fac == nullptr) {
        std::cout << "nullptr!" << std::endl;
    }
    auto deviceInfo = std::make_shared<hmm::OckHmmDeviceInfo>();
    deviceInfo->deviceId = DEVICE_ID;
    CPU_SET(1U, &deviceInfo->cpuSet); // 设置1号CPU核
    CPU_SET(2U, &deviceInfo->cpuSet);
    CPU_SET(3U, &deviceInfo->cpuSet);
    CPU_SET(4U, &deviceInfo->cpuSet);
    auto param = OckVsaAnnCreateParam::Create(deviceInfo->cpuSet, deviceInfo->deviceId, MAX_FEATURE_ROW_COUNT, 2500U,
        EXTKEY_ATTR_BYTE_SIZE, EXTKEY_ATTR_BLOCK_SIZE);
    attr::OckTimeSpaceAttrTrait dftTrait{ 2500U };
    return fac->Create(param, dftTrait, errorCode);
}

inline std::shared_ptr<OckVsaAnnIndexBase<int8_t, DIMS, 2U, attr::OckTimeSpaceAttrTrait>> GetNpuTsInstance()
{
    static auto npuTs = BuildIndexBase("NPUTS");
    return npuTs;
}

inline std::shared_ptr<OckVsaAnnIndexBase<int8_t, DIMS, 2U, attr::OckTimeSpaceAttrTrait>> GetHppTsInstance()
{
    static auto hppTs = BuildIndexBase("HPPTS");
    return hppTs;
}

inline double GetMilliSecs()
{
    struct timeval tv = { 0, 0 };
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

class TestOckVsaHppIndexDelete : public testing::Test {
public:
    void SetUp(void) override
    {
        aclrtSetDevice(DEVICE_ID);
    }
    void TearDown(void) override
    {
        aclrtResetDevice(DEVICE_ID);
        std::this_thread::sleep_for(std::chrono::seconds(5U));
    }

    void GenerateNormalFeature(uint64_t count)
    {
        double ts = GetMilliSecs();
        for (uint64_t i = 0; i < DIMS * count; ++i) {
            features.push_back(i % 8UL + 1);
        }
        double te = GetMilliSecs();
        std::cout << "Generate base time: " << (te - ts) << " ms." << std::endl;

        // 生成底库labels
        labels.resize(count, 0);
        for (uint64_t i = 0; i < count; ++i) {
            labels[i] = i;
        }
        // 生成底库时空属性
        attrs.resize(count);
        for (uint64_t i = 0; i < count; ++i) {
            attrs[i] = attr::OckTimeSpaceAttr(int32_t(i % 4U), uint32_t(i % tokenNum));
        }
    }

    void GenerateRandFeature(uint64_t count)
    {
        double ts = GetMilliSecs();
        std::default_random_engine e(time(nullptr));
        std::uniform_real_distribution<float> rCode(0.0f, 1.0f);
        size_t maxSize = DIMS * count;
        std::independent_bits_engine<std::mt19937, 8U, uint8_t> engine(1);

        features.resize(maxSize);
        for (size_t i = 0; i < maxSize; ++i) {
            features[i] = engine();
        }

        double te = GetMilliSecs();
        std::cout << "Generate base time: " << (te - ts) << " ms." << std::endl;

        // 生成底库labels
        labels.resize(count, 0);
        for (uint64_t i = 0; i < count; ++i) {
            labels[i] = i;
        }
        // 生成底库时空属性
        attrs.resize(count);
        for (uint64_t i = 0; i < count; ++i) {
            attrs[i] = attr::OckTimeSpaceAttr(int32_t(i % 4U), uint32_t(i % tokenNum));
        }
    }

    void GenerateGetLabelFeature(uint64_t count)
    {
        double ts = GetMilliSecs();
        features.resize(count * DIMS, 0);
        for (uint64_t i = 0; i < DIMS * count; ++i) {
            features[i] = (i / DIMS) % 128ULL;
        }
        double te = GetMilliSecs();
        std::cout << "Generate base time: " << (te - ts) << " ms." << std::endl;

        // 生成底库labels
        labels.resize(count, 0);
        for (uint64_t i = 0; i < count; ++i) {
            labels[i] = i;
        }
        // 生成底库时空属性
        attrs.resize(count);
        for (uint64_t i = 0; i < count; ++i) {
            attrs[i] = attr::OckTimeSpaceAttr(int32_t(i % 4U), uint32_t(i % tokenNum));
        }
    }

    void GeneratePerfFeature(uint64_t count)
    {
        uint64_t sampleCount = 50000ULL;
        std::vector<int8_t> sampleFeature(sampleCount);
        GenerateRandFeatureSample(sampleFeature.data(), sampleCount);

        uint64_t stepCopyCount = 48000UL;
        features = std::vector<int8_t>(DIMS * count);
        for (uint64_t i = 0; i < DIMS * count; i += stepCopyCount) {
            uint64_t copyCount = std::min(stepCopyCount, DIMS * count - i);
            uint64_t srcOffsetPos = rand() % (sampleCount - stepCopyCount);
            // 注意， memcpy_s这里第二个参数不能使用 DIM_SIZE * count, 因为很容易超过 memcpy_s的
            // size_t的限制，导致copy失败
            memcpy_s(features.data() + i, copyCount, sampleFeature.data() + srcOffsetPos, copyCount);
        }

        // 生成底库labels
        labels.resize(count, 0);
        for (uint64_t i = 0; i < count; ++i) {
            labels[i] = i;
        }
        // 生成底库时空属性
        attrs.resize(count);
        for (uint64_t i = 0; i < count; ++i) {
            attrs[i] = attr::OckTimeSpaceAttr(int32_t(i % 4U), uint32_t(i % tokenNum));
        }
    }

    void GenerateRandFeatureSample(int8_t *sampleFeatures, uint32_t sampleSize)
    {
        for (uint64_t i = 0; i < sampleSize; ++i) {
            sampleFeatures[i] = static_cast<int8_t>(rand());
        }
    }

    void GenerateQueryData(uint64_t count, int seed = 5678)
    {
        // 生成查询向量
        std::default_random_engine e(seed);
        std::uniform_real_distribution<float> rCode(0.0f, 1.0f);
        for (size_t i = 0; i < count * DIMS; ++i) {
            queryFeature.push_back(i % 8UL + 1);
        }
        // 生成时空过滤条件
        for (uint32_t i = 0; i < count; ++i) {
            int32_t minTime = 0;
            int32_t maxTime = 100;
            attrFilter.push_back(attr::OckTimeSpaceAttrTrait(2500U));
            attrFilter[i].minTime = minTime;
            attrFilter[i].maxTime = maxTime;
            // 本测试文件tokenId数量为2048(确保可以整除底库数量，每个tokenId对应等量label)
            for (uint32_t j = 0; j < 2048U; ++j) {
                attrFilter[i].bitSet.Set(j);
            }
        }
    }

    void GenerateExtraMask(uint64_t count, uint32_t queryNum)
    {
        extraMaskLenEachQuery = (count + 7ULL) / 8ULL;
        extraMask.resize(extraMaskLenEachQuery * queryNum, 255U);
    }

    void GenerateCustomAttr(uint64_t count)
    {
        uint32_t customAttrBlockNum = (count + EXTKEY_ATTR_BLOCK_SIZE - 1) / EXTKEY_ATTR_BLOCK_SIZE;
        customAttr.resize(customAttrBlockNum * EXTKEY_ATTR_BLOCK_SIZE * EXTKEY_ATTR_BYTE_SIZE);
        for (uint32_t i = 0; i < customAttrBlockNum; ++i) {
            for (uint32_t j = 0; j < EXTKEY_ATTR_BLOCK_SIZE * EXTKEY_ATTR_BYTE_SIZE; ++j) {
                customAttr[i * EXTKEY_ATTR_BLOCK_SIZE * EXTKEY_ATTR_BYTE_SIZE + j] = i + j % EXTKEY_ATTR_BYTE_SIZE;
            }
        }
    }

    void PrepareOutData(uint32_t queryBatch, uint32_t queryTopK)
    {
        outLabels.resize(queryBatch * queryTopK, -1);
        outDistances.resize(queryBatch * queryTopK, -1);
        validNums.resize(queryBatch, 0);
    }

    // 删除
    std::vector<int64_t> GenerateRangeLabels(int64_t start, int64_t end)
    {
        std::vector<int64_t> labelVec;
        for (size_t i = start; i < end; i++) {
            labelVec.push_back(i);
        }
        return labelVec;
    }

    std::vector<int64_t> GenerateSampleLabels(int64_t start, int64_t end, int64_t num)
    {
        // 确保离散采样效率
        assert(num <= (end - start) / 2UL);
        std::random_device seed;
        std::ranlux48 engine(seed());
        std::uniform_int_distribution<> distrib(start, end);
        std::unordered_set<int64_t> labelSet;
        while (labelSet.size() < num) {
            labelSet.insert(distrib(engine));
        }
        return std::vector<int64_t>(std::make_move_iterator(labelSet.begin()), std::make_move_iterator(labelSet.end()));
    }

    void DeleteAllBaseByToken(
        std::shared_ptr<OckVsaAnnIndexBase<int8_t, DIMS, 2U, attr::OckTimeSpaceAttrTrait>> indexBase, uint64_t count,
        uint64_t beginPos)
    {
        std::vector<uint32_t> deleteTokens(count);
        std::iota(deleteTokens.begin(), deleteTokens.end(), beginPos);
        EXPECT_EQ(indexBase->DeleteFeatureByToken(count, deleteTokens.data()), VSA_SUCCESS);
        EXPECT_EQ(indexBase->GetFeatureNum(), 0);
        std::cout << "DeleteAllBaseByToken SUCCESS" << std::endl;
    }

    uint64_t nTotal{ 262144ULL }; // 特征底库数量
    uint32_t batch{ 1U };         // 搜索条数
    uint32_t tokenNum{ 2048 };    // 时空属性数量
    // 底库数据
    std::vector<int8_t> features;              // 底库数据
    std::vector<int64_t> labels;               // 底库labels
    std::vector<attr::OckTimeSpaceAttr> attrs; // 底库时空属性
    std::vector<uint8_t> customAttr;           // 自定义属性
    // 查询数据
    std::vector<int8_t> queryFeature;
    std::vector<attr::OckTimeSpaceAttrTrait> attrFilter;
    bool shareAttrFilter{ true }; //  是否共享属性
    uint32_t topK{ 16 };
    std::vector<uint8_t> extraMask;
    uint64_t extraMaskLenEachQuery{ nTotal / 8 };
    bool extraMaskIsAtDevice{ false };
    bool enableTimeFilter{ true };
    // 结果数据
    std::vector<int64_t> outLabels;
    std::vector<float> outDistances;
    std::vector<uint32_t> validNums;
};

// 测试删除一半label后的查询添加场景
TEST_F(TestOckVsaHppIndexDelete, AccIndex_HPPTS_DeleteFeatureByLabel_Delete_Half1)
{
    auto hppTs = GetHppTsInstance();
    GeneratePerfFeature(nTotal * 64U * 2U); // 生成底库数据
    GenerateCustomAttr(nTotal * 64U * 2U);
    GenerateQueryData(batch); // 生成查询数据
    PrepareOutData(batch, topK);

    // 准备待添加的底库数据
    auto addFeatureParam =
        OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal * 64U * 2U, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr.data());
    // 添加底库
    EXPECT_EQ(hppTs->GetFeatureNum(), 0);
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), VSA_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal * 64U * 2U);

    // 删除偶数label的底库, 循环删除，每次100w
    std::vector<int64_t> deleteLabels(hppTs->GetFeatureNum() / 2U);
    for (uint32_t i = 0; i < deleteLabels.size(); ++i) {
        deleteLabels[i] = i * 2U;
    }
    uint32_t deletedNum = 0;
    uint32_t leftNumInDeleteLabels = deleteLabels.size();
    while (leftNumInDeleteLabels > 0) {
        uint32_t removeSize = leftNumInDeleteLabels < 1000000U ? leftNumInDeleteLabels : 1000000;
        EXPECT_EQ(hppTs->DeleteFeatureByLabel(removeSize, deleteLabels.data() + deletedNum), VSA_SUCCESS);
        deletedNum += removeSize;
        leftNumInDeleteLabels -= removeSize;
    }
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal * 64U);
    std::vector<int8_t> result(deleteLabels.size() * DIMS);
    EXPECT_EQ(hppTs->GetFeatureByLabel(1, deleteLabels.data(), result.data()), VSA_ERROR_INVALID_OUTTER_LABEL);
    std::vector<attr::OckTimeSpaceAttrTrait::KeyTypeTuple> attrResult(deleteLabels.size());
    EXPECT_EQ(hppTs->GetFeatureAttrByLabel(1, deleteLabels.data(), attrResult.data()), VSA_ERROR_INVALID_OUTTER_LABEL);

    // 准备查询数据
    auto queryCondition = OckVsaAnnQueryCondition<int8_t, DIMS, attr::OckTimeSpaceAttrTrait>(batch, queryFeature.data(),
        attrFilter.data(), shareAttrFilter, topK, nullptr, extraMaskLenEachQuery, extraMaskIsAtDevice,
        enableTimeFilter);
    auto outResult = OckVsaAnnQueryResult<int8_t, attr::OckTimeSpaceAttrTrait>(batch, topK, outLabels.data(),
        outDistances.data(), validNums.data());
    // 查询
    EXPECT_EQ(hppTs->Search(queryCondition, outResult), hmm::HMM_SUCCESS);
    for (uint i = 0; i < batch * topK; ++i) {
        EXPECT_NE(outResult.labels[i] % 2U, 0);
    }

    // 添加底库
    std::vector<int64_t> newLabels(nTotal * 64ULL);
    std::iota(newLabels.begin(), newLabels.end(), nTotal * 64ULL * 2ULL);
    addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal * 64U, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), newLabels.data(),
        customAttr.data());
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), VSA_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal * 64ULL * 2ULL);
    EXPECT_EQ(hppTs->GetFeatureByLabel(1, newLabels.data(), result.data()), VSA_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureAttrByLabel(1, newLabels.data(), attrResult.data()), VSA_SUCCESS);
    EXPECT_EQ(hppTs->Search(queryCondition, outResult), hmm::HMM_SUCCESS);
    DeleteAllBaseByToken(hppTs, 2048ULL, 0);
}

// 测试删除一半label后重复添加回来的场景
TEST_F(TestOckVsaHppIndexDelete, AccIndex_HPPTS_DeleteFeatureByLabel_Delete_Half2)
{
    auto hppTs = GetHppTsInstance();
    GeneratePerfFeature(nTotal * 64U * 2U); // 生成底库数据
    GenerateCustomAttr(nTotal * 64U * 2U);
    GenerateQueryData(batch); // 生成查询数据
    PrepareOutData(batch, topK);

    // 准备待添加的底库数据
    auto addFeatureParam =
        OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal * 64U * 2U, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr.data());
    EXPECT_EQ(std::get<0>(*(addFeatureParam.attributes + 2U)).tokenId, 2U);
    EXPECT_EQ(addFeatureParam.count, nTotal * 64U * 2U);

    // 添加底库
    EXPECT_EQ(hppTs->GetFeatureNum(), 0);
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), VSA_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal * 64U * 2U);

    // 删除偶数label的底库, 循环删除，每次100w
    std::vector<int64_t> deleteLabels(hppTs->GetFeatureNum() / 2U);
    for (uint32_t i = 0; i < deleteLabels.size(); ++i) {
        deleteLabels[i] = i * 2U;
    }
    uint32_t deletedNum = 0;
    uint32_t leftNumInDeleteLabels = deleteLabels.size();
    while (leftNumInDeleteLabels > 0) {
        uint32_t removeSize = leftNumInDeleteLabels < 1000000U ? leftNumInDeleteLabels : 1000000;
        EXPECT_EQ(hppTs->DeleteFeatureByLabel(removeSize, deleteLabels.data() + deletedNum), VSA_SUCCESS);
        deletedNum += removeSize;
        leftNumInDeleteLabels -= removeSize;
    }
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal * 64U);
    std::vector<int8_t> result(deleteLabels.size() * DIMS);
    EXPECT_EQ(hppTs->GetFeatureByLabel(1, deleteLabels.data(), result.data()), VSA_ERROR_INVALID_OUTTER_LABEL);
    std::vector<attr::OckTimeSpaceAttrTrait::KeyTypeTuple> attrResult(deleteLabels.size());
    EXPECT_EQ(hppTs->GetFeatureAttrByLabel(1, deleteLabels.data(), attrResult.data()), VSA_ERROR_INVALID_OUTTER_LABEL);

    // 准备查询数据
    auto queryCondition = OckVsaAnnQueryCondition<int8_t, DIMS, attr::OckTimeSpaceAttrTrait>(batch, queryFeature.data(),
        attrFilter.data(), shareAttrFilter, topK, nullptr, extraMaskLenEachQuery, extraMaskIsAtDevice,
        enableTimeFilter);
    auto outResult = OckVsaAnnQueryResult<int8_t, attr::OckTimeSpaceAttrTrait>(batch, topK, outLabels.data(),
        outDistances.data(), validNums.data());
    // 查询
    EXPECT_EQ(hppTs->Search(queryCondition, outResult), hmm::HMM_SUCCESS);
    for (uint i = 0; i < batch * topK; ++i) {
        EXPECT_NE(outResult.labels[i] % 2U, 0);
    }

    // 添加底库
    addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal * 64U, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), deleteLabels.data(),
        customAttr.data());
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), VSA_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal * 64ULL * 2ULL);
    EXPECT_EQ(hppTs->GetFeatureByLabel(1, deleteLabels.data(), result.data()), VSA_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureAttrByLabel(1, deleteLabels.data(), attrResult.data()), VSA_SUCCESS);
    EXPECT_EQ(hppTs->Search(queryCondition, outResult), hmm::HMM_SUCCESS);
    DeleteAllBaseByToken(hppTs, 2048ULL, 0);
}


// 测试删除连续的一半token后新增查询的场景
TEST_F(TestOckVsaHppIndexDelete, AccIndex_HPPTS_DeleteFeatureByToken_Delete_Half_Consecutive)
{
    auto hppTs = GetHppTsInstance();
    GeneratePerfFeature(nTotal * 64U * 2U); // 生成底库数据
    GenerateCustomAttr(nTotal * 64U * 2U);
    GenerateQueryData(batch); // 生成查询数据
    PrepareOutData(batch, topK);

    // 准备待添加的底库数据
    auto addFeatureParam =
        OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal * 64U * 2U, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr.data());
    EXPECT_EQ(std::get<0>(*(addFeatureParam.attributes + 2U)).tokenId, 2U);
    EXPECT_EQ(addFeatureParam.count, nTotal * 64U * 2U);

    // 添加底库
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), hmm::HMM_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal * 64U * 2U);
    uint32_t deleteNum = tokenNum / 2;
    std::vector<uint32_t> deleteTokenIds(deleteNum);
    for (uint32_t i = 0; i < deleteNum; ++i) {
        deleteTokenIds[i] = i;
    }
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal * 64U * 2U);
    EXPECT_EQ(hppTs->DeleteFeatureByToken(deleteTokenIds.size(), deleteTokenIds.data()), VSA_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal * 64ULL);
    std::vector<int64_t> queryLabels(nTotal * 64U);
    for (uint32_t i = 0; i < nTotal * 64U * 2U / tokenNum; ++i) {
        for (uint32_t j = 0; j < deleteNum; ++j) {
            queryLabels[i] = i * tokenNum + j;
        }
    }
    std::vector<int8_t> result(queryLabels.size() * DIMS);
    EXPECT_EQ(hppTs->GetFeatureByLabel(1, queryLabels.data(), result.data()), VSA_ERROR_INVALID_OUTTER_LABEL);
    std::vector<attr::OckTimeSpaceAttrTrait::KeyTypeTuple> attrResult(queryLabels.size());
    EXPECT_EQ(hppTs->GetFeatureAttrByLabel(1, queryLabels.data(), attrResult.data()), VSA_ERROR_INVALID_OUTTER_LABEL);

    // 准备查询数据
    auto queryCondition = OckVsaAnnQueryCondition<int8_t, DIMS, attr::OckTimeSpaceAttrTrait>(batch, queryFeature.data(),
        attrFilter.data(), shareAttrFilter, topK, nullptr, extraMaskLenEachQuery, extraMaskIsAtDevice,
        enableTimeFilter);
    auto outResult = OckVsaAnnQueryResult<int8_t, attr::OckTimeSpaceAttrTrait>(batch, topK, outLabels.data(),
        outDistances.data(), validNums.data());

    // 查询
    EXPECT_EQ(hppTs->Search(queryCondition, outResult), hmm::HMM_SUCCESS);

    addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal * 64U, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), queryLabels.data(),
        customAttr.data());
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), VSA_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal * 64ULL * 2ULL);
    EXPECT_EQ(hppTs->GetFeatureByLabel(1, queryLabels.data(), result.data()), VSA_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureAttrByLabel(1, queryLabels.data(), attrResult.data()), VSA_SUCCESS);
    EXPECT_EQ(hppTs->Search(queryCondition, outResult), hmm::HMM_SUCCESS);
    DeleteAllBaseByToken(hppTs, 2048ULL, 0);
}

// 测试删除间隔的一半token后新增查询的场景
TEST_F(TestOckVsaHppIndexDelete, AccIndex_HPPTS_DeleteFeatureByToken_Delete_Half_Interval)
{
    auto hppTs = GetHppTsInstance();
    GeneratePerfFeature(nTotal * 64U * 2U); // 生成底库数据
    GenerateCustomAttr(nTotal * 64U * 2U);
    GenerateQueryData(batch); // 生成查询数据
    PrepareOutData(batch, topK);

    // 准备待添加的底库数据
    auto addFeatureParam =
        OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal * 64U * 2U, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr.data());
    EXPECT_EQ(std::get<0>(*(addFeatureParam.attributes + 2U)).tokenId, 2U);
    EXPECT_EQ(addFeatureParam.count, nTotal * 64U * 2U);

    // 添加底库
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), hmm::HMM_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal * 64U * 2U);
    uint32_t deleteNum = tokenNum / 2;
    std::vector<uint32_t> deleteTokenIds(deleteNum);
    for (uint32_t i = 0; i < deleteNum; ++i) {
        deleteTokenIds[i] = i * 2U;
    }
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal * 64U * 2U);
    EXPECT_EQ(hppTs->DeleteFeatureByToken(deleteTokenIds.size(), deleteTokenIds.data()), VSA_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal * 64ULL);
    std::vector<int64_t> queryLabels(nTotal * 64U);
    for (uint32_t i = 0; i < queryLabels.size(); ++i) {
        queryLabels[i] = i * 2U;
    }
    std::vector<int8_t> result(queryLabels.size() * DIMS);
    EXPECT_EQ(hppTs->GetFeatureByLabel(1, queryLabels.data(), result.data()), VSA_ERROR_INVALID_OUTTER_LABEL);
    std::vector<attr::OckTimeSpaceAttrTrait::KeyTypeTuple> attrResult(queryLabels.size());
    EXPECT_EQ(hppTs->GetFeatureAttrByLabel(1, queryLabels.data(), attrResult.data()), VSA_ERROR_INVALID_OUTTER_LABEL);

    // 准备查询数据
    auto queryCondition = OckVsaAnnQueryCondition<int8_t, DIMS, attr::OckTimeSpaceAttrTrait>(batch, queryFeature.data(),
        attrFilter.data(), shareAttrFilter, topK, nullptr, extraMaskLenEachQuery, extraMaskIsAtDevice,
        enableTimeFilter);
    auto outResult = OckVsaAnnQueryResult<int8_t, attr::OckTimeSpaceAttrTrait>(batch, topK, outLabels.data(),
        outDistances.data(), validNums.data());

    // 查询
    EXPECT_EQ(hppTs->Search(queryCondition, outResult), hmm::HMM_SUCCESS);
    for (uint i = 0; i < batch * topK; ++i) {
        EXPECT_NE(outResult.labels[i] % 2U, 0);
    }

    addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal * 64U, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), queryLabels.data(),
        customAttr.data());
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), VSA_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal * 64ULL * 2ULL);
    EXPECT_EQ(hppTs->GetFeatureByLabel(1, queryLabels.data(), result.data()), VSA_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureAttrByLabel(1, queryLabels.data(), attrResult.data()), VSA_SUCCESS);
    EXPECT_EQ(hppTs->Search(queryCondition, outResult), hmm::HMM_SUCCESS);
    DeleteAllBaseByToken(hppTs, 2048ULL, 0);
}

// 测试软删除host侧的一整个group后，触发硬删除的场景
TEST_F(TestOckVsaHppIndexDelete, AccIndex_HPPTS_CLEAR_INVALID_GROUP)
{
    auto hppTs = GetHppTsInstance();
    GeneratePerfFeature(nTotal * 64U * 3U); // 生成底库数据
    GenerateCustomAttr(nTotal * 64U * 3U);
    GenerateQueryData(batch); // 生成查询数据
    PrepareOutData(batch, topK);

    // 准备待添加的底库数据
    auto addFeatureParam =
        OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal * 64U * 3U, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr.data());
    // 添加底库
    EXPECT_EQ(hppTs->GetFeatureNum(), 0);
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), VSA_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal * 64U * 3U);
    // 删除第一个group, 循环删除，每次100w
    std::vector<int64_t> deleteLabels(nTotal * 64U);
    for (uint32_t i = 0; i < deleteLabels.size(); ++i) {
        deleteLabels[i] = i;
    }
    uint32_t deletedNum = 0;
    uint32_t leftNumInDeleteLabels = deleteLabels.size();
    while (leftNumInDeleteLabels > 0) {
        uint32_t removeSize = leftNumInDeleteLabels < 1000000U ? leftNumInDeleteLabels : 1000000;
        EXPECT_EQ(hppTs->DeleteFeatureByLabel(removeSize, deleteLabels.data() + deletedNum), VSA_SUCCESS);
        deletedNum += removeSize;
        leftNumInDeleteLabels -= removeSize;
    }
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal * 64U * 2U);
    std::vector<int8_t> result(deleteLabels.size() * DIMS);
    EXPECT_EQ(hppTs->GetFeatureByLabel(1, deleteLabels.data(), result.data()), VSA_ERROR_INVALID_OUTTER_LABEL);
    std::vector<attr::OckTimeSpaceAttrTrait::KeyTypeTuple> attrResult(deleteLabels.size());
    EXPECT_EQ(hppTs->GetFeatureAttrByLabel(1, deleteLabels.data(), attrResult.data()), VSA_ERROR_INVALID_OUTTER_LABEL);

    // 准备查询数据
    auto queryCondition = OckVsaAnnQueryCondition<int8_t, DIMS, attr::OckTimeSpaceAttrTrait>(batch, queryFeature.data(),
        attrFilter.data(), shareAttrFilter, topK, nullptr, extraMaskLenEachQuery, extraMaskIsAtDevice,
        enableTimeFilter);
    auto outResult = OckVsaAnnQueryResult<int8_t, attr::OckTimeSpaceAttrTrait>(batch, topK, outLabels.data(),
        outDistances.data(), validNums.data());
    // 查询
    EXPECT_EQ(hppTs->Search(queryCondition, outResult), hmm::HMM_SUCCESS);
    for (uint i = 0; i < batch * topK; ++i) {
        EXPECT_GE(outResult.labels[i], 16777216ULL);
    }
    DeleteAllBaseByToken(hppTs, 2048ULL, 0);
}

// 测试一直添加直到超出范围后返回错误码的场景
TEST_F(TestOckVsaHppIndexDelete, AccIndex_HPPTS_ADD_EXCEED)
{
    auto hppTs = GetHppTsInstance();
    GeneratePerfFeature(nTotal * 64U * 3U); // 生成底库数据
    GenerateCustomAttr(nTotal * 64U * 3U);
    GenerateQueryData(batch); // 生成查询数据
    PrepareOutData(batch, topK);

    // 准备待添加的底库数据
    auto addFeatureParam =
        OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal * 64U * 3U, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr.data());
    // 添加底库
    EXPECT_EQ(hppTs->GetFeatureNum(), 0);
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), VSA_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal * 64U * 3U);

    // 继续添加底库
    std::vector<int64_t> newLabels(nTotal * 64ULL);
    std::iota(newLabels.begin(), newLabels.end(), nTotal * 64ULL * 3ULL);
    auto addFeatureParam2 = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal * 64U, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), newLabels.data(),
        customAttr.data());
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam2), VSA_ERROR_EXCEED_HPP_INDEX_MAX_FEATURE_NUMBER);
    DeleteAllBaseByToken(hppTs, 2048ULL, 0);
}

// 测试添加超出maxGroupCount后，利用预留空间存储不触发硬删除的场景
TEST_F(TestOckVsaHppIndexDelete, AccIndex_HPPTS_Out_Of_Add)
{
    auto hppTs = GetHppTsInstance();
    GeneratePerfFeature(nTotal * 64U * 3U); // 生成底库数据
    GenerateCustomAttr(nTotal * 64U * 3U);
    GenerateQueryData(batch); // 生成查询数据
    PrepareOutData(batch, topK);

    // 准备待添加的底库数据
    auto addFeatureParam =
        OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal * 64U * 3U, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr.data());
    // 添加底库
    EXPECT_EQ(hppTs->GetFeatureNum(), 0);
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), VSA_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal * 64U * 3U);

    // 第一个group删除超过一半的数据, 循环删除，每次100w
    std::vector<int64_t> deleteLabels(nTotal * 64U - 10U);
    for (uint32_t i = 0; i < deleteLabels.size(); ++i) {
        deleteLabels[i] = i;
    }
    uint32_t deletedNum = 0;
    uint32_t leftNumInDeleteLabels = deleteLabels.size();
    while (leftNumInDeleteLabels > 0) {
        uint32_t removeSize = leftNumInDeleteLabels < 1000000U ? leftNumInDeleteLabels : 1000000;
        EXPECT_EQ(hppTs->DeleteFeatureByLabel(removeSize, deleteLabels.data() + deletedNum), VSA_SUCCESS);
        deletedNum += removeSize;
        leftNumInDeleteLabels -= removeSize;
    }
    // 第二个group删除超过一半的数据, 循环删除，每次100w
    std::vector<int64_t> deleteLabels2(nTotal * 64U - 10U);
    for (uint32_t i = 0; i < deleteLabels2.size(); ++i) {
        deleteLabels2[i] = nTotal * 64U + i;
    }
    uint32_t deletedNum2 = 0;
    uint32_t leftNumInDeleteLabels2 = deleteLabels2.size();
    while (leftNumInDeleteLabels2 > 0) {
        uint32_t removeSize = leftNumInDeleteLabels2 < 1000000U ? leftNumInDeleteLabels2 : 1000000;
        EXPECT_EQ(hppTs->DeleteFeatureByLabel(removeSize, deleteLabels2.data() + deletedNum2), VSA_SUCCESS);
        deletedNum2 += removeSize;
        leftNumInDeleteLabels2 -= removeSize;
    }

    // 底库数据已满且host侧底库的数据空洞累计超过一个group后，继续添加数据
    std::vector<int64_t> newLabels(nTotal * 64ULL);
    std::iota(newLabels.begin(), newLabels.end(), nTotal * 64ULL * 3ULL);
    auto addFeatureParam2 = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal * 64U, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), newLabels.data(),
        customAttr.data());
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam2), VSA_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), 33554452ULL);
    // 查询结果是否符合预期
    std::vector<int8_t> result(deleteLabels.size() * DIMS);
    EXPECT_EQ(hppTs->GetFeatureByLabel(1, deleteLabels.data(), result.data()), VSA_ERROR_INVALID_OUTTER_LABEL);
    std::vector<attr::OckTimeSpaceAttrTrait::KeyTypeTuple> attrResult(deleteLabels.size());
    EXPECT_EQ(hppTs->GetFeatureAttrByLabel(1, deleteLabels.data(), attrResult.data()), VSA_ERROR_INVALID_OUTTER_LABEL);
    EXPECT_EQ(hppTs->GetFeatureAttrByLabel(1, deleteLabels2.data(), attrResult.data()), VSA_ERROR_INVALID_OUTTER_LABEL);
    DeleteAllBaseByToken(hppTs, 2048ULL, 0);
}
// 测试添加超出maxGroupCount后，预留空间已占满，触发host侧硬删除的场景
TEST_F(TestOckVsaHppIndexDelete, AccIndex_HPPTS_Del_Invalid_Data_Add_To_Device)
{
    auto hppTs = GetHppTsInstance();
    GeneratePerfFeature(nTotal * 64U * 3U); // 生成底库数据
    GenerateCustomAttr(nTotal * 64U * 3U);
    GenerateQueryData(batch); // 生成查询数据
    PrepareOutData(batch, topK);

    // 准备待添加的底库数据
    auto addFeatureParam =
        OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal * 64U * 3U, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr.data());
    // 添加底库
    EXPECT_EQ(hppTs->GetFeatureNum(), 0);
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), VSA_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal * 64U * 3U);

    // 前两个group累计删除超过一个group的数据, 循环删除，每次100w
    std::vector<int64_t> deleteLabels(nTotal * 64U - 10U);
    for (uint32_t i = 0; i < deleteLabels.size(); ++i) {
        deleteLabels[i] = i;
    }
    uint32_t deletedNum = 0;
    uint32_t leftNumInDeleteLabels = deleteLabels.size();
    while (leftNumInDeleteLabels > 0) {
        uint32_t removeSize = leftNumInDeleteLabels < 1000000U ? leftNumInDeleteLabels : 1000000;
        EXPECT_EQ(hppTs->DeleteFeatureByLabel(removeSize, deleteLabels.data() + deletedNum), VSA_SUCCESS);
        deletedNum += removeSize;
        leftNumInDeleteLabels -= removeSize;
    }
    std::vector<int64_t> deleteLabels2(1000000U);
    EXPECT_EQ(hppTs->DeleteFeatureByLabel(1000000U, deleteLabels2.data()), VSA_SUCCESS);
    // 查询是否删除成功
    std::vector<int8_t> result1(deleteLabels.size() * DIMS);
    EXPECT_EQ(hppTs->GetFeatureByLabel(1, deleteLabels.data(), result1.data()), VSA_ERROR_INVALID_OUTTER_LABEL);
    std::vector<attr::OckTimeSpaceAttrTrait::KeyTypeTuple> attrResult1(deleteLabels.size());
    EXPECT_EQ(hppTs->GetFeatureAttrByLabel(1, deleteLabels.data(), attrResult1.data()), VSA_ERROR_INVALID_OUTTER_LABEL);

    // 底库数据已满且host侧底库的数据空洞累计超过一个group后，继续添加超出一个group的数据
    std::vector<int64_t> newLabels(nTotal * 64ULL + 1);
    std::iota(newLabels.begin(), newLabels.end(), nTotal * 64ULL * 3ULL);
    auto addFeatureParam2 = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal * 64U, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), newLabels.data(),
        customAttr.data());
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam2), VSA_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), 49331659ULL);
    // 查询结果是否符合预期
    std::vector<int8_t> result(deleteLabels.size() * DIMS);
    EXPECT_EQ(hppTs->GetFeatureByLabel(1, deleteLabels.data(), result.data()), VSA_ERROR_INVALID_OUTTER_LABEL);
    std::vector<attr::OckTimeSpaceAttrTrait::KeyTypeTuple> attrResult(deleteLabels.size());
    EXPECT_EQ(hppTs->GetFeatureAttrByLabel(1, deleteLabels.data(), attrResult.data()), VSA_ERROR_INVALID_OUTTER_LABEL);
    DeleteAllBaseByToken(hppTs, 2048ULL, 0);
}
} // namespace test
} // namespace neighbor
} // namespace vsa
} // namespace ock
