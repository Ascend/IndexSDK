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
namespace test {
const int DIMS = 256; // 256: dim
const uint64_t MAX_FEATURE_ROW_COUNT = 262144ULL * 64ULL * 24ULL;
uint32_t g_extKeyAttrByteSize = 22U;
uint32_t g_extKeyAttrBlockSize = 262144U;
uint32_t g_deviceId = 4U;

std::shared_ptr<OckVsaAnnIndexBase<int8_t, DIMS, 2U, attr::OckTimeSpaceAttrTrait>> BuildIndexBase(
    std::string factoryName)
{
    OckVsaErrorCode errorCode = VSA_SUCCESS;
    auto facReg = OckVsaAnnIndexFactoryRegister<int8_t, DIMS, 2U, attr::OckTimeSpaceAttrTrait>::Instance();
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
    auto param = OckVsaAnnCreateParam::Create(deviceInfo->cpuSet, deviceInfo->deviceId, MAX_FEATURE_ROW_COUNT, 2500U,
        g_extKeyAttrByteSize, g_extKeyAttrBlockSize);
    attr::OckTimeSpaceAttrTrait dftTrait{ 2500U };
    return fac->Create(param, dftTrait, errorCode);
}

std::shared_ptr<OckVsaAnnIndexBase<int8_t, DIMS, 2U, attr::OckTimeSpaceAttrTrait>> GetNpuTsInstance()
{
    static auto npuTs = BuildIndexBase("NPUTS");
    return npuTs;
}

std::shared_ptr<OckVsaAnnIndexBase<int8_t, DIMS, 2U, attr::OckTimeSpaceAttrTrait>> GetHppTsInstance()
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

class TestOckVsaHppIndex : public testing::Test {
public:
    void SetUp(void) override
    {
        aclrtSetDevice(g_deviceId);
    }
    void TearDown(void) override
    {
        aclrtResetDevice(g_deviceId);
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
            attrs[i] = attr::OckTimeSpaceAttr(int32_t(i % 4U), uint32_t(i % 2500U));
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
            attrs[i] = attr::OckTimeSpaceAttr(int32_t(i % 4U), uint32_t(i % 2500U));
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
            attrs[i] = attr::OckTimeSpaceAttr(int32_t(i % 4U), uint32_t(i % 2500U));
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
            attrs[i] = attr::OckTimeSpaceAttr(int32_t(i % 4U), uint32_t(i % 2500U));
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
            attrFilter.push_back(attr::OckTimeSpaceAttrTrait(tokenNum));
            attrFilter[i].minTime = minTime;
            attrFilter[i].maxTime = maxTime;
            attrFilter[i].bitSet.Set(0U);
            attrFilter[i].bitSet.Set(1U);
            attrFilter[i].bitSet.Set(2U);
        }
    }

    void GenerateExtraMask(uint64_t count, uint32_t queryNum)
    {
        extraMaskLenEachQuery = (count + 7ULL) / 8ULL;
        extraMask.resize(extraMaskLenEachQuery * queryNum, 255U);
    }

    void GenerateCustomAttr(uint64_t count)
    {
        uint32_t customAttrBlockNum = (count + g_extKeyAttrBlockSize - 1) / g_extKeyAttrBlockSize;
        customAttr.resize(customAttrBlockNum * g_extKeyAttrBlockSize * g_extKeyAttrByteSize);
        for (uint32_t i = 0; i < customAttrBlockNum; ++i) {
            for (uint32_t j = 0; j < g_extKeyAttrBlockSize * g_extKeyAttrByteSize; ++j) {
                customAttr[i * g_extKeyAttrBlockSize * g_extKeyAttrByteSize + j] = i + j % g_extKeyAttrByteSize;
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
    uint32_t tokenNum{ 2500 };    // 时空属性数量
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

TEST_F(TestOckVsaHppIndex, AccIndex_HPPTS_GetFactory_Failed)
{
    auto facReg = OckVsaAnnIndexFactoryRegister<int8_t, DIMS, 2U, attr::OckTimeSpaceAttrTrait>::Instance();
    auto factory = facReg.GetFactory("hpp");
    EXPECT_EQ(factory, nullptr);
}

TEST_F(TestOckVsaHppIndex, AccIndex_HPPTS_GetNPUTS_Failed)
{
    auto facReg = OckVsaAnnIndexFactoryRegister<int8_t, DIMS, 2U, attr::OckTimeSpaceAttrTrait>::Instance();
    auto factory = facReg.GetFactory("NPUTS");
    EXPECT_EQ(factory, nullptr);
}

TEST_F(TestOckVsaHppIndex, AccIndex_HPPTS_CreateIndex_Failed_While_DeviceId_Invalid)
{
    OckVsaErrorCode errorCode = VSA_SUCCESS;
    auto facReg = OckVsaAnnIndexFactoryRegister<int8_t, DIMS, 2U, attr::OckTimeSpaceAttrTrait>::Instance();
    auto factory = facReg.GetFactory("HPPTS");
    EXPECT_NE(factory, nullptr);
    auto deviceInfo = std::make_shared<hmm::OckHmmDeviceInfo>();
    deviceInfo->deviceId = 100U;
    CPU_SET(1U, &deviceInfo->cpuSet); // 设置1号CPU核
    CPU_SET(2U, &deviceInfo->cpuSet);
    CPU_SET(3U, &deviceInfo->cpuSet);
    CPU_SET(4U, &deviceInfo->cpuSet);
    auto param = OckVsaAnnCreateParam::Create(deviceInfo->cpuSet, deviceInfo->deviceId, MAX_FEATURE_ROW_COUNT, 2500U,
        g_extKeyAttrByteSize, g_extKeyAttrBlockSize);
    attr::OckTimeSpaceAttrTrait dftTrait{ 2500U };
    auto index = factory->Create(param, dftTrait, errorCode);
    EXPECT_EQ(index.get(), nullptr);
}

TEST_F(TestOckVsaHppIndex, AccIndex_HPPTS_CreateIndex_Failed_With_ExtKeyAttrByteSize_Zero)
{
    OckVsaErrorCode errorCode = VSA_SUCCESS;
    auto facReg = OckVsaAnnIndexFactoryRegister<int8_t, DIMS, 2U, attr::OckTimeSpaceAttrTrait>::Instance();
    auto factory = facReg.GetFactory("HPPTS");
    EXPECT_NE(factory, nullptr);
    auto deviceInfo = std::make_shared<hmm::OckHmmDeviceInfo>();
    deviceInfo->deviceId = 1U;
    CPU_SET(1U, &deviceInfo->cpuSet); // 设置1号CPU核
    CPU_SET(2U, &deviceInfo->cpuSet);
    CPU_SET(3U, &deviceInfo->cpuSet);
    CPU_SET(4U, &deviceInfo->cpuSet);
    auto param = OckVsaAnnCreateParam::Create(deviceInfo->cpuSet, deviceInfo->deviceId, MAX_FEATURE_ROW_COUNT, 2500U,
                                              0U, g_extKeyAttrBlockSize);
    attr::OckTimeSpaceAttrTrait dftTrait{ 2500U };
    auto index = factory->Create(param, dftTrait, errorCode);
    EXPECT_EQ(index.get(), nullptr);
}

TEST_F(TestOckVsaHppIndex, AccIndex_HPPTS_CreateIndex_Failed_With_ExtKeyAttrBlockSize_Zero)
{
    OckVsaErrorCode errorCode = VSA_SUCCESS;
    auto facReg = OckVsaAnnIndexFactoryRegister<int8_t, DIMS, 2U, attr::OckTimeSpaceAttrTrait>::Instance();
    auto factory = facReg.GetFactory("HPPTS");
    EXPECT_NE(factory, nullptr);
    auto deviceInfo = std::make_shared<hmm::OckHmmDeviceInfo>();
    deviceInfo->deviceId = 1U;
    CPU_SET(1U, &deviceInfo->cpuSet); // 设置1号CPU核
    CPU_SET(2U, &deviceInfo->cpuSet);
    CPU_SET(3U, &deviceInfo->cpuSet);
    CPU_SET(4U, &deviceInfo->cpuSet);
    auto param = OckVsaAnnCreateParam::Create(deviceInfo->cpuSet, deviceInfo->deviceId, MAX_FEATURE_ROW_COUNT, 2500U,
                                              g_extKeyAttrByteSize, 0U);
    attr::OckTimeSpaceAttrTrait dftTrait{ 2500U };
    auto index = factory->Create(param, dftTrait, errorCode);
    EXPECT_EQ(index.get(), nullptr);
}

TEST_F(TestOckVsaHppIndex, AccIndex_HPPTS_CreateIndex_Succeed_With_ExtKeyAttrBlockSize_And_ExtKeyAttrByteSize_Zero)
{
    OckVsaErrorCode errorCode = VSA_SUCCESS;
    auto facReg = OckVsaAnnIndexFactoryRegister<int8_t, DIMS, 2U, attr::OckTimeSpaceAttrTrait>::Instance();
    auto factory = facReg.GetFactory("HPPTS");
    EXPECT_NE(factory, nullptr);
    auto deviceInfo = std::make_shared<hmm::OckHmmDeviceInfo>();
    deviceInfo->deviceId = 1U;
    CPU_SET(1U, &deviceInfo->cpuSet); // 设置1号CPU核
    CPU_SET(2U, &deviceInfo->cpuSet);
    CPU_SET(3U, &deviceInfo->cpuSet);
    CPU_SET(4U, &deviceInfo->cpuSet);
    auto param = OckVsaAnnCreateParam::Create(deviceInfo->cpuSet, deviceInfo->deviceId, MAX_FEATURE_ROW_COUNT, 2500U,
                                              0U, 0U);
    attr::OckTimeSpaceAttrTrait dftTrait{ 2500U };
    auto index = factory->Create(param, dftTrait, errorCode);
    EXPECT_EQ(errorCode, VSA_SUCCESS);
    EXPECT_NE(index.get(), nullptr);
}

TEST_F(TestOckVsaHppIndex, AccIndex_HPPTS_CreateIndex_Failed_While_CpuSet_Invalid)
{
    OckVsaErrorCode errorCode = VSA_SUCCESS;
    auto facReg = OckVsaAnnIndexFactoryRegister<int8_t, DIMS, 2U, attr::OckTimeSpaceAttrTrait>::Instance();
    auto factory = facReg.GetFactory("HPPTS");
    EXPECT_NE(factory, nullptr);
    auto deviceInfo = std::make_shared<hmm::OckHmmDeviceInfo>();
    deviceInfo->deviceId = g_deviceId;
    CPU_ZERO(&deviceInfo->cpuSet);
    CPU_SET(1U, &deviceInfo->cpuSet); // 设置1号CPU核
    CPU_SET(2U, &deviceInfo->cpuSet);
    CPU_SET(3U, &deviceInfo->cpuSet);
    auto param = OckVsaAnnCreateParam::Create(deviceInfo->cpuSet, deviceInfo->deviceId, MAX_FEATURE_ROW_COUNT, 2500U,
        g_extKeyAttrByteSize, g_extKeyAttrBlockSize);
    attr::OckTimeSpaceAttrTrait dftTrait{ 2500U };
    auto index = factory->Create(param, dftTrait, errorCode);
    EXPECT_EQ(index.get(), nullptr);

    CPU_SET(10000U, &deviceInfo->cpuSet);
    param = OckVsaAnnCreateParam::Create(deviceInfo->cpuSet, deviceInfo->deviceId, MAX_FEATURE_ROW_COUNT, 2500U,
        g_extKeyAttrByteSize, g_extKeyAttrBlockSize);
    index = factory->Create(param, dftTrait, errorCode);
    EXPECT_EQ(index.get(), nullptr);
}

TEST_F(TestOckVsaHppIndex, AccIndex_HPPTS_CreateIndex_Failed_While_ParamRange_Invalid)
{
    OckVsaErrorCode errorCode = VSA_SUCCESS;
    auto facReg = OckVsaAnnIndexFactoryRegister<int8_t, DIMS, 2U, attr::OckTimeSpaceAttrTrait>::Instance();
    auto factory = facReg.GetFactory("HPPTS");
    EXPECT_NE(factory, nullptr);
    auto deviceInfo = std::make_shared<hmm::OckHmmDeviceInfo>();
    deviceInfo->deviceId = g_deviceId;
    CPU_ZERO(&deviceInfo->cpuSet);
    CPU_SET(1U, &deviceInfo->cpuSet); // 设置1号CPU核
    CPU_SET(2U, &deviceInfo->cpuSet);
    CPU_SET(3U, &deviceInfo->cpuSet);
    CPU_SET(4U, &deviceInfo->cpuSet);
    auto param = OckVsaAnnCreateParam::Create(deviceInfo->cpuSet, deviceInfo->deviceId, 16777215U, 2500U,
        g_extKeyAttrByteSize, g_extKeyAttrBlockSize);
    attr::OckTimeSpaceAttrTrait dftTrait{ 2500U };
    auto index = factory->Create(param, dftTrait, errorCode);
    EXPECT_EQ(index.get(), nullptr);

    param = OckVsaAnnCreateParam::Create(deviceInfo->cpuSet, deviceInfo->deviceId, 1000000001U, 2500U,
        g_extKeyAttrByteSize, g_extKeyAttrBlockSize);
    index = factory->Create(param, dftTrait, errorCode);
    EXPECT_EQ(index.get(), nullptr);

    param = OckVsaAnnCreateParam::Create(deviceInfo->cpuSet, deviceInfo->deviceId, MAX_FEATURE_ROW_COUNT, 0,
        g_extKeyAttrByteSize, g_extKeyAttrBlockSize);
    index = factory->Create(param, dftTrait, errorCode);
    EXPECT_EQ(index.get(), nullptr);

    param = OckVsaAnnCreateParam::Create(deviceInfo->cpuSet, deviceInfo->deviceId, MAX_FEATURE_ROW_COUNT, 300001U,
        g_extKeyAttrByteSize, g_extKeyAttrBlockSize);
    index = factory->Create(param, dftTrait, errorCode);
    EXPECT_EQ(index.get(), nullptr);

    param = OckVsaAnnCreateParam::Create(deviceInfo->cpuSet, deviceInfo->deviceId, MAX_FEATURE_ROW_COUNT, 2500U, 23U,
        g_extKeyAttrBlockSize);
    index = factory->Create(param, dftTrait, errorCode);
    EXPECT_EQ(index.get(), nullptr);

    param = OckVsaAnnCreateParam::Create(deviceInfo->cpuSet, deviceInfo->deviceId, MAX_FEATURE_ROW_COUNT, 2500U,
        g_extKeyAttrByteSize, 262143U);
    index = factory->Create(param, dftTrait, errorCode);
    EXPECT_EQ(index.get(), nullptr);

    param = OckVsaAnnCreateParam::Create(deviceInfo->cpuSet, deviceInfo->deviceId, MAX_FEATURE_ROW_COUNT, 2500U,
        g_extKeyAttrByteSize, 524287U);
    index = factory->Create(param, dftTrait, errorCode);
    EXPECT_EQ(index.get(), nullptr);

    param = OckVsaAnnCreateParam::Create(deviceInfo->cpuSet, deviceInfo->deviceId, MAX_FEATURE_ROW_COUNT, 2500U,
        g_extKeyAttrByteSize, 16777217U);
    index = factory->Create(param, dftTrait, errorCode);
    EXPECT_EQ(index.get(), nullptr);
}

TEST_F(TestOckVsaHppIndex, AccIndex_HPPTS_Add)
{
    auto hppTs = GetHppTsInstance();
    GenerateNormalFeature(nTotal); // 生成底库数据
    GenerateCustomAttr(nTotal);
    auto addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr.data());
    EXPECT_EQ(std::get<0>(*(addFeatureParam.attributes + 2U)).tokenId, 2U);
    EXPECT_EQ(addFeatureParam.count, nTotal);
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), hmm::HMM_SUCCESS);
    DeleteAllBaseByToken(hppTs, 2500ULL, 0);
}

TEST_F(TestOckVsaHppIndex, AccIndex_HPPTS_Add_Negative)
{
    auto hppTs = GetHppTsInstance();
    GenerateNormalFeature(nTotal); // 生成底库数据
    GenerateCustomAttr(nTotal);
    // features传入空指针
    auto paramInvalidFeatures = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal, nullptr,
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr.data());
    EXPECT_EQ(hppTs->AddFeature(paramInvalidFeatures), VSA_ERROR_INVALID_INPUT_PARAM);

    // attributes传入空指针
    auto paramInvalidAttributes = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal, features.data(),
        nullptr, labels.data(), customAttr.data());
    EXPECT_EQ(hppTs->AddFeature(paramInvalidAttributes), VSA_ERROR_INVALID_INPUT_PARAM);

    // labels传入空指针
    auto paramInvalidLabels = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), nullptr, customAttr.data());
    EXPECT_EQ(hppTs->AddFeature(paramInvalidLabels), VSA_ERROR_INVALID_INPUT_PARAM);

    // ExtKeyAttrByteSize不等于0的情况下，customAttr传入空指针
    auto paramInvalidCustomAttr = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), nullptr);
    EXPECT_EQ(hppTs->AddFeature(paramInvalidCustomAttr), hcps::HCPS_ERROR_CUSTOM_ATTR_EMPTY);
}

TEST_F(TestOckVsaHppIndex, AccIndex_HPPTS_Add_Exceed_Scope_In_Npu)
{
    OckVsaErrorCode errorCode = VSA_SUCCESS;
    auto facReg = OckVsaAnnIndexFactoryRegister<int8_t, DIMS, 2U, attr::OckTimeSpaceAttrTrait>::Instance();
    auto factory = facReg.GetFactory("HPPTS");
    EXPECT_NE(factory, nullptr);
    auto deviceInfo = std::make_shared<hmm::OckHmmDeviceInfo>();
    deviceInfo->deviceId = 0U;
    CPU_ZERO(&deviceInfo->cpuSet);
    CPU_SET(1U, &deviceInfo->cpuSet); // 设置1号CPU核
    CPU_SET(2U, &deviceInfo->cpuSet);
    CPU_SET(3U, &deviceInfo->cpuSet);
    CPU_SET(4U, &deviceInfo->cpuSet);
    auto param = OckVsaAnnCreateParam::Create(deviceInfo->cpuSet, deviceInfo->deviceId, 16777216ULL, 2500U,
        g_extKeyAttrByteSize, g_extKeyAttrBlockSize);
    attr::OckTimeSpaceAttrTrait dftTrait{ 2500U };
    auto index = factory->Create(param, dftTrait, errorCode);
    EXPECT_NE(index.get(), nullptr);

    GenerateNormalFeature(nTotal * 64ULL + 1); // 生成底库数据
    GenerateCustomAttr(nTotal * 64ULL + 1);
    auto addParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal * 64ULL + 1, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr.data());
    EXPECT_EQ(index->AddFeature(addParam), VSA_ERROR_EXCEED_HPP_INDEX_MAX_FEATURE_NUMBER);
}

TEST_F(TestOckVsaHppIndex, AccIndex_HPPTS_GetFeatureNum)
{
    // 创建index
    auto hppTs = GetHppTsInstance();
    GenerateNormalFeature(nTotal * 2ULL); // 生成底库数据
    GenerateCustomAttr(nTotal * 2ULL);
    // 准备待添加的底库数据
    auto addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal * 2ULL, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr.data());
    EXPECT_EQ(std::get<0>(*(addFeatureParam.attributes + 2U)).tokenId, 2U);
    EXPECT_EQ(addFeatureParam.count, nTotal * 2ULL);

    // 添加底库
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), hmm::HMM_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal * 2ULL);
    DeleteAllBaseByToken(hppTs, 2500ULL, 0);
}

TEST_F(TestOckVsaHppIndex, AccIndex_HPPTS_Get_Feature_By_Label)
{
    auto hppTs = GetHppTsInstance();
    GenerateGetLabelFeature(nTotal); // 生成底库数据
    GenerateCustomAttr(nTotal);
    // 准备待添加的底库数据
    auto addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr.data());
    EXPECT_EQ(std::get<0>(*(addFeatureParam.attributes + 2U)).tokenId, 2U);
    EXPECT_EQ(addFeatureParam.count, nTotal);

    // 添加底库
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), hmm::HMM_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal);
    // 准备查询数据
    uint32_t queryNum = 3U;
    std::vector<int64_t> queryLabels = { 1, 5, 14 };
    std::vector<int8_t> outFeatures(queryNum * DIMS);
    // 查询
    auto ret = hppTs->GetFeatureByLabel(queryNum, queryLabels.data(), outFeatures.data());
    EXPECT_EQ(ret, hmm::HMM_SUCCESS);
    for (uint32_t i = 0; i < queryNum; ++i) {
        for (uint32_t j = 0; j < DIMS; ++j) {
            EXPECT_EQ(outFeatures[i * DIMS + j], queryLabels[i]);
        }
    }
    DeleteAllBaseByToken(hppTs, 2500ULL, 0);
}

TEST_F(TestOckVsaHppIndex, AccIndex_HPPTS_Get_Feature_By_Label_Negative)
{
    auto hppTs = GetHppTsInstance();
    GenerateGetLabelFeature(nTotal); // 生成底库数据
    GenerateCustomAttr(nTotal);
    // 准备待添加的底库数据
    auto addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr.data());
    EXPECT_EQ(std::get<0>(*(addFeatureParam.attributes + 2U)).tokenId, 2U);
    EXPECT_EQ(addFeatureParam.count, nTotal);

    // 添加底库
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), hmm::HMM_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal);
    // 准备查询数据
    uint32_t queryNum = 3U;
    std::vector<int64_t> queryLabels = { 1, 5, 14 };
    std::vector<int8_t> outFeatures(queryNum * DIMS);
    // count大于底库数量场景
    auto retCount = hppTs->GetFeatureByLabel(1.0E6 + 1ULL, queryLabels.data(), outFeatures.data());
    EXPECT_EQ(retCount, VSA_ERROR_INVALID_INPUT_PARAM);
    // labels传入空指针
    auto retLabels = hppTs->GetFeatureByLabel(queryNum, nullptr, outFeatures.data());
    EXPECT_EQ(retLabels, VSA_ERROR_INVALID_INPUT_PARAM);
    // features传入空指针
    auto retFeatures = hppTs->GetFeatureByLabel(queryNum, queryLabels.data(), nullptr);
    EXPECT_EQ(retFeatures, VSA_ERROR_INVALID_INPUT_PARAM);
    DeleteAllBaseByToken(hppTs, 2500ULL, 0);
}

TEST_F(TestOckVsaHppIndex, AccIndex_HPPTS_Get_Feature_Attr_By_Label)
{
    auto hppTs = GetHppTsInstance();
    GenerateNormalFeature(nTotal); // 生成底库数据
    GenerateCustomAttr(nTotal);
    // 准备待添加的底库数据
    auto addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr.data());
    EXPECT_EQ(std::get<0>(*(addFeatureParam.attributes + 2U)).tokenId, 2U);
    EXPECT_EQ(addFeatureParam.count, nTotal);

    // 添加底库
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), hmm::HMM_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal);
    // 准备查询数据
    uint32_t queryNum = 3U;
    std::vector<int64_t> queryLabels = { 1, 5, 14 };
    std::vector<attr::OckTimeSpaceAttr> outAttrs(queryNum);
    // 查询
    auto ret = hppTs->GetFeatureAttrByLabel(queryNum, queryLabels.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(outAttrs.data()));
    EXPECT_EQ(ret, hmm::HMM_SUCCESS);
    for (uint32_t i = 0; i < queryNum; ++i) {
        EXPECT_EQ(outAttrs[i].time, queryLabels[i] % 4U);
        EXPECT_EQ(outAttrs[i].tokenId, queryLabels[i] % 2500U);
    }
    DeleteAllBaseByToken(hppTs, 2500ULL, 0);
}

TEST_F(TestOckVsaHppIndex, AccIndex_HPPTS_Get_Feature_Attr_By_Label_Negative)
{
    auto hppTs = GetHppTsInstance();
    GenerateNormalFeature(nTotal); // 生成底库数据
    GenerateCustomAttr(nTotal);
    // 准备待添加的底库数据
    auto addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr.data());
    EXPECT_EQ(std::get<0>(*(addFeatureParam.attributes + 2U)).tokenId, 2U);
    EXPECT_EQ(addFeatureParam.count, nTotal);

    // 添加底库
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), hmm::HMM_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal);
    // 准备查询数据
    uint32_t queryNum = 3U;
    std::vector<int64_t> queryLabels = { 1, 5, 14 };
    std::vector<attr::OckTimeSpaceAttr> outAttrs(queryNum);
    // count大于底库数量
    auto retCount = hppTs->GetFeatureAttrByLabel(1.0E6 + 1, queryLabels.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(outAttrs.data()));
    EXPECT_EQ(retCount, VSA_ERROR_INVALID_INPUT_PARAM);
    // labels传入空指针
    auto retLabels = hppTs->GetFeatureAttrByLabel(queryNum, nullptr,
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(outAttrs.data()));
    EXPECT_EQ(retLabels, VSA_ERROR_INVALID_INPUT_PARAM);
    // attributes传入空指针
    auto retAttributes = hppTs->GetFeatureAttrByLabel(queryNum, queryLabels.data(), nullptr);
    EXPECT_EQ(retAttributes, VSA_ERROR_INVALID_INPUT_PARAM);
    DeleteAllBaseByToken(hppTs, 2500ULL, 0);
}

TEST_F(TestOckVsaHppIndex, AccIndex_HPPTS_Search)
{
    auto hppTs = GetHppTsInstance();
    GenerateRandFeature(nTotal); // 生成底库数据
    GenerateCustomAttr(nTotal);
    GenerateQueryData(batch); // 生成查询数据
    PrepareOutData(batch, topK);

    // 准备待添加的底库数据
    auto addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr.data());
    EXPECT_EQ(std::get<0>(*(addFeatureParam.attributes + 2U)).tokenId, 2U);
    EXPECT_EQ(addFeatureParam.count, nTotal);

    // 添加底库
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), hmm::HMM_SUCCESS);
    // 准备查询数据
    auto queryCondition = OckVsaAnnQueryCondition<int8_t, DIMS, attr::OckTimeSpaceAttrTrait>(batch, queryFeature.data(),
        attrFilter.data(), shareAttrFilter, topK, nullptr, extraMaskLenEachQuery, extraMaskIsAtDevice,
        enableTimeFilter);
    auto outResult = OckVsaAnnQueryResult<int8_t, attr::OckTimeSpaceAttrTrait>(batch, topK, outLabels.data(),
        outDistances.data(), validNums.data());

    // 查询
    EXPECT_EQ(hppTs->Search(queryCondition, outResult), hmm::HMM_SUCCESS);
    std::cout << outResult << std::endl;
    DeleteAllBaseByToken(hppTs, 2500ULL, 0);
}

TEST_F(TestOckVsaHppIndex, AccIndex_HPPTS_Search_Negative)
{
    auto hppTs = GetHppTsInstance();
    GenerateRandFeature(nTotal); // 生成底库数据
    GenerateCustomAttr(nTotal);
    GenerateQueryData(batch); // 生成查询数据
    PrepareOutData(batch, topK);

    // 准备待添加的底库数据
    auto addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr.data());
    EXPECT_EQ(std::get<0>(*(addFeatureParam.attributes + 2U)).tokenId, 2U);
    EXPECT_EQ(addFeatureParam.count, nTotal);

    // 添加底库
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), hmm::HMM_SUCCESS);
    // 准备查询数据
    auto outResult = OckVsaAnnQueryResult<int8_t, attr::OckTimeSpaceAttrTrait>(batch, topK, outLabels.data(),
        outDistances.data(), validNums.data());

    // queryBatchCount传入小于等于0的值
    auto invalidQueryBatchCount =
        OckVsaAnnQueryCondition<int8_t, DIMS, attr::OckTimeSpaceAttrTrait>(0, queryFeature.data(), attrFilter.data(),
        shareAttrFilter, topK, nullptr, extraMaskLenEachQuery, extraMaskIsAtDevice, enableTimeFilter);
    EXPECT_EQ(hppTs->Search(invalidQueryBatchCount, outResult), VSA_ERROR_INVALID_INPUT_PARAM);

    // queryFeature传入空指针
    auto invalidQueryFeature =
        OckVsaAnnQueryCondition<int8_t, DIMS, attr::OckTimeSpaceAttrTrait>(batch, nullptr, attrFilter.data(),
        shareAttrFilter, topK, nullptr, extraMaskLenEachQuery, extraMaskIsAtDevice, enableTimeFilter);
    EXPECT_EQ(hppTs->Search(invalidQueryFeature, outResult), VSA_ERROR_INVALID_INPUT_PARAM);

    // attrFilter传入空指针
    auto invalidAttrFilter =
        OckVsaAnnQueryCondition<int8_t, DIMS, attr::OckTimeSpaceAttrTrait>(batch, queryFeature.data(), nullptr,
        shareAttrFilter, topK, nullptr, extraMaskLenEachQuery, extraMaskIsAtDevice, enableTimeFilter);
    EXPECT_EQ(hppTs->Search(invalidAttrFilter, outResult), VSA_ERROR_INVALID_INPUT_PARAM);

    // topk传入0
    auto invalidTopk = OckVsaAnnQueryCondition<int8_t, DIMS, attr::OckTimeSpaceAttrTrait>(batch, queryFeature.data(),
        attrFilter.data(), shareAttrFilter, 0, nullptr, extraMaskLenEachQuery, extraMaskIsAtDevice, enableTimeFilter);
    EXPECT_EQ(hppTs->Search(invalidTopk, outResult), VSA_ERROR_INVALID_INPUT_PARAM);

    // topK大于100000
    auto biggerTopk = OckVsaAnnQueryCondition<int8_t, DIMS, attr::OckTimeSpaceAttrTrait>(batch, queryFeature.data(),
        attrFilter.data(), shareAttrFilter, 100001ULL, nullptr, extraMaskLenEachQuery, extraMaskIsAtDevice,
        enableTimeFilter);
    EXPECT_EQ(hppTs->Search(invalidTopk, outResult), VSA_ERROR_INVALID_INPUT_PARAM);
    DeleteAllBaseByToken(hppTs, 2500ULL, 0);
}

TEST_F(TestOckVsaHppIndex, AccIndex_HPPTS_Search_with_extra_mask)
{
    auto hppTs = GetHppTsInstance();
    GenerateRandFeature(nTotal * 2ULL); // 生成底库数据
    GenerateCustomAttr(nTotal * 2ULL);
    GenerateQueryData(batch); // 生成查询数据
    GenerateExtraMask(nTotal * 2ULL, batch);
    PrepareOutData(batch, topK);

    // 准备待添加的底库数据
    auto addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal * 2ULL, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr.data());
    EXPECT_EQ(std::get<0>(*(addFeatureParam.attributes + 2U)).tokenId, 2U);
    EXPECT_EQ(addFeatureParam.count, nTotal * 2ULL);

    // 添加底库
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), hmm::HMM_SUCCESS);

    // 准备查询数据
    auto queryCondition = OckVsaAnnQueryCondition<int8_t, DIMS, attr::OckTimeSpaceAttrTrait>(batch, queryFeature.data(),
        attrFilter.data(), false, topK, extraMask.data(), extraMaskLenEachQuery, extraMaskIsAtDevice, enableTimeFilter);
    auto outResult = OckVsaAnnQueryResult<int8_t, attr::OckTimeSpaceAttrTrait>(batch, topK, outLabels.data(),
        outDistances.data(), validNums.data());

    // 查询
    EXPECT_EQ(hppTs->Search(queryCondition, outResult), hmm::HMM_SUCCESS);
    std::cout << outResult << std::endl;
    DeleteAllBaseByToken(hppTs, 2500ULL, 0);
}

TEST_F(TestOckVsaHppIndex, AccIndex_HPPTS_Search_With_Extra_Mask_Negative)
{
    auto hppTs = GetHppTsInstance();
    GenerateRandFeature(nTotal * 2ULL); // 生成底库数据
    GenerateCustomAttr(nTotal * 2ULL);
    GenerateQueryData(batch); // 生成查询数据
    GenerateExtraMask(nTotal * 2ULL, batch);
    PrepareOutData(batch, topK);

    // 准备待添加的底库数据
    auto addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal * 2ULL, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr.data());
    EXPECT_EQ(std::get<0>(*(addFeatureParam.attributes + 2U)).tokenId, 2U);
    EXPECT_EQ(addFeatureParam.count, nTotal * 2ULL);

    // 添加底库
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), hmm::HMM_SUCCESS);

    // extraMask不为空时 extraMaskLenEachQuery 值不等于底库数量 / 8的场景
    auto invalidExtraMaskLenEachQuery = OckVsaAnnQueryCondition<int8_t, DIMS, attr::OckTimeSpaceAttrTrait>(batch,
        queryFeature.data(), attrFilter.data(), false, topK, extraMask.data(), extraMaskLenEachQuery + 1,
        extraMaskIsAtDevice, enableTimeFilter);
    auto outResult = OckVsaAnnQueryResult<int8_t, attr::OckTimeSpaceAttrTrait>(batch, 200, outLabels.data(),
        outDistances.data(), validNums.data());

    EXPECT_EQ(hppTs->Search(invalidExtraMaskLenEachQuery, outResult), VSA_ERROR_INVALID_INPUT_PARAM);
    DeleteAllBaseByToken(hppTs, 2500ULL, 0);
}

// 每次测试完功能后，注意将本用例添加的底库全部清空，否则多个用例共有一个Index对象，可能影响其他用例测试的正确性。
TEST_F(TestOckVsaHppIndex, AccIndex_HPPTS_DeleteFeatureByLabel_Delete_All)
{
    auto hppTs = GetHppTsInstance();
    GeneratePerfFeature(nTotal * 3U); // 生成底库数据
    GenerateCustomAttr(nTotal * 3U);
    GenerateQueryData(batch); // 生成查询数据
    PrepareOutData(batch, topK);

    // 准备待添加的底库数据
    auto addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal * 3U, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr.data());
    EXPECT_EQ(std::get<0>(*(addFeatureParam.attributes + 2U)).tokenId, 2U);
    EXPECT_EQ(addFeatureParam.count, nTotal * 3U);

    // 添加底库
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), hmm::HMM_SUCCESS);

    // 准备查询数据
    auto queryCondition = OckVsaAnnQueryCondition<int8_t, DIMS, attr::OckTimeSpaceAttrTrait>(batch, queryFeature.data(),
        attrFilter.data(), shareAttrFilter, topK, nullptr, extraMaskLenEachQuery, extraMaskIsAtDevice,
        enableTimeFilter);
    auto outResult = OckVsaAnnQueryResult<int8_t, attr::OckTimeSpaceAttrTrait>(batch, topK, outLabels.data(),
        outDistances.data(), validNums.data());

    // 查询
    EXPECT_EQ(hppTs->Search(queryCondition, outResult), hmm::HMM_SUCCESS);
    for (uint32_t i = 0; i < batch; ++i) {
        EXPECT_EQ(outResult.validNums[i], topK);
    }
    std::cout << outResult << std::endl;

    auto deleteLabels = std::move(GenerateRangeLabels(0, nTotal * 3ULL));
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal * 3ULL);
    auto errorCode = hppTs->DeleteFeatureByLabel(deleteLabels.size(), deleteLabels.data());
    EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), 0);

    std::cout << "after delete" << std::endl;
    std::vector<int64_t> newOutLabels(batch * topK, -1);
    std::vector<float> newOutDistances(batch * topK, -1);
    std::vector<uint32_t> newValidNums(batch, 0);
    auto newOutResult = OckVsaAnnQueryResult<int8_t, attr::OckTimeSpaceAttrTrait>(batch, topK, newOutLabels.data(),
        newOutDistances.data(), newValidNums.data());
    EXPECT_EQ(hppTs->Search(queryCondition, newOutResult), VSA_ERROR_EMPTY_BASE);
    for (uint32_t i = 0; i < batch; ++i) {
        EXPECT_EQ(newOutResult.validNums[0], 0);
    }
    std::cout << newOutResult << std::endl;
}

TEST_F(TestOckVsaHppIndex, AccIndex_HPPTS_DeleteFeatureByLabel_Delete_Half)
{
    auto hppTs = GetHppTsInstance();
    GeneratePerfFeature(nTotal * 4U); // 生成底库数据
    GenerateCustomAttr(nTotal * 4U);
    GenerateQueryData(batch); // 生成查询数据
    PrepareOutData(batch, topK);

    // 准备待添加的底库数据
    auto addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal * 4U, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr.data());
    EXPECT_EQ(std::get<0>(*(addFeatureParam.attributes + 2U)).tokenId, 2U);
    EXPECT_EQ(addFeatureParam.count, nTotal * 4U);

    // 添加底库
    EXPECT_EQ(hppTs->GetFeatureNum(), 0);
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), VSA_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal * 4ULL);

    // 删除偶数label的底库
    std::vector<int64_t> deleteLabels(hppTs->GetFeatureNum() / 2U);
    for (uint32_t i = 0; i < deleteLabels.size(); ++i) {
        deleteLabels[i] = i * 2U;
    }
    EXPECT_EQ(hppTs->DeleteFeatureByLabel(deleteLabels.size(), deleteLabels.data()), VSA_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal * 2ULL);
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
    addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal * 2U, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), deleteLabels.data(),
        customAttr.data());
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), VSA_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal * 4ULL);
    EXPECT_EQ(hppTs->GetFeatureByLabel(1, deleteLabels.data(), result.data()), VSA_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureAttrByLabel(1, deleteLabels.data(), attrResult.data()), VSA_SUCCESS);
    EXPECT_EQ(hppTs->Search(queryCondition, outResult), hmm::HMM_SUCCESS);
    DeleteAllBaseByToken(hppTs, 2500ULL, 0);
}

TEST_F(TestOckVsaHppIndex, AccIndex_HPPTS_DeleteFeatureByLabel_Negative)
{
    auto hppTs = GetHppTsInstance();
    GeneratePerfFeature(nTotal * 3U); // 生成底库数据
    GenerateCustomAttr(nTotal * 3U);
    auto addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal * 3U, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr.data());
    EXPECT_EQ(std::get<0>(*(addFeatureParam.attributes + 2U)).tokenId, 2U);
    EXPECT_EQ(addFeatureParam.count, nTotal * 3U);

    // 添加底库
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), hmm::HMM_SUCCESS);

    auto deleteLabels = std::move(GenerateRangeLabels(0, nTotal * 3ULL));
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal * 3ULL);

    // 测试count数值超出范围
    auto errorCode = hppTs->DeleteFeatureByLabel(1.0E6 + 1, deleteLabels.data());
    EXPECT_EQ(errorCode, VSA_ERROR_INVALID_INPUT_PARAM);

    // 测试labels传入空指针
    auto errorCode2 = hppTs->DeleteFeatureByLabel(deleteLabels.size(), nullptr);
    EXPECT_EQ(errorCode2, VSA_ERROR_INVALID_INPUT_PARAM);
    DeleteAllBaseByToken(hppTs, 2500ULL, 0);
}

// 每次测试完功能后，注意将本用例添加的底库全部清空，否则多个用例共有一个Index对象，可能影响其他用例测试的正确性。
TEST_F(TestOckVsaHppIndex, AccIndex_HPPTS_DeleteFeatureByToken_Delete_All)
{
    auto hppTs = GetHppTsInstance();
    GeneratePerfFeature(nTotal * 3U); // 生成底库数据
    GenerateCustomAttr(nTotal * 3U);
    GenerateQueryData(batch); // 生成查询数据
    PrepareOutData(batch, topK);

    // 准备待添加的底库数据
    auto addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal * 3U, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr.data());
    EXPECT_EQ(std::get<0>(*(addFeatureParam.attributes + 2U)).tokenId, 2U);
    EXPECT_EQ(addFeatureParam.count, nTotal * 3U);

    // 添加底库
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), hmm::HMM_SUCCESS);

    // 准备查询数据
    auto queryCondition = OckVsaAnnQueryCondition<int8_t, DIMS, attr::OckTimeSpaceAttrTrait>(batch, queryFeature.data(),
        attrFilter.data(), shareAttrFilter, topK, nullptr, extraMaskLenEachQuery, extraMaskIsAtDevice,
        enableTimeFilter);
    auto outResult = OckVsaAnnQueryResult<int8_t, attr::OckTimeSpaceAttrTrait>(batch, topK, outLabels.data(),
        outDistances.data(), validNums.data());

    // 查询
    EXPECT_EQ(hppTs->Search(queryCondition, outResult), hmm::HMM_SUCCESS);
    std::cout << outResult.validNums[0] << std::endl;
    std::cout << outResult << std::endl;

    std::vector<uint32_t> deleteTokenIds(tokenNum);
    std::iota(deleteTokenIds.begin(), deleteTokenIds.end(), 0);
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal * 3ULL);
    auto errorCode = hppTs->DeleteFeatureByToken(deleteTokenIds.size(), deleteTokenIds.data());
    EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), 0);

    std::cout << "after delete" << std::endl;
    std::vector<int64_t> newOutLabels(batch * topK, -1);
    std::vector<float> newOutDistances(batch * topK, -1);
    std::vector<uint32_t> newValidNums(batch, 0);
    auto newOutResult = OckVsaAnnQueryResult<int8_t, attr::OckTimeSpaceAttrTrait>(batch, topK, newOutLabels.data(),
        newOutDistances.data(), newValidNums.data());
    EXPECT_EQ(hppTs->Search(queryCondition, newOutResult), VSA_ERROR_EMPTY_BASE);
    std::cout << newOutResult.validNums[0] << std::endl;
    std::cout << newOutResult << std::endl;
}

TEST_F(TestOckVsaHppIndex, AccIndex_HPPTS_DeleteFeatureByToken_Delete_Half)
{
    auto hppTs = GetHppTsInstance();
    GeneratePerfFeature(2500ULL * 600ULL); // 生成底库数据
    GenerateCustomAttr(2500ULL * 600ULL);
    GenerateQueryData(batch); // 生成查询数据
    PrepareOutData(batch, topK);

    // 准备待添加的底库数据
    auto addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(2500ULL * 600ULL,
        features.data(), reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()),
        labels.data(), customAttr.data());
    EXPECT_EQ(std::get<0>(*(addFeatureParam.attributes + 2U)).tokenId, 2U);
    EXPECT_EQ(addFeatureParam.count, 2500ULL * 600ULL);

    // 添加底库
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), hmm::HMM_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), 2500ULL * 600ULL);

    std::vector<uint32_t> deleteTokenIds(tokenNum / 2U);
    for (uint32_t i = 0; i < deleteTokenIds.size(); ++i) {
        deleteTokenIds[i] = i * 2U;
    }
    EXPECT_EQ(hppTs->GetFeatureNum(), 2500ULL * 600ULL);
    EXPECT_EQ(hppTs->DeleteFeatureByToken(deleteTokenIds.size(), deleteTokenIds.data()), VSA_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), 2500ULL * 300ULL);
    std::vector<int64_t> queryLabels(2500ULL * 300ULL);
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

    addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(2500ULL * 300ULL, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), queryLabels.data(),
        customAttr.data());
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), VSA_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureNum(), 2500ULL * 600ULL);
    EXPECT_EQ(hppTs->GetFeatureByLabel(1, queryLabels.data(), result.data()), VSA_SUCCESS);
    EXPECT_EQ(hppTs->GetFeatureAttrByLabel(1, queryLabels.data(), attrResult.data()), VSA_SUCCESS);
    EXPECT_EQ(hppTs->Search(queryCondition, outResult), hmm::HMM_SUCCESS);
    DeleteAllBaseByToken(hppTs, 2500ULL, 0);
}

TEST_F(TestOckVsaHppIndex, AccIndex_HPPTS_DeleteFeatureByToken_Negative)
{
    auto hppTs = GetHppTsInstance();
    GeneratePerfFeature(nTotal * 3U); // 生成底库数据
    GenerateCustomAttr(nTotal * 3U);
    GenerateQueryData(batch); // 生成查询数据
    PrepareOutData(batch, topK);

    // 准备待添加的底库数据
    auto addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal * 3U, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr.data());
    EXPECT_EQ(std::get<0>(*(addFeatureParam.attributes + 2U)).tokenId, 2U);
    EXPECT_EQ(addFeatureParam.count, nTotal * 3U);

    // 添加底库
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), hmm::HMM_SUCCESS);

    std::vector<uint32_t> deleteTokenIds = { 0, 1, 2, 3 };
    EXPECT_EQ(hppTs->GetFeatureNum(), nTotal * 3ULL);
    // 测试count数值超出范围
    auto errorCode = hppTs->DeleteFeatureByToken(1.0E6 + 1, deleteTokenIds.data());
    EXPECT_EQ(errorCode, VSA_ERROR_INVALID_INPUT_PARAM);

    // 测试tokens传入空指针
    auto errorCode2 = hppTs->DeleteFeatureByToken(deleteTokenIds.size(), nullptr);
    EXPECT_EQ(errorCode2, VSA_ERROR_INVALID_INPUT_PARAM);
    DeleteAllBaseByToken(hppTs, 2500ULL, 0);
}

TEST_F(TestOckVsaHppIndex, AccIndex_HPPTS_Get_Custom_Attr_By_BlockId)
{
    auto hppTs = GetHppTsInstance();
    GeneratePerfFeature(nTotal * 3U); // 生成底库数据
    GenerateCustomAttr(nTotal * 3U);

    // 准备待添加的底库数据
    auto addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal * 3U, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr.data());
    EXPECT_EQ(std::get<0>(*(addFeatureParam.attributes + 2U)).tokenId, 2U);
    EXPECT_EQ(addFeatureParam.count, nTotal * 3U);

    // 添加底库
    EXPECT_EQ(hppTs->AddFeature(addFeatureParam), hmm::HMM_SUCCESS);
    EXPECT_EQ(hppTs->GetCustomAttrBlockCount(), 3U);
    std::vector<uint8_t> customAttrResult(g_extKeyAttrBlockSize * g_extKeyAttrByteSize, 255U);
    auto errorCode = VSA_SUCCESS;
    uintptr_t address = hppTs->GetCustomAttrByBlockId(0, errorCode);
    aclrtMemcpy(reinterpret_cast<void *>(customAttrResult.data()), customAttrResult.size(),
        reinterpret_cast<void *>(address), customAttrResult.size(), ACL_MEMCPY_DEVICE_TO_HOST);
    for (uint i = 0; i < g_extKeyAttrBlockSize; ++i) {
        EXPECT_EQ(customAttrResult[i], 0);
    }
    DeleteAllBaseByToken(hppTs, 2500ULL, 0);
}

// batch = 1
TEST_F(TestOckVsaHppIndex, AccIndex_HPPTS_Search_With_Host_Memory)
{
    auto hppTs = GetHppTsInstance();
    GeneratePerfFeature(nTotal * (64U * 2U)); // 生成底库数据
    GenerateCustomAttr(nTotal * (64U * 2U));
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

    // 准备查询数据
    auto queryCondition = OckVsaAnnQueryCondition<int8_t, DIMS, attr::OckTimeSpaceAttrTrait>(batch, queryFeature.data(),
        attrFilter.data(), shareAttrFilter, topK, nullptr, extraMaskLenEachQuery, extraMaskIsAtDevice,
        enableTimeFilter);
    auto outResult = OckVsaAnnQueryResult<int8_t, attr::OckTimeSpaceAttrTrait>(batch, topK, outLabels.data(),
        outDistances.data(), validNums.data());

    // 查询
    EXPECT_EQ(hppTs->Search(queryCondition, outResult), hmm::HMM_SUCCESS);
    std::cout << outResult.validNums[0] << std::endl;
    std::cout << outResult << std::endl;
    DeleteAllBaseByToken(hppTs, 2500ULL, 0);
}
} // namespace test
} // namespace neighbor
} // namespace vsa
} // namespace ock
