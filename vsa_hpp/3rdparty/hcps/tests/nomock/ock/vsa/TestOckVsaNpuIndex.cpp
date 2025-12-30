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
#include <sys/time.h>
#include <sys/stat.h>
#include "acl/acl.h"
#include "ock/log/OckHcpsLogger.h"
#include "ock/vsa/neighbor/base/OckVsaAnnFactory.h"
#include "ock/vsa/neighbor/base/OckVsaAnnCreateParam.h"
#include "ock/vsa/neighbor/base/OckVsaAnnAddFeatureParam.h"
#include "ock/vsa/attr/OckTimeSpaceAttrTrait.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace test {
const int DIMS = 256; // 256: dim
class TestOckVsaNpuIndex : public testing::Test {
public:
    void SetUp(void) override
    {
        aclInit(nullptr);
        aclrtSetDevice(0U);
        BuildDeviceInfo();
    }
    void TearDown(void) override
    {
        aclrtResetDevice(0U);
        aclFinalize();
    }
    void BuildIndexBase()
    {
        OckVsaErrorCode errorCode = hmm::HMM_SUCCESS;
        auto facReg = OckVsaAnnIndexFactoryRegister<int8_t, DIMS, 2U, attr::OckTimeSpaceAttrTrait>::Instance();
        auto fac = facReg.GetFactory("NPUTS");
        auto param = OckVsaAnnCreateParam::Create(deviceInfo->cpuSet, deviceInfo->deviceId, maxFeatureRowCount,
            tokenNum, extKeyAttrsByteSize);
        indexBase = fac->Create(param, dftTrait, errorCode);
    }

    void GenerateFeature(uint64_t count)
    {
        double ts = GetMillisecs();
        for (uint64_t i = 0; i < DIMS * count; ++i) {
            features.push_back(i%8UL+1);
        }
        double te = GetMillisecs();

        // 生成底库labels
        for (uint64_t i = 0; i < count; ++i) {
            labels.push_back(i);
        }
        // 生成底库时空属性
        for (uint64_t i = 0; i < count; ++i) {
            attrs.push_back((attr::OckTimeSpaceAttr(int32_t(i % 4U), uint32_t(i % 4U))));
        }
        OCK_HCPS_LOG_ERROR("GenerateFeature Used Time=" << (te - ts) << "ms");
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
            queryFeature.push_back(i%8UL +1);
        }
        // 生成时空过滤条件
        int32_t minTime = 0;
        int32_t maxTime = 100;
        attrFilter = std::make_shared<attr::OckTimeSpaceAttrTrait>(tokenNum);
        attrFilter->minTime = minTime;
        attrFilter->maxTime = maxTime;
        attrFilter->bitSet.Set(0U);
        attrFilter->bitSet.Set(1U);
        attrFilter->bitSet.Set(2U);
    }

    void PrepareOutdata()
    {
        outLabels.resize(batch * topk, -1);
        outDistances.resize(batch * topk, -1);
        validNums.resize(batch, 1);
    }

    std::shared_ptr<hmm::OckHmmDeviceInfo> deviceInfo;
    uint64_t ntotal{ 262144ULL };
    uint32_t batch{ 1U };
    uint64_t maxFeatureRowCount{ 262144ULL * 64ULL * 6ULL };
    uint32_t tokenNum{ 2500 };
    uint32_t extKeyAttrsByteSize{ 20 };
    attr::OckTimeSpaceAttrTrait dftTrait{ tokenNum };
    std::shared_ptr<OckVsaAnnIndexBase<int8_t, DIMS, 2U, attr::OckTimeSpaceAttrTrait>> indexBase;
    // 底库数据
    std::vector<int8_t> features;
    std::vector<int64_t> labels;
    std::vector<attr::OckTimeSpaceAttr> attrs;
    uint8_t *customAttr;
    // 查询数据
    std::vector<int8_t> queryFeature;
    std::shared_ptr<attr::OckTimeSpaceAttrTrait> attrFilter;
    bool shareAttrFilter{ true };
    uint32_t topk{ 16 };
    uint8_t *extraMask = nullptr;
    uint64_t extraMaskLenEachQuery{ 0 };
    bool extraMaskIsAtDevice{ false };
    bool enableTimeFilter{ true };
    // 结果数据
    std::vector<int64_t> outLabels;
    std::vector<float> outDistances;
    std::vector<uint32_t> validNums;

private:
    void BuildDeviceInfo()
    {
        deviceInfo = std::make_shared<hmm::OckHmmDeviceInfo>();
        deviceInfo->deviceId = 0U;
        CPU_SET(1U, &deviceInfo->cpuSet);                                                   // 设置1号CPU核
        CPU_SET(2U, &deviceInfo->cpuSet);                                                   // 设置2号CPU核
        deviceInfo->memorySpec.devSpec.maxDataCapacity = 1024ULL * 1024ULL * 1024ULL;       // 1G
        deviceInfo->memorySpec.devSpec.maxSwapCapacity = 3ULL * 64ULL * 1024ULL * 1024ULL;  // 3 * 64M
        deviceInfo->memorySpec.hostSpec.maxDataCapacity = 1024ULL * 1024ULL * 1024ULL;      // 1G
        deviceInfo->memorySpec.hostSpec.maxSwapCapacity = 3ULL * 64ULL * 1024ULL * 1024ULL; // 3 * 64M
        deviceInfo->transferThreadNum = 2ULL;                                               // 2个线程
    }
    inline double GetMillisecs()
    {
        struct timeval tv = { 0, 0 };
        gettimeofday(&tv, nullptr);
        return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
    }
};

TEST_F(TestOckVsaNpuIndex, add)
{
    BuildIndexBase();
    GenerateFeature(ntotal);     // 生成底库数据
    GenerateQueryData(batch); // 生成查询数据
    PrepareOutdata();
    auto addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(ntotal, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr);
    EXPECT_EQ(std::get<0>(*(addFeatureParam.attributes + 2U)).tokenId, 2U);
    EXPECT_EQ(addFeatureParam.count, ntotal);
    EXPECT_EQ(indexBase->AddFeature(addFeatureParam), hmm::HMM_SUCCESS);
}

TEST_F(TestOckVsaNpuIndex, search)
{
    // 创建index
    BuildIndexBase();
    GenerateFeature(ntotal);     // 生成底库数据
    GenerateQueryData(batch); // 生成查询数据
    PrepareOutdata();

    // 准备待添加的底库数据
    auto addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(ntotal, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr);
    EXPECT_EQ(std::get<0>(*(addFeatureParam.attributes + 2U)).tokenId, 2U);
    EXPECT_EQ(addFeatureParam.count, ntotal);

    // 添加底库
    EXPECT_EQ(indexBase->AddFeature(addFeatureParam), hmm::HMM_SUCCESS);

    // 准备查询数据
    auto queryCondition =
        OckVsaAnnQueryCondition<int8_t, DIMS, attr::OckTimeSpaceAttrTrait>(batch, queryFeature.data(), attrFilter.get(),
        shareAttrFilter, topk, extraMask, extraMaskLenEachQuery, extraMaskIsAtDevice, enableTimeFilter);
    auto outResult = OckVsaAnnQueryResult<int8_t, attr::OckTimeSpaceAttrTrait>(batch, topk, outLabels.data(),
        outDistances.data(), validNums.data());

    // 查询
    EXPECT_EQ(indexBase->Search(queryCondition, outResult), hmm::HMM_SUCCESS);
}
TEST_F(TestOckVsaNpuIndex, get_feature_by_label)
{
    // 创建index
    BuildIndexBase();
    GenerateFeature(ntotal);     // 生成底库数据
    GenerateQueryData(batch); // 生成查询数据
    PrepareOutdata();
    // 准备待添加的底库数据
    auto addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(ntotal, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr);
    EXPECT_EQ(std::get<0>(*(addFeatureParam.attributes + 2U)).tokenId, 2U);
    EXPECT_EQ(addFeatureParam.count, ntotal);

    // 添加底库
    EXPECT_EQ(indexBase->AddFeature(addFeatureParam), hmm::HMM_SUCCESS);
    EXPECT_EQ(indexBase->GetFeatureNum(), ntotal);
    // 准备查询数据
    uint32_t queryNum = 3;
    std::vector<int64_t> labels = { 1, 5, 14 };
    std::vector<int8_t> outFeatures(queryNum * DIMS);
    // 查询
    auto ret = indexBase->GetFeatureByLabel(queryNum, labels.data(), outFeatures.data());
    EXPECT_EQ(ret, hmm::HMM_SUCCESS);
}
TEST_F(TestOckVsaNpuIndex, get_feature_attr_by_label)
{
    // 创建index
    BuildIndexBase();
    GenerateFeature(ntotal);     // 生成底库数据
    GenerateQueryData(batch); // 生成查询数据
    PrepareOutdata();
    // 准备待添加的底库数据
    auto addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(ntotal, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr);
    EXPECT_EQ(std::get<0>(*(addFeatureParam.attributes + 2U)).tokenId, 2U);
    EXPECT_EQ(addFeatureParam.count, ntotal);

    // 添加底库
    EXPECT_EQ(indexBase->AddFeature(addFeatureParam), hmm::HMM_SUCCESS);
    EXPECT_EQ(indexBase->GetFeatureNum(), ntotal);
    // 准备查询数据
    uint32_t queryNum = 3;
    std::vector<int64_t> labels = { 1, 5, 14 };
    std::vector<attr::OckTimeSpaceAttr> outAttrs(queryNum);
    // 查询
    auto ret = indexBase->GetFeatureAttrByLabel(queryNum, labels.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(outAttrs.data()));
    EXPECT_EQ(ret, hmm::HMM_SUCCESS);
}
} // namespace test
} // namespace neighbor
} // namespace vsa
} // namespace ock
