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

#include <iostream>
#include <random>
#include <chrono>
#include <vector>
#include <numeric>
#include "gtest/gtest.h"
#include "acl/acl.h"
#include "ock/log/OckHcpsLogger.h"
#include "ock/hcps/handler/OckHeteroHandler.h"
#include "ock/hcps/handler/OckHandlerHmmHelper.h"
#include "ock/vsa/neighbor/base/OckVsaAnnFactory.h"
#include "ock/vsa/neighbor/base/OckVsaAnnCreateParam.h"
#include "ock/vsa/neighbor/base/OckVsaAnnAddFeatureParam.h"
#include "ock/vsa/attr/OckTimeSpaceAttrTrait.h"
#include "ock/vsa/neighbor/npu/OckVsaAnnNpuIndex.h"
namespace ock {
namespace vsa {
namespace neighbor {
namespace npu {
const uint64_t DIMS = 256ULL;
inline double GetMilliSecs()
{
    struct timeval tv = { 0, 0 };
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}
class TestNpuSingleDeviceSearch : public testing::Test {
public:
    void SetUp(void) override
    {
        BuildDeviceInfo();
        aclrtSetDevice(deviceInfo->deviceId);
    }
    void TearDown(void) override
    {
        aclrtResetDevice(deviceInfo->deviceId);
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
        std::cout << "==========begin to generate base data==========" << std::endl;
        double ts = GetMilliSecs();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
        for (uint64_t i = 0; i < DIMS * count; ++i) {
            features.push_back(static_cast<int8_t>(dis(gen)));
        }
        double te = GetMilliSecs();
        std::cout << "GenerateFeature Used Time=" << static_cast<float>(te - ts) << "ms" << std::endl;
        std::cout << "==========finish to generate base data==========" << std::endl;

        // 生成底库labels
        for (uint64_t i = 0; i < count; ++i) {
            labels.push_back(i);
        }
        // 生成底库时空属性
        for (uint64_t i = 0; i < count; ++i) {
            attrs.push_back((attr::OckTimeSpaceAttr(int32_t(i % 4U), uint32_t(i % 4U))));
        }
    }

    void GenerateCustomAttr()
    {
        customAttr.resize(nTotal * extKeyAttrsByteSize);
        uint32_t blockSize = extKeyAttrsBlockSize * extKeyAttrsByteSize;
        uint32_t blockNum = utils::SafeDivUp(nTotal, extKeyAttrsBlockSize);
        for (uint32_t i = 0; i < blockNum; ++i) {
            for (uint32_t j = 0; j < blockSize; ++j) {
                customAttr[i * blockSize + j] = j % extKeyAttrsByteSize + i;
            }
        }
    }

    void GenerateQueryData(uint64_t count, int seed = 5678)
    {
        // 生成查询向量
        for (uint32_t i = 0; i < count * DIMS; ++i) {
            queryFeature.push_back(static_cast<int8_t>(i % 8U + 1));
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

        extraMask.resize(count * extraMaskLenEachQuery, 255U);
    }

    void PrepareOutputData()
    {
        outLabels.resize(batch * topK, -1);
        outDistances.resize(batch * topK, -1);
        validNums.resize(batch, 1);
    }

    template <typename T> void WriteHmo(const std::shared_ptr<hmm::OckHmmSubHMObject> &hmo, const std::vector<T> &vec)
    {
        aclrtMemcpy(reinterpret_cast<void *>(hmo->Addr()), hmo->GetByteSize(), vec.data(), vec.size() * sizeof(T),
                    ACL_MEMCPY_HOST_TO_DEVICE);
    }
    template <typename T> void ReadHmo(const std::shared_ptr<hmm::OckHmmSubHMObject> &hmo, std::vector<T> &vec)
    {
        aclrtMemcpy(vec.data(), vec.size() * sizeof(T), reinterpret_cast<void *>(hmo->Addr()), vec.size() * sizeof(T),
                    ACL_MEMCPY_DEVICE_TO_HOST);
    }

    std::shared_ptr<hmm::OckHmmDeviceInfo> deviceInfo;
    uint64_t nTotal{ 262144ULL * 128ULL };
    uint32_t batch{ 256U };
    uint64_t maxFeatureRowCount{ 262144ULL * 64ULL * 6ULL };
    uint32_t tokenNum{ 2500 };
    uint32_t extKeyAttrsByteSize{ 10 };
    uint32_t extKeyAttrsBlockSize{ 262144 };
    attr::OckTimeSpaceAttrTrait dftTrait{ tokenNum };
    std::shared_ptr<OckVsaAnnIndexBase<int8_t, DIMS, 2U, attr::OckTimeSpaceAttrTrait>> indexBase;
    // 底库数据
    std::vector<int8_t> features;
    std::vector<int64_t> labels;
    std::vector<attr::OckTimeSpaceAttr> attrs;
    std::vector<uint8_t> customAttr;
    // 查询数据
    std::vector<int8_t> queryFeature;
    std::shared_ptr<attr::OckTimeSpaceAttrTrait> attrFilter;
    bool shareAttrFilter{ true };
    uint32_t topK{ 128 };
    std::vector<uint8_t> extraMask;
    uint64_t extraMaskLenEachQuery{ nTotal / 8ULL };
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
        deviceInfo->deviceId = 1U;
        CPU_ZERO(&deviceInfo->cpuSet);
        CPU_SET(1U, &deviceInfo->cpuSet);                                                   // 设置1号CPU核
        CPU_SET(2U, &deviceInfo->cpuSet);                                                   // 设置2号CPU核
        deviceInfo->memorySpec.devSpec.maxDataCapacity = 1024ULL * 1024ULL * 1024ULL;       // 1G
        deviceInfo->memorySpec.devSpec.maxSwapCapacity = 3ULL * 64ULL * 1024ULL * 1024ULL;  // 3 * 64M
        deviceInfo->memorySpec.hostSpec.maxDataCapacity = 1024ULL * 1024ULL * 1024ULL;      // 1G
        deviceInfo->memorySpec.hostSpec.maxSwapCapacity = 3ULL * 64ULL * 1024ULL * 1024ULL; // 3 * 64M
        deviceInfo->transferThreadNum = 2ULL;                                               // 2个线程
    }
};

TEST_F(TestNpuSingleDeviceSearch, qps)
{
    // 创建index对象
    BuildIndexBase();

    // 数据准备
    GenerateFeature(nTotal);
    GenerateQueryData(batch);
    PrepareOutputData();

    // 准备待添加的底库数据
    auto addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr.data());
    EXPECT_EQ(std::get<0>(*(addFeatureParam.attributes + 2U)).tokenId, 2U);
    EXPECT_EQ(addFeatureParam.count, nTotal);

    // 添加底库
    EXPECT_EQ(indexBase->AddFeature(addFeatureParam), hmm::HMM_SUCCESS);

    // 准备查询数据
    auto queryCondition =
        OckVsaAnnQueryCondition<int8_t, DIMS, attr::OckTimeSpaceAttrTrait>(batch, queryFeature.data(), attrFilter.get(),
        shareAttrFilter, topK, extraMask.data(), extraMaskLenEachQuery, extraMaskIsAtDevice, enableTimeFilter);
    auto outResult = OckVsaAnnQueryResult<int8_t, attr::OckTimeSpaceAttrTrait>(batch, topK, outLabels.data(),
        outDistances.data(), validNums.data());

    // 预热查询
    std::cout << "==========begin to warm up search==========" << std::endl;
    for (uint32_t i = 0; i < 10U; ++i) {
        double ts = GetMilliSecs();
        auto ret = indexBase->Search(queryCondition, outResult);
        double te = GetMilliSecs();
        EXPECT_EQ(ret, hmm::HMM_SUCCESS);
        std::cout << "warm up search time[" << i << "], qps:" << 1000U * batch / static_cast<float>(te - ts) <<
            std::endl;
    }
    std::cout << "==========end to warm up search==========" << std::endl;

    std::cout << "==========begin to formal search==========" << std::endl;
    std::vector<float> result;
    // 正式搜索
    for (uint32_t i = 0; i < 100U; ++i) {
        double ts = GetMilliSecs();
        auto ret = indexBase->Search(queryCondition, outResult);
        double te = GetMilliSecs();
        EXPECT_EQ(ret, hmm::HMM_SUCCESS);
        float qps = 1000U * batch / static_cast<float>(te - ts);
        result.push_back(qps);
        std::cout << "formal search time[" << i << "], qps:" << qps << std::endl;
    }
    std::cout << "==========end to formal search==========" << std::endl;
    std::cout << "average qps: " << std::accumulate(result.begin(), result.end(), 0.0) / result.size() << std::endl;
}

TEST_F(TestNpuSingleDeviceSearch, add_and_get_custom_attr)
{
    // 创建index对象
    BuildIndexBase();

    // 数据准备
    GenerateFeature(nTotal);
    GenerateCustomAttr();

    // 准备待添加的底库数据
    auto addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(nTotal, features.data(),
        reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(attrs.data()), labels.data(), customAttr.data());
    EXPECT_EQ(std::get<0>(*(addFeatureParam.attributes + 2U)).tokenId, 2U);
    EXPECT_EQ(addFeatureParam.count, nTotal);

    // 添加底库
    EXPECT_EQ(indexBase->AddFeature(addFeatureParam), hmm::HMM_SUCCESS);

    std::vector<uint8_t> attrs(extKeyAttrsBlockSize * extKeyAttrsByteSize);
    auto errorCode = VSA_SUCCESS;
    uintptr_t address = indexBase->GetCustomAttrByBlockId(0U, errorCode);
    aclrtMemcpy(reinterpret_cast<void *>(attrs.data()), attrs.size(), reinterpret_cast<void *>(address), attrs.size(),
                ACL_MEMCPY_DEVICE_TO_HOST);
    for (uint i = 0; i < extKeyAttrsBlockSize; ++i) {
        if (attrs[i] != 0) {
            std::cout << "wrong result:" << int(attrs[i]) << std::endl;
            break;
        }
    }
}

TEST_F(TestNpuSingleDeviceSearch, get_mask_result)
{
    GenerateQueryData(batch);
    OckVsaErrorCode errorCode = hmm::HMM_SUCCESS;
    auto param = OckVsaAnnCreateParam::Create(deviceInfo->cpuSet, deviceInfo->deviceId, maxFeatureRowCount, tokenNum,
        extKeyAttrsByteSize);
    auto handler = hcps::handler::OckHeteroHandler::CreateSingleDeviceHandler(deviceInfo->deviceId, deviceInfo->cpuSet,
        deviceInfo->memorySpec, errorCode);
    auto npuIndex = OckVsaAnnNpuIndex<int8_t, DIMS, 2U, attr::OckTimeSpaceAttrTrait>(handler, dftTrait, param);
    auto queryCondition = OckVsaAnnQueryCondition<int8_t, DIMS, attr::OckTimeSpaceAttrTrait>(batch,
        queryFeature.data(), attrFilter.get(), shareAttrFilter, topK, extraMask.data(),
        extraMaskLenEachQuery, extraMaskIsAtDevice, enableTimeFilter);
    std::vector<int32_t> times;
    times.resize(nTotal, 0);
    for (uint32_t i = 0; i < times.size(); ++i) {
        times[i] = i % 4U;
    }
    std::vector<int32_t> tokenIdQs;
    tokenIdQs.resize(nTotal, 0);
    std::vector<uint8_t> tokenIdRs;
    tokenIdRs.resize(nTotal * 2U, 64U);
    for (uint32_t i = 0; i < nTotal; ++i) {
        tokenIdRs[OPS_DATA_TYPE_TIMES * i] = 1 << (i % 8U);
    }
    std::deque<OckVsaAnnKeyAttrInfo> attrFeatureGroups;
    for (uint32_t i = 0; i < 2U; ++i) {
        OckVsaAnnKeyAttrInfo attrInfo;
        attrInfo.keyAttrTime =
            hcps::handler::helper::MakeDeviceHmo(*handler, times.size() * sizeof(int32_t), errorCode);
        attrInfo.keyAttrQuotient =
            hcps::handler::helper::MakeDeviceHmo(*handler, tokenIdQs.size() * sizeof(int32_t), errorCode);
        attrInfo.keyAttrRemainder =
            hcps::handler::helper::MakeDeviceHmo(*handler, tokenIdRs.size() * sizeof(uint8_t), errorCode);
        WriteHmo(attrInfo.keyAttrTime, times);
        WriteHmo(attrInfo.keyAttrQuotient, tokenIdQs);
        WriteHmo(attrInfo.keyAttrRemainder, tokenIdRs);
        attrFeatureGroups.push_back(attrInfo);
    }
    auto maskHmo = hcps::handler::helper::MakeDeviceHmo(*handler, extraMaskLenEachQuery * 2U, errorCode);
    npuIndex.GetMaskResult(queryCondition, attrFeatureGroups, nTotal, maskHmo);
    std::vector<uint8_t> maskResult(maskHmo->GetByteSize());
    ReadHmo(maskHmo, maskResult);
    for (uint i = 0; i < maskResult.size(); ++i) {
        EXPECT_EQ(maskResult[i], 7U);
    }
}
}
}
}
}
