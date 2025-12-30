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


#include <memory>
#include <vector>
#include <unistd.h>
#include "gtest/gtest.h"
#include "acl/acl.h"
#include "mockcpp/mockcpp.hpp"
#include "ock/hmm/OckHmmFactory.h"
#include "ock/acladapter/WithEnvAclMock.h"

namespace ock {
namespace hmm {
namespace test {

class ApiTestParamCheck : public acladapter::WithEnvAclMock<testing::Test> {
public:
    void SetUp(void) override
    {
        acladapter::WithEnvAclMock<testing::Test>::SetUp();
        aclInit(nullptr);
        factory = OckHmmFactory::Create();
    }

    void TearDown(void) override
    {
        factory.reset();
        singleMgr.reset();
        composeMgr.reset();
        aclFinalize();
        acladapter::WithEnvAclMock<testing::Test>::TearDown();
    }

    void BuildSingleDeviceMgr(void)
    {
        auto deviceInfo = std::make_shared<OckHmmDeviceInfo>(BuildDeviceInfo(deviceIdA, {0U, 1U}));
        auto ret = factory->CreateSingleDeviceMemoryMgr(deviceInfo);
        ASSERT_EQ(ret.first, HMM_SUCCESS);
        ASSERT_NE(ret.second.get(), nullptr);
        singleMgr = ret.second;
    }

    void BuildComposeDeviceMgr(void)
    {
        auto deviceInfoVec = std::make_shared<OckHmmDeviceInfoVec>();
        deviceInfoVec->push_back(BuildDeviceInfo(deviceIdA, {2U, 3U}));
        deviceInfoVec->push_back(BuildDeviceInfo(deviceIdB, {4U, 5U}));
        auto ret = factory->CreateComposeMemoryMgr(deviceInfoVec);
        ASSERT_EQ(ret.first, HMM_SUCCESS);
        ASSERT_NE(ret.second.get(), nullptr);
        composeMgr = ret.second;
    }

    OckHmmDeviceInfo BuildDeviceInfo(OckHmmDeviceId deviceId, std::vector<uint32_t> cpuIds)
    {
        OckHmmDeviceInfo devInfo;
        devInfo.deviceId = deviceId;
        CPU_ZERO(&devInfo.cpuSet);
        for (uint32_t i = 0; i < cpuIds.size(); ++i) {
            CPU_SET(cpuIds[i], &devInfo.cpuSet);
        }
        devInfo.transferThreadNum = 2ULL;
        devInfo.memorySpec.devSpec.maxDataCapacity = normalBaseCapacity;
        devInfo.memorySpec.devSpec.maxSwapCapacity = normalSwapCapacity;
        devInfo.memorySpec.hostSpec.maxDataCapacity = normalBaseCapacity;
        devInfo.memorySpec.hostSpec.maxSwapCapacity = normalSwapCapacity;
        return devInfo;
    }

    OckHmmDeviceId deviceIdA = 0U;
    OckHmmDeviceId deviceIdB = 1U;
    OckHmmDeviceId deviceIdC = 2U;
    OckHmmDeviceId deviceIdD = 3U;
    OckHmmDeviceId maxDeviceId = 15U;
    OckHmmDeviceId minDeviceId = 0U;
    uint32_t minTransferThreadNum = 1U;
    uint32_t maxTransferThreadNum = 128U;
    uint64_t minBaseCapacity = 1ULL * 1024ULL * 1024ULL * 1024ULL;
    uint64_t maxBaseCapacity = 512ULL * 1024ULL * 1024ULL * 1024ULL;
    uint64_t normalBaseCapacity = 1ULL * 1024ULL * 1024ULL * 1024ULL;
    uint64_t minSwapCapacity = 64ULL * 1024ULL * 1024ULL;
    uint64_t maxSwapCapacity = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    uint64_t normalSwapCapacity = 2ULL * 64ULL * 1024ULL * 1024ULL;
    uint64_t minHmoSize = 0ULL;
    uint64_t maxHmoSize = 128ULL * 1024ULL * 1024ULL * 1024ULL;
    uint64_t normalHmoSize = 64ULL * 1024ULL * 1024ULL;
    uint64_t minFragThreshold = 2ULL * 1024ULL * 1024ULL;
    uint64_t maxFragThreshold = 4ULL * 1024ULL * 1024ULL * 1024ULL;
    std::shared_ptr<OckHmmFactory> factory;
    std::shared_ptr<OckHmmSingleDeviceMgr> singleMgr;
    std::shared_ptr<OckHmmComposeDeviceMgr> composeMgr;
};

TEST_F(ApiTestParamCheck, create_single_device_mgr_failed_while_input_nullptr)
{
    std::shared_ptr<OckHmmDeviceInfo> devInfo = nullptr;
    auto ret = factory->CreateSingleDeviceMemoryMgr(devInfo);
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_EMPTY);
    EXPECT_EQ(ret.second.get(), nullptr);
}

TEST_F(ApiTestParamCheck, create_compose_device_mgr_failed_while_input_nullptr)
{
    std::shared_ptr<OckHmmDeviceInfoVec> devInfoVec = nullptr;
    auto ret = factory->CreateComposeMemoryMgr(devInfoVec);
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_EMPTY);
    EXPECT_EQ(ret.second.get(), nullptr);
}

TEST_F(ApiTestParamCheck, create_compose_device_mgr_failed_while_empty_deviceInfoVec)
{
    OckHmmDeviceInfoVec devInfoVec;
    auto ret = factory->CreateComposeMemoryMgr(std::make_shared<OckHmmDeviceInfoVec>(devInfoVec));
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_EMPTY);
    EXPECT_EQ(ret.second.get(), nullptr);
}

TEST_F(ApiTestParamCheck, create_single_device_mgr_failed_while_deviceId_out_of_range)
{
    OckHmmDeviceInfo devInfo = BuildDeviceInfo(maxDeviceId + 1U, {0U, 1U});
    auto ret = factory->CreateSingleDeviceMemoryMgr(std::make_shared<OckHmmDeviceInfo>(devInfo));
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_DEVICE_NOT_EXISTS);
    EXPECT_EQ(ret.second.get(), nullptr);
}

TEST_F(ApiTestParamCheck, create_compose_device_mgr_failed_while_deviceId_out_of_range)
{
    auto devInfoVec = std::make_shared<OckHmmDeviceInfoVec>();
    devInfoVec->push_back(BuildDeviceInfo(maxDeviceId + 1U, {0U, 1U}));
    devInfoVec->push_back(BuildDeviceInfo(deviceIdA, {0U, 1U}));
    auto ret = factory->CreateComposeMemoryMgr(devInfoVec);

    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_DEVICE_NOT_EXISTS);
    EXPECT_EQ(ret.second.get(), nullptr);
}

TEST_F(ApiTestParamCheck, create_single_device_mgr_failed_while_deviceId_does_not_exist)
{
    uint32_t deviceCount = 0;
    aclrtGetDeviceCount(&deviceCount);
    OckHmmDeviceInfo devInfo = BuildDeviceInfo(deviceCount, {0U, 1U});
    auto ret = factory->CreateSingleDeviceMemoryMgr(std::make_shared<OckHmmDeviceInfo>(devInfo));
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_DEVICE_NOT_EXISTS);
    EXPECT_EQ(ret.second.get(), nullptr);
}

TEST_F(ApiTestParamCheck, create_compose_device_mgr_failed_while_deviceId_does_not_exist)
{
    uint32_t deviceCount = 0;
    aclrtGetDeviceCount(&deviceCount);
    auto devInfoVec = std::make_shared<OckHmmDeviceInfoVec>();
    devInfoVec->push_back(BuildDeviceInfo(deviceCount, {0U, 1U}));
    devInfoVec->push_back(BuildDeviceInfo(deviceIdA, {0U, 1U}));
    auto ret = factory->CreateComposeMemoryMgr(devInfoVec);

    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_DEVICE_NOT_EXISTS);
    EXPECT_EQ(ret.second.get(), nullptr);
}

TEST_F(ApiTestParamCheck, create_single_device_mgr_failed_while_cpuId_does_not_exist)
{
    uint32_t cpuCount = sysconf(_SC_NPROCESSORS_CONF);
    OckHmmDeviceInfo devInfo = BuildDeviceInfo(deviceIdA, {cpuCount - 1U, cpuCount});
    auto ret = factory->CreateSingleDeviceMemoryMgr(std::make_shared<OckHmmDeviceInfo>(devInfo));
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_CPUID_NOT_EXISTS);
    EXPECT_EQ(ret.second.get(), nullptr);
}

TEST_F(ApiTestParamCheck, create_compose_device_mgr_failed_while_cpuId_does_not_exist)
{
    uint32_t cpuCount = sysconf(_SC_NPROCESSORS_CONF);
    auto devInfoVec = std::make_shared<OckHmmDeviceInfoVec>();
    devInfoVec->push_back(BuildDeviceInfo(deviceIdA, {cpuCount - 1U, cpuCount}));
    devInfoVec->push_back(BuildDeviceInfo(deviceIdB, {0U, 1U}));
    auto ret = factory->CreateComposeMemoryMgr(devInfoVec);

    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_CPUID_NOT_EXISTS);
    EXPECT_EQ(ret.second.get(), nullptr);
}

TEST_F(ApiTestParamCheck, create_single_device_mgr_failed_while_transferThreadNum_out_of_range)
{
    OckHmmDeviceInfo devInfo = BuildDeviceInfo(deviceIdA, {0U, 1U});
    devInfo.transferThreadNum = minTransferThreadNum - 1U;
    auto ret = factory->CreateSingleDeviceMemoryMgr(std::make_shared<OckHmmDeviceInfo>(devInfo));
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ret.second.get(), nullptr);

    devInfo.transferThreadNum = maxTransferThreadNum + 1U;
    ret = factory->CreateSingleDeviceMemoryMgr(std::make_shared<OckHmmDeviceInfo>(devInfo));
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ret.second.get(), nullptr);
}

TEST_F(ApiTestParamCheck, create_compose_device_mgr_failed_while_transferThreadNum_out_of_range)
{
    OckHmmDeviceInfo devInfoA = BuildDeviceInfo(deviceIdA, {0U, 1U});
    OckHmmDeviceInfo devInfoB = BuildDeviceInfo(deviceIdB, {2U, 3U});
    OckHmmDeviceInfoVec devInfoVec = {devInfoA, devInfoB};
    devInfoVec[0].transferThreadNum = minTransferThreadNum - 1U;
    auto ret = factory->CreateComposeMemoryMgr(std::make_shared<OckHmmDeviceInfoVec>(devInfoVec));
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ret.second.get(), nullptr);

    devInfoVec[0].transferThreadNum = maxTransferThreadNum + 1U;
    ret = factory->CreateComposeMemoryMgr(std::make_shared<OckHmmDeviceInfoVec>(devInfoVec));
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ret.second.get(), nullptr);
}

TEST_F(ApiTestParamCheck, create_single_device_mgr_failed_while_device_base_capacity_out_of_range)
{
    OckHmmDeviceInfo devInfo = BuildDeviceInfo(deviceIdA, {0U, 1U});
    devInfo.memorySpec.devSpec.maxDataCapacity = minBaseCapacity - 1ULL;
    auto ret = factory->CreateSingleDeviceMemoryMgr(std::make_shared<OckHmmDeviceInfo>(devInfo));
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ret.second.get(), nullptr);

    devInfo.memorySpec.devSpec.maxDataCapacity = maxBaseCapacity + 1ULL;
    ret = factory->CreateSingleDeviceMemoryMgr(std::make_shared<OckHmmDeviceInfo>(devInfo));
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ret.second.get(), nullptr);
}

TEST_F(ApiTestParamCheck, create_compose_device_mgr_failed_while_device_base_capacity_out_of_range)
{
    OckHmmDeviceInfo devInfoA = BuildDeviceInfo(deviceIdA, {0U, 1U});
    OckHmmDeviceInfo devInfoB = BuildDeviceInfo(deviceIdB, {2U, 3U});
    OckHmmDeviceInfoVec devInfoVec = {devInfoA, devInfoB};
    devInfoVec[0].memorySpec.devSpec.maxDataCapacity = minBaseCapacity - 1U;
    auto ret = factory->CreateComposeMemoryMgr(std::make_shared<OckHmmDeviceInfoVec>(devInfoVec));
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ret.second.get(), nullptr);

    devInfoVec[0].memorySpec.devSpec.maxDataCapacity = maxBaseCapacity + 1U;
    ret = factory->CreateComposeMemoryMgr(std::make_shared<OckHmmDeviceInfoVec>(devInfoVec));
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ret.second.get(), nullptr);
}

TEST_F(ApiTestParamCheck, create_single_device_mgr_failed_while_device_swap_capacity_out_of_range)
{
    OckHmmDeviceInfo devInfo = BuildDeviceInfo(deviceIdA, {0U, 1U});
    devInfo.memorySpec.devSpec.maxSwapCapacity = minSwapCapacity - 1ULL;
    auto ret = factory->CreateSingleDeviceMemoryMgr(std::make_shared<OckHmmDeviceInfo>(devInfo));
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ret.second.get(), nullptr);

    devInfo.memorySpec.devSpec.maxSwapCapacity = maxSwapCapacity + 1ULL;
    ret = factory->CreateSingleDeviceMemoryMgr(std::make_shared<OckHmmDeviceInfo>(devInfo));
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ret.second.get(), nullptr);
}

TEST_F(ApiTestParamCheck, create_compose_device_mgr_failed_while_device_swap_capacity_out_of_range)
{
    OckHmmDeviceInfo devInfoA = BuildDeviceInfo(deviceIdA, {0U, 1U});
    OckHmmDeviceInfo devInfoB = BuildDeviceInfo(deviceIdB, {2U, 3U});
    OckHmmDeviceInfoVec devInfoVec = {devInfoA, devInfoB};
    devInfoVec[0].memorySpec.devSpec.maxSwapCapacity = minSwapCapacity - 1U;
    auto ret = factory->CreateComposeMemoryMgr(std::make_shared<OckHmmDeviceInfoVec>(devInfoVec));
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ret.second.get(), nullptr);

    devInfoVec[0].memorySpec.devSpec.maxSwapCapacity = maxSwapCapacity + 1U;
    ret = factory->CreateComposeMemoryMgr(std::make_shared<OckHmmDeviceInfoVec>(devInfoVec));
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ret.second.get(), nullptr);
}

TEST_F(ApiTestParamCheck, create_single_device_mgr_failed_while_host_base_capacity_out_of_range)
{
    OckHmmDeviceInfo devInfo = BuildDeviceInfo(deviceIdA, {0U, 1U});
    devInfo.memorySpec.hostSpec.maxDataCapacity = minBaseCapacity - 1ULL;
    auto ret = factory->CreateSingleDeviceMemoryMgr(std::make_shared<OckHmmDeviceInfo>(devInfo));
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ret.second.get(), nullptr);

    devInfo.memorySpec.hostSpec.maxDataCapacity = maxBaseCapacity + 1ULL;
    ret = factory->CreateSingleDeviceMemoryMgr(std::make_shared<OckHmmDeviceInfo>(devInfo));
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ret.second.get(), nullptr);
}

TEST_F(ApiTestParamCheck, create_compose_device_mgr_failed_while_host_base_capacity_out_of_range)
{
    OckHmmDeviceInfo devInfoA = BuildDeviceInfo(deviceIdA, {0U, 1U});
    OckHmmDeviceInfo devInfoB = BuildDeviceInfo(deviceIdB, {2U, 3U});
    OckHmmDeviceInfoVec devInfoVec = {devInfoA, devInfoB};
    devInfoVec[0].memorySpec.hostSpec.maxDataCapacity = minBaseCapacity - 1U;
    auto ret = factory->CreateComposeMemoryMgr(std::make_shared<OckHmmDeviceInfoVec>(devInfoVec));
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ret.second.get(), nullptr);

    devInfoVec[0].memorySpec.hostSpec.maxDataCapacity = maxBaseCapacity + 1U;
    ret = factory->CreateComposeMemoryMgr(std::make_shared<OckHmmDeviceInfoVec>(devInfoVec));
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ret.second.get(), nullptr);
}

TEST_F(ApiTestParamCheck, create_single_device_mgr_failed_while_host_swap_capacity_out_of_range)
{
    OckHmmDeviceInfo devInfo = BuildDeviceInfo(deviceIdA, {0U, 1U});
    devInfo.memorySpec.hostSpec.maxSwapCapacity = minSwapCapacity - 1ULL;
    auto ret = factory->CreateSingleDeviceMemoryMgr(std::make_shared<OckHmmDeviceInfo>(devInfo));
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ret.second.get(), nullptr);

    devInfo.memorySpec.hostSpec.maxSwapCapacity = maxSwapCapacity + 1ULL;
    ret = factory->CreateSingleDeviceMemoryMgr(std::make_shared<OckHmmDeviceInfo>(devInfo));
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ret.second.get(), nullptr);
}

TEST_F(ApiTestParamCheck, create_compose_device_mgr_failed_while_host_swap_capacity_out_of_range)
{
    OckHmmDeviceInfo devInfoA = BuildDeviceInfo(deviceIdA, {0U, 1U});
    OckHmmDeviceInfo devInfoB = BuildDeviceInfo(deviceIdB, {2U, 3U});
    OckHmmDeviceInfoVec devInfoVec = {devInfoA, devInfoB};
    devInfoVec[0].memorySpec.hostSpec.maxSwapCapacity = minSwapCapacity - 1U;
    auto ret = factory->CreateComposeMemoryMgr(std::make_shared<OckHmmDeviceInfoVec>(devInfoVec));
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ret.second.get(), nullptr);

    devInfoVec[0].memorySpec.hostSpec.maxSwapCapacity = maxSwapCapacity + 1U;
    ret = factory->CreateComposeMemoryMgr(std::make_shared<OckHmmDeviceInfoVec>(devInfoVec));
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ret.second.get(), nullptr);
}

TEST_F(ApiTestParamCheck, alloc_hmo_by_single_device_mgr_failed_while_hmo_size_out_of_range)
{
    BuildSingleDeviceMgr();
    auto ret = singleMgr->Alloc(minHmoSize);
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ret.second.get(), nullptr);

    ret = singleMgr->Alloc(maxHmoSize + 1ULL);
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ret.second.get(), nullptr);
}

TEST_F(ApiTestParamCheck, alloc_hmo_by_compose_device_mgr_failed_while_hmo_size_out_of_range)
{
    BuildComposeDeviceMgr();
    auto ret = composeMgr->Alloc(minHmoSize);
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ret.second.get(), nullptr);

    ret = composeMgr->Alloc(maxHmoSize + 1ULL);
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ret.second.get(), nullptr);
}

TEST_F(ApiTestParamCheck, alloc_hmo_with_deviceId_by_compose_device_mgr_failed_while_hmo_size_out_of_range)
{
    BuildComposeDeviceMgr();
    auto ret = composeMgr->Alloc(deviceIdA, minHmoSize);
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ret.second.get(), nullptr);

    ret = composeMgr->Alloc(deviceIdB, minHmoSize);
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ret.second.get(), nullptr);

    ret = composeMgr->Alloc(deviceIdA, maxHmoSize + 1ULL);
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ret.second.get(), nullptr);

    ret = composeMgr->Alloc(deviceIdB, maxHmoSize + 1ULL);
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    EXPECT_EQ(ret.second.get(), nullptr);
}

TEST_F(ApiTestParamCheck, alloc_hmo_with_deviceId_by_compose_device_mgr_failed_while_deviceId_does_not_exist)
{
    BuildComposeDeviceMgr();
    auto ret = composeMgr->Alloc(deviceIdC, normalHmoSize);
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_DEVICE_NOT_EXISTS);
    EXPECT_EQ(ret.second.get(), nullptr);

    ret = composeMgr->Alloc(deviceIdD, normalHmoSize);
    EXPECT_EQ(ret.first, HMM_ERROR_INPUT_PARAM_DEVICE_NOT_EXISTS);
    EXPECT_EQ(ret.second.get(), nullptr);
}

TEST_F(ApiTestParamCheck, alloc_hmo_by_single_device_mgr_failed_while_space_not_enough)
{
    BuildSingleDeviceMgr();
    auto ret = singleMgr->Alloc(normalBaseCapacity + 1ULL);
    EXPECT_EQ(ret.first, HMM_ERROR_SPACE_NOT_ENOUGH);

    ret = singleMgr->Alloc(normalBaseCapacity + 1ULL, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
    EXPECT_EQ(ret.first, HMM_ERROR_DEVICE_DATA_SPACE_NOT_ENOUGH);

    ret = singleMgr->Alloc(normalBaseCapacity + 1ULL, OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY);
    EXPECT_EQ(ret.first, HMM_ERROR_HOST_DATA_SPACE_NOT_ENOUGH);
}

TEST_F(ApiTestParamCheck, alloc_hmo_by_compose_device_mgr_failed_while_space_not_enough)
{
    BuildComposeDeviceMgr();
    auto ret = composeMgr->Alloc(normalBaseCapacity + 1ULL);
    EXPECT_EQ(ret.first, HMM_ERROR_SPACE_NOT_ENOUGH);

    ret = composeMgr->Alloc(normalBaseCapacity + 1ULL, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
    EXPECT_EQ(ret.first, HMM_ERROR_SPACE_NOT_ENOUGH);

    ret = composeMgr->Alloc(normalBaseCapacity + 1ULL, OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY);
    EXPECT_EQ(ret.first, HMM_ERROR_SPACE_NOT_ENOUGH);
}

TEST_F(ApiTestParamCheck, alloc_hmo_with_deviceId_by_compose_device_mgr_while_space_not_enough)
{
    BuildComposeDeviceMgr();
    auto ret = composeMgr->Alloc(deviceIdA, normalBaseCapacity + 1ULL);
    EXPECT_EQ(ret.first, HMM_ERROR_SPACE_NOT_ENOUGH);

    ret = composeMgr->Alloc(deviceIdB, normalBaseCapacity + 1ULL);
    EXPECT_EQ(ret.first, HMM_ERROR_SPACE_NOT_ENOUGH);

    ret = composeMgr->Alloc(deviceIdA, normalBaseCapacity + 1ULL, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
    EXPECT_EQ(ret.first, HMM_ERROR_DEVICE_DATA_SPACE_NOT_ENOUGH);

    ret = composeMgr->Alloc(deviceIdB, normalBaseCapacity + 1ULL, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
    EXPECT_EQ(ret.first, HMM_ERROR_DEVICE_DATA_SPACE_NOT_ENOUGH);

    ret = composeMgr->Alloc(deviceIdA, normalBaseCapacity + 1ULL, OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY);
    EXPECT_EQ(ret.first, HMM_ERROR_HOST_DATA_SPACE_NOT_ENOUGH);

    ret = composeMgr->Alloc(deviceIdB, normalBaseCapacity + 1ULL, OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY);
    EXPECT_EQ(ret.first, HMM_ERROR_HOST_DATA_SPACE_NOT_ENOUGH);
}

TEST_F(ApiTestParamCheck, copy_hmo_by_single_device_mgr_failed_while_dstHmo_and_srcHmo_belong_to_diff_devices)
{
    BuildSingleDeviceMgr();
    OckHmmDeviceInfo devInfo = BuildDeviceInfo(deviceIdB, {2U, 3U});
    auto devMgr = factory->CreateSingleDeviceMemoryMgr(std::make_shared<OckHmmDeviceInfo>(devInfo));
    EXPECT_EQ(devMgr.first, HMM_SUCCESS);

    auto dstHmo = devMgr.second->Alloc(normalHmoSize).second;
    auto srcHmo = singleMgr->Alloc(normalHmoSize).second;
    auto ret = singleMgr->CopyHMO(*dstHmo, 0ULL, *srcHmo, 0ULL, normalHmoSize);
    EXPECT_EQ(ret, HMM_ERROR_INPUT_PARAM_DEVICEID_NOT_EQUAL);
}

TEST_F(ApiTestParamCheck, copy_hmo_by_compose_device_mgr_failed_while_dstHmo_and_srcHmo_belong_to_diff_devices)
{
    BuildComposeDeviceMgr();
    auto dstHmo = composeMgr->Alloc(normalHmoSize, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY).second;
    auto srcHmo = composeMgr->Alloc(normalHmoSize, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY).second;
    auto ret = composeMgr->CopyHMO(*dstHmo, 0ULL, *srcHmo, 0ULL, normalBaseCapacity);
    EXPECT_EQ(ret, HMM_ERROR_INPUT_PARAM_DEVICEID_NOT_EQUAL);

    dstHmo = composeMgr->Alloc(deviceIdA, normalHmoSize).second;
    srcHmo = composeMgr->Alloc(deviceIdB, normalHmoSize).second;
    ret = composeMgr->CopyHMO(*dstHmo, 0ULL, *srcHmo, 0ULL, normalBaseCapacity);
    EXPECT_EQ(ret, HMM_ERROR_INPUT_PARAM_DEVICEID_NOT_EQUAL);
}

TEST_F(ApiTestParamCheck, copy_hmo_by_single_device_mgr_failed_while_hmo_is_already_freed)
{
    BuildSingleDeviceMgr();
    auto dstHmo = singleMgr->Alloc(normalHmoSize).second;
    auto srcHmo = singleMgr->Alloc(normalHmoSize).second;
    singleMgr->Free(dstHmo);
    auto ret = singleMgr->CopyHMO(*dstHmo, 0ULL, *srcHmo, 0ULL, normalHmoSize);
    EXPECT_EQ(ret, HMM_ERROR_INPUT_PARAM_DST_OFFSET_EXCEED_SCOPE);
}

TEST_F(ApiTestParamCheck, copy_hmo_by_compose_device_mgr_failed_while_hmo_is_already_freed)
{
    BuildComposeDeviceMgr();
    auto dstHmo = composeMgr->Alloc(deviceIdA, normalHmoSize).second;
    auto srcHmo = composeMgr->Alloc(deviceIdA, normalHmoSize).second;
    composeMgr->Free(dstHmo);
    auto ret = composeMgr->CopyHMO(*dstHmo, 0ULL, *srcHmo, 0ULL, normalHmoSize);
    EXPECT_EQ(ret, HMM_ERROR_INPUT_PARAM_DST_OFFSET_EXCEED_SCOPE);
}

TEST_F(ApiTestParamCheck, copy_hmo_by_single_device_mgr_failed_while_offset_out_of_range)
{
    BuildSingleDeviceMgr();
    auto dstHmo = singleMgr->Alloc(normalHmoSize).second;
    auto srcHmo = singleMgr->Alloc(normalHmoSize).second;
    auto ret = singleMgr->CopyHMO(*dstHmo, normalHmoSize, *srcHmo, 0ULL, 0ULL);
    EXPECT_EQ(ret, HMM_ERROR_INPUT_PARAM_DST_OFFSET_EXCEED_SCOPE);

    ret = singleMgr->CopyHMO(*dstHmo, 0ULL, *srcHmo, normalHmoSize, 0ULL);
    EXPECT_EQ(ret, HMM_ERROR_INPUT_PARAM_SRC_OFFSET_EXCEED_SCOPE);
}

TEST_F(ApiTestParamCheck, copy_hmo_by_compose_device_mgr_failed_while_offset_out_of_range)
{
    BuildComposeDeviceMgr();
    auto dstHmo = composeMgr->Alloc(deviceIdA, normalHmoSize).second;
    auto srcHmo = composeMgr->Alloc(deviceIdA, normalHmoSize).second;
    auto ret = composeMgr->CopyHMO(*dstHmo, normalHmoSize, *srcHmo, 0ULL, 0ULL);
    EXPECT_EQ(ret, HMM_ERROR_INPUT_PARAM_DST_OFFSET_EXCEED_SCOPE);

    ret = composeMgr->CopyHMO(*dstHmo, 0ULL, *srcHmo, normalHmoSize, 0ULL);
    EXPECT_EQ(ret, HMM_ERROR_INPUT_PARAM_SRC_OFFSET_EXCEED_SCOPE);
}

TEST_F(ApiTestParamCheck, copy_hmo_by_single_device_mgr_failed_while_length_out_of_range)
{
    BuildSingleDeviceMgr();
    auto dstHmo = singleMgr->Alloc(normalHmoSize).second;
    auto srcHmo = singleMgr->Alloc(normalHmoSize).second;
    auto ret = singleMgr->CopyHMO(*dstHmo, 1ULL, *srcHmo, 0ULL, normalHmoSize);
    EXPECT_EQ(ret, HMM_ERROR_INPUT_PARAM_DST_LENGTH_EXCEED_SCOPE);

    ret = singleMgr->CopyHMO(*dstHmo, 0ULL, *srcHmo, 1ULL, normalHmoSize);
    EXPECT_EQ(ret, HMM_ERROR_INPUT_PARAM_SRC_LENGTH_EXCEED_SCOPE);
}

TEST_F(ApiTestParamCheck, copy_hmo_by_compose_device_mgr_failed_while_length_out_of_range)
{
    BuildComposeDeviceMgr();
    auto dstHmo = composeMgr->Alloc(deviceIdA, normalHmoSize).second;
    auto srcHmo = composeMgr->Alloc(deviceIdA, normalHmoSize).second;
    auto ret = composeMgr->CopyHMO(*dstHmo, 1ULL, *srcHmo, 0ULL, normalHmoSize);
    EXPECT_EQ(ret, HMM_ERROR_INPUT_PARAM_DST_LENGTH_EXCEED_SCOPE);

    ret = composeMgr->CopyHMO(*dstHmo, 0ULL, *srcHmo, 1ULL, normalHmoSize);
    EXPECT_EQ(ret, HMM_ERROR_INPUT_PARAM_SRC_LENGTH_EXCEED_SCOPE);
}

TEST_F(ApiTestParamCheck, copy_hmo_successfully_while_length_is_zero)
{
    BuildSingleDeviceMgr();
    auto dstHmo = singleMgr->Alloc(normalHmoSize).second;
    auto srcHmo = singleMgr->Alloc(normalHmoSize).second;
    EXPECT_EQ(singleMgr->CopyHMO(*dstHmo, 0ULL, *srcHmo, 0ULL, 0ULL), HMM_SUCCESS);

    BuildComposeDeviceMgr();
    dstHmo = composeMgr->Alloc(deviceIdA, normalHmoSize).second;
    srcHmo = composeMgr->Alloc(deviceIdA, normalHmoSize).second;
    EXPECT_EQ(composeMgr->CopyHMO(*dstHmo, 0ULL, *srcHmo, 0ULL, 0ULL), HMM_SUCCESS);
}

TEST_F(ApiTestParamCheck, get_used_info_failed_while_fragThreshold_out_of_range)
{
    BuildSingleDeviceMgr();
    EXPECT_EQ(singleMgr->GetUsedInfo(minFragThreshold - 1ULL), nullptr);
    EXPECT_EQ(singleMgr->GetUsedInfo(maxFragThreshold + 1ULL), nullptr);

    BuildComposeDeviceMgr();
    EXPECT_EQ(composeMgr->GetUsedInfo(minFragThreshold - 1ULL), nullptr);
    EXPECT_EQ(composeMgr->GetUsedInfo(maxFragThreshold + 1ULL), nullptr);
    EXPECT_EQ(composeMgr->GetUsedInfo(minFragThreshold - 1ULL, deviceIdA), nullptr);
    EXPECT_EQ(composeMgr->GetUsedInfo(maxFragThreshold + 1ULL, deviceIdA), nullptr);
    EXPECT_EQ(composeMgr->GetUsedInfo(minFragThreshold - 1ULL, deviceIdB), nullptr);
    EXPECT_EQ(composeMgr->GetUsedInfo(maxFragThreshold + 1ULL, deviceIdB), nullptr);
}

TEST_F(ApiTestParamCheck, get_used_info_with_deviceId_failed_while_deviceId_does_not_exist)
{
    BuildComposeDeviceMgr();
    EXPECT_EQ(composeMgr->GetUsedInfo(minFragThreshold, deviceIdC), nullptr);
    EXPECT_EQ(composeMgr->GetUsedInfo(maxFragThreshold, deviceIdC), nullptr);
    EXPECT_EQ(composeMgr->GetUsedInfo(minFragThreshold, deviceIdD), nullptr);
    EXPECT_EQ(composeMgr->GetUsedInfo(maxFragThreshold, deviceIdD), nullptr);
}

TEST_F(ApiTestParamCheck, get_TrafficStatistics_info_with_deviceId_failed_while_deviceId_does_not_exist)
{
    BuildComposeDeviceMgr();
    EXPECT_EQ(composeMgr->GetTrafficStatisticsInfo(deviceIdC), nullptr);
    EXPECT_EQ(composeMgr->GetTrafficStatisticsInfo(deviceIdD), nullptr);
}

TEST_F(ApiTestParamCheck, get_cpuSet_with_deviceId_failed_while_deviceId_does_not_exist)
{
    BuildComposeDeviceMgr();
    EXPECT_EQ(composeMgr->GetCpuSet(deviceIdC), nullptr);
    EXPECT_EQ(composeMgr->GetCpuSet(deviceIdD), nullptr);
}

TEST_F(ApiTestParamCheck, get_specific_with_deviceId_failed_while_deviceId_does_not_exist)
{
    BuildComposeDeviceMgr();
    EXPECT_EQ(composeMgr->GetSpecific(deviceIdC), nullptr);
    EXPECT_EQ(composeMgr->GetSpecific(deviceIdD), nullptr);
}

TEST_F(ApiTestParamCheck, get_buffer_failed_while_offset_out_of_range)
{
    BuildSingleDeviceMgr();
    auto hmo = singleMgr->Alloc(normalHmoSize).second;
    EXPECT_EQ(hmo->GetBuffer(OckHmmHeteroMemoryLocation::DEVICE_DDR, normalHmoSize), nullptr);
    EXPECT_EQ(hmo->GetBuffer(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, normalHmoSize), nullptr);
    EXPECT_EQ(hmo->GetBufferAsync(OckHmmHeteroMemoryLocation::DEVICE_DDR, normalHmoSize), nullptr);
    EXPECT_EQ(hmo->GetBufferAsync(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, normalHmoSize), nullptr);
}

TEST_F(ApiTestParamCheck, get_buffer_failed_while_length_out_of_range)
{
    BuildSingleDeviceMgr();
    auto hmo = singleMgr->Alloc(normalHmoSize).second;
    EXPECT_EQ(hmo->GetBuffer(OckHmmHeteroMemoryLocation::DEVICE_DDR, 1ULL, normalHmoSize), nullptr);
    EXPECT_EQ(hmo->GetBuffer(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 1ULL, normalHmoSize), nullptr);
    EXPECT_EQ(hmo->GetBufferAsync(OckHmmHeteroMemoryLocation::DEVICE_DDR, 1ULL, normalHmoSize), nullptr);
    EXPECT_EQ(hmo->GetBufferAsync(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 1ULL, normalHmoSize), nullptr);
}

TEST_F(ApiTestParamCheck, get_buffer_failed_while_swap_space_is_not_enough)
{
    BuildSingleDeviceMgr();
    auto hmo = singleMgr->Alloc(4ULL * normalHmoSize).second;
    uint64_t length = normalSwapCapacity + 1ULL;
    EXPECT_EQ(hmo->GetBuffer(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0ULL, length), nullptr);
    EXPECT_EQ(hmo->GetBufferAsync(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0ULL, length), nullptr);
}

TEST_F(ApiTestParamCheck, flush_data_failed_while_hmo_has_been_released)
{
    BuildSingleDeviceMgr();
    auto hmo = singleMgr->Alloc(normalHmoSize).second;
    auto deviceBuffer = hmo->GetBuffer(OckHmmHeteroMemoryLocation::DEVICE_DDR, 0U, normalHmoSize);
    auto hostBuffer = hmo->GetBuffer(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0U, normalHmoSize);
    singleMgr->Free(hmo);
    EXPECT_EQ(deviceBuffer->FlushData(), HMM_ERROR_HMO_OBJECT_NOT_EXISTS);
    EXPECT_EQ(hostBuffer->FlushData(), HMM_ERROR_HMO_OBJECT_NOT_EXISTS);
}
}  // namespace test
}  // namespace hmm
}  // namespace ock
