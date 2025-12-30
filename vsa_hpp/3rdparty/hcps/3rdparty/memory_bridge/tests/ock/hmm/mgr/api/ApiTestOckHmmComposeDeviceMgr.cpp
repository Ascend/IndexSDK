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
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include "ock/utils/OckSafeUtils.h"
#include "ock/log/OckLogger.h"
#include "ock/hmm/OckHmmFactory.h"
#include "ock/acladapter/WithEnvAclMock.h"

namespace ock {
namespace hmm {
namespace test {

class ApiTestOckHmmComposeDeviceMgr : public acladapter::WithEnvAclMock<testing::Test> {
public:
    ApiTestOckHmmComposeDeviceMgr()
        : mindxHmoBytes(64ULL * 1024ULL * 1024ULL), fragThreshold(2ULL * 1024ULL * 1024ULL), deviceIdA(0U),
          deviceIdB(1U), deviceIdD(3U)
    {}
    void SetUp(void) override
    {
        acladapter::WithEnvAclMock<testing::Test>::SetUp();
        aclInit(nullptr);
        deviceInfoVec = std::make_shared<OckHmmDeviceInfoVec>();
        deviceInfoVec->push_back(BuildDeviceInfo(deviceIdA, {0U, 1U}));
        deviceInfoVec->push_back(BuildDeviceInfo(deviceIdB, {2U, 3U}));
        BuildComposeDeviceMgr();
    }

    void TearDown(void) override
    {
        composeMgr.reset();
        aclrtResetDevice(0);
        aclFinalize();
        acladapter::WithEnvAclMock<testing::Test>::TearDown();
    }

    void BuildComposeDeviceMgr(void)
    {
        auto factory = OckHmmFactory::Create();
        auto ret = factory->CreateComposeMemoryMgr(deviceInfoVec);
        ASSERT_EQ(ret.first, HMM_SUCCESS);
        composeMgr = ret.second;
    }

    OckHmmDeviceInfo BuildDeviceInfo(OckHmmDeviceId deviceId, std::vector<uint32_t> cpuIds)
    {
        OckHmmDeviceInfo deviceInfo;
        deviceInfo.deviceId = deviceId;
        CPU_ZERO(&deviceInfo.cpuSet);
        for (uint32_t i = 0; i < cpuIds.size(); ++i) {
            CPU_SET(cpuIds[i], &deviceInfo.cpuSet);
        }
        deviceInfo.memorySpec.devSpec.maxDataCapacity = 1024ULL * 1024ULL * 1024ULL;
        deviceInfo.memorySpec.devSpec.maxSwapCapacity = 3ULL * 64ULL * 1024ULL * 1024ULL;
        deviceInfo.memorySpec.hostSpec.maxDataCapacity = 1024ULL * 1024ULL * 1024ULL;
        deviceInfo.memorySpec.hostSpec.maxSwapCapacity = 3ULL * 64ULL * 1024ULL * 1024ULL;
        deviceInfo.transferThreadNum = 2ULL;
        return deviceInfo;
    }

    void TestCompareUsedInfo(OckHmmMemoryUsedInfo usedInfo, uint64_t expectValue)
    {
        EXPECT_EQ(usedInfo.usedBytes, expectValue);
        EXPECT_EQ(usedInfo.unusedFragBytes, 0ULL);
        EXPECT_EQ(usedInfo.leftBytes, deviceInfoVec->back().memorySpec.devSpec.maxDataCapacity - expectValue);
        EXPECT_EQ(usedInfo.swapUsedBytes, 0ULL);
        EXPECT_EQ(usedInfo.swapLeftBytes, deviceInfoVec->back().memorySpec.devSpec.maxSwapCapacity);
    }

    void ExpectDeviceToHostTrafficData(void)
    {
        auto trafficData = composeMgr->GetTrafficStatisticsInfo(1U);
        EXPECT_GE(trafficData->device2hostMovedBytes, mindxHmoBytes);
        EXPECT_EQ(trafficData->host2DeviceMovedBytes, 0ULL);
        EXPECT_GT(trafficData->device2hostSpeed, 0ULL);
        EXPECT_EQ(trafficData->host2DeviceSpeed, 0ULL);
    }

    void ExpectDeviceToHostTrafficData(OckHmmDeviceId deviceId)
    {
        auto trafficData = composeMgr->GetTrafficStatisticsInfo(deviceId, 1U);
        EXPECT_EQ(trafficData->device2hostMovedBytes, mindxHmoBytes);
        EXPECT_EQ(trafficData->host2DeviceMovedBytes, 0ULL);
        EXPECT_GT(trafficData->device2hostSpeed, 0ULL);
        EXPECT_EQ(trafficData->host2DeviceSpeed, 0ULL);
    }

    void ExpectTrafficDataZero(OckHmmDeviceId deviceId)
    {
        auto trafficData = composeMgr->GetTrafficStatisticsInfo(deviceId, 1U);
        EXPECT_EQ(trafficData->device2hostMovedBytes, 0ULL);
        EXPECT_EQ(trafficData->host2DeviceMovedBytes, 0ULL);
        EXPECT_EQ(trafficData->device2hostSpeed, 0ULL);
        EXPECT_EQ(trafficData->host2DeviceSpeed, 0ULL);
    }

    const uint64_t mindxHmoBytes;
    const uint64_t fragThreshold;
    OckHmmDeviceId deviceIdA;
    OckHmmDeviceId deviceIdB;
    OckHmmDeviceId deviceIdD;

    std::shared_ptr<OckHmmDeviceInfoVec> deviceInfoVec;
    std::shared_ptr<OckHmmComposeDeviceMgr> composeMgr;
};
TEST_F(ApiTestOckHmmComposeDeviceMgr, no_traffic_with_device_id)
{
    ExpectTrafficDataZero(deviceIdA);
    ExpectTrafficDataZero(deviceIdB);
}
TEST_F(ApiTestOckHmmComposeDeviceMgr, traffic_with_device_id)
{
    auto devHmoRetA = composeMgr->Alloc(deviceIdA, mindxHmoBytes);
    auto bufferA = devHmoRetA.second->GetBuffer(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0UL, mindxHmoBytes);
    auto devHmoRetB = composeMgr->Alloc(deviceIdB, mindxHmoBytes);
    auto bufferB = devHmoRetB.second->GetBuffer(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0UL, mindxHmoBytes);
    ExpectDeviceToHostTrafficData(deviceIdB);
}
TEST_F(ApiTestOckHmmComposeDeviceMgr, traffic_without_device_id)
{
    auto devHmoRet1 = composeMgr->Alloc(mindxHmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
    auto devHmoRet2 = composeMgr->Alloc(mindxHmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
    auto devHmoRet3 = composeMgr->Alloc(mindxHmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
    auto buffer1 = devHmoRet1.second->GetBuffer(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0UL, mindxHmoBytes);
    auto buffer2 = devHmoRet2.second->GetBuffer(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0UL, mindxHmoBytes);
    auto buffer3 = devHmoRet3.second->GetBuffer(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0UL, mindxHmoBytes);
    ExpectDeviceToHostTrafficData();
}
TEST_F(ApiTestOckHmmComposeDeviceMgr, build_compose_device_mgr_successfully)
{
    EXPECT_NE(composeMgr, nullptr);
}
TEST_F(ApiTestOckHmmComposeDeviceMgr, alloc_hmo)
{
    EXPECT_EQ(composeMgr->Alloc(deviceIdA, mindxHmoBytes).first, HMM_SUCCESS);
    EXPECT_EQ(composeMgr->Alloc(deviceIdD, mindxHmoBytes).first, HMM_ERROR_INPUT_PARAM_DEVICE_NOT_EXISTS);
}
TEST_F(ApiTestOckHmmComposeDeviceMgr, get_used_info_with_deviceId)
{
    auto devHmoRet = composeMgr->Alloc(deviceIdA, mindxHmoBytes);
    auto usedInfo = composeMgr->GetUsedInfo(fragThreshold, deviceIdA);
    TestCompareUsedInfo(usedInfo->devUsedInfo, mindxHmoBytes);
    TestCompareUsedInfo(usedInfo->hostUsedInfo, 0ULL);
}
TEST_F(ApiTestOckHmmComposeDeviceMgr, copy_hmo_with_same_device)
{
    aclrtSetDevice(0);
    std::vector<uint8_t> dataZeros(mindxHmoBytes, 0);
    std::vector<uint8_t> dataOnes(mindxHmoBytes, 1);
    std::vector<uint8_t> dataDevice(mindxHmoBytes, 0);

    auto devHmoRet = composeMgr->Alloc(deviceIdA, mindxHmoBytes);
    aclrtMemcpy(reinterpret_cast<void *>(devHmoRet.second->Addr()),
        mindxHmoBytes,
        dataZeros.data(),
        mindxHmoBytes,
        ACL_MEMCPY_HOST_TO_DEVICE);
    auto hostHmoRet = composeMgr->Alloc(deviceIdA, mindxHmoBytes, OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY);
    aclrtMemcpy(reinterpret_cast<void *>(hostHmoRet.second->Addr()),
        mindxHmoBytes,
        dataOnes.data(),
        mindxHmoBytes,
        ACL_MEMCPY_HOST_TO_HOST);

    auto ret = composeMgr->CopyHMO(*devHmoRet.second, 0ULL, *hostHmoRet.second, 0ULL, mindxHmoBytes);
    EXPECT_EQ(ret, HMM_SUCCESS);
    aclrtMemcpy(dataDevice.data(),
        mindxHmoBytes,
        reinterpret_cast<void *>(devHmoRet.second->Addr()),
        mindxHmoBytes,
        ACL_MEMCPY_DEVICE_TO_HOST);
    EXPECT_EQ(dataDevice, dataOnes);
}
TEST_F(ApiTestOckHmmComposeDeviceMgr, copy_hmo_with_different_device)
{
    aclrtSetDevice(0);
    std::vector<uint8_t> dataZeros(mindxHmoBytes, 0);
    std::vector<uint8_t> dataOnes(mindxHmoBytes, 1);
    std::vector<uint8_t> dataDevice(mindxHmoBytes, 0);

    auto devHmoRet = composeMgr->Alloc(deviceIdA, mindxHmoBytes);
    aclrtMemcpy(reinterpret_cast<void *>(devHmoRet.second->Addr()),
        mindxHmoBytes,
        dataZeros.data(),
        mindxHmoBytes,
        ACL_MEMCPY_HOST_TO_DEVICE);
    auto hostHmoRet = composeMgr->Alloc(deviceIdB, mindxHmoBytes, OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY);
    aclrtMemcpy(reinterpret_cast<void *>(hostHmoRet.second->Addr()),
        mindxHmoBytes,
        dataOnes.data(),
        mindxHmoBytes,
        ACL_MEMCPY_HOST_TO_HOST);

    auto ret = composeMgr->CopyHMO(*devHmoRet.second, 0ULL, *hostHmoRet.second, 0ULL, mindxHmoBytes);
    EXPECT_EQ(ret, HMM_ERROR_INPUT_PARAM_DEVICEID_NOT_EQUAL);
    aclrtMemcpy(dataDevice.data(),
        mindxHmoBytes,
        reinterpret_cast<void *>(devHmoRet.second->Addr()),
        mindxHmoBytes,
        ACL_MEMCPY_DEVICE_TO_HOST);
    EXPECT_EQ(dataDevice, dataZeros);
}
}  // namespace test
}  // namespace hmm
}  // namespace ock
