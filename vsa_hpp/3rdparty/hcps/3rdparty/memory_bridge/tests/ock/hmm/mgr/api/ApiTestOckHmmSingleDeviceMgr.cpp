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
#include "ock/log/OckLogger.h"
#include "ock/hmm/OckHmmFactory.h"
#include "ock/acladapter/WithEnvAclMock.h"
namespace ock {
namespace hmm {
namespace test {

class ApiTestOckHmmSingleDeviceMgr : public acladapter::WithEnvAclMock<testing::Test> {
public:
    // specDataOffset 随便一个位置， 不要超过minxHmoBytes大小， specDataValue 随便一个特殊值
    ApiTestOckHmmSingleDeviceMgr(void)
        : mindxHmoBytes(64ULL * 1024ULL * 1024ULL),
          specDataOffset(5151U),
          specDataValue(0xae),
          maxNumDeviceAndHost(32U),
          maxNumDeviceOrHost(16U),
          incHmoBytes(512ULL * 1024ULL * 1024ULL),
          incDevMemoryByteSize(4ULL * 1024ULL * 1024ULL * 1024ULL + 4ULL * 1024U * 1024U),
          incHostMemoryByteSize(4ULL * 1024ULL * 1024ULL * 1024ULL + 4ULL * 1024U * 1024U),
          maxIncNumDeviceOrHost(8U),
          threshold(5ULL * 1024U * 1024U)
    {}
    void SetUp(void) override
    {
        acladapter::WithEnvAclMock<testing::Test>::SetUp();
        aclInit(nullptr);
        deviceInfo = std::make_shared<OckHmmDeviceInfo>();
        deviceInfo->deviceId = 0U;
        CPU_SET(1U, &deviceInfo->cpuSet);                                                      // 设置1号CPU核
        CPU_SET(2U, &deviceInfo->cpuSet);                                                      // 设置2号CPU核
        deviceInfo->memorySpec.devSpec.maxDataCapacity = 1024ULL * 1024ULL * 1024ULL;   // 1G
        deviceInfo->memorySpec.devSpec.maxSwapCapacity = 3ULL * 64ULL * 1024ULL * 1024ULL;     // 3 * 64M
        deviceInfo->memorySpec.hostSpec.maxDataCapacity = 1024ULL * 1024ULL * 1024ULL;  // 1G
        deviceInfo->memorySpec.hostSpec.maxSwapCapacity = 3ULL * 64ULL * 1024ULL * 1024ULL;    // 3 * 64M
        deviceInfo->transferThreadNum = 2ULL;                                                  // 2个线程
    }

    void TearDown(void) override
    {
        DestroyHmmDeviceMgr();  // 需要提前reset，否则打桩不生效。
        aclrtResetDevice(0);
        aclFinalize();
        acladapter::WithEnvAclMock<testing::Test>::TearDown();
    }
    void BuildHmmDeviceMgr(void)
    {
        auto factory = OckHmmFactory::Create();
        auto ret = factory->CreateSingleDeviceMemoryMgr(deviceInfo);
        ASSERT_EQ(ret.first, HMM_SUCCESS);
        devMgrErrorCode = ret.first;
        devMgr = ret.second;
    }
    void DestroyHmmDeviceMgr(void)
    {
        if (devMgr.get() != nullptr) {
            devMgr.reset();
        }
    }
    void ExpectGoodHmoAllocRet(std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmHMObject>> hmoAllRet,
        OckHmmHeteroMemoryLocation location, uint32_t expectHmoSize)
    {
        EXPECT_EQ(hmoAllRet.first, HMM_SUCCESS);
        ASSERT_TRUE(hmoAllRet.second.get() != nullptr);
        EXPECT_EQ(hmoAllRet.second->Location(), location);
        EXPECT_EQ(hmoAllRet.second->GetByteSize(), expectHmoSize);
        EXPECT_TRUE(hmoAllRet.second->Addr() != 0ULL);
    }
    void ExpectDeviceToHostTrafficData(void)
    {
        auto trafficData = devMgr->GetTrafficStatisticsInfo(1U);
        EXPECT_EQ(trafficData->device2hostMovedBytes, mindxHmoBytes);
        EXPECT_EQ(trafficData->host2DeviceMovedBytes, 0ULL);
        EXPECT_GT(trafficData->device2hostSpeed, 0ULL);
        EXPECT_EQ(trafficData->host2DeviceSpeed, 0ULL);
    }
    void ExpectZeroTrafficData(void)
    {
        auto trafficData = devMgr->GetTrafficStatisticsInfo(1U);
        EXPECT_EQ(trafficData->device2hostMovedBytes, 0ULL);
        EXPECT_EQ(trafficData->host2DeviceMovedBytes, 0ULL);
        EXPECT_EQ(trafficData->device2hostSpeed, 0ULL);
        EXPECT_EQ(trafficData->host2DeviceSpeed, 0ULL);
    }
    void ExpectUsedInfo(uint32_t devUsedBytes)
    {
        auto usedInfo = devMgr->GetUsedInfo(2ULL * 1024ULL * 1024ULL);
        EXPECT_EQ(usedInfo->devUsedInfo.usedBytes, devUsedBytes);
        EXPECT_EQ(usedInfo->devUsedInfo.unusedFragBytes, 0ULL);
    }
    void WriteLocalHostData(uintptr_t intAddr, uint32_t pos, uint8_t value)
    {
        uint8_t *addr = reinterpret_cast<uint8_t *>(intAddr);
        addr[pos] = value;
    }
    void ExpectLocalHostData(uintptr_t intAddr, uint32_t pos, uint8_t value)
    {
        uint8_t *addr = reinterpret_cast<uint8_t *>(intAddr);
        EXPECT_EQ(addr[pos], value);
    }
    void CompareUsedInfo(OckHmmMemoryUsedInfo usedInfo, uint64_t usedByteSize,
                         uint64_t unusedFragByteSize, uint64_t leftByteSize)
    {
        EXPECT_EQ(usedInfo.usedBytes, usedByteSize);
        EXPECT_EQ(usedInfo.unusedFragBytes, unusedFragByteSize);
        EXPECT_EQ(usedInfo.leftBytes, leftByteSize);
    }
    void CompareMemorySpecification(OckHmmMemorySpecification specification, uint64_t maxHostDataCapacity)
    {
        EXPECT_EQ(specification.devSpec.maxDataCapacity, deviceInfo->memorySpec.devSpec.maxDataCapacity);
        EXPECT_EQ(specification.devSpec.maxSwapCapacity, deviceInfo->memorySpec.devSpec.maxSwapCapacity);
        EXPECT_EQ(specification.hostSpec.maxDataCapacity, maxHostDataCapacity);
        EXPECT_EQ(specification.hostSpec.maxSwapCapacity, deviceInfo->memorySpec.hostSpec.maxSwapCapacity);
    }
    void WriteToHMO(std::shared_ptr<OckHmmHMObject> writeHmo)
    {
        try {
            int *hmoPtr = reinterpret_cast<int *>(writeHmo->Addr());
            *hmoPtr = 100L;
        } catch (std::exception &e) {
            OCK_HMM_LOG_ERROR("Address is invalid!");
        }
    }
    const uint64_t mindxHmoBytes;
    const uint32_t specDataOffset;
    const uint8_t specDataValue;
    std::shared_ptr<OckHmmDeviceInfo> deviceInfo;
    std::shared_ptr<OckHmmSingleDeviceMgr> devMgr;
    OckHmmErrorCode devMgrErrorCode;
    int maxNumDeviceAndHost;
    int maxNumDeviceOrHost;
    uint64_t incHmoBytes;
    uint64_t incDevMemoryByteSize;      // 新增的 dev 内存大小
    uint64_t incHostMemoryByteSize;     // 新增的 host 内存大小
    int maxIncNumDeviceOrHost;          // 可新分配的最大 hmo 数量
    uint64_t threshold;
};
TEST_F(ApiTestOckHmmSingleDeviceMgr, zero_traffic)
{
    BuildHmmDeviceMgr();
    ExpectZeroTrafficData();
}
TEST_F(ApiTestOckHmmSingleDeviceMgr, get_traffic)
{
    BuildHmmDeviceMgr();
    auto devHmoRet = devMgr->Alloc(mindxHmoBytes);
    ExpectGoodHmoAllocRet(devHmoRet, OckHmmHeteroMemoryLocation::DEVICE_DDR, mindxHmoBytes);
    auto buffer = devHmoRet.second->GetBuffer(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0UL, mindxHmoBytes);
    ExpectDeviceToHostTrafficData();
}
TEST_F(ApiTestOckHmmSingleDeviceMgr, copy_hmo_correct)
{
    BuildHmmDeviceMgr();
    auto devHmoRet = devMgr->Alloc(mindxHmoBytes);
    ExpectGoodHmoAllocRet(devHmoRet, OckHmmHeteroMemoryLocation::DEVICE_DDR, mindxHmoBytes);
    auto buffer = devHmoRet.second->GetBuffer(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0UL, mindxHmoBytes);
    WriteLocalHostData(buffer->Address(), specDataOffset, specDataValue);

    buffer->FlushData();
    devHmoRet.second->ReleaseBuffer(buffer);

    auto hostHmoRet = devMgr->Alloc(mindxHmoBytes, OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY);
    ExpectGoodHmoAllocRet(hostHmoRet, OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, mindxHmoBytes);

    EXPECT_EQ(devMgr->CopyHMO(*hostHmoRet.second, 0ULL, *devHmoRet.second, 0ULL, mindxHmoBytes), HMM_SUCCESS);
    auto rawData = devMgr->Malloc(mindxHmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_FIRST);

    ExpectLocalHostData(hostHmoRet.second->Addr(), specDataOffset, specDataValue);
}

TEST_F(ApiTestOckHmmSingleDeviceMgr, create_and_destroy_mgr)
{
    BuildHmmDeviceMgr();
    EXPECT_EQ(devMgrErrorCode, HMM_SUCCESS);
    EXPECT_NE(devMgr.get(), nullptr);

    DestroyHmmDeviceMgr();
    EXPECT_EQ(devMgr.get(), nullptr);
}

TEST_F(ApiTestOckHmmSingleDeviceMgr, creat_hmo_default_policy_device_space_enough)
{
    BuildHmmDeviceMgr();
    auto devHmoRet = devMgr->Alloc(mindxHmoBytes);
    ExpectGoodHmoAllocRet(devHmoRet, OckHmmHeteroMemoryLocation::DEVICE_DDR, mindxHmoBytes);
}

TEST_F(ApiTestOckHmmSingleDeviceMgr, creat_hmo_default_policy_use_host_space)
{
    BuildHmmDeviceMgr();
    // 已经使用默认策略创建了 maxNumDeviceOrHost 个HMO对象，用尽了device侧分配的内存
    std::vector<std::shared_ptr<OckHmmHMObject>> hmos;
    for (int i = 0; i < maxNumDeviceOrHost; i++) {
        auto ret = devMgr->Alloc(mindxHmoBytes);
        if (ret.first == HMM_SUCCESS) {
            hmos.push_back(ret.second);
        }
    }
    auto hostHmoRet = devMgr->Alloc(mindxHmoBytes);
    ExpectGoodHmoAllocRet(hostHmoRet, OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, mindxHmoBytes);
}

TEST_F(ApiTestOckHmmSingleDeviceMgr, creat_hmo_default_no_space)
{
    BuildHmmDeviceMgr();
    // 已经使用默认策略创建了 maxNumDeviceAndHost 个HMO对象，用尽了device侧和host侧分配的内存
    std::vector<std::shared_ptr<OckHmmHMObject>> hmos;
    for (int i = 0; i < maxNumDeviceAndHost; i++) {
        auto ret = devMgr->Alloc(mindxHmoBytes);
        if (ret.first == HMM_SUCCESS) {
            hmos.push_back(ret.second);
        }
    }
    auto hmoRet = devMgr->Alloc(mindxHmoBytes);
    EXPECT_NE(hmoRet.first, HMM_SUCCESS);
}

TEST_F(ApiTestOckHmmSingleDeviceMgr, creat_hmo_device_policy_device_space_enough)
{
    BuildHmmDeviceMgr();
    auto devHmoRet = devMgr->Alloc(mindxHmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
    ExpectGoodHmoAllocRet(devHmoRet, OckHmmHeteroMemoryLocation::DEVICE_DDR, mindxHmoBytes);
}

TEST_F(ApiTestOckHmmSingleDeviceMgr, creat_hmo_device_policy_device_space_not_enough)
{
    BuildHmmDeviceMgr();
    // 已经使用默认策略创建了 maxNumDeviceOrHost 个HMO对象，用尽了device侧分配的内存
    std::vector<std::shared_ptr<OckHmmHMObject>> hmos;
    for (int i = 0; i < maxNumDeviceOrHost; i++) {
        auto ret = devMgr->Alloc(mindxHmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
        if (ret.first == HMM_SUCCESS) {
            hmos.push_back(ret.second);
        }
    }
    auto devHmoRet = devMgr->Alloc(mindxHmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
    EXPECT_NE(devHmoRet.first, HMM_SUCCESS);
}

TEST_F(ApiTestOckHmmSingleDeviceMgr, creat_hmo_host_policy_host_space_enough)
{
    BuildHmmDeviceMgr();
    auto hostHmoRet = devMgr->Alloc(mindxHmoBytes, OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY);
    ExpectGoodHmoAllocRet(hostHmoRet, OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, mindxHmoBytes);
}

TEST_F(ApiTestOckHmmSingleDeviceMgr, creat_hmo_host_policy_host_space_not_enough)
{
    BuildHmmDeviceMgr();
    // 已经使用默认策略创建了 maxNumDeviceOrHost 个HMO对象，用尽了device侧分配的内存
    std::vector<std::shared_ptr<OckHmmHMObject>> hmos;
    for (int i = 0; i < maxNumDeviceOrHost; i++) {
        auto ret = devMgr->Alloc(mindxHmoBytes, OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY);
        if (ret.first == HMM_SUCCESS) {
            hmos.push_back(ret.second);
        }
    }
    auto hostHmoRet = devMgr->Alloc(mindxHmoBytes, OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY);
    EXPECT_NE(hostHmoRet.first, HMM_SUCCESS);
}

TEST_F(ApiTestOckHmmSingleDeviceMgr, destory_and_create_one_hmo_when_device_full)
{
    BuildHmmDeviceMgr();

    // 已经使用默认策略创建了 maxNumDeviceOrHost 个HMO对象，用尽了device侧分配的内存
    std::vector<std::shared_ptr<OckHmmHMObject>> hmos;
    for (int i = 0; i < maxNumDeviceOrHost; i++) {
        auto ret = devMgr->Alloc(mindxHmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
        if (ret.first == HMM_SUCCESS) {
            hmos.push_back(ret.second);
        }
    }

    devMgr->Free(hmos.back());
    auto devHmoRet = devMgr->Alloc(mindxHmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
    ExpectGoodHmoAllocRet(devHmoRet, OckHmmHeteroMemoryLocation::DEVICE_DDR, mindxHmoBytes);
}

TEST_F(ApiTestOckHmmSingleDeviceMgr, destory_and_create_one_hmo_when_host_full)
{
    BuildHmmDeviceMgr();
    // 已经使用默认策略创建了 maxNumDeviceOrHost 个HMO对象，用尽了host侧分配的内存
    std::vector<std::shared_ptr<OckHmmHMObject>> hmos;
    for (int i = 0; i < maxNumDeviceOrHost; i++) {
        auto ret = devMgr->Alloc(mindxHmoBytes, OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY);
        if (ret.first == HMM_SUCCESS) {
            hmos.push_back(ret.second);
        }
    }
    devMgr->Free(hmos.back());
    auto hostHmoRet = devMgr->Alloc(mindxHmoBytes, OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY);
    ExpectGoodHmoAllocRet(hostHmoRet, OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, mindxHmoBytes);
}

TEST_F(ApiTestOckHmmSingleDeviceMgr, get_used_info_after_create_one_hmo)
{
    BuildHmmDeviceMgr();
    auto devHmoRet = devMgr->Alloc(mindxHmoBytes);
    // 将小于2M的内存块视为碎片
    auto usedInfo = devMgr->GetUsedInfo(2ULL * 1024ULL * 1024ULL);
    EXPECT_EQ(usedInfo->devUsedInfo.usedBytes, mindxHmoBytes);
    EXPECT_EQ(usedInfo->devUsedInfo.unusedFragBytes, 0ULL);
    EXPECT_EQ(usedInfo->devUsedInfo.leftBytes, deviceInfo->memorySpec.devSpec.maxDataCapacity - mindxHmoBytes);
    EXPECT_EQ(usedInfo->devUsedInfo.swapUsedBytes, 0ULL);
    EXPECT_EQ(usedInfo->devUsedInfo.swapLeftBytes, deviceInfo->memorySpec.devSpec.maxSwapCapacity);
    EXPECT_EQ(usedInfo->hostUsedInfo.usedBytes, 0ULL);
    EXPECT_EQ(usedInfo->hostUsedInfo.unusedFragBytes, 0ULL);
    EXPECT_EQ(usedInfo->hostUsedInfo.leftBytes, deviceInfo->memorySpec.hostSpec.maxDataCapacity);
    EXPECT_EQ(usedInfo->hostUsedInfo.swapUsedBytes, 0ULL);
    EXPECT_EQ(usedInfo->hostUsedInfo.swapLeftBytes, deviceInfo->memorySpec.hostSpec.maxSwapCapacity);
}

TEST_F(ApiTestOckHmmSingleDeviceMgr, copy_hmo_from_host_to_device)
{
    BuildHmmDeviceMgr();
    aclrtSetDevice(0);
    std::vector<uint8_t> dataZeros(mindxHmoBytes, 0);
    std::vector<uint8_t> dataOnes(mindxHmoBytes, 1);
    std::vector<uint8_t> dataDevice(mindxHmoBytes, 0);

    auto devHmoRet = devMgr->Alloc(mindxHmoBytes);
    aclrtMemcpy(reinterpret_cast<void *>(devHmoRet.second->Addr()),
        mindxHmoBytes,
        dataZeros.data(),
        mindxHmoBytes,
        ACL_MEMCPY_HOST_TO_DEVICE);
    auto hostHmoRet = devMgr->Alloc(mindxHmoBytes, OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY);
    aclrtMemcpy(reinterpret_cast<void *>(hostHmoRet.second->Addr()),
        mindxHmoBytes,
        dataOnes.data(),
        mindxHmoBytes,
        ACL_MEMCPY_HOST_TO_HOST);

    auto ret = devMgr->CopyHMO(*devHmoRet.second, 0ULL, *hostHmoRet.second, 0ULL, mindxHmoBytes);
    EXPECT_EQ(ret, HMM_SUCCESS);
    aclrtMemcpy(dataDevice.data(),
        mindxHmoBytes,
        reinterpret_cast<void *>(devHmoRet.second->Addr()),
        mindxHmoBytes,
        ACL_MEMCPY_DEVICE_TO_HOST);
    EXPECT_EQ(dataDevice, dataOnes);
}

TEST_F(ApiTestOckHmmSingleDeviceMgr, get_host_buffer_in_host_and_free)
{
    BuildHmmDeviceMgr();
    auto hostHmoRet = devMgr->Alloc(mindxHmoBytes, OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY);
    auto buffer = hostHmoRet.second->GetBuffer(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0UL, mindxHmoBytes);
    EXPECT_EQ(buffer->Address(), hostHmoRet.second->Addr());
    hostHmoRet.second->ReleaseBuffer(buffer);
    EXPECT_EQ(reinterpret_cast<void *>(buffer->Address()), nullptr);
}

TEST_F(ApiTestOckHmmSingleDeviceMgr, get_device_buffer_in_device_and_free)
{
    BuildHmmDeviceMgr();
    auto devHmoRet = devMgr->Alloc(mindxHmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
    auto buffer = devHmoRet.second->GetBuffer(OckHmmHeteroMemoryLocation::DEVICE_DDR, 0UL, mindxHmoBytes);
    EXPECT_EQ(buffer->Address(), devHmoRet.second->Addr());
    devHmoRet.second->ReleaseBuffer(buffer);
    EXPECT_EQ(reinterpret_cast<void *>(buffer->Address()), nullptr);
}

TEST_F(ApiTestOckHmmSingleDeviceMgr, host_space_not_enough_increase_host_space)
{
    BuildHmmDeviceMgr();
    // 已经使用默认策略创建了 maxNumDeviceOrHost 个HMO对象，用尽了 host 侧分配的内存
    std::vector<std::shared_ptr<OckHmmHMObject>> hmos;
    for (int i = 0; i < maxNumDeviceOrHost; i++) {
        auto ret = devMgr->Alloc(mindxHmoBytes, OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY);
        if (ret.first == HMM_SUCCESS) {
            hmos.push_back(ret.second);
        }
    }
    auto hostHmoRet = devMgr->Alloc(mindxHmoBytes, OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY);
    EXPECT_NE(hostHmoRet.first, HMM_SUCCESS);
    // increase host space
    devMgr->IncBindMemory(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, incHostMemoryByteSize);
    std::vector<std::shared_ptr<OckHmmHMObject>> hmosInc;
    for (int i = 0; i < maxIncNumDeviceOrHost; i++) {
        auto ret = devMgr->Alloc(incHmoBytes, OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY);
        EXPECT_EQ(ret.first, HMM_SUCCESS);
        if (ret.first == HMM_SUCCESS) {
            hmosInc.push_back(ret.second);
        }
    }
    auto incHostHmoRet = devMgr->Alloc(incHmoBytes, OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY);
    EXPECT_NE(incHostHmoRet.first, HMM_SUCCESS);
    EXPECT_NO_THROW(WriteToHMO(hmosInc.front()));
    devMgr->Free(hmosInc.back());

    auto usedInfo = devMgr->GetUsedInfo(threshold);
    CompareUsedInfo(usedInfo->devUsedInfo, 0ULL, 0ULL, deviceInfo->memorySpec.devSpec.maxDataCapacity);
    EXPECT_EQ(usedInfo->hostUsedInfo.usedBytes,
        deviceInfo->memorySpec.hostSpec.maxDataCapacity + incHmoBytes * (maxIncNumDeviceOrHost - 1ULL));
    EXPECT_EQ(usedInfo->hostUsedInfo.unusedFragBytes, 0ULL);
    EXPECT_EQ(usedInfo->hostUsedInfo.leftBytes, incHostMemoryByteSize - incHmoBytes * (maxIncNumDeviceOrHost - 1ULL));

    auto specific = devMgr->GetSpecific();
    CompareMemorySpecification(specific, deviceInfo->memorySpec.hostSpec.maxDataCapacity + incHostMemoryByteSize);
    incHostHmoRet = devMgr->Alloc(incHmoBytes, OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY);
    EXPECT_EQ(incHostHmoRet.first, HMM_SUCCESS);
}

TEST_F(ApiTestOckHmmSingleDeviceMgr, IncBindMemoryParam)
{
    BuildHmmDeviceMgr();
    auto incRet = devMgr->IncBindMemory(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 2ULL * 1024U * 1024U * 1024U);
    EXPECT_EQ(incRet, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    incRet = devMgr->IncBindMemory(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 4ULL * 1024U * 1024U * 1024U);
    EXPECT_EQ(incRet, HMM_SUCCESS);
    incRet = devMgr->IncBindMemory(OckHmmHeteroMemoryLocation::DEVICE_DDR, 4ULL * 1024U * 1024U * 1024U);
    EXPECT_EQ(incRet, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    incRet = devMgr->IncBindMemory(OckHmmHeteroMemoryLocation::DEVICE_HBM, 4ULL * 1024U * 1024U * 1024U);
    EXPECT_EQ(incRet, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
    incRet = devMgr->IncBindMemory(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 100ULL * 1024U * 1024U * 1024U);
    EXPECT_EQ(incRet, HMM_SUCCESS);
    incRet = devMgr->IncBindMemory(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 101ULL * 1024U * 1024U * 1024U);
    EXPECT_EQ(incRet, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
}
}  // namespace test
}  // namespace hmm
}  // namespace ock
