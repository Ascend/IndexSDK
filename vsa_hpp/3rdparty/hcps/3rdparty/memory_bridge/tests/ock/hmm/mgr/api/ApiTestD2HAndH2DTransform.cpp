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
#include "acl/acl.h"
#include "mockcpp/mockcpp.hpp"
#include "ock/hmm/OckHmmFactory.h"
#include "ock/acladapter/WithEnvAclMock.h"

namespace ock {
namespace hmm {
namespace test {

class ApiTestD2HAndH2DTransform : public acladapter::WithEnvAclMock<testing::Test> {
public:
    ApiTestD2HAndH2DTransform(void) : hmoBytes(64ULL * 1024ULL * 1024ULL), specDataOffset(1998U), specDataValue(0xae)
    {}

    void SetUp(void) override
    {
        acladapter::WithEnvAclMock<testing::Test>::SetUp();
        aclInit(nullptr);
        deviceInfo = std::make_shared<OckHmmDeviceInfo>();
        deviceInfo->deviceId = 0U;
        CPU_ZERO(&deviceInfo->cpuSet);
        CPU_SET(1U, &deviceInfo->cpuSet);
        CPU_SET(2U, &deviceInfo->cpuSet);
        deviceInfo->memorySpec.devSpec.maxDataCapacity = 1ULL * 1024ULL * 1024ULL * 1024ULL;   // 1G
        deviceInfo->memorySpec.devSpec.maxSwapCapacity = 64ULL * 1024ULL * 1024ULL;            // 64M
        deviceInfo->memorySpec.hostSpec.maxDataCapacity = 1ULL * 1024ULL * 1024ULL * 1024ULL;  // 1G
        deviceInfo->memorySpec.hostSpec.maxSwapCapacity = 64ULL * 1024ULL * 1024ULL;           // 64M
        deviceInfo->transferThreadNum = 2ULL;
        BuildSingleDeviceMgr();
    }

    void TearDown(void)
    {
        devMgr.reset();
        aclrtResetDevice(0U);
        aclFinalize();
        acladapter::WithEnvAclMock<testing::Test>::TearDown();
    }

    void BuildSingleDeviceMgr(void)
    {
        auto factory = OckHmmFactory::Create();
        auto ret = factory->CreateSingleDeviceMemoryMgr(deviceInfo);
        ASSERT_EQ(ret.first, HMM_SUCCESS);
        devMgr = ret.second;
    }

    void ExpectGoodHmoAllocRet(std::pair<OckHmmErrorCode, std::shared_ptr<OckHmmHMObject>> ret,
        OckHmmHeteroMemoryLocation location, uint64_t expectHmoSize)
    {
        EXPECT_EQ(ret.first, HMM_SUCCESS);
        ASSERT_TRUE(ret.second.get() != nullptr);
        EXPECT_EQ(ret.second->Location(), location);
        EXPECT_EQ(ret.second->GetByteSize(), expectHmoSize);
        EXPECT_TRUE(ret.second->Addr() != 0ULL);
    }

    void ExpectReleasedHMOBuffer(std::shared_ptr<OckHmmHMOBuffer> buffer)
    {
        EXPECT_EQ(buffer->Address(), 0ULL);
        EXPECT_EQ(buffer->Size(), 0ULL);
        EXPECT_EQ(buffer->Offset(), 0ULL);
        EXPECT_EQ(buffer->Location(), OckHmmHeteroMemoryLocation::DEVICE_DDR);
        EXPECT_EQ(buffer->GetId(), 0ULL);
        EXPECT_EQ(buffer->FlushData(), HMM_ERROR_HMO_BUFFER_NOT_ALLOCED);
        EXPECT_EQ(buffer->ErrorCode(), HMM_ERROR_WAIT_TIME_OUT);
    }

    void WriteDataIntoHost(uintptr_t address, uint32_t pos, uint8_t value)
    {
        uint8_t *addr = reinterpret_cast<uint8_t *>(address);
        addr[pos] = value;
    }

    void ExpectValueInHost(uintptr_t address, uint32_t pos, uint8_t value)
    {
        uint8_t *addr = reinterpret_cast<uint8_t *>(address);
        EXPECT_EQ(addr[pos], value);
    }

    void WriteDataIntoDevice(uintptr_t address, uint32_t pos, uint8_t value)
    {
        aclrtSetDevice(0U);
        uint8_t tempNum = value;
        uint8_t *srcAddr = &tempNum;
        uint8_t *dstAddr = reinterpret_cast<uint8_t *>(address + pos);
        aclrtMemcpy(dstAddr, sizeof(value), srcAddr, sizeof(value), ACL_MEMCPY_HOST_TO_DEVICE);
    }

    void ExpectValueInDevice(uintptr_t address, uint32_t pos, uint8_t value)
    {
        aclrtSetDevice(0U);
        uint8_t expectValue = 0;
        uint8_t *srcAddr = reinterpret_cast<uint8_t *>(address + pos);
        uint8_t *dstAddr = &expectValue;
        aclrtMemcpy(dstAddr, sizeof(value), srcAddr, sizeof(value), ACL_MEMCPY_DEVICE_TO_HOST);
        EXPECT_EQ(expectValue, value);
    }

    const uint64_t hmoBytes;
    const uint32_t specDataOffset;
    const uint8_t specDataValue;
    std::shared_ptr<OckHmmDeviceInfo> deviceInfo;
    std::shared_ptr<OckHmmSingleDeviceMgr> devMgr;
};

TEST_F(ApiTestD2HAndH2DTransform, enough_space_in_buffer)
{
    auto ret = devMgr->Alloc(hmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_FIRST);
    ExpectGoodHmoAllocRet(ret, OckHmmHeteroMemoryLocation::DEVICE_DDR, hmoBytes);
    
    auto deviceBuffer = ret.second->GetBuffer(OckHmmHeteroMemoryLocation::DEVICE_DDR, 0ULL, hmoBytes);
    EXPECT_NE(deviceBuffer, nullptr);
    ret.second->ReleaseBuffer(deviceBuffer);
    ExpectReleasedHMOBuffer(deviceBuffer);

    auto hostBuffer = ret.second->GetBuffer(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0ULL, hmoBytes);
    EXPECT_NE(hostBuffer, nullptr);
    ret.second->ReleaseBuffer(hostBuffer);
    ExpectReleasedHMOBuffer(hostBuffer);
}

TEST_F(ApiTestD2HAndH2DTransform, getBuffer_async_while_enough_space)
{
    auto ret = devMgr->Alloc(hmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_FIRST);
    ExpectGoodHmoAllocRet(ret, OckHmmHeteroMemoryLocation::DEVICE_DDR, hmoBytes);

    auto devAsyncRet = ret.second->GetBufferAsync(OckHmmHeteroMemoryLocation::DEVICE_DDR, 0ULL, hmoBytes);
    EXPECT_NE(devAsyncRet, nullptr);
    auto deviceBuffer = devAsyncRet->WaitResult();
    EXPECT_NE(deviceBuffer->Address(), 0ULL);
    EXPECT_EQ(deviceBuffer->Size(), hmoBytes);
    EXPECT_EQ(deviceBuffer->Offset(), 0ULL);
    EXPECT_EQ(deviceBuffer->Location(), OckHmmHeteroMemoryLocation::DEVICE_DDR);
    EXPECT_EQ(deviceBuffer->GetId(), ret.second->GetId());
    EXPECT_EQ(deviceBuffer->FlushData(), HMM_SUCCESS);
    EXPECT_NE(deviceBuffer->ErrorCode(), HMM_ERROR_WAIT_TIME_OUT);
}

TEST_F(ApiTestD2HAndH2DTransform, not_enough_space_in_buffer)
{
    auto ret = devMgr->Alloc(hmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_FIRST);
    ExpectGoodHmoAllocRet(ret, OckHmmHeteroMemoryLocation::DEVICE_DDR, hmoBytes);
    EXPECT_EQ(ret.second->GetBuffer(OckHmmHeteroMemoryLocation::DEVICE_DDR, 1ULL, hmoBytes), nullptr);
    EXPECT_EQ(ret.second->GetBuffer(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 1ULL, hmoBytes), nullptr);

    // 异步
    EXPECT_EQ(ret.second->GetBufferAsync(OckHmmHeteroMemoryLocation::DEVICE_DDR, 1ULL, hmoBytes), nullptr);
    EXPECT_EQ(ret.second->GetBufferAsync(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 1ULL, hmoBytes), nullptr);
}

TEST_F(ApiTestD2HAndH2DTransform, flush_data_from_device_to_hmo_in_device)
{
    auto ret = devMgr->Alloc(hmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
    ExpectGoodHmoAllocRet(ret, OckHmmHeteroMemoryLocation::DEVICE_DDR, hmoBytes);
    
    auto deviceBuffer = ret.second->GetBuffer(OckHmmHeteroMemoryLocation::DEVICE_DDR, 0ULL, hmoBytes);
    WriteDataIntoDevice(deviceBuffer->Address(), specDataOffset, specDataValue);
    deviceBuffer->FlushData();
    ExpectValueInDevice(ret.second->Addr(), specDataOffset, specDataValue);

    ret.second->ReleaseBuffer(deviceBuffer);
    ExpectReleasedHMOBuffer(deviceBuffer);

    // 异步
    auto asyncRet = ret.second->GetBufferAsync(OckHmmHeteroMemoryLocation::DEVICE_DDR, 0ULL, hmoBytes);
    EXPECT_NE(asyncRet, nullptr);
    auto devAsyncBuffer = asyncRet->WaitResult();
    WriteDataIntoDevice(devAsyncBuffer->Address(), specDataOffset, specDataValue);
    devAsyncBuffer->FlushData();
    ExpectValueInDevice(ret.second->Addr(), specDataOffset, specDataValue);

    ret.second->ReleaseBuffer(devAsyncBuffer);
    ExpectReleasedHMOBuffer(devAsyncBuffer);
}

TEST_F(ApiTestD2HAndH2DTransform, flush_data_from_host_to_hmo_in_device)
{
    auto ret = devMgr->Alloc(hmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
    ExpectGoodHmoAllocRet(ret, OckHmmHeteroMemoryLocation::DEVICE_DDR, hmoBytes);
    
    auto hostBuffer = ret.second->GetBuffer(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0ULL, hmoBytes);
    WriteDataIntoHost(hostBuffer->Address(), specDataOffset, specDataValue);
    hostBuffer->FlushData();
    ExpectValueInDevice(ret.second->Addr(), specDataOffset, specDataValue);

    ret.second->ReleaseBuffer(hostBuffer);
    ExpectReleasedHMOBuffer(hostBuffer);

    // 异步
    auto asyncRet = ret.second->GetBufferAsync(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0ULL, hmoBytes);
    EXPECT_NE(asyncRet, nullptr);
    auto hostAsyncBuffer = asyncRet->WaitResult();
    WriteDataIntoHost(hostAsyncBuffer->Address(), specDataOffset, specDataValue);
    hostAsyncBuffer->FlushData();
    ExpectValueInDevice(ret.second->Addr(), specDataOffset, specDataValue);

    ret.second->ReleaseBuffer(hostAsyncBuffer);
    ExpectReleasedHMOBuffer(hostAsyncBuffer);
}

TEST_F(ApiTestD2HAndH2DTransform, flush_data_from_device_to_hmo_in_host)
{
    auto ret = devMgr->Alloc(hmoBytes, OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY);
    ExpectGoodHmoAllocRet(ret, OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, hmoBytes);
    
    auto deviceBuffer = ret.second->GetBuffer(OckHmmHeteroMemoryLocation::DEVICE_DDR, 0ULL, hmoBytes);
    WriteDataIntoDevice(deviceBuffer->Address(), specDataOffset, specDataValue);
    deviceBuffer->FlushData();
    ExpectValueInHost(ret.second->Addr(), specDataOffset, specDataValue);

    ret.second->ReleaseBuffer(deviceBuffer);
    ExpectReleasedHMOBuffer(deviceBuffer);

    // 异步
    auto asyncRet = ret.second->GetBufferAsync(OckHmmHeteroMemoryLocation::DEVICE_DDR, 0ULL, hmoBytes);
    EXPECT_NE(asyncRet, nullptr);
    auto devAsyncBuffer = asyncRet->WaitResult();
    WriteDataIntoDevice(devAsyncBuffer->Address(), specDataOffset, specDataValue);
    devAsyncBuffer->FlushData();
    ExpectValueInHost(ret.second->Addr(), specDataOffset, specDataValue);

    ret.second->ReleaseBuffer(devAsyncBuffer);
    ExpectReleasedHMOBuffer(devAsyncBuffer);
}

TEST_F(ApiTestD2HAndH2DTransform, flush_data_from_host_to_hmo_in_host)
{
    auto ret = devMgr->Alloc(hmoBytes, OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY);
    ExpectGoodHmoAllocRet(ret, OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, hmoBytes);

    auto hostBuffer = ret.second->GetBuffer(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0ULL, hmoBytes);
    WriteDataIntoHost(hostBuffer->Address(), specDataOffset, specDataValue);
    hostBuffer->FlushData();
    ExpectValueInHost(ret.second->Addr(), specDataOffset, specDataValue);

    ret.second->ReleaseBuffer(hostBuffer);
    ExpectReleasedHMOBuffer(hostBuffer);

    // 异步
    auto asyncRet = ret.second->GetBufferAsync(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0ULL, hmoBytes);
    EXPECT_NE(asyncRet, nullptr);
    auto hostAsyncBuffer = asyncRet->WaitResult();
    WriteDataIntoHost(hostAsyncBuffer->Address(), specDataOffset, specDataValue);
    hostAsyncBuffer->FlushData();
    ExpectValueInHost(ret.second->Addr(), specDataOffset, specDataValue);

    ret.second->ReleaseBuffer(hostAsyncBuffer);
    ExpectReleasedHMOBuffer(hostAsyncBuffer);
}
}  // namespace test
}  // namespace hmm
}  // namespace ock