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


#include "gtest/gtest.h"
#include "secodeFuzz.h"
#include "acl/acl.h"
#include "ock/acladapter/WithEnvAclMock.h"
#include "ock/hmm/OckHmmFactory.h"

namespace ock {
namespace hmm {
namespace test {
static constexpr size_t DF_FUZZ_EXEC_COUNT = 300000;
static constexpr size_t DF_FUZZ_EXEC_SECOND = 10800;
class FuzzTestOckHmmHMObject : public acladapter::WithEnvAclMock<testing::Test> {
public:
    void SetUpForFuzz(void)
    {
        DT_Set_Running_Time_Second(DF_FUZZ_EXEC_SECOND);
    }

    void SetUp(void) override
    {
        acladapter::WithEnvAclMock<testing::Test>::SetUp();
        aclInit(nullptr);
        singleDeviceInfo = std::make_shared<OckHmmDeviceInfo>();
        singleDeviceInfo->deviceId = 0U;
        CPU_SET(1U, &singleDeviceInfo->cpuSet);                                                     // 设置1号CPU核
        CPU_SET(2U, &singleDeviceInfo->cpuSet);                                                     // 设置2号CPU核
        singleDeviceInfo->memorySpec.devSpec.maxDataCapacity = 2ULL * 1024ULL * 1024ULL * 1024ULL;  // 2G
        singleDeviceInfo->memorySpec.devSpec.maxSwapCapacity = 3ULL * 64ULL * 1024ULL * 1024ULL;    // 3 * 64M
        singleDeviceInfo->memorySpec.hostSpec.maxDataCapacity = 2ULL * 1024ULL * 1024ULL * 1024ULL; // 2G
        singleDeviceInfo->memorySpec.hostSpec.maxSwapCapacity = 3ULL * 64ULL * 1024ULL * 1024ULL;   // 3 * 64M
        singleDeviceInfo->transferThreadNum = 2ULL;                                                 // 2个线程
    }

    void TearDown(void) override
    {
        DestroyHmmDeviceMgr(); // 需要提前reset，否则打桩不生效。
        aclFinalize();
        acladapter::WithEnvAclMock<testing::Test>::TearDown();
    }

    void BuildSingleMgr(void)
    {
        auto factory = ock::hmm::OckHmmFactory::Create();
        auto ret = factory->CreateSingleDeviceMemoryMgr(singleDeviceInfo);
        singleMgr = ret.second;
    }

    void DestroyHmmDeviceMgr(void)
    {
        if (singleMgr.get() != nullptr) {
            singleMgr.reset();
        }
    }

    std::shared_ptr<OckHmmDeviceInfo> singleDeviceInfo;
    std::shared_ptr<OckHmmSingleDeviceMgr> singleMgr;
    int32_t minHmoBytes = 1;
    int32_t maxHmoBytes = 3 * 64 * 1024 * 1024;
    uint64_t fragThreshold = 2ULL * 1024ULL * 1024ULL;
};

TEST_F(FuzzTestOckHmmHMObject, get_buffer_and_release_with_diff_location_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "get_buffer_with_diff_location", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRet = singleMgr->Alloc(mindxHmoBytes);
        auto buffer = devHmoRet.second->GetBuffer(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0, mindxHmoBytes);
        devHmoRet.second->ReleaseBuffer(buffer);
        singleMgr->Free(devHmoRet.second);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmHMObject, get_buffer_and_release_with_same_location_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "get_buffer_with_same_location_with_no_memory_leak", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRet = singleMgr->Alloc(mindxHmoBytes);
        auto buffer = devHmoRet.second->GetBuffer(OckHmmHeteroMemoryLocation::DEVICE_DDR, 0, mindxHmoBytes);
        devHmoRet.second->ReleaseBuffer(buffer);
        singleMgr->Free(devHmoRet.second);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmHMObject, release_buffer_with_hmo_released)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "get_buffer_with_same_location_with_no_memory_leak", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRet = singleMgr->Alloc(mindxHmoBytes);
        auto buffer = devHmoRet.second->GetBuffer(OckHmmHeteroMemoryLocation::DEVICE_DDR, 0, mindxHmoBytes);
        devHmoRet.second->ReleaseBuffer(buffer);
        singleMgr->Free(devHmoRet.second);
        devHmoRet.second->ReleaseBuffer(buffer);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmHMObject, get_buffer_async_with_diff_location_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "get_buffer_async_with_diff_location_with_no_memory_leak", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRet = singleMgr->Alloc(mindxHmoBytes);
        auto buffer = devHmoRet.second->GetBufferAsync(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0, mindxHmoBytes);
        devHmoRet.second->ReleaseBuffer(buffer->WaitResult());
        singleMgr->Free(devHmoRet.second);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmHMObject, get_buffer_async_with_same_location_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "get_buffer_with_same_location_with_no_memory_leak", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRet = singleMgr->Alloc(mindxHmoBytes);
        auto buffer = devHmoRet.second->GetBufferAsync(OckHmmHeteroMemoryLocation::DEVICE_DDR, 0, mindxHmoBytes);
        devHmoRet.second->ReleaseBuffer(buffer->WaitResult());
        singleMgr->Free(devHmoRet.second);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmHMObject, hmo_location_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "hmo_location_with_no_memory_leak", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRet = singleMgr->Alloc(mindxHmoBytes);
        devHmoRet.second->Location();
        singleMgr->Free(devHmoRet.second);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmHMObject, hmo_location_with_hmo_released)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "hmo_location_with_no_memory_leak", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRet = singleMgr->Alloc(mindxHmoBytes);
        devHmoRet.second->Location();
        singleMgr->Free(devHmoRet.second);
        devHmoRet.second->Location();
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmHMObject, hmo_addr_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "hmo_addr_with_no_memory_leak", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRet = singleMgr->Alloc(mindxHmoBytes);
        devHmoRet.second->Addr();
        singleMgr->Free(devHmoRet.second);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmHMObject, hmo_addr_with_hmo_released)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "hmo_addr_with_no_memory_leak", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRet = singleMgr->Alloc(mindxHmoBytes);
        devHmoRet.second->Addr();
        singleMgr->Free(devHmoRet.second);
        devHmoRet.second->Addr();
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmHMObject, get_byte_size_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "get_byte_size_with_no_memory_leak", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRet = singleMgr->Alloc(mindxHmoBytes);
        devHmoRet.second->GetByteSize();
        singleMgr->Free(devHmoRet.second);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmHMObject, get_byte_size_with_hmo_released)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "get_byte_size_with_no_memory_leak", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRet = singleMgr->Alloc(mindxHmoBytes);
        devHmoRet.second->GetByteSize();
        singleMgr->Free(devHmoRet.second);
        devHmoRet.second->GetByteSize();
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmHMObject, hmo_get_id_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "hmo_get_id_with_no_memory_leak", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRet = singleMgr->Alloc(mindxHmoBytes);
        devHmoRet.second->GetId();
        singleMgr->Free(devHmoRet.second);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmHMObject, hmo_get_id_with_hmo_released)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "hmo_get_id_with_no_memory_leak", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRet = singleMgr->Alloc(mindxHmoBytes);
        devHmoRet.second->GetId();
        singleMgr->Free(devHmoRet.second);
        devHmoRet.second->GetId();
    }
    DT_FUZZ_END()
}

// hmo buffer interface test
TEST_F(FuzzTestOckHmmHMObject, buffer_address_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "buffer_address_with_no_memory_leak", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRet = singleMgr->Alloc(mindxHmoBytes);
        auto buffer = devHmoRet.second->GetBuffer(OckHmmHeteroMemoryLocation::DEVICE_DDR, 0, mindxHmoBytes);
        buffer->Address();
        devHmoRet.second->ReleaseBuffer(buffer);
        singleMgr->Free(devHmoRet.second);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmHMObject, buffer_address_with_buffer_released)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "buffer_address_with_no_memory_leak", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRet = singleMgr->Alloc(mindxHmoBytes);
        auto buffer = devHmoRet.second->GetBuffer(OckHmmHeteroMemoryLocation::DEVICE_DDR, 0, mindxHmoBytes);
        devHmoRet.second->ReleaseBuffer(buffer);
        buffer->Address();
        singleMgr->Free(devHmoRet.second);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmHMObject, buffer_size_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "buffer_size_with_no_memory_leak", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRet = singleMgr->Alloc(mindxHmoBytes);
        auto buffer = devHmoRet.second->GetBuffer(OckHmmHeteroMemoryLocation::DEVICE_DDR, 0, mindxHmoBytes);
        buffer->Size();
        devHmoRet.second->ReleaseBuffer(buffer);
        singleMgr->Free(devHmoRet.second);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmHMObject, buffer_size_with_buffer_released)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "buffer_size_with_no_memory_leak", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRet = singleMgr->Alloc(mindxHmoBytes);
        auto buffer = devHmoRet.second->GetBuffer(OckHmmHeteroMemoryLocation::DEVICE_DDR, 0, mindxHmoBytes);
        devHmoRet.second->ReleaseBuffer(buffer);
        buffer->Size();
        singleMgr->Free(devHmoRet.second);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmHMObject, buffer_offset_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "buffer_offset_with_no_memory_leak", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRet = singleMgr->Alloc(mindxHmoBytes);
        auto buffer = devHmoRet.second->GetBuffer(OckHmmHeteroMemoryLocation::DEVICE_DDR, 0, mindxHmoBytes);
        buffer->Offset();
        devHmoRet.second->ReleaseBuffer(buffer);
        singleMgr->Free(devHmoRet.second);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmHMObject, buffer_offset_with_buffer_released)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "buffer_offset_with_no_memory_leak", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRet = singleMgr->Alloc(mindxHmoBytes);
        auto buffer = devHmoRet.second->GetBuffer(OckHmmHeteroMemoryLocation::DEVICE_DDR, 0, mindxHmoBytes);
        devHmoRet.second->ReleaseBuffer(buffer);
        buffer->Offset();
        singleMgr->Free(devHmoRet.second);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmHMObject, buffer_location_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "buffer_location_with_no_memory_leak", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRet = singleMgr->Alloc(mindxHmoBytes);
        auto buffer = devHmoRet.second->GetBuffer(OckHmmHeteroMemoryLocation::DEVICE_DDR, 0, mindxHmoBytes);
        buffer->Location();
        devHmoRet.second->ReleaseBuffer(buffer);
        singleMgr->Free(devHmoRet.second);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmHMObject, buffer_location_with_buffer_released)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "buffer_location_with_no_memory_leak", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRet = singleMgr->Alloc(mindxHmoBytes);
        auto buffer = devHmoRet.second->GetBuffer(OckHmmHeteroMemoryLocation::DEVICE_DDR, 0, mindxHmoBytes);
        devHmoRet.second->ReleaseBuffer(buffer);
        buffer->Location();
        singleMgr->Free(devHmoRet.second);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmHMObject, buffer_get_id_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "buffer_get_id_with_no_memory_leak", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRet = singleMgr->Alloc(mindxHmoBytes);
        auto buffer = devHmoRet.second->GetBuffer(OckHmmHeteroMemoryLocation::DEVICE_DDR, 0, mindxHmoBytes);
        buffer->GetId();
        devHmoRet.second->ReleaseBuffer(buffer);
        singleMgr->Free(devHmoRet.second);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmHMObject, buffer_get_id_with_buffer_released)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "buffer_get_id_with_no_memory_leak", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRet = singleMgr->Alloc(mindxHmoBytes);
        auto buffer = devHmoRet.second->GetBuffer(OckHmmHeteroMemoryLocation::DEVICE_DDR, 0, mindxHmoBytes);
        devHmoRet.second->ReleaseBuffer(buffer);
        buffer->GetId();
        singleMgr->Free(devHmoRet.second);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmHMObject, buffer_flush_data_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "buffer_flush_data_with_no_memory_leak", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRet = singleMgr->Alloc(mindxHmoBytes);
        auto buffer = devHmoRet.second->GetBuffer(OckHmmHeteroMemoryLocation::DEVICE_DDR, 0, mindxHmoBytes);
        buffer->FlushData();
        devHmoRet.second->ReleaseBuffer(buffer);
        singleMgr->Free(devHmoRet.second);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmHMObject, buffer_flush_data_with_buffer_released)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "buffer_flush_data_with_no_memory_leak", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRet = singleMgr->Alloc(mindxHmoBytes);
        auto buffer = devHmoRet.second->GetBuffer(OckHmmHeteroMemoryLocation::DEVICE_DDR, 0, mindxHmoBytes);
        devHmoRet.second->ReleaseBuffer(buffer);
        buffer->FlushData();
        singleMgr->Free(devHmoRet.second);
    }
    DT_FUZZ_END()
}

// OckHmmAsyncResult interface test
TEST_F(FuzzTestOckHmmHMObject, async_result_wait_result_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "async_result_wait_result_with_no_memory_leak", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRet = singleMgr->Alloc(mindxHmoBytes);
        auto bufferAsync = devHmoRet.second->GetBufferAsync(OckHmmHeteroMemoryLocation::DEVICE_DDR, 0, mindxHmoBytes);
        auto buffer = bufferAsync->WaitResult();
        devHmoRet.second->ReleaseBuffer(buffer);
        singleMgr->Free(devHmoRet.second);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmHMObject, async_result_cancel_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "async_result_cancel_with_no_memory_leak", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRet = singleMgr->Alloc(mindxHmoBytes);
        auto bufferAsync = devHmoRet.second->GetBufferAsync(OckHmmHeteroMemoryLocation::DEVICE_DDR, 0, mindxHmoBytes);
        bufferAsync->Cancel();
        auto buffer = bufferAsync->WaitResult();
        if (buffer != nullptr) {
            devHmoRet.second->ReleaseBuffer(buffer);
        }
        singleMgr->Free(devHmoRet.second);
    }
    DT_FUZZ_END()
}
}
}
}