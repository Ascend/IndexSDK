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
static constexpr size_t DF_FUZZ_EXEC_COUNT = 3000000;
static constexpr size_t DF_FUZZ_EXEC_SECOND = 10800;

class FuzzTestOckHmmSingleMemoryMgr : public acladapter::WithEnvAclMock<testing::Test> {
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

    void WriteHMO(const std::shared_ptr<OckHmmHMObject> &dstHmo, std::vector<uint8_t> &srcVec)
    {
        auto len = srcVec.size() * sizeof(uint8_t);
        auto buffer = dstHmo->GetBuffer(hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0UL, len);
        memcpy_s(reinterpret_cast<uint8_t *>(dstHmo->Addr()), len, srcVec.data(), len);
        buffer->FlushData();
        dstHmo->ReleaseBuffer(buffer);
    }

    void DestroyHmmDeviceMgr(void)
    {
        if (singleMgr.get() != nullptr) {
            singleMgr.reset();
        }
    }

    std::shared_ptr<OckHmmDeviceInfo> singleDeviceInfo;
    std::shared_ptr<OckHmmSingleDeviceMgr> singleMgr;
    uint64_t fragThreshold = 2ULL * 1024ULL * 1024ULL;
    int32_t minHmoBytes = 1;
    int32_t maxHmoBytes = 3 * 64 * 1024 * 1024;
};

TEST_F(FuzzTestOckHmmSingleMemoryMgr, alloc_and_free_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();

    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "single_mgr_alloc_and_free", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRet = singleMgr->Alloc((uint64_t)mindxHmoBytes);
        singleMgr->Free(devHmoRet.second);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmSingleMemoryMgr, copy_hmo_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "single_mgr_copy_hmo", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        std::vector<uint8_t> dataZeros(mindxHmoBytes, 0);
        std::vector<uint8_t> dataOnes(mindxHmoBytes, 1);
        auto devHmoRet = singleMgr->Alloc((uint64_t)mindxHmoBytes);
        WriteHMO(devHmoRet.second, dataZeros);
        auto hostHmoRet = singleMgr->Alloc((uint64_t)mindxHmoBytes, OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY);
        WriteHMO(hostHmoRet.second, dataOnes);
        singleMgr->CopyHMO(*devHmoRet.second, 0ULL, *hostHmoRet.second, 0ULL, (uint64_t)mindxHmoBytes);
        singleMgr->Free(devHmoRet.second);
        singleMgr->Free(hostHmoRet.second);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmSingleMemoryMgr, get_used_info_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "single_mgr_get_used_info", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);

        auto devHmoRet = singleMgr->Alloc((uint64_t)mindxHmoBytes);
        singleMgr->GetUsedInfo(fragThreshold);
        singleMgr->Free(devHmoRet.second);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmSingleMemoryMgr, get_traffic_statistics_info_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "single_mgr_get_traffic_statistics_info", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);

        auto devHmoRet = singleMgr->Alloc((uint64_t)mindxHmoBytes);
        devHmoRet.second->GetBuffer(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0UL, (uint64_t)mindxHmoBytes);
        singleMgr->GetTrafficStatisticsInfo(1U);
        singleMgr->Free(devHmoRet.second);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmSingleMemoryMgr, get_specific_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "single_mgr_get_specific", 0)
    {
        singleMgr->GetSpecific();
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmSingleMemoryMgr, get_cpu_set_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "single_mgr_get_cpu_set", 0)
    {
        singleMgr->GetCpuSet();
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmSingleMemoryMgr, get_device_id_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "single_mgr_get_device_id", 0)
    {
        singleMgr->GetDeviceId();
    }
    DT_FUZZ_END()
}

// memory pool interface test
TEST_F(FuzzTestOckHmmSingleMemoryMgr, memory_pool_malloc_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;

    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "single_mgr_memory_pool_malloc", 0)
    {
        BuildSingleMgr();
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        singleMgr->Malloc((uint64_t)mindxHmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_FIRST);
    }
    DT_FUZZ_END()
}

// memory guard interface test
TEST_F(FuzzTestOckHmmSingleMemoryMgr, memory_guard_addr_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;

    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "single_mgr_memory_guard_addr", 0)
    {
        BuildSingleMgr();
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto ret = singleMgr->Malloc(mindxHmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_FIRST);
        ret->Addr();
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmSingleMemoryMgr, memory_guard_location_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;

    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "single_mgr_memory_guard_location", 0)
    {
        BuildSingleMgr();
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto ret = singleMgr->Malloc(mindxHmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_FIRST);
        ret->Location();
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmSingleMemoryMgr, memory_guard_byte_size_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildSingleMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "single_mgr_memory_guard_byte_size", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto ret = singleMgr->Malloc(mindxHmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_FIRST);
        ret->ByteSize();
    }
    DT_FUZZ_END()
}
}
}