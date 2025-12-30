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
static constexpr size_t DF_FUZZ_EXEC_COUNT = 3000000;
static constexpr size_t DF_FUZZ_EXEC_SECOND = 10800;

class FuzzTestOckHmmComposeMemoryMgr : public acladapter::WithEnvAclMock<testing::Test> {
public:
    void SetUpForFuzz(void)
    {
        DT_Set_Running_Time_Second(DF_FUZZ_EXEC_SECOND);
    }

    void SetUp(void) override
    {
        acladapter::WithEnvAclMock<testing::Test>::SetUp();
        aclInit(nullptr);
        deviceInfoVec = std::make_shared<OckHmmDeviceInfoVec>();
        deviceInfoVec->push_back(BuildDeviceInfo(deviceIdA, { 0U, 1U }));
        deviceInfoVec->push_back(BuildDeviceInfo(deviceIdB, { 2U, 3U }));
    }

    void TearDown(void) override
    {
        composeMgr.reset();
        aclFinalize();
        acladapter::WithEnvAclMock<testing::Test>::TearDown();
    }

    void BuildComposeMgr(void)
    {
        auto factory = ock::hmm::OckHmmFactory::Create();
        auto ret = factory->CreateComposeMemoryMgr(deviceInfoVec);
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
        deviceInfo.memorySpec.devSpec.maxDataCapacity = 2ULL * 1024ULL * 1024ULL * 1024ULL;
        deviceInfo.memorySpec.devSpec.maxSwapCapacity = 3ULL * 64ULL * 1024ULL * 1024ULL;
        deviceInfo.memorySpec.hostSpec.maxDataCapacity = 2ULL * 1024ULL * 1024ULL * 1024ULL;
        deviceInfo.memorySpec.hostSpec.maxSwapCapacity = 3ULL * 64ULL * 1024ULL * 1024ULL;
        deviceInfo.transferThreadNum = 2ULL;
        return deviceInfo;
    }

    void WriteHMO(const std::shared_ptr<OckHmmHMObject> &dstHmo, std::vector<uint8_t> &srcVec)
    {
        auto len = srcVec.size() * sizeof(uint8_t);
        auto buffer = dstHmo->GetBuffer(hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0UL, len);
        memcpy_s(reinterpret_cast<uint8_t *>(dstHmo->Addr()), len, srcVec.data(), len);
        buffer->FlushData();
        dstHmo->ReleaseBuffer(buffer);
    }

    std::shared_ptr<OckHmmDeviceInfoVec> deviceInfoVec;
    std::shared_ptr<OckHmmComposeDeviceMgr> composeMgr;
    OckHmmDeviceId deviceIdA = 0U;
    OckHmmDeviceId deviceIdB = 1U;
    int32_t minHmoBytes = 1;
    int32_t maxHmoBytes = 3 * 64 * 1024 * 1024;
    uint64_t fragThreshold = 2ULL * 1024ULL * 1024ULL;
};


TEST_F(FuzzTestOckHmmComposeMemoryMgr, alloc_and_free_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildComposeMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "compose_mgr_alloc_and_free", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRetA = composeMgr->Alloc(mindxHmoBytes);
        composeMgr->Free(devHmoRetA.second);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmComposeMemoryMgr, alloc_device_id_and_free_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildComposeMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "compose_mgr_free", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRetA = composeMgr->Alloc(deviceIdA, mindxHmoBytes);
        auto devHmoRetB = composeMgr->Alloc(deviceIdB, mindxHmoBytes);
        composeMgr->Free(devHmoRetA.second);
        composeMgr->Free(devHmoRetB.second);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmComposeMemoryMgr, copy_hmo_with_same_device_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildComposeMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "compose_mgr_copy_hmo_with_same_device", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        std::vector<uint8_t> dataZeros(mindxHmoBytes, 0);
        std::vector<uint8_t> dataOnes(mindxHmoBytes, 1);
        auto devHmoRet = composeMgr->Alloc(deviceIdA, mindxHmoBytes);
        WriteHMO(devHmoRet.second, dataZeros);
        auto hostHmoRet = composeMgr->Alloc(deviceIdA, mindxHmoBytes, OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY);
        WriteHMO(hostHmoRet.second, dataOnes);
        auto ret = composeMgr->CopyHMO(*devHmoRet.second, 0ULL, *hostHmoRet.second, 0ULL, mindxHmoBytes);
        composeMgr->Free(devHmoRet.second);
        composeMgr->Free(hostHmoRet.second);
    }

    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmComposeMemoryMgr, get_used_info_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildComposeMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "compose_mgr_get_used_info", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRetA = composeMgr->Alloc(deviceIdA, mindxHmoBytes);
        auto devHmoRetB = composeMgr->Alloc(deviceIdB, mindxHmoBytes);
        composeMgr->GetUsedInfo(fragThreshold);
        composeMgr->Free(devHmoRetA.second);
        composeMgr->Free(devHmoRetB.second);
    }

    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmComposeMemoryMgr, get_traffic_statistics_info_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildComposeMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "compose_mgr_get_traffic_statistics_info", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRetA = composeMgr->Alloc(deviceIdA, mindxHmoBytes);
        auto devHmoRetB = composeMgr->Alloc(deviceIdB, mindxHmoBytes);
        devHmoRetA.second->GetBuffer(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0UL, mindxHmoBytes);
        devHmoRetB.second->GetBuffer(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0UL, mindxHmoBytes);
        auto trafficData = composeMgr->GetTrafficStatisticsInfo(1U);
        composeMgr->Free(devHmoRetA.second);
        composeMgr->Free(devHmoRetB.second);
    }

    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmComposeMemoryMgr, get_used_info_with_device_id_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildComposeMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "compose_mgr_get_used_info_with_device_id", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRetA = composeMgr->Alloc(deviceIdA, mindxHmoBytes);
        auto devHmoRetB = composeMgr->Alloc(deviceIdB, mindxHmoBytes);
        composeMgr->GetUsedInfo(fragThreshold, deviceIdA);
        composeMgr->GetUsedInfo(fragThreshold, deviceIdB);
        composeMgr->Free(devHmoRetA.second);
        composeMgr->Free(devHmoRetB.second);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmComposeMemoryMgr, get_traffic_statistics_info_with_device_id_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildComposeMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "compose_mgr_get_traffic_statistics_info_with_device_id", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        auto devHmoRetB = composeMgr->Alloc(deviceIdB, mindxHmoBytes);
        devHmoRetB.second->GetBuffer(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0UL, mindxHmoBytes);
        composeMgr->GetTrafficStatisticsInfo(deviceIdB, 1U);
        composeMgr->Free(devHmoRetB.second);
    }

    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmComposeMemoryMgr, get_cpu_set_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildComposeMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "compose_mgr_get_cpu_set", 0)
    {
        composeMgr->GetCpuSet(deviceIdA);
        composeMgr->GetCpuSet(deviceIdB);
    }

    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmComposeMemoryMgr, alloc_with_device_id_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildComposeMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "compose_mgr_alloc_with_device_id", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        composeMgr->Alloc(deviceIdA, mindxHmoBytes);
        composeMgr->Alloc(deviceIdB, mindxHmoBytes);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmComposeMemoryMgr, get_specific_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildComposeMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "compose_mgr_get_specific", 0)
    {
        composeMgr->GetSpecific(deviceIdA);
        composeMgr->GetSpecific(deviceIdB);
    }
    DT_FUZZ_END()
}

// compose mgr memory pool interface
TEST_F(FuzzTestOckHmmComposeMemoryMgr, memory_pool_malloc_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    BuildComposeMgr();
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "compose_mgr_memory_pool_malloc", 0)
    {
        s32 mindxHmoBytes = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 64 * 1024 * 1024, minHmoBytes, maxHmoBytes);
        composeMgr->Malloc(mindxHmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_FIRST);
    }
    DT_FUZZ_END()
}
}
}
}
