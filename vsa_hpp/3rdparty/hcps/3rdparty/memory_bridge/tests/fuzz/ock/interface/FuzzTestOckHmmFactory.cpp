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

#include <unistd.h>
#include <iostream>
#include "gtest/gtest.h"
#include "secodeFuzz.h"
#include "ock/acladapter/WithEnvAclMock.h"
#include "acl/acl.h"
#include "ock/hmm/OckHmmFactory.h"

namespace ock {
namespace hmm {
static constexpr size_t DF_FUZZ_EXEC_COUNT = 300000;
static constexpr size_t DF_FUZZ_EXEC_SECOND = 10800;
class FuzzTestOckHmmFactory : public acladapter::WithEnvAclMock<testing::Test> {
public:
    void SetUpForFuzz(void)
    {
        DT_Set_Running_Time_Second(DF_FUZZ_EXEC_SECOND);
    }

    void SetUp(void) override
    {
        acladapter::WithEnvAclMock<testing::Test>::SetUp();
        aclInit(nullptr);
    }

    void SetUpForSingleMgr(uint64_t devData, uint64_t devSwap, uint64_t hostData, uint64_t hostSwap)
    {
        singleDeviceInfo = std::make_shared<OckHmmDeviceInfo>();
        singleDeviceInfo->deviceId = 0U;
        CPU_SET(1U, &singleDeviceInfo->cpuSet);                           // 设置1号CPU核
        CPU_SET(2U, &singleDeviceInfo->cpuSet);                           // 设置2号CPU核
        singleDeviceInfo->memorySpec.devSpec.maxDataCapacity = devData;   // 2G
        singleDeviceInfo->memorySpec.devSpec.maxSwapCapacity = devSwap;   // 3 * 64M
        singleDeviceInfo->memorySpec.hostSpec.maxDataCapacity = hostData; // 2G
        singleDeviceInfo->memorySpec.hostSpec.maxSwapCapacity = hostSwap; // 3 * 64M
        singleDeviceInfo->transferThreadNum = 2ULL;
    }

    void TearDown(void) override
    {
        aclFinalize();
        acladapter::WithEnvAclMock<testing::Test>::TearDown();
    }

    void SetUpForComposeMgr(uint64_t devData, uint64_t devSwap, uint64_t hostData, uint64_t hostSwap)
    {
        OckHmmMemorySpecification memorySpec;
        memorySpec.devSpec.maxDataCapacity = devData;
        memorySpec.devSpec.maxSwapCapacity = devSwap;
        memorySpec.hostSpec.maxDataCapacity = hostData;
        memorySpec.hostSpec.maxSwapCapacity = hostSwap;
        deviceInfoVec = std::make_shared<OckHmmDeviceInfoVec>();
        deviceInfoVec->push_back(BuildDeviceInfo(0U, { 0U, 1U }, memorySpec));
        deviceInfoVec->push_back(BuildDeviceInfo(1U, { 2U, 3U }, memorySpec));
    }

    OckHmmDeviceInfo BuildDeviceInfo(OckHmmDeviceId deviceId, std::vector<uint32_t> cpuIds,
        OckHmmMemorySpecification memorySpec)
    {
        OckHmmDeviceInfo deviceInfo;
        deviceInfo.deviceId = deviceId;
        CPU_ZERO(&deviceInfo.cpuSet);
        for (uint32_t i = 0; i < cpuIds.size(); ++i) {
            CPU_SET(cpuIds[i], &deviceInfo.cpuSet);
        }
        deviceInfo.memorySpec = memorySpec;
        deviceInfo.transferThreadNum = 2ULL;
        return deviceInfo;
    }

    std::shared_ptr<OckHmmDeviceInfo> singleDeviceInfo;
    std::shared_ptr<OckHmmDeviceInfoVec> deviceInfoVec;
    int minDataBase = 32 * 1024;
    int maxDataBase = 256 * 1024;
    int minSwapBase = 8 * 1024;
    int maxSwapBase = 64 * 1024;
};

TEST_F(FuzzTestOckHmmFactory, create_factory_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "create_factory_with_no_memory_leak", 0)
    {
        auto factory = ock::hmm::OckHmmFactory::Create();
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmFactory, create_single_mgr_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    SetUpForFuzz();
    uint32_t seed = 0;
    auto factory = ock::hmm::OckHmmFactory::Create();
    factory->CreateSingleDeviceMemoryMgr(singleDeviceInfo, 0);
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "create_single_mgr_with_no_memory_leak", 0)
    {
        s32 dataBase = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 32 * 1024, minDataBase, maxDataBase);
        s32 swapBase = *(s32 *)DT_SetGetNumberRange(&g_Element[1], 32 * 1024, minSwapBase, maxSwapBase);
        u64 devData = u64(dataBase) * u64(dataBase);
        u64 devSwap = u64(swapBase) * u64(swapBase);
        u64 hostData = u64(dataBase) * u64(dataBase);
        u64 hostSwap = u64(swapBase) * u64(swapBase);
        SetUpForSingleMgr(devData, devSwap, hostData, hostSwap);
        factory->CreateSingleDeviceMemoryMgr(singleDeviceInfo, 0);
    }
    DT_FUZZ_END()
}

TEST_F(FuzzTestOckHmmFactory, create_compose_mgr_with_no_memory_leak)
{
    DT_Enable_Leak_Check(0, 0);
    SetUpForFuzz();
    uint32_t seed = 0;
    auto factory = ock::hmm::OckHmmFactory::Create();
    factory->CreateComposeMemoryMgr(deviceInfoVec, 0);
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "create_compose_mgr_with_no_memory_leak", 0)
    {
        s32 dataBase = *(s32 *)DT_SetGetNumberRange(&g_Element[0], 32 * 1024, minDataBase, maxDataBase);
        s32 swapBase = *(s32 *)DT_SetGetNumberRange(&g_Element[1], 32 * 1024, minSwapBase, maxSwapBase);
        u64 devData = u64(dataBase) * u64(dataBase);
        u64 devSwap = u64(swapBase) * u64(swapBase);
        u64 hostData = u64(dataBase) * u64(dataBase);
        u64 hostSwap = u64(swapBase) * u64(swapBase);
        SetUpForComposeMgr(devData, devSwap, hostData, hostSwap);
        auto factory = ock::hmm::OckHmmFactory::Create();
        factory->CreateComposeMemoryMgr(deviceInfoVec, 0);
    }
    DT_FUZZ_END()
}
}
}
