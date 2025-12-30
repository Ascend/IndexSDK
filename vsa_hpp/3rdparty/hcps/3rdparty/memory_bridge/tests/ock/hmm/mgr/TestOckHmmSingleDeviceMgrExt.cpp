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
#include <thread>
#include <chrono>
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include "ock/hmm/mgr/WithEnvOckHmmSingleDeviceMgrExt.h"
#include "ock/hmm/mgr/data/OckHmmHMOObjectIDGenerator.h"
namespace ock {
namespace hmm {
namespace test {
class TestOckHmmSingleDeviceMgrExt : public WithEnvOckHmmSingleDeviceMgrExt<testing::Test> {
public:
    using BaseT = WithEnvOckHmmSingleDeviceMgrExt<testing::Test>;
    void ExpectCpuSetEQ(const cpu_set_t &lhs, const cpu_set_t &rhs)
    {
        for (uint32_t i = 0; i < __CPU_SETSIZE / __NCPUBITS; ++i) {
            EXPECT_EQ(lhs.__bits[i], rhs.__bits[i]);
        }
    }
};

TEST_F(TestOckHmmSingleDeviceMgrExt, base_info_query)
{
    ExpectCpuSetEQ(mgr->GetCpuSet(), this->cpuSet);
    EXPECT_EQ(mgr->GetSpecific(), this->deviceInfo->memorySpec);
}
TEST_F(TestOckHmmSingleDeviceMgrExt, get_used_info)
{
    const uint64_t fragThreshold = 2ULL * 1024ULL * 1024ULL;
    MockGetUsedInfo(*hostSwapAlloc, fragThreshold, hostSwapUsedBytes, hostSwapUnusedFragBytes, hostSwapLeftBytes);
    MockGetUsedInfo(*hostDataAlloc, fragThreshold, hostDataUsedBytes, hostDataUnusedFragBytes, hostDataLeftBytes);
    MockGetUsedInfo(*devSwapAlloc, fragThreshold, devSwapUsedBytes, devSwapUnusedFragBytes, devSwapLeftBytes);
    MockGetUsedInfo(*devDataAlloc, fragThreshold, devDataUsedBytes, devDataUnusedFragBytes, devDataLeftBytes);

    auto usedInfo = mgr->GetUsedInfo(fragThreshold);
    EXPECT_EQ(usedInfo->devUsedInfo.swapUsedBytes, devSwapUsedBytes);
    EXPECT_EQ(usedInfo->devUsedInfo.swapLeftBytes, devSwapLeftBytes);
    EXPECT_EQ(usedInfo->devUsedInfo.usedBytes, devDataUsedBytes);
    EXPECT_EQ(usedInfo->devUsedInfo.unusedFragBytes, devDataUnusedFragBytes);
    EXPECT_EQ(usedInfo->devUsedInfo.leftBytes, devDataLeftBytes);

    EXPECT_EQ(usedInfo->hostUsedInfo.swapUsedBytes, hostSwapUsedBytes);
    EXPECT_EQ(usedInfo->hostUsedInfo.swapLeftBytes, hostSwapLeftBytes);
    EXPECT_EQ(usedInfo->hostUsedInfo.usedBytes, hostDataUsedBytes);
    EXPECT_EQ(usedInfo->hostUsedInfo.unusedFragBytes, hostDataUnusedFragBytes);
    EXPECT_EQ(usedInfo->hostUsedInfo.leftBytes, hostDataLeftBytes);
}
TEST_F(TestOckHmmSingleDeviceMgrExt, alloc_free)
{
    const uint32_t hmoBytes = 1024U * 1024U * 64;
    this->MockAllocFreeWithNewDelete(*devDataAlloc);
    this->MockAllocFreeWithNewDelete(*hostDataAlloc);
    auto policyLocationMap = BuildPolicyLocationMap();

    for (auto policyLocation : policyLocationMap) {
        auto ret = mgr->Alloc(hmoBytes, policyLocation.first);
        ASSERT_EQ(ret.first, HMM_SUCCESS);
        EXPECT_EQ(ret.second->GetByteSize(), hmoBytes);
        EXPECT_EQ(ret.second->Location(), policyLocation.second);
        EXPECT_EQ(OckHmmHMOObjectIDGenerator::ParseDeviceId(ret.second->GetId()), this->deviceInfo->deviceId);
        mgr->Free(ret.second);
    }
}
TEST_F(TestOckHmmSingleDeviceMgrExt, alloc_exceed_number)
{
    const uint32_t hmoBytes = 1024U * 1024U * 64U;
    this->MockAllocFreeWithNewDelete(*devDataAlloc);
    conf::OckHmmConf hmmConf;
    hmmConf.maxHMOCountPerDevice = 0;
    MOCKER(&conf::OckSysConf::HmmConf).expects(exactly(3U)).will(returnValue(hmmConf));
    auto ret = mgr->Alloc(hmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
    ASSERT_EQ(ret.first, HMM_ERROR_HMO_OBJECT_NUM_EXCEED);
}
TEST_F(TestOckHmmSingleDeviceMgrExt, alloc_failed_while_dev_space_not_enough)
{
    const uint32_t hmoBytes = 1024U * 1024U * 64U;
    EXPECT_CALL(*devDataAlloc, Alloc(testing::_)).WillRepeatedly(testing::Return(0UL));
    auto ret = mgr->Alloc(hmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
    ASSERT_EQ(ret.first, HMM_ERROR_DEVICE_DATA_SPACE_NOT_ENOUGH);
}
TEST_F(TestOckHmmSingleDeviceMgrExt, alloc_failed_while_host_space_not_enough)
{
    const uint32_t hmoBytes = 1024U * 1024U * 64U;
    EXPECT_CALL(*hostDataAlloc, Alloc(testing::_)).WillRepeatedly(testing::Return(0UL));
    auto ret = mgr->Alloc(hmoBytes, OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY);
    ASSERT_EQ(ret.first, HMM_ERROR_HOST_DATA_SPACE_NOT_ENOUGH);
}
TEST_F(TestOckHmmSingleDeviceMgrExt, malloc_multi_location)
{
    const uint32_t hmoBytes = 1024U * 1024U * 64U;
    this->MockAllocFreeWithNewDelete(*devDataAlloc);
    this->MockAllocFreeWithNewDelete(*hostDataAlloc);
    auto policyLocationMap = BuildPolicyLocationMap();

    for (auto policyLocation : policyLocationMap) {
        auto ret = mgr->Malloc(hmoBytes, policyLocation.first);
        ASSERT_TRUE(ret.get() != nullptr);
        EXPECT_EQ(ret->ByteSize(), hmoBytes);
        EXPECT_EQ(ret->Location(), policyLocation.second);
        EXPECT_NE(ret->Addr(), 0U);
    }
}
}  // namespace test
}  // namespace hmm
}  // namespace ock
