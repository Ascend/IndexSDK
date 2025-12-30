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


#include <vector>
#include "gtest/gtest.h"
#include "ock/hmm/helper/OckHmmAllocator.h"
#include "ock/hmm/OckHmmFactory.h"
#include "acl/acl.h"
#include "ock/acladapter/WithEnvAclMock.h"

namespace ock {
namespace hmm {
namespace helper {
namespace test {

class TestOckHmmAllocator : public acladapter::WithEnvAclMock<testing::Test> {
public:
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

    template <typename T> using VecTest = std::vector<T, OckHmmAllocator<T>>;
    uint64_t minFragThreshold = 2ULL * 1024ULL * 1024ULL;
    std::shared_ptr<OckHmmDeviceInfo> deviceInfo;
    std::shared_ptr<OckHmmSingleDeviceMgr> devMgr;
};

TEST_F(TestOckHmmAllocator, create_vector)
{
    BuildSingleDeviceMgr();
    uint64_t minFragThreshold = 2ULL * 1024ULL * 1024ULL;
    uint32_t vecCapacity = 5000ULL;
    uint64_t testNum = 100000;
    VecTest<uint32_t> testVec{ OckHmmAllocator<uint32_t>(*devMgr.get()) };

    testVec.emplace_back(testNum);
    EXPECT_EQ(testVec[0], testNum);

    auto usedInfo = devMgr->GetUsedInfo(minFragThreshold);
    EXPECT_EQ(usedInfo->hostUsedInfo.usedBytes, sizeof(uint32_t));

    testVec.reserve(vecCapacity);
    usedInfo = devMgr->GetUsedInfo(minFragThreshold);
    EXPECT_EQ(usedInfo->hostUsedInfo.usedBytes, vecCapacity * sizeof(uint32_t));

    VecTest<uint32_t>(*devMgr.get()).swap(testVec);
    usedInfo = devMgr->GetUsedInfo(minFragThreshold);
    EXPECT_EQ(usedInfo->hostUsedInfo.usedBytes, 0ULL);
}
}
}
}
}