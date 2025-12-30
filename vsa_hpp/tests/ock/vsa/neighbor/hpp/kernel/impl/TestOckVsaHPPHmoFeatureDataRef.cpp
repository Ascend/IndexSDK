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

#include <gtest/gtest.h>
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPHmoFeatureDataRef.h"
#include "ock/acladapter/WithEnvAclMock.h"
#include "ock/hmm/OckHmmFactory.h"
namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
namespace impl {
namespace test {
class TestOckVsaHPPHmoFeatureDataRef : public acladapter::WithEnvAclMock<testing::Test> {
public:
    using BaseT = acladapter::WithEnvAclMock<testing::Test>;
    void SetUp(void) override
    {
        BaseT::SetUp();
        singleDeviceInfo = std::make_shared<hmm::OckHmmDeviceInfo>();
        singleDeviceInfo->deviceId = 0U;
        CPU_SET(1U, &singleDeviceInfo->cpuSet);                                                      // 设置1号CPU核
        CPU_SET(2U, &singleDeviceInfo->cpuSet);                                                      // 设置2号CPU核
        singleDeviceInfo->memorySpec.devSpec.maxDataCapacity = 2ULL * 1024ULL * 1024ULL * 1024ULL;   // 2G
        singleDeviceInfo->memorySpec.devSpec.maxSwapCapacity = 3ULL * 64ULL * 1024ULL * 1024ULL;     // 3 * 64M
        singleDeviceInfo->memorySpec.hostSpec.maxDataCapacity = 2ULL * 1024ULL * 1024ULL * 1024ULL;  // 2G
        singleDeviceInfo->memorySpec.hostSpec.maxSwapCapacity = 3ULL * 64ULL * 1024ULL * 1024ULL;    // 3 * 64M
        singleDeviceInfo->transferThreadNum = 2ULL;                                                  // 2个线程
    }

    void TearDown(void) override
    {
        DestroyHmmDeviceMgr();  // 需要提前reset，否则打桩不生效。
        BaseT::TearDown();
    }

    void BuildSingleMgr(void)
    {
        auto factory = hmm::OckHmmFactory::Create();
        auto ret = factory->CreateSingleDeviceMemoryMgr(singleDeviceInfo);
        singleMgr = ret.second;
    }

    void DestroyHmmDeviceMgr(void)
    {
        if (singleMgr.get() != nullptr) {
            singleMgr.reset();
        }
    }

    std::shared_ptr<hmm::OckHmmDeviceInfo> singleDeviceInfo;
    std::shared_ptr<hmm::OckHmmSingleDeviceMgr> singleMgr;
    uint64_t hmoByte = 64ULL * 1024ULL * 1024ULL;
    uint32_t containerSize = 5UL;
};

TEST_F(TestOckVsaHPPHmoFeatureDataRef, createOckVsaHPPHmoFeatureDataRef)
{
    BuildSingleMgr();
    auto devHmoRet = singleMgr->Alloc(hmoByte);
    EXPECT_EQ(devHmoRet.first, hmm::HMM_SUCCESS);
    auto srcData = devHmoRet.second;

    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> unusedContainer;
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> usedContainer;
    for (uint32_t i = 0; i < containerSize; ++i) {
        auto containerHmoRet = singleMgr->Alloc(hmoByte);
        EXPECT_EQ(containerHmoRet.first, hmm::HMM_SUCCESS);
        unusedContainer.emplace_back(containerHmoRet.second);
        usedContainer.emplace_back(containerHmoRet.second);
    }

    std::shared_ptr<OckVsaHPPHmoFeatureDataRef> dataRef =
        std::make_shared<OckVsaHPPHmoFeatureDataRef>(*srcData, unusedContainer, usedContainer);
    EXPECT_NE(dataRef, nullptr);
}

TEST_F(TestOckVsaHPPHmoFeatureDataRef, popUnusedToUsed)
{
    BuildSingleMgr();
    auto devHmoRet = singleMgr->Alloc(hmoByte);
    EXPECT_EQ(devHmoRet.first, hmm::HMM_SUCCESS);
    auto srcData = devHmoRet.second;

    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> unusedContainer;
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> usedContainer;
    for (uint32_t i = 0; i < containerSize; ++i) {
        auto containerHmoRet = singleMgr->Alloc(hmoByte);
        EXPECT_EQ(containerHmoRet.first, hmm::HMM_SUCCESS);
        unusedContainer.emplace_back(containerHmoRet.second);
        usedContainer.emplace_back(containerHmoRet.second);
    }

    std::shared_ptr<OckVsaHPPHmoFeatureDataRef> dataRef =
            std::make_shared<OckVsaHPPHmoFeatureDataRef>(*srcData, unusedContainer, usedContainer);
    EXPECT_NE(dataRef, nullptr);
    EXPECT_EQ(dataRef->usedContainer.size(), containerSize);
    EXPECT_EQ(dataRef->unusedContainer.size(), containerSize);
    dataRef->PopUnusedToUsed();
    EXPECT_EQ(dataRef->usedContainer.size(), containerSize + 1UL);
    EXPECT_EQ(dataRef->unusedContainer.size(), containerSize - 1UL);
}

TEST_F(TestOckVsaHPPHmoFeatureDataRef, pickUnused)
{
    BuildSingleMgr();
    auto devHmoRet = singleMgr->Alloc(hmoByte);
    EXPECT_EQ(devHmoRet.first, hmm::HMM_SUCCESS);
    auto srcData = devHmoRet.second;

    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> unusedContainer;
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> usedContainer;
    for (uint32_t i = 0; i < containerSize; ++i) {
        auto containerHmoRet = singleMgr->Alloc(hmoByte);
        EXPECT_EQ(containerHmoRet.first, hmm::HMM_SUCCESS);
        unusedContainer.emplace_back(containerHmoRet.second);
        usedContainer.emplace_back(containerHmoRet.second);
    }

    std::shared_ptr<OckVsaHPPHmoFeatureDataRef> dataRef =
            std::make_shared<OckVsaHPPHmoFeatureDataRef>(*srcData, unusedContainer, usedContainer);
    EXPECT_NE(dataRef, nullptr);
    EXPECT_EQ(dataRef->usedContainer.size(), containerSize);
    EXPECT_EQ(dataRef->unusedContainer.size(), containerSize);
    auto usedData = dataRef->PickUnused();
    EXPECT_NE(usedData, nullptr);
}
}
}
}
}
}
}
