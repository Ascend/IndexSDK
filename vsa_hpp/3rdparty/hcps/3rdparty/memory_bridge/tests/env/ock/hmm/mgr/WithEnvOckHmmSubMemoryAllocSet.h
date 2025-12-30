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


#ifndef OCK_MEMORY_BRIDGE_OCK_ADAPTER_TASK_WITH_ENV_OCK_HMM_SUB_MEMORY_ALLOC_SET_H
#define OCK_MEMORY_BRIDGE_OCK_ADAPTER_TASK_WITH_ENV_OCK_HMM_SUB_MEMORY_ALLOC_SET_H
#include <thread>
#include <chrono>
#include "ock/hmm/mgr/MockOckHmmSubMemoryAlloc.h"
#include "ock/hmm/mgr/OckHmmSingleDeviceMgrExt.h"
namespace ock {
namespace hmm {

template <typename BaseT>
class WithEnvOckHmmSubMemoryAllocSet : public BaseT {
public:
    void SetUp(void) override
    {
        BaseT::SetUp();
        hostSwapAlloc = new MockOckHmmSubMemoryAlloc();
        hostDataAlloc = new MockOckHmmSubMemoryAlloc();
        devSwapAlloc = new MockOckHmmSubMemoryAlloc();
        devDataAlloc = new MockOckHmmSubMemoryAlloc();
        allocSet = std::make_shared<OckHmmSubMemoryAllocSet>(std::shared_ptr<OckHmmSubMemoryAlloc>(devDataAlloc),
            std::shared_ptr<OckHmmSubMemoryAlloc>(devSwapAlloc),
            std::shared_ptr<OckHmmSubMemoryAlloc>(hostDataAlloc),
            std::shared_ptr<OckHmmSubMemoryAlloc>(hostSwapAlloc));
        EXPECT_CALL(*hostDataAlloc, Location())
            .WillRepeatedly(testing::Return(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY));
        EXPECT_CALL(*hostSwapAlloc, Location())
            .WillRepeatedly(testing::Return(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY));
        EXPECT_CALL(*devSwapAlloc, Location()).WillRepeatedly(testing::Return(OckHmmHeteroMemoryLocation::DEVICE_DDR));
        EXPECT_CALL(*devDataAlloc, Location()).WillRepeatedly(testing::Return(OckHmmHeteroMemoryLocation::DEVICE_DDR));

        devDataUsedBytes = {3};          // 分配质数，测试容易发现功能问题
        devDataUnusedFragBytes = {5};    // 分配质数，测试容易发现功能问题
        devDataLeftBytes = {7};          // 分配质数，测试容易发现功能问题
        devSwapUsedBytes = {11};         // 分配质数，测试容易发现功能问题
        devSwapUnusedFragBytes = {13};   // 分配质数，测试容易发现功能问题
        devSwapLeftBytes = {17};         // 分配质数，测试容易发现功能问题
        hostDataUsedBytes = {19};        // 分配质数，测试容易发现功能问题
        hostDataUnusedFragBytes = {23};  // 分配质数，测试容易发现功能问题
        hostDataLeftBytes = {29};        // 分配质数，测试容易发现功能问题
        hostSwapUsedBytes = {31};        // 分配质数，测试容易发现功能问题
        hostSwapUnusedFragBytes = {37};  // 分配质数，测试容易发现功能问题
        hostSwapLeftBytes = {41};        // 分配质数，测试容易发现功能问题

        allocSleepTimeMilliSecond = {100};  // Alloc时等待时间，单位ms
    }
    void TearDown(void) override
    {
        BaseT::TearDown();
    }
    void MockAllocFreeWithNewDelete(MockOckHmmSubMemoryAlloc &alloc)
    {
        EXPECT_CALL(alloc, Alloc(testing::_)).WillRepeatedly(testing::Invoke([](uint64_t byteSize) {
            return uintptr_t(new uint8_t[byteSize]);
        }));
        EXPECT_CALL(alloc, Alloc(testing::_, testing::_))
            .WillRepeatedly(testing::Invoke([](uint64_t byteSize, const acladapter::OckUserWaitInfoBase &waitInfo) {
                return uintptr_t(new uint8_t[byteSize]);
            }));
        EXPECT_CALL(alloc, Free(testing::_, testing::_))
            .WillRepeatedly(testing::Invoke([](uintptr_t addr, uint64_t byteSize) { delete[] (uint8_t *)addr; }));
    }
    void MockAllocFreeWithNewDeleteTimeOut(MockOckHmmSubMemoryAlloc &alloc)
    {
        EXPECT_CALL(alloc, Alloc(testing::_, testing::_))
            .WillRepeatedly(testing::Invoke([this](uint64_t byteSize, const acladapter::OckUserWaitInfoBase &waitInfo) {
                std::this_thread::sleep_for(std::chrono::milliseconds(this->allocSleepTimeMilliSecond));
                if (waitInfo.WaitTimeOut()) {
                    return 0UL;
                }
                return uintptr_t(new uint8_t[byteSize]);
            }));
        EXPECT_CALL(alloc, Free(testing::_, testing::_))
            .WillRepeatedly(testing::Invoke([](uintptr_t addr, uint64_t byteSize) { delete[] (uint8_t *)addr; }));
    }
    void MockGetUsedInfo(MockOckHmmSubMemoryAlloc &alloc, const uint64_t fragThreshold, uint64_t usedBytes,
        uint64_t unusedFragBytes, uint64_t leftBytes)
    {
        EXPECT_CALL(alloc, GetUsedInfo(fragThreshold))
            .WillRepeatedly(testing::Invoke([usedBytes, unusedFragBytes, leftBytes](uint64_t fragThreshold) {
                return std::make_shared<OckHmmMemoryUsedInfoLocal>(usedBytes, unusedFragBytes, leftBytes);
            }));
    }
    uint64_t devDataUsedBytes;
    uint64_t devDataUnusedFragBytes;
    uint64_t devDataLeftBytes;
    uint64_t devSwapUsedBytes;
    uint64_t devSwapUnusedFragBytes;
    uint64_t devSwapLeftBytes;
    uint64_t hostDataUsedBytes;
    uint64_t hostDataUnusedFragBytes;
    uint64_t hostDataLeftBytes;
    uint64_t hostSwapUsedBytes;
    uint64_t hostSwapUnusedFragBytes;
    uint64_t hostSwapLeftBytes;
    uint64_t allocSleepTimeMilliSecond;
    MockOckHmmSubMemoryAlloc *hostSwapAlloc;
    MockOckHmmSubMemoryAlloc *hostDataAlloc;
    MockOckHmmSubMemoryAlloc *devSwapAlloc;
    MockOckHmmSubMemoryAlloc *devDataAlloc;
    std::shared_ptr<OckHmmSubMemoryAllocSet> allocSet;
};

}  // namespace hmm
}  // namespace ock
#endif