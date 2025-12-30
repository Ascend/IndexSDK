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

#include <ostream>
#include <cstdint>
#include <gtest/gtest.h>
#include <chrono>
#include <unordered_map>
#include <ctime>
#include "ptest/ptest.h"
#include "gmock/gmock.h"
#include "ock/acladapter/utils/MockOckAdapterMemoryGuard.h"
#include "ock/acladapter/executor/MockOckUserWaitInfoBase.h"
#include "ock/hmm/mgr/OckHmmSubMemoryAlloc.h"

namespace ock {
namespace hmm {
namespace test {

class PTestPerfSubMemoryAlloc : public testing::Test {
public:
    PTestPerfSubMemoryAlloc(void)
        : memoryByteSize(10000ULL * 1024ULL * 1024ULL) /* 10000M */,
          onePercentByte(100ULL * 1024ULL * 1024ULL) /* 100M */,
          memoryAddr(37U) /* 一个任意数，这里选择了一个素数 */,
          loopTimes(10000U) /* 性能测试的迭代轮数 */
    {
        mockWaitInfo = new acladapter::MockOckUserWaitInfoBase();
        waitInfo = std::shared_ptr<acladapter::OckUserWaitInfoBase>(mockWaitInfo);
    }

    void BuildMemAlloc(void)
    {
        auto mockGuard = new acladapter::MockOckAdapterMemoryGuard();
        EXPECT_CALL(*mockGuard, ByteSize()).WillRepeatedly(testing::Return(memoryByteSize));
        EXPECT_CALL(*mockGuard, Addr()).WillRepeatedly(testing::Return(memoryAddr));
        EXPECT_CALL(*mockGuard, Location()).WillRepeatedly(testing::Return(location));
        memAlloc = OckHmmSubMemoryAlloc::Create(std::unique_ptr<acladapter::OckAdapterMemoryGuard>(mockGuard),
            memoryByteSize);
    }

    uint64_t memoryByteSize;
    uint64_t onePercentByte;
    uintptr_t memoryAddr;
    uint32_t loopTimes;

    OckHmmHeteroMemoryLocation location;
    std::shared_ptr<OckHmmSubMemoryAlloc> memAlloc;
    acladapter::MockOckUserWaitInfoBase *mockWaitInfo;
    std::shared_ptr<acladapter::OckUserWaitInfoBase> waitInfo;

    std::list<std::pair<uintptr_t, uint64_t>> allocated;
    std::vector<int> allocOrFree;
    std::vector<uint64_t> randomLen;
    std::vector<size_t> randomIndex;

    void BuildHighLoadScene(uint32_t percentage)
    {
        srand(static_cast<uint64_t>(time(nullptr)));
        for (uint32_t i = 0; i < loopTimes; ++i) {
            allocOrFree.push_back(rand() % 2U);  // 0 or 1
            randomLen.push_back(((rand() % 100U) + 1U) * 1024ULL * 1024ULL);  // 1M~100M
            randomIndex.push_back(rand());
        }
        BuildMemAlloc();
        for (uint32_t i = 0; i < percentage; ++i) {
            memAlloc->Alloc(onePercentByte);
        }
    }

    void TryRandomAlloc(uint32_t i)
    {
        auto addr = memAlloc->Alloc(randomLen[i]);
        if (addr != 0) {
            allocated.push_back(std::make_pair(addr, randomLen[i]));
        } else {
            TryRandomFree(i);
        }
    }

    void TryRandomFree(uint32_t i)
    {
        if (!allocated.empty()) {
            auto iter = allocated.begin();
            int random = randomIndex[i] % allocated.size();
            std::advance(iter, random);
            memAlloc->Free(iter->first, iter->second);
            allocated.erase(iter);
        } else {
            TryRandomAlloc(i);
        }
    }

    void DoRandomAllocAndFree()
    {
        for (uint32_t i = 0; i < loopTimes; ++i) {
            if (allocOrFree[i] == 0) {
                TryRandomAlloc(i);
            } else {
                TryRandomFree(i);
            }
        }
    }
};

// 1%左右空间负载下10000组内存二次随机分配/释放耗时
TEST_F(PTestPerfSubMemoryAlloc, performance_test_1_percent_overload_alloc)
{
    BuildHighLoadScene(1U);
    auto timeGuard = fast::hdt::TestTimeGuard();
    DoRandomAllocAndFree();
    EXPECT_TRUE(FAST_PTEST().Test(
        "OCK.MemoryBridge.HMM.SubMem.LowUseRateRandAllocFreePair", "AllocAndFreeTime",
        timeGuard.ElapsedMicroSeconds()));
}

// 50%左右空间负载下10000组内存二次随机分配/释放耗时
TEST_F(PTestPerfSubMemoryAlloc, performance_test_50_percent_overload_alloc)
{
    BuildHighLoadScene(50U);
    auto timeGuard = fast::hdt::TestTimeGuard();
    DoRandomAllocAndFree();
    EXPECT_TRUE(FAST_PTEST().Test(
        "OCK.MemoryBridge.HMM.SubMem.HalfUseRateRandAllocFreePair", "AllocAndFreeTime",
        timeGuard.ElapsedMicroSeconds()));
}

// 99%左右空间负载下10000组内存二次随机分配/释放耗时
TEST_F(PTestPerfSubMemoryAlloc, performance_test_99_percent_overload_alloc)
{
    BuildHighLoadScene(99U);
    auto timeGuard = fast::hdt::TestTimeGuard();
    DoRandomAllocAndFree();
    EXPECT_TRUE(FAST_PTEST().Test(
        "OCK.MemoryBridge.HMM.SubMem.HighUseRateRandAllocFreePair", "AllocAndFreeTime",
        timeGuard.ElapsedMicroSeconds()));
}
}  // namespace test
}  // namespace utils
}  // namespace ock