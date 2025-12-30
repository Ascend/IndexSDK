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
#include <chrono>
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "ock/acladapter/utils/MockOckAdapterMemoryGuard.h"
#include "ock/acladapter/executor/MockOckUserWaitInfoBase.h"

#include "ock/hmm/mgr/OckHmmSubMemoryAlloc.h"
#include "ock/hmm/mgr/OckHmmMgrCreator.h"
namespace ock {
namespace hmm {
namespace test {
class TestOckHmmSubMemoryAlloc : public testing::Test {
public:
    TestOckHmmSubMemoryAlloc(void)
        : memoryByteSize(4ULL * 1024U * 1024U * 1024U) /* 4G = 4294967296 */,
          memoryByteSizeNotAligned(11ULL * 1024ULL * 1024ULL + 2ULL * 1024ULL + 277ULL),
          blockByteSize(64ULL * 1024U * 1024U) /* 64M */,
          nonBlockByteSize(63ULL * 512U * 1024U) /* 31M */,
          wastedFragByteSize(1ULL * 512U * 1024U) /* 1M */,
          byte256B(256U) /* 256字节, 字节级别的block的大小 */,
          byte257B(257U) /* 257字节, 字节级别的block的大小 */,
          byte2K(2ULL * 1024U) /* 2K, K级别的block的大小 */,
          byte2K2B(2ULL * 1024U + 2U) /* 2K + 2字节, K级别的block的大小 */,
          byte5M(5ULL * 1024U * 1024U) /* 5M, M级别的block的大小 */,
          byte5M2B(5ULL * 1024U * 1024U + 2U) /* 5M + 2字节, M级别的block的大小 */,
          threshold(5ULL * 1024U * 1024U) /* 大小block组合测试的碎片阈值 */,
          memoryAddr(37U) /* 一个任意数，这里选择了一个素数 */,
          incMemoryAddr(137U + 4ULL * 1024U * 1024U * 1024U) /* 新增内存的首地址 4294967433 */,
          incMemoryByteSize(32ULL * 1024U * 1024U) /* 新增内存的大小 32M */
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
        memAlloc =
            OckHmmSubMemoryAlloc::Create(std::unique_ptr<acladapter::OckAdapterMemoryGuard>(mockGuard), memoryByteSize);
    }

    void BuildMemAllocWithName(std::string name)
    {
        auto mockGuard = new acladapter::MockOckAdapterMemoryGuard();
        EXPECT_CALL(*mockGuard, ByteSize()).WillRepeatedly(testing::Return(memoryByteSize));
        EXPECT_CALL(*mockGuard, Addr()).WillRepeatedly(testing::Return(memoryAddr));
        EXPECT_CALL(*mockGuard, Location()).WillRepeatedly(testing::Return(location));
        memAlloc = OckHmmSubMemoryAlloc::Create(std::unique_ptr<acladapter::OckAdapterMemoryGuard>(mockGuard),
            memoryByteSize, name);
    }

    acladapter::OckAdapterMemoryGuard *BuildOckAdapterMemoryGuard(uint64_t needMemoryByteSize, uintptr_t needMemoryAddr,
        OckHmmHeteroMemoryLocation needLocation)
    {
        auto mockGuard = new acladapter::MockOckAdapterMemoryGuard();
        EXPECT_CALL(*mockGuard, ByteSize()).WillRepeatedly(testing::Return(needMemoryByteSize));
        EXPECT_CALL(*mockGuard, Addr()).WillRepeatedly(testing::Return(needMemoryAddr));
        EXPECT_CALL(*mockGuard, Location()).WillRepeatedly(testing::Return(needLocation));
        return mockGuard;
    }

    void CompareUsedInfo(std::shared_ptr<OckHmmMemoryUsedInfoLocal> usedInfo, uint64_t usedByteSize,
        uint64_t unusedFragByteSize, uint64_t leftByteSize)
    {
        EXPECT_EQ(usedInfo->usedBytes, usedByteSize);
        EXPECT_EQ(usedInfo->unusedFragBytes, unusedFragByteSize);
        EXPECT_EQ(usedInfo->leftBytes, leftByteSize);
    }

    uint64_t memoryByteSize;
    uint64_t memoryByteSizeNotAligned;
    uint64_t blockByteSize;
    uint64_t nonBlockByteSize;
    uint64_t wastedFragByteSize;
    uint64_t byte256B;
    uint64_t byte257B;
    uint64_t byte2K;
    uint64_t byte2K2B;
    uint64_t byte5M;
    uint64_t byte5M2B;
    uint64_t threshold;
    uintptr_t memoryAddr;
    uintptr_t incMemoryAddr;
    uint64_t incMemoryByteSize;
    OckHmmHeteroMemoryLocation location;
    std::shared_ptr<OckHmmSubMemoryAlloc> memAlloc;
    acladapter::MockOckUserWaitInfoBase *mockWaitInfo;
    std::shared_ptr<acladapter::OckUserWaitInfoBase> waitInfo;
};

TEST_F(TestOckHmmSubMemoryAlloc, location)
{
    BuildMemAlloc();
    EXPECT_EQ(memAlloc->Location(), location);
}

TEST_F(TestOckHmmSubMemoryAlloc, Alloc)
{
    BuildMemAlloc();

    auto addr = memAlloc->Alloc(blockByteSize);
    EXPECT_EQ(addr, memoryAddr);

    addr = memAlloc->Alloc(blockByteSize);
    EXPECT_EQ(addr, memoryAddr + blockByteSize);
}

TEST_F(TestOckHmmSubMemoryAlloc, AllocBeyondAlignedRange)
{
    auto mockGuard = new acladapter::MockOckAdapterMemoryGuard();
    EXPECT_CALL(*mockGuard, ByteSize()).WillRepeatedly(testing::Return(memoryByteSizeNotAligned));
    EXPECT_CALL(*mockGuard, Addr()).WillRepeatedly(testing::Return(memoryAddr));
    EXPECT_CALL(*mockGuard, Location()).WillRepeatedly(testing::Return(location));
    memAlloc = OckHmmSubMemoryAlloc::Create(std::unique_ptr<acladapter::OckAdapterMemoryGuard>(mockGuard),
                                            memoryByteSizeNotAligned);
    auto info = memAlloc->GetUsedInfo(threshold);
    EXPECT_EQ(info->unusedFragBytes, 21U);
    EXPECT_EQ(info->leftBytes, memoryByteSizeNotAligned);
    EXPECT_EQ(info->usedBytes, 0ULL);

    auto addr = memAlloc->Alloc(memoryByteSizeNotAligned);
    EXPECT_EQ(addr, 0ULL);
}

TEST_F(TestOckHmmSubMemoryAlloc, AllocWithWaitInfoTrue)
{
    BuildMemAlloc();
    EXPECT_CALL(*mockWaitInfo, WaitTimeOut()).WillRepeatedly(testing::Return(true));
    auto addr = memAlloc->Alloc(blockByteSize, *waitInfo);
    EXPECT_EQ(addr, 0UL);
}

TEST_F(TestOckHmmSubMemoryAlloc, AllocWithWaitInfoFalse)
{
    BuildMemAlloc();
    EXPECT_CALL(*mockWaitInfo, WaitTimeOut()).WillRepeatedly(testing::Return(false));
    auto addr = memAlloc->Alloc(blockByteSize, *waitInfo);
    EXPECT_EQ(addr, memoryAddr);
}

TEST_F(TestOckHmmSubMemoryAlloc, Free)
{
    BuildMemAlloc();

    auto addr = memAlloc->Alloc(blockByteSize);
    EXPECT_EQ(addr, memoryAddr);

    memAlloc->Free(addr, blockByteSize);
    addr = memAlloc->Alloc(blockByteSize);
    EXPECT_EQ(addr, memoryAddr);
}

TEST_F(TestOckHmmSubMemoryAlloc, GetUsedInfo)
{
    BuildMemAlloc();
    memAlloc->Alloc(blockByteSize);
    auto addr = memAlloc->Alloc(blockByteSize);
    memAlloc->Alloc(nonBlockByteSize);
    memAlloc->Free(addr, blockByteSize);
    auto info = memAlloc->GetUsedInfo(128ULL * 1024U * 1024U);

    EXPECT_EQ(info->usedBytes, blockByteSize + nonBlockByteSize);
    EXPECT_EQ(info->unusedFragBytes, blockByteSize + wastedFragByteSize);
    EXPECT_EQ(info->leftBytes, memoryByteSize - blockByteSize - nonBlockByteSize);
}

TEST_F(TestOckHmmSubMemoryAlloc, Name)
{
    std::string name = "DEVICE_DATA";
    BuildMemAllocWithName(name);
    EXPECT_EQ(memAlloc->Name(), "DEVICE_DATA");
}

// 测试场景：[2K+2B][5M][2K][256B]<5M>[257B][5M2B]<5M2B><257B><2K+2B><256B><2K>
// 其中，[]表示Alloc, <>表示Free,B,K,M表示对齐的字节级、KB级、M级内存块，B1,K1,M1表示非对齐的
TEST_F(TestOckHmmSubMemoryAlloc, alloc_big_and_small_blocks)
{
    BuildMemAlloc();
    std::shared_ptr<OckHmmMemoryUsedInfoLocal> info;

    // pool: [total] -> <3K空洞> [total-3K]
    auto addrK = memAlloc->Alloc(byte2K2B);
    info = memAlloc->GetUsedInfo(threshold);
    EXPECT_EQ(info->unusedFragBytes, 1022U); // 1022: 对齐引起的碎片, 1K-2B

    // pool: <3K空洞> [total-3K] -> <3K+5M+2K+256B空洞> [total-256B-2K-5M-3K]
    auto addrMAligned = memAlloc->Alloc(byte5M);
    auto addrKAligned = memAlloc->Alloc(byte2K);
    auto addrBAligned = memAlloc->Alloc(byte256B);
    // pool: <3K+5M+2K+256B空洞> [total-256B-2K-5M-3K] -> <3K空洞> [5M] <2K+256B空洞> [total-256B-2K-5M-3K]
    memAlloc->Free(addrMAligned, byte5M);
    info = memAlloc->GetUsedInfo(threshold);
    EXPECT_EQ(info->unusedFragBytes, 1022U); // 1022: 归还5M不会引起小于5M的碎片

    //  <3K空洞> [5M] <2K+256B空洞> [total-256B-2K-5M-3K] ->  <3K+512k空洞> [5M-512] <2K+256B空洞> [total-256B-2K-5M-3K]
    auto addrB = memAlloc->Alloc(byte257B); // 底层会从归还的5M中划走512B，剩余大小<5M，引起了小于阈值的碎片
    info = memAlloc->GetUsedInfo(threshold);
    EXPECT_EQ(info->unusedFragBytes,
        1022U + 255U + 5ULL * 1024U * 1024U - 512U); // 对齐引起新增碎片 255B；小于阈值的碎片 5M-512B

    // <3K+512k空洞> [5M-512] <2K+256B空洞> [total-256B-2K-5M-3K] ->
    // <3K+512k空洞> [5M-512] <2K+256B+6M空洞> [total-6M-256B-2K-5M-3K]
    auto addrM = memAlloc->Alloc(byte5M2B); // 小空间够，从最大那块划分
    info = memAlloc->GetUsedInfo(threshold);
    EXPECT_EQ(info->unusedFragBytes,
        1022U + 255U + 5ULL * 1024U * 1024U - 512U + 1ULL * 1024U * 1024U - 2U); // 对齐引起碎片 1M-2B

    // <3K+512k空洞> [5M-512] <2K+256B+6M空洞> [total-6M-256B-2K-5M-3K] ->
    // <3K+512k空洞> [5M-512] <2K+256B空洞> [total-256B-2K-5M-3K]
    memAlloc->Free(addrM, byte5M2B);
    info = memAlloc->GetUsedInfo(threshold);
    EXPECT_EQ(info->unusedFragBytes, 1022U + 255U + 5ULL * 1024U * 1024U - 512U); // 归还对齐碎片 1M-2B

    // <3K+512k空洞> [5M-512] <2K+256B空洞> [total-256B-2K-5M-3K] -> <3K空洞> [5M] <2K+256B空洞> [total-256B-2K-5M-3K]
    memAlloc->Free(addrB, byte257B);
    info = memAlloc->GetUsedInfo(threshold);
    EXPECT_EQ(info->unusedFragBytes, 1022U); // 归还对齐引起的碎片和不足5M的碎片

    // <3K空洞> [5M] <2K+256B空洞> [total-256B-2K-5M-3K] -> [5M+3K] <2K+256B空洞> [total-256B-2K-5M-3K]
    memAlloc->Free(addrK, byte2K2B);
    info = memAlloc->GetUsedInfo(threshold);
    EXPECT_EQ(info->unusedFragBytes, 0U); // 归还对齐引起的碎片

    // [5M+3K] <2K+256B空洞> [total-256B-2K-5M-3K] -> [5M+3K] <2K空洞> [total-2K-5M-3K]
    memAlloc->Free(addrBAligned, byte256B);
    // [5M+3K] <2K空洞> [total-2K-5M-3K] -> [total]
    memAlloc->Free(addrKAligned, byte2K);
    info = memAlloc->GetUsedInfo(threshold);
    EXPECT_EQ(info->unusedFragBytes, 0U);
    EXPECT_EQ(info->leftBytes, memoryByteSize);
}
TEST_F(TestOckHmmSubMemoryAlloc, IncBindMemory)
{
    BuildMemAlloc();

    auto incMockGuard = new acladapter::MockOckAdapterMemoryGuard();
    EXPECT_CALL(*incMockGuard, ByteSize()).WillRepeatedly(testing::Return(incMemoryByteSize));
    EXPECT_CALL(*incMockGuard, Addr()).WillRepeatedly(testing::Return(incMemoryAddr));
    EXPECT_CALL(*incMockGuard, Location()).WillRepeatedly(testing::Return(location));
    OCK_HMM_LOG_DEBUG("incMockGuard succeed !");

    memAlloc->IncBindMemoryToMemPool(std::unique_ptr<acladapter::OckAdapterMemoryGuard>(incMockGuard),
        incMemoryByteSize, "HOST_DATA");

    auto info = memAlloc->GetUsedInfo(threshold);
    EXPECT_EQ(info->usedBytes, 0ULL);
    EXPECT_EQ(info->unusedFragBytes, 0ULL);
    EXPECT_EQ(info->leftBytes, memoryByteSize + incMemoryByteSize);

    auto addr = memAlloc->Alloc(blockByteSize);
    EXPECT_EQ(addr, memoryAddr);

    addr = memAlloc->Alloc(blockByteSize);
    EXPECT_EQ(addr, memoryAddr + blockByteSize);
}
TEST_F(TestOckHmmSubMemoryAlloc, multiple_increase_alloc_free)
{
    BuildMemAlloc();
    auto incMockGuard = BuildOckAdapterMemoryGuard(incMemoryByteSize, incMemoryAddr, location);
    memAlloc->IncBindMemoryToMemPool(std::unique_ptr<acladapter::OckAdapterMemoryGuard>(incMockGuard),
        incMemoryByteSize, "HOST_DATA");

    // 分配完初始的全部内存
    auto addr = memAlloc->Alloc(memoryByteSize);
    auto info = memAlloc->GetUsedInfo(threshold);
    EXPECT_EQ(addr, memoryAddr);
    CompareUsedInfo(info, memoryByteSize, 0ULL, incMemoryByteSize);

    // 分配完新增内存
    addr = memAlloc->Alloc(incMemoryByteSize);
    EXPECT_EQ(addr, incMemoryAddr);
    memAlloc->Free(addr, incMemoryByteSize);

    // 继续新增内存
    incMockGuard = BuildOckAdapterMemoryGuard(incMemoryByteSize, incMemoryAddr + incMemoryByteSize, location);
    memAlloc->IncBindMemoryToMemPool(std::unique_ptr<acladapter::OckAdapterMemoryGuard>(incMockGuard),
        incMemoryByteSize, "HOST_DATA");
    addr = memAlloc->Alloc(incMemoryByteSize + incMemoryByteSize / 2ULL);
    info = memAlloc->GetUsedInfo(threshold);
    EXPECT_EQ(addr, incMemoryAddr);
    CompareUsedInfo(info, memoryByteSize + incMemoryByteSize + incMemoryByteSize / 2ULL, 0ULL,
        incMemoryByteSize / 2ULL);
}
} // namespace test
} // namespace hmm
} // namespace ock
