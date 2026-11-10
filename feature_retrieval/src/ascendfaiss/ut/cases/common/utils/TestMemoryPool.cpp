/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include "ascendfaiss/ascenddaemon/utils/MemoryPool.h"

using namespace testing;

namespace ascend
{

class FakeUpstream : public IUpstream
{
   public:
    void *alloc(size_t size) override
    {
        totalBytes += size;
        ++allocCount;
        if (failNext > 0)
        {
            --failNext;
            return nullptr;
        }
        return std::malloc(size);
    }

    void free(void *p) override
    {
        ++freeCount;
        std::free(p);
    }

    size_t totalBytes = 0;
    int allocCount = 0;
    int freeCount = 0;
    int failNext = 0;
};

// Exercise alignment repair and upstream-pointer preservation.
class MisalignedUpstream : public IUpstream
{
   public:
    void *alloc(size_t size) override
    {
        auto *raw = static_cast<uint8_t *>(std::malloc(size + MemoryPool::CANN_ADDR_ALIGN * 2));
        if (raw == nullptr)
        {
            return nullptr;
        }
        const uintptr_t aligned = (reinterpret_cast<uintptr_t>(raw) + MemoryPool::CANN_ADDR_ALIGN - 1) /
                                  MemoryPool::CANN_ADDR_ALIGN * MemoryPool::CANN_ADDR_ALIGN;
        auto *p = reinterpret_cast<uint8_t *>(aligned) + 16;
        bases[p] = raw;
        ++allocCount;
        return p;
    }

    void free(void *p) override
    {
        ++freeCount;
        auto it = bases.find(p);
        if (it != bases.end())
        {
            std::free(it->second);
            bases.erase(it);
        }
    }

    std::map<void *, uint8_t *> bases;  // Returned pointer -> malloc base.
    int allocCount = 0;
    int freeCount = 0;
};

class TestMemoryPool : public Test
{
   protected:
    FakeUpstream up;
    MemoryPool pool;
};

TEST_F(TestMemoryPool, InitInvalidArg)
{
    MemoryPool p;
    EXPECT_EQ(p.init(nullptr), ACL_ERROR_INVALID_PARAM);
    EXPECT_EQ(p.init(&up, 0), ACL_ERROR_INVALID_PARAM);
    void *ptr = nullptr;
    EXPECT_EQ(p.allocate(16, &ptr), ACL_ERROR_UNINITIALIZE);
}

TEST_F(TestMemoryPool, AllocateRoundsSizeByGranularity)
{
    ASSERT_EQ(pool.init(&up, 8ull * 1024 * 1024, 1), ACL_SUCCESS);
    void *a = nullptr;
    void *b = nullptr;
    void *c = nullptr;
    ASSERT_EQ(pool.allocate(1000, &a), ACL_SUCCESS);
    ASSERT_EQ(pool.allocate(5000, &b), ACL_SUCCESS);
    ASSERT_EQ(pool.allocate(3ull * 1024 * 1024, &c), ACL_SUCCESS);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    ASSERT_NE(c, nullptr);
    EXPECT_NE(a, b);
    EXPECT_NE(b, c);

    MemoryPool::Stat s = pool.stat();
    // 1000 -> 64 B granularity; 5000 -> 4 KiB granularity; 3 MiB keeps 4 KiB
    // granularity because rounding it to 4 MiB would waste a third of it.
    EXPECT_EQ(s.inUse, static_cast<size_t>(1088 + 8192 + 3ull * 1024 * 1024 + 4096));
    EXPECT_EQ(s.numChunks, 1u);
}

TEST_F(TestMemoryPool, FreeAndReuseNoGrowth)
{
    ASSERT_EQ(pool.init(&up, 8ull * 1024 * 1024, 1), ACL_SUCCESS);
    void *a = nullptr;
    void *b = nullptr;
    ASSERT_EQ(pool.allocate(4096, &a), ACL_SUCCESS);
    ASSERT_EQ(pool.allocate(4096, &b), ACL_SUCCESS);
    size_t chunksBefore = pool.stat().numChunks;

    ASSERT_EQ(pool.deallocate(b), ACL_SUCCESS);
    void *b2 = nullptr;
    ASSERT_EQ(pool.allocate(4096, &b2), ACL_SUCCESS);
    EXPECT_EQ(pool.stat().numChunks, chunksBefore);
}

TEST_F(TestMemoryPool, CoalesceAdjacentFreeBlocks)
{
    ASSERT_EQ(pool.init(&up, 8ull * 1024 * 1024, 1), ACL_SUCCESS);
    void *a = nullptr;
    void *b = nullptr;
    void *c = nullptr;
    ASSERT_EQ(pool.allocate(1ull * 1024 * 1024, &a), ACL_SUCCESS);
    ASSERT_EQ(pool.allocate(1ull * 1024 * 1024, &b), ACL_SUCCESS);
    ASSERT_EQ(pool.allocate(1ull * 1024 * 1024, &c), ACL_SUCCESS);
    size_t chunksBefore = pool.stat().numChunks;

    ASSERT_EQ(pool.deallocate(a), ACL_SUCCESS);
    ASSERT_EQ(pool.deallocate(b), ACL_SUCCESS);
    EXPECT_GE(pool.stat().largestFree, 2ull * 1024 * 1024);

    void *big = nullptr;
    ASSERT_EQ(pool.allocate(2ull * 1024 * 1024, &big), ACL_SUCCESS);
    EXPECT_EQ(pool.stat().numChunks, chunksBefore);
}

TEST_F(TestMemoryPool, GrowMultipleChunks)
{
    ASSERT_EQ(pool.init(&up, 8ull * 1024 * 1024, 1), ACL_SUCCESS);
    std::vector<void *> ptrs;
    for (int i = 0; i < 6; ++i)
    {
        void *p = nullptr;
        ASSERT_EQ(pool.allocate(4ull * 1024 * 1024, &p), ACL_SUCCESS);
        ptrs.push_back(p);
    }
    EXPECT_GE(pool.stat().numChunks, 3u);
    for (void *p : ptrs)
    {
        EXPECT_EQ(pool.deallocate(p), ACL_SUCCESS);
    }
}

TEST_F(TestMemoryPool, TrimKeepsAtMostMaxFreeChunks)
{
    ASSERT_EQ(pool.init(&up, 8ull * 1024 * 1024, /*maxFreeChunks=*/1), ACL_SUCCESS);
    std::vector<void *> ptrs;
    for (int i = 0; i < 6; ++i)
    {
        void *p = nullptr;
        ASSERT_EQ(pool.allocate(4ull * 1024 * 1024, &p), ACL_SUCCESS);
        ptrs.push_back(p);
    }
    for (void *p : ptrs)
    {
        ASSERT_EQ(pool.deallocate(p), ACL_SUCCESS);
    }
    MemoryPool::Stat s = pool.stat();
    EXPECT_LE(s.numFreeChunks, 1u);
    EXPECT_LE(s.numChunks, 1u);
}

TEST_F(TestMemoryPool, ShrinkReleasesFreeChunks)
{
    ASSERT_EQ(pool.init(&up, 8ull * 1024 * 1024, 2), ACL_SUCCESS);
    void *a = nullptr;
    ASSERT_EQ(pool.allocate(4ull * 1024 * 1024, &a), ACL_SUCCESS);
    ASSERT_EQ(pool.deallocate(a), ACL_SUCCESS);
    ASSERT_GE(pool.stat().numChunks, 1u);

    size_t released = pool.shrink();
    EXPECT_GE(released, 1u);
    EXPECT_EQ(pool.stat().numChunks, 0u);
}

TEST_F(TestMemoryPool, DeallocateInvalidPtr)
{
    ASSERT_EQ(pool.init(&up), ACL_SUCCESS);
    int local = 0;
    EXPECT_EQ(pool.deallocate(&local), ACL_ERROR_INVALID_PARAM);
    EXPECT_EQ(pool.deallocate(nullptr), ACL_SUCCESS);
}

TEST_F(TestMemoryPool, OutOfMemoryWhenUpstreamFails)
{
    ASSERT_EQ(pool.init(&up, 8ull * 1024 * 1024, 1), ACL_SUCCESS);
    up.failNext = 100;
    void *p = nullptr;
    EXPECT_EQ(pool.allocate(4096, &p), ACL_ERROR_BAD_ALLOC);
}

TEST_F(TestMemoryPool, OutOfMemoryTriggersShrinkThenRetry)
{
    ASSERT_EQ(pool.init(&up, 4ull * 1024 * 1024, 1), ACL_SUCCESS);

    void *small = nullptr;
    ASSERT_EQ(pool.allocate(1024, &small), ACL_SUCCESS);
    ASSERT_EQ(pool.deallocate(small), ACL_SUCCESS);
    ASSERT_EQ(pool.stat().numFreeChunks, 1u);
    ASSERT_EQ(up.freeCount, 0);

    // Force growth beyond the cached chunk, then fail the first attempt.
    up.failNext = 1;
    void *big = nullptr;
    EXPECT_EQ(pool.allocate(8ull * 1024 * 1024, &big), ACL_SUCCESS);
    EXPECT_NE(big, nullptr);
    EXPECT_EQ(up.freeCount, 1);
    EXPECT_EQ(pool.stat().numFreeChunks, 0u);
    EXPECT_EQ(pool.deallocate(big), ACL_SUCCESS);
}

TEST_F(TestMemoryPool, DoubleFreeIsRejectedAndKeepsAccounting)
{
    ASSERT_EQ(pool.init(&up, 8ull * 1024 * 1024, 1), ACL_SUCCESS);
    void *a = nullptr;
    void *b = nullptr;
    ASSERT_EQ(pool.allocate(4096, &a), ACL_SUCCESS);
    const size_t blockSize = pool.stat().inUse;
    ASSERT_EQ(pool.allocate(4096, &b), ACL_SUCCESS);
    const size_t inUseBefore = pool.stat().inUse;

    ASSERT_EQ(pool.deallocate(a), ACL_SUCCESS);
    EXPECT_EQ(pool.deallocate(a), ACL_ERROR_INVALID_PARAM);

    EXPECT_EQ(pool.stat().inUse, inUseBefore - blockSize);
    EXPECT_EQ(pool.deallocate(b), ACL_SUCCESS);
    EXPECT_EQ(pool.stat().inUse, 0u);
}

// Characterizes unsafe reinitialization: old chunks are freed through the new upstream.
// Re-initializing a live pool would release the old upstream's chunks through
// the new upstream. It must be rejected, and the pool must keep working against
// the original upstream.
TEST_F(TestMemoryPool, ReinitOnLivePoolIsRejectedAndKeepsOriginalUpstream)
{
    FakeUpstream upOld;
    FakeUpstream upNew;
    MemoryPool p;

    ASSERT_EQ(p.init(&upOld, 1ull * 1024 * 1024, 1), ACL_SUCCESS);
    void *a = nullptr;
    ASSERT_EQ(p.allocate(4096, &a), ACL_SUCCESS);
    ASSERT_EQ(upOld.allocCount, 1);

    EXPECT_EQ(p.init(&upNew, 1ull * 1024 * 1024, 1), ACL_ERROR_REPEAT_INITIALIZE);

    p.destroy();
    EXPECT_EQ(upOld.freeCount, 1);  // Released through the upstream that owns it.
    EXPECT_EQ(upNew.allocCount, 0);
    EXPECT_EQ(upNew.freeCount, 0);
}

// destroy() must fully reset the pool so that it can be re-initialized, and so
// that a stale upstream pointer cannot be used afterwards.
TEST_F(TestMemoryPool, DestroyResetsPoolSoItCanBeReinitialized)
{
    FakeUpstream upOld;
    FakeUpstream upNew;
    MemoryPool p;

    ASSERT_EQ(p.init(&upOld, 1ull * 1024 * 1024, 1), ACL_SUCCESS);
    void *a = nullptr;
    ASSERT_EQ(p.allocate(4096, &a), ACL_SUCCESS);
    p.destroy();
    EXPECT_EQ(upOld.allocCount, upOld.freeCount);

    // Uninitialized again: no allocation may be served from the dropped upstream.
    void *b = nullptr;
    EXPECT_EQ(p.allocate(4096, &b), ACL_ERROR_UNINITIALIZE);
    EXPECT_EQ(upOld.allocCount, 1);

    ASSERT_EQ(p.init(&upNew, 1ull * 1024 * 1024, 1), ACL_SUCCESS);
    ASSERT_EQ(p.allocate(4096, &b), ACL_SUCCESS);
    EXPECT_EQ(upNew.allocCount, 1);
    EXPECT_EQ(p.deallocate(b), ACL_SUCCESS);
}

TEST_F(TestMemoryPool, DestroyBalancesUpstream)
{
    ASSERT_EQ(pool.init(&up, 8ull * 1024 * 1024, 4), ACL_SUCCESS);
    std::vector<void *> ptrs;
    for (int i = 0; i < 5; ++i)
    {
        void *p = nullptr;
        ASSERT_EQ(pool.allocate(3ull * 1024 * 1024, &p), ACL_SUCCESS);
        ptrs.push_back(p);
    }
    for (void *p : ptrs)
    {
        ASSERT_EQ(pool.deallocate(p), ACL_SUCCESS);
    }
    pool.destroy();
    EXPECT_EQ(up.allocCount, up.freeCount);
    EXPECT_EQ(pool.stat().numChunks, 0u);
}

TEST_F(TestMemoryPool, SubBlocksHonorCannAlignmentAndTailPad)
{
    ASSERT_EQ(pool.init(&up, 64ull * 1024 * 1024, 1), ACL_SUCCESS);
    const size_t sizes[] = {
        1, 31, 32, 33, 64, 1000, 4064, 4096, 4097, 8192, 2ull * 1024 * 1024, 2ull * 1024 * 1024 + 1};

    size_t prevInUse = 0;
    std::vector<std::pair<unsigned char *, size_t>> blocks;
    for (size_t len : sizes)
    {
        void *p = nullptr;
        ASSERT_EQ(pool.allocate(len, &p), ACL_SUCCESS) << "len=" << len;
        EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % MemoryPool::CANN_ADDR_ALIGN, 0u) << "len=" << len;
        const size_t required =
            (len + MemoryPool::CANN_SIZE_ALIGN - 1) / MemoryPool::CANN_SIZE_ALIGN * MemoryPool::CANN_SIZE_ALIGN +
            MemoryPool::CANN_SIZE_PAD;
        const size_t got = pool.stat().inUse - prevInUse;
        EXPECT_GE(got, required) << "len=" << len;
        std::memset(p, static_cast<int>(blocks.size() + 1), required);
        blocks.emplace_back(static_cast<unsigned char *>(p), required);
        prevInUse = pool.stat().inUse;
    }
    // Tail writes must not overlap neighboring allocations.
    for (size_t i = 0; i < blocks.size(); ++i)
    {
        const auto &block = blocks[i];
        EXPECT_TRUE(
            std::all_of(block.first, block.first + block.second, [i](unsigned char value) { return value == i + 1; }));
        EXPECT_EQ(pool.deallocate(block.first), ACL_SUCCESS);
    }
}

// Tail padding must not add an entire huge page.
TEST_F(TestMemoryPool, TailPadDoesNotCrossHugeGranularity)
{
    ASSERT_EQ(pool.init(&up, 64ull * 1024 * 1024, 1), ACL_SUCCESS);
    void *p = nullptr;
    ASSERT_EQ(pool.allocate(4ull * 1024 * 1024, &p), ACL_SUCCESS);
    const size_t first = pool.stat().inUse;
    EXPECT_GE(first, 4ull * 1024 * 1024 + MemoryPool::CANN_SIZE_PAD);
    EXPECT_LE(first, 4ull * 1024 * 1024 + MemoryPool::SMALL_ALIGN);

    void *q = nullptr;
    ASSERT_EQ(pool.allocate(2ull * 1024 * 1024, &q), ACL_SUCCESS);
    const size_t second = pool.stat().inUse - first;
    EXPECT_GE(second, 2ull * 1024 * 1024 + MemoryPool::CANN_SIZE_PAD);
    EXPECT_LE(second, 2ull * 1024 * 1024 + MemoryPool::SMALL_ALIGN);
}

TEST_F(TestMemoryPool, TailPadIsFreeWhenGranularityAlreadyCovers)
{
    ASSERT_EQ(pool.init(&up, 64ull * 1024 * 1024, 1), ACL_SUCCESS);
    void *a = nullptr;
    ASSERT_EQ(pool.allocate(5000, &a), ACL_SUCCESS);
    EXPECT_EQ(pool.stat().inUse, 8192u);

    // Just under 4 MiB: huge-page rounding absorbs the tail pad for only 4 KiB
    // of slack, which is within the waste budget, so it is taken.
    void *b = nullptr;
    const size_t nearHuge = 4ull * 1024 * 1024 - MemoryPool::SMALL_ALIGN;
    ASSERT_EQ(pool.allocate(nearHuge, &b), ACL_SUCCESS);
    EXPECT_EQ(pool.stat().inUse, 8192u + 4ull * 1024 * 1024);
}

// Rounding a request that barely crosses a huge page up to the next huge page
// used to nearly double its footprint (2 MiB + 32 B -> 4 MiB, 3 MiB -> 4 MiB).
TEST_F(TestMemoryPool, HugeGranularityDoesNotInflateSmallOverhang)
{
    ASSERT_EQ(pool.init(&up, 64ull * 1024 * 1024, 1), ACL_SUCCESS);
    void *a = nullptr;
    const size_t overhang = 2ull * 1024 * 1024 + 32;
    ASSERT_EQ(pool.allocate(overhang, &a), ACL_SUCCESS);
    const size_t firstUse = pool.stat().inUse;
    EXPECT_GE(firstUse, overhang + MemoryPool::CANN_SIZE_PAD);
    EXPECT_LE(firstUse, overhang + MemoryPool::SMALL_ALIGN * 2);

    void *b = nullptr;
    ASSERT_EQ(pool.allocate(3ull * 1024 * 1024, &b), ACL_SUCCESS);
    const size_t secondUse = pool.stat().inUse - firstUse;
    EXPECT_LE(secondUse, 3ull * 1024 * 1024 + MemoryPool::SMALL_ALIGN);
}

TEST_F(TestMemoryPool, AlignsFirstBlockWhenUpstreamBaseIsMisaligned)
{
    MisalignedUpstream misaligned;
    MemoryPool p;
    ASSERT_EQ(p.init(&misaligned, 8ull * 1024 * 1024, 1), ACL_SUCCESS);

    std::vector<void *> ptrs;
    for (int i = 0; i < 8; ++i)
    {
        void *ptr = nullptr;
        ASSERT_EQ(p.allocate(100000, &ptr), ACL_SUCCESS);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % MemoryPool::CANN_ADDR_ALIGN, 0u);
        ptrs.push_back(ptr);
    }
    for (void *ptr : ptrs)
    {
        ASSERT_EQ(p.deallocate(ptr), ACL_SUCCESS);
    }
    p.destroy();

    EXPECT_EQ(misaligned.allocCount, misaligned.freeCount);
    EXPECT_TRUE(misaligned.bases.empty());
}

TEST_F(TestMemoryPool, RejectsRequestThatWouldOverflowRounding)
{
    ASSERT_EQ(pool.init(&up, 4ull * 1024 * 1024, 1), ACL_SUCCESS);
    void *p = nullptr;
    EXPECT_EQ(pool.allocate(MemoryPool::MAX_REQUEST + 1, &p), ACL_ERROR_INVALID_PARAM);
    EXPECT_EQ(p, nullptr);
}

TEST_F(TestMemoryPool, TinyListAllocationsDoNotConsumeOnePageEach)
{
    ASSERT_EQ(pool.init(&up, 128 * 1024, 1), ACL_SUCCESS);
    std::vector<void *> blocks;
    for (size_t i = 0; i < 1024; ++i)
    {
        void *p = nullptr;
        ASSERT_EQ(pool.allocate(sizeof(uint64_t), &p), ACL_SUCCESS);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % 64, 0u);
        blocks.push_back(p);
    }
    EXPECT_EQ(up.allocCount, 1);
    EXPECT_LE(pool.stat().inUse, 1024u * 64);
    for (void *p : blocks)
    {
        EXPECT_EQ(pool.deallocate(p), ACL_SUCCESS);
    }
    EXPECT_EQ(pool.stat().inUse, 0u);
}

TEST_F(TestMemoryPool, TinyBlocksCoalesceAndSplitAtAddressAlignment)
{
    ASSERT_EQ(pool.init(&up, 16384, 1), ACL_SUCCESS);
    void *first = nullptr;
    void *second = nullptr;
    void *guard = nullptr;
    ASSERT_EQ(pool.allocate(8, &first), ACL_SUCCESS);
    ASSERT_EQ(pool.allocate(8, &second), ACL_SUCCESS);
    ASSERT_EQ(pool.allocate(8, &guard), ACL_SUCCESS);
    ASSERT_EQ(pool.deallocate(first), ACL_SUCCESS);
    ASSERT_EQ(pool.deallocate(second), ACL_SUCCESS);

    // A padded 96B request needs both adjacent 64B blocks.
    void *merged = nullptr;
    ASSERT_EQ(pool.allocate(96, &merged), ACL_SUCCESS);
    EXPECT_EQ(merged, first);
    ASSERT_EQ(pool.deallocate(merged), ACL_SUCCESS);

    // Split the merged block back into two reusable 64B blocks.
    void *left = nullptr;
    void *right = nullptr;
    ASSERT_EQ(pool.allocate(8, &left), ACL_SUCCESS);
    ASSERT_EQ(pool.allocate(8, &right), ACL_SUCCESS);
    EXPECT_EQ(left, first);
    EXPECT_EQ(right, second);
    EXPECT_EQ(pool.stat().inUse, 3u * 64);
    EXPECT_EQ(up.allocCount, 1);
    EXPECT_EQ(pool.deallocate(left), ACL_SUCCESS);
    EXPECT_EQ(pool.deallocate(right), ACL_SUCCESS);
    EXPECT_EQ(pool.deallocate(guard), ACL_SUCCESS);
    EXPECT_EQ(pool.shrink(), 1u);
}

TEST_F(TestMemoryPool, StatCountersTrackAllocateFreeAndShrink)
{
    ASSERT_EQ(pool.init(&up, 16384, 1), ACL_SUCCESS);
    EXPECT_EQ(pool.stat().reserved, 0u);
    EXPECT_EQ(pool.stat().numFreeBlocks, 0u);

    std::vector<void *> live;
    for (int i = 0; i < 32; ++i)
    {
        void *p = nullptr;
        ASSERT_EQ(pool.allocate(8, &p), ACL_SUCCESS);
        live.push_back(p);
    }
    MemoryPool::Stat s = pool.stat();
    EXPECT_EQ(s.reserved, 16384u);
    EXPECT_EQ(s.inUse, 32u * 64);
    EXPECT_EQ(s.numChunks, 1u);
    EXPECT_EQ(s.numFreeChunks, 0u);
    // Only the trailing remainder is free.
    EXPECT_EQ(s.numFreeBlocks, 1u);
    EXPECT_GE(s.largestFree, 64u);

    // Non-adjacent holes cannot coalesce.
    for (size_t i = 0; i < live.size(); i += 2)
    {
        ASSERT_EQ(pool.deallocate(live[i]), ACL_SUCCESS);
    }
    s = pool.stat();
    EXPECT_EQ(s.inUse, 16u * 64);
    EXPECT_EQ(s.numFreeBlocks, 17u);

    for (size_t i = 1; i < live.size(); i += 2)
    {
        ASSERT_EQ(pool.deallocate(live[i]), ACL_SUCCESS);
    }
    s = pool.stat();
    EXPECT_EQ(s.inUse, 0u);
    EXPECT_EQ(s.numFreeBlocks, 1u);
    EXPECT_EQ(s.numFreeChunks, 1u);

    EXPECT_EQ(pool.shrink(), 1u);
    s = pool.stat();
    EXPECT_EQ(s.reserved, 0u);
    EXPECT_EQ(s.numChunks, 0u);
    EXPECT_EQ(s.numFreeBlocks, 0u);
    EXPECT_EQ(s.largestFree, 0u);
}

TEST_F(TestMemoryPool, FailedGrowLeavesPoolUnchanged)
{
    ASSERT_EQ(pool.init(&up, 16384, 1), ACL_SUCCESS);
    up.failNext = 1;
    void *p = nullptr;
    EXPECT_EQ(pool.allocate(64, &p), ACL_ERROR_BAD_ALLOC);
    MemoryPool::Stat s = pool.stat();
    EXPECT_EQ(s.numChunks, 0u);
    EXPECT_EQ(s.reserved, 0u);
    EXPECT_EQ(s.inUse, 0u);
    EXPECT_EQ(s.numFreeBlocks, 0u);
    EXPECT_EQ(up.freeCount, 0);

    ASSERT_EQ(pool.allocate(64, &p), ACL_SUCCESS);
    EXPECT_EQ(pool.stat().numChunks, 1u);
    EXPECT_EQ(pool.deallocate(p), ACL_SUCCESS);
}

namespace
{
constexpr int CONCURRENT_THREADS = 4;
constexpr int CONCURRENT_ITERS = 400;
constexpr size_t CONCURRENT_MAX_REQUEST = 48ull * 1024;
constexpr size_t CONCURRENT_HOLD_DEPTH = 8;

// Overlapping blocks cannot be caught by a sanitizer, because the whole chunk is a single
// upstream allocation. Giving every thread its own byte pattern turns an overlap into a
// deterministic mismatch instead.
bool PatternIntact(const void *ptr, size_t size, uint8_t pattern)
{
    const auto *bytes = static_cast<const uint8_t *>(ptr);
    for (size_t i = 0; i < size; ++i)
    {
        if (bytes[i] != pattern)
        {
            return false;
        }
    }
    return true;
}
}  // namespace

TEST_F(TestMemoryPool, TryDeallocateOnlyClaimsItsOwnPointers)
{
    FakeUpstream other;
    MemoryPool otherPool;
    ASSERT_EQ(pool.init(&up, 16384, 1), ACL_SUCCESS);
    ASSERT_EQ(otherPool.init(&other, 16384, 1), ACL_SUCCESS);

    void *mine = nullptr;
    void *theirs = nullptr;
    ASSERT_EQ(pool.allocate(128, &mine), ACL_SUCCESS);
    ASSERT_EQ(otherPool.allocate(128, &theirs), ACL_SUCCESS);
    const size_t mineInUse = pool.stat().inUse;
    const size_t theirsInUse = otherPool.stat().inUse;
    ASSERT_GE(mineInUse, 128u);
    ASSERT_GE(theirsInUse, 128u);

    // Probing a foreign pointer must be a silent no-op, so a caller can scan pools.
    EXPECT_FALSE(pool.tryDeallocate(theirs));
    EXPECT_FALSE(otherPool.tryDeallocate(mine));
    EXPECT_EQ(pool.stat().inUse, mineInUse);
    EXPECT_EQ(otherPool.stat().inUse, theirsInUse);

    EXPECT_TRUE(pool.tryDeallocate(mine));
    EXPECT_EQ(pool.stat().inUse, 0u);
    EXPECT_FALSE(pool.tryDeallocate(mine));  // Second release must not double-free.

    EXPECT_TRUE(otherPool.tryDeallocate(theirs));
    EXPECT_EQ(otherPool.stat().inUse, 0u);
}

TEST_F(TestMemoryPool, TryDeallocateIsSafeOnNullAndUninitializedPool)
{
    EXPECT_FALSE(pool.tryDeallocate(nullptr));
    void *stack = &up;
    EXPECT_FALSE(pool.tryDeallocate(stack));  // Uninitialized pool owns nothing.

    ASSERT_EQ(pool.init(&up, 16384, 1), ACL_SUCCESS);
    void *p = nullptr;
    ASSERT_EQ(pool.allocate(64, &p), ACL_SUCCESS);
    pool.destroy();
    EXPECT_FALSE(pool.tryDeallocate(p));  // destroy() already reclaimed the chunk.
}

TEST_F(TestMemoryPool, ConcurrentAllocFreeKeepsBlocksDisjoint)
{
    ASSERT_EQ(pool.init(&up, 1ull * 1024 * 1024, 2), ACL_SUCCESS);

    std::atomic<int> corruption{0};
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    threads.reserve(CONCURRENT_THREADS);

    for (int t = 0; t < CONCURRENT_THREADS; ++t)
    {
        threads.emplace_back(
            [this, t, &corruption, &failures]
            {
                std::mt19937 rng(static_cast<unsigned>(t) + 1u);
                std::uniform_int_distribution<size_t> sizeDist(1, CONCURRENT_MAX_REQUEST);
                const auto pattern = static_cast<uint8_t>(0xA0 + t);
                std::vector<std::pair<void *, size_t>> held;

                for (int i = 0; i < CONCURRENT_ITERS; ++i)
                {
                    const size_t want = sizeDist(rng);
                    void *p = nullptr;
                    if (pool.allocate(want, &p) != ACL_SUCCESS || p == nullptr)
                    {
                        ++failures;
                        continue;
                    }
                    std::memset(p, pattern, want);
                    held.emplace_back(p, want);

                    if (held.size() > CONCURRENT_HOLD_DEPTH)
                    {
                        const auto victim = held.front();
                        held.erase(held.begin());
                        if (!PatternIntact(victim.first, victim.second, pattern))
                        {
                            ++corruption;
                        }
                        (void)pool.deallocate(victim.first);
                    }
                }

                for (const auto &kv : held)
                {
                    if (!PatternIntact(kv.first, kv.second, pattern))
                    {
                        ++corruption;
                    }
                    (void)pool.deallocate(kv.first);
                }
            });
    }
    for (auto &th : threads)
    {
        th.join();
    }

    EXPECT_EQ(corruption.load(), 0);
    EXPECT_EQ(failures.load(), 0);

    // Everything was released, so each surviving chunk must have coalesced back into one block.
    const MemoryPool::Stat s = pool.stat();
    EXPECT_EQ(s.inUse, 0u);
    EXPECT_EQ(s.numFreeBlocks, s.numChunks);
    EXPECT_EQ(s.numFreeChunks, s.numChunks);
    EXPECT_LE(s.numChunks, 2u);  // maxFreeChunks trim must still hold under contention.
}

TEST_F(TestMemoryPool, ConcurrentShrinkDoesNotRaceAllocations)
{
    ASSERT_EQ(pool.init(&up, 256ull * 1024, 0), ACL_SUCCESS);

    std::atomic<bool> stop{false};
    std::atomic<int> corruption{0};
    std::vector<std::thread> threads;
    threads.reserve(CONCURRENT_THREADS);

    for (int t = 0; t < CONCURRENT_THREADS - 1; ++t)
    {
        threads.emplace_back(
            [this, t, &corruption]
            {
                std::mt19937 rng(static_cast<unsigned>(t) + 101u);
                std::uniform_int_distribution<size_t> sizeDist(1, 16ull * 1024);
                const auto pattern = static_cast<uint8_t>(0x30 + t);
                for (int i = 0; i < CONCURRENT_ITERS; ++i)
                {
                    const size_t want = sizeDist(rng);
                    void *p = nullptr;
                    if (pool.allocate(want, &p) != ACL_SUCCESS || p == nullptr)
                    {
                        continue;
                    }
                    std::memset(p, pattern, want);
                    if (!PatternIntact(p, want, pattern))
                    {
                        ++corruption;
                    }
                    (void)pool.deallocate(p);
                }
            });
    }

    // A chunk may only be handed back to the upstream once no live block points into it,
    // even while other threads are carving blocks out of neighbouring chunks.
    threads.emplace_back(
        [this, &stop]
        {
            while (!stop.load())
            {
                (void)pool.shrink();
            }
        });

    for (size_t i = 0; i + 1 < threads.size(); ++i)
    {
        threads[i].join();
    }
    stop.store(true);
    threads.back().join();

    EXPECT_EQ(corruption.load(), 0);

    const MemoryPool::Stat s = pool.stat();
    EXPECT_EQ(s.inUse, 0u);
    EXPECT_EQ(up.allocCount, up.freeCount + static_cast<int>(s.numChunks));
}

}  // namespace ascend
