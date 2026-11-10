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

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mockcpp/mockcpp.hpp>
#include <string>
#include <vector>

#include "../TestEnvGuard.h"
#include "acl.h"
#include "ascendfaiss/ascenddaemon/utils/DeviceMemArena.h"
#include "ascendfaiss/ascenddaemon/utils/MemorySpace.h"

using namespace testing;
using namespace std;

namespace ascend
{
namespace
{
size_t g_chunkAllocCount = 0;
size_t g_lastChunkSize = 0;
aclrtMemMallocPolicy g_lastPolicy = ACL_MEM_MALLOC_NORMAL_ONLY;
size_t g_aclFreeCount = 0;
size_t g_directAllocCount = 0;
size_t g_failChunkAllocs = 0;
int32_t g_fakeDevice = 0;
bool g_failGetDevice = false;

aclError StubMallocAlign32(void **devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    ++g_chunkAllocCount;
    g_lastChunkSize = size;
    g_lastPolicy = policy;
    if (g_failChunkAllocs > 0)
    {
        --g_failChunkAllocs;
        *devPtr = nullptr;
        return ACL_ERROR_BAD_ALLOC;
    }
    *devPtr = std::malloc(size);
    return (*devPtr != nullptr) ? ACL_ERROR_NONE : 1;
}

aclError StubFree(void *devPtr)
{
    ++g_aclFreeCount;
    std::free(devPtr);
    return ACL_ERROR_NONE;
}

aclError StubGetDevice(int32_t *deviceId)
{
    if (g_failGetDevice)
    {
        return ACL_ERROR_UNINITIALIZE;
    }
    *deviceId = g_fakeDevice;
    return ACL_ERROR_NONE;
}

aclError StubMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy)
{
    ++g_directAllocCount;
    *devPtr = std::malloc(size);
    return (*devPtr != nullptr) ? ACL_ERROR_NONE : 1;
}
}  // namespace

class TestMemorySpace : public Test
{
   public:
    void SetUp() override
    {
        g_chunkAllocCount = 0;
        g_lastChunkSize = 0;
        g_lastPolicy = ACL_MEM_MALLOC_NORMAL_ONLY;
        g_aclFreeCount = 0;
        g_directAllocCount = 0;
        g_failChunkAllocs = 0;
        g_fakeDevice = 0;
        g_failGetDevice = false;
        // Limit host memory use for newly created pools.
        chunkMbEnv_.reset(new ScopedEnv("ASCENDFAISS_MEM_POOL_CHUNK_MB", "4"));
        SetMemoryPoolEnabled(false);
        MOCKER_CPP(&aclrtGetDevice).stubs().will(invoke(StubGetDevice));
    }

    void TearDown() override
    {
        // Destroy the pools while the ACL stubs are still installed. Shrinking alone is
        // not enough: a chunk that still backs a leaked pointer would stay resident and
        // later be freed through the real aclrtFree(). It also keeps each test from
        // inheriting pool state (and chunk size) from whichever test ran first.
        ResetMemoryPoolsForTest();
        SetMemoryPoolEnabled(false);
        GlobalMockObject::verify();
        chunkMbEnv_.reset();
    }

   private:
    std::unique_ptr<ScopedEnv> chunkMbEnv_;
};

// Preserve direct-allocation behavior and error messages.
TEST_F(TestMemorySpace, construct)
{
    int8_t *start = nullptr;
    size_t size = 0;
    string actualMsg = "";

    MemorySpace spaceType = MemorySpace::DEVICE;
    MOCKER_CPP(&aclrtMalloc)
        .stubs()
        .will(returnValue(1))
        .then(returnValue(0))
        .then(returnValue(1))
        .then(returnValue(0));
    try
    {
        AllocMemorySpace(spaceType, &start, size);
    }
    catch (std::exception &e)
    {
        actualMsg = e.what();
    }
    string expectMsg("failed to aclrtMalloc");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);

    actualMsg = "";
    try
    {
        AllocMemorySpace(spaceType, &start, size);
    }
    catch (std::exception &e)
    {
        actualMsg = e.what();
    }
    EXPECT_TRUE(actualMsg.empty());

    spaceType = MemorySpace::DEVICE_HUGEPAGE;
    try
    {
        AllocMemorySpace(spaceType, &start, size);
    }
    catch (std::exception &e)
    {
        actualMsg = e.what();
    }
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);

    actualMsg = "";
    try
    {
        AllocMemorySpace(spaceType, &start, size);
    }
    catch (std::exception &e)
    {
        actualMsg = e.what();
    }
    EXPECT_TRUE(actualMsg.empty());

    spaceType = static_cast<MemorySpace>(0);
    try
    {
        AllocMemorySpace(spaceType, &start, size);
    }
    catch (std::exception &e)
    {
        actualMsg = e.what();
    }
    expectMsg = string("Unsupported memoryspace type");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);
}

TEST_F(TestMemorySpace, PoolPathSplitsChunkAndKeepsSpacePolicy)
{
    MOCKER_CPP(&aclrtMallocAlign32).stubs().will(invoke(StubMallocAlign32));
    MOCKER_CPP(&aclrtFree).stubs().will(invoke(StubFree));
    SetMemoryPoolEnabled(true);

    int8_t *first = nullptr;
    ASSERT_NO_THROW(AllocMemorySpace(MemorySpace::DEVICE, &first, 4096));
    EXPECT_NE(first, nullptr);
    EXPECT_EQ(g_chunkAllocCount, 1u);
    EXPECT_EQ(g_lastPolicy, ACL_MEM_MALLOC_NORMAL_ONLY);

    int8_t *second = nullptr;
    ASSERT_NO_THROW(AllocMemorySpace(MemorySpace::DEVICE, &second, 4096));
    EXPECT_NE(second, first);
    EXPECT_EQ(g_chunkAllocCount, 1u);

    int8_t *huge = nullptr;
    ASSERT_NO_THROW(AllocMemorySpace(MemorySpace::DEVICE_HUGEPAGE, &huge, 4096));
    EXPECT_EQ(g_chunkAllocCount, 2u);
    EXPECT_EQ(g_lastPolicy, ACL_MEM_MALLOC_HUGE_FIRST);

    // Each pool retains its single idle chunk until explicitly shrunk.
    EXPECT_NO_THROW(FreeMemorySpace(MemorySpace::DEVICE, first));
    EXPECT_NO_THROW(FreeMemorySpace(MemorySpace::DEVICE, second));
    EXPECT_NO_THROW(FreeMemorySpace(MemorySpace::DEVICE_HUGEPAGE, huge));
    EXPECT_EQ(g_aclFreeCount, 0u);

    EXPECT_GE(ReleaseAllMemoryPools(), 2u);
    EXPECT_GE(g_aclFreeCount, 2u);
}

TEST_F(TestMemorySpace, PoolPathFreeRoutesByOwnerAcrossDevices)
{
    MOCKER_CPP(&aclrtMallocAlign32).stubs().will(invoke(StubMallocAlign32));
    MOCKER_CPP(&aclrtFree).stubs().will(invoke(StubFree));
    SetMemoryPoolEnabled(true);

    g_fakeDevice = 0;
    int8_t *p = nullptr;
    ASSERT_NO_THROW(AllocMemorySpace(MemorySpace::DEVICE, &p, 4096));
    const size_t chunksAfterAlloc = g_chunkAllocCount;

    g_fakeDevice = 1;
    EXPECT_NO_THROW(FreeMemorySpace(MemorySpace::DEVICE, p));
    EXPECT_EQ(g_aclFreeCount, 0u);

    g_fakeDevice = 0;
    int8_t *q = nullptr;
    ASSERT_NO_THROW(AllocMemorySpace(MemorySpace::DEVICE, &q, 4096));
    EXPECT_EQ(q, p);
    EXPECT_EQ(g_chunkAllocCount, chunksAfterAlloc);
    EXPECT_NO_THROW(FreeMemorySpace(MemorySpace::DEVICE, q));
}

TEST_F(TestMemorySpace, PoolPathAllocFailureReportsPoolStat)
{
    MOCKER_CPP(&aclrtMallocAlign32).stubs().will(returnValue(1));
    SetMemoryPoolEnabled(true);

    int8_t *start = nullptr;
    string actualMsg;
    // Bypass cached blocks to force upstream allocation.
    const size_t hugeSize = 4ull * 1024 * 1024 * 1024;
    try
    {
        AllocMemorySpace(MemorySpace::DEVICE, &start, hugeSize);
    }
    catch (std::exception &e)
    {
        actualMsg = e.what();
    }
    EXPECT_TRUE(actualMsg.find("MemoryPool") != string::npos);
    EXPECT_TRUE(actualMsg.find("reserved=") != string::npos);
}

TEST_F(TestMemorySpace, PoolPathRejectsUnsupportedSpace)
{
    MOCKER_CPP(&aclrtMallocAlign32).stubs().will(invoke(StubMallocAlign32));
    SetMemoryPoolEnabled(true);

    int8_t *start = nullptr;
    string actualMsg;
    try
    {
        AllocMemorySpace(static_cast<MemorySpace>(0), &start, 4096);
    }
    catch (std::exception &e)
    {
        actualMsg = e.what();
    }
    EXPECT_TRUE(actualMsg.find("Unsupported memoryspace type") != string::npos);
    EXPECT_EQ(g_chunkAllocCount, 0u);
}

TEST_F(TestMemorySpace, PoolPathHandlesZeroSize)
{
    MOCKER_CPP(&aclrtMallocAlign32).stubs().will(invoke(StubMallocAlign32));
    MOCKER_CPP(&aclrtFree).stubs().will(invoke(StubFree));
    SetMemoryPoolEnabled(true);

    int8_t *start = nullptr;
    ASSERT_NO_THROW(AllocMemorySpace(MemorySpace::DEVICE, &start, 0));
    EXPECT_NE(start, nullptr);
    EXPECT_NO_THROW(FreeMemorySpace(MemorySpace::DEVICE, start));
}

TEST_F(TestMemorySpace, NonPooledPointerIsFreedThroughAcl)
{
    MOCKER_CPP(&aclrtMalloc).stubs().will(invoke(StubMalloc));
    MOCKER_CPP(&aclrtFree).stubs().will(invoke(StubFree));

    int8_t *start = nullptr;
    ASSERT_NO_THROW(AllocMemorySpace(MemorySpace::DEVICE, &start, 4096));
    EXPECT_NE(start, nullptr);

    SetMemoryPoolEnabled(true);
    EXPECT_NO_THROW(FreeMemorySpace(MemorySpace::DEVICE, start));
    EXPECT_EQ(g_aclFreeCount, 1u);
}

TEST_F(TestMemorySpace, MemoryPoolEnvValueIsOptIn)
{
    // Only an explicit "1" enables the pool. The device-side binary links this code and
    // does not inherit the host environment, so defaulting to on would leave it with no
    // reachable way back to plain aclrtMalloc.
    EXPECT_TRUE(MemoryPoolEnabledByEnvValue("1"));

    EXPECT_FALSE(MemoryPoolEnabledByEnvValue(nullptr));
    EXPECT_FALSE(MemoryPoolEnabledByEnvValue(""));
    EXPECT_FALSE(MemoryPoolEnabledByEnvValue("0"));
    EXPECT_FALSE(MemoryPoolEnabledByEnvValue("true"));
    EXPECT_FALSE(MemoryPoolEnabledByEnvValue("11"));
    EXPECT_FALSE(MemoryPoolEnabledByEnvValue("1 "));
    EXPECT_FALSE(MemoryPoolEnabledByEnvValue("01"));
}

TEST_F(TestMemorySpace, AllocBypassesPoolWhenDeviceQueryFails)
{
    // deviceId is part of the pool key, so a failed query must not fall back to a
    // placeholder id that would merge unrelated devices into one pool.
    g_failGetDevice = true;
    MOCKER_CPP(&aclrtMalloc).stubs().will(invoke(StubMalloc));
    MOCKER_CPP(&aclrtFree).stubs().will(invoke(StubFree));
    SetMemoryPoolEnabled(true);

    int8_t *start = nullptr;
    ASSERT_NO_THROW(AllocMemorySpace(MemorySpace::DEVICE, &start, 4096));
    EXPECT_NE(start, nullptr);
    EXPECT_EQ(g_directAllocCount, 1u);
    EXPECT_EQ(g_chunkAllocCount, 0u);

    EXPECT_NO_THROW(FreeMemorySpace(MemorySpace::DEVICE, start));
    EXPECT_EQ(g_aclFreeCount, 1u);
}

TEST_F(TestMemorySpace, ResetMemoryPoolsReleasesChunksBehindLivePointers)
{
    MOCKER_CPP(&aclrtMallocAlign32).stubs().will(invoke(StubMallocAlign32));
    MOCKER_CPP(&aclrtFree).stubs().will(invoke(StubFree));
    SetMemoryPoolEnabled(true);

    int8_t *leaked = nullptr;
    ASSERT_NO_THROW(AllocMemorySpace(MemorySpace::DEVICE, &leaked, 4096));
    ASSERT_EQ(g_chunkAllocCount, 1u);

    // A chunk with a live block is pinned, so shrinking cannot reclaim it.
    EXPECT_EQ(ReleaseAllMemoryPools(), 0u);
    EXPECT_EQ(g_aclFreeCount, 0u);

    // Resetting must release it anyway, otherwise the chunk would outlive the stubs and
    // later be handed to the real aclrtFree().
    ResetMemoryPoolsForTest();
    EXPECT_EQ(g_aclFreeCount, 1u);

    // The registry is empty again, so the next allocation starts from a fresh pool.
    int8_t *fresh = nullptr;
    ASSERT_NO_THROW(AllocMemorySpace(MemorySpace::DEVICE, &fresh, 4096));
    EXPECT_EQ(g_chunkAllocCount, 2u);
    EXPECT_NO_THROW(FreeMemorySpace(MemorySpace::DEVICE, fresh));
}

TEST_F(TestMemorySpace, ArenaAndMemorySpaceShareChunkAndReuseBlocks)
{
    MOCKER_CPP(&aclrtMallocAlign32).stubs().will(invoke(StubMallocAlign32));
    MOCKER_CPP(&aclrtFree).stubs().will(invoke(StubFree));
    SetMemoryPoolEnabled(true);

    void *anchor = nullptr;
    AllocMemorySpaceV(MemorySpace::DEVICE_HUGEPAGE, &anchor, 256);
    DeviceMemArena arena;
    void *owned = arena.Allocate(256);
    EXPECT_EQ(g_chunkAllocCount, 1u);
    EXPECT_EQ(g_lastPolicy, ACL_MEM_MALLOC_HUGE_FIRST);
    arena.Deallocate(owned);

    void *plain = nullptr;
    AllocMemorySpaceV(MemorySpace::DEVICE_HUGEPAGE, &plain, 256);
    EXPECT_EQ(plain, owned);
    EXPECT_EQ(g_chunkAllocCount, 1u);
    FreeMemorySpace(MemorySpace::DEVICE_HUGEPAGE, plain);
    void *reused = arena.Allocate(256);
    EXPECT_EQ(reused, plain);
    arena.Deallocate(reused);
    FreeMemorySpace(MemorySpace::DEVICE_HUGEPAGE, anchor);
}

TEST_F(TestMemorySpace, ArenaResetKeepsOtherArenaAndMemorySpaceAllocationsAlive)
{
    MOCKER_CPP(&aclrtMallocAlign32).stubs().will(invoke(StubMallocAlign32));
    MOCKER_CPP(&aclrtFree).stubs().will(invoke(StubFree));
    SetMemoryPoolEnabled(true);

    DeviceMemArena first;
    DeviceMemArena second;
    void *released = first.Allocate(256);
    auto *peer = static_cast<unsigned char *>(second.Allocate(256));
    unsigned char *plain = nullptr;
    AllocMemorySpace(MemorySpace::DEVICE_HUGEPAGE, &plain, 256);
    std::memset(peer, 0x5a, 256);
    std::memset(plain, 0xa5, 256);

    first.Reset();
    first.Reset();
    EXPECT_EQ(g_aclFreeCount, 0u);
    EXPECT_EQ(ReleaseAllMemoryPools(), 0u);
    void *reused = second.Allocate(256);
    EXPECT_EQ(reused, released);
    EXPECT_EQ(g_chunkAllocCount, 1u);
    for (size_t i = 0; i < 256; ++i)
    {
        EXPECT_EQ(peer[i], 0x5a);
        EXPECT_EQ(plain[i], 0xa5);
    }

    second.Reset();
    EXPECT_EQ(ReleaseAllMemoryPools(), 0u);
    FreeMemorySpace(MemorySpace::DEVICE_HUGEPAGE, plain);
    EXPECT_EQ(ReleaseAllMemoryPools(), 1u);
}

TEST_F(TestMemorySpace, ArenaDestructorReturnsBlocksToSharedPool)
{
    MOCKER_CPP(&aclrtMallocAlign32).stubs().will(invoke(StubMallocAlign32));
    MOCKER_CPP(&aclrtFree).stubs().will(invoke(StubFree));
    SetMemoryPoolEnabled(true);

    void *previous = nullptr;
    {
        DeviceMemArena arena;
        previous = arena.Allocate(1024);
    }
    EXPECT_EQ(g_aclFreeCount, 0u);
    void *plain = nullptr;
    AllocMemorySpaceV(MemorySpace::DEVICE_HUGEPAGE, &plain, 1024);
    EXPECT_EQ(plain, previous);
    EXPECT_EQ(g_chunkAllocCount, 1u);
    FreeMemorySpace(MemorySpace::DEVICE_HUGEPAGE, plain);
}

// An arena built while the pool was on keeps using it, and stays correct even after the
// switch goes off: FreeMemorySpace() routes by allocation origin, not by the current
// switch value, so the pooled block and the direct one each go back the way they came.
TEST_F(TestMemorySpace, PooledArenaSurvivesTheSwitchFlippingOff)
{
    MOCKER_CPP(&aclrtMallocAlign32).stubs().will(invoke(StubMallocAlign32));
    MOCKER_CPP(&aclrtMalloc).stubs().will(invoke(StubMalloc));
    MOCKER_CPP(&aclrtFree).stubs().will(invoke(StubFree));
    SetMemoryPoolEnabled(true);
    DeviceMemArena arena;
    EXPECT_TRUE(arena.IsPooled());

    void *pooled = arena.Allocate(1024);
    SetMemoryPoolEnabled(false);
    void *direct = arena.Allocate(1024);
    EXPECT_EQ(g_directAllocCount, 1u);
    EXPECT_EQ(g_chunkAllocCount, 1u);
    // Still no private slabs: a pooled arena never carves its own.
    EXPECT_EQ(arena.SlabCount(), 0u);

    arena.Deallocate(pooled);
    EXPECT_EQ(g_aclFreeCount, 0u);
    SetMemoryPoolEnabled(true);
    arena.Deallocate(direct);
    EXPECT_EQ(g_aclFreeCount, 1u);
    EXPECT_EQ(ReleaseAllMemoryPools(), 1u);
}

// The regression the slab backend exists for. With the pool off -- the default, and what
// CI runs -- AllocMemorySpaceV() is one aclrtMalloc per call, and an IVF index asks the
// arena for O(nlist) sub-allocations. Assert on the device-call counters, not on timing.
TEST_F(TestMemorySpace, ArenaSubAllocationsDoNotEachHitTheDeviceWhenPoolIsOff)
{
    constexpr size_t SLAB = 1024u * 1024u;
    constexpr size_t COUNT = 200;
    MOCKER_CPP(&aclrtMallocAlign32).stubs().will(invoke(StubMallocAlign32));
    MOCKER_CPP(&aclrtMalloc).stubs().will(invoke(StubMalloc));
    MOCKER_CPP(&aclrtFree).stubs().will(invoke(StubFree));
    ASSERT_FALSE(MemoryPoolEnabled());

    DeviceMemArena arena(SLAB);
    ASSERT_FALSE(arena.IsPooled());
    std::vector<void *> held;
    held.reserve(COUNT);
    for (size_t i = 0; i < COUNT; ++i)
    {
        void *p = arena.Allocate(256);
        ASSERT_NE(p, nullptr);
        std::memset(p, 0x5a, 256);
        held.push_back(p);
    }

    // One slab for 200 sub-allocations. Routed through AllocMemorySpaceV this read 200.
    EXPECT_EQ(g_chunkAllocCount, 1u);
    EXPECT_EQ(g_directAllocCount, 0u);
    EXPECT_EQ(arena.SlabCount(), 1u);
    EXPECT_EQ(arena.TotalReservedBytes(), SLAB);
    EXPECT_EQ(arena.AllocationCount(), COUNT);
    EXPECT_EQ(arena.AllocatedBytes(), COUNT * 256u);

    // Overlapping sub-allocations would show up as a torn pattern.
    for (void *p : held)
    {
        const auto *bytes = static_cast<const unsigned char *>(p);
        for (size_t k = 0; k < 256; ++k)
        {
            ASSERT_EQ(bytes[k], 0x5a);
        }
    }

    // Releasing returns ranges to the arena, not to the device.
    for (void *p : held)
    {
        arena.Deallocate(p);
    }
    EXPECT_EQ(arena.AllocationCount(), 0u);
    EXPECT_EQ(arena.AllocatedBytes(), 0u);
    EXPECT_EQ(g_aclFreeCount, 0u);
    EXPECT_EQ(arena.SlabCount(), 1u);
}

// The backend is latched in the constructor. Without that, a slab interior pointer would
// reach FreeMemorySpace(), where no pool claims it and aclrtFree() gets it instead --
// ASan calls that a bad-free.
TEST_F(TestMemorySpace, ArenaStaysOnSlabsAfterTheSwitchFlipsOn)
{
    MOCKER_CPP(&aclrtMallocAlign32).stubs().will(invoke(StubMallocAlign32));
    MOCKER_CPP(&aclrtMalloc).stubs().will(invoke(StubMalloc));
    MOCKER_CPP(&aclrtFree).stubs().will(invoke(StubFree));
    ASSERT_FALSE(MemoryPoolEnabled());

    DeviceMemArena arena(1024u * 1024u);
    ASSERT_FALSE(arena.IsPooled());
    void *before = arena.Allocate(256);
    ASSERT_NE(before, nullptr);

    SetMemoryPoolEnabled(true);
    void *after = arena.Allocate(256);
    ASSERT_NE(after, nullptr);
    EXPECT_FALSE(arena.IsPooled());
    EXPECT_EQ(g_chunkAllocCount, 1u);  // Second allocation took no pool chunk.
    EXPECT_EQ(g_directAllocCount, 0u);
    EXPECT_EQ(arena.SlabCount(), 1u);

    // The release path has to match the allocation path.
    arena.Deallocate(before);
    arena.Deallocate(after);
    EXPECT_EQ(g_aclFreeCount, 0u);
    EXPECT_EQ(arena.AllocatedBytes(), 0u);
}

TEST_F(TestMemorySpace, ArenaSlabReusesReleasedSubBlocksInsteadOfGrowing)
{
    MOCKER_CPP(&aclrtMallocAlign32).stubs().will(invoke(StubMallocAlign32));
    MOCKER_CPP(&aclrtFree).stubs().will(invoke(StubFree));
    ASSERT_FALSE(MemoryPoolEnabled());

    DeviceMemArena arena(1024u * 1024u);
    void *first = arena.Allocate(4096);
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(g_chunkAllocCount, 1u);

    arena.Deallocate(first);
    void *again = arena.Allocate(4096);
    EXPECT_EQ(again, first);  // Handed the range back rather than bumping past it.
    EXPECT_EQ(g_chunkAllocCount, 1u);

    // A realloc-style grow loop: hold the new block, drop the old, repeat. Coalescing has
    // to keep this inside one slab, which is what bounds DeviceVector growth.
    void *current = again;
    for (size_t size = 8192; size <= 128u * 1024u; size *= 2)
    {
        void *grown = arena.Allocate(size);
        ASSERT_NE(grown, nullptr);
        arena.Deallocate(current);
        current = grown;
    }
    EXPECT_EQ(arena.SlabCount(), 1u);
    EXPECT_EQ(g_chunkAllocCount, 1u);
    arena.Deallocate(current);
    EXPECT_EQ(arena.AllocatedBytes(), 0u);
}

TEST_F(TestMemorySpace, ArenaSlabHonoursCannAddressAlignmentAndRejectsForeignPointers)
{
    MOCKER_CPP(&aclrtMallocAlign32).stubs().will(invoke(StubMallocAlign32));
    MOCKER_CPP(&aclrtFree).stubs().will(invoke(StubFree));
    ASSERT_FALSE(MemoryPoolEnabled());

    DeviceMemArena arena(1024u * 1024u);
    EXPECT_EQ(arena.Allocate(0), nullptr);
    EXPECT_NO_THROW(arena.Deallocate(nullptr));

    void *mine = nullptr;
    for (size_t want : {1u, 17u, 63u, 64u, 65u, 1000u})
    {
        void *p = arena.Allocate(want);
        ASSERT_NE(p, nullptr);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % DeviceMemArena::kAddrAlign, 0u) << "want=" << want;
        mine = p;
    }

    int stackish = 0;
    EXPECT_THROW(arena.Deallocate(&stackish), std::exception);
    // An interior address is not an allocation either.
    EXPECT_THROW(arena.Deallocate(static_cast<unsigned char *>(mine) + 8), std::exception);
    arena.Deallocate(mine);
    EXPECT_THROW(arena.Deallocate(mine), std::exception);
}

TEST_F(TestMemorySpace, ArenaResetReturnsSlabsToAclAndStaysUsable)
{
    MOCKER_CPP(&aclrtMallocAlign32).stubs().will(invoke(StubMallocAlign32));
    MOCKER_CPP(&aclrtFree).stubs().will(invoke(StubFree));
    ASSERT_FALSE(MemoryPoolEnabled());

    {
        DeviceMemArena arena(1024u * 1024u);
        (void)arena.Allocate(256);
        (void)arena.Allocate(256);
        ASSERT_EQ(arena.SlabCount(), 1u);

        arena.Reset();
        arena.Reset();  // Idempotent.
        EXPECT_EQ(arena.SlabCount(), 0u);
        EXPECT_EQ(arena.TotalReservedBytes(), 0u);
        EXPECT_EQ(arena.AllocationCount(), 0u);
        EXPECT_EQ(arena.AllocatedBytes(), 0u);
        // The slab, and only the slab, went back to ACL. No pool was ever created.
        EXPECT_EQ(g_aclFreeCount, 1u);
        EXPECT_EQ(ReleaseAllMemoryPools(), 0u);

        void *p = arena.Allocate(256);  // Usable again, still on the slab backend.
        ASSERT_NE(p, nullptr);
        EXPECT_EQ(arena.SlabCount(), 1u);
    }  // ~DeviceMemArena releases the second slab.
    EXPECT_EQ(g_aclFreeCount, 2u);
    EXPECT_EQ(g_chunkAllocCount, 2u);
}

TEST_F(TestMemorySpace, ArenaTracksOnlyLiveRequestedBytesAndRejectsForeignPointers)
{
    MOCKER_CPP(&aclrtMallocAlign32).stubs().will(invoke(StubMallocAlign32));
    MOCKER_CPP(&aclrtFree).stubs().will(invoke(StubFree));
    SetMemoryPoolEnabled(true);
    DeviceMemArena arena;
    DeviceMemArena peer;
    EXPECT_EQ(arena.Allocate(0), nullptr);
    EXPECT_NO_THROW(arena.Deallocate(nullptr));
    EXPECT_EQ(arena.AllocationCount(), 0u);

    g_failChunkAllocs = 1;
    EXPECT_THROW(arena.Allocate(256), std::exception);
    EXPECT_EQ(arena.AllocationCount(), 0u);
    EXPECT_EQ(arena.AllocatedBytes(), 0u);
    void *first = arena.Allocate(33);
    void *second = arena.Allocate(129);
    EXPECT_EQ(arena.AllocationCount(), 2u);
    EXPECT_EQ(arena.AllocatedBytes(), 162u);
    EXPECT_THROW(peer.Deallocate(first), std::exception);
    arena.Deallocate(first);
    EXPECT_THROW(arena.Deallocate(first), std::exception);
    EXPECT_EQ(arena.AllocationCount(), 1u);
    EXPECT_EQ(arena.AllocatedBytes(), 129u);
    arena.Reset();
    EXPECT_THROW(arena.Deallocate(second), std::exception);
    EXPECT_EQ(arena.AllocationCount(), 0u);
    EXPECT_EQ(arena.AllocatedBytes(), 0u);
}

TEST_F(TestMemorySpace, ArenaAllocationRetriesAfterReleasingSharedIdleChunk)
{
    MOCKER_CPP(&aclrtMallocAlign32).stubs().will(invoke(StubMallocAlign32));
    MOCKER_CPP(&aclrtFree).stubs().will(invoke(StubFree));
    SetMemoryPoolEnabled(true);

    void *plain = nullptr;
    AllocMemorySpaceV(MemorySpace::DEVICE_HUGEPAGE, &plain, 256);
    FreeMemorySpace(MemorySpace::DEVICE_HUGEPAGE, plain);
    EXPECT_EQ(g_aclFreeCount, 0u);
    // Use the actual chunk size; earlier tests may have initialized the pool.
    const size_t largeSize = g_lastChunkSize + 2ull * 1024 * 1024;
    g_failChunkAllocs = 1;
    DeviceMemArena arena;
    void *large = nullptr;
    EXPECT_NO_THROW(large = arena.Allocate(largeSize));
    EXPECT_NE(large, nullptr);
    EXPECT_EQ(g_chunkAllocCount, 3u);  // Initial allocation, failed growth, retry.
    EXPECT_EQ(g_aclFreeCount, 1u);
    arena.Deallocate(large);
}

TEST_F(TestMemorySpace, AllocRetriesAfterReleasingIdleChunkOfOtherSpacePool)
{
    MOCKER_CPP(&aclrtMallocAlign32).stubs().will(invoke(StubMallocAlign32));
    MOCKER_CPP(&aclrtFree).stubs().will(invoke(StubFree));
    SetMemoryPoolEnabled(true);

    // Leave an idle chunk cached in the hugepage pool.
    void *cached = nullptr;
    ASSERT_NO_THROW(AllocMemorySpaceV(MemorySpace::DEVICE_HUGEPAGE, &cached, 256));
    ASSERT_NO_THROW(FreeMemorySpace(MemorySpace::DEVICE_HUGEPAGE, cached));
    EXPECT_EQ(g_aclFreeCount, 0u);

    // The normal pool has nothing to reclaim on its own.
    g_failChunkAllocs = 1;
    void *p = nullptr;
    ASSERT_NO_THROW(AllocMemorySpaceV(MemorySpace::DEVICE, &p, 256));
    EXPECT_NE(p, nullptr);
    EXPECT_EQ(g_chunkAllocCount, 3u);  // Hugepage chunk, failed growth, retry.
    EXPECT_EQ(g_aclFreeCount, 1u);     // The hugepage idle chunk was released.
    FreeMemorySpace(MemorySpace::DEVICE, p);
}
TEST_F(TestMemorySpace, ShutdownDropsTheChunksReleaseAllMemoryPoolsHasToKeep)
{
    MOCKER_CPP(&aclrtMallocAlign32).stubs().will(invoke(StubMallocAlign32));
    MOCKER_CPP(&aclrtFree).stubs().will(invoke(StubFree));
    SetMemoryPoolEnabled(true);

    static_assert(noexcept(ShutdownMemoryPools()), "shutdown must never throw across the ACL boundary");

    void *p = nullptr;
    ASSERT_NO_THROW(AllocMemorySpaceV(MemorySpace::DEVICE, &p, 4096));
    ASSERT_NE(p, nullptr);

    // A live chunk cannot be shrunk away; that is exactly why a finalize-time hook
    // that only shrinks would leave stale device memory behind.
    EXPECT_EQ(ReleaseAllMemoryPools(), 0u);
    EXPECT_EQ(g_aclFreeCount, 0u);

    EXPECT_EQ(ShutdownMemoryPools(), 1u);
    EXPECT_EQ(g_aclFreeCount, 1u);
}

TEST_F(TestMemorySpace, ShutdownLeavesNoChunkForASecondAclInitToCarveFrom)
{
    MOCKER_CPP(&aclrtMallocAlign32).stubs().will(invoke(StubMallocAlign32));
    MOCKER_CPP(&aclrtFree).stubs().will(invoke(StubFree));
    SetMemoryPoolEnabled(true);

    void *p = nullptr;
    ASSERT_NO_THROW(AllocMemorySpaceV(MemorySpace::DEVICE, &p, 4096));
    ASSERT_NO_THROW(FreeMemorySpace(MemorySpace::DEVICE, p));
    ASSERT_EQ(g_chunkAllocCount, 1u);
    EXPECT_EQ(g_aclFreeCount, 0u);  // Idle chunk is cached, not freed.

    ASSERT_EQ(ShutdownMemoryPools(), 1u);
    EXPECT_EQ(g_aclFreeCount, 1u);

    // A second aclInit must not inherit a chunk backed by the finalized context.
    void *q = nullptr;
    ASSERT_NO_THROW(AllocMemorySpaceV(MemorySpace::DEVICE, &q, 4096));
    EXPECT_EQ(g_chunkAllocCount, 2u);
    ASSERT_NO_THROW(FreeMemorySpace(MemorySpace::DEVICE, q));
}

TEST_F(TestMemorySpace, ShutdownCoversEveryDeviceAndSpaceAndIsIdempotent)
{
    MOCKER_CPP(&aclrtMallocAlign32).stubs().will(invoke(StubMallocAlign32));
    MOCKER_CPP(&aclrtFree).stubs().will(invoke(StubFree));
    SetMemoryPoolEnabled(true);

    EXPECT_EQ(ShutdownMemoryPools(), 0u);  // Empty registry.

    void *a = nullptr;
    void *b = nullptr;
    void *c = nullptr;
    ASSERT_NO_THROW(AllocMemorySpaceV(MemorySpace::DEVICE, &a, 4096));
    ASSERT_NO_THROW(AllocMemorySpaceV(MemorySpace::DEVICE_HUGEPAGE, &b, 4096));
    g_fakeDevice = 1;
    ASSERT_NO_THROW(AllocMemorySpaceV(MemorySpace::DEVICE, &c, 4096));
    g_fakeDevice = 0;

    EXPECT_EQ(ShutdownMemoryPools(), 3u);
    EXPECT_EQ(g_aclFreeCount, 3u);

    // Idempotent: nothing left to destroy, and no double free.
    EXPECT_EQ(ShutdownMemoryPools(), 0u);
    EXPECT_EQ(g_aclFreeCount, 3u);
}

TEST_F(TestMemorySpace, DirectPointerStillRoutesToAclAfterShutdown)
{
    MOCKER_CPP(&aclrtMallocAlign32).stubs().will(invoke(StubMallocAlign32));
    MOCKER_CPP(&aclrtMalloc).stubs().will(invoke(StubMalloc));
    MOCKER_CPP(&aclrtFree).stubs().will(invoke(StubFree));

    // Allocated while the pool is off: this pointer is owned by ACL, not by any pool.
    void *direct = nullptr;
    ASSERT_NO_THROW(AllocMemorySpaceV(MemorySpace::DEVICE, &direct, 4096));
    ASSERT_EQ(g_directAllocCount, 1u);

    SetMemoryPoolEnabled(true);
    void *pooled = nullptr;
    ASSERT_NO_THROW(AllocMemorySpaceV(MemorySpace::DEVICE, &pooled, 4096));
    ASSERT_NO_THROW(FreeMemorySpace(MemorySpace::DEVICE, pooled));

    ASSERT_EQ(ShutdownMemoryPools(), 1u);

    // Shutdown must unpublish the pool snapshot before destroying the pools, otherwise
    // this lookup would touch a destroyed pool's mutex.
    EXPECT_NO_THROW(FreeMemorySpace(MemorySpace::DEVICE, direct));
    EXPECT_EQ(g_aclFreeCount, 2u);  // One chunk plus the direct pointer.
}

TEST_F(TestMemorySpace, ChunkSizeOverrideIsValidatedBeforeItIsScaledToBytes)
{
    MOCKER_CPP(&aclrtMallocAlign32).stubs().will(invoke(StubMallocAlign32));
    MOCKER_CPP(&aclrtFree).stubs().will(invoke(StubFree));
    SetMemoryPoolEnabled(true);

    struct Case
    {
        const char *value;
        bool accepted;
    };
    const std::vector<Case> cases = {
        {"8", true},                      // Plain decimal, well under the default.
        {"17592186044416", false},        // 2^44: chunkMb * 1MB wraps to 0.
        {"17592186044417", false},        // 2^44 + 1: wraps to a 1 MB chunk.
        {"18446744073709551616", false},  // Past unsigned long long; strtoull saturates.
        {"-1", false},                    // strtoull would wrap this to SIZE_MAX.
        {" 4", false},                    // strtoull would skip the leading space.
        {"4junk", false},                 // strtoull would stop at the junk.
        {"0", false},                     // A zero-byte chunk cannot back anything.
    };

    const size_t defaultChunkBytes = 128ull * 1024 * 1024;
    for (const auto &c : cases)
    {
        // Each case needs a fresh pool: the chunk size is latched at first use.
        ResetMemoryPoolsForTest();
        g_chunkAllocCount = 0;
        g_lastChunkSize = 0;
        g_aclFreeCount = 0;
        {
            ScopedEnv override("ASCENDFAISS_MEM_POOL_CHUNK_MB", c.value);
            void *p = nullptr;
            ASSERT_NO_THROW(AllocMemorySpaceV(MemorySpace::DEVICE, &p, 4096)) << "value=" << c.value;
            ASSERT_NE(p, nullptr) << "value=" << c.value;
            ASSERT_EQ(g_chunkAllocCount, 1u) << "value=" << c.value;
            if (c.accepted)
            {
                EXPECT_LT(g_lastChunkSize, defaultChunkBytes) << "value=" << c.value;
            }
            else
            {
                // Rejected values must fall back to the default, not to a wrapped size.
                EXPECT_GE(g_lastChunkSize, defaultChunkBytes) << "value=" << c.value;
            }
            ASSERT_NO_THROW(FreeMemorySpace(MemorySpace::DEVICE, p));
        }
    }
}

}  // namespace ascend
