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

#include "ascenddaemon/utils/MemorySpace.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <utility>

#include "acl/acl.h"
#include "ascenddaemon/utils/MemDebug.h"
#include "ascenddaemon/utils/MemoryPool.h"
#include "common/utils/AscendAssert.h"
#include "common/utils/LogUtils.h"
#include "common/utils/SocUtils.h"

namespace ascend
{
namespace
{
constexpr const char *MEM_POOL_ENV = "ASCENDFAISS_MEM_POOL";
constexpr const char *MEM_POOL_CHUNK_MB_ENV = "ASCENDFAISS_MEM_POOL_CHUNK_MB";
constexpr const char *MEM_POOL_MAX_FREE_CHUNKS_ENV = "ASCENDFAISS_MEM_POOL_MAX_FREE_CHUNKS";

constexpr size_t ONE_MB = 1024ULL * 1024;
constexpr size_t DEFAULT_CHUNK_MB = 128;
constexpr size_t DEFAULT_MAX_FREE_CHUNKS = 1;
constexpr size_t UPSTREAM_ALIGN = 4ULL * 1024;

// One pool per (deviceId, space); two spaces means this covers 16 devices.
constexpr size_t MAX_TRACKED_POOLS = 32;

// Ceiling for ASCENDFAISS_MEM_POOL_CHUNK_MB. PoolForLocked() multiplies the value by
// ONE_MB, and a product that wraps size_t is worse than a loud failure: 2^44 wraps to
// 0 and aborts pool creation, while 2^44 + 128 wraps back to exactly the 128 MB
// default and is then indistinguishable from it in the log.
constexpr size_t MAX_CHUNK_MB = SIZE_MAX / ONE_MB;

// Parses a bare decimal override. Empty, signed, whitespace-padded, trailing-garbage,
// out-of-range and above-maxValue inputs all fall back with a warning.
//
// The digits-only scan is the actual guard, not decoration: std::strtoull() skips
// leading whitespace, accepts a '-' sign and silently wraps it, so " -1" would
// otherwise parse as SIZE_MAX, and it stops at trailing garbage instead of rejecting
// it. ERANGE is still checked on top because callers that leave maxValue at SIZE_MAX
// cannot otherwise tell a saturated result from a deliberate one.
size_t EnvSize(const char *name, size_t fallback, size_t maxValue = SIZE_MAX)
{
    const char *raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0')
    {
        return fallback;
    }
    for (const char *c = raw; *c != '\0'; ++c)
    {
        if (*c < '0' || *c > '9')
        {
            APP_LOG_WARNING("[MemorySpace] ignore invalid %s=%s, fallback to %zu", name, raw, fallback);
            return fallback;
        }
    }

    errno = 0;
    const unsigned long long parsed = std::strtoull(raw, nullptr, 10);
    if (errno == ERANGE || parsed > static_cast<unsigned long long>(maxValue))
    {
        APP_LOG_WARNING("[MemorySpace] ignore out-of-range %s=%s (max %zu), fallback to %zu", name, raw, maxValue,
                        fallback);
        return fallback;
    }
    return static_cast<size_t>(parsed);
}

bool ReadPoolEnabledEnv() { return MemoryPoolEnabledByEnvValue(std::getenv(MEM_POOL_ENV)); }

std::atomic<bool> &PoolEnabledFlag()
{
    static std::atomic<bool> enabled(ReadPoolEnabledEnv());
    return enabled;
}

aclrtMemMallocPolicy PolicyOf(MemorySpace space)
{
    return space == MemorySpace::DEVICE_HUGEPAGE ? ACL_MEM_MALLOC_HUGE_FIRST : ACL_MEM_MALLOC_NORMAL_ONLY;
}

void ValidateSpace(MemorySpace space)
{
    if (space != MemorySpace::DEVICE && space != MemorySpace::DEVICE_HUGEPAGE)
    {
        ASCEND_THROW_MSG("Unsupported memoryspace type\n");
    }
}

// The device id is part of the pool key, so a failed query must not be papered over
// with a placeholder: that would merge unrelated devices into one pool and hand out
// memory belonging to the wrong device.
bool TryCurrentDevice(int32_t *deviceId)
{
    *deviceId = -1;
    const aclError err = aclrtGetDevice(deviceId);
    if (err != ACL_ERROR_NONE || *deviceId < 0)
    {
        APP_LOG_WARNING("[MemorySpace] aclrtGetDevice failed (error %d, deviceId %d), bypassing the memory pool",
                        static_cast<int>(err), *deviceId);
        return false;
    }
    return true;
}

void AllocWithDebug(MemorySpace space, void **const p, size_t size, aclrtMemMallocPolicy policy)
{
    int deviceId = -1;
    (void)aclrtGetDevice(&deviceId);

    size_t freeBefore = 0;
    size_t totalBefore = 0;
    if (MemDebugEnabled())
    {
        (void)QueryHbm(&freeBefore, &totalBefore);
        (void)NoteAlloc(space, size, deviceId, freeBefore, totalBefore);
    }

    aclError err = aclrtMalloc(p, size, policy);
    if (err != ACL_ERROR_NONE)
    {
        if (MemDebugEnabled())
        {
            DumpAllocRingOnFailure(space, size, static_cast<int>(err), deviceId);
        }
        ASCEND_THROW_FMT(
            "failed to aclrtMalloc %zu bytes (error %d) space=%s device=%d "
            "HBM_free_before=%zu HBM_total_before=%zu\n",
            size, static_cast<int>(err), MemorySpaceName(space), deviceId, freeBefore, totalBefore);
    }
}

// Debug accounting tracks chunks, not suballocations.
class DeviceUpstream : public IUpstream
{
   public:
    DeviceUpstream(int32_t deviceId, MemorySpace space) : deviceId_(deviceId), space_(space) {}

    void *alloc(size_t size) override
    {
        size = (size + UPSTREAM_ALIGN - 1) / UPSTREAM_ALIGN * UPSTREAM_ALIGN;

        size_t freeBefore = 0;
        size_t totalBefore = 0;
        if (MemDebugEnabled())
        {
            (void)QueryHbm(&freeBefore, &totalBefore);
            (void)NoteAlloc(space_, size, deviceId_, freeBefore, totalBefore);
        }

        void *p = nullptr;
        aclError err = aclrtMallocAlign32(&p, size, PolicyOf(space_));
        if (err != ACL_ERROR_NONE)
        {
            if (MemDebugEnabled())
            {
                DumpAllocRingOnFailure(space_, size, static_cast<int>(err), deviceId_);
            }
            // MemoryPool handles reclamation and retry.
            APP_LOG_ERROR(
                "[MemorySpace] pool chunk alloc %zu bytes failed (error %d) space=%s device=%d "
                "HBM_free_before=%zu HBM_total_before=%zu",
                size, static_cast<int>(err), MemorySpaceName(space_), deviceId_, freeBefore, totalBefore);
            return nullptr;
        }
        return p;
    }

    void free(void *p) override { (void)aclrtFree(p); }

   private:
    int32_t deviceId_;
    MemorySpace space_;
};

struct PoolEntry
{
    std::unique_ptr<DeviceUpstream> upstream;
    std::unique_ptr<MemoryPool> pool;
};

// Separate devices and ACL allocation policies.
using PoolKey = std::pair<int32_t, int>;

// Keep registries alive to avoid ACL calls after aclFinalize().
std::map<PoolKey, PoolEntry> &Pools()
{
    static auto *pools = new std::map<PoolKey, PoolEntry>();
    return *pools;
}

// Lock order: registry, then pool.
std::mutex &RegistryMutex()
{
    static auto *mtx = new std::mutex();
    return *mtx;
}

// A pool is created once per key and then lives until ResetMemoryPoolsForTest(), so a
// published pointer stays valid. FreeMemorySpace can therefore ask each pool whether it
// owns a pointer without taking RegistryMutex, which keeps the global lock off a path
// that may call aclrtFree() while trimming idle chunks. It also removes the need for a
// pointer -> pool side table, which used to duplicate MemoryPool's own bookkeeping and
// could silently drop an entry when its insert failed.
std::array<std::atomic<MemoryPool *>, MAX_TRACKED_POOLS> &PoolSnapshot()
{
    static auto *snapshot = new std::array<std::atomic<MemoryPool *>, MAX_TRACKED_POOLS>();
    return *snapshot;
}

std::atomic<size_t> &PoolSnapshotSize()
{
    static std::atomic<size_t> size(0);
    return size;
}

// Requires RegistryMutex(). Also forces both snapshot statics to initialize, so that
// PublishPoolLocked() cannot run their allocating initializers.
bool CanPublishPoolLocked() { return PoolSnapshot().size() > PoolSnapshotSize().load(std::memory_order_relaxed); }

// Requires RegistryMutex() and a preceding CanPublishPoolLocked(). Must not fail: an
// unpublished pool would hand out pointers FreeMemorySpace could not route back.
void PublishPoolLocked(MemoryPool *pool) noexcept
{
    const size_t next = PoolSnapshotSize().load(std::memory_order_relaxed);
    PoolSnapshot()[next].store(pool, std::memory_order_relaxed);
    // Releasing the size publishes the pointer store above to readers that acquire it.
    PoolSnapshotSize().store(next + 1, std::memory_order_release);
}

// Requires RegistryMutex().
MemoryPool *PoolForLocked(int32_t deviceId, MemorySpace space)
{
    const PoolKey key(deviceId, static_cast<int>(space));
    auto it = Pools().find(key);
    if (it != Pools().end())
    {
        return it->second.pool.get();
    }

    // Checked before the pool exists so that publishing it afterwards cannot fail.
    ASCEND_THROW_IF_NOT_FMT(CanPublishPoolLocked(),
                            "MemoryPool registry is full (%zu pools), cannot create one for device %d space %s\n",
                            MAX_TRACKED_POOLS, deviceId, MemorySpaceName(space));

    size_t chunkMb = EnvSize(MEM_POOL_CHUNK_MB_ENV, DEFAULT_CHUNK_MB, MAX_CHUNK_MB);
    if (chunkMb == 0)
    {
        APP_LOG_WARNING("[MemorySpace] %s=0 is invalid, fallback to %zu", MEM_POOL_CHUNK_MB_ENV, DEFAULT_CHUNK_MB);
        chunkMb = DEFAULT_CHUNK_MB;
    }
    const size_t maxFreeChunks = EnvSize(MEM_POOL_MAX_FREE_CHUNKS_ENV, DEFAULT_MAX_FREE_CHUNKS);

    PoolEntry entry;
    entry.upstream.reset(new DeviceUpstream(deviceId, space));
    entry.pool.reset(new MemoryPool());
    aclError err = entry.pool->init(entry.upstream.get(), chunkMb * ONE_MB, maxFreeChunks);
    ASCEND_THROW_IF_NOT_FMT(err == ACL_SUCCESS,
                            "failed to init MemoryPool (error %d) space=%s device=%d chunkMB=%zu maxFreeChunks=%zu\n",
                            static_cast<int>(err), MemorySpaceName(space), deviceId, chunkMb, maxFreeChunks);

    APP_LOG_INFO("[MemorySpace] create MemoryPool space=%s device=%d chunkMB=%zu maxFreeChunks=%zu",
                 MemorySpaceName(space), deviceId, chunkMb, maxFreeChunks);
    it = Pools().emplace(key, std::move(entry)).first;
    MemoryPool *pool = it->second.pool.get();
    PublishPoolLocked(pool);
    return pool;
}

// Requires RegistryMutex(). Returns the number of idle chunks released. A null exclude
// shrinks every pool of the device.
size_t ShrinkDevicePoolsLocked(int32_t deviceId, const MemoryPool *exclude)
{
    size_t released = 0;
    for (auto &kv : Pools())
    {
        if (kv.first.first != deviceId || kv.second.pool.get() == exclude)
        {
            continue;
        }
        released += kv.second.pool->shrink();
    }
    return released;
}

// Requires RegistryMutex(). Returns the number of pools destroyed.
//
// Unpublishes every pool before destroying it: a concurrent FreeMemorySpace() reads
// the snapshot without the registry lock, and must not be able to reach a pool that is
// going away. After the size store below it falls through to aclrtFree(), which is the
// right destination once the owning pool is gone.
size_t DestroyAllPoolsLocked()
{
    PoolSnapshotSize().store(0, std::memory_order_release);
    for (auto &slot : PoolSnapshot())
    {
        slot.store(nullptr, std::memory_order_relaxed);
    }
    const size_t destroyed = Pools().size();
    // ~MemoryPool() releases every chunk through the upstream, which PoolEntry keeps
    // alive until the pool is gone.
    Pools().clear();
    return destroyed;
}
}  // namespace

void AllocMemorySpaceV(MemorySpace space, void **const p, size_t size)
{
    ValidateSpace(space);

    if (!MemoryPoolEnabled())
    {
        AllocWithDebug(space, p, size, PolicyOf(space));
        return;
    }

    int32_t deviceId = -1;
    if (!TryCurrentDevice(&deviceId))
    {
        AllocWithDebug(space, p, size, PolicyOf(space));
        return;
    }

    MemoryPool *pool = nullptr;
    {
        std::lock_guard<std::mutex> lock(RegistryMutex());
        pool = PoolForLocked(deviceId, space);
    }

    // Pools are stable and synchronize allocations internally.
    aclError err = pool->allocate(size, p);
    if (err != ACL_SUCCESS)
    {
        // The failing pool already dropped its own idle chunks; other pools on the
        // same device may still cache device memory.
        size_t releasedElsewhere = 0;
        {
            std::lock_guard<std::mutex> lock(RegistryMutex());
            releasedElsewhere = ShrinkDevicePoolsLocked(deviceId, pool);
        }
        if (releasedElsewhere > 0)
        {
            APP_LOG_WARNING("[MemorySpace] released %zu idle chunks from other pools on device %d, retry %zu bytes",
                            releasedElsewhere, deviceId, size);
            err = pool->allocate(size, p);
        }
    }
    if (err != ACL_SUCCESS)
    {
        const MemoryPool::Stat stat = pool->stat();
        size_t freeBytes = 0;
        size_t totalBytes = 0;
        if (MemDebugEnabled())
        {
            (void)QueryHbm(&freeBytes, &totalBytes);
            DumpAllocRingOnFailure(space, size, static_cast<int>(err), deviceId);
        }
        ASCEND_THROW_FMT(
            "failed to allocate %zu bytes from MemoryPool (error %d) space=%s device=%d "
            "reserved=%zu inUse=%zu largestFree=%zu numChunks=%zu HBM_free=%zu HBM_total=%zu\n",
            size, static_cast<int>(err), MemorySpaceName(space), deviceId, stat.reserved, stat.inUse, stat.largestFree,
            stat.numChunks, freeBytes, totalBytes);
    }
}

void FreeMemorySpace(MemorySpace, void *p) noexcept
{
    if (p == nullptr)
    {
        return;
    }

    // Route by allocation origin rather than by the current device or the current value
    // of the pool switch: ask every pool whether it owns the pointer. MemoryPool already
    // tracks its live blocks, so this needs no side table and cannot disagree with it.
    //
    // The scan is linear in the number of pools, which is two per device the process has
    // actually touched, and each probe is one uncontended mutex. That is cheaper than the
    // extra global lock and hash insert that keeping a pointer -> pool table cost on
    // every single allocation.
    const size_t poolCount = PoolSnapshotSize().load(std::memory_order_acquire);
    for (size_t i = 0; i < poolCount; ++i)
    {
        MemoryPool *pool = PoolSnapshot()[i].load(std::memory_order_relaxed);
        if (pool != nullptr && pool->tryDeallocate(p))
        {
            return;
        }
    }

    // No pool claims it, so it came straight from aclrtMalloc: either the pool was
    // disabled at the time, or the device query failed and the alloc bypassed the pool.
    // Never pass a pooled interior pointer here; aclrtFree would reject or corrupt it.
    (void)aclrtFree(p);
}

bool MemoryPoolEnabledByEnvValue(const char *value)
{
    // Opt-in: the device-side binary links this file too and does not inherit the host
    // environment, so an opt-out switch would leave it with no way back to aclrtMalloc.
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

bool MemoryPoolEnabled() { return PoolEnabledFlag().load(std::memory_order_relaxed); }

void SetMemoryPoolEnabled(bool enabled) { PoolEnabledFlag().store(enabled, std::memory_order_relaxed); }

size_t ReleaseMemoryPoolsOfDevice(int32_t deviceId)
{
    std::lock_guard<std::mutex> lock(RegistryMutex());
    return ShrinkDevicePoolsLocked(deviceId, nullptr);
}

size_t ReleaseAllMemoryPools()
{
    std::lock_guard<std::mutex> lock(RegistryMutex());
    size_t released = 0;
    for (auto &kv : Pools())
    {
        released += kv.second.pool->shrink();
    }
    return released;
}

void ResetMemoryPoolsForTest()
{
    std::lock_guard<std::mutex> lock(RegistryMutex());
    (void)DestroyAllPoolsLocked();
}

size_t ShutdownMemoryPools() noexcept
{
    // Reached from ~SocUtils() and from an atexit() handler. Letting anything escape
    // there would terminate the process during shutdown, so the body is contained.
    try
    {
        std::lock_guard<std::mutex> lock(RegistryMutex());
        size_t liveBytes = 0;
        for (auto &kv : Pools())
        {
            liveBytes += kv.second.pool->stat().inUse;
        }
        if (liveBytes > 0)
        {
            // Not fatal: aclFinalize() invalidates this memory a moment later either
            // way. It does mean something outlived the index objects, which is worth
            // seeing in the log.
            APP_LOG_WARNING("[MemorySpace] destroying memory pools with %zu bytes still in use", liveBytes);
        }
        const size_t destroyed = DestroyAllPoolsLocked();
        if (destroyed > 0)
        {
            APP_LOG_INFO("[MemorySpace] destroyed %zu memory pools before ACL shutdown", destroyed);
        }
        return destroyed;
    }
    catch (...)
    {
        APP_LOG_ERROR("[MemorySpace] unexpected exception while destroying memory pools");
        return 0;
    }
}
}  // namespace ascend
