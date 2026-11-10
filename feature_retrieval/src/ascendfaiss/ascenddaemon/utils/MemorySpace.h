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

#ifndef ASCEND_MEMORYSPACE_INCLUDED
#define ASCEND_MEMORYSPACE_INCLUDED

#include <cstddef>
#include <cstdint>

namespace ascend
{
enum class MemorySpace
{
    DEVICE = 1,
    DEVICE_HUGEPAGE = 2,
};

void AllocMemorySpaceV(MemorySpace space, void **p, size_t size);

template <typename T>
inline void AllocMemorySpace(MemorySpace space, T **p, size_t size)
{
    AllocMemorySpaceV(space, reinterpret_cast<void **>(p), size);
}

// Runs on destructor paths, so it reports failures through the log instead of throwing.
// The space argument is ignored: pooled pointers are routed back to their owning pool
// and everything else goes to aclrtFree().
void FreeMemorySpace(MemorySpace space, void *p) noexcept;

// Opt-in via ASCENDFAISS_MEM_POOL=1 (read once). It stays opt-in while the pool is a
// PoC because MemorySpace.cpp is linked into the device-side binary too, whose process
// environment the host cannot set, so an opt-out switch would have no escape hatch there.
// Pools are shared per (deviceId, space), including DeviceMemArena.
// Per-pool defaults: ASCENDFAISS_MEM_POOL_CHUNK_MB=128, ASCENDFAISS_MEM_POOL_MAX_FREE_CHUNKS=1.
bool MemoryPoolEnabled();

// Affects new allocations only; existing pointers retain their release path.
void SetMemoryPoolEnabled(bool enabled);

// Returns true only for the exact string "1".
bool MemoryPoolEnabledByEnvValue(const char *value);

// Releases the fully idle chunks of one device and returns their count; live
// allocations stay valid. Call this while deviceId is the current ACL device and
// before resetting it, since it releases that device's memory through aclrtFree().
size_t ReleaseMemoryPoolsOfDevice(int32_t deviceId);

// Same, across every device. Only safe where any device's memory may be released
// under the current ACL context, such as process shutdown and tests. Prefer
// ReleaseMemoryPoolsOfDevice() when tearing a single device down.
size_t ReleaseAllMemoryPools();

// Destroys every pool and empties the registry, returning the number of pools
// destroyed. SocUtils calls this immediately before aclFinalize(); there is no other
// production caller, and nothing re-registers a pool except the next allocation.
//
// Unlike ReleaseAllMemoryPools() this also drops chunks that still back live
// allocations. That is deliberate: aclFinalize() invalidates every device address a
// moment later anyway, while leaving the chunks in the registry would let a second
// aclInit() in the same process carve sub-blocks out of memory the previous finalize
// already killed -- an illegal device address that surfaces as a stray ACL error or
// silently wrong data, not as a crash at the point of reuse.
//
// Only call this on a path that really does shut ACL down. Callers that skip
// aclFinalize() (MX_INDEX_FINALIZE=0, or a repeated init) must not call it: the live
// pointers it invalidates are still in use by the running process.
size_t ShutdownMemoryPools() noexcept;

// Destroys every pool, including chunks that still back live allocations, and empties
// the registry so the next allocation starts from a fresh pool.
//
// Test-only: all arena and MemorySpace pointers must already be dead and no other
// thread may be inside Alloc/FreeMemorySpace. Same teardown as ShutdownMemoryPools(),
// without the shutdown-time logging.
void ResetMemoryPoolsForTest();
}  // namespace ascend

#endif  // ASCEND_MEMORYSPACE_H
