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

#ifndef ASCEND_DEVICE_MEM_ARENA_INCLUDED
#define ASCEND_DEVICE_MEM_ARENA_INCLUDED

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ascend
{

// Sub-allocates device memory for the per-list vectors of an IVF index.
//
// Two backends, picked once at construction from the ASCENDFAISS_MEM_POOL switch:
//
//   pool on   every sub-allocation is an AllocMemorySpaceV() request, so the shared
//             BFC MemoryPool does the splitting and the coalescing.
//   pool off  private DEVICE_HUGEPAGE slabs carved here, i.e. the pre-pool behaviour.
//
// The slab backend is not dead weight. AllocMemorySpaceV() degrades to one aclrtMalloc
// per call while the pool is off, and an IVF index asks this class for O(nlist)
// sub-allocations, so routing the disabled path through it would turn a handful of slab
// mallocs into hundreds of thousands of device mallocs. Keeping the carver behind the
// switch is also what makes "pool on vs pool off" a fair measurement rather than a race
// against a strawman that never shipped.
//
// The backend is latched in the constructor instead of being read per call: callers may
// flip the switch at runtime, and a pointer carved out of a slab must never reach the
// pool release path, nor the reverse.
class DeviceMemArena
{
   public:
    static constexpr size_t kDefaultSlabBytes = 256ULL * 1024ULL * 1024ULL;
    // CANN secondary allocations: 64B address alignment, alignUp(len, 32) + 32 bytes.
    static constexpr size_t kAddrAlign = 64;
    static constexpr size_t kSizeAlign = 32;
    static constexpr size_t kSizePad = 32;

    explicit DeviceMemArena(size_t slabBytes = kDefaultSlabBytes);
    ~DeviceMemArena();

    DeviceMemArena(const DeviceMemArena &) = delete;
    DeviceMemArena &operator=(const DeviceMemArena &) = delete;

    // Zero bytes returns nullptr.
    void *Allocate(size_t nbytes);

    // Accepts only pointers owned by this arena, or nullptr.
    void Deallocate(void *ptr);

    // Invalidates this arena's allocations; shared chunks may remain cached.
    // Callers must stop using all arena pointers before reset.
    void Reset();

    size_t AllocationCount() const;

    // Live requested bytes, excluding pool padding and cached chunks.
    size_t AllocatedBytes() const;

    // True when this arena routes through the shared MemoryPool. Latched at construction.
    bool IsPooled() const { return pooled_; }

    // Private slabs held by this arena. Always 0 in pooled mode, where the shared pool
    // owns the chunks. This is the number the O(nlist) regression is visible in: it must
    // stay flat while sub-allocations pile up.
    size_t SlabCount() const;

    // Bytes reserved by the private slabs, including space not handed out yet.
    size_t TotalReservedBytes() const;

   private:
    struct Slab
    {
        void *ptr{nullptr};
        size_t capacity{0};
        size_t used{0};
    };

    struct FreeBlock
    {
        size_t bytes{0};
        uintptr_t slabBegin{0};
    };

    struct Allocation
    {
        // What the caller asked for. Drives AllocatedBytes() in both backends.
        size_t requested{0};
        // Slab backend only: the carved range to hand back on release.
        uintptr_t begin{0};
        size_t carved{0};
        uintptr_t slabBegin{0};
    };

    static size_t AlignUp(size_t value, size_t align);
    static size_t CarveBytes(size_t nbytes);

    void *AllocatePooled(size_t nbytes);
    void *AllocateFromSlab(size_t nbytes);
    void DeallocatePooled(void *ptr);
    void DeallocateFromSlab(void *ptr);
    // Both require mutex_.
    void Grow(size_t minCarveBytes);
    void AddFreeBlock(uintptr_t begin, size_t bytes, uintptr_t slabBegin);
    // Requires mutex_. Returns false and logs when ptr is already live.
    bool Record(void *ptr, const Allocation &allocation);

    const bool pooled_;
    const size_t slabBytes_;
    std::vector<Slab> slabs_;
    // Address-ordered free blocks allow O(log n) neighbour coalescing.
    std::map<uintptr_t, FreeBlock> freeBlocks_;
    std::unordered_map<void *, Allocation> allocations_;
    size_t allocatedBytes_{0};
    mutable std::mutex mutex_;
};
}  // namespace ascend

#endif  // ASCEND_DEVICE_MEM_ARENA_INCLUDED
