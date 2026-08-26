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

// Allocates from a few large device HUGE slabs to avoid O(nlist) aclrtMalloc.
// Sub-blocks follow CANN secondary-allocation constraints: 64B VA align, size rounded to ALIGN_UP(len, 32) + 32.
class DeviceMemArena
{
   public:
    static constexpr size_t kDefaultSlabBytes = 256ULL * 1024ULL * 1024ULL;
    static constexpr size_t kAddrAlign = 64;
    static constexpr size_t kSizeAlign = 32;
    static constexpr size_t kSizePad = 32;

    explicit DeviceMemArena(size_t slabBytes = kDefaultSlabBytes);
    ~DeviceMemArena();

    DeviceMemArena(const DeviceMemArena &) = delete;
    DeviceMemArena &operator=(const DeviceMemArena &) = delete;

    // Returns a device pointer with at least nbytes usable (caller sees nbytes).
    void *Allocate(size_t nbytes);

    // Returns a sub-block to the arena. Adjacent free blocks are coalesced and
    // reused by later allocations. The pointer must come from Allocate().
    void Deallocate(void *ptr);

    // Frees all slabs. Callers must drop all pointers obtained from Allocate first.
    void Reset();

    size_t SlabCount() const;

    size_t TotalReservedBytes() const;

   private:
    struct Slab
    {
        void *ptr{nullptr};
        size_t capacity{0};
        size_t used{0};
    };

    struct Allocation
    {
        uintptr_t begin{0};
        size_t bytes{0};
        uintptr_t slabBegin{0};
    };

    struct FreeBlock
    {
        size_t bytes{0};
        uintptr_t slabBegin{0};
    };

    static size_t AlignUp(size_t value, size_t align);
    static size_t CarveBytes(size_t nbytes);
    void Grow(size_t minCarveBytes);
    void AddFreeBlock(uintptr_t begin, size_t bytes, uintptr_t slabBegin);

    size_t slabBytes_;
    std::vector<Slab> slabs_;
    // Address-ordered free blocks allow O(log n) neighbour coalescing.
    std::map<uintptr_t, FreeBlock> freeBlocks_;
    std::unordered_map<void *, Allocation> allocations_;
    mutable std::mutex mutex_;
};
}  // namespace ascend

#endif  // ASCEND_DEVICE_MEM_ARENA_INCLUDED
