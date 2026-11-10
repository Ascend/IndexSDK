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

#include "ascenddaemon/utils/DeviceMemArena.h"

#include <algorithm>
#include <iterator>

#include "acl/acl.h"
#include "ascenddaemon/utils/MemDebug.h"
#include "ascenddaemon/utils/MemorySpace.h"
#include "common/utils/AscendAssert.h"
#include "common/utils/LogUtils.h"

namespace ascend
{

// Latching the switch here is deliberate; see the header. Reading it per allocation
// would let a runtime flip strand slab pointers on the pool release path.
DeviceMemArena::DeviceMemArena(size_t slabBytes)
    : pooled_(MemoryPoolEnabled()), slabBytes_(slabBytes > 0 ? slabBytes : kDefaultSlabBytes)
{
}

DeviceMemArena::~DeviceMemArena()
{
    // Reset() only frees, and FreeMemorySpace() is noexcept, so nothing here should
    // throw. Guard anyway: escaping a destructor would turn a leak into a terminate.
    try
    {
        Reset();
    }
    catch (...)
    {
        APP_LOG_ERROR("[DeviceMemArena] unexpected exception while resetting in destructor");
    }
}

size_t DeviceMemArena::AlignUp(size_t value, size_t align)
{
    if (align == 0)
    {
        return value;
    }
    return (value + align - 1) / align * align;
}

size_t DeviceMemArena::CarveBytes(size_t nbytes)
{
    if (nbytes == 0)
    {
        return 0;
    }
    return AlignUp(nbytes, kSizeAlign) + kSizePad;
}

size_t DeviceMemArena::AllocatedBytes() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return allocatedBytes_;
}

size_t DeviceMemArena::AllocationCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return allocations_.size();
}

size_t DeviceMemArena::SlabCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return slabs_.size();
}

size_t DeviceMemArena::TotalReservedBytes() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    for (const auto &slab : slabs_)
    {
        total += slab.capacity;
    }
    return total;
}

bool DeviceMemArena::Record(void *ptr, const Allocation &allocation)
{
    // emplace() leaves an existing entry untouched, so a duplicate cannot corrupt the
    // recorded size. Still refuse to double-count it: the byte total feeds the MemDebug
    // diagnostics, and silently drifting there would discredit exactly the numbers
    // someone reads while chasing the pool bug that caused this.
    const auto inserted = allocations_.emplace(ptr, allocation);
    if (!inserted.second)
    {
        APP_LOG_ERROR(
            "[DeviceMemArena] %p is already registered as %zu bytes, refusing to re-register it as "
            "%zu; the backend handed out a block that is still live",
            ptr, inserted.first->second.requested, allocation.requested);
        return false;
    }
    allocatedBytes_ += allocation.requested;
    return true;
}

void *DeviceMemArena::Allocate(size_t nbytes)
{
    if (nbytes == 0)
    {
        return nullptr;
    }
    return pooled_ ? AllocatePooled(nbytes) : AllocateFromSlab(nbytes);
}

void *DeviceMemArena::AllocatePooled(size_t nbytes)
{
    // Deliberately outside mutex_. The backend can block for milliseconds growing a
    // pool chunk, and holding the arena lock across that would serialize every vector
    // sharing this arena behind one device malloc.
    void *ptr = nullptr;
    AllocMemorySpaceV(MemorySpace::DEVICE_HUGEPAGE, &ptr, nbytes);

    try
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // The map is diagnostics here: MemoryPool owns the block and tryDeallocate()
        // resolves it from the pool's own table, so a stale record cannot misroute a
        // release. Logging and carrying on keeps a bookkeeping glitch from taking down a
        // build that is otherwise fine.
        Allocation allocation;
        allocation.requested = nbytes;
        (void)Record(ptr, allocation);
    }
    catch (...)
    {
        FreeMemorySpace(MemorySpace::DEVICE_HUGEPAGE, ptr);
        throw;
    }
    return ptr;
}

// Carves out of private slabs, which is what the pool-off path used before the pool
// existed. Alignment padding is owned by the allocation so Deallocate can restore and
// coalesce the whole region, rather than leaving small holes between reallocations.
void *DeviceMemArena::AllocateFromSlab(size_t nbytes)
{
    const size_t carve = CarveBytes(nbytes);
    std::lock_guard<std::mutex> lock(mutex_);

    // Reuse released sub-blocks before growing the arena.
    for (auto it = freeBlocks_.begin(); it != freeBlocks_.end(); ++it)
    {
        const uintptr_t blockBegin = it->first;
        const size_t blockBytes = it->second.bytes;
        const uintptr_t slabBegin = it->second.slabBegin;
        const uintptr_t aligned = AlignUp(blockBegin, kAddrAlign);
        const size_t prefix = static_cast<size_t>(aligned - blockBegin);
        if (prefix + carve > blockBytes)
        {
            continue;
        }

        const size_t allocationBytes = prefix + carve;
        const size_t suffixBytes = blockBytes - allocationBytes;
        freeBlocks_.erase(it);
        if (suffixBytes > 0)
        {
            freeBlocks_.emplace(blockBegin + allocationBytes, FreeBlock{suffixBytes, slabBegin});
        }

        void *result = reinterpret_cast<void *>(aligned);
        Allocation allocation;
        allocation.requested = nbytes;
        allocation.begin = blockBegin;
        allocation.carved = allocationBytes;
        allocation.slabBegin = slabBegin;
        if (!Record(result, allocation))
        {
            // Unlike the pooled path, this map is load-bearing: Deallocate reads
            // begin/carved/slabBegin from it to coalesce. A live address coming back out
            // of the free list means the free list is corrupt, and handing the same range
            // to a second owner would silently fuse two vectors. Put it back and fail.
            AddFreeBlock(blockBegin, allocationBytes, slabBegin);
            ASCEND_THROW_MSG("DeviceMemArena carved an address that is already live");
        }
        return result;
    }

    if (slabs_.empty() || (AlignUp(slabs_.back().used, kAddrAlign) + carve > slabs_.back().capacity))
    {
        Grow(carve);
    }

    Slab &slab = slabs_.back();
    const size_t offset = AlignUp(slab.used, kAddrAlign);
    // Grow() reserves max(slabBytes_, carve), so the fresh tail always fits.
    ASCEND_THROW_IF_NOT_MSG(offset + carve <= slab.capacity, "DeviceMemArena slab tail is too small");

    const size_t oldUsed = slab.used;
    void *result = static_cast<uint8_t *>(slab.ptr) + offset;
    slab.used = offset + carve;
    const uintptr_t slabBegin = reinterpret_cast<uintptr_t>(slab.ptr);

    Allocation allocation;
    allocation.requested = nbytes;
    allocation.begin = slabBegin + oldUsed;
    allocation.carved = slab.used - oldUsed;
    allocation.slabBegin = slabBegin;
    if (!Record(result, allocation))
    {
        slab.used = oldUsed;
        ASCEND_THROW_MSG("DeviceMemArena bump-allocated an address that is already live");
    }
    return result;
}

void DeviceMemArena::Grow(size_t minCarveBytes)
{
    const size_t allocSize = std::max(slabBytes_, minCarveBytes);
    int deviceId = -1;
    (void)aclrtGetDevice(&deviceId);

    size_t freeBefore = 0;
    size_t totalBefore = 0;
    if (MemDebugEnabled())
    {
        (void)QueryHbm(&freeBefore, &totalBefore);
        (void)NoteAlloc(MemorySpace::DEVICE_HUGEPAGE, allocSize, deviceId, freeBefore, totalBefore);
        MemDebugPrintf("[MemDebug] DeviceMemArena::Grow slabBytes=%zu device=%d HBM_free=%zu\n", allocSize, deviceId,
                       freeBefore);
    }

    // Straight to ACL, not AllocMemorySpaceV: this backend exists precisely because the
    // pool is off, and a slab must not become a pool block that FreeMemorySpace would
    // later claim. Reset() pairs this with aclrtFree.
    void *ptr = nullptr;
    aclError err = aclrtMallocAlign32(&ptr, allocSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != ACL_ERROR_NONE)
    {
        if (MemDebugEnabled())
        {
            DumpAllocRingOnFailure(MemorySpace::DEVICE_HUGEPAGE, allocSize, static_cast<int>(err), deviceId);
        }
        ASCEND_THROW_FMT(
            "DeviceMemArena failed to aclrtMallocAlign32 %zu bytes (error %d) device=%d "
            "HBM_free_before=%zu HBM_total_before=%zu\n",
            allocSize, static_cast<int>(err), deviceId, freeBefore, totalBefore);
    }

    Slab slab;
    slab.ptr = ptr;
    slab.capacity = allocSize;
    slab.used = 0;
    slabs_.push_back(slab);
}

void DeviceMemArena::AddFreeBlock(uintptr_t begin, size_t bytes, uintptr_t slabBegin)
{
    if (bytes == 0)
    {
        return;
    }

    auto next = freeBlocks_.lower_bound(begin);
    if (next != freeBlocks_.begin())
    {
        auto prev = std::prev(next);
        if (prev->second.slabBegin == slabBegin && prev->first + prev->second.bytes == begin)
        {
            begin = prev->first;
            bytes += prev->second.bytes;
            freeBlocks_.erase(prev);
        }
    }

    next = freeBlocks_.lower_bound(begin);
    const bool canMergeWithNext =
        next != freeBlocks_.end() && next->second.slabBegin == slabBegin && begin + bytes == next->first;
    if (canMergeWithNext)
    {
        bytes += next->second.bytes;
        freeBlocks_.erase(next);
    }

    // If the released range reaches the bump tail, roll the tail back instead of keeping
    // it in the free map. Continue through preceding free ranges so a grow-copy-free cycle
    // can reuse the same slab without fragmentation.
    auto slabIt = std::find_if(slabs_.begin(), slabs_.end(), [slabBegin](const Slab &slab)
                               { return reinterpret_cast<uintptr_t>(slab.ptr) == slabBegin; });
    ASCEND_THROW_IF_NOT_MSG(slabIt != slabs_.end(), "DeviceMemArena free block does not belong to a slab");
    const bool isBumpSlab = slabIt == std::prev(slabs_.end());
    if (isBumpSlab && begin + bytes == slabBegin + slabIt->used)
    {
        slabIt->used = begin - slabBegin;
        while (true)
        {
            auto tail = freeBlocks_.lower_bound(slabBegin + slabIt->used);
            if (tail == freeBlocks_.begin())
            {
                break;
            }
            --tail;
            if (tail->second.slabBegin != slabBegin || tail->first + tail->second.bytes != slabBegin + slabIt->used)
            {
                break;
            }
            slabIt->used = tail->first - slabBegin;
            freeBlocks_.erase(tail);
        }
        return;
    }
    freeBlocks_.emplace(begin, FreeBlock{bytes, slabBegin});
}

void DeviceMemArena::Deallocate(void *ptr)
{
    if (ptr == nullptr)
    {
        return;
    }
    if (pooled_)
    {
        DeallocatePooled(ptr);
    }
    else
    {
        DeallocateFromSlab(ptr);
    }
}

void DeviceMemArena::DeallocatePooled(void *ptr)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = allocations_.find(ptr);
        ASCEND_THROW_IF_NOT_MSG(it != allocations_.end(), "DeviceMemArena Deallocate received unknown pointer");
        allocatedBytes_ -= it->second.requested;
        allocations_.erase(it);
    }
    // Dropped the record first, so the release below cannot be reached twice and runs
    // without the arena lock, matching Allocate().
    FreeMemorySpace(MemorySpace::DEVICE_HUGEPAGE, ptr);
}

void DeviceMemArena::DeallocateFromSlab(void *ptr)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocations_.find(ptr);
    ASCEND_THROW_IF_NOT_MSG(it != allocations_.end(), "DeviceMemArena Deallocate received unknown pointer");
    const Allocation allocation = it->second;
    allocatedBytes_ -= allocation.requested;
    allocations_.erase(it);
    // Returns the range to this arena, not to the device: slabs live until Reset().
    AddFreeBlock(allocation.begin, allocation.carved, allocation.slabBegin);
}

void DeviceMemArena::Reset()
{
    if (pooled_)
    {
        std::unordered_map<void *, Allocation> doomed;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            doomed.swap(allocations_);
            allocatedBytes_ = 0;
        }
        for (const auto &allocation : doomed)
        {
            FreeMemorySpace(MemorySpace::DEVICE_HUGEPAGE, allocation.first);
        }
        return;
    }

    std::vector<Slab> doomed;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        doomed.swap(slabs_);
        freeBlocks_.clear();
        allocations_.clear();
        allocatedBytes_ = 0;
    }
    // Outside the lock to match the pooled path: aclrtFree can block, and nothing below
    // touches arena state.
    for (const auto &slab : doomed)
    {
        if (slab.ptr != nullptr)
        {
            (void)aclrtFree(slab.ptr);
        }
    }
}

}  // namespace ascend
