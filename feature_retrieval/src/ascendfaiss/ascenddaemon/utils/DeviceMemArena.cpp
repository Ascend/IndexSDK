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

namespace ascend
{

DeviceMemArena::DeviceMemArena(size_t slabBytes) : slabBytes_(slabBytes > 0 ? slabBytes : kDefaultSlabBytes) {}

DeviceMemArena::~DeviceMemArena() { Reset(); }

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

size_t DeviceMemArena::SlabCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return slabs_.size();
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

void *DeviceMemArena::Allocate(size_t nbytes)
{
    if (nbytes == 0)
    {
        return nullptr;
    }

    const size_t carve = CarveBytes(nbytes);
    std::lock_guard<std::mutex> lock(mutex_);

    // Reuse released sub-blocks before growing the arena. Alignment padding is
    // owned by the allocation so Deallocate can restore and coalesce the whole
    // region, rather than leaving small holes between successive reallocations.
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
        allocations_.emplace(result, Allocation{blockBegin, allocationBytes, slabBegin});
        return result;
    }

    if (slabs_.empty() || (AlignUp(slabs_.back().used, kAddrAlign) + carve > slabs_.back().capacity))
    {
        Grow(carve);
    }

    Slab &slab = slabs_.back();
    size_t offset = AlignUp(slab.used, kAddrAlign);
    if (offset + carve > slab.capacity)
    {
        // Pathological: Grow should have ensured capacity; allocate a dedicated slab.
        Grow(carve);
        Slab &fresh = slabs_.back();
        const size_t oldUsed = fresh.used;
        offset = AlignUp(oldUsed, kAddrAlign);
        ASCEND_THROW_IF_NOT(offset + carve <= fresh.capacity);
        void *result = static_cast<uint8_t *>(fresh.ptr) + offset;
        fresh.used = offset + carve;
        const uintptr_t slabBegin = reinterpret_cast<uintptr_t>(fresh.ptr);
        allocations_.emplace(result, Allocation{slabBegin + oldUsed, fresh.used - oldUsed, slabBegin});
        return result;
    }

    const size_t oldUsed = slab.used;
    void *result = static_cast<uint8_t *>(slab.ptr) + offset;
    slab.used = offset + carve;
    const uintptr_t slabBegin = reinterpret_cast<uintptr_t>(slab.ptr);
    allocations_.emplace(result, Allocation{slabBegin + oldUsed, slab.used - oldUsed, slabBegin});
    return result;
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

    // If the released range reaches the bump tail, roll the tail back instead
    // of keeping it in the free map. Continue through preceding free ranges so
    // a grow-copy-free cycle can reuse the same slab without fragmentation.
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

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocations_.find(ptr);
    ASCEND_THROW_IF_NOT_MSG(it != allocations_.end(), "DeviceMemArena Deallocate received unknown pointer");
    AddFreeBlock(it->second.begin, it->second.bytes, it->second.slabBegin);
    allocations_.erase(it);
}

void DeviceMemArena::Reset()
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &slab : slabs_)
    {
        if (slab.ptr != nullptr)
        {
            (void)aclrtFree(slab.ptr);
            slab.ptr = nullptr;
        }
    }
    slabs_.clear();
    freeBlocks_.clear();
    allocations_.clear();
}

}  // namespace ascend
