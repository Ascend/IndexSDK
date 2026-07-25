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
    size_t total = 0;
    for (const auto &slab : slabs_)
    {
        total += slab.capacity;
    }
    return total;
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
        offset = AlignUp(fresh.used, kAddrAlign);
        ASCEND_THROW_IF_NOT(offset + carve <= fresh.capacity);
        void *result = static_cast<uint8_t *>(fresh.ptr) + offset;
        fresh.used = offset + carve;
        return result;
    }

    void *result = static_cast<uint8_t *>(slab.ptr) + offset;
    slab.used = offset + carve;
    return result;
}

void DeviceMemArena::Reset()
{
    for (auto &slab : slabs_)
    {
        if (slab.ptr != nullptr)
        {
            (void)aclrtFree(slab.ptr);
            slab.ptr = nullptr;
        }
    }
    slabs_.clear();
}

}  // namespace ascend
