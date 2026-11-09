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

#ifndef ASCEND_SEGMENTED_DEVICE_VECTOR_INCLUDED
#define ASCEND_SEGMENTED_DEVICE_VECTOR_INCLUDED

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "ascenddaemon/utils/DeviceMemArena.h"
#include "ascenddaemon/utils/MemorySpace.h"
#include "common/utils/AscendAssert.h"

namespace ascend
{

// A contiguous device allocation partitioned into fixed-size slots. Owns the slot<->element mapping
// so callers address it in slots and never touch the elementsPerSlot arithmetic directly.
template <typename T>
class DeviceSegment
{
   public:
    // arena != nullptr carves this segment out of the shared arena instead of calling aclrtMalloc per
    // segment. With growthStep < slotsPerSegment the tail segment is reallocated once per growth step, so
    // filling one segment costs slotsPerSegment/growthStep allocations per vector; multiplied by nlist
    // vectors that is the O(nlist) aclrtMalloc storm DeviceMemArena exists to absorb.
    //
    // The segment holds a raw pointer into the arena, so the arena must outlive it. Callers that reset an
    // arena must drop every segment first (see DeviceMemArena::Reset).
    DeviceSegment(size_t slotCapacity, size_t elementsPerSlot, std::shared_ptr<DeviceMemArena> arena = nullptr,
                  MemorySpace space = MemorySpace::DEVICE_HUGEPAGE)
        : slotCapacity_(slotCapacity), elementsPerSlot_(elementsPerSlot), arena_(std::move(arena)), space_(space)
    {
        size_t elements = slotCapacity_ * elementsPerSlot_;
        if (elements != 0)
        {
            if (arena_ != nullptr)
            {
                data_ = static_cast<T*>(arena_->Allocate(elements * sizeof(T)));
                ASCEND_THROW_IF_NOT_MSG(data_ != nullptr, "DeviceMemArena Allocate returned null");
            }
            else
            {
                AllocMemorySpace(space_, &data_, elements * sizeof(T));
            }
        }
    }

    ~DeviceSegment()
    {
        if (data_ != nullptr)
        {
            if (arena_ != nullptr)
            {
                arena_->Deallocate(data_);
            }
            else
            {
                FreeMemorySpace(space_, data_);
            }
        }
    }

    DeviceSegment(const DeviceSegment&) = delete;
    DeviceSegment& operator=(const DeviceSegment&) = delete;

    T* data() const { return data_; }

    T* slotPtr(size_t localSlot) const { return data_ + localSlot * elementsPerSlot_; }

    size_t slotCapacity() const { return slotCapacity_; }

    size_t bytes() const { return slotCapacity_ * elementsPerSlot_ * sizeof(T); }

   private:
    T* data_{nullptr};
    size_t slotCapacity_{0};
    size_t elementsPerSlot_{0};
    std::shared_ptr<DeviceMemArena> arena_;
    MemorySpace space_{MemorySpace::DEVICE_HUGEPAGE};
};

// A logically-contiguous vector stored as a list of fixed-size device segments, so it can grow
// without reallocating/copying the whole buffer and so snapshots can share unchanged segments.
//
// Terminology and units (read before touching the addressing math). The container is index-agnostic.
//   element  - one T of underlying storage.
//   slot     - one logical entry the caller stores and indexes as a unit; occupies elementsPerSlot elements.
//   segment  - one contiguous device allocation (a DeviceSegment<T>) holding up to slotsPerSegment slots.
//
//   elementsPerSlot - elements that make up one slot.
//   slotsPerSegment - slots per full segment; segment boundaries fall on multiples of it.
//   growthStep      - tail-segment allocation granularity, in slots.
//   length          - number of slots currently in use.
//
// Only the last (tail) segment may be partial: it is allocated in growthStep increments up to
// slotsPerSegment, after which a new segment is started. Every earlier segment is full (holds and
// is allocated for slotsPerSegment slots).
//
// Everything at this level is counted in slots; DeviceSegment owns the slot<->element mapping, so
// elementsPerSlot only appears when a segment is built.
//
// Addressing: slot s lives in segment (s / slotsPerSegment), at local slot (s % slotsPerSegment)
// within that segment.
template <typename T>
class SegmentedDeviceVector
{
   public:
    using Segment = DeviceSegment<T>;
    using SegmentPtr = std::shared_ptr<Segment>;

    // Segment geometry is fixed at construction and never mutated afterwards: reconfiguring after
    // segments are allocated would make already-allocated segments inconsistent with the new geometry
    // (slot()/segmentLength()/ensureCapacity() all rely on it). slotsPerSegment is the logical slot
    // count per full segment; elementsPerSlot is how many T make up one logical slot.
    // arena, when set, backs every segment this vector allocates (see DeviceSegment).
    SegmentedDeviceVector(size_t slotsPerSegment, size_t growthStep, size_t elementsPerSlot = 1,
                          std::shared_ptr<DeviceMemArena> arena = nullptr)
        : slotsPerSegment_(slotsPerSegment),
          growthStep_(growthStep),
          elementsPerSlot_(elementsPerSlot),
          arena_(std::move(arena))
    {
        ASCEND_THROW_IF_NOT_MSG(slotsPerSegment_ != 0, "slotsPerSegment must be > 0");
        ASCEND_THROW_IF_NOT_MSG(growthStep_ != 0, "growthStep must be > 0");
        ASCEND_THROW_IF_NOT_MSG(elementsPerSlot_ != 0, "elementsPerSlot must be > 0");
    }

    // Default copy/move are intentional: they shallow-copy the segment shared_ptrs so
    // snapshots structurally share segments (copy-on-write). ensureCapacity/adoptClonedSegment
    // clone a segment into a fresh shared_ptr before mutating it. Do not delete these.

    size_t length() const { return length_; }

    size_t elementsPerSlot() const { return elementsPerSlot_; }

    // Set the number of in-use slots (the logical length). Must be backed by already-allocated segments
    // (call ensureCapacity with at least newLength first), otherwise slot()/segmentLength() would index
    // past the storage.
    void commitLength(size_t newLength)
    {
        ASCEND_THROW_IF_NOT_FMT(newLength <= capacity_, "commitLength %zu exceeds allocated capacity %zu", newLength,
                                capacity_);
        length_ = newLength;
    }

    // Number of slots currently allocated (addressable). Only the tail segment may hold fewer than
    // slotsPerSegment slots, so this is not simply segmentCount * slotsPerSegment.
    size_t capacity() const { return capacity_; }

    size_t segmentCount() const { return segments_.size(); }

    size_t segmentLength(size_t segmentId) const
    {
        if (segmentId + 1 < segments_.size())
        {
            return slotsPerSegment_;
        }
        if (segments_.empty())
        {
            return 0;
        }
        size_t tail = length_ % slotsPerSegment_;
        return tail == 0 && length_ != 0 ? slotsPerSegment_ : tail;
    }

    T* slot(size_t index) const
    {
        ASCEND_THROW_IF_NOT_FMT(index < capacity_, "slot index %zu out of range (capacity %zu)", index, capacity_);
        return segments_.at(index / slotsPerSegment_)->slotPtr(index % slotsPerSegment_);
    }

    T* segmentData(size_t segmentId) const
    {
        ASCEND_THROW_IF_NOT_FMT(segmentId < segments_.size(), "segmentId %zu out of range (segmentCount %zu)",
                                segmentId, segments_.size());
        return segments_.at(segmentId)->data();
    }

    size_t segmentBytes(size_t segmentId) const
    {
        ASCEND_THROW_IF_NOT_FMT(segmentId < segments_.size(), "segmentId %zu out of range (segmentCount %zu)",
                                segmentId, segments_.size());
        return segments_.at(segmentId)->bytes();
    }

    size_t allocatedBytes() const { return capacity_ * elementsPerSlot_ * sizeof(T); }

    // Bytes held by segments this vector uniquely owns (not shared with another snapshot).
    size_t uniquelyOwnedBytes() const
    {
        size_t bytes = 0;
        for (const auto& segment : segments_)
        {
            if (segment.use_count() == 1)
            {
                bytes += segment->bytes();
            }
        }
        return bytes;
    }

    // Grow (or shrink) storage so logical slots [0, requiredLength) are addressable. Tail segment
    // is copy-on-write cloned when it must grow, preserving already-written data.
    void ensureCapacity(size_t requiredLength, aclrtStream stream)
    {
        size_t requiredSegments = (requiredLength + slotsPerSegment_ - 1) / slotsPerSegment_;
        segments_.resize(requiredSegments);
        for (size_t segmentId = 0; segmentId < requiredSegments; ++segmentId)
        {
            size_t required =
                segmentId + 1 < requiredSegments ? slotsPerSegment_ : requiredLength - segmentId * slotsPerSegment_;
            size_t capacity = segmentId + 1 < requiredSegments ? slotsPerSegment_ : nextTailCapacity(required);
            auto& segment = segments_[segmentId];
            if (segment == nullptr)
            {
                segment = std::make_shared<Segment>(capacity, elementsPerSlot_, arena_);
            }
            else if (segment->slotCapacity() < capacity)
            {
                segment = cloneSegment(segmentId, capacity, stream);
            }
        }
        refreshCapacity();
    }

    // Drop trailing segments after a shrink; caller commits the new length separately.
    void resizeSegments(size_t count)
    {
        segments_.resize(count);
        refreshCapacity();
    }

    // Replace segment[segmentId] with a copy-on-write clone of src's same-indexed segment.
    void adoptClonedSegment(size_t segmentId, const SegmentedDeviceVector& src, size_t capacity, aclrtStream stream)
    {
        segments_[segmentId] = src.cloneSegment(segmentId, capacity, stream);
        refreshCapacity();
    }

   private:
    // Recompute the cached allocated slot count. Non-tail segments are always full (slotsPerSegment
    // slots); only the tail may hold fewer, so read its actual allocation.
    void refreshCapacity()
    {
        capacity_ =
            segments_.empty() ? 0 : (segments_.size() - 1) * slotsPerSegment_ + segments_.back()->slotCapacity();
    }

    size_t nextTailCapacity(size_t required) const
    {
        return std::min(slotsPerSegment_, ((required + growthStep_ - 1) / growthStep_) * growthStep_);
    }

    // Copy-on-write clone of segment[segmentId], grown to `capacity` slots.
    //
    // The copy length is bounded by the SOURCE segment's own allocation, never by segmentLength(): callers
    // reach this from ensureCapacity, which has already resized segments_, so a segment that used to be the
    // tail is no longer last and segmentLength() would report a full slotsPerSegment_ for it regardless of
    // how little was actually allocated (the tail grows in growthStep_ increments). Reading that many slots
    // would run off the end of the source allocation -- into a neighbouring arena block or unmapped device
    // memory -- so the allocation is the hard bound and the logical length only trims it further.
    //
    // The copy is synchronous on purpose. The caller replaces segments_[segmentId] with this clone as soon
    // as we return, dropping the last reference to the source in the normal single-writer growth path, and
    // DeviceSegment's destructor hands that memory straight back to the shared arena. An async copy would
    // still be in flight at that point, so the arena could re-issue the block to another list while it is
    // being read. ArenaDevMemStrategy copies synchronously before freeing for the same reason.
    SegmentPtr cloneSegment(size_t segmentId, size_t capacity, aclrtStream stream) const
    {
        const auto& src = segments_.at(segmentId);
        auto clone = std::make_shared<Segment>(capacity, elementsPerSlot_, arena_);
        size_t copySlots = std::min({src->slotCapacity(), segmentLength(segmentId), capacity});
        size_t copyCount = copySlots * elementsPerSlot_;
        if (copyCount != 0)
        {
            (void)stream;  // synchronous copy; see the note above on why this must not be stream-ordered
            auto ret = aclrtMemcpy(clone->data(), clone->bytes(), src->data(), copyCount * sizeof(T),
                                   ACL_MEMCPY_DEVICE_TO_DEVICE);
            ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy error %d", ret);
        }
        return clone;
    }

   private:
    std::vector<SegmentPtr> segments_;
    size_t length_{0};
    size_t capacity_{0};
    size_t slotsPerSegment_;
    size_t growthStep_;
    size_t elementsPerSlot_;
    std::shared_ptr<DeviceMemArena> arena_;
};

}  // namespace ascend

#endif  // ASCEND_SEGMENTED_DEVICE_VECTOR_INCLUDED
