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
 * -------------------------------------------------------------------------
 */

#include <gtest/gtest.h>

#include "ascenddaemon/utils/AscendUtils.h"
#include "ascenddaemon/utils/DeviceMemArena.h"
#include "ascenddaemon/utils/SegmentedDeviceVector.h"
#include "common/utils/AscendException.h"

namespace ascend
{

TEST(SegmentedDeviceVector, ConstructorRejectsZeroGeometry)
{
    EXPECT_THROW((SegmentedDeviceVector<uint8_t>(0, 4)), AscendException);     // slotsPerSegment == 0
    EXPECT_THROW((SegmentedDeviceVector<uint8_t>(8, 0)), AscendException);     // growthStep == 0
    EXPECT_THROW((SegmentedDeviceVector<uint8_t>(8, 4, 0)), AscendException);  // elementsPerSlot == 0
}

TEST(SegmentedDeviceVector, CommitLengthRejectsLengthBeyondAllocatedTail)
{
    SegmentedDeviceVector<uint8_t> vector(8, 4);
    vector.ensureCapacity(3, nullptr);  // tail rounded up to growthStep => 4 slots allocated, not 8

    EXPECT_THROW(vector.commitLength(5), AscendException);  // only 4 slots are actually allocated
    EXPECT_NO_THROW(vector.commitLength(4));                // exactly fits the allocated tail
}

TEST(SegmentedDeviceVector, SlotRejectsIndexBeyondCapacity)
{
    SegmentedDeviceVector<uint8_t> vector(8, 4);
    vector.ensureCapacity(3, nullptr);  // capacity == 4 (partial tail), not 8

    EXPECT_NO_THROW(vector.slot(3));                  // in range
    EXPECT_THROW(vector.slot(4), AscendException);    // at capacity, out of range
    EXPECT_THROW(vector.slot(100), AscendException);  // far past the last segment
}

TEST(SegmentedDeviceVector, SegmentAccessorsRejectBadSegmentId)
{
    SegmentedDeviceVector<uint8_t> vector(8, 4);
    vector.ensureCapacity(3, nullptr);  // one segment

    EXPECT_NO_THROW(vector.segmentData(0));
    EXPECT_THROW(vector.segmentData(1), AscendException);
    EXPECT_THROW(vector.segmentBytes(1), AscendException);
}

TEST(SegmentedDeviceVector, RecordsAllocatedCapacity)
{
    SegmentedDeviceVector<uint8_t> vector(8, 4);
    EXPECT_EQ(vector.capacity(), 0U);

    vector.ensureCapacity(3, nullptr);
    EXPECT_EQ(vector.capacity(), 4U);  // partial tail: nextTailCapacity(3) == 4

    vector.ensureCapacity(10, nullptr);
    EXPECT_EQ(vector.capacity(), 12U);  // full segment (8) + tail nextTailCapacity(2) == 4
}

TEST(SegmentedDeviceVector, ConstructorAppliesElementsPerSlot)
{
    SegmentedDeviceVector<uint8_t> vector(8, 4, 2);  // 2 bytes per logical slot
    vector.ensureCapacity(1, nullptr);
    EXPECT_EQ(vector.allocatedBytes(), 8U);  // tail 4 slots * 2 elementsPerSlot * 1 byte
}

TEST(SegmentedDeviceVector, ComputesLogicalSegmentLengths)
{
    SegmentedDeviceVector<uint8_t> vector(32768, 4096);
    vector.ensureCapacity(32768 + 17, nullptr);
    vector.commitLength(32768 + 17);

    EXPECT_EQ(vector.length(), 32768U + 17U);
    EXPECT_EQ(vector.segmentCount(), 2U);
    EXPECT_EQ(vector.segmentLength(0), 32768U);
    EXPECT_EQ(vector.segmentLength(1), 17U);
}

TEST(SegmentedDeviceVector, GrowsTailByGrowthStep)
{
    SegmentedDeviceVector<uint8_t> vector(32768, 4096);  // elementsPerSlot defaults to 1

    vector.ensureCapacity(1, nullptr);
    EXPECT_EQ(vector.allocatedBytes(), 4096U);

    vector.ensureCapacity(4097, nullptr);
    EXPECT_EQ(vector.allocatedBytes(), 8192U);

    vector.ensureCapacity(32768, nullptr);
    EXPECT_EQ(vector.allocatedBytes(), 32768U);
}

// Growing past a segment boundary promotes the old partial tail to a full segment, which means cloning it.
// The clone must copy only what the SOURCE had allocated: ensureCapacity resizes segments_ before cloning,
// so the promoted segment is no longer last and segmentLength() reports a full slotsPerSegment_ for it no
// matter how few slots the growthStep tail actually held. Sizing the copy from that would read past the
// source allocation. Asserting the payload (not just the pointer) is what makes the over-read visible.
TEST(SegmentedDeviceVector, CrossSegmentGrowthPreservesPayload)
{
    SegmentedDeviceVector<uint8_t> vector(8, 4);  // slotsPerSegment 8, growthStep 4
    vector.ensureCapacity(3, nullptr);
    vector.commitLength(3);
    EXPECT_EQ(vector.capacity(), 4U);  // partial tail: only 4 of the 8 slots exist

    for (uint8_t i = 0; i < 3; ++i)
    {
        *vector.slot(i) = static_cast<uint8_t>(0xA0 + i);
    }

    // newLen 10 > slotsPerSegment, so segment 0 is promoted to a full 8 slots and cloned from a source
    // that only ever had 4.
    vector.ensureCapacity(10, nullptr);
    vector.commitLength(10);
    EXPECT_EQ(vector.segmentCount(), 2U);

    for (uint8_t i = 0; i < 3; ++i)
    {
        EXPECT_EQ(*vector.slot(i), static_cast<uint8_t>(0xA0 + i)) << "slot " << static_cast<int>(i) << " lost data";
    }
}

// Same promotion, but with elementsPerSlot > 1 so an over-read would be scaled by the slot width -- this is
// the shape `codes` actually uses (elementsPerSlot == code bytes per vector).
TEST(SegmentedDeviceVector, CrossSegmentGrowthPreservesPayloadWithWideSlots)
{
    constexpr size_t kElementsPerSlot = 4;
    SegmentedDeviceVector<uint8_t> vector(8, 4, kElementsPerSlot);
    vector.ensureCapacity(2, nullptr);
    vector.commitLength(2);

    for (size_t s = 0; s < 2; ++s)
    {
        uint8_t *slot = vector.slot(s);
        for (size_t e = 0; e < kElementsPerSlot; ++e)
        {
            slot[e] = static_cast<uint8_t>(0x10 * (s + 1) + e);
        }
    }

    vector.ensureCapacity(9, nullptr);  // promotes segment 0 to full width
    vector.commitLength(9);

    for (size_t s = 0; s < 2; ++s)
    {
        const uint8_t *slot = vector.slot(s);
        for (size_t e = 0; e < kElementsPerSlot; ++e)
        {
            EXPECT_EQ(slot[e], static_cast<uint8_t>(0x10 * (s + 1) + e));
        }
    }
}

// A snapshot sharing the segment must keep seeing the pre-growth payload: the clone is what gets mutated,
// never the shared source.
TEST(SegmentedDeviceVector, CopyOnWriteGrowthLeavesSharedSnapshotIntact)
{
    SegmentedDeviceVector<uint8_t> vector(8, 4);
    vector.ensureCapacity(3, nullptr);
    vector.commitLength(3);
    for (uint8_t i = 0; i < 3; ++i)
    {
        *vector.slot(i) = static_cast<uint8_t>(0x50 + i);
    }

    SegmentedDeviceVector<uint8_t> shared = vector;

    vector.ensureCapacity(10, nullptr);
    vector.commitLength(10);
    *vector.slot(0) = 0xFF;  // mutate the clone

    EXPECT_EQ(*shared.slot(0), 0x50);  // snapshot untouched
    EXPECT_EQ(*shared.slot(1), 0x51);
    EXPECT_EQ(*shared.slot(2), 0x52);
}

TEST(SegmentedDeviceVector, CopyOnWriteGrowthClonesTailSegment)
{
    SegmentedDeviceVector<uint8_t> vector(8, 4);
    vector.ensureCapacity(3, nullptr);
    vector.commitLength(3);
    uint8_t *original = vector.segmentData(0);

    SegmentedDeviceVector<uint8_t> shared = vector;  // shares the underlying segment (COW)

    vector.ensureCapacity(6, nullptr);  // tail capacity 4 -> 8 forces a clone
    vector.commitLength(6);

    EXPECT_NE(vector.segmentData(0), original);  // grown vector cloned its tail segment
    EXPECT_EQ(shared.segmentData(0), original);  // the shared snapshot keeps the original segment
}

TEST(SegmentedDeviceVector, UniquelyOwnedBytesExcludesSharedSegments)
{
    SegmentedDeviceVector<uint8_t> vector(8, 4);
    vector.ensureCapacity(3, nullptr);
    vector.commitLength(3);

    EXPECT_EQ(vector.uniquelyOwnedBytes(), 4U);

    SegmentedDeviceVector<uint8_t> shared = vector;
    EXPECT_EQ(vector.uniquelyOwnedBytes(), 0U);
}

// When an arena is supplied, segments are carved from it instead of calling AllocMemorySpace per
// segment -- that is what keeps tail growth from issuing O(nlist) aclrtMalloc calls. Each growth step
// returns the replaced block, so filling a whole segment stays within one slab.
TEST(SegmentedDeviceVector, ArenaBackedSegmentsAllocateFromArena)
{
    constexpr size_t kSlabBytes = 128 * 1024;
    auto arena = std::make_shared<DeviceMemArena>(kSlabBytes);
    SegmentedDeviceVector<uint8_t> vector(32768, 4096, 1, arena);
    EXPECT_EQ(arena->TotalReservedBytes(), 0U);  // construction allocates nothing

    vector.ensureCapacity(1, nullptr);
    ASSERT_NE(vector.segmentData(0), nullptr);
    EXPECT_EQ(vector.allocatedBytes(), 4096U);
    EXPECT_EQ(arena->TotalReservedBytes(), kSlabBytes);  // came from the arena, not AllocMemorySpace

    vector.ensureCapacity(4097, nullptr);
    EXPECT_EQ(vector.allocatedBytes(), 8192U);
    vector.ensureCapacity(32768, nullptr);
    EXPECT_EQ(vector.allocatedBytes(), 32768U);
    EXPECT_EQ(arena->SlabCount(), 1U);
}

// A segment holds a raw pointer into the arena, so every segment must be dropped before the arena is
// reset. Resetting while a vector still holds segments would leave it pointing at freed slabs.
TEST(SegmentedDeviceVector, ArenaResetRequiresDroppingSegmentsFirst)
{
    auto arena = std::make_shared<DeviceMemArena>(4096);
    {
        SegmentedDeviceVector<uint8_t> vector(8, 4, 1, arena);
        vector.ensureCapacity(3, nullptr);
        EXPECT_GT(arena->TotalReservedBytes(), 0U);
    }
    EXPECT_NO_THROW(arena->Reset());
    EXPECT_EQ(arena->TotalReservedBytes(), 0U);
}

}  // namespace ascend

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    ascend::DeviceScope deviceScope;
    return RUN_ALL_TESTS();
}
