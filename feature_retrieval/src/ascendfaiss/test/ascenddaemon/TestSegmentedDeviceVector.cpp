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

#include <vector>

#include "ascenddaemon/utils/AscendUtils.h"
#include "ascenddaemon/utils/DeviceMemArena.h"
#include "ascenddaemon/utils/MemorySpace.h"
#include "common/utils/AscendException.h"

namespace ascend
{
namespace
{
// Only this standalone test executable's header instantiations use these wrappers. Successful
// operations still use the real allocator and synchronous ACL copy; no mock library is required.
struct SegmentFailure;
thread_local SegmentFailure* activeSegmentFailure = nullptr;

struct SegmentFailure
{
    SegmentFailure(size_t allocation, size_t copy) : failAllocation(allocation), failCopy(copy)
    {
        activeSegmentFailure = this;
    }
    ~SegmentFailure() { activeSegmentFailure = nullptr; }

    size_t failAllocation;
    size_t failCopy;
    size_t allocations{0};
    size_t copies{0};
    size_t copyBytes{0};
};

template <typename T>
void SegmentTestAllocMemorySpace(MemorySpace space, T** pointer, size_t bytes)
{
    if (activeSegmentFailure != nullptr && ++activeSegmentFailure->allocations == activeSegmentFailure->failAllocation)
    {
        ASCEND_THROW_MSG("injected segment allocation failure");
    }
    AllocMemorySpace(space, pointer, bytes);
}

aclError SegmentTestMemcpy(void* destination, size_t destinationBytes, const void* source, size_t bytes,
                           aclrtMemcpyKind kind)
{
    if (activeSegmentFailure != nullptr)
    {
        activeSegmentFailure->copyBytes = bytes;
        if (++activeSegmentFailure->copies == activeSegmentFailure->failCopy)
        {
            return static_cast<aclError>(1);
        }
    }
    return aclrtMemcpy(destination, destinationBytes, source, bytes, kind);
}
}  // namespace
}  // namespace ascend

// Scope fault injection to the code under test, not gtest, DeviceScope, or payload transfers.
#define AllocMemorySpace SegmentTestAllocMemorySpace
#define aclrtMemcpy SegmentTestMemcpy
#include "ascenddaemon/utils/SegmentedDeviceVector.h"
#undef aclrtMemcpy
#undef AllocMemorySpace

namespace ascend
{

TEST(SegmentedDeviceVector, InitialAllocationFailureLeavesEmptyVectorRetryable)
{
    for (size_t allocation = 1; allocation <= 3; ++allocation)
    {
        SCOPED_TRACE(allocation);
        SegmentedDeviceVector<uint8_t> vector(8, 4);
        {
            SegmentFailure failure(allocation, 0);
            EXPECT_THROW(vector.ensureCapacity(18, nullptr), AscendException);
            EXPECT_EQ(failure.allocations, allocation);
        }
        EXPECT_EQ(vector.segmentCount(), 0U);
        EXPECT_EQ(vector.length(), 0U);
        EXPECT_EQ(vector.capacity(), 0U);
        EXPECT_EQ(vector.allocatedBytes(), 0U);

        EXPECT_NO_THROW(vector.ensureCapacity(18, nullptr));
        ASSERT_EQ(vector.segmentCount(), 3U);
        EXPECT_EQ(vector.length(), 0U);
        EXPECT_EQ(vector.capacity(), 20U);
        for (size_t segmentId = 0; segmentId < vector.segmentCount(); ++segmentId)
        {
            EXPECT_NE(vector.segmentData(segmentId), nullptr);
        }
        EXPECT_NO_THROW(vector.commitLength(18));
    }
}

TEST(SegmentedDeviceVector, FailedGrowthPreservesSharedStorageAndRetriesWithWideSlots)
{
    for (size_t elementsPerSlot : {1U, 4U})
    {
        for (size_t requiredLength : {6U, 18U})
        {
            // Tail growth allocates one clone. Cross-segment growth then allocates two new segments;
            // failing either later allocation must roll back even the already-copied clone.
            size_t allocationCount = requiredLength == 6 ? 1 : 3;
            for (size_t fault = 0; fault <= allocationCount; ++fault)
            {
                SCOPED_TRACE(elementsPerSlot);
                SCOPED_TRACE(requiredLength);
                SCOPED_TRACE(fault);
                SegmentedDeviceVector<uint8_t> vector(8, 4, elementsPerSlot);
                vector.ensureCapacity(3, nullptr);
                vector.commitLength(3);
                uint8_t* original = vector.segmentData(0);
                // Include the allocated fourth slot beyond logical length as a sentinel.
                std::vector<uint8_t> payload(4 * elementsPerSlot);
                for (size_t i = 0; i < payload.size(); ++i)
                {
                    payload[i] = static_cast<uint8_t>(0xA0 + i);
                }
                ASSERT_EQ(aclrtMemcpy(original, vector.segmentBytes(0), payload.data(), payload.size(),
                                      ACL_MEMCPY_HOST_TO_DEVICE),
                          ACL_SUCCESS);
                SegmentedDeviceVector<uint8_t> shared = vector;
                {
                    // fault == 0 injects the synchronous clone copy failure instead of allocation failure.
                    SegmentFailure failure(fault, fault == 0 ? 1 : 0);
                    EXPECT_THROW(vector.ensureCapacity(requiredLength, nullptr), AscendException);
                    EXPECT_EQ(failure.allocations, fault == 0 ? 1U : fault);
                    EXPECT_EQ(failure.copies, fault == 1 ? 0U : 1U);
                }
                std::vector<uint8_t> actual(payload.size());
                for (const auto* unchanged : {&vector, &shared})
                {
                    EXPECT_EQ(unchanged->segmentCount(), 1U);
                    EXPECT_EQ(unchanged->length(), 3U);
                    EXPECT_EQ(unchanged->capacity(), 4U);
                    EXPECT_EQ(unchanged->segmentLength(0), 3U);
                    EXPECT_EQ(unchanged->segmentData(0), original);
                    EXPECT_EQ(unchanged->segmentBytes(0), payload.size());
                    EXPECT_EQ(unchanged->allocatedBytes(), payload.size());
                    EXPECT_EQ(unchanged->uniquelyOwnedBytes(), 0U);
                    ASSERT_EQ(aclrtMemcpy(actual.data(), actual.size(), unchanged->segmentData(0), actual.size(),
                                          ACL_MEMCPY_DEVICE_TO_HOST),
                              ACL_SUCCESS);
                    EXPECT_EQ(actual, payload);
                }
                {
                    SegmentFailure observe(0, 0);
                    EXPECT_NO_THROW(vector.ensureCapacity(requiredLength, nullptr));
                    EXPECT_EQ(observe.allocations, allocationCount);
                    EXPECT_EQ(observe.copies, 1U);
                    // The resized target layout promotes the source to non-tail: copy all four allocated
                    // slots, not just three logical slots, while still respecting the allocation bound.
                    EXPECT_EQ(observe.copyBytes, (requiredLength == 6 ? 3U : 4U) * elementsPerSlot);
                }
                EXPECT_EQ(vector.length(), 3U);  // ensureCapacity never commits a new logical length
                EXPECT_EQ(vector.capacity(), requiredLength == 6 ? 8U : 20U);
                EXPECT_EQ(vector.segmentCount(), requiredLength == 6 ? 1U : 3U);
                EXPECT_NE(vector.segmentData(0), original);
                size_t preservedBytes = (requiredLength == 6 ? 3U : 4U) * elementsPerSlot;
                actual.resize(preservedBytes);
                ASSERT_EQ(aclrtMemcpy(actual.data(), actual.size(), vector.segmentData(0), actual.size(),
                                      ACL_MEMCPY_DEVICE_TO_HOST),
                          ACL_SUCCESS);
                EXPECT_EQ(actual, std::vector<uint8_t>(payload.begin(), payload.begin() + preservedBytes));
                EXPECT_EQ(shared.segmentCount(), 1U);
                EXPECT_EQ(shared.length(), 3U);
                EXPECT_EQ(shared.capacity(), 4U);
                EXPECT_EQ(shared.segmentData(0), original);
                actual.resize(payload.size());
                ASSERT_EQ(aclrtMemcpy(actual.data(), actual.size(), shared.segmentData(0), actual.size(),
                                      ACL_MEMCPY_DEVICE_TO_HOST),
                          ACL_SUCCESS);
                EXPECT_EQ(actual, payload);
                EXPECT_NO_THROW(vector.commitLength(requiredLength));
            }
        }
    }
}

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
// The copy must stay within the source allocation even though segmentLength() now reports a full segment.
// These payload checks verify preservation, not the absence of an over-read.
TEST(SegmentedDeviceVector, CrossSegmentGrowthPreservesPayload)
{
    SegmentedDeviceVector<uint8_t> vector(8, 4);  // slotsPerSegment 8, growthStep 4
    vector.ensureCapacity(3, nullptr);
    vector.commitLength(3);
    EXPECT_EQ(vector.capacity(), 4U);  // partial tail: only 4 of the 8 slots exist

    // The fourth allocated slot is beyond logical length and must survive promotion as well.
    const std::vector<uint8_t> payload{0xA0, 0xA1, 0xA2, 0xAF};
    ASSERT_EQ(aclrtMemcpy(vector.segmentData(0), vector.segmentBytes(0), payload.data(), payload.size(),
                          ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_SUCCESS);

    // newLen 10 > slotsPerSegment, so segment 0 is promoted to a full 8 slots and cloned from a source
    // that only ever had 4.
    vector.ensureCapacity(10, nullptr);
    EXPECT_EQ(vector.length(), 3U);
    vector.commitLength(10);
    EXPECT_EQ(vector.segmentCount(), 2U);

    std::vector<uint8_t> actual(payload.size());
    ASSERT_EQ(
        aclrtMemcpy(actual.data(), actual.size(), vector.segmentData(0), actual.size(), ACL_MEMCPY_DEVICE_TO_HOST),
        ACL_SUCCESS);
    EXPECT_EQ(actual, payload);
}

// The same promotion with multiple elements per slot checks that all valid slot payload is preserved.
TEST(SegmentedDeviceVector, CrossSegmentGrowthPreservesPayloadWithWideSlots)
{
    constexpr size_t kElementsPerSlot = 4;
    SegmentedDeviceVector<uint8_t> vector(8, 4, kElementsPerSlot);
    vector.ensureCapacity(2, nullptr);
    vector.commitLength(2);

    std::vector<uint8_t> payload(2 * kElementsPerSlot);
    for (size_t s = 0; s < 2; ++s)
    {
        for (size_t e = 0; e < kElementsPerSlot; ++e)
        {
            payload[s * kElementsPerSlot + e] = static_cast<uint8_t>(0x10 * (s + 1) + e);
        }
    }
    ASSERT_EQ(aclrtMemcpy(vector.segmentData(0), vector.segmentBytes(0), payload.data(), payload.size(),
                          ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_SUCCESS);

    vector.ensureCapacity(9, nullptr);  // promotes segment 0 to full width
    vector.commitLength(9);

    std::vector<uint8_t> actual(payload.size());
    ASSERT_EQ(
        aclrtMemcpy(actual.data(), actual.size(), vector.segmentData(0), actual.size(), ACL_MEMCPY_DEVICE_TO_HOST),
        ACL_SUCCESS);
    EXPECT_EQ(actual, payload);
}

// A snapshot sharing the segment must keep seeing the pre-growth payload: the clone is what gets mutated,
// never the shared source.
TEST(SegmentedDeviceVector, CopyOnWriteGrowthLeavesSharedSnapshotIntact)
{
    SegmentedDeviceVector<uint8_t> vector(8, 4);
    vector.ensureCapacity(3, nullptr);
    vector.commitLength(3);
    const std::vector<uint8_t> payload{0x50, 0x51, 0x52};
    ASSERT_EQ(aclrtMemcpy(vector.segmentData(0), vector.segmentBytes(0), payload.data(), payload.size(),
                          ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_SUCCESS);

    SegmentedDeviceVector<uint8_t> shared = vector;

    vector.ensureCapacity(10, nullptr);
    vector.commitLength(10);
    const uint8_t replacement = 0xFF;
    ASSERT_EQ(
        aclrtMemcpy(vector.slot(0), sizeof(replacement), &replacement, sizeof(replacement), ACL_MEMCPY_HOST_TO_DEVICE),
        ACL_SUCCESS);

    std::vector<uint8_t> actual(payload.size());
    ASSERT_EQ(
        aclrtMemcpy(actual.data(), actual.size(), shared.segmentData(0), actual.size(), ACL_MEMCPY_DEVICE_TO_HOST),
        ACL_SUCCESS);
    EXPECT_EQ(actual, payload);
}

TEST(SegmentedDeviceVector, CopyOnWriteGrowthClonesTailSegment)
{
    SegmentedDeviceVector<uint8_t> vector(8, 4);
    vector.ensureCapacity(3, nullptr);
    vector.commitLength(3);
    uint8_t* original = vector.segmentData(0);
    const std::vector<uint8_t> payload{0x70, 0x71, 0x72};
    ASSERT_EQ(aclrtMemcpy(original, vector.segmentBytes(0), payload.data(), payload.size(), ACL_MEMCPY_HOST_TO_DEVICE),
              ACL_SUCCESS);

    SegmentedDeviceVector<uint8_t> shared = vector;  // shares the underlying segment (COW)

    vector.ensureCapacity(6, nullptr);  // tail capacity 4 -> 8 forces a clone
    vector.commitLength(6);

    EXPECT_NE(vector.segmentData(0), original);  // grown vector cloned its tail segment
    EXPECT_EQ(shared.segmentData(0), original);  // the shared snapshot keeps the original segment
    EXPECT_EQ(vector.length(), 6U);
    EXPECT_EQ(vector.capacity(), 8U);
    EXPECT_EQ(shared.length(), 3U);
    EXPECT_EQ(shared.capacity(), 4U);

    std::vector<uint8_t> actual(payload.size());
    ASSERT_EQ(
        aclrtMemcpy(actual.data(), actual.size(), vector.segmentData(0), actual.size(), ACL_MEMCPY_DEVICE_TO_HOST),
        ACL_SUCCESS);
    EXPECT_EQ(actual, payload);
    ASSERT_EQ(
        aclrtMemcpy(actual.data(), actual.size(), shared.segmentData(0), actual.size(), ACL_MEMCPY_DEVICE_TO_HOST),
        ACL_SUCCESS);
    EXPECT_EQ(actual, payload);
}

TEST(SegmentedDeviceVector, AdoptClonedSegmentSeparatesFullAndPartialTailWithinCapacity)
{
    constexpr size_t kSlotsPerSegment = 8;
    constexpr size_t kGrowthStep = 4;
    constexpr size_t kElementsPerSlot = 3;
    constexpr size_t kLength = 11;
    constexpr size_t kCapacity = 12;
    std::vector<uint8_t> payload(kLength * kElementsPerSlot);
    for (size_t i = 0; i < payload.size(); ++i)
    {
        payload[i] = static_cast<uint8_t>(0x20 + i);
    }

    for (size_t target = 0; target < 2; ++target)
    {
        SCOPED_TRACE(target);
        SegmentedDeviceVector<uint8_t> source(kSlotsPerSegment, kGrowthStep, kElementsPerSlot);
        source.ensureCapacity(kLength, nullptr);
        source.commitLength(kLength);
        ASSERT_EQ(source.segmentCount(), 2U);
        ASSERT_EQ(source.length(), kLength);
        ASSERT_EQ(source.capacity(), kCapacity);
        ASSERT_EQ(source.segmentLength(0), kSlotsPerSegment);
        ASSERT_EQ(source.segmentLength(1), kLength - kSlotsPerSegment);
        for (size_t segmentId = 0; segmentId < source.segmentCount(); ++segmentId)
        {
            size_t offset = segmentId * kSlotsPerSegment * kElementsPerSlot;
            size_t bytes = source.segmentLength(segmentId) * kElementsPerSlot;
            ASSERT_EQ(aclrtMemcpy(source.segmentData(segmentId), source.segmentBytes(segmentId),
                                  payload.data() + offset, bytes, ACL_MEMCPY_HOST_TO_DEVICE),
                      ACL_SUCCESS);
        }

        SegmentedDeviceVector<uint8_t> destination = source;
        for (size_t segmentId = 0; segmentId < source.segmentCount(); ++segmentId)
        {
            ASSERT_EQ(destination.segmentData(segmentId), source.segmentData(segmentId));
        }

        size_t targetCapacity = target == 0 ? kSlotsPerSegment : kGrowthStep;
        destination.adoptClonedSegment(target, source, targetCapacity, nullptr);
        ASSERT_EQ(destination.length(), kLength);
        ASSERT_EQ(destination.capacity(), kCapacity);
        ASSERT_EQ(destination.segmentCount(), 2U);
        EXPECT_NE(destination.segmentData(target), source.segmentData(target));
        EXPECT_EQ(destination.segmentData(1 - target), source.segmentData(1 - target));
        EXPECT_EQ(destination.segmentBytes(target), source.segmentBytes(target));

        std::vector<uint8_t> actual(payload.size());
        for (size_t segmentId = 0; segmentId < destination.segmentCount(); ++segmentId)
        {
            size_t offset = segmentId * kSlotsPerSegment * kElementsPerSlot;
            size_t bytes = destination.segmentLength(segmentId) * kElementsPerSlot;
            ASSERT_EQ(aclrtMemcpy(actual.data() + offset, actual.size() - offset, destination.segmentData(segmentId),
                                  bytes, ACL_MEMCPY_DEVICE_TO_HOST),
                      ACL_SUCCESS);
        }
        EXPECT_EQ(actual, payload);

        size_t targetOffset = target * kSlotsPerSegment * kElementsPerSlot;
        std::vector<uint8_t> replacement(source.segmentLength(target) * kElementsPerSlot);
        std::vector<uint8_t> expected = payload;
        for (size_t i = 0; i < replacement.size(); ++i)
        {
            replacement[i] = static_cast<uint8_t>(0xC0 + i);
            expected[targetOffset + i] = replacement[i];
        }
        ASSERT_EQ(aclrtMemcpy(destination.segmentData(target), destination.segmentBytes(target), replacement.data(),
                              replacement.size(), ACL_MEMCPY_HOST_TO_DEVICE),
                  ACL_SUCCESS);

        for (const auto* vector : {&source, &destination})
        {
            EXPECT_EQ(vector->length(), kLength);
            EXPECT_EQ(vector->capacity(), kCapacity);
            EXPECT_EQ(vector->segmentCount(), 2U);
            EXPECT_EQ(vector->segmentLength(0), kSlotsPerSegment);
            EXPECT_EQ(vector->segmentLength(1), kLength - kSlotsPerSegment);
            for (size_t segmentId = 0; segmentId < vector->segmentCount(); ++segmentId)
            {
                size_t offset = segmentId * kSlotsPerSegment * kElementsPerSlot;
                size_t bytes = vector->segmentLength(segmentId) * kElementsPerSlot;
                ASSERT_EQ(aclrtMemcpy(actual.data() + offset, actual.size() - offset, vector->segmentData(segmentId),
                                      bytes, ACL_MEMCPY_DEVICE_TO_HOST),
                          ACL_SUCCESS);
            }
            EXPECT_EQ(actual, vector == &source ? payload : expected);
        }
    }
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

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);

    ascend::DeviceScope deviceScope;
    return RUN_ALL_TESTS();
}
