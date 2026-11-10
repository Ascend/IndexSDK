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

#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <limits>
#include <map>
#include <new>

#include "ascend/impl/LocationTable.h"

namespace
{
thread_local int64_t allocationsUntilFailure = -1;
thread_local size_t allocationCalls = 0;

class AllocationFailure
{
   public:
    explicit AllocationFailure(int64_t after) { allocationsUntilFailure = after; }
    ~AllocationFailure() { allocationsUntilFailure = -1; }
    AllocationFailure(const AllocationFailure &) = delete;
    AllocationFailure &operator=(const AllocationFailure &) = delete;
};
}  // namespace

void *operator new(std::size_t bytes)
{
    ++allocationCalls;
    if (allocationsUntilFailure == 0) throw std::bad_alloc();
    if (allocationsUntilFailure > 0) --allocationsUntilFailure;
    if (void *p = std::malloc(bytes == 0 ? 1 : bytes)) return p;
    throw std::bad_alloc();
}

void *operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }

namespace
{
using faiss::ascend::Location;
using faiss::ascend::LocationDelta;
using faiss::ascend::LocationTable;
using LogicalMap = std::map<int64_t, std::array<uint32_t, 3>>;

LogicalMap Snapshot(const LocationTable &table)
{
    LogicalMap result;
    table.forEachValidId(
        [&](int64_t id)
        {
            const auto loc = table.lookupLocation(id);
            result.emplace(id, std::array<uint32_t, 3>{loc->deviceIndex(), loc->listId(), loc->slot()});
        });
    return result;
}

TEST(TestLocationTableStandalone, LayoutAndConfiguredRanges)
{
    EXPECT_EQ(sizeof(Location), 8U);
    EXPECT_THROW(LocationTable(0, 1), std::invalid_argument);
    EXPECT_THROW(LocationTable(257, 1), std::invalid_argument);
    EXPECT_THROW(LocationTable(1, 0), std::invalid_argument);
    EXPECT_THROW(LocationTable(1, 65537), std::invalid_argument);
    LocationTable table(256, 65536);
    table.setLocation(INT64_MAX, UINT8_MAX, UINT16_MAX, UINT32_MAX);
    const auto loc = table.lookupLocation(INT64_MAX);
    ASSERT_TRUE(loc);
    EXPECT_EQ(loc->deviceIndex(), UINT8_MAX);
    EXPECT_EQ(loc->listId(), UINT16_MAX);
    EXPECT_EQ(loc->slot(), UINT32_MAX);
}

TEST(TestLocationTableStandalone, DispersedIdsAndInt64MaxUseSparsePayloads)
{
    LocationTable table(2, 10);
    std::vector<int64_t> ids{0, 4095, 4096, INT64_MAX - 4096, INT64_MAX};
    for (size_t i = 0; i < 128; ++i) ids.push_back((int64_t{1} << 40) + i * 8192);
    for (size_t i = 0; i < ids.size(); ++i)
    {
        table.setLocation(ids[i], static_cast<uint8_t>(i % 2), 7, static_cast<uint32_t>(i));
    }
    EXPECT_EQ(table.allocatedBlockCount(), ids.size() - 1);
    EXPECT_EQ(table.denseBlockCount(), 0U);
    EXPECT_LT(table.memoryBytes(), ids.size() * 64);
    EXPECT_EQ(Snapshot(table).size(), ids.size());
    for (size_t i = 0; i < ids.size(); ++i)
    {
        ASSERT_TRUE(table.contains(ids[i]));
        EXPECT_EQ(table.lookupLocation(ids[i])->slot(), i);
    }
    EXPECT_FALSE(table.contains(-1));
    EXPECT_FALSE(table.contains(2));
    EXPECT_FALSE(table.lookupLocation(2));
}

TEST(TestLocationTableStandalone, ReserveMaterializesInvisibleIndividualSlots)
{
    LocationTable table(2, 10, 10000);
    table.setLocation(3, 1, 8, 20);
    table.reserveForIds({1, 2, 1, 4096, 10000});
    EXPECT_EQ(table.allocatedBlockCount(), 3U);
    EXPECT_EQ(table.denseBlockCount(), 0U);
    EXPECT_EQ(Snapshot(table).size(), 1U);
    EXPECT_FALSE(table.contains(1));
    LocationDelta delta;
    delta.addedIds.emplace_back(1, Location(0, 1, 90));
    EXPECT_NO_THROW(table.validateDelta(delta));
    delta.addedIds.emplace_back(4, Location(0, 1, 90));
    EXPECT_THROW(table.validateDelta(delta), std::logic_error);
    delta.addedIds.pop_back();
    table.applyLocationDeltaNoFail(delta);
    EXPECT_TRUE(table.contains(1));
    EXPECT_EQ(table.lookupLocation(3)->slot(), 20U);
}

TEST(TestLocationTableStandalone, EntireReserveBatchValidatedBeforeAllocation)
{
    LocationTable table(1, 2, 9000);
    table.setLocation(7, 0, 1, 4);
    const auto before = Snapshot(table);
    const size_t bytes = table.memoryBytes();
    EXPECT_THROW(table.reserveForIds({4096, -1}), std::out_of_range);
    EXPECT_THROW(table.reserveForIds({8192, 9001}), std::out_of_range);
    EXPECT_EQ(table.allocatedBlockCount(), 1U);
    EXPECT_EQ(table.memoryBytes(), bytes);
    EXPECT_EQ(Snapshot(table), before);
    EXPECT_THROW(LocationTable::collectTouchedBlockKeys({1, -1}), std::out_of_range);
}

TEST(TestLocationTableStandalone, PromotionCountsInvalidMaterializedSlots)
{
    LocationTable table(1, 2);
    table.setLocation(0, 0, 1, 77);
    std::vector<int64_t> ids;
    for (size_t i = 0; i < LocationTable::kDenseThreshold - 1; ++i) ids.push_back(i);
    table.reserveForIds(ids);
    EXPECT_EQ(table.denseBlockCount(), 0U);
    EXPECT_EQ(Snapshot(table).size(), 1U);
    table.reserveForIds({1023});
    EXPECT_EQ(table.denseBlockCount(), 1U);
    EXPECT_EQ(table.memoryBytes(), LocationTable::kBlockSlots * sizeof(Location));
    EXPECT_EQ(table.lookupLocation(0)->slot(), 77U);
    EXPECT_FALSE(table.contains(1023));

    const auto keys = LocationTable::collectTouchedBlockKeys(ids);
    table.cleanupTouchedBlocks(keys);
    EXPECT_EQ(table.denseBlockCount(), 1U);  // No demotion while nonempty.
    table.clearValid(0);
    table.cleanupTouchedBlocks(keys);
    EXPECT_EQ(table.allocatedBlockCount(), 0U);
    EXPECT_EQ(table.memoryBytes(), 0U);
}

TEST(TestLocationTableStandalone, FailedPromotionPreservesOriginalSparsePayload)
{
    LocationTable table(1, 2);
    std::vector<int64_t> ids;
    for (size_t i = 0; i < LocationTable::kDenseThreshold - 1; ++i) ids.push_back(i);
    table.reserveForIds(ids);
    table.setLocation(42, 0, 1, 9);
    const size_t bytes = table.memoryBytes();
    const std::vector<int64_t> finalId{1023};
    bool failed = false;
    {
        AllocationFailure fault(0);
        try
        {
            table.reserveForIds(finalId);
        }
        catch (const std::bad_alloc &)
        {
            failed = true;
        }
    }
    EXPECT_TRUE(failed);
    EXPECT_EQ(table.denseBlockCount(), 0U);
    EXPECT_EQ(table.memoryBytes(), bytes);
    EXPECT_EQ(table.lookupLocation(42)->slot(), 9U);
    LocationDelta unreserved;
    unreserved.addedIds.emplace_back(1023, Location(0, 1, 0));
    EXPECT_THROW(table.validateDelta(unreserved), std::logic_error);
}

TEST(TestLocationTableStandalone, FailedReservationKeepsLogicalMapAndPrecollectedCleanupWorks)
{
    bool sawFailure = false;
    bool sawPartialReservation = false;
    bool sawSuccess = false;
    std::vector<int64_t> ids;
    for (size_t i = 0; i < 1024; ++i) ids.push_back(i);
    ids.push_back(8192);
    ids.push_back(INT64_MAX);
    const auto keys = LocationTable::collectTouchedBlockKeys(ids);
    for (int64_t failAfter = 0; failAfter < 32; ++failAfter)
    {
        LocationTable table(2, 10);
        table.setLocation(0, 1, 7, 999);
        const auto before = Snapshot(table);
        const size_t initialBytes = table.memoryBytes();
        bool failed = false;
        {
            AllocationFailure fault(failAfter);
            try
            {
                table.reserveForIds(ids);
            }
            catch (const std::bad_alloc &)
            {
                failed = true;
            }
        }
        EXPECT_EQ(Snapshot(table), before);
        sawFailure |= failed;
        sawSuccess |= !failed;
        sawPartialReservation |= failed && table.memoryBytes() > initialBytes;
        const size_t allocationsBefore = allocationCalls;
        {
            AllocationFailure fault(0);
            table.cleanupTouchedBlocks(keys);
        }
        EXPECT_EQ(allocationCalls, allocationsBefore);
        EXPECT_EQ(Snapshot(table), before);
        EXPECT_EQ(table.allocatedBlockCount(), 1U);
        table.clearValid(0);
        table.cleanupTouchedBlocks(keys);
        EXPECT_EQ(table.allocatedBlockCount(), 0U);
    }
    EXPECT_TRUE(sawFailure);
    EXPECT_TRUE(sawPartialReservation);
    EXPECT_TRUE(sawSuccess);
}

TEST(TestLocationTableStandalone, SetAllocationFailureKeepsMappings)
{
    for (int64_t failAfter = 0; failAfter < 4; ++failAfter)
    {
        LocationTable table(1, 2);
        table.setLocation(1, 0, 1, 8);
        const auto before = Snapshot(table);
        bool failed = false;
        {
            AllocationFailure fault(failAfter);
            try
            {
                table.setLocation(8192, 0, 1, 9);
            }
            catch (const std::bad_alloc &)
            {
                failed = true;
            }
        }
        if (failed) EXPECT_EQ(Snapshot(table), before);
        EXPECT_EQ(table.lookupLocation(1)->slot(), 8U);
    }
}

TEST(TestLocationTableStandalone, ValidateDeltaRejectsEveryMalformedWriteWithoutMutation)
{
    LocationTable table(2, 3, 100);
    table.setLocation(1, 0, 2, 8);
    table.reserveForIds({2});
    const auto before = Snapshot(table);
    LocationDelta delta;
    delta.erasedIds = {1, -1};
    EXPECT_THROW(table.validateDelta(delta), std::out_of_range);
    delta.erasedIds = {1, 101};
    EXPECT_THROW(table.validateDelta(delta), std::out_of_range);
    delta.erasedIds = {1};
    delta.addedIds.emplace_back(2, Location());
    EXPECT_THROW(table.validateDelta(delta), std::invalid_argument);
    delta.addedIds.back().second = Location(2, 0, 0);
    EXPECT_THROW(table.validateDelta(delta), std::out_of_range);
    delta.addedIds.back().second = Location(0, 3, 0);
    EXPECT_THROW(table.validateDelta(delta), std::out_of_range);
    delta.addedIds.back() = {3, Location(0, 1, 0)};
    EXPECT_THROW(table.validateDelta(delta), std::logic_error);
    delta.addedIds.back() = {-1, Location(0, 1, 0)};
    EXPECT_THROW(table.validateDelta(delta), std::out_of_range);
    delta.addedIds.back() = {101, Location(0, 1, 0)};
    EXPECT_THROW(table.validateDelta(delta), std::out_of_range);
    delta.addedIds.clear();
    delta.movedIds.emplace_back(2, Location(0, 1, 9));
    EXPECT_THROW(table.validateDelta(delta), std::logic_error);
    delta.movedIds.back() = {3, Location(0, 1, 9)};
    EXPECT_THROW(table.validateDelta(delta), std::logic_error);
    delta.movedIds.back() = {1, Location(0, 1, 9)};
    EXPECT_NO_THROW(table.validateDelta(delta));
    EXPECT_EQ(Snapshot(table), before);
}

TEST(TestLocationTableStandalone, CommitAndCleanupDoNotAllocateOrPromote)
{
    LocationTable table(2, 4);
    table.setLocation(1, 0, 1, 7);
    table.setLocation(4096, 0, 2, 8);
    const std::vector<int64_t> ids{1, 2, 3, 4096, INT64_MAX};
    const auto keys = LocationTable::collectTouchedBlockKeys(ids);
    table.reserveForIds(ids);
    LocationDelta delta;
    delta.erasedIds = {1, 4096};
    delta.movedIds.emplace_back(1, Location(1, 3, 10));
    delta.addedIds.emplace_back(2, Location(1, 3, 11));
    delta.addedIds.emplace_back(INT64_MAX, Location(0, 3, 12));
    table.validateDelta(delta);
    const size_t allocationsBefore = allocationCalls;
    {
        AllocationFailure fault(0);
        table.applyLocationDeltaNoFail(delta);
        table.cleanupTouchedBlocks(keys);
    }
    EXPECT_EQ(allocationCalls, allocationsBefore);
    EXPECT_EQ(table.denseBlockCount(), 0U);
    EXPECT_EQ(table.allocatedBlockCount(), 2U);
    EXPECT_EQ(table.lookupLocation(1)->slot(), 10U);
    EXPECT_EQ(table.lookupLocation(2)->slot(), 11U);
    EXPECT_EQ(table.lookupLocation(INT64_MAX)->slot(), 12U);
    EXPECT_FALSE(table.contains(3));
    EXPECT_FALSE(table.contains(4096));
    LocationDelta pruned;
    pruned.addedIds.emplace_back(3, Location(0, 0, 0));
    EXPECT_THROW(table.validateDelta(pruned), std::logic_error);
    const size_t retainedCapacity = table.memoryBytes();
    table.reserveForIds({3});
    EXPECT_EQ(table.memoryBytes(), retainedCapacity);  // Pruning retains vector capacity.
}

TEST(TestLocationTableStandalone, EraseAndAddSameReservedIdIsAllowed)
{
    LocationTable table(1, 2);
    table.reserveForIds({5});
    LocationDelta delta;
    delta.erasedIds = {5};
    delta.addedIds.emplace_back(5, Location(0, 1, 7));
    table.validateDelta(delta);
    table.applyLocationDeltaNoFail(delta);
    EXPECT_EQ(table.lookupLocation(5)->slot(), 7U);
}

TEST(TestLocationTableStandalone, MixedSparseDenseClearDeviceAndReset)
{
    LocationTable table(2, 3);
    std::vector<int64_t> denseIds;
    for (size_t i = 0; i < 1024; ++i) denseIds.push_back(i);
    table.reserveForIds(denseIds);
    table.setLocation(0, 0, 1, 1);
    table.setLocation(1, 1, 1, 2);
    table.setLocation(8192, 0, 2, 3);
    table.setLocation(INT64_MAX, 1, 2, 4);
    const auto keys = LocationTable::collectTouchedBlockKeys({0, 8192, INT64_MAX});
    const size_t allocationsBefore = allocationCalls;
    {
        AllocationFailure fault(0);
        table.clearDevice(0);
        table.cleanupTouchedBlocks(keys);
    }
    EXPECT_EQ(allocationCalls, allocationsBefore);
    EXPECT_FALSE(table.contains(0));
    EXPECT_FALSE(table.contains(8192));
    EXPECT_TRUE(table.contains(1));
    EXPECT_TRUE(table.contains(INT64_MAX));
    EXPECT_EQ(table.denseBlockCount(), 1U);
    EXPECT_EQ(table.allocatedBlockCount(), 2U);
    table.reset();
    EXPECT_TRUE(Snapshot(table).empty());
    EXPECT_EQ(table.memoryBytes(), 0U);
}

TEST(TestLocationTableStandalone, DeterministicOperationsMatchStdMap)
{
    LocationTable table(2, 8);
    LogicalMap expected;
    uint32_t state = 17;
    for (size_t i = 0; i < 1000; ++i)
    {
        state = state * 1664525U + 1013904223U;
        const int64_t id = static_cast<int64_t>(state % 300) * 4096 + state % 7;
        if ((state & 7U) == 0)
        {
            table.clearValid(id);
            expected.erase(id);
        }
        else
        {
            const uint8_t device = state % 2;
            const uint16_t list = state % 8;
            table.setLocation(id, device, list, state);
            expected[id] = {device, list, state};
        }
        if (i % 31 == 0)
        {
            table.cleanupTouchedBlocks(LocationTable::collectTouchedBlockKeys({id}));
            EXPECT_EQ(Snapshot(table), expected);
        }
    }
    EXPECT_EQ(Snapshot(table), expected);
}

#if GTEST_HAS_DEATH_TEST
const LocationTable *terminateTable = nullptr;
void CheckUnmodifiedAndExit() { std::_Exit(terminateTable && terminateTable->contains(1) ? 71 : 72); }

TEST(TestLocationTableStandalone, CommitPrechecksWholeDeltaBeforeFirstErase)
{
    for (int malformed = 0; malformed < 7; ++malformed)
    {
        EXPECT_EXIT(
            {
                LocationTable table(2, 3, 100);
                table.setLocation(1, 0, 1, 8);
                table.reserveForIds({2});
                LocationDelta delta;
                delta.erasedIds = {1};
                switch (malformed)
                {
                    case 0:
                        delta.erasedIds.push_back(-1);
                        break;
                    case 1:
                        delta.addedIds.emplace_back(2, Location());
                        break;
                    case 2:
                        delta.addedIds.emplace_back(2, Location(2, 0, 0));
                        break;
                    case 3:
                        delta.addedIds.emplace_back(2, Location(0, 3, 0));
                        break;
                    case 4:
                        delta.addedIds.emplace_back(3, Location(0, 0, 0));
                        break;
                    case 5:
                        delta.movedIds.emplace_back(2, Location(0, 0, 0));
                        break;
                    case 6:
                        delta.addedIds.emplace_back(101, Location(0, 0, 0));
                        break;
                }
                terminateTable = &table;
                std::set_terminate(CheckUnmodifiedAndExit);
                table.applyLocationDeltaNoFail(delta);
                std::_Exit(73);
            },
            testing::ExitedWithCode(71), "");
    }
}
#endif
}  // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
