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

#include <memory>
#include <stdexcept>
#include <vector>

#include "ascenddaemon/utils/ChunkedCowVector.h"
#include "common/utils/AscendException.h"

namespace ascend
{

namespace
{
// Small chunk size so a handful of elements span several chunks, exercising the per-chunk COW boundary.
template <typename T>
using Vec4 = ChunkedCowVector<T, 4>;

std::shared_ptr<const int> boxed(int v) { return std::make_shared<const int>(v); }

Vec4<int> makeSequence(size_t n)
{
    Vec4<int> vec;
    vec.resizeFill(n, [](size_t i) { return std::make_shared<const int>(static_cast<int>(i)); });
    return vec;
}
}  // namespace

TEST(ChunkedCowVector, ResizeFillPopulatesEveryElement)
{
    Vec4<int> vec = makeSequence(10);
    ASSERT_EQ(vec.size(), 10u);
    for (size_t i = 0; i < 10; ++i)
    {
        EXPECT_EQ(*vec[i], static_cast<int>(i));
    }
}

TEST(ChunkedCowVector, ResizeFillFailureLeavesEmptyVectorAndReleasesTemporaries)
{
    for (size_t failAt : {0u, 5u, 9u})  // first slot, middle chunk, partial tail
    {
        SCOPED_TRACE(failAt);
        Vec4<int> vec;
        std::vector<std::weak_ptr<const int>> created;
        created.reserve(10);
        size_t calls = 0;
        EXPECT_THROW(vec.resizeFill(10,
                                    [&](size_t i)
                                    {
                                        EXPECT_EQ(i, calls++);
                                        if (i == failAt)
                                        {
                                            throw std::runtime_error("factory failure");
                                        }
                                        auto value = boxed(static_cast<int>(i));
                                        created.push_back(value);
                                        return value;
                                    }),
                     std::runtime_error);
        EXPECT_EQ(calls, failAt + 1);
        EXPECT_EQ(vec.size(), 0u);
        size_t visited = 0;
        vec.forEach([&](const std::shared_ptr<const int>&) { ++visited; });
        EXPECT_EQ(visited, 0u);
        ASSERT_EQ(created.size(), failAt);
        for (const auto& value : created)
        {
            EXPECT_TRUE(value.expired());
        }
        vec.resizeFill(1, [](size_t) { return boxed(42); });
        ASSERT_EQ(vec.size(), 1u);
        EXPECT_EQ(*vec.at(0), 42);
    }
}

TEST(ChunkedCowVector, ResizeFillFailurePreservesPopulatedVectorAndItsChunks)
{
    for (size_t failAt : {0u, 5u, 9u})
    {
        SCOPED_TRACE(failAt);
        Vec4<int> vec = makeSequence(10);
        std::vector<const Vec4<int>::Ptr*> slots;
        for (size_t i = 0; i < vec.size(); ++i)
        {
            slots.push_back(&vec[i]);
        }
        EXPECT_THROW(vec.resizeFill(10,
                                    [&](size_t i)
                                    {
                                        if (i == failAt)
                                        {
                                            throw std::runtime_error("factory failure");
                                        }
                                        return boxed(100 + static_cast<int>(i));
                                    }),
                     std::runtime_error);
        ASSERT_EQ(vec.size(), 10u);
        for (size_t i = 0; i < vec.size(); ++i)
        {
            EXPECT_EQ(&vec.at(i), slots[i]);  // original chunks, not deep copies
            EXPECT_EQ(*vec[i], static_cast<int>(i));
            EXPECT_EQ(vec[i].use_count(), 1);
        }
        size_t visited = 0;
        vec.forEachInUniqueChunks([&](const std::shared_ptr<const int>&) { ++visited; });
        EXPECT_EQ(visited, 10u);
    }
}

TEST(ChunkedCowVector, ResizeFillFailurePreservesSharedOriginalAndCopy)
{
    for (size_t failAt : {0u, 5u, 9u})
    {
        SCOPED_TRACE(failAt);
        Vec4<int> orig = makeSequence(10);
        Vec4<int> copy = orig;
        EXPECT_THROW(copy.resizeFill(10,
                                     [&](size_t i)
                                     {
                                         if (i == failAt)
                                         {
                                             throw std::runtime_error("factory failure");
                                         }
                                         return boxed(100 + static_cast<int>(i));
                                     }),
                     std::runtime_error);
        ASSERT_EQ(orig.size(), 10u);
        ASSERT_EQ(copy.size(), 10u);
        for (size_t i = 0; i < orig.size(); ++i)
        {
            EXPECT_EQ(&copy.at(i), &orig.at(i));  // still shares the same chunks
            EXPECT_EQ(*orig[i], static_cast<int>(i));
            EXPECT_EQ(*copy[i], static_cast<int>(i));
            EXPECT_EQ(orig[i].use_count(), 1);
        }
        size_t visited = 0;
        orig.forEachInUniqueChunks([&](const std::shared_ptr<const int>&) { ++visited; });
        copy.forEachInUniqueChunks([&](const std::shared_ptr<const int>&) { ++visited; });
        EXPECT_EQ(visited, 0u);
        copy.set(1, boxed(100));
        EXPECT_EQ(*copy[1], 100);
        EXPECT_EQ(*orig[1], 1);
    }
}

TEST(ChunkedCowVector, CopyShallowlySharesElementObjects)
{
    Vec4<int> orig = makeSequence(10);
    Vec4<int> copy = orig;
    ASSERT_EQ(copy.size(), orig.size());
    // Shallow copy: both snapshots point at the very same element objects, no deep clone.
    for (size_t i = 0; i < 10; ++i)
    {
        EXPECT_EQ(copy[i].get(), orig[i].get());
    }
}

TEST(ChunkedCowVector, SetIsolatesTheTouchedSlotFromTheOriginal)
{
    Vec4<int> orig = makeSequence(10);
    Vec4<int> copy = orig;

    copy.set(1, boxed(100));

    EXPECT_EQ(*copy[1], 100);  // copy sees the new value
    EXPECT_EQ(*orig[1], 1);    // original is untouched -> COW isolation holds
}

TEST(ChunkedCowVector, SetClonesOnlyTheContainingChunk)
{
    Vec4<int> orig = makeSequence(10);  // chunks: [0..3] [4..7] [8..9]
    Vec4<int> copy = orig;

    copy.set(1, boxed(100));  // touches chunk 0 only

    // Other slots in the touched chunk still share the same element objects (only slot 1 was replaced).
    EXPECT_EQ(copy[0].get(), orig[0].get());
    EXPECT_EQ(copy[2].get(), orig[2].get());
    EXPECT_EQ(copy[3].get(), orig[3].get());
    // Untouched chunks remain fully shared.
    for (size_t i = 4; i < 10; ++i)
    {
        EXPECT_EQ(copy[i].get(), orig[i].get());
    }
}

TEST(ChunkedCowVector, RepeatedSetsOnSameChunkStayIsolated)
{
    Vec4<int> orig = makeSequence(10);
    Vec4<int> copy = orig;

    copy.set(0, boxed(200));
    copy.set(1, boxed(201));
    copy.set(2, boxed(202));

    EXPECT_EQ(*copy[0], 200);
    EXPECT_EQ(*copy[1], 201);
    EXPECT_EQ(*copy[2], 202);
    // Original never changes no matter how many times the copy mutates the shared chunk.
    EXPECT_EQ(*orig[0], 0);
    EXPECT_EQ(*orig[1], 1);
    EXPECT_EQ(*orig[2], 2);
}

TEST(ChunkedCowVector, SetOnUniquelyOwnedVectorMutatesInPlace)
{
    Vec4<int> vec = makeSequence(4);
    const int* before = vec[3].get();
    (void)before;
    vec.set(3, boxed(300));
    EXPECT_EQ(*vec[3], 300);
    EXPECT_EQ(vec.size(), 4u);
}

TEST(ChunkedCowVector, AtRejectsOutOfRangeIndex)
{
    Vec4<int> vec = makeSequence(5);
    EXPECT_NO_THROW((void)vec.at(4));
    EXPECT_THROW((void)vec.at(5), AscendException);
}

TEST(ChunkedCowVector, ForEachVisitsEveryElementInOrder)
{
    Vec4<int> vec = makeSequence(10);
    std::vector<int> seen;
    vec.forEach([&](const std::shared_ptr<const int>& p) { seen.push_back(*p); });
    ASSERT_EQ(seen.size(), 10u);
    for (size_t i = 0; i < 10; ++i)
    {
        EXPECT_EQ(seen[i], static_cast<int>(i));
    }
}

TEST(ChunkedCowVector, UniqueChunkVisitorTracksSharingAndLaterWrites)
{
    Vec4<int> orig = makeSequence(10);
    const auto uniqueValues = [](const Vec4<int>& vec)
    {
        std::vector<int> seen;
        vec.forEachInUniqueChunks([&](const std::shared_ptr<const int>& p) { seen.push_back(*p); });
        return seen;
    };
    const std::vector<int> all = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_EQ(uniqueValues(orig), all);
    {
        Vec4<int> copy = orig;
        EXPECT_EQ(orig[4].use_count(), 1);  // a shared chunk still has only one shared_ptr slot per element
        EXPECT_TRUE(uniqueValues(orig).empty());
        EXPECT_TRUE(uniqueValues(copy).empty());

        copy.set(1, boxed(100));
        EXPECT_EQ(uniqueValues(orig), (std::vector<int>{0, 1, 2, 3}));
        EXPECT_EQ(uniqueValues(copy), (std::vector<int>{0, 100, 2, 3}));
        EXPECT_EQ(orig[0].use_count(), 2);  // the visitor includes shared elements inside an exclusive chunk

        copy.set(4, boxed(400));
        EXPECT_EQ(uniqueValues(orig), (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}));
        EXPECT_EQ(uniqueValues(copy), (std::vector<int>{0, 100, 2, 3, 400, 5, 6, 7}));
        EXPECT_EQ(orig[8].use_count(), 1);  // the shared partial tail chunk remains excluded
    }
    EXPECT_EQ(uniqueValues(orig), all);  // releasing the clone makes every chunk exclusive again
}

TEST(ChunkedCowVector, EmptyVectorHasZeroSize)
{
    Vec4<int> vec;
    EXPECT_EQ(vec.size(), 0u);
    size_t count = 0;
    vec.forEach([&](const std::shared_ptr<const int>&) { ++count; });
    EXPECT_EQ(count, 0u);
}

}  // namespace ascend

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
