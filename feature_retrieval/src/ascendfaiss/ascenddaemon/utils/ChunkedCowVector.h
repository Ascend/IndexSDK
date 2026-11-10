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

#ifndef ASCEND_CHUNKED_COW_VECTOR_INCLUDED
#define ASCEND_CHUNKED_COW_VECTOR_INCLUDED

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

#include "common/utils/AscendAssert.h"

namespace ascend
{

// A fixed-length vector of shared_ptr<const T> elements, partitioned into fixed-size chunks so that a
// copy shares chunks with its source and a single-element write copies only the one chunk it touches
// (copy-on-write at chunk granularity).
//
// Motivation: the IVF snapshot holds one entry per inverted list (numLists can be very large). Publishing
// a write clones the whole snapshot, but a write touches only a few lists. A flat vector makes every clone
// O(numLists) (allocate N slots + N atomic refcount bumps). Chunking cuts a clone to O(numLists/ChunkSize)
// for the top array plus one chunk copy per touched chunk, while keeping O(1) indexed access on the hot
// search path.
//
// Concurrency: a published instance is immutable. Readers only call size()/operator[]/at()/forEach(); the
// single writer mutates its own private clone via set(), which copies a chunk before writing it, so shared
// chunks (held by the base snapshot or retired snapshots) are never modified in place.
template <typename T, size_t ChunkSize = 1024>
class ChunkedCowVector
{
   public:
    using Ptr = std::shared_ptr<const T>;

    static_assert(ChunkSize > 0, "ChunkSize must be > 0");

    // Default copy/move are intentional: they shallow-copy the chunk shared_ptrs so a clone structurally
    // shares chunks with its source (copy-on-write). set() clones a chunk into a fresh shared_ptr before
    // mutating it. Do not delete these.

    size_t size() const { return size_; }

    // Unchecked indexed access for the hot path. Two array indexings, no bounds check.
    const Ptr& operator[](size_t index) const { return (*chunks_[index / ChunkSize])[index % ChunkSize]; }

    const Ptr& at(size_t index) const
    {
        ASCEND_THROW_IF_NOT_FMT(index < size_, "index %zu out of range (size %zu)", index, size_);
        return (*this)[index];
    }

    // (Re)build the vector with n elements, filling slot i with factory(i). Drops any previous contents.
    template <typename Factory>
    void resizeFill(size_t n, Factory&& factory)
    {
        std::vector<std::shared_ptr<Chunk>> newChunks;
        size_t chunkCount = (n + ChunkSize - 1) / ChunkSize;
        newChunks.reserve(chunkCount);
        for (size_t c = 0; c < chunkCount; ++c)
        {
            size_t begin = c * ChunkSize;
            size_t len = std::min(ChunkSize, n - begin);
            auto chunk = std::make_shared<Chunk>();
            chunk->reserve(len);
            for (size_t k = 0; k < len; ++k)
            {
                chunk->push_back(factory(begin + k));
            }
            newChunks.push_back(std::move(chunk));
        }
        chunks_.swap(newChunks);
        size_ = n;
    }

    // Replace element index. If the containing chunk is shared with another instance, clone it first so the
    // write is isolated (copy-on-write); otherwise mutate the already-exclusive chunk in place.
    void set(size_t index, Ptr value)
    {
        ASCEND_THROW_IF_NOT_FMT(index < size_, "set index %zu out of range (size %zu)", index, size_);
        auto& chunkPtr = chunks_[index / ChunkSize];
        if (chunkPtr.use_count() > 1)
        {
            chunkPtr = std::make_shared<Chunk>(*chunkPtr);
        }
        (*chunkPtr)[index % ChunkSize] = std::move(value);
    }

    // Visits slots only in chunks this instance alone owns. Slot refcounts cannot tell whether a whole
    // chunk is shared: two vectors can reach the very same shared_ptr slot without incrementing it.
    // Callers must hold any external lock that serializes copying/mutating instances.
    template <typename F>
    void forEachInUniqueChunks(F&& f) const
    {
        for (const auto& chunk : chunks_)
        {
            if (chunk.use_count() != 1)
            {
                continue;
            }
            for (const auto& slot : *chunk)
            {
                f(slot);
            }
        }
    }

    template <typename F>
    void forEach(F&& f) const
    {
        for (const auto& chunk : chunks_)
        {
            for (const auto& slot : *chunk)
            {
                f(slot);
            }
        }
    }

   private:
    using Chunk = std::vector<Ptr>;

    std::vector<std::shared_ptr<Chunk>> chunks_;
    size_t size_{0};
};

}  // namespace ascend

#endif  // ASCEND_CHUNKED_COW_VECTOR_INCLUDED
