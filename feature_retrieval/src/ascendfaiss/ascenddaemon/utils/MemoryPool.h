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

// Thread-safe best-fit-with-coalescing (BFC) pool with bounded idle-chunk caching.
//
// Free-list layout: a fixed array of power-of-two size bins, each holding an
// intrusive doubly-linked list sorted by ascending block size. This keeps
// best-fit exact while making insert/remove allocation-free, which is what
// lets deallocate() be noexcept. See the note on deallocate().

#ifndef ASCEND_MEMORYPOOL_INCLUDED
#define ASCEND_MEMORYPOOL_INCLUDED

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

#include "acl/acl.h"
#include "common/utils/LogUtils.h"

namespace ascend
{

class IUpstream
{
   public:
    virtual ~IUpstream() = default;
    // Returns at least size bytes, or nullptr on failure.
    virtual void *alloc(size_t size) = 0;
    virtual void free(void *p) = 0;
};

class MemoryPool
{
   public:
    static constexpr size_t SMALL_ALIGN = 4ULL * 1024;
    static constexpr size_t HUGE_PAGE_SIZE = 2ULL * 1024 * 1024;
    static constexpr size_t DEFAULT_CHUNK_SIZE = 128ULL * 1024 * 1024;
    static constexpr size_t STAT_LOG_INTERVAL = 100;

    // CANN suballocations require 64B address alignment and alignUp(size, 32) + 32 bytes.
    static constexpr size_t CANN_ADDR_ALIGN = 64;
    static constexpr size_t CANN_SIZE_ALIGN = 32;
    static constexpr size_t CANN_SIZE_PAD = 32;
    static constexpr size_t MIN_SPLIT = CANN_ADDR_ALIGN;

    // Leave headroom for rounding, padding and alignment.
    static constexpr size_t MAX_REQUEST = SIZE_MAX - HUGE_PAGE_SIZE - CANN_SIZE_ALIGN - CANN_SIZE_PAD;

    // Huge-page rounding is only worth it while the waste stays under 1/8 of the
    // request. Without this, 2 MiB + 32 B would round to 4 MiB.
    static constexpr size_t HUGE_WASTE_DIVISOR = 8;

    MemoryPool() = default;
    ~MemoryPool() { destroy(); }
    MemoryPool(const MemoryPool &) = delete;
    MemoryPool &operator=(const MemoryPool &) = delete;

    aclError init(IUpstream *upstream, size_t chunkSize = DEFAULT_CHUNK_SIZE, size_t maxFreeChunks = 1)
    {
        if (upstream == nullptr || chunkSize == 0)
        {
            APP_LOG_ERROR("[MemoryPool] init invalid arg: upstream=%p chunkSize=%zu", (void *)upstream, chunkSize);
            return ACL_ERROR_INVALID_PARAM;
        }
        std::lock_guard<std::mutex> lock(mtx_);
        // Re-initializing would retarget upstream_ while chunks from the previous
        // upstream are still live, so they would later be released through the
        // wrong allocator. Require an explicit destroy() instead.
        if (inited_)
        {
            APP_LOG_ERROR("[MemoryPool] init on already initialized pool (%zu chunks live), call destroy() first",
                          chunks_.size());
            return ACL_ERROR_REPEAT_INITIALIZE;
        }
        upstream_ = upstream;
        chunkSize_ = chunkSize;
        maxFreeChunks_ = maxFreeChunks;
        inited_ = true;
        return ACL_SUCCESS;
    }

    aclError allocate(size_t size, void **outPtr)
    {
        if (outPtr == nullptr)
        {
            APP_LOG_ERROR("[MemoryPool] allocate invalid arg: outPtr is null (size=%zu)", size);
            return ACL_ERROR_INVALID_PARAM;
        }
        std::lock_guard<std::mutex> lock(mtx_);
        if (!inited_)
        {
            APP_LOG_ERROR("[MemoryPool] allocate on uninitialized pool (size=%zu)", size);
            return ACL_ERROR_UNINITIALIZE;
        }
        if (size == 0)
        {
            size = 1;
        }
        if (size > MAX_REQUEST)
        {
            APP_LOG_ERROR("[MemoryPool] allocate size too large: %zu (max %zu)", size, MAX_REQUEST);
            return ACL_ERROR_INVALID_PARAM;
        }
        const size_t need = roundRequest(size);

        Block *b = findBestFit(need);
        if (b == nullptr)
        {
            aclError s = growChunk(need);
            if (s != ACL_SUCCESS)
            {
                // Reclaim idle chunks before retrying upstream allocation.
                size_t released = 0;
                while (releaseOneFreeChunk())
                {
                    --freeChunkCount_;
                    ++released;
                }
                if (released > 0)
                {
                    s = growChunk(need);
                }
                if (s != ACL_SUCCESS)
                {
                    APP_LOG_ERROR("[MemoryPool] growChunk failed after shrink(%zu chunks): need=%zu", released, need);
                    return s;
                }
            }
            b = findBestFit(need);
            if (b == nullptr)
            {
                APP_LOG_ERROR("[MemoryPool] allocate failed after grow: need=%zu bytes", need);
                return ACL_ERROR_BAD_ALLOC;
            }
        }

        // Record ownership first: on metadata failure the block stays free.
        try
        {
            allocated_[b->ptr] = b;
        }
        catch (const std::bad_alloc &)
        {
            APP_LOG_ERROR("[MemoryPool] allocate failed: index update for %zu bytes", need);
            return ACL_ERROR_BAD_ALLOC;
        }
        removeFree(b);
        splitIfNeeded(b, need);

        b->free = false;
        if (b->chunk->liveAllocCount == 0)
        {
            if (freeChunkCount_ > 0)
            {
                --freeChunkCount_;
            }
        }
        ++b->chunk->liveAllocCount;
        inUseBytes_ += b->size;
        *outPtr = b->ptr;

        if (++allocCount_ % STAT_LOG_INTERVAL == 0)
        {
            Stat s = statNoLock();
            APP_LOG_DEBUG(
                "[MemoryPool] stat after %zu allocs: reserved=%zu inUse=%zu largestFree=%zu "
                "numChunks=%zu numFreeChunks=%zu numFreeBlocks=%zu",
                allocCount_, s.reserved, s.inUse, s.largestFree, s.numChunks, s.numFreeChunks, s.numFreeBlocks);
        }
        return ACL_SUCCESS;
    }

    // Never allocates: the free lists are intrusive and the physical-neighbour
    // links are preallocated, so the only failure mode is an unknown pointer.
    // This is what makes it safe to call from destructors.
    aclError deallocate(void *ptr)
    {
        if (ptr == nullptr)
        {
            return ACL_SUCCESS;
        }
        std::lock_guard<std::mutex> lock(mtx_);
        if (!inited_)
        {
            APP_LOG_ERROR("[MemoryPool] deallocate on uninitialized pool, ptr %p", ptr);
            return ACL_ERROR_UNINITIALIZE;
        }
        if (!deallocateNoLock(ptr))
        {
            APP_LOG_ERROR("[MemoryPool] deallocate unknown ptr %p (not allocated by this pool)", ptr);
            return ACL_ERROR_INVALID_PARAM;
        }
        return ACL_SUCCESS;
    }

    // Releases ptr if this pool owns it and reports whether it did. Lets a caller
    // route a pointer among several pools without keeping a second ownership
    // registry, and without the error spam that probing deallocate() would produce.
    bool tryDeallocate(void *ptr) noexcept
    {
        if (ptr == nullptr)
        {
            return false;
        }
        std::lock_guard<std::mutex> lock(mtx_);
        if (!inited_)
        {
            return false;
        }
        return deallocateNoLock(ptr);
    }

    // Returns the number of fully idle chunks released.
    size_t shrink()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        size_t released = 0;
        while (releaseOneFreeChunk())
        {
            --freeChunkCount_;
            ++released;
        }
        return released;
    }

    // Releases all chunks, including live allocations.
    void destroy()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        for (auto &c : chunks_)
        {
            for (Block *b = c->firstBlock; b != nullptr;)
            {
                Block *nx = b->nextPhys;
                delete b;
                b = nx;
            }
            if (upstream_ != nullptr)
            {
                upstream_->free(c->base);
            }
        }
        chunks_.clear();
        bins_.fill(nullptr);
        allocated_.clear();
        freeChunkCount_ = 0;
        reservedBytes_ = 0;
        inUseBytes_ = 0;
        freeBlockCount_ = 0;
        // Drop the upstream reference too: it may dangle after the owner is gone,
        // and leaving inited_ set would let a later allocate() use it.
        upstream_ = nullptr;
        allocCount_ = 0;
        inited_ = false;
    }

    struct Stat
    {
        size_t reserved;  // Upstream bytes.
        size_t inUse;     // Allocated bytes, including rounding and padding.
        size_t largestFree;
        size_t numChunks;
        size_t numFreeChunks;
        size_t numFreeBlocks;
    };

    Stat stat()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        return statNoLock();
    }

   private:
    struct Chunk;
    struct Block
    {
        void *ptr = nullptr;
        size_t size = 0;
        bool free = true;
        Chunk *chunk = nullptr;
        // Address-adjacent blocks within the same chunk.
        Block *prevPhys = nullptr;
        Block *nextPhys = nullptr;
        // Intrusive free-list links; valid only while binIdx >= 0.
        Block *freePrev = nullptr;
        Block *freeNext = nullptr;
        // Index into bins_, or NOT_BINNED (-1) when not on any free list.
        int binIdx = -1;
    };
    struct Chunk
    {
        void *base = nullptr;  // Original upstream pointer.
        size_t size = 0;
        size_t usable = 0;  // Excludes leading alignment padding.
        Block *firstBlock = nullptr;
        size_t liveAllocCount = 0;
    };

    // Bin b holds blocks with size in [2^b, 2^(b+1)). Blocks are never smaller
    // than MIN_SPLIT, so bins below MIN_BIN stay empty.
    static constexpr int NUM_BINS = 64;
    static constexpr int NOT_BINNED = -1;

    static int binIndexOf(size_t size)
    {
        // size is always >= MIN_SPLIT (64), so clz is well defined.
        const int idx = 63 - __builtin_clzll(static_cast<unsigned long long>(size));
        return idx < 0 ? 0 : idx;
    }

    static size_t alignUp(size_t n, size_t a) { return (n + a - 1) / a * a; }

    // Requires mtx_.
    Stat statNoLock() const
    {
        // Counters are maintained incrementally to keep stats O(1).
        Stat s{reservedBytes_, inUseBytes_, 0, chunks_.size(), freeChunkCount_, freeBlockCount_};
        s.largestFree = largestFreeNoLock();
        return s;
    }

    // Sub-page requests stay compact at 64 B. Everything else rounds to 4 KiB,
    // except that requests of at least one huge page may round to 2 MiB when the
    // resulting waste is bounded. The chunk base is only 64 B aligned, so 2 MiB
    // size rounding does not by itself place a block on a huge-page boundary; it
    // exists to keep large neighbouring blocks uniformly sized and coalescable.
    // That is not worth an unbounded amount of slack, hence HUGE_WASTE_DIVISOR.
    size_t roundRequest(size_t n) const
    {
        const size_t required = alignUp(n, CANN_SIZE_ALIGN) + CANN_SIZE_PAD;
        if (n < SMALL_ALIGN)
        {
            return alignUp(required, CANN_ADDR_ALIGN);
        }
        const size_t smallRounded = alignUp(required, SMALL_ALIGN);
        if (n < HUGE_PAGE_SIZE)
        {
            const size_t rounded = alignUp(n, SMALL_ALIGN);
            return rounded >= required ? rounded : smallRounded;
        }
        const size_t hugeRounded = alignUp(n, HUGE_PAGE_SIZE);
        if (hugeRounded >= required && (hugeRounded - n) <= n / HUGE_WASTE_DIVISOR)
        {
            return hugeRounded;
        }
        return smallRounded;
    }

    // Allocation-free. Keeps each bin sorted by ascending size so that
    // findBestFit stays exact. Equal sizes are inserted at the front of their
    // run, so the scan stops at the first block of the same size.
    void insertFree(Block *b) noexcept
    {
        const int idx = binIndexOf(b->size);
        Block *cur = bins_[static_cast<size_t>(idx)];
        Block *prev = nullptr;
        while (cur != nullptr && cur->size < b->size)
        {
            prev = cur;
            cur = cur->freeNext;
        }
        b->freePrev = prev;
        b->freeNext = cur;
        if (cur != nullptr)
        {
            cur->freePrev = b;
        }
        if (prev != nullptr)
        {
            prev->freeNext = b;
        }
        else
        {
            bins_[static_cast<size_t>(idx)] = b;
        }
        b->binIdx = idx;
        ++freeBlockCount_;
    }

    // Allocation-free. Tolerates blocks that are not currently binned.
    void removeFree(Block *b) noexcept
    {
        if (b->binIdx == NOT_BINNED)
        {
            return;
        }
        if (b->freePrev != nullptr)
        {
            b->freePrev->freeNext = b->freeNext;
        }
        else
        {
            bins_[static_cast<size_t>(b->binIdx)] = b->freeNext;
        }
        if (b->freeNext != nullptr)
        {
            b->freeNext->freePrev = b->freePrev;
        }
        b->freePrev = nullptr;
        b->freeNext = nullptr;
        b->binIdx = NOT_BINNED;
        if (freeBlockCount_ > 0)
        {
            --freeBlockCount_;
        }
    }

    Block *findBestFit(size_t need) noexcept
    {
        const int start = binIndexOf(need);
        // Within the starting bin sizes straddle need, so scan for the first fit.
        for (Block *b = bins_[static_cast<size_t>(start)]; b != nullptr; b = b->freeNext)
        {
            if (b->size >= need)
            {
                return b;
            }
        }
        // Every block in a higher bin is >= 2^(start+1) > need, and each bin is
        // size-sorted, so the head of the first non-empty bin is the best fit.
        for (int idx = start + 1; idx < NUM_BINS; ++idx)
        {
            if (bins_[static_cast<size_t>(idx)] != nullptr)
            {
                return bins_[static_cast<size_t>(idx)];
            }
        }
        return nullptr;
    }

    // Requires mtx_. Largest free block, or 0 when no block is free.
    size_t largestFreeNoLock() const noexcept
    {
        for (int idx = NUM_BINS - 1; idx >= 0; --idx)
        {
            Block *b = bins_[static_cast<size_t>(idx)];
            if (b == nullptr)
            {
                continue;
            }
            while (b->freeNext != nullptr)
            {  // Bins are sorted ascending.
                b = b->freeNext;
            }
            return b->size;
        }
        return 0;
    }

    void splitIfNeeded(Block *b, size_t need)
    {
        if (b->size - need < MIN_SPLIT)
        {
            return;
        }
        // Splitting is optional: on metadata failure keep b whole instead of failing.
        std::unique_ptr<Block> hold(new (std::nothrow) Block);
        if (hold == nullptr)
        {
            return;
        }
        Block *rem = hold.get();
        rem->ptr = static_cast<void *>(static_cast<uint8_t *>(b->ptr) + need);
        rem->size = b->size - need;
        rem->free = true;
        rem->chunk = b->chunk;
        rem->prevPhys = b;
        rem->nextPhys = b->nextPhys;
        insertFree(rem);
        // Link the remainder only after it is indexed.
        if (b->nextPhys != nullptr)
        {
            b->nextPhys->prevPhys = rem;
        }
        b->nextPhys = rem;
        b->size = need;
        (void)hold.release();
    }

    // src must immediately follow dst within the same chunk.
    void mergeInto(Block *dst, Block *src)
    {
        dst->size += src->size;
        dst->nextPhys = src->nextPhys;
        if (src->nextPhys != nullptr)
        {
            src->nextPhys->prevPhys = dst;
        }
        if (src->chunk->firstBlock == src)
        {
            src->chunk->firstBlock = dst;
        }
        delete src;
    }

    // Requires mtx_. Returns false when ptr was not handed out by this pool.
    // Allocation-free, so the caller can rely on this on destructor paths.
    bool deallocateNoLock(void *ptr) noexcept
    {
        auto it = allocated_.find(ptr);
        if (it == allocated_.end())
        {
            return false;
        }
        Block *b = it->second;
        allocated_.erase(it);
        inUseBytes_ -= b->size;
        b->free = true;
        Chunk *chunk = b->chunk;

        // The chunk comparisons encode the invariant that nextPhys/prevPhys never cross
        // a chunk boundary, so a merge can never join memory from two upstream blocks.
        if (b->nextPhys != nullptr && b->nextPhys->free && b->nextPhys->chunk == chunk)
        {
            Block *n = b->nextPhys;
            removeFree(n);
            mergeInto(b, n);
        }
        if (b->prevPhys != nullptr && b->prevPhys->free && b->prevPhys->chunk == chunk)
        {
            Block *p = b->prevPhys;
            removeFree(p);
            mergeInto(p, b);
            b = p;
        }
        insertFree(b);

        if (--chunk->liveAllocCount == 0)
        {
            ++freeChunkCount_;
            trimIfNeeded();
        }
        return true;
    }

    aclError growChunk(size_t need)
    {
        // Reserve headroom for aligning the first block.
        const size_t want = need + CANN_ADDR_ALIGN;
        const size_t csize = chunkSize_ > want ? chunkSize_ : want;
        // Allocate metadata first: a failure then costs no device memory.
        std::unique_ptr<Chunk> holdChunk(new (std::nothrow) Chunk);
        std::unique_ptr<Block> holdBlock(new (std::nothrow) Block);
        if (holdChunk == nullptr || holdBlock == nullptr)
        {
            APP_LOG_ERROR("[MemoryPool] grow chunk failed: metadata alloc for %zu bytes", csize);
            return ACL_ERROR_BAD_ALLOC;
        }
        void *base = upstream_->alloc(csize);
        if (base == nullptr)
        {
            APP_LOG_ERROR("[MemoryPool] grow chunk failed: upstream alloc %zu bytes returned null", csize);
            return ACL_ERROR_BAD_ALLOC;
        }
        auto *rawBase = static_cast<uint8_t *>(base);
        auto *alignedBase = reinterpret_cast<uint8_t *>(alignUp(reinterpret_cast<uintptr_t>(rawBase), CANN_ADDR_ALIGN));
        const size_t lead = static_cast<size_t>(alignedBase - rawBase);

        Chunk *c = holdChunk.get();
        c->base = base;
        c->size = csize;
        c->usable = csize - lead;
        c->liveAllocCount = 0;

        Block *b = holdBlock.get();
        b->ptr = alignedBase;
        b->size = c->usable;
        b->free = true;
        b->chunk = c;
        b->prevPhys = nullptr;
        b->nextPhys = nullptr;
        c->firstBlock = b;
        // Reserve first so that the push_back below cannot throw. insertFree is
        // allocation-free, so after this point publishing the chunk cannot fail.
        try
        {
            chunks_.reserve(chunks_.size() + 1);
        }
        catch (const std::bad_alloc &)
        {
            upstream_->free(base);
            APP_LOG_ERROR("[MemoryPool] grow chunk failed: index update for %zu bytes", csize);
            return ACL_ERROR_BAD_ALLOC;
        }
        insertFree(b);
        (void)holdBlock.release();
        chunks_.push_back(std::move(holdChunk));
        reservedBytes_ += csize;
        ++freeChunkCount_;
        return ACL_SUCCESS;
    }

    void trimIfNeeded()
    {
        while (freeChunkCount_ > maxFreeChunks_)
        {
            if (!releaseOneFreeChunk())
            {
                break;
            }
            --freeChunkCount_;
        }
    }

    // Releases the largest fully idle chunk, if any.
    bool releaseOneFreeChunk()
    {
        int target = -1;
        size_t bestSize = 0;
        for (size_t i = 0; i < chunks_.size(); ++i)
        {
            if (chunks_[i]->liveAllocCount == 0 && chunks_[i]->size >= bestSize)
            {
                bestSize = chunks_[i]->size;
                target = static_cast<int>(i);
            }
        }
        if (target < 0)
        {
            return false;
        }
        Chunk *c = chunks_[static_cast<size_t>(target)].get();
        // A fully idle chunk must have coalesced into exactly one block covering
        // it. If not, an internal invariant is broken; bail out instead of
        // handing the base pointer back to upstream while blocks may still be
        // referenced. Callers treat false as "nothing reclaimable".
        if (c->firstBlock == nullptr || c->firstBlock->nextPhys != nullptr || c->firstBlock->size != c->usable)
        {
            APP_LOG_ERROR(
                "[MemoryPool] free chunk not fully coalesced: chunkSize=%zu "
                "firstBlockSize=%zu hasNextBlock=%d",
                c->usable, c->firstBlock != nullptr ? c->firstBlock->size : 0,
                (c->firstBlock != nullptr && c->firstBlock->nextPhys != nullptr) ? 1 : 0);
            return false;
        }
        for (Block *b = c->firstBlock; b != nullptr;)
        {
            Block *nx = b->nextPhys;
            removeFree(b);
            delete b;
            b = nx;
        }
        upstream_->free(c->base);
        if (reservedBytes_ >= c->size)
        {
            reservedBytes_ -= c->size;
        }
        chunks_.erase(chunks_.begin() + target);
        return true;
    }

    IUpstream *upstream_ = nullptr;
    size_t chunkSize_ = DEFAULT_CHUNK_SIZE;
    size_t maxFreeChunks_ = 1;
    size_t freeChunkCount_ = 0;
    size_t allocCount_ = 0;
    size_t reservedBytes_ = 0;
    size_t inUseBytes_ = 0;
    size_t freeBlockCount_ = 0;
    bool inited_ = false;

    std::vector<std::unique_ptr<Chunk>> chunks_;
    std::unordered_map<void *, Block *> allocated_;
    // Intrusive free lists; no heap allocation on the free path.
    std::array<Block *, NUM_BINS> bins_{};
    std::mutex mtx_;
};

}  // namespace ascend

#endif  // ASCEND_MEMORYPOOL_INCLUDED
