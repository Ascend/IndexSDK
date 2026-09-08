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

#include "ascend/AscendIDSelectorRoaring.h"

#include <faiss/impl/FaissAssert.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "ascend/impl/AscendRoaringFilter.h"
#include "common/utils/CommonUtils.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"
#include "roaring.h"

namespace faiss
{
namespace ascend
{
namespace
{
constexpr size_t kRoaringAddChunk = 4096;

char *AlignUp32(uint8_t *p)
{
    const uintptr_t addr = reinterpret_cast<uintptr_t>(p);
    const size_t align = aicpu::RABITQ_ROARING_FROZEN_ALIGN;
    const size_t off = (align - (addr % align)) % align;
    return reinterpret_cast<char *>(p + off);
}

const roaring_bitmap_t *AsRoaring(const void *p) { return static_cast<const roaring_bitmap_t *>(p); }

const roaring_bitmap_t *MakeFrozenView(const uint8_t *frozen, size_t n, std::vector<uint8_t> &tmp)
{
    if (frozen == nullptr || n == 0)
    {
        return nullptr;
    }
    const char *ptr = reinterpret_cast<const char *>(frozen);
    if ((reinterpret_cast<uintptr_t>(frozen) & (aicpu::RABITQ_ROARING_FROZEN_ALIGN - 1)) != 0)
    {
        tmp.resize(n + aicpu::RABITQ_ROARING_FROZEN_ALIGN);
        char *aligned = AlignUp32(tmp.data());
        std::memcpy(aligned, frozen, n);
        ptr = aligned;
    }
    return roaring_bitmap_frozen_view(ptr, n);
}

bool FreezeRoaringToHost(roaring_bitmap_t *rb, ::ascend::RabitqIdFilterHost &out, bool runOptimize)
{
    if (rb == nullptr)
    {
        return false;
    }
    if (runOptimize)
    {
        roaring_bitmap_run_optimize(rb);
    }
    const size_t sz = roaring_bitmap_frozen_size_in_bytes(rb);
    out.mode = aicpu::RABITQ_ID_FILTER_ROARING;
    out.sortedView = nullptr;
    out.bitmapView = nullptr;
    out.viewBytes = 0;
    if (sz == 0)
    {
        out.aux0 = 0;
        out.bitmap.clear();
        return true;
    }
    std::vector<uint8_t> tmp(sz + aicpu::RABITQ_ROARING_FROZEN_ALIGN);
    char *aligned = AlignUp32(tmp.data());
    roaring_bitmap_frozen_serialize(rb, aligned);
    out.aux0 = static_cast<int64_t>(sz);
    out.bitmap.assign(reinterpret_cast<const uint8_t *>(aligned), reinterpret_cast<const uint8_t *>(aligned) + sz);
    return true;
}

bool AddUint32Runs(roaring_bitmap_t *rb, const uint32_t *vals, size_t n)
{
    if (rb == nullptr)
    {
        return false;
    }
    uint32_t chunk[kRoaringAddChunk];
    size_t filled = 0;
    auto flushChunk = [&]()
    {
        if (filled > 0)
        {
            roaring_bitmap_add_many(rb, filled, chunk);
            filled = 0;
        }
    };
    size_t i = 0;
    while (i < n)
    {
        size_t j = i + 1;
        while (j < n && vals[j] > vals[j - 1] && vals[j] == vals[j - 1] + 1U)
        {
            ++j;
        }
        if (j - i >= 2)
        {
            flushChunk();
            roaring_bitmap_add_range(rb, vals[i], static_cast<uint64_t>(vals[j - 1]) + 1ULL);
        }
        else
        {
            chunk[filled++] = vals[i];
            if (filled == kRoaringAddChunk)
            {
                flushChunk();
            }
        }
        i = j;
    }
    flushChunk();
    return true;
}

bool IdsFitUint32(int64_t id) { return id >= 0 && static_cast<uint64_t>(id) <= std::numeric_limits<uint32_t>::max(); }

bool FreezeRoaringIdxsChunked(const idx_t *ids, size_t n, ::ascend::RabitqIdFilterHost &out)
{
    roaring_bitmap_t *rb = roaring_bitmap_create();
    if (rb == nullptr)
    {
        return false;
    }
    uint32_t chunk[kRoaringAddChunk];
    size_t filled = 0;
    auto flushChunk = [&]()
    {
        if (filled > 0)
        {
            roaring_bitmap_add_many(rb, filled, chunk);
            filled = 0;
        }
    };
    size_t i = 0;
    while (i < n)
    {
        if (!IdsFitUint32(ids[i]))
        {
            roaring_bitmap_free(rb);
            return false;
        }
        size_t j = i + 1;
        while (j < n && IdsFitUint32(ids[j]) && ids[j] == ids[j - 1] + static_cast<idx_t>(1))
        {
            ++j;
        }
        if (j - i >= 2)
        {
            flushChunk();
            roaring_bitmap_add_range(rb, static_cast<uint32_t>(ids[i]), static_cast<uint64_t>(ids[j - 1]) + 1ULL);
        }
        else
        {
            chunk[filled++] = static_cast<uint32_t>(ids[i]);
            if (filled == kRoaringAddChunk)
            {
                flushChunk();
            }
        }
        i = j;
    }
    flushChunk();
    const bool ok = FreezeRoaringToHost(rb, out, true);
    roaring_bitmap_free(rb);
    return ok;
}
}  // namespace

IDSelectorRoaring::IDSelectorRoaring(size_t nBytes, const uint8_t *frozenBytes)
    : n(nBytes), frozen(frozenBytes), live(nullptr)
{
    FAISS_THROW_IF_NOT_MSG(nBytes == 0 || frozenBytes != nullptr,
                           "IDSelectorRoaring frozen buffer cannot be nullptr when n > 0");
    InitFrozenView();
}

IDSelectorRoaring::IDSelectorRoaring(const void *rb) : n(0), frozen(nullptr), live(rb)
{
    FAISS_THROW_IF_NOT_MSG(rb != nullptr, "IDSelectorRoaring live bitmap cannot be nullptr");
}

IDSelectorRoaring::~IDSelectorRoaring()
{
    if (frozenView_ != nullptr)
    {
        roaring_bitmap_free(const_cast<roaring_bitmap_t *>(AsRoaring(frozenView_)));
        frozenView_ = nullptr;
    }
}

void IDSelectorRoaring::InitFrozenView() { frozenView_ = MakeFrozenView(frozen, n, frozenAligned_); }

bool IDSelectorRoaring::is_member(idx_t id) const
{
    if (id < 0 || static_cast<uint64_t>(id) > std::numeric_limits<uint32_t>::max())
    {
        return false;
    }
    const uint32_t uid = static_cast<uint32_t>(id);
    if (live != nullptr)
    {
        return roaring_bitmap_contains(AsRoaring(live), uid);
    }
    if (frozenView_ == nullptr)
    {
        return false;
    }
    return roaring_bitmap_contains(AsRoaring(frozenView_), uid);
}

bool FreezeRoaringLive(const void *liveBitmap, ::ascend::RabitqIdFilterHost &out)
{
    if (liveBitmap == nullptr)
    {
        return false;
    }
    roaring_bitmap_t *owned = roaring_bitmap_copy(AsRoaring(liveBitmap));
    if (owned == nullptr)
    {
        return false;
    }
    roaring_bitmap_run_optimize(owned);
    const bool ok = FreezeRoaringToHost(owned, out, false);
    roaring_bitmap_free(owned);
    return ok;
}

bool FreezeRoaringUint32Sorted(const uint32_t *vals, size_t n, ::ascend::RabitqIdFilterHost &out)
{
    roaring_bitmap_t *rb = roaring_bitmap_create();
    if (rb == nullptr)
    {
        return false;
    }
    if (!AddUint32Runs(rb, vals, n))
    {
        roaring_bitmap_free(rb);
        return false;
    }
    const bool ok = FreezeRoaringToHost(rb, out, true);
    roaring_bitmap_free(rb);
    return ok;
}

bool IdsFormArithmeticProgression(const idx_t *ids, size_t n, int64_t step)
{
    if (ids == nullptr || n <= 1 || step <= 0)
    {
        return n <= 1 && step > 0;
    }
    const int64_t base = static_cast<int64_t>(ids[0]);
    const int64_t last = static_cast<int64_t>(ids[n - 1]);
    const int64_t n1 = static_cast<int64_t>(n - 1);
    if (n1 > std::numeric_limits<int64_t>::max() / step)
    {
        return false;
    }
    if (last != base + n1 * step)
    {
        return false;
    }
    std::atomic<int> mismatch{0};
    constexpr size_t kParallelMinN = 65536;
#pragma omp parallel num_threads(::ascend::CommonUtils::GetThreadMaxNums()) if (n >= kParallelMinN)
    {
        const int tid = omp_get_thread_num();
        const int nth = omp_get_num_threads();
        const size_t chunk = (n + static_cast<size_t>(nth) - 1U) / static_cast<size_t>(nth);
        const size_t begin = std::min(n, static_cast<size_t>(tid) * chunk);
        const size_t end = std::min(n, begin + chunk);
        for (size_t i = begin; i < end; ++i)
        {
            if (mismatch.load(std::memory_order_relaxed) != 0)
            {
                break;
            }
            if (static_cast<int64_t>(ids[i]) != base + static_cast<int64_t>(i) * step)
            {
                mismatch.store(1, std::memory_order_relaxed);
                break;
            }
        }
    }
    return mismatch.load(std::memory_order_relaxed) == 0;
}

bool FreezeRoaringFromRange(uint64_t minInclusive, uint64_t maxExclusive, uint32_t step,
                            ::ascend::RabitqIdFilterHost &out)
{
    if (step == 0 || maxExclusive < minInclusive)
    {
        return false;
    }
    roaring_bitmap_t *rb = roaring_bitmap_from_range(minInclusive, maxExclusive, step);
    if (rb == nullptr)
    {
        return false;
    }
    const bool ok = FreezeRoaringToHost(rb, out, true);
    roaring_bitmap_free(rb);
    return ok;
}

bool FreezeRoaringIdxs(const idx_t *ids, size_t n, ::ascend::RabitqIdFilterHost &out)
{
    if (ids == nullptr)
    {
        return n == 0;
    }
    if (n >= 2 && IdsFitUint32(ids[0]) && IdsFitUint32(ids[n - 1]))
    {
        const int64_t step = static_cast<int64_t>(ids[1]) - static_cast<int64_t>(ids[0]);
        if (step > 0 && step <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) &&
            IdsFormArithmeticProgression(ids, n, step))
        {
            return FreezeRoaringFromRange(static_cast<uint64_t>(ids[0]), static_cast<uint64_t>(ids[n - 1]) + 1ULL,
                                          static_cast<uint32_t>(step), out);
        }
    }
    return FreezeRoaringIdxsChunked(ids, n, out);
}

void AssertValidFrozenRoaring(const uint8_t *frozen, size_t n)
{
    if (n == 0)
    {
        return;
    }
    FAISS_THROW_IF_NOT_MSG(frozen != nullptr, "IDSelectorRoaring frozen buffer cannot be nullptr when n > 0");
    std::vector<uint8_t> tmp;
    const roaring_bitmap_t *view = MakeFrozenView(frozen, n, tmp);
    FAISS_THROW_IF_NOT_MSG(view != nullptr, "IDSelectorRoaring frozen buffer is not a valid CRoaring frozen view");
    roaring_bitmap_free(const_cast<roaring_bitmap_t *>(view));
}

}  // namespace ascend
}  // namespace faiss
