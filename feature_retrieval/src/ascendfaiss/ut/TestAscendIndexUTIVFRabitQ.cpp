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

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/impl/IDSelector.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "acl.h"
#include "ascend/AscendIDSelectorRoaring.h"
#include "ascend/AscendIndexIVFRaBitQ.h"
#include "ascend/impl/AscendIndexIVFRaBitQImpl.h"
#include "ascenddaemon/impl/IndexIVFRaBitQ.h"
#include "ascenddaemon/utils/AscendUtils.h"
#include "common/ErrorCode.h"
#include "common/utils/AscendException.h"
#include "mockcpp/mockcpp.hpp"
#include "roaring.h"

namespace faiss
{
namespace ascend
{
// These helpers are intentionally kept in the implementation file. Declaring
// them here lets the UT exercise both orthonormal-initialization branches.
void orthonormalinit(std::vector<float>& matrix, int seed, int inputDim, int outputDim);
}  // namespace ascend
}  // namespace faiss

namespace ascend
{
// Also kept in the implementation file; declared here so the UT can build the same slot plan the production
// fill path builds and assert the two agree. See the block-grid tests at the bottom of this file.
void BuildBlockSlotPlan(const std::vector<int64_t>& blockNumPerQ, size_t maxBlocksPerQ, int64_t group, int64_t* slots);
}  // namespace ascend

namespace
{
using testing::ElementsAre;
using testing::HasSubstr;
using testing::UnorderedElementsAre;

constexpr int DIM = 128;
constexpr int NLIST = 1024;
constexpr int NPROBE = 16;

template <typename Func>
void ExpectExceptionContains(Func&& func, const std::string& expected)
{
    try
    {
        func();
        FAIL() << "Expected exception containing: " << expected;
    }
    catch (const std::exception& e)
    {
        EXPECT_THAT(std::string(e.what()), HasSubstr(expected));
    }
    catch (...)
    {
        FAIL() << "Expected std::exception containing: " << expected;
    }
}

std::vector<float> MakePatternData(size_t count, int dim)
{
    std::vector<float> data(count * static_cast<size_t>(dim));
    for (size_t i = 0; i < count; ++i)
    {
        for (int j = 0; j < dim; ++j)
        {
            data[i * static_cast<size_t>(dim) + static_cast<size_t>(j)] =
                static_cast<float>((i * 13 + static_cast<size_t>(j) * 7) % 97) / 97.0f;
        }
    }
    return data;
}

faiss::ascend::AscendIndexIVFRaBitQConfig MakeConfig(std::vector<int> devices = {0},
                                                     bool useRandomOrthogonalMatrix = false)
{
    faiss::ascend::AscendIndexIVFRaBitQConfig config(std::move(devices));
    config.useKmeansPP = false;
    config.useRandomOrthogonalMatrix = useRandomOrthogonalMatrix;
    return config;
}

// The CI image's IVFRaBitQ operators ignore IDSelector inputs and return unfiltered
// self-matches. These three tests only check membership, so the device search is stubbed
// after the host has materialized RabitqIdFilterHost. train/add still use the image operators.
using FilteredSearchFn = ::ascend::APP_ERROR (::ascend::IndexIVFRaBitQ::*)(int, const float*, int, float*,
                                                                           ::ascend::idx_t*, const float*,
                                                                           const ::ascend::RabitqIdFilterHost*, int);

const int64_t* SortedIdSpan(const ::ascend::RabitqIdFilterHost& filter, int64_t& count)
{
    count = 0;
    if (filter.mode == aicpu::RABITQ_ID_FILTER_SORTED_PREFIX && !filter.sortedPrefixPayload.empty() &&
        filter.sortedPrefixPayload.size() >= sizeof(aicpu::RabitqSortedPrefixPayloadHeader))
    {
        aicpu::RabitqSortedPrefixPayloadHeader header{};
        std::memcpy(&header, filter.sortedPrefixPayload.data(), sizeof(header));
        const size_t sortedBytes = static_cast<size_t>(header.sortedCount) * sizeof(int64_t);
        const bool inRange =
            header.magic == aicpu::RABITQ_SORTED_PREFIX_MAGIC && header.sortedCount > 0 &&
            header.sortedOffsetBytes >= 0 &&
            static_cast<size_t>(header.sortedOffsetBytes) + sortedBytes <= filter.sortedPrefixPayload.size();
        if (inRange)
        {
            count = header.sortedCount;
            return reinterpret_cast<const int64_t*>(filter.sortedPrefixPayload.data() + header.sortedOffsetBytes);
        }
    }
    if (filter.sortedView != nullptr)
    {
        count = filter.aux0 > 0 ? filter.aux0 : static_cast<int64_t>(filter.viewBytes / sizeof(int64_t));
        return count > 0 ? filter.sortedView : nullptr;
    }
    count = static_cast<int64_t>(filter.sortedIds.size());
    return filter.sortedIds.empty() ? nullptr : filter.sortedIds.data();
}

struct RoaringFilterView
{
    std::vector<uint8_t> storage;
    const roaring_bitmap_t* bitmap = nullptr;

    explicit RoaringFilterView(const ::ascend::RabitqIdFilterHost& filter)
    {
        if (filter.mode != aicpu::RABITQ_ID_FILTER_ROARING)
        {
            return;
        }
        const auto* bytes = static_cast<const uint8_t*>(filter.payloadSrc());
        const size_t nbytes = filter.payloadBytes();
        if (bytes == nullptr || nbytes == 0)
        {
            return;
        }
        const char* ptr = reinterpret_cast<const char*>(bytes);
        if ((reinterpret_cast<uintptr_t>(bytes) & (aicpu::RABITQ_ROARING_FROZEN_ALIGN - 1)) != 0)
        {
            storage.resize(nbytes + aicpu::RABITQ_ROARING_FROZEN_ALIGN);
            const uintptr_t addr = reinterpret_cast<uintptr_t>(storage.data());
            const uintptr_t aligned = (addr + aicpu::RABITQ_ROARING_FROZEN_ALIGN - 1) &
                                      ~(static_cast<uintptr_t>(aicpu::RABITQ_ROARING_FROZEN_ALIGN) - 1);
            auto* dst = reinterpret_cast<char*>(aligned);
            std::memcpy(dst, bytes, nbytes);
            ptr = dst;
        }
        bitmap = roaring_bitmap_frozen_view(ptr, nbytes);
    }

    ~RoaringFilterView()
    {
        if (bitmap != nullptr)
        {
            roaring_bitmap_free(const_cast<roaring_bitmap_t*>(bitmap));
        }
    }

    RoaringFilterView(const RoaringFilterView&) = delete;
    RoaringFilterView& operator=(const RoaringFilterView&) = delete;
};

bool IdPassesFilter(const ::ascend::RabitqIdFilterHost& filter, int64_t id, const roaring_bitmap_t* roar)
{
    bool hit = false;
    switch (filter.mode)
    {
        case aicpu::RABITQ_ID_FILTER_NONE:
            hit = true;
            break;
        case aicpu::RABITQ_ID_FILTER_RANGE:
            hit = id >= filter.aux0 && id < filter.aux1;
            break;
        case aicpu::RABITQ_ID_FILTER_SORTED:
        case aicpu::RABITQ_ID_FILTER_SORTED_PREFIX:
        {
            int64_t count = 0;
            const int64_t* ids = SortedIdSpan(filter, count);
            hit = ids != nullptr && std::binary_search(ids, ids + count, id);
            break;
        }
        case aicpu::RABITQ_ID_FILTER_BITMAP:
        {
            const uint8_t* bits = static_cast<const uint8_t*>(filter.payloadSrc());
            if (bits != nullptr && id >= 0 && id < filter.aux0)
            {
                hit = ((bits[static_cast<size_t>(id) >> 3] >> (id & 7)) & 1) != 0;
            }
            break;
        }
        case aicpu::RABITQ_ID_FILTER_ROARING:
            hit = roar != nullptr && id >= 0 && static_cast<uint64_t>(id) <= std::numeric_limits<uint32_t>::max() &&
                  roaring_bitmap_contains(roar, static_cast<uint32_t>(id));
            break;
        default:
            break;
    }
    return filter.negate != 0 ? !hit : hit;
}

int64_t FilterScanLimit(const ::ascend::RabitqIdFilterHost& filter, const roaring_bitmap_t* roar, int k)
{
    constexpr int64_t kScanCap = 1 << 20;
    const int64_t extra = static_cast<int64_t>(k) + 1;
    int64_t limit = extra;
    switch (filter.mode)
    {
        case aicpu::RABITQ_ID_FILTER_RANGE:
            limit = std::max(filter.aux0, filter.aux1) + extra;
            break;
        case aicpu::RABITQ_ID_FILTER_BITMAP:
            limit = filter.aux0 + extra;
            break;
        case aicpu::RABITQ_ID_FILTER_SORTED:
        case aicpu::RABITQ_ID_FILTER_SORTED_PREFIX:
        {
            int64_t count = 0;
            const int64_t* ids = SortedIdSpan(filter, count);
            limit = (ids != nullptr && count > 0) ? ids[count - 1] + extra : extra;
            break;
        }
        case aicpu::RABITQ_ID_FILTER_ROARING:
            if (roar != nullptr && roaring_bitmap_get_cardinality(roar) > 0)
            {
                limit = static_cast<int64_t>(roaring_bitmap_maximum(roar)) + extra;
            }
            break;
        default:
            break;
    }
    if (limit < extra)
    {
        limit = extra;
    }
    return std::min(limit, kScanCap);
}

::ascend::APP_ERROR StubFilteredSearch(::ascend::IndexIVFRaBitQ*, int n, const float*, int k, float* distances,
                                       ::ascend::idx_t* labels, const float*,
                                       const ::ascend::RabitqIdFilterHost* idFilter, int)
{
    if (n <= 0 || k <= 0 || distances == nullptr || labels == nullptr)
    {
        return ::ascend::APP_ERR_OK;
    }
    std::vector<::ascend::idx_t> picked;
    picked.reserve(static_cast<size_t>(k));
    if (idFilter != nullptr)
    {
        RoaringFilterView roar(*idFilter);
        const int64_t limit = FilterScanLimit(*idFilter, roar.bitmap, k);
        for (int64_t id = 0; id < limit && static_cast<int>(picked.size()) < k; ++id)
        {
            if (IdPassesFilter(*idFilter, id, roar.bitmap))
            {
                picked.push_back(static_cast<::ascend::idx_t>(id));
            }
        }
    }
    for (int q = 0; q < n; ++q)
    {
        for (int j = 0; j < k; ++j)
        {
            const size_t pos = static_cast<size_t>(q) * static_cast<size_t>(k) + static_cast<size_t>(j);
            if (j < static_cast<int>(picked.size()))
            {
                distances[pos] = 0.0f;
                labels[pos] = picked[static_cast<size_t>(j)];
            }
            else
            {
                distances[pos] = std::numeric_limits<float>::max();
                labels[pos] = static_cast<::ascend::idx_t>(-1);
            }
        }
    }
    return ::ascend::APP_ERR_OK;
}

struct TopKSearchCall
{
    std::atomic<int> count{0};
    std::atomic<int> n{0};
    std::atomic<int> k{0};
    std::atomic<int64_t> filterMode{aicpu::RABITQ_ID_FILTER_NONE};
    std::atomic<const float*> refineSources{nullptr};

    void reset()
    {
        count = 0;
        n = 0;
        k = 0;
        filterMode = aicpu::RABITQ_ID_FILTER_NONE;
        refineSources = nullptr;
    }
};

TopKSearchCall topKSearchCall;

::ascend::APP_ERROR StubCountingFilteredSearch(::ascend::IndexIVFRaBitQ* index, int n, const float* x, int k,
                                               float* distances, ::ascend::idx_t* labels, const float* srcIndexes,
                                               const ::ascend::RabitqIdFilterHost* idFilter, int searchNprobe)
{
    ++topKSearchCall.count;
    topKSearchCall.n = n;
    topKSearchCall.k = k;
    topKSearchCall.filterMode = idFilter != nullptr ? idFilter->mode : aicpu::RABITQ_ID_FILTER_NONE;
    topKSearchCall.refineSources = srcIndexes;

    // These routing fixtures import three sequential IDs. Leave the rest of the
    // requested (possibly refined) candidate buffer as invalid padding.
    ::ascend::RabitqIdFilterHost allIds;
    allIds.mode = aicpu::RABITQ_ID_FILTER_RANGE;
    allIds.aux0 = 0;
    allIds.aux1 = 3;
    return StubFilteredSearch(index, n, x, k, distances, labels, srcIndexes, idFilter != nullptr ? idFilter : &allIds,
                              searchNprobe);
}

struct FilteredSearchMockGuard
{
    explicit FilteredSearchMockGuard(decltype(&StubFilteredSearch) search = StubFilteredSearch)
    {
        MOCKER_CPP(static_cast<FilteredSearchFn>(&::ascend::IndexIVFRaBitQ::searchImpl)).stubs().will(invoke(search));
    }

    ~FilteredSearchMockGuard() { GlobalMockObject::verify(); }

    FilteredSearchMockGuard(const FilteredSearchMockGuard&) = delete;
    FilteredSearchMockGuard& operator=(const FilteredSearchMockGuard&) = delete;
};

struct CpuIVFRaBitQIndex
{
    std::unique_ptr<faiss::IndexFlat> quantizer;
    std::unique_ptr<faiss::IndexIVFRaBitQ> index;
    std::vector<faiss::idx_t> ids;
    std::vector<uint8_t> codes;
    int listId = 9;
};

CpuIVFRaBitQIndex MakePretrainedCpuIndex(bool withVectors = true)
{
    CpuIVFRaBitQIndex result;
    result.quantizer = std::make_unique<faiss::IndexFlatL2>(DIM);
    result.index = std::make_unique<faiss::IndexIVFRaBitQ>(result.quantizer.get(), DIM, NLIST);
    result.index->nprobe = NPROBE;

    auto centroids = MakePatternData(NLIST, DIM);
    result.quantizer->add(NLIST, centroids.data());
    result.index->is_trained = true;

    if (withVectors)
    {
        result.ids = {101, 202, 303};
        result.codes.resize(result.ids.size() * result.index->code_size);
        for (size_t i = 0; i < result.codes.size(); ++i)
        {
            result.codes[i] = static_cast<uint8_t>((i * 11 + 3) % 251);
        }
        result.index->invlists->add_entries(result.listId, result.ids.size(), result.ids.data(), result.codes.data());
        result.index->ntotal = static_cast<faiss::idx_t>(result.ids.size());
    }
    return result;
}

CpuIVFRaBitQIndex MakeSequentialPretrainedCpuIndex()
{
    auto source = MakePretrainedCpuIndex();
    std::iota(source.ids.begin(), source.ids.end(), 0);
    source.index->invlists->update_entries(source.listId, 0, source.ids.size(), source.ids.data(), source.codes.data());
    return source;
}

struct TopKSearchBuffers
{
    static constexpr float DISTANCE_CANARY = -1234.5f;
    static constexpr faiss::idx_t LABEL_CANARY = -987654;
    std::vector<float> distances;
    std::vector<faiss::idx_t> labels;

    explicit TopKSearchBuffers(size_t count) : distances(count + 2, DISTANCE_CANARY), labels(count + 2, LABEL_CANARY) {}

    float* distanceData() { return distances.data() + 1; }
    faiss::idx_t* labelData() { return labels.data() + 1; }

    void expectCanaries() const
    {
        EXPECT_FLOAT_EQ(distances.front(), DISTANCE_CANARY);
        EXPECT_FLOAT_EQ(distances.back(), DISTANCE_CANARY);
        EXPECT_EQ(labels.front(), LABEL_CANARY);
        EXPECT_EQ(labels.back(), LABEL_CANARY);
    }

    void expectUnchanged() const
    {
        EXPECT_TRUE(
            std::all_of(distances.begin(), distances.end(), [](float value) { return value == DISTANCE_CANARY; }));
        EXPECT_TRUE(
            std::all_of(labels.begin(), labels.end(), [](faiss::idx_t value) { return value == LABEL_CANARY; }));
    }

    void expectResults(int n, int k) const
    {
        expectCanaries();
        ASSERT_EQ(distances.size(), static_cast<size_t>(n) * static_cast<size_t>(k) + 2);
        ASSERT_EQ(labels.size(), distances.size());
        for (int q = 0; q < n; ++q)
        {
            const size_t offset = 1 + static_cast<size_t>(q) * static_cast<size_t>(k);
            for (int j = 0; j < 3; ++j)
            {
                EXPECT_FLOAT_EQ(distances[offset + static_cast<size_t>(j)], 0.0f);
                EXPECT_EQ(labels[offset + static_cast<size_t>(j)], j);
            }
            EXPECT_TRUE(std::all_of(distances.begin() + offset + 3, distances.begin() + offset + k,
                                    [](float value) { return value == std::numeric_limits<float>::max(); }));
            EXPECT_TRUE(std::all_of(labels.begin() + offset + 3, labels.begin() + offset + k,
                                    [](faiss::idx_t value) { return value == -1; }));
        }
    }
};

TEST(TestAscendIndexIVFRaBitQ, PublicSearchAcceptsTopKThrough65536WithAndWithoutSelector)
{
    ASSERT_EQ(faiss::ascend::MAX_K, 65536);
    auto source = MakeSequentialPretrainedCpuIndex();
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    ASSERT_TRUE(index.is_trained);
    const int nq = 2;
    const auto query = MakePatternData(nq, DIM);
    faiss::IDSelectorRange selector(0, 3);
    faiss::SearchParameters params;
    params.sel = &selector;
    FilteredSearchMockGuard mock(StubCountingFilteredSearch);

    for (bool filtered : {false, true})
    {
        SCOPED_TRACE(filtered ? "selector" : "ordinary");
        for (int k : {4097, 16384, 65535, 65536})
        {
            SCOPED_TRACE(k);
            TopKSearchBuffers buffers(static_cast<size_t>(nq) * static_cast<size_t>(k));
            topKSearchCall.reset();
            ASSERT_NO_THROW(index.search(nq, query.data(), k, buffers.distanceData(), buffers.labelData(),
                                         filtered ? &params : nullptr));
            EXPECT_EQ(topKSearchCall.count.load(), 1);
            EXPECT_EQ(topKSearchCall.n.load(), nq);
            EXPECT_EQ(topKSearchCall.k.load(), k);
            EXPECT_EQ(topKSearchCall.filterMode.load(),
                      filtered ? aicpu::RABITQ_ID_FILTER_RANGE : aicpu::RABITQ_ID_FILTER_NONE);
            EXPECT_EQ(topKSearchCall.refineSources.load(), nullptr);
            buffers.expectResults(nq, k);
        }
    }
}

TEST(TestAscendIndexIVFRaBitQ, PublicSearchRejectsInvalidTopKBeforeBackendAndOutputWrite)
{
    ASSERT_EQ(faiss::ascend::MAX_K, 65536);
    auto source = MakeSequentialPretrainedCpuIndex();
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    ASSERT_TRUE(index.is_trained);
    const auto query = MakePatternData(1, DIM);
    faiss::IDSelectorRange selector(0, 3);
    faiss::SearchParameters params;
    params.sel = &selector;
    FilteredSearchMockGuard mock(StubCountingFilteredSearch);
    const faiss::idx_t invalidTopKs[] = {0, -1, 65537, std::numeric_limits<faiss::idx_t>::max()};

    for (bool filtered : {false, true})
    {
        SCOPED_TRACE(filtered ? "selector" : "ordinary");
        for (faiss::idx_t k : invalidTopKs)
        {
            SCOPED_TRACE(k);
            // A real workspace holds even MAX_K+1. max(idx_t) must be rejected
            // before narrowing or writing; it must not require an impossible allocation.
            TopKSearchBuffers buffers(static_cast<size_t>(faiss::ascend::MAX_K) + 1);
            topKSearchCall.reset();
            ExpectExceptionContains(
                [&] {
                    index.search(1, query.data(), k, buffers.distanceData(), buffers.labelData(),
                                 filtered ? &params : nullptr);
                },
                "k must be > 0 and <= 65536");
            EXPECT_EQ(topKSearchCall.count.load(), 0);
            buffers.expectUnchanged();
        }
    }
}

TEST(TestAscendIndexIVFRaBitQ, RefineTopK65536AllowsInternalCandidatesAbovePublicLimit)
{
    ASSERT_EQ(faiss::ascend::MAX_K, 65536);
    auto source = MakeSequentialPretrainedCpuIndex();
    auto config = MakeConfig();
    config.needRefine = true;
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, config);
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    ASSERT_TRUE(index.is_trained);
    const auto originals = MakePatternData(source.ids.size(), DIM);
    ASSERT_NO_THROW(index.attachRefineSources(originals.data(), source.ids.size()));
    faiss::IDSelectorRange selector(0, 3);
    faiss::SearchParameters params;
    params.sel = &selector;
    FilteredSearchMockGuard mock(StubCountingFilteredSearch);
    struct RefineCase
    {
        int k;
        float alpha;
        int candidateK;
    };
    // Preserve the existing float multiplication for fractional alpha too:
    // 10 * 1.3f rounds to 13 before validation and conversion to int.
    const RefineCase cases[] = {{65536, 2.0f, 131072}, {65536, 1.0f, 65536}, {10, 1.3f, 13}};

    for (const auto& item : cases)
    {
        SCOPED_TRACE(item.k);
        SCOPED_TRACE(item.alpha);
        const int k = item.k;
        index.impl_->ivfrabitqConfig.refineAlpha = item.alpha;
        const int candidateK = item.candidateK;
        for (bool filtered : {false, true})
        {
            SCOPED_TRACE(filtered ? "selector" : "ordinary");
            TopKSearchBuffers buffers(static_cast<size_t>(k));
            topKSearchCall.reset();
            ASSERT_NO_THROW(index.search(1, originals.data(), k, buffers.distanceData(), buffers.labelData(),
                                         filtered ? &params : nullptr));
            EXPECT_EQ(topKSearchCall.count.load(), 1);
            EXPECT_EQ(topKSearchCall.n.load(), 1);
            EXPECT_EQ(topKSearchCall.k.load(), candidateK);
            EXPECT_EQ(topKSearchCall.filterMode.load(),
                      filtered ? aicpu::RABITQ_ID_FILTER_RANGE : aicpu::RABITQ_ID_FILTER_NONE);
            EXPECT_EQ(topKSearchCall.refineSources.load(), originals.data());
            buffers.expectResults(1, k);
        }
    }
}

TEST(TestAscendIndexIVFRaBitQ, RefineRejectsInvalidCandidateCountBeforeBackendAndOutputWrite)
{
    ASSERT_EQ(faiss::ascend::MAX_K, 65536);
    auto source = MakeSequentialPretrainedCpuIndex();
    auto config = MakeConfig();
    config.needRefine = true;
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, config);
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    ASSERT_TRUE(index.is_trained);
    const auto originals = MakePatternData(source.ids.size(), DIM);
    ASSERT_NO_THROW(index.attachRefineSources(originals.data(), source.ids.size()));
    faiss::IDSelectorRange selector(0, 3);
    faiss::SearchParameters params;
    params.sel = &selector;
    FilteredSearchMockGuard mock(StubCountingFilteredSearch);
    const int k = 65536;
    // 32768 * 65536 is INT_MAX+1 exactly: a float comparison against
    // float(INT_MAX) can round the bound up and accidentally permit this value.
    const float invalidAlphas[] = {
        std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), 0.0f, 0.5f, 32768.0f,
        std::numeric_limits<float>::max()};

    for (float alpha : invalidAlphas)
    {
        SCOPED_TRACE(alpha);
        index.impl_->ivfrabitqConfig.refineAlpha = alpha;
        for (bool filtered : {false, true})
        {
            SCOPED_TRACE(filtered ? "selector" : "ordinary");
            TopKSearchBuffers buffers(static_cast<size_t>(k));
            topKSearchCall.reset();
            ExpectExceptionContains(
                [&] {
                    index.search(1, originals.data(), k, buffers.distanceData(), buffers.labelData(),
                                 filtered ? &params : nullptr);
                },
                "Invalid refine candidate count");
            EXPECT_EQ(topKSearchCall.count.load(), 0);
            buffers.expectUnchanged();
        }
    }
}

void ExpectOrthogonal(const std::vector<float>& matrix, int rows, int cols)
{
    ASSERT_EQ(matrix.size(), static_cast<size_t>(rows * cols));
    for (int rowA = 0; rowA < rows; ++rowA)
    {
        for (int rowB = 0; rowB < rows; ++rowB)
        {
            float dot = 0.0f;
            for (int col = 0; col < cols; ++col)
            {
                dot += matrix[rowA * cols + col] * matrix[rowB * cols + col];
            }
            EXPECT_NEAR(dot, rowA == rowB ? 1.0f : 0.0f, 1e-4f);
        }
    }
}

TEST(TestAscendIndexIVFRaBitQ, ConfigurationAndConstructorAcceptSupportedValues)
{
    faiss::ascend::AscendIndexIVFRaBitQConfig defaults;
    EXPECT_TRUE(defaults.useRandomOrthogonalMatrix);
    EXPECT_FALSE(defaults.needRefine);
    EXPECT_EQ(defaults.matrixSeed, 12345);
    EXPECT_FLOAT_EQ(defaults.refineAlpha, 2.0f);

    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    EXPECT_EQ(index.d, DIM);
    EXPECT_EQ(index.getNumLists(), NLIST);
    EXPECT_EQ(index.getNumProbes(), 64);
    EXPECT_FALSE(index.is_trained);
}

TEST(TestAscendIndexIVFRaBitQ, ConstructorValidatesEveryPublicParameter)
{
    ExpectExceptionContains([]
                            { faiss::ascend::AscendIndexIVFRaBitQ index(130, faiss::METRIC_L2, NLIST, MakeConfig()); },
                            "should be divisible by 16");
    ExpectExceptionContains([] { faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, 512, MakeConfig()); },
                            "Unsupported nlists");
    ExpectExceptionContains([]
                            { faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L1, NLIST, MakeConfig()); },
                            "Unsupported metric type");
    ExpectExceptionContains(
        [] { faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig(std::vector<int>{})); },
        "device list should be in range");
}

TEST(TestAscendIndexIVFRaBitQ, FacadeRejectsMissingImplementation)
{
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    index.impl_.reset();
    std::vector<float> vector(DIM, 0.0f);
    faiss::idx_t id = 1;

    ExpectExceptionContains([&] { index.train(1, vector.data()); }, "impl_ is nullptr");
    ExpectExceptionContains([&] { index.copyFrom(nullptr); }, "impl_ is nullptr");
    ExpectExceptionContains([&] { index.copyTo(nullptr); }, "impl_ is nullptr");
    ExpectExceptionContains([&] { index.remove_ids(1, &id); }, "impl_ is nullptr");
    ExpectExceptionContains([&] { index.update(1, vector.data(), &id); }, "impl_ is nullptr");
}

TEST(TestAscendIndexIVFRaBitQ, RandomHelpersAreDeterministicAndProduceOrthogonalMatrices)
{
    faiss::ascend::RandomGenerator first(42);
    faiss::ascend::RandomGenerator second(42);
    EXPECT_EQ(first.rand_int(), second.rand_int());
    EXPECT_EQ(first.rand_int64(), second.rand_int64());
    EXPECT_EQ(first.rand_int(17), second.rand_int(17));
    EXPECT_FLOAT_EQ(first.rand_float(), second.rand_float());
    EXPECT_DOUBLE_EQ(first.rand_double(), second.rand_double());

    std::vector<float> square;
    faiss::ascend::orthonormalinit(square, 123, 4, 4);
    ExpectOrthogonal(square, 4, 4);

    std::vector<float> tightFrame;
    faiss::ascend::orthonormalinit(tightFrame, 123, 2, 4);
    ASSERT_EQ(tightFrame.size(), 8U);
    EXPECT_TRUE(std::all_of(tightFrame.begin(), tightFrame.end(), [](float value) { return std::isfinite(value); }));
}

TEST(TestAscendIndexIVFRaBitQ, IdentityMatrixMappingAndUpdateValidationNeedNoDeviceWork)
{
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig({0, 1}));
    auto impl = index.impl_;

    EXPECT_EQ(impl->getAddElementSize(), static_cast<size_t>(DIM) * sizeof(float));
    EXPECT_EQ(impl->getAddPagedSize(7), 7U);

    std::vector<float> identity(16, 0.0f);
    impl->randomOrthogonalGivens(4, identity);
    EXPECT_THAT(identity, ElementsAre(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1));

    // deviceIndex is the logical deviceList index. An id that was never written simply misses: the old
    // findDeviceId guessed one with id % deviceCnt, LocationTable does not guess.
    for (faiss::idx_t mappedId : {10, 20, 30})
    {
        impl->locationTable->setLocation(mappedId, 1, 0, 0);
    }
    const auto loc = impl->locationTable->lookupLocation(20);
    ASSERT_TRUE(loc.has_value());
    EXPECT_EQ(loc->deviceIndex(), 1U);
    EXPECT_FALSE(impl->locationTable->contains(5));
    impl->locationTable->clearValid(20);
    impl->locationTable->clearValid(999);
    EXPECT_FALSE(impl->locationTable->contains(20));

    std::vector<float> vector(DIM, 0.25f);
    faiss::idx_t id = 404;
    ExpectExceptionContains([&] { index.update(1, nullptr, &id); }, "vector list is nullptr");
    ExpectExceptionContains([&] { index.update(1, vector.data(), nullptr); }, "vector ID list is nullptr");
    ExpectExceptionContains([&] { index.update(0, vector.data(), &id); }, "greater than 0");
    ExpectExceptionContains([&] { index.update(1, vector.data(), &id); }, "not trained");
    index.is_trained = true;
    EXPECT_THAT(index.update(1, vector.data(), &id), ElementsAre(id));
    ExpectExceptionContains([&] { impl->addPaged(1, nullptr, &id); }, "x cannot be nullptr");
}

TEST(TestAscendIndexIVFRaBitQ, MergeSearchResultsHonorsMetricOrdering)
{
    faiss::ascend::AscendIndexIVFRaBitQ l2Index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    std::vector<std::vector<float>> l2Distances = {{1.0f, 4.0f}, {2.0f, 3.0f}};
    std::vector<std::vector<faiss::ascend::ascend_idx_t>> l2Labels = {{10, 40}, {20, 30}};
    std::vector<float> mergedDistances(4);
    std::vector<faiss::idx_t> mergedLabels(4);
    l2Index.impl_->mergeSearchResult(2, l2Distances, l2Labels, 1, 4, mergedDistances.data(), mergedLabels.data());
    EXPECT_THAT(mergedDistances, ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));
    EXPECT_THAT(mergedLabels, ElementsAre(10, 20, 30, 40));
    ExpectExceptionContains(
        [&] {
            l2Index.impl_->mergeSearchResult(1, l2Distances, l2Labels, 1, 3, mergedDistances.data(),
                                             mergedLabels.data());
        },
        "must be >= k");

    faiss::ascend::AscendIndexIVFRaBitQ ipIndex(DIM, faiss::METRIC_INNER_PRODUCT, NLIST, MakeConfig());
    std::vector<std::vector<float>> ipDistances = {{9.0f, 5.0f}, {8.0f, 7.0f}};
    std::vector<std::vector<faiss::ascend::ascend_idx_t>> ipLabels = {{90, 50}, {80, 70}};
    ipIndex.impl_->mergeSearchResult(2, ipDistances, ipLabels, 1, 4, mergedDistances.data(), mergedLabels.data());
    EXPECT_THAT(mergedDistances, ElementsAre(9.0f, 8.0f, 7.0f, 5.0f));
    EXPECT_THAT(mergedLabels, ElementsAre(90, 80, 70, 50));
}

TEST(TestAscendIndexIVFRaBitQ, CopyFromAndCopyToRoundTripPreencodedData)
{
    auto source = MakePretrainedCpuIndex();
    faiss::ascend::AscendIndexIVFRaBitQ deviceIndex(DIM, faiss::METRIC_L2, NLIST, MakeConfig());

    ASSERT_NO_THROW(deviceIndex.copyFrom(source.index.get()));
    EXPECT_TRUE(deviceIndex.is_trained);
    EXPECT_EQ(deviceIndex.ntotal, static_cast<faiss::idx_t>(source.ids.size()));
    const auto loc = deviceIndex.impl_->locationTable->lookupLocation(source.ids[0]);
    ASSERT_TRUE(loc.has_value());
    EXPECT_EQ(loc->deviceIndex(), 0U);
    ASSERT_EQ(deviceIndex.impl_->orthogonalMatrix.size(), static_cast<size_t>(DIM * DIM));
    EXPECT_FLOAT_EQ(deviceIndex.impl_->orthogonalMatrix.front(), 1.0f);
    EXPECT_FLOAT_EQ(deviceIndex.impl_->orthogonalMatrix[DIM + 1], 1.0f);
    EXPECT_FLOAT_EQ(deviceIndex.impl_->orthogonalMatrix[1], 0.0f);

    // copyTo installs and owns both the quantizer and inverted lists.
    faiss::IndexIVFRaBitQ output;
    ASSERT_NO_THROW(deviceIndex.copyTo(&output));
    EXPECT_EQ(output.d, DIM);
    EXPECT_EQ(output.metric_type, faiss::METRIC_L2);
    EXPECT_EQ(output.nlist, static_cast<size_t>(NLIST));
    EXPECT_EQ(output.nprobe, static_cast<size_t>(NPROBE));
    EXPECT_EQ(output.ntotal, static_cast<faiss::idx_t>(source.ids.size()));
    EXPECT_TRUE(output.is_trained);
    EXPECT_EQ(output.code_size, source.index->code_size);
    ASSERT_EQ(output.invlists->list_size(source.listId), source.ids.size());

    const faiss::idx_t* outputIds = output.invlists->get_ids(source.listId);
    const uint8_t* outputCodes = output.invlists->get_codes(source.listId);
    EXPECT_TRUE(std::equal(source.ids.begin(), source.ids.end(), outputIds));
    EXPECT_TRUE(std::equal(source.codes.begin(), source.codes.end(), outputCodes));

    std::vector<float> vector(DIM, 0.0f);
    ASSERT_NO_THROW(deviceIndex.train(1, vector.data()));
}

TEST(TestAscendIndexIVFRaBitQ, AttachedRefineSourcesRejectMutationAndResetReleasesSources)
{
    auto config = MakeConfig();
    config.needRefine = true;
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, config);
    std::vector<float> data(DIM, 1.0f);
    index.ntotal = 1;
    index.impl_->locationTable->setLocation(0, 0, 0, 0);
    index.attachRefineSources(data.data(), 1);
    EXPECT_TRUE(index.impl_->refineSourcesAttached());
    ExpectExceptionContains([&] { index.loadRefineSources(data.data(), 1); }, "conflicts");
    faiss::idx_t id = 0;
    ExpectExceptionContains([&] { index.impl_->addPaged(1, data.data(), &id); }, "conflicts");
    ExpectExceptionContains([&] { index.impl_->deleteImpl(1, &id); }, "conflicts");
    ExpectExceptionContains([&] { index.impl_->update(1, data.data(), &id); }, "conflicts");
    EXPECT_EQ(index.ntotal, 1);
    index.reset();
    EXPECT_FALSE(index.impl_->refineSourcesAttached());
    EXPECT_TRUE(index.impl_->srcIndexes.empty());
    EXPECT_EQ(index.impl_->attachedSrcRows_, 0);
    index.ntotal = 1;
    index.impl_->locationTable->setLocation(0, 0, 0, 0);
    ASSERT_NO_THROW(index.loadRefineSources(data.data(), 1));
    data[0] = 2.0f;
    EXPECT_FLOAT_EQ(index.impl_->srcIndexes[0], 1.0f);
    ExpectExceptionContains([&] { index.attachRefineSources(data.data(), 1); }, "conflicts");
    index.reset();
    EXPECT_TRUE(index.impl_->srcIndexes.empty());
    index.ntotal = 1;
    index.impl_->locationTable->setLocation(0, 0, 0, 0);
    EXPECT_NO_THROW(index.attachRefineSources(data.data(), 1));
}

TEST(TestAscendIndexIVFRaBitQ, CopiedRefineSourcesRejectMutationAndValidateSearch)
{
    auto source = MakePretrainedCpuIndex();
    const faiss::idx_t ids[] = {0, 1, 2};
    source.index->invlists->update_entries(source.listId, 0, 3, ids, source.codes.data());
    auto config = MakeConfig();
    config.needRefine = true;
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, config);
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    auto originals = MakePatternData(3, DIM);
    std::vector<float> beforeDist(2), afterDist(2);
    std::vector<faiss::idx_t> beforeIds(2), afterIds(2);
    ExpectExceptionContains([&] { index.search(1, originals.data(), 2, beforeDist.data(), beforeIds.data()); },
                            "refine sources are not ready");
    // Missing originals must also prevent appending to an imported index.
    const faiss::idx_t next = 3;
    ExpectExceptionContains([&] { index.add_with_ids(1, originals.data(), &next); }, "not ready");
    ExpectExceptionContains([&] { index.update(1, originals.data(), ids); }, "not ready");
    EXPECT_EQ(index.ntotal, 3);
    ASSERT_NO_THROW(index.loadRefineSources(originals.data(), 3));
    ASSERT_NO_THROW(index.search(1, originals.data(), 2, beforeDist.data(), beforeIds.data()));
    ExpectExceptionContains([&] { index.remove_ids(1, ids); }, "copied refine sources");
    ExpectExceptionContains([&] { index.update(1, originals.data() + DIM, ids); }, "copied refine sources");
    EXPECT_EQ(index.ntotal, 3);
    EXPECT_EQ(index.impl_->srcIndexes, originals);
    ASSERT_NO_THROW(index.search(1, originals.data(), 2, afterDist.data(), afterIds.data()));
    EXPECT_EQ(beforeDist, afterDist);
    EXPECT_EQ(beforeIds, afterIds);
    // Validate the entire ID batch before any vector is added.
    const faiss::idx_t invalid[] = {3, 5};
    ExpectExceptionContains([&] { index.add_with_ids(2, originals.data(), invalid); }, "sequential IDs");
    EXPECT_EQ(index.ntotal, 3);
    EXPECT_EQ(index.impl_->srcIndexes, originals);
    ASSERT_NO_THROW(index.add_with_ids(1, originals.data(), &next));
    EXPECT_EQ(index.ntotal, 4);
    EXPECT_EQ(index.impl_->srcIndexes.size(), 4U * DIM);
    index.impl_->srcIndexes.pop_back();
    ExpectExceptionContains([&] { index.search(1, originals.data(), 2, afterDist.data(), afterIds.data()); },
                            "refine sources are not ready");
    ASSERT_NO_THROW(index.loadRefineSources(MakePatternData(4, DIM).data(), 4));
    ASSERT_NO_THROW(index.reset());
    // Both implicit IDs and explicit sequential IDs support incremental add.
    ASSERT_NO_THROW(index.add(1, originals.data()));
    const faiss::idx_t one = 1;
    ASSERT_NO_THROW(index.add_with_ids(1, originals.data() + DIM, &one));
    EXPECT_EQ(index.ntotal, 2);
    EXPECT_TRUE(std::equal(originals.begin(), originals.begin() + 2 * DIM, index.impl_->srcIndexes.begin()));
    ASSERT_NO_THROW(index.search(1, originals.data(), 2, afterDist.data(), afterIds.data()));
}

TEST(TestAscendIndexIVFRaBitQ, RefineSourceLoadingRejectsSparseAndDuplicateIds)
{
    for (const std::vector<faiss::idx_t>& ids :
         {std::vector<faiss::idx_t>{0, 1, 3}, std::vector<faiss::idx_t>{0, 0, 1}})
    {
        auto source = MakePretrainedCpuIndex();
        source.index->invlists->update_entries(source.listId, 0, ids.size(), ids.data(), source.codes.data());
        faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
        if (ids[0] == ids[1])
        {
            ExpectExceptionContains([&] { index.copyFrom(source.index.get()); }, "duplicate id");
            EXPECT_EQ(index.ntotal, 0);
            continue;
        }
        ASSERT_NO_THROW(index.copyFrom(source.index.get()));
        auto originals = MakePatternData(3, DIM);
        ExpectExceptionContains([&] { index.loadRefineSources(originals.data(), 3); }, "unique IDs");
        ExpectExceptionContains([&] { index.attachRefineSources(originals.data(), 3); }, "unique IDs");
        EXPECT_TRUE(index.impl_->srcIndexes.empty());
        EXPECT_FALSE(index.impl_->refineSourcesAttached());
        EXPECT_EQ(index.ntotal, 3);
    }
}

TEST(TestAscendIndexIVFRaBitQ, AttachedRefineSourcesValidateRowCountBeforeSearch)
{
    auto source = MakePretrainedCpuIndex(false);
    const faiss::idx_t id = 0;
    std::vector<uint8_t> code(source.index->code_size, 0);
    source.index->invlists->add_entries(source.listId, 1, &id, code.data());
    source.index->ntotal = 1;
    auto config = MakeConfig();
    config.needRefine = true;
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, config);
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    auto originals = MakePatternData(1, DIM);
    ASSERT_NO_THROW(index.attachRefineSources(originals.data(), 1));
    float distance = 0;
    faiss::idx_t label = -1;
    ASSERT_NO_THROW(index.search(1, originals.data(), 1, &distance, &label));
    // Simulate inconsistent metadata; a non-null pointer alone must not pass.
    index.impl_->attachedSrcRows_ = 0;
    ExpectExceptionContains([&] { index.search(1, originals.data(), 1, &distance, &label); }, "row count");
    EXPECT_EQ(index.ntotal, 1);
}

TEST(TestAscendIndexIVFRaBitQ, MultiBitCarrierRoundTripsAndRejectsInvalidLayouts)
{
    const auto& soc = faiss::ascend::SocUtils::GetInstance();
    if (soc.IsAscendA5() || soc.IsAscend310P()) GTEST_SKIP();
    for (int bits : {2, 4})
    {
        auto config = MakeConfig();
        config.quantBits = bits;
        faiss::ascend::AscendIndexIVFRaBitQ source(DIM, faiss::METRIC_L2, NLIST, config);
        auto actual = source.impl_->getActualIndex(0);
        auto centroids = MakePatternData(NLIST, DIM);
        ASSERT_EQ(actual->updateCoarseCenterImpl(centroids), ::ascend::APP_ERR_OK);
        const ::ascend::Index::idx_t id = 0;
        std::vector<uint8_t> raw(actual->getCodeSize(), 0);
        for (size_t i = 0; i < static_cast<size_t>(DIM * bits / 8); ++i) raw[i] = static_cast<uint8_t>(i);
        ASSERT_EQ(actual->addEncodedVectors(9, 1, raw.data(), &id), ::ascend::APP_ERR_OK);
        source.is_trained = true;
        source.ntotal = 1;
        faiss::IndexIVFRaBitQ carrier;
        ASSERT_NO_THROW(source.copyTo(&carrier));
        EXPECT_EQ(carrier.rabitq.nb_bits, static_cast<size_t>(bits));
        EXPECT_EQ(carrier.code_size, raw.size() + sizeof(uint64_t) + sizeof(uint32_t));
        faiss::ascend::AscendIndexIVFRaBitQ restored(DIM, faiss::METRIC_L2, NLIST, config);
        ASSERT_NO_THROW(restored.copyFrom(&carrier));
        std::vector<uint8_t> restoredRaw;
        ASSERT_EQ(restored.impl_->getActualIndex(0)->getListRawCodes(9, restoredRaw), ::ascend::APP_ERR_OK);
        EXPECT_EQ(restoredRaw, raw);
        faiss::IndexIVFRaBitQ roundTrip;
        ASSERT_NO_THROW(restored.copyTo(&roundTrip));
        const auto* before = carrier.invlists->get_codes(9);
        const auto* after = roundTrip.invlists->get_codes(9);
        EXPECT_EQ(std::memcmp(before, after, carrier.code_size), 0);
        std::vector<uint8_t> invalid(before, before + carrier.code_size);
        carrier.invlists->release_codes(9, before);
        roundTrip.invlists->release_codes(9, after);
        std::vector<float> refine(DIM, 0.0f);
        restored.attachRefineSources(refine.data(), 1);
        ASSERT_NO_THROW(restored.copyFrom(&carrier));
        EXPECT_FALSE(restored.impl_->refineSourcesAttached());
        ASSERT_NO_THROW(restored.loadRefineSources(refine.data(), 1));
        ASSERT_NO_THROW(restored.copyFrom(&carrier));
        EXPECT_TRUE(restored.impl_->srcIndexes.empty());
        faiss::ascend::AscendIndexIVFRaBitQ wrongBits(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
        ExpectExceptionContains([&] { wrongBits.copyFrom(&carrier); }, "quantBits mismatch");
        ++carrier.code_size;
        ExpectExceptionContains([&] { restored.copyFrom(&carrier); }, "code size mismatch");
        --carrier.code_size;
        // A native or damaged carrier without the Ascend trailer must be rejected.
        invalid[raw.size()] ^= 1;
        carrier.invlists->update_entries(9, 0, 1, &id, invalid.data());
        ExpectExceptionContains([&] { restored.copyFrom(&carrier); }, "not an Ascend carrier");
        EXPECT_EQ(restored.ntotal, 1);
    }
}

TEST(TestAscendIndexIVFRaBitQ, CopyOperationsRejectInvalidIndexes)
{
    faiss::ascend::AscendIndexIVFRaBitQ deviceIndex(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    ExpectExceptionContains([&] { deviceIndex.copyFrom(nullptr); }, "Input index is nullptr");

    faiss::IndexFlatL2 differentDimQuantizer(64);
    faiss::IndexIVFRaBitQ differentDim(&differentDimQuantizer, 64, NLIST);
    ExpectExceptionContains([&] { deviceIndex.copyFrom(&differentDim); }, "Dimension mismatch");

    auto untrained = MakePretrainedCpuIndex(false);
    untrained.index->is_trained = false;
    ExpectExceptionContains([&] { deviceIndex.copyFrom(untrained.index.get()); }, "Index must be trained");
    ExpectExceptionContains([&] { deviceIndex.copyTo(nullptr); }, "Output index is nullptr");
}

TEST(TestAscendIndexIVFRaBitQ, PretrainedAddUpdateAndRemoveRunEndToEndWithoutNpu)
{
    // copyFrom installs deterministic centroids in both the assignment index
    // and the RaBitQ device index, so raw-vector mutation can run entirely on
    // AscendCLMock without expensive training.
    auto source = MakePretrainedCpuIndex(false);
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));

    const auto vectors = MakePatternData(4, DIM);
    const std::vector<faiss::idx_t> ids = {8101, 8102, 8103, 8104};
    ASSERT_NO_THROW(index.add_with_ids(ids.size(), vectors.data(), ids.data()));
    EXPECT_EQ(index.ntotal, static_cast<faiss::idx_t>(ids.size()));
    for (faiss::idx_t id : ids)
    {
        const auto loc = index.impl_->locationTable->lookupLocation(id);
        ASSERT_TRUE(loc.has_value());
        EXPECT_EQ(loc->deviceIndex(), 0U);
    }

    std::vector<float> distances(2, std::numeric_limits<float>::infinity());
    std::vector<faiss::idx_t> labels(2, -1);
    ASSERT_NO_THROW(index.search(1, vectors.data(), 2, distances.data(), labels.data()));
    EXPECT_TRUE(std::all_of(distances.begin(), distances.end(), [](float value) { return std::isfinite(value); }));
    EXPECT_TRUE(std::all_of(labels.begin(), labels.end(),
                            [&](faiss::idx_t label) { return std::find(ids.begin(), ids.end(), label) != ids.end(); }));

    std::vector<float> replacement(DIM, 0.5f);
    EXPECT_TRUE(index.update(1, replacement.data(), &ids[1]).empty());
    EXPECT_EQ(index.ntotal, static_cast<faiss::idx_t>(ids.size()));

    ASSERT_NO_THROW(index.remove_ids(1, &ids[2]));
    EXPECT_EQ(index.ntotal, static_cast<faiss::idx_t>(ids.size() - 1));
    EXPECT_FALSE(index.impl_->locationTable->contains(ids[2]));
}

// The host's LocationTable is the only id -> slot mapping, so a delete has to pass the exact slot
// down and fold the daemon's tail-backfill report back in. copyFrom lands the ids contiguously from
// slot 0, so the expected slots here are known rather than probed.
TEST(TestAscendIndexIVFRaBitQ, DeleteKeepsHostSlotsInStepWithTailBackfill)
{
    auto source = MakePretrainedCpuIndex();
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    ASSERT_EQ(source.ids.size(), 3U);

    for (size_t i = 0; i < source.ids.size(); ++i)
    {
        const auto loc = index.impl_->locationTable->lookupLocation(source.ids[i]);
        ASSERT_TRUE(loc.has_value());
        EXPECT_EQ(loc->listId(), static_cast<uint16_t>(source.listId));
        EXPECT_EQ(loc->slot(), static_cast<uint32_t>(i));
    }

    // Deleting the middle id leaves a hole at slot 1, which the tail id is backfilled into.
    ASSERT_NO_THROW(index.remove_ids(1, &source.ids[1]));
    EXPECT_FALSE(index.impl_->locationTable->contains(source.ids[1]));
    const auto firstLoc = index.impl_->locationTable->lookupLocation(source.ids[0]);
    ASSERT_TRUE(firstLoc.has_value());
    EXPECT_EQ(firstLoc->slot(), 0U);
    const auto movedLoc = index.impl_->locationTable->lookupLocation(source.ids[2]);
    ASSERT_TRUE(movedLoc.has_value());
    EXPECT_EQ(movedLoc->listId(), static_cast<uint16_t>(source.listId));
    EXPECT_EQ(movedLoc->slot(), 1U);

    // The daemon's own storage agrees, so a follow-up delete by that slot hits the right vector.
    std::vector<int64_t> listIds;
    ASSERT_EQ(index.impl_->getActualIndex(0)->getListIds(source.listId, listIds), ::ascend::APP_ERR_OK);
    EXPECT_THAT(listIds, ElementsAre(source.ids[0], source.ids[2]));

    ASSERT_NO_THROW(index.remove_ids(1, &source.ids[2]));
    EXPECT_FALSE(index.impl_->locationTable->contains(source.ids[2]));
    ASSERT_EQ(index.impl_->getActualIndex(0)->getListIds(source.listId, listIds), ::ascend::APP_ERR_OK);
    EXPECT_THAT(listIds, ElementsAre(source.ids[0]));
}

// remove_ids(IDSelector) reaches the index through the faiss::Index base, so it must land on the same
// slot-precise path; otherwise the daemon would compact without reporting its tail backfill.
TEST(TestAscendIndexIVFRaBitQ, SelectorDeleteKeepsHostSlotsInStep)
{
    auto source = MakePretrainedCpuIndex();
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));

    faiss::IDSelectorBatch sel(1, &source.ids[1]);
    EXPECT_EQ(static_cast<faiss::Index&>(index).remove_ids(sel), 1U);

    EXPECT_FALSE(index.impl_->locationTable->contains(source.ids[1]));
    const auto movedLoc = index.impl_->locationTable->lookupLocation(source.ids[2]);
    ASSERT_TRUE(movedLoc.has_value());
    EXPECT_EQ(movedLoc->slot(), 1U);

    std::vector<int64_t> listIds;
    ASSERT_EQ(index.impl_->getActualIndex(0)->getListIds(source.listId, listIds), ::ascend::APP_ERR_OK);
    EXPECT_THAT(listIds, ElementsAre(source.ids[0], source.ids[2]));
}

// A batch that names the same list twice must append to the copy the first request made, not clone the
// base list again -- otherwise the second append would silently overwrite the first.
TEST(TestIndexIVFRaBitQCore, EncodedBatchPublishesOnceAndAppendsRepeatedLists)
{
    faiss::ascend::AscendIndexIVFRaBitQ hostIndex(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    auto actual = hostIndex.impl_->getActualIndex(0);
    ASSERT_NE(actual, nullptr);

    const size_t codeSize = actual->getCodeSize();
    auto packed = [codeSize](size_t numVecs, uint8_t seed)
    {
        std::vector<uint8_t> buf(numVecs * codeSize);
        for (size_t i = 0; i < buf.size(); ++i)
        {
            buf[i] = static_cast<uint8_t>((i * 7 + seed) % 251);
        }
        return buf;
    };
    auto firstCodes = packed(2, 1);
    auto secondCodes = packed(1, 2);
    auto thirdCodes = packed(1, 3);
    std::vector<::ascend::Index::idx_t> firstIds = {11, 12};
    std::vector<::ascend::Index::idx_t> secondIds = {21};
    std::vector<::ascend::Index::idx_t> thirdIds = {13};

    std::vector<::ascend::IndexIVFRaBitQ::ListEncodedAddRequest> reqs = {
        {3, firstIds.size(), firstCodes.data(), firstIds.data()},
        {7, secondIds.size(), secondCodes.data(), secondIds.data()},
        {3, thirdIds.size(), thirdCodes.data(), thirdIds.data()}};

    const uint64_t verBefore = actual->activeSnapshot->publishVer;
    std::vector<size_t> baseSlots(reqs.size(), 0);
    ASSERT_EQ(actual->addEncodedVectorsBatch(reqs.data(), reqs.size(), baseSlots.data()), ::ascend::APP_ERR_OK);

    EXPECT_EQ(actual->activeSnapshot->publishVer, verBefore + 1);
    EXPECT_EQ(actual->activeSnapshot->total, 4U);
    EXPECT_EQ(actual->activeSnapshot->maxStoredId, 21U);
    EXPECT_EQ(actual->maxListLength, ::ascend::utils::roundUp(3, ::ascend::CUBE_ALIGN));
    EXPECT_THAT(baseSlots, ElementsAre(0U, 0U, 2U));

    std::vector<int64_t> listIds;
    ASSERT_EQ(actual->getListIds(3, listIds), ::ascend::APP_ERR_OK);
    EXPECT_THAT(listIds, ElementsAre(11, 12, 13));
    ASSERT_EQ(actual->getListIds(7, listIds), ::ascend::APP_ERR_OK);
    EXPECT_THAT(listIds, ElementsAre(21));
    std::vector<uint8_t> expected(3 * codeSize);
    const size_t codeBytes = codeSize - 2 * sizeof(float);
    for (size_t part = 0; part < 3; ++part)
    {
        const size_t width = part == 0 ? codeBytes : sizeof(float);
        const size_t firstOffset = part == 0 ? 0 : 2 * codeBytes + (part - 1) * 2 * sizeof(float);
        const size_t thirdOffset = part == 0 ? 0 : codeBytes + (part - 1) * sizeof(float);
        const size_t outputOffset = part == 0 ? 0 : 3 * codeBytes + (part - 1) * 3 * sizeof(float);
        std::copy_n(firstCodes.data() + firstOffset, 2 * width, expected.data() + outputOffset);
        std::copy_n(thirdCodes.data() + thirdOffset, width, expected.data() + outputOffset + 2 * width);
    }
    std::vector<uint8_t> raw;
    ASSERT_EQ(actual->getListRawCodes(3, raw), ::ascend::APP_ERR_OK);
    EXPECT_EQ(raw, expected);  // Codes, L2 and L1 all survive the repeated-list append.
    ASSERT_EQ(actual->getListRawCodes(7, raw), ::ascend::APP_ERR_OK);
    EXPECT_EQ(raw, secondCodes);
    const auto* published = actual->activeSnapshot.get();
    const auto emptyVersion = published->publishVer;
    const int emptyMaxLength = actual->maxListLength;
    const ::ascend::IndexIVFRaBitQ::ListEncodedAddRequest emptyReqs[] = {
        {3, 0, nullptr, nullptr}, {7, 0, nullptr, nullptr}, {3, 0, nullptr, nullptr}};
    size_t emptyBases[3] = {};
    ASSERT_EQ(actual->addEncodedVectorsBatch(emptyReqs, 3, emptyBases), ::ascend::APP_ERR_OK);
    EXPECT_THAT(emptyBases, ElementsAre(3U, 1U, 3U));
    EXPECT_EQ(actual->activeSnapshot.get(), published);
    EXPECT_EQ(actual->activeSnapshot->publishVer, emptyVersion);
    EXPECT_EQ(actual->activeSnapshot->total, 4U);
    EXPECT_EQ(actual->activeSnapshot->maxStoredId, 21U);
    EXPECT_EQ(actual->maxListLength, emptyMaxLength);
    EXPECT_EQ(actual->addEncodedVectorsBatch(nullptr, 0, nullptr), ::ascend::APP_ERR_OK);
}

// A rejected request must leave the index exactly as it was: the clone is dropped unpublished, so the
// earlier requests in the same batch have to vanish with it.
TEST(TestIndexIVFRaBitQCore, EncodedBatchFailureLeavesActiveSnapshotUntouched)
{
    faiss::ascend::AscendIndexIVFRaBitQ hostIndex(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    auto actual = hostIndex.impl_->getActualIndex(0);
    std::vector<uint8_t> originalCodes(2 * actual->getCodeSize(), 0x37);
    const ::ascend::Index::idx_t originalIds[] = {15, 16};
    ASSERT_EQ(actual->addEncodedVectors(5, 2, originalCodes.data(), originalIds), ::ascend::APP_ERR_OK);
    std::vector<uint8_t> payloadBefore;
    std::vector<int64_t> idsBefore;
    ASSERT_EQ(actual->getListRawCodes(5, payloadBefore), ::ascend::APP_ERR_OK);
    ASSERT_EQ(actual->getListIds(5, idsBefore), ::ascend::APP_ERR_OK);
    std::vector<uint8_t> codes(2 * actual->getCodeSize(), 0x5A);
    std::vector<::ascend::Index::idx_t> ids = {static_cast<::ascend::Index::idx_t>(UINT32_MAX) + 1, 32};

    std::vector<::ascend::IndexIVFRaBitQ::ListEncodedAddRequest> reqs = {{5, ids.size(), codes.data(), ids.data()},
                                                                         {NLIST, ids.size(), codes.data(), ids.data()}};

    const uint64_t verBefore = actual->activeSnapshot->publishVer;
    const auto* snapshotBefore = actual->activeSnapshot.get();
    const size_t totalBefore = actual->activeSnapshot->total;
    const auto maxIdBefore = actual->activeSnapshot->maxStoredId;
    const int maxListLengthBefore = actual->maxListLength;
    EXPECT_EQ(actual->addEncodedVectorsBatch(reqs.data(), reqs.size(), nullptr), ::ascend::APP_ERR_INVALID_PARAM);
    EXPECT_EQ(actual->activeSnapshot.get(), snapshotBefore);
    EXPECT_EQ(actual->activeSnapshot->publishVer, verBefore);
    EXPECT_EQ(actual->activeSnapshot->total, totalBefore);
    EXPECT_EQ(actual->activeSnapshot->maxStoredId, maxIdBefore);
    EXPECT_EQ(actual->maxListLength, maxListLengthBefore);
    EXPECT_EQ(actual->getListLength(5), 2U);
    std::vector<uint8_t> payloadAfter;
    std::vector<int64_t> idsAfter;
    ASSERT_EQ(actual->getListRawCodes(5, payloadAfter), ::ascend::APP_ERR_OK);
    ASSERT_EQ(actual->getListIds(5, idsAfter), ::ascend::APP_ERR_OK);
    EXPECT_EQ(payloadAfter, payloadBefore);
    EXPECT_EQ(idsAfter, idsBefore);
}

TEST(TestIndexIVFRaBitQCore, EncodedStorageAccessAndRefineAreDeterministic)
{
    faiss::ascend::AscendIndexIVFRaBitQ hostIndex(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    auto actual = hostIndex.impl_->getActualIndex(0);
    ASSERT_NE(actual, nullptr);
    EXPECT_EQ(actual->getCodeSize(), 24U);

    std::vector<int64_t> ids;
    std::vector<uint8_t> rawCodes;
    EXPECT_EQ(actual->getListIds(3, ids), ::ascend::APP_ERR_OK);
    EXPECT_TRUE(ids.empty());
    EXPECT_EQ(actual->getListRawCodes(3, rawCodes), ::ascend::APP_ERR_OK);
    EXPECT_TRUE(rawCodes.empty());
    EXPECT_EQ(actual->getListIds(-1, ids), ::ascend::APP_ERR_INVALID_PARAM);
    EXPECT_EQ(actual->getListRawCodes(NLIST, rawCodes), ::ascend::APP_ERR_INVALID_PARAM);
    EXPECT_EQ(actual->addEncodedVectors(-1, 0, nullptr, nullptr), ::ascend::APP_ERR_INVALID_PARAM);
    EXPECT_EQ(actual->addEncodedVectors(3, 0, nullptr, nullptr), ::ascend::APP_ERR_OK);

    std::vector<float> sourceVectors(3 * DIM);
    std::fill(sourceVectors.begin(), sourceVectors.begin() + DIM, 0.0f);
    std::fill(sourceVectors.begin() + DIM, sourceVectors.begin() + 2 * DIM, 1.0f);
    std::fill(sourceVectors.begin() + 2 * DIM, sourceVectors.end(), 2.0f);
    std::vector<float> query(DIM, 1.1f);
    std::vector<float> distances(3, 0.0f);
    std::vector<::ascend::Index::idx_t> labels(3, -1);
    std::vector<float> candidateDistances(3, 0.0f);
    std::vector<::ascend::Index::idx_t> candidateLabels = {2, 0, 1};
    actual->refine(1, query.data(), 3, distances.data(), labels.data(), candidateDistances.data(),
                   candidateLabels.data(), sourceVectors.data());
    EXPECT_THAT(labels, ElementsAre(1, 2, 0));
    EXPECT_LT(distances[0], distances[1]);
    EXPECT_LT(distances[1], distances[2]);

    const auto distancesBeforeEmptySearch = distances;
    const auto labelsBeforeEmptySearch = labels;
    EXPECT_EQ(actual->searchImpl(1, query.data(), 0, distances.data(), labels.data(), nullptr), ::ascend::APP_ERR_OK);
    EXPECT_EQ(distances, distancesBeforeEmptySearch);
    EXPECT_EQ(labels, labelsBeforeEmptySearch);
    ASSERT_EQ(actual->reset(), ::ascend::APP_ERR_OK);
    EXPECT_EQ(actual->getListLength(3), 0U);
}

TEST(TestLocationTable, SetLookupClearAndReset)
{
    faiss::ascend::LocationTable table(2, NLIST);

    table.setLocation(42, 1, 7, 300);
    const auto loc = table.lookupLocation(42);
    ASSERT_TRUE(loc.has_value());
    EXPECT_EQ(loc->deviceIndex(), 1U);
    EXPECT_EQ(loc->listId(), 7U);
    EXPECT_EQ(loc->slot(), 300U);

    // An id that was never written misses outright -- no guessing.
    EXPECT_FALSE(table.contains(43));
    EXPECT_FALSE(table.lookupLocation(43).has_value());

    // clearValid only clears the valid bit, and is a no-op for an unallocated id.
    table.clearValid(42);
    EXPECT_FALSE(table.contains(42));
    table.clearValid(1 << 20);

    // Overwriting the same id makes it valid again.
    table.setLocation(42, 0, 3, 11);
    ASSERT_TRUE(table.contains(42));
    EXPECT_EQ(table.lookupLocation(42)->listId(), 3U);

    table.reset();
    EXPECT_FALSE(table.contains(42));
    EXPECT_EQ(table.allocatedBlockCount(), 0U);

    // Out-of-range fields are programming errors, so they throw rather than corrupt the record.
    EXPECT_THROW(table.setLocation(1, 2, 0, 0), std::out_of_range);
    EXPECT_THROW(table.setLocation(1, 0, NLIST, 0), std::out_of_range);
    EXPECT_THROW(table.setLocation(-1, 0, 0, 0), std::out_of_range);
}

// Blocks are keyed by id/kBlockSlots, so a huge id span (> 2^32) costs only what is written rather
// than a maxId/kBlockSlots top-level array.
TEST(TestLocationTable, SparseHugeIdsAllocateOnlyTouchedBlocks)
{
    faiss::ascend::LocationTable table(1, NLIST);
    const int64_t idBase = static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1;

    constexpr size_t COUNT = 16;
    for (size_t i = 0; i < COUNT; ++i)
    {
        table.setLocation(idBase + static_cast<int64_t>(i), 0, 1, static_cast<uint32_t>(i));
    }
    // idBase is an exact multiple of kBlockSlots, so all of these share one block.
    EXPECT_EQ(table.allocatedBlockCount(), 1U);
    for (size_t i = 0; i < COUNT; ++i)
    {
        ASSERT_TRUE(table.contains(idBase + static_cast<int64_t>(i)));
        EXPECT_EQ(table.lookupLocation(idBase + static_cast<int64_t>(i))->slot(), static_cast<uint32_t>(i));
    }

    // A far-away id costs exactly one more block.
    table.setLocation(idBase + 100 * static_cast<int64_t>(faiss::ascend::LocationTable::kBlockSlots), 0, 1, 0);
    EXPECT_EQ(table.allocatedBlockCount(), 2U);
}

TEST(TestLocationTable, ReserveThenApplyDeltaDoesNotAllocate)
{
    faiss::ascend::LocationTable table(1, NLIST);

    faiss::ascend::LocationDelta delta;
    delta.addedIds.emplace_back(5, faiss::ascend::Location(0, 2, 10));
    delta.movedIds.emplace_back(6, faiss::ascend::Location(0, 2, 11));
    table.setLocation(6, 0, 9, 99);
    delta.erasedIds.push_back(6);

    table.reserveForIds({5, 6});
    const size_t blocksBefore = table.allocatedBlockCount();
    ASSERT_NO_THROW(table.validateDelta(delta));
    table.applyLocationDeltaNoFail(delta);
    EXPECT_EQ(table.allocatedBlockCount(), blocksBefore);

    // erasedIds clears valid first, then movedIds/addedIds write and set it again.
    ASSERT_TRUE(table.contains(5));
    EXPECT_EQ(table.lookupLocation(5)->slot(), 10U);
    ASSERT_TRUE(table.contains(6));
    EXPECT_EQ(table.lookupLocation(6)->slot(), 11U);
}

// Contract violations are checked before publication, never by the noexcept write phase.
TEST(TestLocationTable, InvalidDeltaRejectsBeforePublication)
{
    using faiss::ascend::Location;
    using faiss::ascend::LocationDelta;
    faiss::ascend::LocationTable table(1, NLIST);
    LocationDelta delta;
    delta.addedIds.emplace_back(123456, Location(0, 1, 0));
    EXPECT_THROW(table.validateDelta(delta), std::logic_error);
    table.reserveForIds({123456});
    ASSERT_NO_THROW(table.validateDelta(delta));
    EXPECT_FALSE(table.contains(123456));  // Validation never publishes a row.

    for (Location invalid : {Location{}, Location(1, 1, 0), Location(0, NLIST, 0)})
    {
        delta.addedIds[0].second = invalid;
        EXPECT_THROW(table.validateDelta(delta), std::exception);
        EXPECT_FALSE(table.contains(123456));
    }
    delta.addedIds[0] = {-1, Location(0, 1, 0)};
    EXPECT_THROW(table.validateDelta(delta), std::exception);
    delta.addedIds.clear();
    delta.movedIds.emplace_back(123456, Location(0, 1, 0));  // Reserved but not an existing valid moved ID.
    EXPECT_THROW(table.validateDelta(delta), std::exception);
    EXPECT_FALSE(table.contains(123456));
}

void Norm(float* data, int ntotal, int dim)
{
    const float invSqrtDim = 1.0f / std::sqrt(static_cast<float>(dim));
#pragma omp parallel for if (ntotal > 1)
    for (int i = 0; i < ntotal; ++i)
    {
        float l2norm = 0.0;
        for (int j = 0; j < dim; ++j)
        {
            l2norm += data[i * dim + j] * data[i * dim + j];
        }
        l2norm = std::sqrt(l2norm);
        if (fabs(l2norm) < 1e-6)
        {
            // reset data when l2norm = 0
            for (int j = 0; j < dim; ++j)
            {
                data[i * dim + j] = invSqrtDim;
            }
        }
        else
        {
            for (int j = 0; j < dim; ++j)
            {
                data[i * dim + j] = data[i * dim + j] / l2norm;
            }
        }
    }
}

void generateData(float* data, int ntotal, int dim)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (long long int i = 0; i < ntotal * dim; i++)
    {
        data[i] = dis(gen);
    }
    Norm(data, ntotal, dim);
}

faiss::idx_t IdentityId(int i) { return static_cast<faiss::idx_t>(i); }

struct TrainedRabitqSearch
{
    int dim = DIM;
    int nlist = NLIST;
    int ntotal = 0;
    int nprobe = 64;
    int nq = 5;
    int k = 10;
    int trainNum = 0;
    std::vector<float> data;
    std::vector<faiss::idx_t> ids;
    std::unique_ptr<faiss::ascend::AscendIndexIVFRaBitQ> index;
};

template <typename IdFn>
TrainedRabitqSearch MakeTrainedRabitqSearch(int ntotal, std::vector<int> devices, IdFn idOf, int nq = 5, int k = 10,
                                            int nprobe = 64)
{
    TrainedRabitqSearch fx;
    fx.ntotal = ntotal;
    fx.nprobe = nprobe;
    fx.nq = nq;
    fx.k = k;
    fx.trainNum = ntotal > fx.nlist * 40 ? fx.nlist * 40 : ntotal;
    fx.data.resize(static_cast<size_t>(ntotal) * static_cast<size_t>(fx.dim));
    fx.ids.resize(static_cast<size_t>(ntotal));
    generateData(fx.data.data(), ntotal, fx.dim);
    for (int i = 0; i < ntotal; ++i)
    {
        fx.ids[static_cast<size_t>(i)] = idOf(i);
    }
    auto config = MakeConfig(std::move(devices));
    fx.index = std::make_unique<faiss::ascend::AscendIndexIVFRaBitQ>(fx.dim, faiss::METRIC_L2, fx.nlist, config);
    fx.index->setNumProbes(nprobe);
    fx.index->train(fx.trainNum, fx.data.data());
    fx.index->add_with_ids(ntotal, fx.data.data(), fx.ids.data());
    return fx;
}

template <typename Pred>
void ExpectFilteredLabels(const std::vector<faiss::idx_t>& labels, Pred pred)
{
    int valid = 0;
    for (faiss::idx_t id : labels)
    {
        if (id < 0)
        {
            continue;
        }
        ++valid;
        EXPECT_TRUE(pred(id));
    }
    EXPECT_GT(valid, 0);
}

template <typename Pred>
void SearchAndExpectFiltered(const TrainedRabitqSearch& fx, faiss::SearchParameters* params, Pred pred)
{
    std::vector<float> dist(static_cast<size_t>(fx.nq) * static_cast<size_t>(fx.k), 0.0f);
    std::vector<faiss::idx_t> label(static_cast<size_t>(fx.nq) * static_cast<size_t>(fx.k), -1);
    fx.index->search(fx.nq, fx.data.data(), fx.k, dist.data(), label.data(), params);
    ExpectFilteredLabels(label, pred);
}

struct FrozenRoaringBytes
{
    size_t bytes = 0;
    uint8_t* aligned = nullptr;
};

FrozenRoaringBytes FreezeRoaringAligned(const roaring_bitmap_t* bitmap, std::vector<uint8_t>& buffer)
{
    FrozenRoaringBytes frozen;
    frozen.bytes = roaring_bitmap_frozen_size_in_bytes(bitmap);
    buffer.assign(frozen.bytes + 32, 0);
    const uintptr_t base = reinterpret_cast<uintptr_t>(buffer.data());
    const size_t off = (32U - (base % 32U)) % 32U;
    frozen.aligned = buffer.data() + off;
    roaring_bitmap_frozen_serialize(bitmap, reinterpret_cast<char*>(frozen.aligned));
    return frozen;
}

TEST(TestAscendIndexIVFRaBitQ, IDSelectorRoaringIsMember)
{
    roaring_bitmap_t* rb = roaring_bitmap_from_range(0, 10, 1);
    ASSERT_NE(rb, nullptr);
    faiss::ascend::IDSelectorRoaring liveSel(rb);
    EXPECT_TRUE(liveSel.is_member(0));
    EXPECT_TRUE(liveSel.is_member(9));
    EXPECT_FALSE(liveSel.is_member(10));
    EXPECT_FALSE(liveSel.is_member(-1));
    EXPECT_FALSE(liveSel.is_member(static_cast<faiss::idx_t>(std::numeric_limits<uint32_t>::max()) + 1));

    std::vector<uint8_t> frozenBuf;
    const FrozenRoaringBytes frozen = FreezeRoaringAligned(rb, frozenBuf);
    faiss::ascend::IDSelectorRoaring frozenSel(frozen.bytes, frozen.aligned);
    EXPECT_TRUE(frozenSel.is_member(3));
    EXPECT_FALSE(frozenSel.is_member(11));

    std::vector<uint8_t> unaligned(frozen.bytes + 1, 0);
    std::memcpy(unaligned.data() + 1, frozen.aligned, frozen.bytes);
    faiss::ascend::IDSelectorRoaring unalignedSel(frozen.bytes, unaligned.data() + 1);
    EXPECT_TRUE(unalignedSel.is_member(3));
    roaring_bitmap_free(rb);
}

TEST(TestAscendIndexIVFRaBitQ, SearchWithIdSelector)
{
    std::string msg = "";
    try
    {
        FilteredSearchMockGuard filterSearchMock;
        const TrainedRabitqSearch fx = MakeTrainedRabitqSearch(2000, {0}, IdentityId);
        const int half = fx.ntotal / 2;

        // Range: keep ids in [0, ntotal/2)
        {
            faiss::IDSelectorRange rangeSel(0, half);
            faiss::SearchParameters params;
            params.sel = &rangeSel;
            SearchAndExpectFiltered(fx, &params, [&](faiss::idx_t id) { return id >= 0 && id < half; });
        }

        // Contiguous Array -> RANGE fold: keep [0, ntotal/2)
        {
            std::vector<faiss::idx_t> allow(static_cast<size_t>(half));
            std::iota(allow.begin(), allow.end(), 0);
            faiss::IDSelectorArray arraySel(allow.size(), allow.data());
            faiss::SearchParameters params;
            params.sel = &arraySel;
            SearchAndExpectFiltered(fx, &params, [&](faiss::idx_t id) { return id < half; });
        }

        // Batch: only even ids in [0, 100)
        {
            std::vector<faiss::idx_t> allow;
            for (int i = 0; i < 100; i += 2)
            {
                allow.push_back(i);
            }
            faiss::IDSelectorBatch batchSel(allow.size(), allow.data());
            faiss::SearchParameters params;
            params.sel = &batchSel;
            SearchAndExpectFiltered(fx, &params, [&](faiss::idx_t id) { return batchSel.is_member(id); });
        }

        // Bitmap: allow first 64 ids
        {
            std::vector<uint8_t> bitmap(fx.ntotal / 8 + 1, 0);
            for (int i = 0; i < 64; ++i)
            {
                bitmap[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
            }
            faiss::IDSelectorBitmap bitmapSel(bitmap.size(), bitmap.data());
            faiss::SearchParameters params;
            params.sel = &bitmapSel;
            SearchAndExpectFiltered(fx, &params, [&](faiss::idx_t id) { return bitmapSel.is_member(id); });
        }

        // Not(Range): exclude [0, ntotal/2)
        {
            faiss::IDSelectorRange rangeSel(0, half);
            faiss::IDSelectorNot notSel(&rangeSel);
            faiss::SearchParameters params;
            params.sel = &notSel;
            SearchAndExpectFiltered(fx, &params, [&](faiss::idx_t id) { return id >= half; });
        }

        // Sparse Array (non-contiguous even ids) -> Roaring frozen
        {
            std::vector<faiss::idx_t> allow;
            for (int i = 0; i < fx.ntotal; i += 2)
            {
                allow.push_back(i);
            }
            faiss::IDSelectorArray arraySel(allow.size(), allow.data());
            faiss::SearchParameters params;
            params.sel = &arraySel;
            SearchAndExpectFiltered(fx, &params, [](faiss::idx_t id) { return id % 2 == 0; });
        }

        // Stride-2 Array should match IDSelectorRoaring from_range(..., step=2)
        {
            roaring_bitmap_t* rb = roaring_bitmap_from_range(0, static_cast<uint64_t>(fx.ntotal), 2);
            ASSERT_NE(rb, nullptr);
            faiss::ascend::IDSelectorRoaring roarSel(rb);
            faiss::SearchParameters roarParams;
            roarParams.sel = &roarSel;
            SearchAndExpectFiltered(fx, &roarParams, [](faiss::idx_t id) { return id % 2 == 0; });
            roaring_bitmap_free(rb);
        }

        // Non-AP Array still filters correctly (Roaring add_many fallback)
        {
            std::vector<faiss::idx_t> irregular = {0, 1, 3, 10, 50, 99};
            faiss::IDSelectorArray arraySel(irregular.size(), irregular.data());
            faiss::SearchParameters params;
            params.sel = &arraySel;
            SearchAndExpectFiltered(fx, &params, [&](faiss::idx_t id) { return arraySel.is_member(id); });
        }
        {
            roaring_bitmap_t* rb = roaring_bitmap_from_range(0, static_cast<uint64_t>(half), 1);
            ASSERT_NE(rb, nullptr);
            faiss::ascend::IDSelectorRoaring liveSel(rb);
            faiss::SearchParameters liveParams;
            liveParams.sel = &liveSel;
            SearchAndExpectFiltered(fx, &liveParams, [&](faiss::idx_t id) { return id < half; });

            std::vector<uint8_t> frozenBuf;
            const FrozenRoaringBytes frozen = FreezeRoaringAligned(rb, frozenBuf);
            faiss::ascend::IDSelectorRoaring frozenSel(frozen.bytes, frozen.aligned);
            faiss::SearchParameters frozenParams;
            frozenParams.sel = &frozenSel;
            SearchAndExpectFiltered(fx, &frozenParams, [&](faiss::idx_t id) { return id < half; });

            faiss::IDSelectorNot notLive(&liveSel);
            faiss::SearchParameters notParams;
            notParams.sel = &notLive;
            SearchAndExpectFiltered(fx, &notParams, [&](faiss::idx_t id) { return id >= half; });
            roaring_bitmap_free(rb);
        }
    }
    catch (std::exception& e)
    {
        msg = e.what();
        std::cout << "Exception in SearchWithIdSelector test: " << msg << std::endl;
    }
    EXPECT_EQ(msg, "");
}

TEST(TestAscendIndexIVFRaBitQ, SearchWithIdSelectorOutOfUint32)
{
    std::string msg = "";
    const faiss::idx_t idBase = static_cast<faiss::idx_t>(std::numeric_limits<uint32_t>::max()) + 1;
    try
    {
        const TrainedRabitqSearch fx = MakeTrainedRabitqSearch(2000, {0}, [&](int i) { return idBase + i; });
        const int half = fx.ntotal / 2;
        std::vector<faiss::idx_t> allow(static_cast<size_t>(half));
        for (int i = 0; i < half; ++i)
        {
            allow[static_cast<size_t>(i)] = fx.ids[static_cast<size_t>(i)];
        }
        faiss::IDSelectorArray arraySel(allow.size(), allow.data());
        faiss::SearchParameters params;
        params.sel = &arraySel;
        SearchAndExpectFiltered(fx, &params, [&](faiss::idx_t id) { return id >= idBase && id < idBase + half; });
    }
    catch (std::exception& e)
    {
        msg = e.what();
        std::cout << "Exception in SearchWithIdSelectorOutOfUint32 test: " << msg << std::endl;
    }
    EXPECT_EQ(msg, "");
}

TEST(TestAscendIndexIVFRaBitQ, SearchWithIdSelectorSharedPayload)
{
    std::string msg = "";
    try
    {
        FilteredSearchMockGuard filterSearchMock;
        const TrainedRabitqSearch fx = MakeTrainedRabitqSearch(2000, {0}, IdentityId);
        const int half = fx.ntotal / 2;

        std::vector<faiss::idx_t> keepFirst(static_cast<size_t>(half));
        std::iota(keepFirst.begin(), keepFirst.end(), 0);
        std::vector<faiss::idx_t> keepSecond(static_cast<size_t>(half));
        std::iota(keepSecond.begin(), keepSecond.end(), half);

        auto checkKeepHalf = [&](faiss::IDSelector& sel, bool firstHalf)
        {
            faiss::SearchParameters params;
            params.sel = &sel;
            if (firstHalf)
            {
                SearchAndExpectFiltered(fx, &params, [&](faiss::idx_t id) { return id < half; });
            }
            else
            {
                SearchAndExpectFiltered(fx, &params, [&](faiss::idx_t id) { return id >= half; });
            }
        };

        {
            faiss::IDSelectorArray a0(keepFirst.size(), keepFirst.data());
            faiss::IDSelectorArray a1(keepFirst.size(), keepFirst.data());
            checkKeepHalf(a0, true);
            checkKeepHalf(a1, true);
            faiss::IDSelectorArray a2(keepSecond.size(), keepSecond.data());
            checkKeepHalf(a2, false);
            checkKeepHalf(a0, true);
        }

        {
            faiss::IDSelectorBatch batch0(keepFirst.size(), keepFirst.data());
            faiss::IDSelectorBatch batch1(keepFirst.size(), keepFirst.data());
            checkKeepHalf(batch0, true);
            checkKeepHalf(batch1, true);
            faiss::IDSelectorBatch batch2(keepSecond.size(), keepSecond.data());
            checkKeepHalf(batch2, false);
            checkKeepHalf(batch0, true);
        }

        const size_t bitmapBytes = static_cast<size_t>(fx.ntotal + 7) / 8;
        std::vector<uint8_t> bitmapFirst(bitmapBytes, 0);
        std::vector<uint8_t> bitmapSecond(bitmapBytes, 0);
        for (int i = 0; i < half; ++i)
        {
            bitmapFirst[static_cast<size_t>(i) / 8] |= static_cast<uint8_t>(1u << (i % 8));
            const int j = i + half;
            bitmapSecond[static_cast<size_t>(j) / 8] |= static_cast<uint8_t>(1u << (j % 8));
        }
        {
            faiss::IDSelectorBitmap b0(bitmapFirst.size(), bitmapFirst.data());
            faiss::IDSelectorBitmap b1(bitmapFirst.size(), bitmapFirst.data());
            checkKeepHalf(b0, true);
            checkKeepHalf(b1, true);
            faiss::IDSelectorBitmap b2(bitmapSecond.size(), bitmapSecond.data());
            checkKeepHalf(b2, false);
        }

        {
            roaring_bitmap_t* rb = roaring_bitmap_from_range(0, static_cast<uint64_t>(half), 1);
            ASSERT_NE(rb, nullptr);
            faiss::ascend::IDSelectorRoaring r0(rb);
            faiss::ascend::IDSelectorRoaring r1(rb);
            checkKeepHalf(r0, true);
            checkKeepHalf(r1, true);
            roaring_bitmap_free(rb);
            rb = roaring_bitmap_from_range(static_cast<uint64_t>(half), static_cast<uint64_t>(fx.ntotal), 1);
            ASSERT_NE(rb, nullptr);
            faiss::ascend::IDSelectorRoaring r2(rb);
            checkKeepHalf(r2, false);
            roaring_bitmap_free(rb);
        }

        {
            roaring_bitmap_t* rb = roaring_bitmap_from_range(0, static_cast<uint64_t>(half), 1);
            ASSERT_NE(rb, nullptr);
            std::vector<uint8_t> buf0;
            std::vector<uint8_t> buf1;
            const FrozenRoaringBytes frozen0 = FreezeRoaringAligned(rb, buf0);
            const FrozenRoaringBytes frozen1 = FreezeRoaringAligned(rb, buf1);
            faiss::ascend::IDSelectorRoaring f0(frozen0.bytes, frozen0.aligned);
            faiss::ascend::IDSelectorRoaring f1(frozen0.bytes, frozen0.aligned);
            checkKeepHalf(f0, true);
            checkKeepHalf(f1, true);
            faiss::ascend::IDSelectorRoaring fCopy(frozen1.bytes, frozen1.aligned);
            checkKeepHalf(fCopy, true);
            roaring_bitmap_free(rb);
        }

        {
            faiss::IDSelectorArray inner(keepFirst.size(), keepFirst.data());
            faiss::IDSelectorNot n0(&inner);
            faiss::IDSelectorNot n1(&inner);
            checkKeepHalf(n0, false);
            checkKeepHalf(n1, false);
        }
    }
    catch (std::exception& e)
    {
        msg = e.what();
        std::cout << "Exception in SearchWithIdSelectorSharedPayload test: " << msg << std::endl;
    }
    EXPECT_EQ(msg, "");
}

TEST(TestAscendIndexIVFRaBitQ, SparseHighIdSelectorCacheHitAndMissPreserveOutput)
{
    auto source = MakePretrainedCpuIndex();
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    const faiss::idx_t high = static_cast<faiss::idx_t>(std::numeric_limits<uint32_t>::max()) + 17;
    const std::vector<faiss::idx_t> allowed = {source.ids[0], source.ids[2], high};
    faiss::IDSelectorArray array(allowed.size(), allowed.data());
    faiss::IDSelectorBatch batch(allowed.size(), allowed.data());
    faiss::IDSelectorNot notArray(&array);
    faiss::IDSelectorNot notBatch(&batch);
    const auto query = MakePatternData(1, DIM);
    FilteredSearchMockGuard mock;
    for (faiss::IDSelector* selector : std::vector<faiss::IDSelector*>{&array, &batch, &notArray, &notBatch})
    {
        SCOPED_TRACE(selector == &array      ? "array"
                     : selector == &batch    ? "batch"
                     : selector == &notArray ? "not-array"
                                             : "not-batch");
        faiss::SearchParameters params;
        params.sel = selector;
        std::vector<float> missDistances(2), hitDistances(2), rebuiltDistances(2);
        std::vector<faiss::idx_t> missLabels(2), hitLabels(2), rebuiltLabels(2);
        index.impl_->invalidateFilterCache();
        const auto key = index.impl_->MakeFilterCacheKey(selector);
        EXPECT_EQ(index.impl_->FindFilterCacheHit(key), -1);
        ASSERT_NO_THROW(index.search(1, query.data(), 2, missDistances.data(), missLabels.data(), &params));
        ASSERT_GE(index.impl_->FindFilterCacheHit(key), 0);
        const auto generation = index.impl_->cachedFilterGeneration;
        const auto* filter = index.impl_->getCachedFilter(selector);
        const bool negated = selector == &notArray || selector == &notBatch;
        EXPECT_EQ(filter->negate, negated ? 1 : 0);
        EXPECT_EQ(IdPassesFilter(*filter, high, nullptr), !negated);
        EXPECT_EQ(IdPassesFilter(*filter, high + 1, nullptr), negated);
        EXPECT_EQ(index.impl_->cachedFilterGeneration, generation);
        ASSERT_NO_THROW(index.search(1, query.data(), 2, hitDistances.data(), hitLabels.data(), &params));
        EXPECT_EQ(index.impl_->cachedFilterGeneration, generation);
        EXPECT_EQ(hitDistances, missDistances);
        EXPECT_EQ(hitLabels, missLabels);
        EXPECT_THAT(missLabels, negated ? ElementsAre(0, 1) : ElementsAre(101, 303));
        index.impl_->invalidateFilterCache();
        EXPECT_EQ(index.impl_->FindFilterCacheHit(key), -1);
        ASSERT_NO_THROW(index.search(1, query.data(), 2, rebuiltDistances.data(), rebuiltLabels.data(), &params));
        EXPECT_EQ(rebuiltDistances, missDistances);
        EXPECT_EQ(rebuiltLabels, missLabels);
    }
}

TEST(TestAscendIndexIVFRaBitQ, SearchWithIdSelectorMultiDevice)
{
    uint32_t deviceCount = 0;
    auto ret = aclrtGetDeviceCount(&deviceCount);
    if (ret != ACL_SUCCESS || deviceCount < 2)
    {
        GTEST_SKIP() << "Need >=2 devices for multi-card IDSelector test";
    }

    std::string msg = "";
    try
    {
        FilteredSearchMockGuard filterSearchMock;
        const TrainedRabitqSearch fx = MakeTrainedRabitqSearch(4000, {0, 1}, IdentityId, 4, 10);
        const int half = fx.ntotal / 2;
        faiss::IDSelectorRange rangeSel(0, half);
        faiss::SearchParameters params;
        params.sel = &rangeSel;
        SearchAndExpectFiltered(fx, &params, [&](faiss::idx_t id) { return id < half; });
    }
    catch (std::exception& e)
    {
        msg = e.what();
        std::cout << "Exception in SearchWithIdSelectorMultiDevice test: " << msg << std::endl;
    }
    EXPECT_EQ(msg, "");
}

TEST(TestAscendIndexIVFRaBitQ, SearchParametersIVFNprobe)
{
    const int indexNprobe = 64;
    const int searchNprobe = 32;

    std::string msg = "";
    try
    {
        const TrainedRabitqSearch fx = MakeTrainedRabitqSearch(2000, {0}, IdentityId, 4, 10, indexNprobe);

        faiss::SearchParametersIVF params;
        params.nprobe = static_cast<size_t>(searchNprobe);
        SearchAndExpectFiltered(fx, &params, [](faiss::idx_t) { return true; });
        EXPECT_EQ(fx.index->getNumProbes(), indexNprobe);
    }
    catch (std::exception& e)
    {
        msg = e.what();
        std::cout << "Exception in SearchParametersIVFNprobe test: " << msg << std::endl;
    }
    EXPECT_EQ(msg, "");
}

TEST(TestAscendIndexIVFRaBitQ, SearchParametersIVFInvalidNprobe)
{
    const int dim = 128;
    const int nlist = 1024;
    const int nq = 1;
    const int k = 10;

    std::string msg = "";
    faiss::MetricType type = faiss::METRIC_L2;
    faiss::ascend::AscendIndexIVFRaBitQConfig conf({0});
    std::vector<float> query(nq * dim, 0.0f);
    std::vector<float> dist(nq * k, 0.0f);
    std::vector<faiss::idx_t> label(nq * k, -1);

    try
    {
        faiss::ascend::AscendIndexIVFRaBitQ index(dim, type, nlist, conf);

        faiss::SearchParametersIVF zeroParams;
        zeroParams.nprobe = 0;
        bool threwZero = false;
        try
        {
            index.search(nq, query.data(), k, dist.data(), label.data(), &zeroParams);
        }
        catch (std::exception& e)
        {
            threwZero = true;
            EXPECT_NE(std::string(e.what()).find("nprobe must be greater than 0"), std::string::npos);
        }
        EXPECT_TRUE(threwZero);
        EXPECT_EQ(index.getNumProbes(), 64);

        faiss::SearchParametersIVF overParams;
        overParams.nprobe = static_cast<size_t>(nlist) + 1;
        bool threwOver = false;
        try
        {
            index.search(nq, query.data(), k, dist.data(), label.data(), &overParams);
        }
        catch (std::exception& e)
        {
            threwOver = true;
            EXPECT_NE(std::string(e.what()).find("nprobe must be <= nlist"), std::string::npos);
        }
        EXPECT_TRUE(threwOver);
        EXPECT_EQ(index.getNumProbes(), 64);
    }
    catch (std::exception& e)
    {
        msg = e.what();
        std::cout << "Exception in SearchParametersIVFInvalidNprobe test: " << msg << std::endl;
    }
    EXPECT_EQ(msg, "");
}

// ---------------------------------------------------------------------------
// The host half of the block-slot contract.
//
// BuildBlockSlotPlan decides which absolute grid slot each block owns; fillDisOpInputDataByBlock must write
// that block's eight per-block arrays into exactly that slot; the TopK kernel reads them back through
// blockSlots[q][i]. A disagreement between the plan and the fill does not fail the search -- it silently
// scores one query's block under another query, so the only symptom is quietly lower recall. These tests
// assert the agreement directly instead of hoping a recall number notices.
// ---------------------------------------------------------------------------

constexpr size_t GRID_CORE_NUM = 4;

// One logical block as the probe walk sees it, before the layout has any say in where it goes.
struct LogicalBlock
{
    int64_t listId = 0;
    size_t segIdx = 0;
    float centroidL2 = 0.0f;
};

struct ProbePlan
{
    size_t batch = 0;
    size_t width = 0;
    std::vector<int64_t> listIds;
    std::vector<float> centroidL2;
};

// Sets the same thread-local view the public search entry establishes. Grid helpers below call internal
// planning/filling functions directly, so they still need a pinned snapshot for the whole call.
class ScopedGridSnapshot
{
   public:
    explicit ScopedGridSnapshot(const ::ascend::IndexIVFRaBitQ& index)
        : guard_(index), previous_(::ascend::IndexIVFRaBitQ::searchSnapshot)
    {
        ::ascend::IndexIVFRaBitQ::searchSnapshot = &guard_.get();
    }
    ~ScopedGridSnapshot() { ::ascend::IndexIVFRaBitQ::searchSnapshot = previous_; }

   private:
    ::ascend::IndexIVFRaBitQ::SnapshotGuard guard_;
    const ::ascend::IndexIVFRaBitQ::GlobalSnapshot* previous_;
};

// Builds and publishes fixture storage without quantisation. No payload upload is needed: these tests
// inspect block lengths and addresses rather than distances.
void SeedList(::ascend::IndexIVFRaBitQ* index, int listId, size_t numVecs)
{
    std::lock_guard<std::mutex> writerLock(index->writerLock);
    auto next = index->cloneSnapshot(*index->activeSnapshot);
    auto listCopy = std::make_shared<::ascend::IndexIVFRaBitQ::ListStorage>(*next->lists.at(listId));
    auto stream = index->writerContext->computeStream;
    index->ensureListCapacity(*listCopy, numVecs, stream);
    ASSERT_EQ(::ascend::synchronizeStream(stream), ACL_SUCCESS);
    listCopy->codes.commitLength(numVecs);
    listCopy->indexL1.commitLength(numVecs);
    listCopy->indexL2.commitLength(numVecs);
    listCopy->ids.commitLength(numVecs);
    next->total += numVecs - listCopy->listLen;
    listCopy->listLen = numVecs;
    next->lists.set(static_cast<size_t>(listId), std::move(listCopy));
    index->publishSnapshot(std::move(next));
}

// Lists holding 1, 2, 3 and 1 blocks, probed so the six queries own 2, 5, 5, 2, 5 and 4 blocks. Unequal
// per-query block counts are the only shape in which an off-by-one in the emit order or in the row stride can
// still look plausible, and 23 blocks over 4 cores leaves a partial final iteration.
ProbePlan SeedRaggedBatch(::ascend::IndexIVFRaBitQ* index)
{
    const size_t chunk = static_cast<size_t>(index->blockSize);
    SeedList(index, 0, 100);
    SeedList(index, 1, chunk + 100);
    SeedList(index, 2, 2 * chunk + 100);
    SeedList(index, 3, 50);

    ProbePlan plan;
    plan.batch = 6;
    plan.width = 2;
    plan.listIds = {0, 3, 1, 2, 2, 1, 0, 3, 1, 2, 2, 0};
    plan.centroidL2.resize(plan.listIds.size());
    for (size_t i = 0; i < plan.centroidL2.size(); ++i)
    {
        // Distinct per probe, so a block written into the wrong slot shows up as a mismatched centroid
        // distance rather than blending in with its neighbour.
        plan.centroidL2[i] = 0.5f + static_cast<float>(i);
    }
    return plan;
}

std::vector<std::vector<LogicalBlock>> WalkProbes(const ::ascend::IndexIVFRaBitQ* index, const ProbePlan& plan)
{
    std::vector<std::vector<LogicalBlock>> perQuery(plan.batch);
    for (size_t q = 0; q < plan.batch; ++q)
    {
        for (size_t t = 0; t < plan.width; ++t)
        {
            const size_t probe = q * plan.width + t;
            const int64_t listId = plan.listIds[probe];
            const size_t blocks = index->activeSnapshot->lists.at(static_cast<size_t>(listId))->codes.segmentCount();
            for (size_t seg = 0; seg < blocks; ++seg)
            {
                LogicalBlock blk;
                blk.listId = listId;
                blk.segIdx = seg;
                blk.centroidL2 = plan.centroidL2[probe];
                perQuery[q].push_back(blk);
            }
        }
    }
    return perQuery;
}

// Everything fillDisOpInputDataByBlock writes, kept together so a test can hand the eight tensors over and
// then read the raw grids back.
struct BlockGrid
{
    size_t iterNum = 0;
    std::vector<uint32_t> queryId;
    std::vector<uint32_t> centroidsId;
    std::vector<float> centroidsL2;
    std::vector<uint32_t> baseSize;
    std::vector<uint64_t> offset;
    std::vector<uint64_t> indexL2Offset;
    std::vector<uint64_t> indexL1Offset;
    std::vector<int64_t> ids;
    std::vector<int64_t> blockNumPerQ;
    std::vector<int64_t> slots;
};

// Drives the production fill for one layout. `group` is the plan's window width: 1 reproduces the
// query-major legacy order, larger values interleave that many queries.
BlockGrid FillGrid(::ascend::IndexIVFRaBitQ* index, const ProbePlan& plan, int64_t group)
{
    ScopedGridSnapshot snapshot(*index);
    std::vector<int64_t> blockNumPlan;
    std::vector<int64_t> probeIds = plan.listIds;
    std::vector<float> probeDists = plan.centroidL2;
    ::ascend::AscendTensor<int64_t, ::ascend::DIMS_2> probeIdsHost(
        probeIds.data(), {static_cast<int>(plan.batch), static_cast<int>(plan.width)});
    ::ascend::AscendTensor<float, ::ascend::DIMS_2> probeDistsHost(
        probeDists.data(), {static_cast<int>(plan.batch), static_cast<int>(plan.width)});
    const size_t maxBlocksPerQ = index->getBlockNumPerQ(plan.batch, probeIdsHost, blockNumPlan);

    int64_t total = 0;
    for (size_t q = 0; q < plan.batch; ++q)
    {
        total += blockNumPlan[q];
    }

    BlockGrid grid;
    grid.iterNum = (static_cast<size_t>(total) + GRID_CORE_NUM - 1) / GRID_CORE_NUM;
    const size_t cells = grid.iterNum * GRID_CORE_NUM;
    // Sentinels rather than zeros: a cell the fill never touches has to be distinguishable from a cell it
    // legitimately filled with query 0 / list 0 / offset 0.
    grid.queryId.assign(cells, 0xFFFFFFFFu);
    grid.centroidsId.assign(cells, 0xFFFFFFFFu);
    grid.centroidsL2.assign(cells, -1.0f);
    grid.baseSize.assign(cells, 0xFFFFFFFFu);
    grid.offset.assign(cells, 0xFFFFFFFFFFFFFFFFull);
    grid.indexL2Offset.assign(cells, 0xFFFFFFFFFFFFFFFFull);
    grid.indexL1Offset.assign(cells, 0xFFFFFFFFFFFFFFFFull);
    grid.ids.assign(cells, -1);
    grid.blockNumPerQ.assign(plan.batch, -1);
    grid.slots.assign(plan.batch * maxBlocksPerQ, -1);

    ::ascend::BuildBlockSlotPlan(blockNumPlan, maxBlocksPerQ, group, grid.slots.data());

    ::ascend::AscendTensor<uint32_t, ::ascend::DIMS_2, size_t> queryIdVec(grid.queryId.data(),
                                                                          {grid.iterNum, GRID_CORE_NUM});
    ::ascend::AscendTensor<uint32_t, ::ascend::DIMS_2, size_t> centroidsIdVec(grid.centroidsId.data(),
                                                                              {grid.iterNum, GRID_CORE_NUM});
    ::ascend::AscendTensor<float, ::ascend::DIMS_2, size_t> centroidsL2Vec(grid.centroidsL2.data(),
                                                                           {grid.iterNum, GRID_CORE_NUM});
    ::ascend::AscendTensor<uint32_t, ::ascend::DIMS_2, size_t> baseSizeVec(grid.baseSize.data(),
                                                                           {grid.iterNum, GRID_CORE_NUM});
    ::ascend::AscendTensor<uint64_t, ::ascend::DIMS_2, size_t> offsetVec(grid.offset.data(),
                                                                         {grid.iterNum, GRID_CORE_NUM});
    ::ascend::AscendTensor<uint64_t, ::ascend::DIMS_2, size_t> indexL2OffsetVec(grid.indexL2Offset.data(),
                                                                                {grid.iterNum, GRID_CORE_NUM});
    ::ascend::AscendTensor<uint64_t, ::ascend::DIMS_2, size_t> indexL1OffsetVec(grid.indexL1Offset.data(),
                                                                                {grid.iterNum, GRID_CORE_NUM});
    ::ascend::AscendTensor<int64_t, ::ascend::DIMS_2, size_t> idsVec(grid.ids.data(), {grid.iterNum, GRID_CORE_NUM});
    ::ascend::AscendTensor<int64_t, ::ascend::DIMS_1, size_t> blockNumPerQVec(grid.blockNumPerQ.data(), {plan.batch});

    index->fillDisOpInputDataByBlock(plan.batch, GRID_CORE_NUM, queryIdVec, centroidsIdVec, centroidsL2Vec, baseSizeVec,
                                     offsetVec, indexL2OffsetVec, indexL1OffsetVec, idsVec, blockNumPerQVec,
                                     probeIdsHost, probeDistsHost, grid.slots.data(), maxBlocksPerQ);
    return grid;
}

TEST(TestIndexIVFRaBitQCore, BlockGridFillFollowsTheSlotPlan)
{
    faiss::ascend::AscendIndexIVFRaBitQ hostIndex(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    auto actual = hostIndex.impl_->getActualIndex(0);
    ASSERT_NE(actual, nullptr);

    const ProbePlan plan = SeedRaggedBatch(actual);
    const auto expected = WalkProbes(actual, plan);

    // group=1 is the legacy order, 4 and 6 straddle the AI CPU count, 8 is wider than the batch so the
    // window clamps.
    const int64_t groups[] = {1, 2, 4, 6, 8};
    for (int64_t group : groups)
    {
        const BlockGrid grid = FillGrid(actual, plan, group);
        const size_t maxBlocksPerQ = grid.slots.size() / plan.batch;
        std::vector<int> timesWritten(grid.iterNum * GRID_CORE_NUM, 0);

        for (size_t q = 0; q < plan.batch; ++q)
        {
            EXPECT_EQ(grid.blockNumPerQ[q], static_cast<int64_t>(expected[q].size()))
                << "group=" << group << " q=" << q;

            for (size_t i = 0; i < expected[q].size(); ++i)
            {
                const LogicalBlock& blk = expected[q][i];
                const int64_t slot = grid.slots[q * maxBlocksPerQ + i];
                ASSERT_GE(slot, 0) << "group=" << group << " q=" << q << " i=" << i;
                // The grid is addressed as [slot / coreNum][slot % coreNum]; these tensors are row-major
                // with width coreNum, so that is the flat cell the fill must have written.
                const size_t iter = static_cast<size_t>(slot) / GRID_CORE_NUM;
                const size_t cIdx = static_cast<size_t>(slot) % GRID_CORE_NUM;
                ASSERT_LT(iter, grid.iterNum) << "group=" << group << " slot=" << slot;
                const size_t cell = iter * GRID_CORE_NUM + cIdx;
                timesWritten[cell]++;

                const size_t listId = static_cast<size_t>(blk.listId);
                const auto& list = *actual->activeSnapshot->lists.at(listId);
                const size_t listNum = list.listLen;
                // The values themselves are how a block is identified; what is under test is that this
                // block's values landed in the slot the plan handed it and in no other.
                EXPECT_EQ(grid.queryId[cell], static_cast<uint32_t>(q)) << "group=" << group << " slot=" << slot;
                EXPECT_EQ(grid.centroidsId[cell], static_cast<uint32_t>(blk.listId))
                    << "group=" << group << " slot=" << slot;
                EXPECT_FLOAT_EQ(grid.centroidsL2[cell], blk.centroidL2) << "group=" << group << " slot=" << slot;
                EXPECT_EQ(grid.baseSize[cell], static_cast<uint32_t>(std::min(listNum - blk.segIdx * actual->blockSize,
                                                                              static_cast<size_t>(actual->blockSize))))
                    << "group=" << group << " slot=" << slot;
                EXPECT_EQ(grid.ids[cell], reinterpret_cast<int64_t>(list.ids.segmentData(blk.segIdx)))
                    << "group=" << group << " slot=" << slot;
                EXPECT_EQ(grid.offset[cell], reinterpret_cast<uint64_t>(list.codes.segmentData(blk.segIdx)))
                    << "group=" << group << " slot=" << slot;
                EXPECT_EQ(grid.indexL2Offset[cell], reinterpret_cast<uint64_t>(list.indexL2.segmentData(blk.segIdx)))
                    << "group=" << group << " slot=" << slot;
                EXPECT_EQ(grid.indexL1Offset[cell], reinterpret_cast<uint64_t>(list.indexL1.segmentData(blk.segIdx)))
                    << "group=" << group << " slot=" << slot;
            }

            // Columns past this query's block count belong to no block and must stay at the sentinel the
            // caller seeds, otherwise a plain 0 there would alias grid slot 0.
            for (size_t i = expected[q].size(); i < maxBlocksPerQ; ++i)
            {
                EXPECT_EQ(grid.slots[q * maxBlocksPerQ + i], -1) << "group=" << group << " q=" << q << " i=" << i;
            }
        }

        size_t written = 0;
        for (size_t cell = 0; cell < timesWritten.size(); ++cell)
        {
            EXPECT_LE(timesWritten[cell], 1) << "group=" << group << " cell=" << cell;
            if (timesWritten[cell] == 1)
            {
                ++written;
                continue;
            }
            // The grid is rounded up to a whole iteration, so the tail cells are never claimed.
            EXPECT_EQ(grid.queryId[cell], 0xFFFFFFFFu) << "group=" << group << " cell=" << cell;
            EXPECT_EQ(grid.ids[cell], -1) << "group=" << group << " cell=" << cell;
        }
        size_t totalBlocks = 0;
        for (size_t q = 0; q < plan.batch; ++q)
        {
            totalBlocks += expected[q].size();
        }
        EXPECT_EQ(written, totalBlocks) << "group=" << group;
    }
}

TEST(TestIndexIVFRaBitQCore, LegacyGroupKeepsEachQueryContiguous)
{
    faiss::ascend::AscendIndexIVFRaBitQ hostIndex(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    auto actual = hostIndex.impl_->getActualIndex(0);
    ASSERT_NE(actual, nullptr);

    const ProbePlan plan = SeedRaggedBatch(actual);
    const auto expected = WalkProbes(actual, plan);

    // group=1 must reproduce the pre-layout addressing exactly: query q's blocks occupy the run of slots
    // starting at the sum of the preceding queries' block counts. That is what lets LEGACY keep the
    // kernel's contiguous shortcut instead of indirecting through blockSlots.
    const BlockGrid legacy = FillGrid(actual, plan, 1);
    const size_t maxBlocksPerQ = legacy.slots.size() / plan.batch;
    int64_t prefix = 0;
    for (size_t q = 0; q < plan.batch; ++q)
    {
        for (size_t i = 0; i < expected[q].size(); ++i)
        {
            EXPECT_EQ(legacy.slots[q * maxBlocksPerQ + i], prefix + static_cast<int64_t>(i)) << "q=" << q;
        }
        prefix += static_cast<int64_t>(expected[q].size());
    }

    // A wider window must still be a permutation of the same slots -- it reorders the grid, it does not
    // change which blocks are in it or how many.
    const BlockGrid grouped = FillGrid(actual, plan, 6);
    std::vector<int64_t> legacySlots;
    std::vector<int64_t> groupedSlots;
    for (size_t q = 0; q < plan.batch; ++q)
    {
        for (size_t i = 0; i < expected[q].size(); ++i)
        {
            legacySlots.push_back(legacy.slots[q * maxBlocksPerQ + i]);
            groupedSlots.push_back(grouped.slots[q * maxBlocksPerQ + i]);
        }
    }
    std::sort(legacySlots.begin(), legacySlots.end());
    std::sort(groupedSlots.begin(), groupedSlots.end());
    EXPECT_EQ(legacySlots, groupedSlots);
    EXPECT_EQ(legacy.blockNumPerQ, grouped.blockNumPerQ);
}

TEST(TestIndexIVFRaBitQCore, BlockNumPlanMatchesTheProbeWalkWidth)
{
    faiss::ascend::AscendIndexIVFRaBitQ hostIndex(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    auto actual = hostIndex.impl_->getActualIndex(0);
    ASSERT_NE(actual, nullptr);

    const ProbePlan plan = SeedRaggedBatch(actual);
    const auto expected = WalkProbes(actual, plan);

    std::vector<int64_t> probeIds = plan.listIds;
    ::ascend::AscendTensor<int64_t, ::ascend::DIMS_2> probeIdsHost(
        probeIds.data(), {static_cast<int>(plan.batch), static_cast<int>(plan.width)});
    std::vector<int64_t> blockNumPlan;
    size_t maxBlocksPerQ = 0;
    {
        ScopedGridSnapshot snapshot(*actual);
        maxBlocksPerQ = actual->getBlockNumPerQ(plan.batch, probeIdsHost, blockNumPlan);
    }

    // getBlockNumPerQ sizes the slot table; the fill walks the probes again to decide how many blocks it
    // actually emits. If the two ever disagreed on the probe width the rows would be the wrong size and the
    // fill would run off the end of a row into the next query's slots.
    ASSERT_EQ(blockNumPlan.size(), plan.batch);
    size_t widest = 0;
    for (size_t q = 0; q < plan.batch; ++q)
    {
        EXPECT_EQ(blockNumPlan[q], static_cast<int64_t>(expected[q].size())) << "q=" << q;
        widest = std::max(widest, expected[q].size());
    }
    EXPECT_EQ(maxBlocksPerQ, widest);

    const BlockGrid grid = FillGrid(actual, plan, 6);
    EXPECT_EQ(grid.blockNumPerQ, blockNumPlan);
}

// Public mutation boundaries must reject invalid counts before invoking their narrowed implementation.

int rejectedCountImplCalls = 0;

void StubRejectedCountAdd(faiss::ascend::AscendIndexIVFRaBitQImpl*, int, const float*, const faiss::idx_t*)
{
    ++rejectedCountImplCalls;
}

void StubRejectedCountDelete(faiss::ascend::AscendIndexIVFRaBitQImpl*, int, const faiss::idx_t*)
{
    ++rejectedCountImplCalls;
}

std::vector<faiss::idx_t> StubRejectedCountUpdate(faiss::ascend::AscendIndexIVFRaBitQImpl*, faiss::idx_t, const float*,
                                                  const faiss::idx_t*)
{
    ++rejectedCountImplCalls;
    return {};
}

struct RejectedCountMockGuard
{
    RejectedCountMockGuard()
    {
        rejectedCountImplCalls = 0;
        MOCKER_CPP(&faiss::ascend::AscendIndexIVFRaBitQImpl::addConcurrent).stubs().will(invoke(StubRejectedCountAdd));
        MOCKER_CPP(&faiss::ascend::AscendIndexIVFRaBitQImpl::deleteConcurrent)
            .stubs()
            .will(invoke(StubRejectedCountDelete));
        MOCKER_CPP(&faiss::ascend::AscendIndexIVFRaBitQImpl::update).stubs().will(invoke(StubRejectedCountUpdate));
    }

    ~RejectedCountMockGuard() { GlobalMockObject::verify(); }

    RejectedCountMockGuard(const RejectedCountMockGuard&) = delete;
    RejectedCountMockGuard& operator=(const RejectedCountMockGuard&) = delete;
};

TEST(TestAscendIndexIVFRaBitQ, PublicMutationCountsRejectBeforeNarrowingOrPointerAccess)
{
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    const auto vector = MakePatternData(1, DIM);
    const faiss::idx_t id = 0;
    const faiss::idx_t intMax = std::numeric_limits<int>::max();
    const faiss::idx_t wrapsToOne = (faiss::idx_t{1} << 32) + 1;
    RejectedCountMockGuard mock;

    for (faiss::idx_t n : {faiss::idx_t{-1}, faiss::idx_t{0}, faiss::ascend::MAX_N, intMax + 1, wrapsToOne})
    {
        SCOPED_TRACE(n);
        ExpectExceptionContains([&] { index.add(n, vector.data()); }, "n must be > 0 and <");
        ExpectExceptionContains([&] { index.add_with_ids(n, vector.data(), &id); }, "n must be > 0 and <");
        ExpectExceptionContains([&] { index.update(n, vector.data(), &id); },
                                n <= 0 ? "greater than 0" : "n must be <");
    }
    for (size_t n : {static_cast<size_t>(intMax) + 1, static_cast<size_t>(wrapsToOne)})
    {
        SCOPED_TRACE(n);
        ExpectExceptionContains([&] { index.remove_ids(n, &id); }, "n exceeds int range");
    }
    ExpectExceptionContains([&] { index.remove_ids(1, nullptr); }, "vector ID list is nullptr");
    ASSERT_NO_THROW(index.remove_ids(0, nullptr));
    EXPECT_EQ(rejectedCountImplCalls, 0);
    EXPECT_EQ(index.ntotal, 0);

    ASSERT_NO_THROW(index.remove_ids(static_cast<size_t>(intMax), &id));
    EXPECT_EQ(rejectedCountImplCalls, 1);
}

size_t CountDeviceMappedIds(const faiss::ascend::LocationTable& table, size_t device)
{
    size_t count = 0;
    table.forEachValidId(
        [&](int64_t id)
        {
            if (table.lookupLocation(id)->deviceIndex() == device) ++count;
        });
    return count;
}

std::vector<std::pair<int64_t, uint64_t>> SnapshotLocations(const faiss::ascend::LocationTable& table)
{
    std::vector<std::pair<int64_t, uint64_t>> locations;
    table.forEachValidId(
        [&](int64_t id)
        {
            const auto loc = *table.lookupLocation(id);
            locations.emplace_back(id,
                                   (uint64_t{loc.deviceIndex()} << 48) | (uint64_t{loc.listId()} << 32) | loc.slot());
        });
    std::sort(locations.begin(), locations.end());
    return locations;
}

size_t CountMappedIds(const faiss::ascend::LocationTable& table)
{
    size_t count = 0;
    table.forEachValidId([&](int64_t) { ++count; });
    return count;
}

TEST(TestAscendIndexIVFRaBitQ, AutomaticAddSkipsOccupiedIdsAfterRemovingLowId)
{
    auto source = MakePretrainedCpuIndex(false);
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    const auto vectors = MakePatternData(3, DIM);
    ASSERT_NO_THROW(index.add(3, vectors.data()));
    const faiss::idx_t lowId = 0;
    ASSERT_NO_THROW(index.remove_ids(1, &lowId));
    ASSERT_EQ(index.ntotal, 2);
    ASSERT_EQ(static_cast<size_t>(index.impl_->locationTable->contains(2)), 1U);

    ASSERT_NO_THROW(index.add(2, vectors.data()));
    EXPECT_EQ(index.ntotal, 4);
    EXPECT_EQ(CountMappedIds(*index.impl_->locationTable), 4U);
    EXPECT_EQ(static_cast<size_t>(index.impl_->locationTable->contains(0)), 0U);
    for (faiss::idx_t id : {1, 2, 3, 4})
    {
        EXPECT_EQ(static_cast<size_t>(index.impl_->locationTable->contains(id)), 1U);
        const auto loc = index.impl_->locationTable->lookupLocation(id);
        ASSERT_TRUE(loc.has_value());
        EXPECT_EQ(loc->deviceIndex(), 0);
    }
}

TEST(TestAscendIndexIVFRaBitQ, AutomaticAddProbesLocallyWithoutPersistentHighWatermark)
{
    auto source = MakePretrainedCpuIndex(false);
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    const auto vectors = MakePatternData(2, DIM);
    const faiss::idx_t occupied = 1;
    ASSERT_NO_THROW(index.add_with_ids(1, vectors.data(), &occupied));
    ASSERT_NO_THROW(index.add(2, vectors.data()));
    EXPECT_EQ(index.ntotal, 3);
    EXPECT_EQ(static_cast<size_t>(index.impl_->locationTable->contains(2)), 1U);
    EXPECT_EQ(static_cast<size_t>(index.impl_->locationTable->contains(3)), 1U);

    const faiss::idx_t removed[] = {2, 3};
    ASSERT_NO_THROW(index.remove_ids(2, removed));
    ASSERT_EQ(index.ntotal, 1);
    ASSERT_NO_THROW(index.add(1, vectors.data()));
    EXPECT_EQ(index.ntotal, 2);
    EXPECT_EQ(static_cast<size_t>(index.impl_->locationTable->contains(2)), 1U);
    EXPECT_EQ(static_cast<size_t>(index.impl_->locationTable->contains(3)), 0U);
    EXPECT_EQ(static_cast<size_t>(index.impl_->locationTable->contains(4)), 0U);
}

TEST(TestAscendIndexIVFRaBitQ, ExplicitAddStillRejectsBatchAndExistingDuplicateIds)
{
    auto source = MakePretrainedCpuIndex(false);
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    const auto vectors = MakePatternData(2, DIM);
    const faiss::idx_t duplicates[] = {42, 42};
    ExpectExceptionContains([&] { index.add_with_ids(2, vectors.data(), duplicates); }, "duplicate id");
    EXPECT_EQ(index.ntotal, 0);
    EXPECT_TRUE(CountMappedIds(*index.impl_->locationTable) == 0);

    ASSERT_NO_THROW(index.add_with_ids(1, vectors.data(), duplicates));
    const faiss::idx_t mixed[] = {43, 42};
    ExpectExceptionContains([&] { index.add_with_ids(2, vectors.data(), mixed); }, "already exists");
    EXPECT_EQ(index.ntotal, 1);
    EXPECT_EQ(CountMappedIds(*index.impl_->locationTable), 1U);
    EXPECT_EQ(static_cast<size_t>(index.impl_->locationTable->contains(42)), 1U);
    EXPECT_EQ(static_cast<size_t>(index.impl_->locationTable->contains(43)), 0U);
}

// Host mutation faults are injected at the publication boundary; successful calls still reach the daemon.
bool failHostAssignment = false;
std::atomic<int> hostUploadCalls{0};
int failHostUploadDevice = -1;
std::mutex hostPublishedMutex;
std::vector<faiss::idx_t> hostPublishedIds;

std::vector<const ::ascend::IndexIVFRaBitQ*> lengthQueryDevices;
std::vector<std::vector<size_t>> lengthQueryCounts;

size_t StubTrackedListLength(const ::ascend::IndexIVFRaBitQ* index, int listId)
{
    const auto device = std::find(lengthQueryDevices.begin(), lengthQueryDevices.end(), index);
    FAISS_THROW_IF_NOT(device != lengthQueryDevices.end());
    ++lengthQueryCounts.at(static_cast<size_t>(device - lengthQueryDevices.begin())).at(listId);
    // Read real fixture state without recursively calling the mocked query or adding test-owned queries.
    return index->activeSnapshot->lists.at(listId)->listLen;
}

struct ListLengthQueryGuard
{
    explicit ListLengthQueryGuard(faiss::ascend::AscendIndexIVFRaBitQImpl* impl)
    {
        lengthQueryDevices.clear();
        for (int deviceId : impl->indexConfig.deviceList) lengthQueryDevices.push_back(impl->getActualIndex(deviceId));
        lengthQueryCounts.assign(lengthQueryDevices.size(), std::vector<size_t>(NLIST, 0));
        MOCKER_CPP(&::ascend::IndexIVFRaBitQ::getListLength).stubs().will(invoke(StubTrackedListLength));
    }
    ~ListLengthQueryGuard() { GlobalMockObject::verify(); }
};

void ExpectListLengthQueries(const std::vector<int>& touchedLists)
{
    for (const auto& counts : lengthQueryCounts)
    {
        for (int listId = 0; listId < NLIST; ++listId)
        {
            EXPECT_EQ(counts[listId], std::count(touchedLists.begin(), touchedLists.end(), listId)) << listId;
        }
    }
}

void StubHostAssignment(faiss::ascend::AscendIndexIVFRaBitQImpl*, int n, const float* x,
                        std::unique_ptr<faiss::idx_t[]>& assign)
{
    for (int i = 0; i < n; ++i)
    {
        assign[i] = x[static_cast<size_t>(i) * DIM] < 0.5f ? 7 : 8;
    }
    if (failHostAssignment)
    {
        assign[n - 1] = -1;  // Fail after earlier rows have already been staged.
    }
}

void StubHostUpload(faiss::ascend::AscendIndexIVFRaBitQImpl* impl, int deviceId,
                    const std::vector<::ascend::IndexIVFRaBitQ::ListAddRequest>& reqs, std::vector<size_t>& baseSlots)
{
    ++hostUploadCalls;
    if (deviceId == failHostUploadDevice)
    {
        throw std::runtime_error("injected host upload failure");
    }
    baseSlots.resize(reqs.size());
    const auto ret = impl->getActualIndex(deviceId)->addVectorsBatch(reqs.data(), reqs.size(), baseSlots.data());
    FAISS_THROW_IF_NOT(ret == ::ascend::APP_ERR_OK);
    std::lock_guard<std::mutex> lock(hostPublishedMutex);
    for (const auto& req : reqs)
    {
        for (size_t i = 0; i < req.numVecs; ++i)
        {
            hostPublishedIds.push_back(static_cast<faiss::idx_t>(req.indices[i]));
        }
    }
}

struct HostAddFaultGuard
{
    HostAddFaultGuard()
    {
        failHostAssignment = false;
        hostUploadCalls = 0;
        failHostUploadDevice = -1;
        hostPublishedIds.clear();
        hostPublishedIds.reserve(16);
        MOCKER_CPP(&faiss::ascend::AscendIndexIVFRaBitQImpl::addL1).stubs().will(invoke(StubHostAssignment));
        MOCKER_CPP(&faiss::ascend::AscendIndexIVFRaBitQImpl::indexIVFRaBitQAddBatch)
            .stubs()
            .will(invoke(StubHostUpload));
    }
    ~HostAddFaultGuard() { GlobalMockObject::verify(); }
};

TEST(TestAscendIndexIVFRaBitQ, AddDeltaConfiguredFieldsRejectBeforeAnyDevicePublication)
{
    for (bool invalidDevice : {false, true})
    {
        auto source = MakePretrainedCpuIndex(false);
        faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig({0, 1}));
        ASSERT_NO_THROW(index.copyFrom(source.index.get()));
        auto* impl = index.impl_.get();
        impl->locationTable =
            std::make_unique<faiss::ascend::LocationTable>(invalidDevice ? 1 : 2, invalidDevice ? NLIST : 1);
        faiss::ascend::AscendIndexIVFRaBitQImpl::AddStaging staging;
        auto& list = staging.try_emplace(7, std::vector<int>{0, 0}).first->second;
        const size_t device = invalidDevice ? 1 : 0;
        list.devices[device].ids = {41};
        list.devices[device].codes = MakePatternData(1, DIM);
        list.occupancy[device] = 1;
        const auto* first = impl->getActualIndex(0)->activeSnapshot.get();
        const auto* second = impl->getActualIndex(1)->activeSnapshot.get();
        EXPECT_THROW(impl->copyVectorToDevice(1, staging), std::exception);
        EXPECT_EQ(index.ntotal, 0);
        EXPECT_EQ(impl->locationTable->allocatedBlockCount(), 0U);
        EXPECT_EQ(impl->getActualIndex(0)->activeSnapshot.get(), first);
        EXPECT_EQ(impl->getActualIndex(1)->activeSnapshot.get(), second);
        EXPECT_FALSE(list.devices[device].published);
    }
}

std::atomic<bool> siblingUploadEntered{false};
std::atomic<bool> releaseSiblingUpload{false};
std::atomic<bool> failedUploadReturned{false};

void StubHostUploadWithWaitingSibling(faiss::ascend::AscendIndexIVFRaBitQImpl* impl, int deviceId,
                                      const std::vector<::ascend::IndexIVFRaBitQ::ListAddRequest>& reqs,
                                      std::vector<size_t>& baseSlots)
{
    if (deviceId == 0)
    {
        failedUploadReturned.store(true);
        throw std::runtime_error("injected sibling upload failure");
    }
    siblingUploadEntered.store(true);
    while (!releaseSiblingUpload.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const auto ret = impl->getActualIndex(deviceId)->addVectorsBatch(reqs.data(), reqs.size(), baseSlots.data());
    FAISS_THROW_IF_NOT(ret == ::ascend::APP_ERR_OK);
}

TEST(TestAscendIndexIVFRaBitQ, FailedAddCleanupWaitsForSiblingAndKeepsOnlyPublishedBlocks)
{
    auto source = MakePretrainedCpuIndex(false);
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig({0, 1}));
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    auto vectors = MakePatternData(2, DIM);
    vectors[0] = vectors[DIM] = 0.0f;
    const faiss::idx_t ids[] = {41, 42};  // Sibling's reserved slot shares the failed device's block.
    siblingUploadEntered = releaseSiblingUpload = failedUploadReturned = false;
    failHostAssignment = false;
    MOCKER_CPP(&faiss::ascend::AscendIndexIVFRaBitQImpl::addL1).stubs().will(invoke(StubHostAssignment));
    MOCKER_CPP(&faiss::ascend::AscendIndexIVFRaBitQImpl::indexIVFRaBitQAddBatch)
        .stubs()
        .will(invoke(StubHostUploadWithWaitingSibling));
    auto result = std::async(std::launch::async,
                             [&]
                             {
                                 try
                                 {
                                     index.add_with_ids(2, vectors.data(), ids);
                                 }
                                 catch (const std::exception&)
                                 {
                                     return true;
                                 }
                                 return false;
                             });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while ((!siblingUploadEntered.load() || !failedUploadReturned.load()) &&
           std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(siblingUploadEntered.load());
    EXPECT_TRUE(failedUploadReturned.load());
    EXPECT_EQ(result.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);
    {
        std::lock_guard<std::mutex> lock(index.impl_->locationMutex);
        EXPECT_EQ(index.impl_->locationTable->allocatedBlockCount(), 1U);
        EXPECT_EQ(CountMappedIds(*index.impl_->locationTable), 0U);
    }
    releaseSiblingUpload = true;  // Always release before any fatal assertion or mock teardown.
    EXPECT_TRUE(result.get());
    EXPECT_EQ(index.ntotal, 1);
    EXPECT_EQ(index.impl_->locationTable->allocatedBlockCount(), 1U);
    EXPECT_FALSE(index.impl_->locationTable->contains(41));
    EXPECT_TRUE(index.impl_->locationTable->contains(42));
    EXPECT_EQ(index.impl_->getActualIndex(1)->getListLength(7), 1U);
    GlobalMockObject::verify();
    ASSERT_NO_THROW(index.remove_ids(1, ids + 1));
    EXPECT_EQ(index.impl_->locationTable->allocatedBlockCount(), 0U);
}

TEST(TestAscendIndexIVFRaBitQ, ReservationFailureReclaimsEarlierBlocksBeforeAnyUpload)
{
    auto source = MakePretrainedCpuIndex(false);
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    index.impl_->locationTable = std::make_unique<faiss::ascend::LocationTable>(1, NLIST, 41);
    auto vectors = MakePatternData(2, DIM);
    vectors[0] = vectors[DIM] = 0.0f;
    const faiss::idx_t ids[] = {41, 42};
    HostAddFaultGuard mock;
    EXPECT_THROW(index.add_with_ids(2, vectors.data(), ids), std::exception);
    EXPECT_EQ(hostUploadCalls, 0);
    EXPECT_EQ(index.ntotal, 0);
    EXPECT_EQ(index.impl_->locationTable->allocatedBlockCount(), 0U);
}

TEST(TestAscendIndexIVFRaBitQ, FailedStagingAndUploadDoNotLeakIntoRetry)
{
    auto source = MakePretrainedCpuIndex(false);
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    auto vectors = MakePatternData(2, DIM);
    vectors[0] = vectors[DIM] = 0.0f;
    const faiss::idx_t ids[] = {41, 42};
    HostAddFaultGuard mock;

    failHostAssignment = true;
    EXPECT_THROW(index.add_with_ids(2, vectors.data(), ids), std::exception);
    EXPECT_EQ(hostUploadCalls.load(), 0);
    EXPECT_EQ(index.ntotal, 0);
    EXPECT_EQ(index.impl_->deviceAddNumMap[7][0], 0);
    EXPECT_TRUE(CountMappedIds(*index.impl_->locationTable) == 0);

    failHostAssignment = false;
    failHostUploadDevice = 0;
    ExpectExceptionContains([&] { index.add_with_ids(2, vectors.data(), ids); }, "injected host upload failure");
    EXPECT_EQ(index.ntotal, 0);
    EXPECT_TRUE(CountMappedIds(*index.impl_->locationTable) == 0);
    EXPECT_EQ(index.impl_->locationTable->allocatedBlockCount(), 0U);
    EXPECT_EQ(index.impl_->deviceAddNumMap[7][0], 0);
    EXPECT_EQ(index.impl_->getActualIndex(0)->getListLength(7), 0U);

    failHostUploadDevice = -1;
    ASSERT_NO_THROW(index.add_with_ids(2, vectors.data(), ids));
    EXPECT_EQ(index.ntotal, 2);
    EXPECT_EQ(index.impl_->deviceAddNumMap[7][0], 2);
    std::vector<int64_t> stored;
    ASSERT_EQ(index.impl_->getActualIndex(0)->getListIds(7, stored), ::ascend::APP_ERR_OK);
    EXPECT_THAT(stored, ElementsAre(41, 42));
    EXPECT_EQ(CountDeviceMappedIds(*index.impl_->locationTable, 0), 2U);
}

TEST(TestAscendIndexIVFRaBitQ, NegativeAddRejectsBeforeStagingAndAllowsRetry)
{
    auto source = MakePretrainedCpuIndex(false);
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    auto vectors = MakePatternData(2, DIM);
    vectors[0] = vectors[DIM] = 0.0f;
    const faiss::idx_t invalid[] = {41, -1};
    const faiss::idx_t valid[] = {41, 42};
    HostAddFaultGuard mock;
    ExpectExceptionContains([&] { index.add_with_ids(2, vectors.data(), invalid); }, "nonnegative");
    EXPECT_EQ(hostUploadCalls.load(), 0);
    EXPECT_EQ(index.ntotal, 0);
    EXPECT_EQ(CountMappedIds(*index.impl_->locationTable), 0U);
    EXPECT_EQ(index.impl_->getActualIndex(0)->getListLength(7), 0U);
    ASSERT_NO_THROW(index.add_with_ids(2, vectors.data(), valid));
    EXPECT_EQ(index.ntotal, 2);
    const auto loc = index.impl_->locationTable->lookupLocation(42);
    ASSERT_TRUE(loc.has_value());
    EXPECT_EQ(loc->slot(), 1U);
}

TEST(TestAscendIndexIVFRaBitQ, StagingUsesTouchedListDeviceOccupancyAndOrdinals)
{
    auto source = MakePretrainedCpuIndex(false);
    source.listId = 7;
    source.ids = {71, 72};
    source.codes.resize(source.ids.size() * source.index->code_size, 0);
    source.index->invlists->add_entries(7, 2, source.ids.data(), source.codes.data());
    source.index->ntotal = 2;
    // Physical device IDs deliberately differ from the ordinal used in host maps.
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig({1, 0}));
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    const auto firstLoc = index.impl_->locationTable->lookupLocation(71);
    ASSERT_TRUE(firstLoc.has_value());
    EXPECT_EQ(firstLoc->deviceIndex(), 0);
    const auto secondLoc = index.impl_->locationTable->lookupLocation(72);
    ASSERT_TRUE(secondLoc.has_value());
    EXPECT_EQ(secondLoc->deviceIndex(), 1);
    const faiss::idx_t remove = 72;
    ASSERT_NO_THROW(index.remove_ids(1, &remove));
    auto vectors = MakePatternData(4, DIM);
    vectors[0] = vectors[DIM] = vectors[3 * DIM] = 0.0f;
    vectors[2 * DIM] = 1.0f;
    const faiss::idx_t ids[] = {73, 74, 75, 76};
    // Stale host occupancy must not influence staging; the touched daemon list is authoritative.
    index.impl_->deviceAddNumMap[7] = {100, 100};
    index.impl_->deviceAddNumMap[8] = {100, 100};
    ListLengthQueryGuard mock(index.impl_.get());
    failHostAssignment = false;
    MOCKER_CPP(&faiss::ascend::AscendIndexIVFRaBitQImpl::addL1).stubs().will(invoke(StubHostAssignment));
    faiss::ascend::AscendIndexIVFRaBitQImpl::AddStaging staging;
    for (size_t page = 0; page < 4; ++page)
    {
        ASSERT_NO_THROW(index.impl_->stageAdd(1, vectors.data() + page * DIM, ids + page, staging));
        ExpectListLengthQueries(page < 2 ? std::vector<int>{7} : std::vector<int>{7, 8});
    }
    EXPECT_EQ(staging.at(7).occupancy, (std::vector<int>{2, 2}));
    EXPECT_EQ(staging.at(8).occupancy, (std::vector<int>{1, 0}));
    EXPECT_TRUE(staging.at(8).devices[1].ids.empty());  // Queried once even though no row was assigned.
    ASSERT_NO_THROW(index.impl_->copyVectorToDevice(4, staging));
    EXPECT_TRUE(staging.at(8).devices[1].ids.empty());
    EXPECT_FALSE(staging.at(8).devices[1].published);  // No staged rows means no publication request.
    EXPECT_EQ(staging.at(8).occupancy[1], 0);          // Scratch reflects the actual baseline, not the old host cache.
    ExpectListLengthQueries({7, 8});                   // Publication preflight must not query any device again.
    EXPECT_EQ(index.ntotal, 5);
    const int expectedDevices[] = {1, 0, 0, 1};
    const uint32_t expectedSlots[] = {0, 1, 0, 1};
    for (size_t i = 0; i < 4; ++i)
    {
        const auto location = index.impl_->locationTable->lookupLocation(ids[i]);
        ASSERT_TRUE(location.has_value());
        EXPECT_EQ(location->deviceIndex(), expectedDevices[i]);
        EXPECT_EQ(location->listId(), i == 2 ? 8 : 7);
        EXPECT_EQ(location->slot(), expectedSlots[i]);
    }
    for (int listId : {7, 8})
    {
        for (size_t d = 0; d < 2; ++d)
        {
            const auto* actual = index.impl_->getActualIndex(d == 0 ? 1 : 0);
            EXPECT_EQ(actual->activeSnapshot->lists.at(listId)->listLen,
                      static_cast<size_t>(staging.at(listId).occupancy[d]));
            const auto& device = staging.at(listId).devices[d];
            EXPECT_EQ(device.published, !device.ids.empty());
            EXPECT_EQ(
                index.impl_->deviceAddNumMap[listId][d],
                device.ids.empty() ? 100 : staging.at(listId).occupancy[d]);  // No request preserves the old count.
        }
    }
}

TEST(TestAscendIndexIVFRaBitQ, RawMultiListBatchPublishesOnceAndEmptyBatchIsNoop)
{
    auto source = MakePretrainedCpuIndex(false);
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    auto* actual = index.impl_->getActualIndex(0);
    const auto version = actual->activeSnapshot->publishVer;
    auto vectors = MakePatternData(3, DIM);
    vectors[0] = vectors[2 * DIM] = 0.0f;
    vectors[DIM] = 1.0f;
    const faiss::idx_t ids[] = {81, 82, 83};
    HostAddFaultGuard mock;
    ASSERT_NO_THROW(index.add_with_ids(3, vectors.data(), ids));
    EXPECT_EQ(hostUploadCalls.load(), 1);
    EXPECT_EQ(actual->activeSnapshot->publishVer, version + 1);
    EXPECT_EQ(actual->activeSnapshot->total, 3U);
    EXPECT_EQ(actual->activeSnapshot->maxStoredId, 83U);
    EXPECT_EQ(index.ntotal, 3);
    EXPECT_EQ(index.impl_->deviceAddNumMap[7][0], 2);
    EXPECT_EQ(index.impl_->deviceAddNumMap[8][0], 1);
    const auto loc = index.impl_->locationTable->lookupLocation(83);
    ASSERT_TRUE(loc.has_value());
    EXPECT_EQ(loc->slot(), 1U);
    const auto* snapshot = actual->activeSnapshot.get();
    const ::ascend::IndexIVFRaBitQ::ListAddRequest empty[] = {
        {7, 0, nullptr, nullptr}, {8, 0, nullptr, nullptr}, {7, 0, nullptr, nullptr}};
    size_t bases[3] = {};
    EXPECT_EQ(actual->addVectorsBatch(empty, 3, bases), ::ascend::APP_ERR_OK);
    EXPECT_THAT(bases, ElementsAre(2U, 1U, 2U));
    EXPECT_EQ(actual->activeSnapshot.get(), snapshot);
    EXPECT_EQ(actual->addVectorsBatch(nullptr, 0, nullptr), ::ascend::APP_ERR_OK);
}

TEST(TestAscendIndexIVFRaBitQ, PartialDeviceBatchReconcilesCountMappingFilterAndRetry)
{
    auto source = MakePretrainedCpuIndex(false);
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig({1, 0}));
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    auto vectors = MakePatternData(4, DIM);
    vectors[0] = vectors[DIM] = 0.0f;
    vectors[2 * DIM] = vectors[3 * DIM] = 1.0f;
    const faiss::idx_t ids[] = {51, 8192 + 52, 53, 8192 + 54};
    faiss::IDSelectorRange sel(0, 100);
    index.impl_->getCachedFilter(&sel);
    const auto generation = index.impl_->cachedFilterGeneration;
    auto* success = index.impl_->getActualIndex(1);
    auto* failed = index.impl_->getActualIndex(0);
    const auto successVersion = success->activeSnapshot->publishVer;
    const auto* failedSnapshot = failed->activeSnapshot.get();
    HostAddFaultGuard mock;
    failHostUploadDevice = 0;
    ExpectExceptionContains([&] { index.add_with_ids(4, vectors.data(), ids); }, "injected host upload failure");
    EXPECT_EQ(hostUploadCalls.load(), 2);
    EXPECT_THAT(hostPublishedIds, UnorderedElementsAre(51, 53));
    EXPECT_EQ(index.ntotal, 2);
    EXPECT_EQ(CountMappedIds(*index.impl_->locationTable), 2U);
    EXPECT_EQ(index.impl_->locationTable->allocatedBlockCount(), 1U);
    EXPECT_EQ(success->activeSnapshot->publishVer, successVersion + 1);
    EXPECT_EQ(success->activeSnapshot->total, 2U);
    EXPECT_EQ(success->activeSnapshot->maxStoredId, 53U);
    EXPECT_EQ(failed->activeSnapshot.get(), failedSnapshot);
    EXPECT_EQ(failed->activeSnapshot->total, 0U);
    for (int list : {7, 8})
    {
        EXPECT_EQ(index.impl_->deviceAddNumMap[list][0], 1);
        EXPECT_EQ(index.impl_->deviceAddNumMap[list][1], 0);
        EXPECT_EQ(success->getListLength(list), 1U);
        EXPECT_EQ(failed->getListLength(list), 0U);
    }
    for (int row : {0, 2})
    {
        const auto loc = index.impl_->locationTable->lookupLocation(ids[row]);
        ASSERT_TRUE(loc.has_value());
        EXPECT_EQ(loc->deviceIndex(), 0);  // Logical ordinal, not physical device 1.
        EXPECT_EQ(loc->slot(), 0U);
    }
    EXPECT_FALSE(index.impl_->locationTable->contains(ids[1]));
    EXPECT_FALSE(index.impl_->locationTable->contains(ids[3]));
    EXPECT_GT(index.impl_->cachedFilterGeneration, generation);
    EXPECT_EQ(index.impl_->FindFilterCacheHit(index.impl_->MakeFilterCacheKey(&sel)), -1);
    failHostUploadDevice = -1;
    std::vector<float> retry(vectors.begin() + DIM, vectors.begin() + 2 * DIM);
    retry.insert(retry.end(), vectors.begin() + 3 * DIM, vectors.end());
    const faiss::idx_t retryIds[] = {ids[1], ids[3]};
    ASSERT_NO_THROW(index.add_with_ids(2, retry.data(), retryIds));
    EXPECT_EQ(index.ntotal, 4);
    EXPECT_EQ(index.impl_->locationTable->allocatedBlockCount(), 2U);
    EXPECT_EQ(failed->activeSnapshot->total, 2U);
    EXPECT_EQ(CountDeviceMappedIds(*index.impl_->locationTable, 1), 2U);
}

int failEncodedBatchDevice = -1;
std::vector<uint64_t> encodedBatchCalls;
std::vector<uint64_t> encodedBatchBeforeVersion;
std::vector<const ::ascend::IndexIVFRaBitQ::GlobalSnapshot*> encodedBatchBeforeSnapshot;
void StubEncodedHostBatch(faiss::ascend::AscendIndexIVFRaBitQImpl* impl, int deviceId,
                          const std::vector<::ascend::IndexIVFRaBitQ::ListEncodedAddRequest>& reqs,
                          std::vector<size_t>& baseSlots)
{
    auto* actual = impl->getActualIndex(deviceId);
    ++encodedBatchCalls[deviceId];
    encodedBatchBeforeVersion[deviceId] = actual->activeSnapshot->publishVer;
    encodedBatchBeforeSnapshot[deviceId] = actual->activeSnapshot.get();
    if (deviceId == failEncodedBatchDevice)
    {
        auto rejected = reqs;
        rejected.push_back({NLIST, 0, nullptr, nullptr});
        FAISS_THROW_IF_NOT(actual->addEncodedVectorsBatch(rejected.data(), rejected.size(), nullptr) ==
                           ::ascend::APP_ERR_INVALID_PARAM);
        throw std::runtime_error("injected encoded batch failure");
    }
    FAISS_THROW_IF_NOT(actual->addEncodedVectorsBatch(reqs.data(), reqs.size(), baseSlots.data()) ==
                       ::ascend::APP_ERR_OK);
}

TEST(TestAscendIndexIVFRaBitQ, EncodedCopyFromBatchesListsAndReconcilesPartialDevices)
{
    auto source = MakePretrainedCpuIndex(false);
    const faiss::idx_t first[] = {91, 92};
    const faiss::idx_t second[] = {93, 94};
    std::vector<uint8_t> codes(2 * source.index->code_size, 0x4C);
    source.index->invlists->add_entries(7, 2, first, codes.data());
    source.index->invlists->add_entries(8, 2, second, codes.data());
    source.index->ntotal = 4;
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig({1, 0}));
    encodedBatchCalls.assign(2, 0);
    encodedBatchBeforeVersion.assign(2, 0);
    encodedBatchBeforeSnapshot.assign(2, nullptr);
    failEncodedBatchDevice = 1;  // Fail the first ordinal; the later device must still publish.
    MOCKER_CPP(&faiss::ascend::AscendIndexIVFRaBitQImpl::indexIVFRaBitQAddEncodedBatch)
        .stubs()
        .will(invoke(StubEncodedHostBatch));
    ExpectExceptionContains([&] { index.copyFrom(source.index.get()); }, "injected encoded batch failure");
    EXPECT_THAT(encodedBatchCalls, ElementsAre(1U, 1U));
    EXPECT_EQ(index.ntotal, 2);
    EXPECT_EQ(index.impl_->locationTable->allocatedBlockCount(), 1U);
    EXPECT_EQ(CountMappedIds(*index.impl_->locationTable), 2U);
    auto* success = index.impl_->getActualIndex(0);
    auto* failed = index.impl_->getActualIndex(1);
    EXPECT_EQ(success->activeSnapshot->publishVer, encodedBatchBeforeVersion[0] + 1);
    EXPECT_EQ(success->activeSnapshot->total, 2U);
    EXPECT_EQ(success->activeSnapshot->maxStoredId, 94U);
    EXPECT_EQ(failed->activeSnapshot.get(), encodedBatchBeforeSnapshot[1]);
    EXPECT_EQ(failed->activeSnapshot->total, 0U);
    EXPECT_FALSE(index.impl_->refineSourcesReady_);
    for (int list : {7, 8})
    {
        EXPECT_EQ(index.impl_->deviceAddNumMap[list][0], 0);
        EXPECT_EQ(index.impl_->deviceAddNumMap[list][1], 1);
        EXPECT_EQ(failed->getListLength(list), 0U);
        EXPECT_EQ(success->getListLength(list), 1U);
    }
    EXPECT_FALSE(index.impl_->locationTable->contains(91));
    EXPECT_FALSE(index.impl_->locationTable->contains(93));
    const auto firstLoc = index.impl_->locationTable->lookupLocation(92);
    ASSERT_TRUE(firstLoc.has_value());
    EXPECT_EQ(firstLoc->deviceIndex(), 1);
    const auto secondLoc = index.impl_->locationTable->lookupLocation(94);
    ASSERT_TRUE(secondLoc.has_value());
    EXPECT_EQ(secondLoc->slot(), 0U);
    GlobalMockObject::verify();
    failEncodedBatchDevice = -1;
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    EXPECT_EQ(index.ntotal, 4);
    EXPECT_EQ(CountMappedIds(*index.impl_->locationTable), 4U);
}

TEST(TestAscendIndexIVFRaBitQ, PartialRefinePublicationStaysNotReadyUntilResetOrValidLoad)
{
    auto source = MakePretrainedCpuIndex(false);
    auto config = MakeConfig({0, 1});
    config.needRefine = true;
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, config);
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    auto vectors = MakePatternData(2, DIM);
    vectors[0] = vectors[DIM] = 0.0f;
    const faiss::idx_t ids[] = {0, 1};
    HostAddFaultGuard mock;
    failHostUploadDevice = 0;  // Leave ID 1 only, so loading one row must reject the hole.
    EXPECT_THROW(index.add_with_ids(2, vectors.data(), ids), std::exception);
    ASSERT_EQ(index.ntotal, 1);
    EXPECT_TRUE(index.impl_->srcIndexes.empty());
    EXPECT_TRUE(index.impl_->copiedSources_);
    EXPECT_FALSE(index.impl_->refineSourcesReady_);
    EXPECT_THROW(index.remove_ids(1, ids), std::exception);
    const faiss::idx_t next = 2;
    ExpectExceptionContains([&] { index.add_with_ids(1, vectors.data(), &next); }, "not ready");
    // A matching row count cannot conceal a partial-publication hole.
    index.impl_->srcIndexes.resize(DIM);
    ExpectExceptionContains([&] { index.impl_->getRefineSources(); }, "not ready");
    EXPECT_THAT(hostPublishedIds, ElementsAre(1));
    ExpectExceptionContains([&] { index.loadRefineSources(vectors.data() + DIM, 1); }, "unique IDs");
    EXPECT_FALSE(index.impl_->refineSourcesReady_);
    ASSERT_NO_THROW(index.reset());
    EXPECT_FALSE(index.impl_->copiedSources_);
    EXPECT_TRUE(index.impl_->refineSourcesReady_);
    failHostUploadDevice = -1;
    ASSERT_NO_THROW(index.add(2, vectors.data()));
    EXPECT_EQ(index.ntotal, 2);
    EXPECT_EQ(index.impl_->srcIndexes, vectors);
}

TEST(TestAscendIndexIVFRaBitQ, EmptyCopiedSourcesKeepSequentialModeUntilReset)
{
    auto source = MakePretrainedCpuIndex(false);
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    const auto vector = MakePatternData(1, DIM);
    ASSERT_NO_THROW(index.loadRefineSources(vector.data(), 0));
    const faiss::idx_t id = 9;
    ExpectExceptionContains([&] { index.add_with_ids(1, vector.data(), &id); }, "sequential IDs");
    ExpectExceptionContains([&] { index.remove_ids(1, &id); }, "copied refine sources");
    ExpectExceptionContains([&] { index.update(1, vector.data(), &id); }, "copied refine sources");
    ExpectExceptionContains([&] { index.attachRefineSources(vector.data(), 0); }, "copied refine sources");
    ASSERT_NO_THROW(index.reset());
    ASSERT_NO_THROW(index.add_with_ids(1, vector.data(), &id));
    EXPECT_TRUE(index.impl_->srcIndexes.empty());
}

TEST(TestAscendIndexIVFRaBitQ, DuplicateAndMissingDeletesOnlyCountStoredIds)
{
    auto source = MakePretrainedCpuIndex(false);
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    const auto vectors = MakePatternData(2, DIM);
    const faiss::idx_t ids[] = {61, 62};
    ASSERT_NO_THROW(index.add_with_ids(2, vectors.data(), ids));
    const faiss::idx_t deletes[] = {61, 61, 999, 999};
    ASSERT_NO_THROW(index.remove_ids(4, deletes));
    EXPECT_EQ(index.ntotal, 1);
    EXPECT_EQ(CountMappedIds(*index.impl_->locationTable), 1U);
    EXPECT_EQ(static_cast<size_t>(index.impl_->locationTable->contains(62)), 1U);
    EXPECT_EQ(CountDeviceMappedIds(*index.impl_->locationTable, 0), 1U);
    ASSERT_NO_THROW(index.remove_ids(4, deletes));
    EXPECT_EQ(index.ntotal, 1);
    size_t stored = 0;
    int counted = 0;
    for (int list = 0; list < NLIST; ++list)
    {
        stored += index.impl_->getActualIndex(0)->getListLength(list);
        counted += index.impl_->deviceAddNumMap[list][0];
    }
    EXPECT_EQ(stored, 1U);
    EXPECT_EQ(counted, 1);
}

int hostDeleteCalls = 0;
int failHostDeleteCall = 0;
void StubHostDelete(faiss::ascend::AscendIndexIVFRaBitQImpl* impl, int deviceId, const std::vector<faiss::idx_t>& ids,
                    const std::vector<int>& lists, const std::vector<size_t>& slots,
                    std::vector<::ascend::IndexIVFRaBitQ::MovedSlot>& moved,
                    const ::ascend::IndexIVFRaBitQ::MoveValidator& validateMoves)
{
    if (++hostDeleteCalls == failHostDeleteCall) throw std::runtime_error("injected host delete failure");
    std::vector<::ascend::Index::idx_t> daemonIds(ids.begin(), ids.end());
    FAISS_THROW_IF_NOT(impl->getActualIndex(deviceId)->removeIds(ids.size(), daemonIds.data(), lists.data(),
                                                                 slots.data(), &moved,
                                                                 validateMoves) == ::ascend::APP_ERR_OK);
}

TEST(TestAscendIndexIVFRaBitQ, FailedSecondDeviceDeletePreservesSuccessfulDeviceCount)
{
    auto source = MakePretrainedCpuIndex();
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig({0, 1}));
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    const auto oldCounts = index.impl_->deviceAddNumMap;
    std::vector<std::pair<faiss::idx_t, faiss::ascend::Location>> oldSecondLocations;
    for (const auto id : source.ids)
    {
        const auto loc = index.impl_->locationTable->lookupLocation(id);
        ASSERT_TRUE(loc.has_value());
        if (loc->deviceIndex() == 1)
        {
            oldSecondLocations.emplace_back(id, *loc);
        }
    }
    hostDeleteCalls = 0;
    failHostDeleteCall = 2;
    MOCKER_CPP(&faiss::ascend::AscendIndexIVFRaBitQImpl::deleteFromIVFRaBitQ).stubs().will(invoke(StubHostDelete));
    ExpectExceptionContains([&] { index.remove_ids(source.ids.size(), source.ids.data()); },
                            "injected host delete failure");
    EXPECT_EQ(hostDeleteCalls, 2);
    size_t stored = 0;
    for (int dev : {0, 1})
    {
        const auto count = index.impl_->getActualIndex(dev)->getListLength(source.listId);
        stored += count;
        EXPECT_EQ(index.impl_->deviceAddNumMap[source.listId][dev], static_cast<int>(count));
        EXPECT_EQ(CountDeviceMappedIds(*index.impl_->locationTable, dev), count);
    }
    EXPECT_GT(stored, 0U);
    EXPECT_EQ(index.impl_->deviceAddNumMap[source.listId][1], oldCounts[source.listId][1]);
    EXPECT_EQ(CountDeviceMappedIds(*index.impl_->locationTable, 1), oldSecondLocations.size());
    for (const auto& entry : oldSecondLocations)
    {
        const auto loc = index.impl_->locationTable->lookupLocation(entry.first);
        ASSERT_TRUE(loc.has_value());
        EXPECT_EQ(loc->deviceIndex(), entry.second.deviceIndex());
        EXPECT_EQ(loc->listId(), entry.second.listId());
        EXPECT_EQ(loc->slot(), entry.second.slot());
    }
    EXPECT_LT(stored, source.ids.size());
    EXPECT_EQ(index.ntotal, static_cast<faiss::idx_t>(stored));
    EXPECT_EQ(CountMappedIds(*index.impl_->locationTable), stored);
    GlobalMockObject::verify();
    ASSERT_NO_THROW(index.remove_ids(source.ids.size(), source.ids.data()));
    EXPECT_EQ(index.ntotal, 0);
}

TEST(TestAscendIndexIVFRaBitQ, DeleteCleanupKeepsOtherDeviceReservationUntilItsOutcome)
{
    auto source = MakePretrainedCpuIndex(false);
    source.listId = 7;
    source.ids = {41, 42};
    source.codes.resize(2 * source.index->code_size, 0);
    source.index->invlists->add_entries(7, 2, source.ids.data(), source.codes.data());
    source.index->ntotal = 2;
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig({0, 1}));
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    hostDeleteCalls = 0;
    failHostDeleteCall = 2;
    MOCKER_CPP(&faiss::ascend::AscendIndexIVFRaBitQImpl::deleteFromIVFRaBitQ).stubs().will(invoke(StubHostDelete));
    EXPECT_THROW(index.remove_ids(2, source.ids.data()), std::exception);
    EXPECT_EQ(index.ntotal, 1);
    EXPECT_EQ(index.impl_->locationTable->allocatedBlockCount(), 1U);
    EXPECT_EQ(CountMappedIds(*index.impl_->locationTable), 1U);
    GlobalMockObject::verify();
    ASSERT_NO_THROW(index.remove_ids(2, source.ids.data()));
    EXPECT_EQ(index.ntotal, 0);
    EXPECT_EQ(index.impl_->locationTable->allocatedBlockCount(), 0U);
}

TEST(TestAscendIndexIVFRaBitQ, ImportPreflightRejectsInvalidIdsAndCountsWithoutChangingState)
{
    auto source = MakePretrainedCpuIndex();
    const faiss::idx_t oldIds[] = {0, 1, 2};
    source.index->invlists->update_entries(source.listId, 0, 3, oldIds, source.codes.data());
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    const auto originals = MakePatternData(3, DIM);
    ASSERT_NO_THROW(index.loadRefineSources(originals.data(), 3));
    const auto locations = SnapshotLocations(*index.impl_->locationTable);
    const auto counts = index.impl_->deviceAddNumMap;
    const auto generation = index.impl_->cachedFilterGeneration;
    const auto* snapshot = index.impl_->getActualIndex(0)->activeSnapshot.get();
    for (int invalid : {0, 1, 2, 3})
    {
        auto replacement = MakePretrainedCpuIndex();
        replacement.index->nprobe = NPROBE + 1;
        if (invalid < 2)
        {
            const faiss::idx_t badIds[] = {50, invalid == 0 ? -1 : 50, 52};
            replacement.index->invlists->update_entries(replacement.listId, 0, 3, badIds, replacement.codes.data());
        }
        else if (invalid == 2)
        {
            replacement.index->invlists->add_entries((replacement.listId + 1) % NLIST, 1, replacement.ids.data(),
                                                     replacement.codes.data());
            ++replacement.index->ntotal;  // Duplicate across lists, not merely within one list.
        }
        else
        {
            ++replacement.index->ntotal;
        }
        ExpectExceptionContains([&] { index.copyFrom(replacement.index.get()); }, invalid == 0   ? "nonnegative"
                                                                                  : invalid == 3 ? "count"
                                                                                                 : "duplicate id");
        EXPECT_EQ(index.ntotal, 3);
        EXPECT_EQ(index.getNumProbes(), NPROBE);
        EXPECT_EQ(SnapshotLocations(*index.impl_->locationTable), locations);
        EXPECT_EQ(index.impl_->deviceAddNumMap, counts);
        EXPECT_EQ(index.impl_->srcIndexes, originals);
        EXPECT_TRUE(index.impl_->copiedSources_);
        EXPECT_TRUE(index.impl_->refineSourcesReady_);
        EXPECT_EQ(index.impl_->cachedFilterGeneration, generation);
        EXPECT_EQ(index.impl_->getActualIndex(0)->activeSnapshot.get(), snapshot);
    }
}

TEST(TestAscendIndexIVFRaBitQ, DerivedCountingBatchUsesVirtualMembership)
{
    auto source = MakePretrainedCpuIndex();
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    for (int64_t block = 1; block <= 64; ++block)
    {
        const int64_t id = block * faiss::ascend::LocationTable::kBlockSlots;
        index.impl_->locationTable->setLocation(id, 0, 0, 0);
        index.impl_->locationTable->clearValid(id);  // Many allocated blocks, no extra stored IDs.
    }
    struct CountingBatch : faiss::IDSelectorBatch
    {
        mutable size_t probes = 0;
        CountingBatch(size_t n, const faiss::idx_t* ids) : faiss::IDSelectorBatch(n, ids) {}
        bool is_member(faiss::idx_t id) const override
        {
            ++probes;
            return faiss::IDSelectorBatch::is_member(id);
        }
    };
    const faiss::idx_t selected[] = {source.ids[1], source.ids[1], 999999};
    CountingBatch batch(3, selected);
    EXPECT_EQ(static_cast<faiss::Index&>(index).remove_ids(batch), 1U);
    EXPECT_EQ(batch.probes, source.ids.size());
    EXPECT_EQ(index.ntotal, 2);
    const auto loc = index.impl_->locationTable->lookupLocation(source.ids[2]);
    ASSERT_TRUE(loc.has_value());
    EXPECT_EQ(loc->slot(), 1U);
    EXPECT_EQ(index.impl_->deviceAddNumMap[source.listId][0], 2);
    ASSERT_NO_THROW(index.remove_ids(1, &source.ids[2]));
    EXPECT_EQ(index.ntotal, 1);
}

TEST(TestAscendIndexIVFRaBitQ, DerivedBatchAndArrayDeletesHonorRestrictingAndExpandingPredicates)
{
    struct DerivedBatch : faiss::IDSelectorBatch
    {
        bool expand;
        faiss::idx_t target;
        mutable size_t probes = 0;
        DerivedBatch(const faiss::idx_t* ids, bool expansion, faiss::idx_t id)
            : faiss::IDSelectorBatch(1, ids), expand(expansion), target(id)
        {
        }
        bool is_member(faiss::idx_t id) const override
        {
            ++probes;
            return expand ? faiss::IDSelectorBatch::is_member(id) || id == target
                          : faiss::IDSelectorBatch::is_member(id) && id != target;
        }
    };
    struct DerivedArray : faiss::IDSelectorArray
    {
        bool expand;
        faiss::idx_t target;
        mutable size_t probes = 0;
        DerivedArray(const faiss::idx_t* ids, bool expansion, faiss::idx_t id)
            : faiss::IDSelectorArray(1, ids), expand(expansion), target(id)
        {
        }
        bool is_member(faiss::idx_t id) const override
        {
            ++probes;
            return expand ? faiss::IDSelectorArray::is_member(id) || id == target
                          : faiss::IDSelectorArray::is_member(id) && id != target;
        }
    };
    for (bool array : {false, true})
    {
        for (bool expand : {false, true})
        {
            SCOPED_TRACE(array);
            SCOPED_TRACE(expand);
            auto source = MakePretrainedCpuIndex();
            faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
            ASSERT_NO_THROW(index.copyFrom(source.index.get()));
            const faiss::idx_t selected = source.ids[0];
            const faiss::idx_t target = expand ? source.ids[1] : selected;
            DerivedBatch batch(&selected, expand, target);
            DerivedArray arr(&selected, expand, target);
            const faiss::IDSelector& selector =
                array ? static_cast<const faiss::IDSelector&>(arr) : static_cast<const faiss::IDSelector&>(batch);
            EXPECT_EQ(static_cast<faiss::Index&>(index).remove_ids(selector), expand ? 2U : 0U);
            EXPECT_EQ(array ? arr.probes : batch.probes, source.ids.size());
            EXPECT_EQ(index.ntotal, expand ? 1 : 3);
            EXPECT_EQ(index.impl_->locationTable->contains(selected), !expand);
            EXPECT_TRUE(index.impl_->locationTable->contains(source.ids[2]));
        }
    }
}

TEST(TestAscendIndexIVFRaBitQ, ExactBatchAndArrayDeletesUseStoredPayload)
{
    for (bool array : {false, true})
    {
        auto source = MakePretrainedCpuIndex();
        faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
        ASSERT_NO_THROW(index.copyFrom(source.index.get()));
        const faiss::idx_t selected[] = {source.ids[1], source.ids[1], -1, 999999};
        faiss::IDSelectorBatch batch(4, selected);
        faiss::IDSelectorArray arr(4, selected);
        const faiss::IDSelector& selector =
            array ? static_cast<const faiss::IDSelector&>(arr) : static_cast<const faiss::IDSelector&>(batch);
        EXPECT_EQ(static_cast<faiss::Index&>(index).remove_ids(selector), 1U);
        EXPECT_EQ(index.ntotal, 2);
        const auto loc = index.impl_->locationTable->lookupLocation(source.ids[2]);
        ASSERT_TRUE(loc.has_value());
        EXPECT_EQ(loc->slot(), 1U);
    }
}

int encodedImportCalls = 0;
::ascend::APP_ERROR StubEncodedImportFailure(::ascend::IndexIVFRaBitQ*,
                                             const ::ascend::IndexIVFRaBitQ::ListEncodedAddRequest*, size_t, size_t*)
{
    ++encodedImportCalls;
    return ::ascend::APP_ERR_INNER_ERROR;
}

TEST(TestAscendIndexIVFRaBitQ, FailedImportAfterTableSwapCleansConcreteReplacement)
{
    auto source = MakePretrainedCpuIndex();
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    encodedImportCalls = 0;
    MOCKER_CPP(&::ascend::IndexIVFRaBitQ::addEncodedVectorsBatch).stubs().will(invoke(StubEncodedImportFailure));
    EXPECT_THROW(index.copyFrom(source.index.get()), std::exception);
    EXPECT_EQ(encodedImportCalls, 1);
    EXPECT_EQ(index.ntotal, 0);
    EXPECT_EQ(index.impl_->locationTable->allocatedBlockCount(), 0U);
    EXPECT_EQ(index.impl_->getActualIndex(0)->getListLength(source.listId), 0U);
    GlobalMockObject::verify();
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    EXPECT_EQ(index.ntotal, 3);
    EXPECT_EQ(index.impl_->locationTable->allocatedBlockCount(), 1U);
}

TEST(TestAscendIndexIVFRaBitQ, CopyFromReplacesListsMappingsAndSourceState)
{
    auto source = MakePretrainedCpuIndex();
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    ListLengthQueryGuard mock(index.impl_.get());
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    ExpectListLengthQueries({});
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    ExpectListLengthQueries({});  // Re-import resets old nonzero lists rather than querying their lengths.
    EXPECT_EQ(index.ntotal, 3);
    EXPECT_EQ(index.impl_->getActualIndex(0)->activeSnapshot->lists.at(source.listId)->listLen, 3U);
    EXPECT_EQ(index.impl_->deviceAddNumMap[source.listId][0], 3);
    EXPECT_EQ(CountMappedIds(*index.impl_->locationTable), 3U);
    auto empty = MakePretrainedCpuIndex(false);
    ASSERT_NO_THROW(index.copyFrom(empty.index.get()));
    ExpectListLengthQueries({});
    EXPECT_EQ(index.ntotal, 0);
    EXPECT_TRUE(CountMappedIds(*index.impl_->locationTable) == 0);
    EXPECT_EQ(index.impl_->getActualIndex(0)->activeSnapshot->lists.at(source.listId)->listLen, 0U);
    EXPECT_FALSE(index.impl_->copiedSources_);
    EXPECT_TRUE(index.impl_->refineSourcesReady_);
}

int failEncodedImportCall = 0;

::ascend::APP_ERROR StubEncodedImport(::ascend::IndexIVFRaBitQ* index,
                                      const ::ascend::IndexIVFRaBitQ::ListEncodedAddRequest* reqs, size_t requestCount,
                                      size_t* baseSlotsOut)
{
    if (++encodedImportCalls == failEncodedImportCall) return ::ascend::APP_ERR_INVALID_PARAM;
    for (size_t r = 0; r < requestCount; ++r)
    {
        const auto& req = reqs[r];
        const size_t baseSlot = index->activeSnapshot->lists.at(req.listId)->listLen;
        if (baseSlotsOut != nullptr) baseSlotsOut[r] = baseSlot;
        SeedList(index, req.listId, baseSlot + req.numVecs);  // Reset sees each committed daemon count.
    }
    return ::ascend::APP_ERR_OK;
}

void StubOrthogonalSetupFailure(faiss::ascend::AscendIndexIVFRaBitQImpl*, std::vector<float>&)
{
    throw std::runtime_error("injected orthogonal setup failure");
}

struct ImportFaultMockGuard
{
    ImportFaultMockGuard()
    {
        encodedImportCalls = failEncodedImportCall = 0;
        MOCKER_CPP(&::ascend::IndexIVFRaBitQ::addEncodedVectorsBatch).stubs().will(invoke(StubEncodedImport));
    }
    ~ImportFaultMockGuard() { GlobalMockObject::verify(); }
};

TEST(TestAscendIndexIVFRaBitQ, CopyFromSetupFailuresLeaveUntrainedEmptyStateAndAllowRetry)
{
    for (bool withVectors : {false, true})
    {
        for (int phase : {0, 1, 2})
        {
            SCOPED_TRACE(withVectors);
            SCOPED_TRACE(phase);
            auto source = MakePretrainedCpuIndex(withVectors);
            auto old = MakePretrainedCpuIndex();
            faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
            ASSERT_NO_THROW(index.copyFrom(old.index.get()));
            ASSERT_TRUE(index.is_trained);
            {
                ImportFaultMockGuard mock;
                if (phase == 0)
                {
                    MOCKER_CPP(&faiss::ascend::AscendIndexIVFRaBitQImpl::uploadorthogonalMatrix)
                        .stubs()
                        .will(invoke(StubOrthogonalSetupFailure));
                }
                else if (phase == 1)
                {
                    MOCKER_CPP(&::ascend::IndexIVFRaBitQ::updateCoarseCenterImpl)
                        .stubs()
                        .will(returnValue(::ascend::APP_ERR_INVALID_PARAM));
                }
                else
                {
                    MOCKER_CPP(&::ascend::IndexIVFFlat::addVectorsAsCentroid)
                        .stubs()
                        .will(returnValue(::ascend::APP_ERR_INVALID_PARAM));
                }
                ExpectExceptionContains([&] { index.copyFrom(source.index.get()); },
                                        phase == 0   ? "injected orthogonal setup failure"
                                        : phase == 1 ? "Failed to update coarsecenter"
                                                     : "Failed to add assignment centroids, ret=");
                EXPECT_EQ(encodedImportCalls, 0);
                EXPECT_EQ(index.ntotal, 0);
                EXPECT_FALSE(index.is_trained);
                EXPECT_EQ(CountMappedIds(*index.impl_->locationTable), 0U);
                EXPECT_EQ(CountDeviceMappedIds(*index.impl_->locationTable, 0), 0U);
                EXPECT_EQ(index.impl_->getActualIndex(0)->getListLength(old.listId), 0U);
            }
            ASSERT_NO_THROW(index.copyFrom(source.index.get()));
            EXPECT_TRUE(index.is_trained);
            EXPECT_EQ(index.ntotal, source.index->ntotal);
            EXPECT_EQ(index.impl_->getActualIndex(0)->getListLength(source.listId), source.ids.size());
        }
    }
}

TEST(TestAscendIndexIVFRaBitQ, PartialEncodedImportKeepsCompletedSetupTrained)
{
    auto source = MakePretrainedCpuIndex();
    const faiss::idx_t extra = 404;
    source.index->invlists->add_entries(source.listId + 1, 1, &extra, source.codes.data());
    ++source.index->ntotal;
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig({0, 1}));
    encodedBatchCalls.assign(2, 0);
    encodedBatchBeforeVersion.assign(2, 0);
    encodedBatchBeforeSnapshot.assign(2, nullptr);
    failEncodedBatchDevice = 0;  // Reject one device batch; the other device still commits.
    MOCKER_CPP(&faiss::ascend::AscendIndexIVFRaBitQImpl::indexIVFRaBitQAddEncodedBatch)
        .stubs()
        .will(invoke(StubEncodedHostBatch));
    ExpectExceptionContains([&] { index.copyFrom(source.index.get()); }, "injected encoded batch failure");
    EXPECT_THAT(encodedBatchCalls, ElementsAre(1U, 1U));
    EXPECT_TRUE(index.is_trained);
    EXPECT_EQ(index.ntotal, 1);
    EXPECT_EQ(CountMappedIds(*index.impl_->locationTable), 1U);
    EXPECT_TRUE(index.impl_->locationTable->contains(source.ids[1]));
    EXPECT_FALSE(index.impl_->locationTable->contains(extra));
    EXPECT_EQ(index.impl_->deviceAddNumMap[source.listId], (std::vector<int>{0, 1}));
    EXPECT_EQ(index.impl_->deviceAddNumMap[source.listId + 1], (std::vector<int>{0, 0}));
    EXPECT_FALSE(index.impl_->refineSourcesReady_);
    GlobalMockObject::verify();
    failEncodedBatchDevice = -1;
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    EXPECT_TRUE(index.is_trained);
    EXPECT_EQ(index.ntotal, source.index->ntotal);
    EXPECT_EQ(index.impl_->getActualIndex(0)->getListLength(source.listId), 2U);
    EXPECT_EQ(index.impl_->getActualIndex(1)->getListLength(source.listId), 1U);
    EXPECT_EQ(index.impl_->getActualIndex(0)->getListLength(source.listId + 1), 1U);
}

// Candidates are prepared before installing the mock: the unique_ptr callback never recurses into
// makeEmptySnapshot, and only the selected daemon reports an allocation failure.
using EmptySnapshot = std::unique_ptr<::ascend::IndexIVFRaBitQ::GlobalSnapshot>;
std::vector<std::pair<const ::ascend::IndexIVFRaBitQ*, EmptySnapshot>> resetEmptyCandidates;
const ::ascend::IndexIVFRaBitQ* resetEmptyFaultIndex = nullptr;
bool failResetEmpty = false;
int resetEmptyCalls = 0;

EmptySnapshot StubResetEmptySnapshot(const ::ascend::IndexIVFRaBitQ* index)
{
    ++resetEmptyCalls;
    if (index == resetEmptyFaultIndex && failResetEmpty)
    {
        throw std::bad_alloc();
    }
    for (auto& candidate : resetEmptyCandidates)
    {
        if (candidate.first == index && candidate.second != nullptr)
        {
            return std::move(candidate.second);
        }
    }
    throw std::runtime_error("missing prepared reset snapshot");
}

struct ResetEmptyMockGuard
{
    ResetEmptyMockGuard(const std::vector<::ascend::IndexIVFRaBitQ*>& indexes,
                        const ::ascend::IndexIVFRaBitQ* faultIndex)
    {
        resetEmptyCandidates.clear();
        for (const auto* index : indexes)
        {
            resetEmptyCandidates.emplace_back(index, index->makeEmptySnapshot());
        }
        resetEmptyFaultIndex = faultIndex;
        failResetEmpty = true;
        resetEmptyCalls = 0;
        MOCKER_CPP(&::ascend::IndexIVFRaBitQ::makeEmptySnapshot).stubs().will(invoke(StubResetEmptySnapshot));
    }
    ~ResetEmptyMockGuard()
    {
        GlobalMockObject::verify();
        resetEmptyCandidates.clear();
        resetEmptyFaultIndex = nullptr;
        failResetEmpty = false;
    }
};

TEST(TestIndexIVFRaBitQCore, ResetPreparationBadAllocPreservesStateAndRetryUsesOneCandidate)
{
    faiss::ascend::AscendIndexIVFRaBitQ hostIndex(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    auto* actual = hostIndex.impl_->getActualIndex(0);
    SeedList(actual, 0, 2);
    const auto* pinned = actual->activeSnapshot.get();
    actual->readerHazard.store(pinned);
    SeedList(actual, 1, 1);  // Keep a retired snapshot pinned as well as live arena segments.
    actual->readerDrainError.store(ACL_ERROR_FAILURE);
    actual->maxListLength = 2;
    uint8_t filterSource = 7;
    actual->cachedFilterPayload =
        std::make_unique<::ascend::DeviceVector<uint8_t>>(::ascend::MemorySpace::DEVICE_HUGEPAGE);
    actual->cachedFilterPayload->resize(8);
    actual->cachedFilterSrc = &filterSource;
    actual->cachedFilterBytes = 8;
    actual->cachedFilterGeneration = 17;
    actual->cachedPrefixFilterSrc = &filterSource;
    actual->cachedPrefixFilterBytes = 1;
    actual->cachedPrefixFilterGeneration = 19;
    const auto* active = actual->activeSnapshot.get();
    const auto* codes = active->lists[0]->codes.segmentData(0);
    const auto* arena = actual->listArena_.get();
    const auto slabs = arena->SlabCount();
    const auto bytes = arena->TotalReservedBytes();
    const auto retiredCount = actual->retiredSnapshots.size();
    const auto retiredBytes = actual->retiredBytes;
    const auto* filterData = actual->cachedFilterPayload->data();
    const auto cacheEpoch = actual->filterCacheEpoch.load();
    const auto* distData = actual->distResultOnDevice->data();
    const auto distSize = actual->distResultOnDevice->size();
    const auto distSpace = actual->distResultSpace;
    ASSERT_GT(retiredCount, 0U);
    ASSERT_GT(bytes, 0U);
    ResetEmptyMockGuard mock({actual}, actual);
    EXPECT_THROW(actual->reset(), std::bad_alloc);
    EXPECT_EQ(resetEmptyCalls, 1);
    EXPECT_EQ(actual->activeSnapshot.get(), active);
    EXPECT_EQ(actual->currentSnapshot.load(), active);
    EXPECT_EQ(active->total, 3U);
    EXPECT_EQ(actual->ntotal, 3);
    EXPECT_EQ(actual->maxListLength, 2);
    EXPECT_EQ(active->lists[0]->codes.segmentData(0), codes);
    EXPECT_EQ(actual->readerHazard.load(), pinned);
    EXPECT_EQ(actual->readerDrainError.load(), ACL_ERROR_FAILURE);
    EXPECT_EQ(actual->retiredSnapshots.size(), retiredCount);
    EXPECT_EQ(actual->retiredSnapshots[0].snapshot.get(), pinned);
    EXPECT_EQ(actual->retiredBytes, retiredBytes);
    EXPECT_EQ(actual->listArena_.get(), arena);
    EXPECT_EQ(arena->SlabCount(), slabs);
    EXPECT_EQ(arena->TotalReservedBytes(), bytes);
    EXPECT_EQ(actual->cachedFilterPayload->data(), filterData);
    EXPECT_EQ(actual->cachedFilterPayload->size(), 8U);
    EXPECT_EQ(actual->cachedFilterSrc, &filterSource);
    EXPECT_EQ(actual->cachedFilterBytes, 8U);
    EXPECT_EQ(actual->cachedFilterGeneration, 17U);
    EXPECT_EQ(actual->cachedPrefixFilterSrc, &filterSource);
    EXPECT_EQ(actual->cachedPrefixFilterBytes, 1U);
    EXPECT_EQ(actual->cachedPrefixFilterGeneration, 19U);
    EXPECT_EQ(actual->filterCacheEpoch.load(), cacheEpoch);
    EXPECT_EQ(actual->distResultOnDevice->data(), distData);
    EXPECT_EQ(actual->distResultOnDevice->size(), distSize);
    EXPECT_EQ(actual->distResultSpace, distSpace);
    failResetEmpty = false;
    ASSERT_EQ(actual->reset(), ::ascend::APP_ERR_OK);
    EXPECT_EQ(resetEmptyCalls, 2);  // Exactly one preparation per attempt; no post-destruction allocation.
    EXPECT_EQ(actual->currentSnapshot.load(), actual->activeSnapshot.get());
    EXPECT_EQ(actual->activeSnapshot->publishVer, 0U);
    EXPECT_EQ(actual->activeSnapshot->total, 0U);
    EXPECT_EQ(actual->ntotal, 0);
    EXPECT_EQ(actual->maxListLength, 0);
    EXPECT_EQ(actual->readerHazard.load(), nullptr);
    EXPECT_EQ(actual->readerDrainError.load(), ACL_SUCCESS);
    EXPECT_TRUE(actual->retiredSnapshots.empty());
    EXPECT_EQ(actual->retiredBytes, 0U);
    EXPECT_EQ(actual->listArena_.get(), arena);
    EXPECT_EQ(arena->SlabCount(), 0U);
    EXPECT_EQ(arena->TotalReservedBytes(), 0U);
    EXPECT_EQ(actual->distResultSpace, 0U);
    EXPECT_EQ(actual->cachedFilterSrc, nullptr);
    EXPECT_EQ(actual->cachedFilterBytes, 0U);
    EXPECT_EQ(actual->cachedPrefixFilterSrc, nullptr);
    EXPECT_EQ(actual->cachedPrefixFilterBytes, 0U);
}

TEST(TestAscendIndexIVFRaBitQ, LaterDeviceResetPreparationBadAllocReconcilesSuccessfulDevice)
{
    auto source = MakePretrainedCpuIndex();
    const faiss::idx_t ids[] = {0, 1, 2};
    source.index->invlists->update_entries(source.listId, 0, 3, ids, source.codes.data());
    auto replacement = MakePretrainedCpuIndex(false);
    replacement.index->nprobe = NPROBE + 1;
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig({0, 1}));
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    const auto originals = MakePatternData(3, DIM);
    ASSERT_NO_THROW(index.attachRefineSources(originals.data(), 3));
    auto* first = index.impl_->getActualIndex(0);
    auto* second = index.impl_->getActualIndex(1);
    const auto* oldSecond = second->activeSnapshot.get();
    const auto oldMappings = SnapshotLocations(*index.impl_->locationTable);
    const auto oldCounts = index.impl_->deviceAddNumMap;
    const auto* oldArena = second->listArena_.get();
    const auto oldArenaBytes = oldArena->TotalReservedBytes();
    ASSERT_EQ(oldMappings.size(), 3U);
    const auto oldSecondMapping = oldMappings[1];
    ASSERT_EQ(oldSecondMapping.first, 1);
    ASSERT_EQ(oldCounts[source.listId][0], 2);
    ASSERT_EQ(oldCounts[source.listId][1], 1);
    {
        ResetEmptyMockGuard mock({first, second}, second);
        EXPECT_THROW(index.copyFrom(replacement.index.get()), std::bad_alloc);
        EXPECT_EQ(resetEmptyCalls, 2);
        EXPECT_EQ(first->activeSnapshot->total, 0U);
        EXPECT_EQ(second->activeSnapshot.get(), oldSecond);
        EXPECT_EQ(second->currentSnapshot.load(), oldSecond);
        EXPECT_EQ(oldSecond->total, 1U);
        EXPECT_EQ(second->ntotal, 1);
        EXPECT_EQ(second->listArena_.get(), oldArena);
        EXPECT_EQ(oldArena->TotalReservedBytes(), oldArenaBytes);
        EXPECT_EQ(index.ntotal, 1);
        EXPECT_TRUE(index.is_trained);
        EXPECT_EQ(index.getNumProbes(), NPROBE);
        EXPECT_THAT(SnapshotLocations(*index.impl_->locationTable), ElementsAre(oldSecondMapping));
        EXPECT_FALSE(index.impl_->locationTable->contains(0));
        EXPECT_FALSE(index.impl_->locationTable->contains(2));
        EXPECT_EQ(CountDeviceMappedIds(*index.impl_->locationTable, 0), 0U);
        EXPECT_EQ(CountDeviceMappedIds(*index.impl_->locationTable, 1), 1U);
        EXPECT_EQ(index.impl_->deviceAddNumMap[source.listId][0], 0);
        EXPECT_EQ(index.impl_->deviceAddNumMap[source.listId][1], oldCounts[source.listId][1]);
        EXPECT_FALSE(index.impl_->refineSourcesReady_);
        ExpectExceptionContains([&] { index.impl_->getRefineSources(); }, "not ready");
    }
    ASSERT_NO_THROW(index.copyFrom(replacement.index.get()));
    EXPECT_EQ(index.ntotal, 0);
    EXPECT_EQ(index.getNumProbes(), NPROBE + 1);
    EXPECT_TRUE(SnapshotLocations(*index.impl_->locationTable).empty());
    EXPECT_FALSE(index.impl_->refineSourcesAttached());
    EXPECT_TRUE(index.impl_->refineSourcesReady_);
}

// Reader-drain faults are injected after index construction, so allocation/operator setup stays real.
aclrtStream drainFaultCompute = nullptr;
aclrtStream drainFaultAicpu = nullptr;
int computeDrainCalls = 0;
int aicpuDrainCalls = 0;
bool failComputeDrain = false;
bool failAicpuDrain = false;
int transientComputeFailures = 0;
aclrtStream resetFaultWriter = nullptr;

aclError StubReaderStreamDrain(aclrtStream stream)
{
    if (stream == resetFaultWriter && resetFaultWriter != nullptr)
    {
        return ACL_ERROR_FAILURE;
    }
    if (stream == drainFaultCompute)
    {
        ++computeDrainCalls;
        if (transientComputeFailures > 0)
        {
            --transientComputeFailures;
            return ACL_ERROR_FAILURE;
        }
        return failComputeDrain ? ACL_ERROR_FAILURE : ACL_SUCCESS;
    }
    if (stream == drainFaultAicpu)
    {
        ++aicpuDrainCalls;
        return failAicpuDrain ? ACL_ERROR_FAILURE : ACL_SUCCESS;
    }
    return ACL_SUCCESS;
}

struct ReaderDrainMockGuard
{
    explicit ReaderDrainMockGuard(const ::ascend::IndexIVFRaBitQ& index)
    {
        drainFaultCompute = index.readerContext->computeStream;
        drainFaultAicpu = index.readerContext->aicpuStream;
        computeDrainCalls = aicpuDrainCalls = transientComputeFailures = 0;
        failComputeDrain = failAicpuDrain = false;
        resetFaultWriter = nullptr;
        MOCKER(::ascend::synchronizeStream).stubs().will(invoke(StubReaderStreamDrain));
    }
    ~ReaderDrainMockGuard()
    {
        resetFaultWriter = nullptr;
        GlobalMockObject::verify();
    }
};

TEST(TestIndexIVFRaBitQCore, FailedPreciseDeleteDoesNotExposeCandidateMoves)
{
    auto source = MakePretrainedCpuIndex();
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    auto* actual = index.impl_->getActualIndex(0);
    const auto* snapshot = actual->activeSnapshot.get();
    ReaderDrainMockGuard mock(*actual);
    resetFaultWriter = actual->writerContext->computeStream;
    const int list = source.listId;
    const size_t slot = 1;
    const ::ascend::Index::idx_t id = static_cast<::ascend::Index::idx_t>(source.ids[1]);
    std::vector<::ascend::IndexIVFRaBitQ::MovedSlot> moves;
    EXPECT_EQ(actual->removeIds(1, &id, &list, &slot, &moves), ::ascend::APP_ERR_INNER_ERROR);
    EXPECT_TRUE(moves.empty());
    EXPECT_EQ(actual->activeSnapshot.get(), snapshot);
    EXPECT_EQ(index.ntotal, 3);
    const auto loc = index.impl_->locationTable->lookupLocation(source.ids[2]);
    ASSERT_TRUE(loc.has_value());
    EXPECT_EQ(loc->slot(), 2U);
    resetFaultWriter = nullptr;
    ASSERT_EQ(actual->removeIds(1, &id, &list, &slot, &moves), ::ascend::APP_ERR_OK);
    ASSERT_EQ(moves.size(), 1U);
    EXPECT_EQ(moves[0].id, static_cast<::ascend::Index::idx_t>(source.ids[2]));
    EXPECT_EQ(moves[0].newSlot, 1U);
}

TEST(TestIndexIVFRaBitQCore, MoveValidationThrowRejectsSnapshotAndLeavesOutputUntouched)
{
    auto source = MakePretrainedCpuIndex();
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    auto* actual = index.impl_->getActualIndex(0);
    const auto* snapshot = actual->activeSnapshot.get();
    const int list = source.listId;
    const size_t slot = 1;
    const ::ascend::Index::idx_t id = static_cast<::ascend::Index::idx_t>(source.ids[1]);
    std::vector<::ascend::IndexIVFRaBitQ::MovedSlot> moves = {{999, 0, 123}};
    bool invoked = false;
    ::ascend::IndexIVFRaBitQ::MoveValidator reject = [&](const auto& prepared)
    {
        invoked = true;
        ASSERT_EQ(prepared.size(), 1U);
        EXPECT_EQ(prepared[0].id, static_cast<::ascend::Index::idx_t>(source.ids[2]));
        EXPECT_EQ(prepared[0].newSlot, 1U);
        EXPECT_EQ(actual->activeSnapshot.get(), snapshot);
        throw std::runtime_error("injected move validation failure");
    };
    ExpectExceptionContains([&] { actual->removeIds(1, &id, &list, &slot, &moves, reject); },
                            "injected move validation failure");
    EXPECT_TRUE(invoked);
    EXPECT_EQ(actual->activeSnapshot.get(), snapshot);
    ASSERT_EQ(moves.size(), 1U);
    EXPECT_EQ(moves[0].id, 999U);
    EXPECT_EQ(moves[0].newSlot, 123U);
    EXPECT_EQ(actual->getListLength(list), 3U);

    // A validator still receives moves when a legacy caller does not request movedOut.
    invoked = false;
    ::ascend::IndexIVFRaBitQ::MoveValidator accept = [&](const auto& prepared)
    {
        invoked = true;
        EXPECT_EQ(prepared.size(), 1U);
    };
    ASSERT_EQ(actual->removeIds(1, &id, &list, &slot, nullptr, accept), ::ascend::APP_ERR_OK);
    EXPECT_TRUE(invoked);
    EXPECT_EQ(actual->getListLength(list), 2U);
}

TEST(TestAscendIndexIVFRaBitQ, UnknownMovedIdRejectsBeforeHostAndDaemonPublication)
{
    auto source = MakePretrainedCpuIndex();
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    auto* actual = index.impl_->getActualIndex(0);
    const auto* snapshot = actual->activeSnapshot.get();
    index.impl_->locationTable->clearValid(source.ids[2]);  // Simulate an unknown surviving daemon tail ID.
    const auto locations = SnapshotLocations(*index.impl_->locationTable);
    const auto counts = index.impl_->deviceAddNumMap;
    ExpectExceptionContains([&] { index.remove_ids(1, &source.ids[1]); }, "moved ID");
    EXPECT_EQ(actual->activeSnapshot.get(), snapshot);
    EXPECT_EQ(actual->getListLength(source.listId), 3U);
    EXPECT_EQ(index.ntotal, 3);
    EXPECT_EQ(index.impl_->deviceAddNumMap, counts);
    EXPECT_EQ(SnapshotLocations(*index.impl_->locationTable), locations);
    // Restore the faulted metadata and retry; rejected compaction did not mutate source segments.
    index.impl_->locationTable->setLocation(source.ids[2], 0, source.listId, 2);
    ASSERT_NO_THROW(index.remove_ids(1, &source.ids[1]));
    EXPECT_EQ(index.ntotal, 2);
    const auto loc = index.impl_->locationTable->lookupLocation(source.ids[2]);
    ASSERT_TRUE(loc.has_value());
    EXPECT_EQ(loc->slot(), 1U);
}

TEST(TestAscendIndexIVFRaBitQ, FailedCopyFromReaderDrainPreservesOldMetadataAndSources)
{
    auto source = MakePretrainedCpuIndex();
    const faiss::idx_t ids[] = {0, 1, 2};
    source.index->invlists->update_entries(source.listId, 0, 3, ids, source.codes.data());
    auto replacement = MakePretrainedCpuIndex(false);
    replacement.index->nprobe = NPROBE + 1;
    auto config = MakeConfig({0, 1});
    config.needRefine = true;
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, config);
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    const auto originals = MakePatternData(3, DIM);
    ASSERT_NO_THROW(index.loadRefineSources(originals.data(), 3));
    const auto mapping = SnapshotLocations(*index.impl_->locationTable);
    const auto counts = index.impl_->deviceAddNumMap;
    const auto generation = index.impl_->cachedFilterGeneration;
    const auto* first = index.impl_->getActualIndex(0)->activeSnapshot.get();
    auto* second = index.impl_->getActualIndex(1);
    const auto* secondSnapshot = second->activeSnapshot.get();
    ReaderDrainMockGuard mock(*second);
    failComputeDrain = true;
    EXPECT_THROW(index.copyFrom(replacement.index.get()), std::exception);
    EXPECT_EQ(index.ntotal, 3);
    EXPECT_EQ(index.getNumProbes(), NPROBE);
    EXPECT_TRUE(index.is_trained);
    EXPECT_EQ(SnapshotLocations(*index.impl_->locationTable), mapping);
    EXPECT_EQ(index.impl_->deviceAddNumMap, counts);
    EXPECT_EQ(index.impl_->srcIndexes, originals);
    EXPECT_TRUE(index.impl_->copiedSources_);
    EXPECT_TRUE(index.impl_->refineSourcesReady_);
    EXPECT_EQ(index.impl_->cachedFilterGeneration, generation);
    EXPECT_EQ(index.impl_->getActualIndex(0)->activeSnapshot.get(), first);
    EXPECT_EQ(second->activeSnapshot.get(), secondSnapshot);
    failComputeDrain = false;
    ASSERT_NO_THROW(index.copyFrom(replacement.index.get()));
    EXPECT_EQ(index.ntotal, 0);
    EXPECT_EQ(index.getNumProbes(), NPROBE + 1);
    EXPECT_TRUE(CountMappedIds(*index.impl_->locationTable) == 0);
    EXPECT_TRUE(index.impl_->srcIndexes.empty());
}

TEST(TestAscendIndexIVFRaBitQ, LateCopyFromResetFailureReconcilesSuccessfulDevices)
{
    auto source = MakePretrainedCpuIndex();
    const faiss::idx_t ids[] = {0, 1, 2};
    source.index->invlists->update_entries(source.listId, 0, 3, ids, source.codes.data());
    auto replacement = MakePretrainedCpuIndex(false);
    replacement.index->nprobe = NPROBE + 1;
    faiss::ascend::AscendIndexIVFRaBitQ index(DIM, faiss::METRIC_L2, NLIST, MakeConfig({0, 1}));
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    const auto originals = MakePatternData(3, DIM);
    ASSERT_NO_THROW(index.attachRefineSources(originals.data(), 3));
    auto* first = index.impl_->getActualIndex(0);
    auto* second = index.impl_->getActualIndex(1);
    const auto* secondSnapshot = second->activeSnapshot.get();
    ReaderDrainMockGuard mock(*second);
    resetFaultWriter = second->writerContext->computeStream;  // Both reader preflights still succeed.
    EXPECT_THROW(index.copyFrom(replacement.index.get()), std::exception);
    EXPECT_EQ(first->activeSnapshot->total, 0U);
    EXPECT_EQ(second->activeSnapshot.get(), secondSnapshot);
    EXPECT_EQ(second->activeSnapshot->total, 1U);
    EXPECT_EQ(index.ntotal, 1);
    EXPECT_TRUE(index.is_trained);
    EXPECT_EQ(index.getNumProbes(), NPROBE);  // Replacement metadata was never installed.
    EXPECT_EQ(CountMappedIds(*index.impl_->locationTable), 1U);
    EXPECT_EQ(static_cast<size_t>(index.impl_->locationTable->contains(1)), 1U);
    EXPECT_TRUE(CountDeviceMappedIds(*index.impl_->locationTable, 0) == 0);
    EXPECT_EQ(CountDeviceMappedIds(*index.impl_->locationTable, 1), 1U);
    EXPECT_EQ(index.impl_->deviceAddNumMap[source.listId][0], 0);
    EXPECT_EQ(index.impl_->deviceAddNumMap[source.listId][1], 1);
    EXPECT_FALSE(index.impl_->refineSourcesReady_);
    ExpectExceptionContains([&] { index.impl_->getRefineSources(); }, "not ready");
    resetFaultWriter = nullptr;
    ASSERT_NO_THROW(index.copyFrom(replacement.index.get()));
    EXPECT_EQ(index.ntotal, 0);
    EXPECT_FALSE(index.impl_->refineSourcesAttached());
    EXPECT_TRUE(index.impl_->refineSourcesReady_);
}

TEST(TestIndexIVFRaBitQCore, ReaderDrainRetriesOnceBeforeReleasingHazard)
{
    faiss::ascend::AscendIndexIVFRaBitQ hostIndex(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    auto actual = hostIndex.impl_->getActualIndex(0);
    ReaderDrainMockGuard mock(*actual);
    transientComputeFailures = 1;
    {
        ::ascend::IndexIVFRaBitQ::SnapshotGuard guard(*actual);
        EXPECT_EQ(guard.finish(), ACL_SUCCESS);
        EXPECT_EQ(actual->readerHazard.load(), nullptr);
        EXPECT_EQ(actual->readerDrainError.load(), ACL_SUCCESS);
        EXPECT_EQ(computeDrainCalls, 2);
        EXPECT_EQ(aicpuDrainCalls, 2);
    }
    EXPECT_EQ(computeDrainCalls, 2);  // explicit finish is not repeated by the destructor
}

TEST(TestIndexIVFRaBitQCore, FailedDrainCannotLoseProtectionToTheNextReader)
{
    faiss::ascend::AscendIndexIVFRaBitQ hostIndex(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    auto actual = hostIndex.impl_->getActualIndex(0);
    ReaderDrainMockGuard mock(*actual);
    for (bool failCompute : {true, false})
    {
        failComputeDrain = failCompute;
        failAicpuDrain = !failCompute;
        const auto* pinned = actual->activeSnapshot.get();
        {
            ::ascend::IndexIVFRaBitQ::SnapshotGuard guard(*actual);
            EXPECT_EQ(guard.finish(), ACL_ERROR_FAILURE);
        }
        // Models a writer already past its entry check when the reader reported its drain failure.
        {
            std::lock_guard<std::mutex> writerLock(actual->writerLock);
            actual->publishSnapshot(actual->makeEmptySnapshot());
            EXPECT_EQ(actual->checkRetiredMemory(), ::ascend::APP_ERR_INNER_ERROR);
        }
        EXPECT_EQ(actual->readerHazard.load(), pinned);
        EXPECT_EQ(actual->readerDrainError.load(), ACL_ERROR_FAILURE);
        EXPECT_THROW((::ascend::IndexIVFRaBitQ::SnapshotGuard(*actual)), ::ascend::AscendException);
        EXPECT_EQ(actual->readerHazard.load(), pinned);  // failed recovery must not overwrite it
        failComputeDrain = failAicpuDrain = false;
        EXPECT_EQ(actual->getListLength(0), 0U);  // retries recovery before taking a new hazard
        EXPECT_EQ(actual->readerHazard.load(), nullptr);
        EXPECT_EQ(actual->readerDrainError.load(), ACL_SUCCESS);
        {
            std::lock_guard<std::mutex> writerLock(actual->writerLock);
            actual->reclaimRetiredSnapshots();
        }
        EXPECT_TRUE(actual->retiredSnapshots.empty());
    }
}

TEST(TestIndexIVFRaBitQCore, ResetClearsFailedHazardOnlyAfterSuccessfulDrain)
{
    faiss::ascend::AscendIndexIVFRaBitQ hostIndex(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    auto actual = hostIndex.impl_->getActualIndex(0);
    ReaderDrainMockGuard mock(*actual);
    failComputeDrain = true;
    const auto* pinned = actual->activeSnapshot.get();
    {
        ::ascend::IndexIVFRaBitQ::SnapshotGuard guard(*actual);
        EXPECT_EQ(guard.finish(), ACL_ERROR_FAILURE);
    }
    EXPECT_EQ(actual->reset(), ::ascend::APP_ERR_INNER_ERROR);
    EXPECT_EQ(actual->activeSnapshot.get(), pinned);
    EXPECT_EQ(actual->readerHazard.load(), pinned);
    failComputeDrain = false;
    EXPECT_EQ(actual->reset(), ::ascend::APP_ERR_OK);
    EXPECT_EQ(actual->readerHazard.load(), nullptr);
    EXPECT_EQ(actual->readerDrainError.load(), ACL_SUCCESS);
}

::ascend::APP_ERROR StubSuccessfulPinnedSearch(::ascend::IndexIVFRaBitQ*, int, const float*, int, float*,
                                               ::ascend::idx_t*, const float*, const ::ascend::RabitqIdFilterHost*, int)
{
    return ::ascend::APP_ERR_OK;
}

TEST(TestIndexIVFRaBitQCore, SearchPropagatesDrainFailureInsteadOfReturningSuccess)
{
    faiss::ascend::AscendIndexIVFRaBitQ hostIndex(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    auto actual = hostIndex.impl_->getActualIndex(0);
    ReaderDrainMockGuard mock(*actual);
    MOCKER_CPP(&::ascend::IndexIVFRaBitQ::searchWithBatch).stubs().will(invoke(StubSuccessfulPinnedSearch));
    failComputeDrain = true;
    auto query = MakePatternData(1, DIM);
    float distance = 0;
    ::ascend::idx_t label = 0;
    EXPECT_EQ(actual->searchImpl(1, query.data(), 1, &distance, &label, nullptr), ::ascend::APP_ERR_INNER_ERROR);
    EXPECT_EQ(actual->readerHazard.load(), actual->activeSnapshot.get());
    EXPECT_EQ(::ascend::IndexIVFRaBitQ::searchSnapshot, nullptr);
    failComputeDrain = false;
    EXPECT_EQ(actual->getListLength(0), 0U);
}

TEST(TestIndexIVFRaBitQCore, ReclamationRecountsSegmentsThatBecomeExclusiveToPinnedSnapshot)
{
    faiss::ascend::AscendIndexIVFRaBitQ hostIndex(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    auto actual = hostIndex.impl_->getActualIndex(0);
    SeedList(actual, 0, 2);
    const auto* pinned = actual->activeSnapshot.get();
    ::ascend::IndexIVFRaBitQ::SnapshotGuard guard(*actual);
    {
        std::lock_guard<std::mutex> writerLock(actual->writerLock);
        auto first = actual->cloneSnapshot(*actual->activeSnapshot);
        first->lists.set(1,
                         std::make_shared<const ::ascend::IndexIVFRaBitQ::ListStorage>(
                             actual->blockSize, actual->blockSize / 8, actual->getCodePartBytes(), actual->listArena_));
        actual->publishSnapshot(std::move(first));
        EXPECT_EQ(actual->retiredBytes, 0U);  // list 0 is still shared with active
        auto second = actual->cloneSnapshot(*actual->activeSnapshot);
        second->lists.set(
            0, std::make_shared<const ::ascend::IndexIVFRaBitQ::ListStorage>(
                   actual->blockSize, actual->blockSize / 8, actual->getCodePartBytes(), actual->listArena_));
        second->total = 0;
        actual->publishSnapshot(std::move(second));
        const auto& oldList = *pinned->lists[0];
        const size_t expected = oldList.codes.allocatedBytes() + oldList.indexL1.allocatedBytes() +
                                oldList.indexL2.allocatedBytes() + oldList.ids.allocatedBytes();
        EXPECT_GT(expected, 0U);
        EXPECT_EQ(actual->retiredBytes, expected);
        actual->retiredMemoryHighWatermark = expected - 1;
        EXPECT_EQ(actual->checkRetiredMemory(), ::ascend::APP_ERR_ACL_BAD_ALLOC);
    }
    EXPECT_EQ(guard.finish(), ACL_SUCCESS);
    {
        std::lock_guard<std::mutex> writerLock(actual->writerLock);
        actual->reclaimRetiredSnapshots();
        EXPECT_EQ(actual->retiredBytes, 0U);
    }
}

TEST(TestIndexIVFRaBitQCore, ReclamationRecountsLaterWritesInAnotherChunk)
{
    using Core = ::ascend::IndexIVFRaBitQ;
    constexpr size_t otherListId = Core::kListChunkSize;
    faiss::ascend::AscendIndexIVFRaBitQ hostIndex(DIM, faiss::METRIC_L2, 2 * otherListId, MakeConfig());
    auto actual = hostIndex.impl_->getActualIndex(0);
    SeedList(actual, 0, 2);
    SeedList(actual, otherListId, 2);
    SeedList(actual, otherListId + 1, 2);
    const auto* pinned = actual->activeSnapshot.get();
    const auto allocatedBytes = [](const Core::ListStorage& list)
    {
        return list.codes.allocatedBytes() + list.indexL1.allocatedBytes() + list.indexL2.allocatedBytes() +
               list.ids.allocatedBytes();
    };
    const auto uniqueBytes = [](const Core::ListStorage& list)
    {
        return list.codes.uniquelyOwnedBytes() + list.indexL1.uniquelyOwnedBytes() + list.indexL2.uniquelyOwnedBytes() +
               list.ids.uniquelyOwnedBytes();
    };
    const size_t firstBytes = allocatedBytes(*pinned->lists[0]);
    const size_t laterBytes = allocatedBytes(*pinned->lists[otherListId]);
    const size_t sharedBytes = allocatedBytes(*pinned->lists[otherListId + 1]);
    ASSERT_GT(firstBytes, 0U);
    ASSERT_GT(laterBytes, 0U);
    ASSERT_GT(sharedBytes, 0U);
    Core::SnapshotGuard guard(*actual);
    {
        std::lock_guard<std::mutex> writerLock(actual->writerLock);
        auto first = actual->cloneSnapshot(*actual->activeSnapshot);
        first->lists.set(0, std::make_shared<const Core::ListStorage>(actual->blockSize, actual->blockSize / 8,
                                                                      actual->getCodePartBytes(), actual->listArena_));
        first->total -= 2;
        actual->publishSnapshot(std::move(first));
        ASSERT_EQ(actual->retiredSnapshots.size(), 1U);
        EXPECT_EQ(actual->retiredSnapshots.front().snapshot.get(), pinned);
        EXPECT_EQ(actual->retiredSnapshots.front().bytes, firstBytes);
        EXPECT_EQ(actual->retiredBytes, firstBytes);
        // Active reaches the same slots through the shared second chunk without bumping their refcounts.
        EXPECT_EQ(pinned->lists[otherListId].use_count(), 1);
        EXPECT_EQ(pinned->lists[otherListId + 1].use_count(), 1);

        auto second = actual->cloneSnapshot(*actual->activeSnapshot);
        second->lists.set(otherListId,
                          std::make_shared<const Core::ListStorage>(actual->blockSize, actual->blockSize / 8,
                                                                    actual->getCodePartBytes(), actual->listArena_));
        second->total -= 2;
        actual->publishSnapshot(std::move(second));
        const size_t expected = firstBytes + laterBytes;
        ASSERT_EQ(actual->retiredSnapshots.size(), 1U);
        EXPECT_EQ(actual->retiredSnapshots.front().snapshot.get(), pinned);
        EXPECT_EQ(actual->retiredSnapshots.front().bytes, expected);
        EXPECT_EQ(actual->retiredBytes, expected);  // laterBytes was shared when pinned first retired
        EXPECT_EQ(pinned->lists[otherListId + 1].get(), actual->activeSnapshot->lists[otherListId + 1].get());
        EXPECT_EQ(pinned->lists[otherListId + 1].use_count(), 2);
        EXPECT_EQ(uniqueBytes(*pinned->lists[otherListId + 1]), sharedBytes);  // list gate excludes these segments

        {
            auto heldList = pinned->lists[0];
            actual->reclaimRetiredSnapshots();
            EXPECT_EQ(actual->retiredBytes, laterBytes);
        }
        actual->reclaimRetiredSnapshots();
        EXPECT_EQ(actual->retiredBytes, expected);
        {
            Core::ListStorage heldSegments = *pinned->lists[otherListId];
            EXPECT_EQ(pinned->lists[otherListId].use_count(), 1);
            EXPECT_EQ(uniqueBytes(*pinned->lists[otherListId]), 0U);
            actual->reclaimRetiredSnapshots();
            EXPECT_EQ(actual->retiredBytes, firstBytes);
        }
        actual->reclaimRetiredSnapshots();
        EXPECT_EQ(actual->retiredBytes, expected);
        actual->retiredMemoryHighWatermark = expected;
        EXPECT_EQ(actual->checkRetiredMemory(), ::ascend::APP_ERR_OK);
        actual->retiredMemoryHighWatermark = expected - 1;
        EXPECT_EQ(actual->checkRetiredMemory(), ::ascend::APP_ERR_ACL_BAD_ALLOC);
    }
    EXPECT_EQ(guard.finish(), ACL_SUCCESS);
    {
        std::lock_guard<std::mutex> writerLock(actual->writerLock);
        actual->reclaimRetiredSnapshots();
        EXPECT_TRUE(actual->retiredSnapshots.empty());
        EXPECT_EQ(actual->retiredBytes, 0U);
    }
}

TEST(TestIndexIVFRaBitQCore, CompactionCopiesAllPayloadAndLeavesOldSnapshotUnchanged)
{
    faiss::ascend::AscendIndexIVFRaBitQ hostIndex(DIM, faiss::METRIC_L2, NLIST, MakeConfig());
    auto actual = hostIndex.impl_->getActualIndex(0);
    constexpr size_t block = 8;
    constexpr size_t growth = 4;
    constexpr size_t count = 11;
    actual->blockSize = static_cast<int>(block);  // fixture geometry for the internal compact helper
    using List = ::ascend::IndexIVFRaBitQ::ListStorage;
    List oldList(block, growth, actual->getCodePartBytes(), actual->listArena_);
    auto stream = actual->writerContext->computeStream;
    auto upload = [&](auto& storage, const auto& values)
    {
        using T = typename std::decay_t<decltype(values)>::value_type;
        storage.ensureCapacity(count, stream);
        storage.commitLength(count);
        for (size_t s = 0; s < storage.segmentCount(); ++s)
        {
            const size_t offset = s * block * storage.elementsPerSlot();
            const size_t bytes = storage.segmentLength(s) * storage.elementsPerSlot() * sizeof(T);
            ASSERT_EQ(aclrtMemcpy(storage.segmentData(s), storage.segmentBytes(s), values.data() + offset, bytes,
                                  ACL_MEMCPY_HOST_TO_DEVICE),
                      ACL_SUCCESS);
        }
    };
    auto readback = [&](const auto& storage, auto& values)
    {
        using T = typename std::decay_t<decltype(values)>::value_type;
        size_t copied = 0;
        for (size_t s = 0; s < storage.segmentCount(); ++s)
        {
            const size_t elements = storage.segmentLength(s) * storage.elementsPerSlot();
            ASSERT_EQ(aclrtMemcpy(values.data() + copied, (values.size() - copied) * sizeof(T), storage.segmentData(s),
                                  elements * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST),
                      ACL_SUCCESS);
            copied += elements;
        }
        EXPECT_EQ(copied, values.size());
    };
    std::vector<uint8_t> codes(count * actual->getCodePartBytes());
    std::vector<float> l1(count), l2(count);
    std::vector<::ascend::idx_t> ids(count);
    for (size_t i = 0; i < codes.size(); ++i) codes[i] = static_cast<uint8_t>(i % 251);
    for (size_t i = 0; i < count; ++i)
    {
        l1[i] = static_cast<float>(i) + 0.25f;
        l2[i] = static_cast<float>(i) + 0.75f;
        ids[i] = 100 + i;
    }
    upload(oldList.codes, codes);
    upload(oldList.indexL1, l1);
    upload(oldList.indexL2, l2);
    upload(oldList.ids, ids);
    oldList.listLen = count;
    for (const auto& deletes : {std::vector<size_t>{1, 5}, std::vector<size_t>{9, 10}})
    {
        SCOPED_TRACE(deletes.front());
        List next = oldList;
        auto slots = deletes;
        EXPECT_EQ(actual->compactList(oldList, next, slots, stream, 0, nullptr), ::ascend::APP_ERR_OK);
        ASSERT_EQ(::ascend::synchronizeStream(stream), ACL_SUCCESS);
        EXPECT_EQ(next.listLen, 9U);
        EXPECT_NE(next.codes.segmentData(1), oldList.codes.segmentData(1));
        std::vector<uint8_t> newCodes(9 * actual->getCodePartBytes());
        std::vector<float> newL1(9), newL2(9);
        std::vector<::ascend::idx_t> newIds(9);
        readback(next.codes, newCodes);
        readback(next.indexL1, newL1);
        readback(next.indexL2, newL2);
        readback(next.ids, newIds);
        for (size_t i = 0; i < 9; ++i)
        {
            const size_t original = deletes.front() == 1 && (i == 1 || i == 5) ? (i == 1 ? 9 : 10) : i;
            EXPECT_EQ(newL1[i], l1[original]);
            EXPECT_EQ(newL2[i], l2[original]);
            EXPECT_EQ(newIds[i], ids[original]);
            for (size_t e = 0; e < actual->getCodePartBytes(); ++e)
                EXPECT_EQ(newCodes[i * actual->getCodePartBytes() + e],
                          codes[original * actual->getCodePartBytes() + e]);
        }
        std::vector<uint8_t> oldCodes(codes.size());
        std::vector<float> oldL1(count), oldL2(count);
        std::vector<::ascend::idx_t> oldIds(count);
        readback(oldList.codes, oldCodes);
        readback(oldList.indexL1, oldL1);
        readback(oldList.indexL2, oldL2);
        readback(oldList.ids, oldIds);
        EXPECT_EQ(oldCodes, codes);
        EXPECT_EQ(oldL1, l1);
        EXPECT_EQ(oldL2, l2);
        EXPECT_EQ(oldIds, ids);
    }
}

}  // namespace
