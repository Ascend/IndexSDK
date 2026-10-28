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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "acl.h"
#include "ascend/AscendIDSelectorRoaring.h"
#include "ascend/AscendIndexIVFRaBitQ.h"
#include "ascend/impl/AscendIndexIVFRaBitQImpl.h"
#include "ascenddaemon/impl/IndexIVFRaBitQ.h"
#include "common/ErrorCode.h"
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

namespace
{
using testing::ElementsAre;
using testing::HasSubstr;

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

struct FilteredSearchMockGuard
{
    FilteredSearchMockGuard()
    {
        MOCKER_CPP(static_cast<FilteredSearchFn>(&::ascend::IndexIVFRaBitQ::searchImpl))
            .stubs()
            .will(invoke(StubFilteredSearch));
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

    const faiss::ascend::ascend_idx_t mappedIds[] = {10, 20, 30};
    impl->updateIdMapping(mappedIds, 1, 3);
    EXPECT_EQ(impl->findDeviceId(20), 1);
    EXPECT_EQ(impl->findDeviceId(5), 1);
    impl->removeIdMapping({20, 999});
    EXPECT_FALSE(impl->idToDeviceMap.count(20));
    EXPECT_FALSE(impl->deviceInfos[1].idSet.count(20));

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
    EXPECT_EQ(deviceIndex.impl_->findDeviceId(source.ids[0]), 0);
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
        EXPECT_EQ(index.impl_->findDeviceId(id), 0);
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
    EXPECT_EQ(index.impl_->idToDeviceMap.count(ids[2]), 0U);
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

}  // namespace
