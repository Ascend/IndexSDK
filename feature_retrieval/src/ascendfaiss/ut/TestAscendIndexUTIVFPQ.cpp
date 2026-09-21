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
#include <faiss/IndexIVFPQ.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "ascend/AscendIndexIVFPQ.h"
#include "ascend/impl/AscendIndexIVFPQImpl.h"
#include "ascenddaemon/impl/IndexIVFPQ.h"
#include "common/utils/SocUtils.h"

namespace
{
using testing::ElementsAre;
using testing::HasSubstr;

constexpr int DIM = 128;
constexpr int NLIST = 1024;
constexpr int MSUB = 4;
constexpr int NBITS = 8;
constexpr int NPROBE = 16;

template <typename Func>
void ExpectExceptionContains(Func &&func, const std::string &expected)
{
    try
    {
        func();
        FAIL() << "Expected exception containing: " << expected;
    }
    catch (const std::exception &e)
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
                static_cast<float>((i * 17 + static_cast<size_t>(j) * 3) % 101) / 101.0f;
        }
    }
    return data;
}

struct CpuIVFPQIndex
{
    std::unique_ptr<faiss::IndexFlat> quantizer;
    std::unique_ptr<faiss::IndexIVFPQ> index;
    std::vector<faiss::idx_t> ids;
    std::vector<uint8_t> codes;
    int listId = 0;
};

CpuIVFPQIndex MakePretrainedCpuIndex(bool withVectors = true)
{
    CpuIVFPQIndex result;
    result.quantizer = std::make_unique<faiss::IndexFlatL2>(DIM);
    result.index = std::make_unique<faiss::IndexIVFPQ>(result.quantizer.get(), DIM, NLIST, MSUB, NBITS);
    result.index->nprobe = NPROBE;

    auto coarseCentroids = MakePatternData(NLIST, DIM);
    result.quantizer->add(NLIST, coarseCentroids.data());

    result.index->pq.centroids.resize(static_cast<size_t>(DIM) * (1U << NBITS));
    for (size_t i = 0; i < result.index->pq.centroids.size(); ++i)
    {
        result.index->pq.centroids[i] = static_cast<float>(i % 257) / 257.0f;
    }
    result.index->is_trained = true;

    if (withVectors)
    {
        result.ids = {11, 22, 33};
        result.codes.resize(result.ids.size() * static_cast<size_t>(MSUB));
        std::iota(result.codes.begin(), result.codes.end(), static_cast<uint8_t>(1));
        result.index->invlists->add_entries(result.listId, result.ids.size(), result.ids.data(), result.codes.data());
        result.index->ntotal = static_cast<faiss::idx_t>(result.ids.size());
    }
    return result;
}

faiss::ascend::AscendIndexIVFPQConfig MakeConfig(std::vector<int> devices = {0})
{
    faiss::ascend::AscendIndexIVFPQConfig config(std::move(devices));
    config.useKmeansPP = false;
    return config;
}

TEST(TestAscendIndexIVFPQ, ConstructorAcceptsSupportedParameters)
{
    faiss::ascend::AscendIndexIVFPQ index(DIM, faiss::METRIC_L2, NLIST, MSUB, NBITS, MakeConfig());

    EXPECT_EQ(index.d, DIM);
    EXPECT_EQ(index.getNumLists(), NLIST);
    EXPECT_EQ(index.getNumProbes(), 64);
    EXPECT_FALSE(index.is_trained);
}

TEST(TestAscendIndexIVFPQ, ConstructorValidatesEveryPublicParameter)
{
    ExpectExceptionContains(
        [] { faiss::ascend::AscendIndexIVFPQ index(64, faiss::METRIC_L2, NLIST, MSUB, NBITS, MakeConfig()); },
        "Unsupported dims");
    ExpectExceptionContains(
        [] { faiss::ascend::AscendIndexIVFPQ index(DIM, faiss::METRIC_L2, 512, MSUB, NBITS, MakeConfig()); },
        "Unsupported nlists");
    ExpectExceptionContains(
        [] { faiss::ascend::AscendIndexIVFPQ index(DIM, faiss::METRIC_L2, NLIST, 3, NBITS, MakeConfig()); },
        "Unsupported msubs");
    ExpectExceptionContains(
        [] { faiss::ascend::AscendIndexIVFPQ index(DIM, faiss::METRIC_L2, NLIST, MSUB, 4, MakeConfig()); },
        "Unsupported nbits");
    ExpectExceptionContains(
        [] { faiss::ascend::AscendIndexIVFPQ index(DIM, faiss::METRIC_L1, NLIST, MSUB, NBITS, MakeConfig()); },
        "Unsupported metric type");
    ExpectExceptionContains(
        [] {
            faiss::ascend::AscendIndexIVFPQ index(DIM, faiss::METRIC_L2, NLIST, MSUB, NBITS,
                                                  MakeConfig(std::vector<int>{}));
        },
        "device");
}

TEST(TestAscendIndexIVFPQ, ConstructorRecognizesLargeNlistBeforeLaterValidation)
{
    for (int nlist : {262144, 524288})
    {
        try
        {
            faiss::ascend::AscendIndexIVFPQ index(DIM, faiss::METRIC_L2, nlist, 3, NBITS, MakeConfig());
            FAIL() << "Expected unsupported msubs for nlist=" << nlist;
        }
        catch (const std::exception &e)
        {
            EXPECT_THAT(std::string(e.what()), HasSubstr("Unsupported msubs"));
            EXPECT_THAT(std::string(e.what()), testing::Not(HasSubstr("Unsupported nlists")));
        }
    }
}

TEST(TestAscendIndexIVFPQ, FacadeRejectsMissingImplementation)
{
    faiss::ascend::AscendIndexIVFPQ index(DIM, faiss::METRIC_L2, NLIST, MSUB, NBITS, MakeConfig());
    index.impl_.reset();
    std::vector<float> vector(DIM, 0.0f);
    faiss::idx_t id = 1;

    ExpectExceptionContains([&] { index.train(1, vector.data()); }, "impl_ is nullptr");
    ExpectExceptionContains([&] { index.copyFrom(nullptr); }, "impl_ is nullptr");
    ExpectExceptionContains([&] { index.copyTo(nullptr); }, "impl_ is nullptr");
    ExpectExceptionContains([&] { index.remove_ids(1, &id); }, "impl_ is nullptr");
    ExpectExceptionContains([&] { index.update(1, vector.data(), &id); }, "impl_ is nullptr");
}

TEST(TestAscendIndexIVFPQ, ProductQuantizerHelpersEncodeNearestCentroids)
{
    faiss::ascend::AscendIndexIVFPQ index(DIM, faiss::METRIC_L2, NLIST, MSUB, NBITS, MakeConfig());
    auto impl = index.impl_;

    EXPECT_EQ(impl->getAddElementSize(), static_cast<size_t>(DIM) * sizeof(float));
    EXPECT_EQ(impl->getAddPagedSize(9), 9U);

    const float lhs[] = {1.0f, 2.0f, 3.0f};
    const float rhs[] = {4.0f, 6.0f, 3.0f};
    EXPECT_FLOAT_EQ(impl->calDistance(lhs, rhs, 3), 25.0f);

    impl->pq.dim = 4;
    impl->pq.M = 2;
    impl->pq.ksub = 3;
    impl->pq.dsub = 2;
    impl->pq.codeBook = {
        0.0f, 0.0f, 1.0f, 1.0f, 5.0f, 5.0f, 0.0f, 0.0f, 2.0f, 2.0f, 7.0f, 7.0f,
    };

    const float vector[] = {0.9f, 1.2f, 6.8f, 7.1f};
    uint8_t code[2] = {0, 0};
    impl->encodeSingleVectorPQ(vector, code);
    EXPECT_THAT(std::vector<uint8_t>(code, code + 2), ElementsAre(1, 2));
    EXPECT_EQ(impl->findCentroidInSubQuantizer(0, nullptr), 0);
    EXPECT_EQ(impl->findCentroidInSubQuantizer(impl->pq.M, vector), 0);

    std::vector<uint8_t> batchCodes(4);
    const float vectors[] = {0.1f, 0.2f, 2.1f, 1.9f, 4.9f, 5.1f, 0.1f, 0.2f};
    index.d = impl->pq.dim;
    impl->addL2(2, vectors, batchCodes);
    index.d = DIM;
    EXPECT_THAT(batchCodes, ElementsAre(0, 1, 2, 0));
}

TEST(TestAscendIndexIVFPQ, SubspaceAndCodebookHelpersPreserveLayout)
{
    faiss::ascend::AscendIndexIVFPQ index(DIM, faiss::METRIC_L2, NLIST, MSUB, NBITS, MakeConfig());
    auto impl = index.impl_;
    impl->pq.dim = 4;
    impl->pq.M = 2;
    impl->pq.ksub = 2;
    impl->pq.dsub = 2;
    impl->pq.codeBook.assign(8, 0.0f);

    const std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<std::vector<float>> subspaces;
    impl->extractAllSubspaces(2, {1, 0}, input.data(), subspaces);
    ASSERT_EQ(subspaces.size(), 2U);
    EXPECT_THAT(subspaces[0], ElementsAre(5, 6, 1, 2));
    EXPECT_THAT(subspaces[1], ElementsAre(7, 8, 3, 4));

    impl->savePQCodeBook(1, {10, 11, 12, 13});
    EXPECT_THAT(impl->pq.codeBook, ElementsAre(0, 0, 0, 0, 10, 11, 12, 13));
    ExpectExceptionContains([&] { impl->savePQCodeBook(2, {1, 2, 3, 4}); }, "out of range");
    ExpectExceptionContains([&] { impl->savePQCodeBook(0, {1, 2}); }, "centroids size error");
}

TEST(TestAscendIndexIVFPQ, IdMappingSupportsLookupFallbackAndRemoval)
{
    faiss::ascend::AscendIndexIVFPQ index(DIM, faiss::METRIC_L2, NLIST, MSUB, NBITS, MakeConfig({2, 5}));
    auto impl = index.impl_;

    impl->updateIdMapping({10, 20, 30}, {1, 1, 3});
    impl->idToDeviceMap[10] = 2;
    impl->idToDeviceMap[20] = 5;
    impl->idToDeviceMap[30] = 2;

    EXPECT_EQ(impl->findListId(20), 1);
    EXPECT_EQ(impl->findDeviceId(20), 5);
    EXPECT_EQ(impl->findListId(1027), 3);
    EXPECT_EQ(impl->findDeviceId(11), 5);
    EXPECT_TRUE(impl->listInfos[1].idSet.count(10));

    impl->removeIdMapping({10, 30, 999});
    EXPECT_FALSE(impl->idToListMap.count(10));
    EXPECT_FALSE(impl->idToDeviceMap.count(30));
    EXPECT_FALSE(impl->listInfos[1].idSet.count(10));
    EXPECT_ANY_THROW(impl->updateIdMapping({1, 2}, {3}));
}

TEST(TestAscendIndexIVFPQ, UpdateValidatesInputsAndReportsMissingIdsWithoutDeviceWork)
{
    faiss::ascend::AscendIndexIVFPQ index(DIM, faiss::METRIC_L2, NLIST, MSUB, NBITS, MakeConfig());
    std::vector<float> vector(DIM, 0.25f);
    faiss::idx_t id = 404;

    ExpectExceptionContains([&] { index.update(1, nullptr, &id); }, "vector list is nullptr");
    ExpectExceptionContains([&] { index.update(1, vector.data(), nullptr); }, "vector ID list is nullptr");
    ExpectExceptionContains([&] { index.update(0, vector.data(), &id); }, "greater than 0");
    ExpectExceptionContains([&] { index.update(1, vector.data(), &id); }, "not trained");

    index.is_trained = true;
    EXPECT_THAT(index.update(1, vector.data(), &id), ElementsAre(id));
}

TEST(TestAscendIndexIVFPQ, CopyFromAndCopyToRoundTripPretrainedData)
{
    auto source = MakePretrainedCpuIndex();
    faiss::ascend::AscendIndexIVFPQ deviceIndex(DIM, faiss::METRIC_L2, NLIST, MSUB, NBITS, MakeConfig());

    ASSERT_NO_THROW(deviceIndex.copyFrom(source.index.get()));
    EXPECT_TRUE(deviceIndex.is_trained);
    EXPECT_EQ(deviceIndex.ntotal, static_cast<faiss::idx_t>(source.ids.size()));
    EXPECT_EQ(deviceIndex.impl_->findListId(source.ids[0]), source.listId);
    EXPECT_EQ(deviceIndex.impl_->findDeviceId(source.ids[0]), 0);

    // copyTo initializes every field and installs an owned quantizer.
    faiss::IndexIVFPQ output;
    ASSERT_NO_THROW(deviceIndex.copyTo(&output));

    EXPECT_EQ(output.d, DIM);
    EXPECT_EQ(output.nlist, static_cast<size_t>(NLIST));
    EXPECT_EQ(output.nprobe, static_cast<size_t>(NPROBE));
    EXPECT_EQ(output.ntotal, static_cast<faiss::idx_t>(source.ids.size()));
    EXPECT_TRUE(output.is_trained);
    ASSERT_EQ(output.invlists->list_size(source.listId), source.ids.size());

    const faiss::idx_t *outputIds = output.invlists->get_ids(source.listId);
    const uint8_t *outputCodes = output.invlists->get_codes(source.listId);
    EXPECT_TRUE(std::equal(source.ids.begin(), source.ids.end(), outputIds));
    EXPECT_TRUE(std::equal(source.codes.begin(), source.codes.end(), outputCodes));
    EXPECT_EQ(output.pq.centroids.size(), source.index->pq.centroids.size());
}

TEST(TestAscendIndexIVFPQ, CopyOperationsRejectNullAndUntrainedIndexes)
{
    faiss::ascend::AscendIndexIVFPQ deviceIndex(DIM, faiss::METRIC_L2, NLIST, MSUB, NBITS, MakeConfig());
    auto untrained = MakePretrainedCpuIndex(false);
    untrained.index->is_trained = false;

    ExpectExceptionContains([&] { deviceIndex.copyFrom(nullptr); }, "index is nullptr");
    ExpectExceptionContains([&] { deviceIndex.copyFrom(untrained.index.get()); }, "not trained");

    faiss::IndexIVFPQ output;
    ExpectExceptionContains([&] { deviceIndex.copyTo(&output); }, "Index is not trained");
    deviceIndex.is_trained = true;
    ExpectExceptionContains([&] { deviceIndex.copyTo(nullptr); }, "index is nullptr");
}

TEST(TestAscendIndexIVFPQ, QueryParallelCopyReplicatesListsAcrossDevices)
{
    auto source = MakePretrainedCpuIndex();
    auto config = MakeConfig({0, 1});
    config.enableQueryParallelSearch = true;
    faiss::ascend::AscendIndexIVFPQ deviceIndex(DIM, faiss::METRIC_L2, NLIST, MSUB, NBITS, config);

    ASSERT_NO_THROW(deviceIndex.copyFrom(source.index.get()));
    EXPECT_TRUE(deviceIndex.impl_->queryParallelSearchReady);
    EXPECT_EQ(deviceIndex.impl_->deviceAddNumMap[source.listId][0], static_cast<int>(source.ids.size()));
    EXPECT_EQ(deviceIndex.impl_->deviceAddNumMap[source.listId][1], static_cast<int>(source.ids.size()));

    const auto queries = MakePatternData(3, DIM);
    std::vector<float> distances(3, std::numeric_limits<float>::infinity());
    std::vector<faiss::idx_t> labels(3, -1);
    ASSERT_NO_THROW(deviceIndex.search(3, queries.data(), 1, distances.data(), labels.data()));
    EXPECT_TRUE(std::all_of(distances.begin(), distances.end(), [](float value) { return std::isfinite(value); }));
    EXPECT_TRUE(
        std::all_of(labels.begin(), labels.end(), [&](faiss::idx_t label) { return label == source.ids.front(); }));

    std::vector<float> vector(DIM, 0.0f);
    faiss::idx_t id = 9999;
    ExpectExceptionContains([&] { deviceIndex.update(1, vector.data(), &id); }, "only supports copyFrom-loaded");
    ExpectExceptionContains([&] { deviceIndex.remove_ids(1, &id); }, "only supports copyFrom-loaded");
}

TEST(TestAscendIndexIVFPQ, PretrainedAddUpdateAndRemoveRunEndToEndWithoutNpu)
{
    // Loading deterministic coarse/PQ centroids avoids expensive public-shape
    // training while still exercising device assignment, PQ encoding, storage,
    // update, deletion and ID bookkeeping through the public interface.
    faiss::ascend::AscendIndexIVFPQ index(DIM, faiss::METRIC_L2, NLIST, MSUB, NBITS, MakeConfig());
    auto source = MakePretrainedCpuIndex(false);
    ASSERT_NO_THROW(index.copyFrom(source.index.get()));
    ASSERT_TRUE(index.is_trained);
    EXPECT_EQ(index.impl_->centroidsOnHost.size(), static_cast<size_t>(NLIST * DIM));
    EXPECT_EQ(index.impl_->pq.codeBook.size(), static_cast<size_t>(DIM) * (1U << NBITS));

    const auto training = MakePatternData(4, DIM);
    const std::vector<faiss::idx_t> ids = {7001, 7002, 7003, 7004};
    ASSERT_NO_THROW(index.add_with_ids(ids.size(), training.data(), ids.data()));
    EXPECT_EQ(index.ntotal, static_cast<faiss::idx_t>(ids.size()));
    for (faiss::idx_t id : ids)
    {
        EXPECT_GE(index.impl_->findListId(id), 0);
        EXPECT_EQ(index.impl_->findDeviceId(id), 0);
    }

    float distance = std::numeric_limits<float>::infinity();
    faiss::idx_t label = -1;
    ASSERT_NO_THROW(index.search(1, training.data(), 1, &distance, &label));
    EXPECT_TRUE(std::isfinite(distance));
    EXPECT_EQ(label, ids.front());

    auto replacement = MakePatternData(1, DIM);
    std::fill(replacement.begin(), replacement.end(), 0.75f);
    EXPECT_TRUE(index.update(1, replacement.data(), &ids[1]).empty());
    EXPECT_EQ(index.ntotal, static_cast<faiss::idx_t>(ids.size()));

    ASSERT_NO_THROW(index.remove_ids(1, &ids[2]));
    EXPECT_EQ(index.ntotal, static_cast<faiss::idx_t>(ids.size() - 1));
    EXPECT_EQ(index.impl_->idToListMap.count(ids[2]), 0U);
}

TEST(TestIndexIVFPQCore, SamplingNormalizationCapacityAndBurstCalculation)
{
    faiss::ascend::AscendIndexIVFPQ hostIndex(DIM, faiss::METRIC_L2, NLIST, MSUB, NBITS, MakeConfig());
    auto actual = hostIndex.impl_->getActualIndex(0);
    ASSERT_NE(actual, nullptr);

    EXPECT_EQ(::ascend::IndexIVFPQ::getActualRngSeed(123), 123U);
    EXPECT_NE(::ascend::IndexIVFPQ::getActualRngSeed(-1), 0U);

    const std::vector<float> input = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    std::vector<float> copied;
    ::ascend::IndexIVFPQ::sampleTrainData(input.data(), 8, 2, 8, 7, copied);
    EXPECT_EQ(copied, input);

    std::vector<float> sampledA;
    std::vector<float> sampledB;
    ::ascend::IndexIVFPQ::sampleTrainData(input.data(), 8, 2, 3, 7, sampledA);
    ::ascend::IndexIVFPQ::sampleTrainData(input.data(), 8, 2, 3, 7, sampledB);
    EXPECT_EQ(sampledA, sampledB);
    ASSERT_EQ(sampledA.size(), 6U);
    for (size_t i = 0; i < sampledA.size(); i += 2)
    {
        bool found = false;
        for (size_t j = 0; j < input.size(); j += 2)
        {
            found = found || (sampledA[i] == input[j] && sampledA[i + 1] == input[j + 1]);
        }
        EXPECT_TRUE(found);
    }

    std::vector<float> normalized = {3.0f, 4.0f, 0.0f, 0.0f};
    actual->normL2(2, 2, normalized.data());
    EXPECT_NEAR(normalized[0], 0.6f, 1e-6f);
    EXPECT_NEAR(normalized[1], 0.8f, 1e-6f);
    EXPECT_FLOAT_EQ(normalized[2], 0.0f);
    EXPECT_FLOAT_EQ(normalized[3], 0.0f);

    const size_t minimum = actual->getPQVecCapacity(1, 0, MSUB);
    EXPECT_GE(minimum, 4U * 1024U);
    EXPECT_EQ(actual->getPQVecCapacity(200000, 1000000, MSUB), 1000000U);
    EXPECT_GT(actual->getPQVecCapacity(200000, 0, MSUB), minimum);

    auto &soc = faiss::ascend::SocUtils::GetInstance();
    const auto originalSoc = soc.socAttr.socType;
    int burstLen = 0;
    soc.socAttr.socType = faiss::ascend::SocUtils::SocType::SOC_910B4;
    EXPECT_GT(::ascend::IndexIVFPQ::GetBurstsOfBlock(64, 128, burstLen), 0);
    EXPECT_EQ(burstLen, 64);
    soc.socAttr.socType = faiss::ascend::SocUtils::SocType::SOC_310P;
    EXPECT_GT(::ascend::IndexIVFPQ::GetBurstsOfBlock(49, 128, burstLen), 0);
    EXPECT_EQ(burstLen, 32);
    EXPECT_GT(::ascend::IndexIVFPQ::GetBurstsOfBlock(1, 128, burstLen), 0);
    EXPECT_EQ(burstLen, 64);
    soc.socAttr.socType = originalSoc;
}

TEST(TestIndexIVFPQCore, SmallDeterministicTrainingProducesReusableDeviceCentroids)
{
    // Exercise the complete daemon-side k-means path with dimensions small enough
    // for a CPU-only mock run. 64 lists keeps the distance/top-k tiling valid.
    constexpr int smallDim = 16;
    constexpr int smallNlist = 64;
    constexpr int smallM = 16;
    ::ascend::IndexIVFPQ actual(smallNlist, smallDim, smallM, NBITS, 4);
    const auto training = MakePatternData(smallNlist, smallDim);

    ASSERT_EQ(actual.trainImpl(smallNlist, training.data(), smallDim, smallNlist, 2, 123, false), ::ascend::APP_ERR_OK);
    ASSERT_NE(actual.getTrainCentroidsDevicePtr(), nullptr);
    ASSERT_NE(actual.getTrainCentroidsSqrSumDevicePtr(), nullptr);

    // Reusing the same operation shapes should be a no-op and preserve the
    // trained buffers instead of rebuilding the mock device operators.
    EXPECT_EQ(actual.initClusterTrainOps(smallNlist, smallDim), ::ascend::APP_ERR_OK);

    std::vector<float> centroids(training.begin(), training.end());
    std::vector<int64_t> assignments(smallNlist, -1);
    ASSERT_EQ(actual.assignCentroid(smallNlist, smallDim, smallNlist, centroids, const_cast<float *>(training.data()),
                                    assignments, false),
              ::ascend::APP_ERR_OK);
    EXPECT_TRUE(std::all_of(assignments.begin(), assignments.end(),
                            [smallNlist](int64_t value) { return value >= 0 && value < smallNlist; }));

    EXPECT_EQ(actual.syncTrainCentroidsFromHost(smallNlist, smallDim, nullptr), ::ascend::APP_ERR_INVALID_PARAM);
    ASSERT_EQ(actual.syncTrainCentroidsFromHost(smallNlist, smallDim, centroids.data()), ::ascend::APP_ERR_OK);
    ASSERT_EQ(actual.copyTrainCentroidsFromDevice(smallNlist, smallDim, actual.getTrainCentroidsDevicePtr(),
                                                  actual.getTrainCentroidsSqrSumDevicePtr()),
              ::ascend::APP_ERR_OK);

    std::fill(assignments.begin(), assignments.end(), -1);
    ASSERT_EQ(actual.assignCentroid(smallNlist, smallDim, smallNlist, centroids, const_cast<float *>(training.data()),
                                    assignments, true),
              ::ascend::APP_ERR_OK);
    EXPECT_TRUE(std::all_of(assignments.begin(), assignments.end(),
                            [smallNlist](int64_t value) { return value >= 0 && value < smallNlist; }));
    EXPECT_GE(actual.getL3SearchBatchCap(), 1);

    actual.resetTrainSession();
    EXPECT_EQ(actual.getTrainCentroidsDevicePtr(), nullptr);
    EXPECT_EQ(actual.getTrainCentroidsSqrSumDevicePtr(), nullptr);
}

TEST(TestIndexIVFPQCore, AddReadDeleteAndInvalidSearchAreDeviceIndependent)
{
    faiss::ascend::AscendIndexIVFPQ hostIndex(DIM, faiss::METRIC_L2, NLIST, MSUB, NBITS, MakeConfig());
    auto actual = hostIndex.impl_->getActualIndex(0);
    ASSERT_NE(actual, nullptr);

    const int listId = 5;
    const std::vector<::ascend::Index::idx_t> ids = {101, 202, 303};
    const std::vector<uint8_t> codes = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    EXPECT_EQ(actual->addPQCodes(-1, ids.size(), codes.data(), ids.data()), ::ascend::APP_ERR_INVALID_PARAM);
    EXPECT_EQ(actual->addPQCodes(listId, 0, codes.data(), ids.data()), ::ascend::APP_ERR_OK);
    ASSERT_EQ(actual->addPQCodes(listId, ids.size(), codes.data(), ids.data()), ::ascend::APP_ERR_OK);
    EXPECT_EQ(actual->getListLength(listId), ids.size());

    std::vector<unsigned char> reshaped;
    ASSERT_EQ(actual->getListVectorsReshaped(listId, reshaped), ::ascend::APP_ERR_OK);
    EXPECT_EQ(reshaped.size(), codes.size());
    EXPECT_TRUE(std::equal(codes.begin(), codes.end(), reshaped.begin()));

    const ::ascend::Index::idx_t deleteId = ids[1];
    ASSERT_EQ(actual->deletePQCodes(listId, 1, &deleteId), ::ascend::APP_ERR_OK);
    EXPECT_EQ(actual->getListLength(listId), 2U);
    EXPECT_ANY_THROW(actual->getListLength(NLIST));

    std::vector<float> query(DIM, 0.0f);
    std::vector<float> distances(1, 0.0f);
    std::vector<::ascend::Index::idx_t> labels(1, -1);
    EXPECT_EQ(actual->searchImpl(1, query.data(), 0, distances.data(), labels.data()), ::ascend::APP_ERR_INVALID_PARAM);

    ASSERT_EQ(actual->reset(), ::ascend::APP_ERR_OK);
    EXPECT_EQ(actual->getListLength(listId), 0U);
}
}  // namespace
