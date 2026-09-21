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
#include <faiss/IndexIVFRaBitQ.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "ascend/AscendIndexIVFRaBitQ.h"
#include "ascend/impl/AscendIndexIVFRaBitQImpl.h"
#include "ascenddaemon/impl/IndexIVFRaBitQ.h"

namespace faiss
{
namespace ascend
{
// These helpers are intentionally kept in the implementation file. Declaring
// them here lets the UT exercise both orthonormal-initialization branches.
void orthonormalinit(std::vector<float> &matrix, int seed, int inputDim, int outputDim);
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

void ExpectOrthogonal(const std::vector<float> &matrix, int rows, int cols)
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

    const faiss::idx_t *outputIds = output.invlists->get_ids(source.listId);
    const uint8_t *outputCodes = output.invlists->get_codes(source.listId);
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
}  // namespace
