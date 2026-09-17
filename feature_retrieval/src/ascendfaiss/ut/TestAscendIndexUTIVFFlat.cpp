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
#include <faiss/IndexIVFFlat.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ascend/AscendIndexIVFFlat.h"
#include "ascend/AscendIndexQuantizerImpl.h"
#include "ascend/impl/AscendIndexIVFFlatImpl.h"

namespace
{
using testing::HasSubstr;
using testing::UnorderedElementsAreArray;

constexpr int DIM = 128;
constexpr int NLIST = 1024;
constexpr int NPROBE = 8;

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

std::vector<float> MakePatternData(size_t count)
{
    std::vector<float> data(count * static_cast<size_t>(DIM));
    for (size_t i = 0; i < count; ++i)
    {
        for (int j = 0; j < DIM; ++j)
        {
            data[i * static_cast<size_t>(DIM) + static_cast<size_t>(j)] =
                static_cast<float>((i * 13 + static_cast<size_t>(j) * 7) % 97) / 97.0f;
        }
    }
    return data;
}

struct CpuIVFFlatIndex
{
    std::unique_ptr<faiss::IndexFlatIP> quantizer;
    std::unique_ptr<faiss::IndexIVFFlat> index;
    std::vector<faiss::idx_t> ids;
    std::vector<float> vectors;
    int listId = 3;
};

CpuIVFFlatIndex MakePretrainedCpuIndex(bool withVectors = true)
{
    CpuIVFFlatIndex result;
    result.quantizer = std::make_unique<faiss::IndexFlatIP>(DIM);
    result.index =
        std::make_unique<faiss::IndexIVFFlat>(result.quantizer.get(), DIM, NLIST, faiss::METRIC_INNER_PRODUCT);
    result.index->nprobe = NPROBE;

    auto centroids = MakePatternData(NLIST);
    result.quantizer->add(NLIST, centroids.data());
    result.index->is_trained = true;

    if (withVectors)
    {
        result.ids = {11, 22, 33, 44};
        result.vectors = MakePatternData(result.ids.size());
        result.index->invlists->add_entries(result.listId, result.ids.size(), result.ids.data(),
                                            reinterpret_cast<const uint8_t *>(result.vectors.data()));
        result.index->ntotal = static_cast<faiss::idx_t>(result.ids.size());
    }
    return result;
}

faiss::ascend::AscendIndexIVFFlatConfig MakeConfig(std::vector<int> devices = {0})
{
    faiss::ascend::AscendIndexIVFFlatConfig config(std::move(devices));
    config.useKmeansPP = false;
    config.cp.niter = 1;
    config.cp.nredo = 1;
    config.cp.min_points_per_centroid = 1;
    config.cp.max_points_per_centroid = 1;
    return config;
}

TEST(TestAscendIndexIVFFlat, ConstructorAcceptsSupportedParameters)
{
    faiss::ascend::AscendIndexIVFFlat index(DIM, faiss::METRIC_INNER_PRODUCT, NLIST, MakeConfig());

    EXPECT_EQ(index.d, DIM);
    EXPECT_EQ(index.getNumLists(), NLIST);
    EXPECT_FALSE(index.is_trained);
    EXPECT_EQ(index.impl_->getAddElementSize(), static_cast<size_t>(DIM) * sizeof(float));
    EXPECT_EQ(index.impl_->getAddPagedSize(7), 7U);
}

TEST(TestAscendIndexIVFFlat, ConstructorValidatesPublicParameters)
{
    ExpectExceptionContains(
        [] { faiss::ascend::AscendIndexIVFFlat index(64, faiss::METRIC_INNER_PRODUCT, NLIST, MakeConfig()); },
        "Unsupported dims");
    ExpectExceptionContains(
        [] { faiss::ascend::AscendIndexIVFFlat index(DIM, faiss::METRIC_INNER_PRODUCT, 512, MakeConfig()); },
        "Unsupported nlists");
    ExpectExceptionContains([] { faiss::ascend::AscendIndexIVFFlat index(DIM, faiss::METRIC_L2, NLIST, MakeConfig()); },
                            "Unsupported metric type");
    ExpectExceptionContains(
        [] {
            faiss::ascend::AscendIndexIVFFlat index(DIM, faiss::METRIC_INNER_PRODUCT, NLIST,
                                                    MakeConfig(std::vector<int>{}));
        },
        "device");
}

TEST(TestAscendIndexIVFFlat, FacadeRejectsMissingImplementation)
{
    faiss::ascend::AscendIndexIVFFlat index(DIM, faiss::METRIC_INNER_PRODUCT, NLIST, MakeConfig());
    index.impl_.reset();
    std::vector<float> vector(DIM, 0.0f);

    ExpectExceptionContains([&] { index.train(1, vector.data()); }, "impl_ is nullptr");
    ExpectExceptionContains([&] { index.copyFrom(nullptr); }, "impl_ is nullptr");
    ExpectExceptionContains([&] { index.copyTo(nullptr); }, "impl_ is nullptr");
}

TEST(TestAscendIndexIVFFlat, CopyFromValidatesSourceState)
{
    faiss::ascend::AscendIndexIVFFlat index(DIM, faiss::METRIC_INNER_PRODUCT, NLIST, MakeConfig());
    auto source = MakePretrainedCpuIndex(false);
    source.index->is_trained = false;

    ExpectExceptionContains([&] { index.copyFrom(nullptr); }, "index is nullptr");
    ExpectExceptionContains([&] { index.copyFrom(source.index.get()); }, "Source index is not trained");
    ExpectExceptionContains([&] { index.copyTo(nullptr); }, "index is nullptr");
    ExpectExceptionContains(
        [&]
        {
            faiss::IndexIVFFlat destination;
            index.copyTo(&destination);
        },
        "Index is not trained");
}

TEST(TestAscendIndexIVFFlat, CopyRoundTripPreservesVectorsAcrossDevices)
{
    auto source = MakePretrainedCpuIndex();
    faiss::ascend::AscendIndexIVFFlat index(DIM, faiss::METRIC_INNER_PRODUCT, NLIST, MakeConfig({0, 1}));

    index.copyFrom(source.index.get());

    EXPECT_TRUE(index.is_trained);
    EXPECT_EQ(index.ntotal, static_cast<faiss::idx_t>(source.ids.size()));
    EXPECT_EQ(index.getNumProbes(), NPROBE);
    EXPECT_EQ(index.getListLength(source.listId), source.ids.size());
    EXPECT_EQ(index.impl_->pQuantizerImpl->cpuQuantizer->ntotal, NLIST);
    EXPECT_TRUE(index.impl_->pQuantizerImpl->cpuQuantizer->is_trained);
    EXPECT_EQ(index.impl_->deviceAddNumMap[source.listId][0], 2);
    EXPECT_EQ(index.impl_->deviceAddNumMap[source.listId][1], 2);

    faiss::IndexIVFFlat destination;
    index.copyTo(&destination);

    ASSERT_TRUE(destination.is_trained);
    ASSERT_EQ(destination.ntotal, index.ntotal);
    ASSERT_EQ(destination.nlist, static_cast<size_t>(NLIST));
    ASSERT_EQ(destination.nprobe, static_cast<size_t>(NPROBE));
    ASSERT_NE(destination.quantizer, nullptr);
    EXPECT_EQ(destination.quantizer->ntotal, NLIST);
    ASSERT_EQ(destination.invlists->list_size(source.listId), source.ids.size());
    const faiss::idx_t *copiedIds = destination.invlists->get_ids(source.listId);
    EXPECT_THAT(std::vector<faiss::idx_t>(copiedIds, copiedIds + source.ids.size()),
                UnorderedElementsAreArray(source.ids));
}

TEST(TestAscendIndexIVFFlat, CopyFromIndexCanAcceptAdditionalVectors)
{
    auto source = MakePretrainedCpuIndex(false);
    faiss::ascend::AscendIndexIVFFlat index(DIM, faiss::METRIC_INNER_PRODUCT, NLIST, MakeConfig());
    index.copyFrom(source.index.get());

    std::vector<faiss::idx_t> ids = {101, 202, 303};
    auto vectors = MakePatternData(ids.size());
    index.add_with_ids(ids.size(), vectors.data(), ids.data());

    EXPECT_EQ(index.ntotal, static_cast<faiss::idx_t>(ids.size()));
    faiss::IndexIVFFlat destination;
    index.copyTo(&destination);

    size_t stored = 0;
    std::vector<faiss::idx_t> storedIds;
    for (int listId = 0; listId < NLIST; ++listId)
    {
        size_t listSize = destination.invlists->list_size(listId);
        stored += listSize;
        const faiss::idx_t *listIds = destination.invlists->get_ids(listId);
        storedIds.insert(storedIds.end(), listIds, listIds + listSize);
    }
    EXPECT_EQ(stored, ids.size());
    EXPECT_THAT(storedIds, UnorderedElementsAreArray(ids));
}

TEST(TestAscendIndexIVFFlat, CpuTrainingBuildsCoarseCentroids)
{
    faiss::ascend::AscendIndexIVFFlat index(DIM, faiss::METRIC_INNER_PRODUCT, NLIST, MakeConfig());
    auto training = MakePatternData(NLIST);

    ExpectExceptionContains([&] { index.train(0, training.data()); }, "n must be > 0");
    ExpectExceptionContains([&] { index.train(NLIST, nullptr); }, "x can not be nullptr");
    index.train(NLIST, training.data());

    EXPECT_TRUE(index.is_trained);
    EXPECT_EQ(index.impl_->pQuantizerImpl->cpuQuantizer->ntotal, NLIST);
    EXPECT_NO_THROW(index.train(NLIST, training.data()));
}
}  // namespace
