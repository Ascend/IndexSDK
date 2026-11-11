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

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "ascendhost/include/impl/AscendIndexCagraImpl.h"
#include "ascendhost/include/index/AscendIndexCagra.h"
#include "common/ErrorCode.h"
using namespace testing;
using namespace std;
namespace faiss
{
namespace ascend
{
namespace
{
constexpr int TEST_DIM = 64;
constexpr int TEST_QUERY_BITS = 6;
constexpr int TEST_BYTES_PER_PLANE = TEST_DIM / 8;

class CagraQuantizationTestImpl : public AscendIndexCagraImpl
{
   public:
    CagraQuantizationTestImpl() : AscendIndexCagraImpl(TEST_DIM, 32, {0})
    {
        dataNum = 2;
        rotatedSize = TEST_BYTES_PER_PLANE * TEST_QUERY_BITS;
    }

    void PrepareTransform()
    {
        g_centroid.assign(TEST_DIM, 2.0F);
        g_inverseRotation.assign(TEST_DIM * TEST_DIM, 0.0F);
        for (int i = 0; i < TEST_DIM; ++i)
        {
            g_inverseRotation[i * TEST_DIM + i] = 1.0F;
        }
    }

    using AscendIndexCagraImpl::g_centroid;
    using AscendIndexCagraImpl::g_inverseRotation;
    using AscendIndexCagraImpl::g_precompute;
    using AscendIndexCagraImpl::g_rotated;
};
}  // namespace

TEST(TestCagraQueryQuantizationUT, RequiresInitializationAndBaseQuantization)
{
    AscendIndexCagra index;
    float query = 0.0F;
    EXPECT_EQ(index.QuantizeQueries(1, &query), ::ascend::APP_ERR_INVALID_PARAM);
    CagraQuantizationTestImpl impl;
    EXPECT_EQ(impl.QuantizeQueries(1, &query), ::ascend::APP_ERR_INVALID_PARAM);
}

TEST(TestCagraQueryQuantizationUT, ReusesTransformAndReplacesQueryBatch)
{
    CagraQuantizationTestImpl impl;
    impl.PrepareTransform();
    const auto centroid = impl.g_centroid;
    const auto rotation = impl.g_inverseRotation;
    std::vector<float> queries(TEST_DIM * 2, 2.0F);
    std::fill(queries.begin(), queries.begin() + TEST_DIM, 3.0F);
    ASSERT_EQ(impl.QuantizeQueries(2, queries.data()), ::ascend::APP_ERR_OK);
    const std::vector<float> expected = {64.0F, 32.0F, 1.0F / 31.0F, 4032.0F, 0.0F, 32.0F, 1.0F, 2048.0F};
    ASSERT_EQ(impl.g_precompute.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_FLOAT_EQ(impl.g_precompute[i], expected[i]);
    }
    // Centered +1 maps to 63 in all dimensions; centered 0 maps to 32.
    std::vector<uint8_t> expectedCodes(TEST_BYTES_PER_PLANE * TEST_QUERY_BITS * 2, 0);
    std::fill(expectedCodes.begin(), expectedCodes.begin() + TEST_BYTES_PER_PLANE * TEST_QUERY_BITS, 0xFF);
    std::fill(expectedCodes.end() - TEST_BYTES_PER_PLANE, expectedCodes.end(), 0xFF);
    EXPECT_EQ(impl.g_rotated, expectedCodes);

    // A smaller subsequent batch must replace, rather than append to, the cache.
    std::vector<float> nextQuery(TEST_DIM, 1.0F);
    ASSERT_EQ(impl.QuantizeQueries(1, nextQuery.data()), ::ascend::APP_ERR_OK);
    ASSERT_EQ(impl.g_precompute.size(), 4U);
    EXPECT_FLOAT_EQ(impl.g_precompute[0], 64.0F);
    EXPECT_FLOAT_EQ(impl.g_precompute[1], 32.0F);
    EXPECT_FLOAT_EQ(impl.g_precompute[2], 1.0F / 31.0F);
    EXPECT_FLOAT_EQ(impl.g_precompute[3], 64.0F);
    expectedCodes.assign(TEST_BYTES_PER_PLANE * TEST_QUERY_BITS, 0);
    std::fill(expectedCodes.begin(), expectedCodes.begin() + TEST_BYTES_PER_PLANE, 0xFF);
    EXPECT_EQ(impl.g_rotated, expectedCodes);
    const auto precompute = impl.g_precompute;
    ASSERT_EQ(impl.QuantizeQueries(1, nextQuery.data()), ::ascend::APP_ERR_OK);
    EXPECT_EQ(impl.g_precompute, precompute);
    EXPECT_EQ(impl.g_rotated, expectedCodes);
    EXPECT_EQ(impl.g_centroid, centroid);
    EXPECT_EQ(impl.g_inverseRotation, rotation);
}

TEST(TestCagraQueryQuantizationUT, RejectsInvalidInputWithoutChangingPreparedQueries)
{
    CagraQuantizationTestImpl impl;
    impl.PrepareTransform();
    std::vector<float> query(TEST_DIM, 3.0F);
    ASSERT_EQ(impl.QuantizeQueries(1, query.data()), ::ascend::APP_ERR_OK);
    const auto precompute = impl.g_precompute;
    const auto codes = impl.g_rotated;
    for (int count : {-1, 0, 4097})
    {
        EXPECT_EQ(impl.QuantizeQueries(count, query.data()), ::ascend::APP_ERR_INVALID_PARAM);
    }
    EXPECT_EQ(impl.QuantizeQueries(1, nullptr), ::ascend::APP_ERR_INVALID_PARAM);
    for (int count : {-1, 0, 1, 3})
    {
        EXPECT_EQ(impl.QuantizeData(1, query.data(), count, query.data()), ::ascend::APP_ERR_INVALID_PARAM);
    }
    EXPECT_EQ(impl.g_precompute, precompute);
    EXPECT_EQ(impl.g_rotated, codes);
}

TEST(TestCagraQueryQuantizationUT, SupportsMaximumQueryBatch)
{
    CagraQuantizationTestImpl impl;
    impl.PrepareTransform();
    constexpr int queryCount = 4096;
    std::vector<float> queries(queryCount * TEST_DIM, 2.0F);
    ASSERT_EQ(impl.QuantizeQueries(queryCount, queries.data()), ::ascend::APP_ERR_OK);
    ASSERT_EQ(impl.g_precompute.size(), queryCount * 4U);
    ASSERT_EQ(impl.g_rotated.size(), queryCount * TEST_BYTES_PER_PLANE * TEST_QUERY_BITS);
    for (int row = 0; row < queryCount; ++row)
    {
        EXPECT_FLOAT_EQ(impl.g_precompute[row * 4], 0.0F);
        EXPECT_FLOAT_EQ(impl.g_precompute[row * 4 + 3], 2048.0F);
    }
}

TEST(TestCagraSearchUT, AcceptsTopK256ForSupportedGraphDegrees)
{
    for (int degree : {64, 128})
    {
        AscendIndexCagra index;
        EXPECT_EQ(index.Init(64, degree, 1024, 256, {0}), ::ascend::APP_ERR_OK);
    }
}

TEST(TestCagraSearchUT, RejectsUnsupportedSearchSpecification)
{
    for (int topK : {-1, 0, 1, 10, 31, 33, 127, 129, 255, 257, 1023, 1025, 4096})
    {
        AscendIndexCagra index;
        EXPECT_EQ(index.Init(128, 64, 10000, topK, {0}), ::ascend::APP_ERR_INVALID_PARAM);
    }
    for (int degree : {0, 32, 96, 256, 512})
    {
        AscendIndexCagra index;
        EXPECT_EQ(index.Init(128, degree, 10000, 32, {0}), ::ascend::APP_ERR_INVALID_PARAM);
    }
    for (int dim : {0, 100, 4096})
    {
        AscendIndexCagra index;
        EXPECT_EQ(index.Init(dim, 64, 10000, 32, {0}), ::ascend::APP_ERR_INVALID_PARAM);
    }
    for (int degree : {64, 128})
    {
        AscendIndexCagra boundary;
        EXPECT_EQ(boundary.Init(128, degree, degree, 32, {0}), ::ascend::APP_ERR_INVALID_PARAM);
    }
    AscendIndexCagra index;
    EXPECT_EQ(index.Init(128, 64, 128, 512, {0}), ::ascend::APP_ERR_INVALID_PARAM);
}

TEST(TestCagraUnifiedUT, DefaultsAndEmptyState)
{
    AscendIndexCagraConfig config;
    EXPECT_EQ(config.graphDegree, 64U);
    EXPECT_EQ(config.intermediateDegree, 128U);
    EXPECT_EQ(config.maxIterations, 20U);
    EXPECT_FLOAT_EQ(config.terminationThreshold, 0.0001F);
    EXPECT_FALSE(config.verbose);
    EXPECT_TRUE(config.guaranteeConnectivity);
    AscendIndexCagra index;
    EXPECT_FALSE(index.IsTrained());
    EXPECT_EQ(index.GetSize(), 0);
    std::vector<uint32_t> graph{123};
    EXPECT_EQ(index.GetKnnGraph(graph), ::ascend::APP_ERR_INDEX_NOT_INIT);
    EXPECT_EQ(graph, std::vector<uint32_t>({123}));
    EXPECT_EQ(index.Reset(), ::ascend::APP_ERR_OK);
    EXPECT_EQ(index.Train(129, nullptr), ::ascend::APP_ERR_INVALID_PARAM);
}

TEST(TestCagraUnifiedUT, ValidatesDimensionsAndBuildConfigurationBeforeDeviceAccess)
{
    for (int dim : {64, 128, 256, 512, 1024, 1536, 3072})
    {
        AscendIndexCagra index;
        ASSERT_EQ(index.Init(dim, {0}), ::ascend::APP_ERR_OK);
        EXPECT_FALSE(index.IsTrained());
        EXPECT_EQ(index.Init(dim, {0}), ::ascend::APP_ERR_INVALID_PARAM);
    }
    for (int dim : {0, 100, 3073})
    {
        AscendIndexCagra index;
        EXPECT_EQ(index.Init(dim, {0}), ::ascend::APP_ERR_INVALID_PARAM);
    }
    for (uint32_t degree : {0U, 32U, 96U, 256U, 512U})
    {
        AscendIndexCagra index;
        AscendIndexCagraConfig config;
        config.graphDegree = degree;
        EXPECT_EQ(index.Init(128, {0}, config), ::ascend::APP_ERR_INVALID_PARAM);
    }
    AscendIndexCagra index;
    AscendIndexCagraConfig config;
    config.intermediateDegree = 63;
    EXPECT_EQ(index.Init(128, {0}, config), ::ascend::APP_ERR_INVALID_PARAM);
    config.intermediateDegree = 129;
    EXPECT_EQ(index.Init(128, {0}, config), ::ascend::APP_ERR_INVALID_PARAM);
    config.intermediateDegree = 128;
    config.maxIterations = 0;
    EXPECT_EQ(index.Init(128, {0}, config), ::ascend::APP_ERR_INVALID_PARAM);
    config.maxIterations = 20;
    for (float threshold : {-0.1F, 1.1F, std::numeric_limits<float>::infinity()})
    {
        config.terminationThreshold = threshold;
        EXPECT_EQ(index.Init(128, {0}, config), ::ascend::APP_ERR_INVALID_PARAM);
    }
    config.terminationThreshold = std::numeric_limits<float>::quiet_NaN();
    EXPECT_EQ(index.Init(128, {0}, config), ::ascend::APP_ERR_INVALID_PARAM);
    EXPECT_EQ(index.Init(128, {-1}), ::ascend::APP_ERR_INVALID_PARAM);
    EXPECT_EQ(index.Init(128, {}), ::ascend::APP_ERR_INVALID_PARAM);
    EXPECT_EQ(index.Init(128, {0, 1}), ::ascend::APP_ERR_INVALID_PARAM);
}

TEST(TestCagraUnifiedUT, InvalidTrainingLeavesAnEmptyReusableIndex)
{
    AscendIndexCagra index;
    ASSERT_EQ(index.Init(64, {0}), ::ascend::APP_ERR_OK);
    std::vector<float> data(129 * 64, 0.0F);
    EXPECT_EQ(index.Train(128, data.data()), ::ascend::APP_ERR_INVALID_PARAM);
    data[0] = std::numeric_limits<float>::infinity();
    EXPECT_EQ(index.Add(129, data.data()), ::ascend::APP_ERR_INVALID_PARAM);
    EXPECT_FALSE(index.IsTrained());
    EXPECT_EQ(index.GetSize(), 0);
    ASSERT_EQ(index.Reset(), ::ascend::APP_ERR_OK);
    EXPECT_EQ(index.GetSize(), 0);
}

TEST(TestCagraUnifiedUT, GraphExportValidatesPathAndTrainingState)
{
    AscendIndexCagra index;
    EXPECT_EQ(index.SaveGraphToFile(""), ::ascend::APP_ERR_INVALID_PARAM);
    EXPECT_EQ(index.SaveGraphToFile(std::string("graph\0.bin", 10)), ::ascend::APP_ERR_INVALID_PARAM);
    EXPECT_EQ(index.SaveGraphToFile("graph.bin"), ::ascend::APP_ERR_INDEX_NOT_INIT);
    ASSERT_EQ(index.Init(64, {0}), ::ascend::APP_ERR_OK);
    EXPECT_EQ(index.SaveGraphToFile("graph.bin"), ::ascend::APP_ERR_INDEX_NOT_INIT);
    ASSERT_EQ(index.Reset(), ::ascend::APP_ERR_OK);
    EXPECT_EQ(index.SaveGraphToFile("graph.bin"), ::ascend::APP_ERR_INDEX_NOT_INIT);
    EXPECT_FALSE(index.IsTrained());
    EXPECT_EQ(index.GetSize(), 0);
}

TEST(TestCagraSearchUT, Search)
{
    try
    {
        int dim = 128;
        int ntotal = 10000;
        int queryNum = 64;
        int topK = 32;
        int graph = 64;
        std::vector<int> deviceList = {0};
        faiss::ascend::AscendIndexCagra index;
        auto ret = index.Init(dim, graph, ntotal, topK, deviceList);
        ASSERT_EQ(ret, 0);

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> distFloat(0.0f, 1.0f);
        std::uniform_int_distribution<uint32_t> distUint(0, static_cast<uint32_t>(ntotal - 1));
        // 生成随机 query 数据
        std::vector<float> queryData(queryNum * dim);
        for (auto &v : queryData)
        {
            v = distFloat(rng);
        }
        // 生成随机 graph 数据
        std::vector<uint32_t> graphData(ntotal * graph);
        for (auto &v : graphData)
        {
            v = distUint(rng);
        }
        // 生成随机 hash 数据
        std::vector<uint32_t> hashData(ntotal * 2);
        for (auto &v : hashData)
        {
            v = distUint(rng);
        }
        // 生成随机 base 数据
        std::vector<float> data_ptrData(ntotal * dim);
        for (auto &v : data_ptrData)
        {
            v = distFloat(rng);
        }
        ret = index.Add(graphData.data(), hashData.data(), data_ptrData.data());
        EXPECT_EQ(ret, 0);
        cout << "add done ..." << endl;

        EXPECT_EQ(index.QuantizeQueries(queryNum, queryData.data()), ::ascend::APP_ERR_INVALID_PARAM);
        ret = index.QuantizeData(queryNum, queryData.data(), ntotal, data_ptrData.data());
        EXPECT_EQ(ret, 0);
        cout << "QuantizeData done ..." << endl;
        // Prepare output vectors
        std::vector<float> dists(queryNum * topK);
        std::vector<uint32_t> labels(queryNum * topK);

        ret = index.Search(queryNum, queryData.data(), topK, dists.data(), labels.data());
        EXPECT_EQ(ret, 0);
        EXPECT_EQ(index.QuantizeQueries(0, queryData.data()), ::ascend::APP_ERR_INVALID_PARAM);
        EXPECT_EQ(index.QuantizeQueries(4097, queryData.data()), ::ascend::APP_ERR_INVALID_PARAM);
        EXPECT_EQ(index.QuantizeQueries(queryNum, nullptr), ::ascend::APP_ERR_INVALID_PARAM);
        ASSERT_EQ(index.QuantizeQueries(1, queryData.data()), ::ascend::APP_ERR_OK);
        EXPECT_EQ(index.Search(queryNum, queryData.data(), topK, dists.data(), labels.data()),
                  ::ascend::APP_ERR_INVALID_PARAM);
        std::fill(queryData.begin(), queryData.end(), 0.25F);
        ASSERT_EQ(index.QuantizeQueries(queryNum, queryData.data()), ::ascend::APP_ERR_OK);
        EXPECT_EQ(index.Search(queryNum, queryData.data(), topK, dists.data(), labels.data()), ::ascend::APP_ERR_OK);
        EXPECT_EQ(index.Search(queryNum, queryData.data(), topK - 1, dists.data(), labels.data()),
                  ::ascend::APP_ERR_INVALID_PARAM);
        EXPECT_EQ(index.Search(queryNum, queryData.data(), topK + 1, dists.data(), labels.data()),
                  ::ascend::APP_ERR_INVALID_PARAM);
        ASSERT_EQ(index.Add(graphData.data(), hashData.data(), data_ptrData.data()), ::ascend::APP_ERR_OK);
        EXPECT_EQ(index.QuantizeQueries(queryNum, queryData.data()), ::ascend::APP_ERR_INVALID_PARAM);
        EXPECT_EQ(index.Search(queryNum, queryData.data(), topK, dists.data(), labels.data()),
                  ::ascend::APP_ERR_INVALID_PARAM);
        cout << "Search done ..." << endl;
    }
    catch (std::exception &e)
    {
        FAIL() << "Exception occurred: " << e.what();
    }
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
}  // namespace ascend
}  // namespace faiss
