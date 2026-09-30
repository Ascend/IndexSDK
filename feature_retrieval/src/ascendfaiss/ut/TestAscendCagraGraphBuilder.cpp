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

#include <cstdint>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "common/ErrorCode.h"
#include "index/AscendCagraGraphBuilder.h"

namespace faiss
{
namespace ascend
{
TEST(TestAscendCagraGraphBuilder, DefaultConfig)
{
    AscendCagraGraphBuildConfig config;
    EXPECT_EQ(config.intermediateDegree, 128U);
    EXPECT_EQ(config.maxIterations, 20U);
    EXPECT_FLOAT_EQ(config.terminationThreshold, 0.0001F);
    EXPECT_FALSE(config.verbose);
    EXPECT_TRUE(config.guaranteeConnectivity);
}

TEST(TestAscendCagraGraphBuilder, RejectsInvalidInitParametersBeforeDeviceAccess)
{
    AscendCagraGraphBuilder builder;
    const std::vector<int> oneDevice{0};

    EXPECT_EQ(builder.Init(0, 64, 10000, oneDevice), ::ascend::APP_ERR_INVALID_PARAM);
    EXPECT_EQ(builder.Init(3073, 64, 10000, oneDevice), ::ascend::APP_ERR_INVALID_PARAM);
    EXPECT_EQ(builder.Init(128, 0, 10000, oneDevice), ::ascend::APP_ERR_INVALID_PARAM);
    EXPECT_EQ(builder.Init(128, 10000, 10000, oneDevice), ::ascend::APP_ERR_INVALID_PARAM);
    EXPECT_EQ(builder.Init(128, 64, 1, oneDevice), ::ascend::APP_ERR_INVALID_PARAM);
    EXPECT_EQ(builder.Init(128, 64, 10000, {}), ::ascend::APP_ERR_INVALID_PARAM);
    EXPECT_EQ(builder.Init(128, 64, 10000, {0, 1}), ::ascend::APP_ERR_INVALID_PARAM);
}

TEST(TestAscendCagraGraphBuilder, BuildRequiresInitialization)
{
    AscendCagraGraphBuilder builder;
    float data = 0.0F;
    uint32_t graph = 0;

    EXPECT_EQ(builder.Build(&data, &graph), ::ascend::APP_ERR_INDEX_NOT_INIT);
    EXPECT_EQ(builder.BuildToFile(&data, "cagra_graph.bin"), ::ascend::APP_ERR_INDEX_NOT_INIT);
}

TEST(TestAscendCagraGraphBuilder, BuildsStructurallyValidGraphOnA5)
{
    const char *enabled = std::getenv("CAGRA_RUN_GRAPH_BUILD_UT");
    if (enabled == nullptr || std::string(enabled) != "1")
    {
        GTEST_SKIP() << "Set CAGRA_RUN_GRAPH_BUILD_UT=1 after generating the default graph-build models";
    }

    constexpr int dim = 128;
    constexpr int dataNum = 10000;
    constexpr int graphDegree = 64;
    const char *deviceIdValue = std::getenv("CAGRA_DEVICE_ID");
    const int deviceId = deviceIdValue == nullptr ? 0 : std::atoi(deviceIdValue);

    std::mt19937 randomEngine(42);
    std::uniform_real_distribution<float> distribution(0.0F, 1.0F);
    std::vector<float> data(static_cast<size_t>(dataNum) * dim);
    for (float &value : data)
    {
        value = distribution(randomEngine);
    }

    AscendCagraGraphBuilder builder;
    ASSERT_EQ(builder.Init(dim, graphDegree, dataNum, {deviceId}), ::ascend::APP_ERR_OK);

    AscendCagraGraphBuildConfig config;
    config.intermediateDegree = 128;
    config.maxIterations = 20;
    std::vector<uint32_t> graph(static_cast<size_t>(dataNum) * graphDegree);
    ASSERT_EQ(builder.Build(data.data(), graph.data(), config), ::ascend::APP_ERR_OK);

    std::vector<int> lastSeen(dataNum, -1);
    for (int row = 0; row < dataNum; ++row)
    {
        for (int column = 0; column < graphDegree; ++column)
        {
            const uint32_t neighbor = graph[static_cast<size_t>(row) * graphDegree + column];
            ASSERT_LT(neighbor, static_cast<uint32_t>(dataNum)) << "row=" << row << ", column=" << column;
            ASSERT_NE(neighbor, static_cast<uint32_t>(row)) << "row=" << row << ", column=" << column;
            ASSERT_NE(lastSeen[neighbor], row) << "duplicate neighbor in row=" << row;
            lastSeen[neighbor] = row;
        }
    }
}
}  // namespace ascend
}  // namespace faiss
