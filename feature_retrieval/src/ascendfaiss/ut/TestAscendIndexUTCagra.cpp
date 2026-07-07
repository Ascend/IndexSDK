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
#include <omp.h>
#include <sys/time.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <numeric>
#include <random>
#include <vector>

#include "AscendIndexCagra.h"
using namespace testing;
using namespace std;
namespace faiss
{
namespace ascend
{
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

        ret = index.QuantizeData(queryNum, queryData.data(), ntotal, data_ptrData.data());
        EXPECT_EQ(ret, 0);
        cout << "QuantizeData done ..." << endl;
        // Prepare output vectors
        std::vector<float> dists(queryNum * topK);
        std::vector<uint32_t> labels(queryNum * topK);

        ret = index.Search(queryNum, queryData.data(), topK, dists.data(), labels.data());
        EXPECT_EQ(ret, 0);
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
