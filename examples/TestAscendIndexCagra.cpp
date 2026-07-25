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

// CAGRA（CUDA Accelerated Graph Index for Vector Search）图检索算法demo
#include <faiss/ascend/AscendIndexCagra.h>
#include <gtest/gtest.h>
#include <sys/time.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace
{
unsigned int g_seed;
const int FAST_RAND_MAX = 0x7FFF;
const int MILLI_SECOND = 1000;

inline double GetMillisecs()
{
    struct timeval tv = {0, 0};
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

inline int FastRand()
{
    const int multiplyNum = 214013;
    const int addNum = 2531011;
    const int rshiftNum = 16;
    g_seed = (multiplyNum * g_seed + addNum);
    return (g_seed >> rshiftNum) & FAST_RAND_MAX;
}

void Norm(float *data, size_t n, int dim)
{
#pragma omp parallel for if (n > 100)
    for (size_t i = 0; i < n; ++i)
    {
        float l2norm = 0;
        for (int j = 0; j < dim; ++j)
        {
            l2norm += data[i * dim + j] * data[i * dim + j];
        }
        l2norm = std::sqrt(l2norm);
        if (fabs(l2norm) < FLT_EPSILON)
        {
            std::cerr << "Error: Invalid l2norm value." << std::endl;
            continue;
        }
        for (int j = 0; j < dim; ++j)
        {
            data[i * dim + j] = data[i * dim + j] / l2norm;
        }
    }
}

TEST(TestAscendIndexCagra, QPS)
{
    int dim = 128;
    int ntotal = 10000;
    int graphDegree = 64;
    int topK = 32;
    try
    {
        std::vector<int> deviceList = {0};
        faiss::ascend::AscendIndexCagra index;
        auto ret = index.Init(dim, graphDegree, ntotal, topK, deviceList);
        ASSERT_EQ(ret, 0);

        // 生成随机数据
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> distFloat(0.0f, 1.0f);
        std::uniform_int_distribution<uint32_t> distUint(0, static_cast<uint32_t>(ntotal - 1));

        std::vector<float> baseData(ntotal * dim);
        for (auto &v : baseData)
        {
            v = distFloat(rng);
        }
        Norm(baseData.data(), ntotal, dim);

        std::vector<uint32_t> graphData(ntotal * graphDegree);
        for (auto &v : graphData)
        {
            v = distUint(rng);
        }
        std::vector<uint32_t> hashData(ntotal * 2);
        for (auto &v : hashData)
        {
            v = distUint(rng);
        }

        ret = index.Add(graphData.data(), hashData.data(), baseData.data());
        EXPECT_EQ(ret, 0);
        printf("add done\n");

        int queryNum = 64;
        std::vector<float> queryData(queryNum * dim);
        for (auto &v : queryData)
        {
            v = distFloat(rng);
        }
        Norm(queryData.data(), queryNum, dim);

        ret = index.QuantizeData(queryNum, queryData.data(), ntotal, baseData.data());
        EXPECT_EQ(ret, 0);
        printf("quantize done\n");

        // 预热
        std::vector<float> distw(queryNum * topK, 0);
        std::vector<uint32_t> labelw(queryNum * topK, 0);
        for (int i = 0; i < 5; i++)
        {
            index.Search(queryNum, queryData.data(), topK, distw.data(), labelw.data());
        }

        // QPS 测试
        int loopTimes = 100;
        std::vector<float> dist(queryNum * topK, 0);
        std::vector<uint32_t> label(queryNum * topK, 0);
        double ts = GetMillisecs();
        for (int i = 0; i < loopTimes; i++)
        {
            index.Search(queryNum, queryData.data(), topK, dist.data(), label.data());
        }
        double te = GetMillisecs();
        printf("CAGRA QPS test: base=%d, dim=%d, graphDegree=%d, topK=%d, queryNum=%d, QPS=%.4f\n", ntotal, dim,
               graphDegree, topK, queryNum, MILLI_SECOND * queryNum * loopTimes / (te - ts));
    }
    catch (std::exception &e)
    {
        FAIL() << "Exception occurred: " << e.what();
    }
}

}  // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
