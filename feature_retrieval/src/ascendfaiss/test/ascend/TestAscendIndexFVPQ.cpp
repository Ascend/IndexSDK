/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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


#include <numeric>
#include <cmath>
#include <random>
#include <cstring>
#include <sys/time.h>

#include <gtest/gtest.h>
#include <faiss/IndexPQ.h>
#include <faiss/index_io.h>

#include <faiss/ascend/fv/AscendIndexFVPQ.h>

namespace {
inline double GetMillisecs()
{
    struct timeval tv = { 0, 0 };
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

inline void AssertEqual(const std::vector<float> &gt, const std::vector<float> &data)
{
    const float epson = 1e-3;
    ASSERT_EQ(gt.size(), data.size());
    for (size_t i = 0; i < gt.size(); i++) {
        ASSERT_TRUE(fabs(gt[i] - data[i]) <= epson) << "i: " << i << " gt: " << gt[i] << " data: " << data[i] <<
            std::endl;
    }
}

namespace {
unsigned int g_seed;
const int FAST_RAND_MAX = 0x7FFF;
}

inline void FastSrand(int seed)
{
    g_seed = seed;
}

// Compute a pseudorandom integer.
// Output value in range [0, 32767]
inline int FastRand(void)
{
    const int mutipliyNum = 214013;
    const int addNum = 2531011;
    const int rshiftNum = 16;
    g_seed = (mutipliyNum * g_seed + addNum);
    return (g_seed >> rshiftNum) & FAST_RAND_MAX;
}

TEST(TestAscendIndexFVPQ, QPS)
{
    int dim = 256;
    size_t ntotal = 1000000;
    size_t ntrain = 100000;
    size_t maxSize = ntotal * dim;

    std::vector<float> data(maxSize);
    for (size_t i = 0; i < maxSize; i++) {
        data[i] = 1.0 * FastRand() / FAST_RAND_MAX;
    }

    std::vector<float> trainData(ntrain * dim);
    for (size_t i = 0; i < ntrain * dim; i++) {
        trainData[i] = 1.0 * FastRand() / FAST_RAND_MAX;
    }

    faiss::ascend::AscendIndexFVPQConfig conf({ 0 });
    faiss::ascend::AscendIndexFVPQ index(dim, faiss::METRIC_L2, conf);

    index.train(ntrain, trainData.data());
    index.add(ntotal, data.data());

    std::vector<int> searchNum = { 1, 2, 4, 8, 16, 32, 64, 128, 1000 };
    for (size_t n = 0; n < searchNum.size(); n++) {
        int k = 100;
        int loopTimes = 100;
        std::vector<float> dist(searchNum[n] * k, 0);
        std::vector<faiss::idx_t> label(searchNum[n] * k, 0);
        double ts = GetMillisecs();
        for (int i = 0; i < loopTimes; i++) {
            index.search(searchNum[n], data.data(), k, dist.data(), label.data());
        }
        double te = GetMillisecs();
        printf("case[%zu]: base:%zu, dim:%d, search num:%d, QPS:%.4f\n", n, ntotal, dim, searchNum[n],
            1000 * searchNum[n] * loopTimes / (te - ts));
    }
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
