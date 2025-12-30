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
#include <gtest/gtest.h>
#include <faiss/ascend/custom/AscendIndexFlatAT.h>
#include <faiss/ascend/AscendCloner.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <cstring>
#include <sys/time.h>
#include <faiss/index_io.h>
#include <cstdlib>

namespace {
const int DIM = 64;
const int K = 1;
const size_t BASE_SIZE = 8192;
const std::vector<int> DEVICES = { 1, 3 };

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

TEST(TestAscendIndexFlatAT, All)
{
    std::vector<float> data(DIM * BASE_SIZE);
    for (size_t i = 0; i < DIM * BASE_SIZE; i++) {
        data[i] = drand48();
    }

    faiss::ascend::AscendIndexFlatATConfig conf(DEVICES);
    faiss::ascend::AscendIndexFlatAT index(DIM, BASE_SIZE, conf);

    index.add(BASE_SIZE, data.data());
    EXPECT_EQ(index.ntotal, BASE_SIZE);

    {
        int batch = 32;
        for (size_t i = BASE_SIZE - batch; i < BASE_SIZE; i += batch) {
            std::vector<float> dist(K * batch, 0);
            std::vector<faiss::idx_t> label(K * batch, 0);
            index.search(batch, data.data() + i * DIM, K, dist.data(), label.data());
            ASSERT_EQ(label[0], i);
            ASSERT_EQ(label[K], i + 1);
            ASSERT_EQ(label[K * 2], i + 2);
            ASSERT_EQ(label[K * 3], i + 3);
            faiss::idx_t assign;
            index.assign(1, data.data() + i * DIM, &assign);
            ASSERT_EQ(assign, i);
        }
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

TEST(TestAscendIndexFlatAT, PRECISION)
{
    size_t num = 250 * 10000;
    size_t maxSize = num * DIM;

    std::vector<float> data(maxSize);
    for (size_t i = 0; i < maxSize; i++) {
        data[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX));
    }

    for (int i = 0; i < 100; i++) {
        printf("%f ", data[i]);
    }

    faiss::ascend::AscendIndexFlatATConfig conf(DEVICES, 768 * 1024 * 1024);
    faiss::ascend::AscendIndexFlatAT index(DIM, BASE_SIZE, conf);

    faiss::IndexFlatL2 cpuIndex(DIM);
    cpuIndex.add(BASE_SIZE, data.data());
    EXPECT_EQ(cpuIndex.ntotal, BASE_SIZE);

    index.add(BASE_SIZE, data.data());
    index.reset();
    index.add(BASE_SIZE, data.data());
    EXPECT_EQ(index.ntotal, BASE_SIZE);

    std::vector<size_t> searchNum = { num / 100 };
    // Since the qps might have some tricky issue
    for (size_t n = 0; n < searchNum.size(); n++) {
        printf("\n\nBatch size %ld\n", searchNum[n]);
        std::vector<float> dist(searchNum[n] * K, 0);
        std::vector<faiss::idx_t> label(searchNum[n] * K, 0);

        std::vector<float> dist__(searchNum[n] * K, 0);
        std::vector<faiss::idx_t> label__(searchNum[n] * K, 0);
        index.search(searchNum[n], data.data(), K, dist.data(), label.data());
        cpuIndex.search(searchNum[n], data.data(), K, dist__.data(), label__.data());

        int wrong = 0;
        float errNpu = 0;
        float errCpu = 0;
        for (size_t i = 0; i < searchNum[n]; i++) {
            for (int j = 0; j < K; ++j) {
                if (label[i * K + j] != label__[i * K + j]) {
                    wrong++;
                }
                errNpu += dist[i * K + j];
                errCpu += dist__[i * K + j];
                if (std::abs(dist[i * K + j] - dist__[i * K + j]) > 0.1) {
                    printf("j=%d, cpuIndex=%f, AscendIndex=%f\n", j, dist[i + j], dist__[i + j]);
                    if (i < 10) {
                        printf("for %ld cpu-dist %f, npu-dist %f \n", i, dist__[i], dist[i]);
                    }
                }
            }
        }
        printf("WRONG / TOTAL = %d / %ld\n", wrong, num / 100);
        printf("error rate= %f\n", 1.0 * wrong * 100 / num);
        printf("errNpu: %f errCpu: %f\n", errNpu, errCpu);
    }
}

TEST(TestAscendIndexFlatAT, ASCEND_QPS)
{
    size_t num = 2500 * 10000;
    size_t maxSize = num * DIM;

    std::vector<float> data(maxSize);
    for (size_t i = 0; i < maxSize; i++) {
        data[i] = 1.0 * FastRand() / FAST_RAND_MAX;
    }

    faiss::ascend::AscendIndexFlatATConfig conf(DEVICES, 768 * 1024 * 1024);
    faiss::ascend::AscendIndexFlatAT index(DIM, BASE_SIZE, conf);

    index.add(BASE_SIZE, data.data());
    EXPECT_EQ(index.ntotal, BASE_SIZE);

    std::vector<size_t> searchNum = { num };
    // Since the qps might have some tricky issue
    for (size_t n = 0; n < searchNum.size(); n++) {
        std::vector<float> dist(searchNum[n] * K, 0);
        std::vector<faiss::idx_t> label(searchNum[n] * K, 0);

        double ts = GetMillisecs();
        index.search(searchNum[n], data.data(), K, dist.data(), label.data());
        double te = GetMillisecs();

        double qps = searchNum[n] / (te - ts) * 1000;
        printf("AscendIndex case[%zu]: base:%zu, dim:%d, search num:%ld, QPS:%.4f\n",
            n, BASE_SIZE, DIM, searchNum[n], qps);
    }
}

TEST(TestAscendIndexFlatAT, CPU_QPS)
{
    size_t num = 2500 * 10000;
    size_t maxSize = num * DIM;

    std::vector<float> data(maxSize);
    for (size_t i = 0; i < maxSize; i++) {
        data[i] = 1.0 * FastRand() / FAST_RAND_MAX;
    }

    faiss::IndexFlatL2 cpuIndex(DIM);
    cpuIndex.add(BASE_SIZE, data.data());

    EXPECT_EQ(cpuIndex.ntotal, BASE_SIZE);

    std::vector<size_t> searchNum = { num };
    // Since the qps might have some tricky issue
    for (size_t n = 0; n < searchNum.size(); n++) {
        std::vector<float> dist(searchNum[n] * K, 0);
        std::vector<faiss::idx_t> label(searchNum[n] * K, 0);

        double ts = GetMillisecs();
        cpuIndex.search(searchNum[n], data.data(), K, dist.data(), label.data());
        double te = GetMillisecs();

        double qps = searchNum[n] / (te - ts) * 1000;
        printf("CpuIndex: case[%zu]: base:%zu, dim:%d, k:%d, search num:%ld, QPS:%.4f\n",
                n, BASE_SIZE, DIM, K, searchNum[n], qps);
    }
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
