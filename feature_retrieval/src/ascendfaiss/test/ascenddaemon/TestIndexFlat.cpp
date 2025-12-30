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


#include <vector>
#include <iostream>
#include <algorithm>
#include <limits>
#include <memory>

#include <cstdlib>
#include <cmath>
#include <unistd.h>
#include <sys/time.h>

#include <arm_fp16.h>
#include <gtest/gtest.h>

#include <securec.h>

#include <ascenddaemon/impl/AuxIndexStructures.h>
#include <ascenddaemon/utils/AscendUtils.h>
#include <ascenddaemon/utils/StaticUtils.h>
#include <ascenddaemon/impl/IndexFlatL2.h>
#include <ascenddaemon/utils/AscendTensor.h>
#include <ascenddaemon/utils/DeviceVector.h>

namespace {
unsigned int g_seed;
const int FAST_RAND_MAX = 0x7FFF;

static double Elapsed()
{
    struct timeval tv {0};
    gettimeofday (&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Used to seed the generator.
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

inline int GetDataSize(const std::vector<std::unique_ptr<ascend::DeviceVector<float16_t>>>& datas)
{
    if (datas.empty()) {
        return 0;
    }

    int ret = 0;
    for (auto &elem : datas) {
        ret += elem->size();
    }
    return ret;
}

TEST(IndexFlat, Add)
{
    srand(0);
    // create Index
    const int dim = 512;
    ascend::IndexFlatL2 indexFlat(dim);
    indexFlat.init();
    const int distComputeBatchSize = indexFlat.getBlockSize();

    // add 1, will resize
    int total = 0;
    int addNum0 = 1;
    ascend::AscendTensor<float16_t, 2> addVec0({addNum0, dim});
    for (int i = 0; i < addNum0; i++) {
        for (int j = 0; j < dim; j++) {
            *(addVec0[i][j].data()) = i * dim + j;
        }
    }
    indexFlat.addVectors(addVec0);
    total += addNum0;

    const auto& normBase = indexFlat.getNormBase();
    const auto& baseShaped = indexFlat.getBaseShaped();

    EXPECT_EQ(indexFlat.getSize(), total);
    EXPECT_EQ(GetDataSize(normBase), ascend::utils::divUp(total, distComputeBatchSize) * distComputeBatchSize);
    EXPECT_EQ(GetDataSize(baseShaped), ascend::utils::divUp(total, distComputeBatchSize) * distComputeBatchSize * dim);

    // add distComputeBatchSize, will resize
    int addNum1 = distComputeBatchSize;
    ascend::AscendTensor<float16_t, 2> addVec1({addNum1, dim});
    for (int i = 0; i < addNum1; i++) {
        for (int j = 0; j < dim; j++) {
            *(addVec1[i][j].data()) = (i * dim + j) % 32768;
        }
    }
    indexFlat.addVectors(addVec1);
    total += addNum1;

    EXPECT_EQ(indexFlat.getSize(), total);
    EXPECT_EQ(GetDataSize(normBase), ascend::utils::divUp(total, distComputeBatchSize) * distComputeBatchSize);
    EXPECT_EQ(GetDataSize(baseShaped), ascend::utils::divUp(total, distComputeBatchSize) * distComputeBatchSize * dim);

    // add distComputeBatchSize - 1, will not resize
    int addNum2 = distComputeBatchSize - addNum0;
    ascend::AscendTensor<float16_t, 2> addVec2({addNum2, dim});
    indexFlat.addVectors(addVec2);
    total += addNum2;

    EXPECT_EQ(indexFlat.getSize(), total);
    EXPECT_EQ(GetDataSize(normBase), ascend::utils::divUp(total, distComputeBatchSize) * distComputeBatchSize);
    EXPECT_EQ(GetDataSize(baseShaped), ascend::utils::divUp(total, distComputeBatchSize) * distComputeBatchSize * dim);

    // reset
    indexFlat.reset();
    EXPECT_EQ(indexFlat.getSize(), 0);
    EXPECT_EQ(GetDataSize(normBase), 0);
    EXPECT_EQ(GetDataSize(baseShaped), 0);
}

TEST(IndexFlat, AddBatch)
{
    srand(0);
    // create Index
    const int dim = 512;
    ascend::IndexFlatL2 indexFlat(dim);
    indexFlat.init();
    const int distComputeBatchSize = indexFlat.getBlockSize();
    const auto& normBase = indexFlat.getNormBase();
    const auto& baseShaped = indexFlat.getBaseShaped();

    const int total = 1500000;
    const int batch = distComputeBatchSize - 1;
    ascend::AscendTensor<float16_t, 2> addVec0({total, dim});
    for (int i = 0; i < total; i++) {
        for (int j = 0; j < dim; j++) {
            *(addVec0[i][j].data()) = 1.0 * FastRand() / FAST_RAND_MAX;
        }
    }

    for (int i = 0; i < ascend::utils::divUp(total, batch); i++) {
        int addNum = std::min(batch, total - i * batch);
        ascend::AscendTensor<float16_t, 2> addVec(addVec0.data(), {addNum, dim});
        indexFlat.addVectors(addVec);
        int expectSize = i * batch + addNum;
        ASSERT_EQ(indexFlat.getSize(), expectSize);
        EXPECT_EQ(GetDataSize(normBase), ascend::utils::divUp(expectSize, distComputeBatchSize) * distComputeBatchSize);
        EXPECT_EQ(GetDataSize(baseShaped),
            ascend::utils::divUp(expectSize, distComputeBatchSize) * distComputeBatchSize * dim);
    }

    // reset
    indexFlat.reset();
}

TEST(IndexFlat, Get)
{
    FastSrand(0);
    const int dim = 512;
    ascend::IndexFlatL2 indexFlat(dim);
    indexFlat.init();
    const int distComputeBatchSize = indexFlat.getBlockSize();

    // get empty base
    std::vector<float16_t> result;
    indexFlat.getVectors(0, distComputeBatchSize, result);
    ASSERT_EQ(result.size(), 0);

    int remain = 1;
    int addNum = distComputeBatchSize + remain;
    ascend::AscendTensor<float16_t, 2> addVec0({addNum, dim});
    for (int i = 0; i < addNum; i++) {
        for (int j = 0; j < dim; j++) {
            *(addVec0[i][j].data()) = 1.0 * FastRand() / FAST_RAND_MAX;
        }
    }
    indexFlat.addVectors(addVec0);

    // get all
    result.clear();
    indexFlat.getVectors(0, addNum, result);
    ASSERT_EQ(result.size(), addNum * dim);
    for (int i = 0; i < addNum * dim; i++) {
        ASSERT_FLOAT_EQ(result[i], *(addVec0.data() + i));
    }

    // get more than add num
    result.clear();
    indexFlat.getVectors(0, addNum + 1, result);
    ASSERT_EQ(result.size(), addNum * dim);
    for (int i = 0; i < addNum * dim; i++) {
        if (fabs(result[i] - *(addVec0.data() + i)) > 1e-4) {
            std::cout << "error" << std::endl;
        }
        ASSERT_FLOAT_EQ(result[i], *(addVec0.data() + i));
    }

    // offset >= total
    result.clear();
    indexFlat.getVectors(addNum, distComputeBatchSize, result);
    ASSERT_EQ(result.size(), 0);

    // offset + num > total
    result.clear();
    indexFlat.getVectors(distComputeBatchSize, distComputeBatchSize, result);
    ASSERT_EQ(result.size(), remain * dim);
    for (int i = 0; i < remain * dim; i++) {
        ASSERT_FLOAT_EQ(result[i], *(addVec0.data() + distComputeBatchSize * dim + i));
    }
}

TEST(IndexFlat, GetLimit)
{
    FastSrand(0);
    const int dim = 512;
    ascend::IndexFlatL2 indexFlat(dim);
    indexFlat.init();

    const int addNum = 2000000;
    ascend::AscendTensor<float16_t, 2> addVec0({addNum, dim});
    for (int i = 0; i < addNum; i++) {
        for (int j = 0; j < dim; j++) {
            *(addVec0[i][j].data()) = 1.0 * FastRand() / FAST_RAND_MAX;
        }
    }
    indexFlat.addVectors(addVec0);

    // get all
    const int getPageSize = 16384;
    std::vector<float16_t> result;
    int offset = 0;
    for (int i = 0; i < ascend::utils::divUp(addNum, getPageSize); i++) {
        int num = std::min(getPageSize, addNum - i * getPageSize);
        indexFlat.getVectors(offset, getPageSize, result);
        ASSERT_EQ(result.size(), num * dim);
        for (int j = 0; j < num * dim; j++) {
            ASSERT_FLOAT_EQ(result[i], *(addVec0.data() + offset * dim + i)) << "(" << i << "," << j << ")";
        }

        result.clear();
        offset += num;
    }
}

TEST(IndexFlat, SearchBasic)
{
    FastSrand(0);
    // Create Index
    const int dim = 512;
    ascend::IndexFlatL2 indexFlat(dim);
    indexFlat.init();
    const int distComputeBatchSize = indexFlat.getBlockSize();

    // Add
    int addNum0 = distComputeBatchSize + 1;
    ascend::AscendTensor<float16_t, 2> addVec0({addNum0, dim});
    for (int i = 0; i < addNum0; i++) {
        for (int j = 0; j < dim; j++) {
            addVec0[i][j] = 1.0 * FastRand() / FAST_RAND_MAX;
        }
    }
    indexFlat.addVectors(addVec0);

    // Search
    const int nqMax = 1000;
    const int k = 1000;
    std::vector<int> queryNum {1, 10, 31, 100, 1000};
    std::vector<float16_t> queries(addVec0.data(), addVec0.data() + nqMax * dim);
    for (auto nq: queryNum) {
        std::vector<float16_t> distances(nq * k);
        std::vector<ascend::Index::idx_t> labels(nq * k);
        indexFlat.search(nq, queries.data(), k, distances.data(), labels.data());
        for (int i = 0; i < nq; i++) {
            EXPECT_EQ(labels[i * k], i);
        }
    }
}


TEST(IndexFlat, SearchLimit)
{
    FastSrand(0);
    // Create Index
    const int dim = 512;
    ascend::IndexFlatL2 indexFlat(dim);
    indexFlat.init();

    // Add
    const int maxBaseSize = 2000000;
    ascend::AscendTensor<float16_t, 2> addVec({maxBaseSize, dim});
    for (int i = 0; i < maxBaseSize; i++) {
        for (int j = 0; j < dim; j++) {
            addVec[i][j] = 1.0 * FastRand() / FAST_RAND_MAX;
        }
    }

    const int nq = 1000;
    const int k = 1000;
    const std::vector<int> baseSizes {
        10000, 50000, 100000, 500000, 1000000, 1500000, maxBaseSize};
    for (auto baseSize : baseSizes) {
        std::cout << "Test base size " << baseSize << "\n";
        ascend::AscendTensor<float16_t, 2> base(addVec.data(), {baseSize, dim});
        indexFlat.addVectors(base);

        const int offset = baseSize / 3;
        const std::vector<float16_t> queries(addVec.data() + offset * dim,
            addVec.data() + offset * dim + nq * dim);
        std::vector<float16_t> distances(nq * k);
        std::vector<ascend::Index::idx_t> labels(nq * k);
        indexFlat.search(nq, queries.data(), k, distances.data(), labels.data());
        for (int i = 0; i < nq; i++) {
            if (labels[i * k] != ascend::Index::idx_t(offset + i)) {
                std::cout << "error" << std::endl;
            }
            ASSERT_EQ(labels[i * k], offset + i);
        }
        indexFlat.reset();
    }
}

TEST(IndexFlat, RemoveRangeBasic)
{
    FastSrand(0);
    // Create Index
    const int dim = 512;
    ascend::IndexFlatL2 indexFlat(dim);
    indexFlat.init();
    const int distComputeBatchSize = indexFlat.getBlockSize();
    const auto& normBase = indexFlat.getNormBase();
    const auto& baseShaped = indexFlat.getBaseShaped();
    int total;

    // add
    int addNum0 = distComputeBatchSize + 2;
    ascend::AscendTensor<float16_t, 2> addVec0({addNum0, dim});
    for (int i = 0; i < addNum0; i++) {
        for (int j = 0; j < dim; j++) {
            addVec0[i][j] = 1.0 * FastRand() / FAST_RAND_MAX;
        }
    }
    total = addNum0;
    indexFlat.addVectors(addVec0);
    ASSERT_EQ(indexFlat.getSize(), total);

    // remove first
    ascend::IDSelectorRange removeRange(0, 1);
    indexFlat.removeIds(removeRange);
    total -= 1;
    ASSERT_EQ(indexFlat.getSize(), total);
    ASSERT_EQ(GetDataSize(normBase), ascend::utils::divUp(total, distComputeBatchSize) * distComputeBatchSize);
    ASSERT_EQ(GetDataSize(baseShaped), ascend::utils::divUp(total, distComputeBatchSize) * distComputeBatchSize * dim);

    // [1, 10) not change
    int nq = 10;
    int k = 100;
    std::vector<float16_t> queries(addVec0.data(), addVec0.data() + nq * dim);
    std::vector<float16_t> distances(nq * k);
    std::vector<ascend::Index::idx_t> labels(nq * k);
    indexFlat.search(nq, queries.data(), k, distances.data(), labels.data());
    for (int i = 1; i < nq; i++) {
        // move forward by 1
        EXPECT_EQ(labels[i * k], i);
    }

    // remove last
    removeRange = ascend::IDSelectorRange(total - 1, total);
    indexFlat.removeIds(removeRange);
    total -= 1;
    ASSERT_EQ(indexFlat.getSize(), total);
    ASSERT_EQ(GetDataSize(normBase), ascend::utils::divUp(total, distComputeBatchSize) * distComputeBatchSize);
    ASSERT_EQ(GetDataSize(baseShaped), ascend::utils::divUp(total, distComputeBatchSize) * distComputeBatchSize * dim);

    queries.assign(addVec0.end() - nq * dim, addVec0.end());
    indexFlat.search(nq, queries.data(), k, distances.data(), labels.data());
    for (int i = 0; i < nq - 2; i++) {
        if (labels[i * k] != ascend::Index::idx_t(addNum0 - nq + i)) {
            std::cout << "error" << std::endl;
        }
        EXPECT_EQ(labels[i * k], addNum0 - nq + i);
    }

    // remove [5, 10), which is original [6, 11)
    const int imin = 5;
    const int imax = 10;
    removeRange = ascend::IDSelectorRange(imin, imax);
    indexFlat.removeIds(removeRange);
    total -= (imax - imin);
    ASSERT_EQ(indexFlat.getSize(), total);
    ASSERT_EQ(GetDataSize(normBase), ascend::utils::divUp(total, distComputeBatchSize) * distComputeBatchSize);
    ASSERT_EQ(GetDataSize(baseShaped), ascend::utils::divUp(total, distComputeBatchSize) * distComputeBatchSize * dim);

    nq = 20;
    distances.resize(nq * k);
    labels.resize(nq * k);
    queries.assign(addVec0.data(), addVec0.data() + nq * dim);
    indexFlat.search(nq, queries.data(), k, distances.data(), labels.data());
    for (int i = 1; i < nq; i++) {
        if (i < 5 || i > 10) {
            EXPECT_EQ(labels[i * k], i);
        }
    }
}

TEST(IndexFlat, RemoveBatchBasic)
{
    FastSrand(0);
    // Create Index
    const int dim = 512;
    ascend::IndexFlatL2 indexFlat(dim);
    indexFlat.init();
    const int distComputeBatchSize = indexFlat.getBlockSize();

    // add
    int addNum0 = distComputeBatchSize * 2;
    ascend::AscendTensor<float16_t, 2> addVec0({addNum0, dim});
    for (int i = 0; i < addNum0; i++) {
        for (int j = 0; j < dim; j++) {
            addVec0[i][j] = 1.0 * FastRand() / FAST_RAND_MAX;
        }
    }
    indexFlat.addVectors(addVec0);
    ASSERT_EQ(indexFlat.getSize(), addNum0);

    // remove vectors with even number
    std::vector<ascend::Index::idx_t> toRemove;
    for (int i = 0; i < distComputeBatchSize; i++) {
        if (i % 2 == 0) {
            toRemove.push_back(static_cast<ascend::Index::idx_t>(i));
        }
    }

    ascend::IDSelectorBatch removeBatch(toRemove.size(), toRemove.data());
    auto removed = indexFlat.removeIds(removeBatch);
    ASSERT_EQ(removed, toRemove.size());
    ASSERT_EQ(indexFlat.getSize(), addNum0 - distComputeBatchSize / 2);
}


/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/
static std::unique_ptr<float[]> FvecsRead(const std::string &fname, size_t &d_out, size_t &n_out)
{
    FILE *f = fopen(fname.c_str(), "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname.c_str());
        perror("");
        abort();
    }
    int d;
    size_t nr = fread(&d, 1, sizeof(int), f);
    static_cast<void>(nr);  // unused variable nr
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    size_t n = sz / ((d + 1) * sizeof(float));

    d_out = d;
    n_out = n;
    auto x = std::make_unique<float[]>(n * (d + 1));
    float *pX = x.get();
    nr = fread(pX, sizeof(float), n * (d + 1), f);
    static_cast<void>(nr);  // unused variable nr

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++) {
        auto err = memmove_s(pX + i * d, (n * (d + 1) - i * d) * sizeof(*pX),
            pX + 1 + i * (d + 1), d * sizeof(*pX));
        ASCEND_THROW_IF_NOT(err == EOK);
    }

    fclose(f);
    return x;
}

/*
 * To run this testcase, download the ANN_SIFT1M dataset from
 * http://corpus-texmex.irisa.fr/
 * and unzip it to the subdirectory sift1M.
 */
TEST(IndexFlat, SIFT1M)
{
    double t0 = Elapsed();

    // check whether Sift1M exists
    if (access("sift1M", F_OK) == 0) {
        printf ("[%.3f s] Loading database\n", Elapsed() - t0);
        size_t nb, dim;
        auto pBase = FvecsRead("sift1M/sift_base.fvecs", dim, nb);
        const float *xb = pBase.get();

        printf ("[%.3f s] Preparing indexFlat, dim=%ld\n", Elapsed() - t0, dim);
        ascend::IndexFlatL2 indexFlat(dim);
        indexFlat.init();

        printf ("[%.3f s] Indexing database, size (%ld,%ld)\n",
            Elapsed() - t0, nb, dim);
        std::vector<float16_t> buf(nb * dim);
        std::transform(xb, xb + nb * dim, buf.data(), [](const float& from) {
            return static_cast<float16_t>(from / 128.0);
        });
        ascend::AscendTensor<float16_t, 2> addVec(buf.data(),
            {static_cast<int>(nb), static_cast<int>(dim)});
        indexFlat.addVectors(addVec);

        printf ("[%.3f s] Loading queries\n", Elapsed() - t0);
        size_t nq;
        size_t d2;
        auto pQuery = FvecsRead("sift1M/sift_query.fvecs", d2, nq);
        const float *xq = pQuery.get();
        std::vector<float16_t> queries(nq * dim);
        std::transform(xq, xq + nq * dim, queries.data(), [](const float& from) {
            return static_cast<float16_t>(from / 128.0);
        });

        printf ("[%.3f s] Loading ground truth for %ld queries\n",
                Elapsed() - t0, nq);
        size_t k; // nb of results per query in the GT
        size_t nq2;
        // not very clean, but works as long as sizeof(int) == sizeof(float)
        auto pGT = FvecsRead("sift1M/sift_groundtruth.ivecs", k, nq2);
        auto gt = reinterpret_cast<const ascend::Index::idx_t *>(pGT.get());

        int searchK = 100;
        nq = 10000;

        {
            printf ("[%.3f s] Perform a search on %ld queries\n",
                    Elapsed() - t0, nq);

            // output buffers
            std::vector<ascend::Index::idx_t> indices(nq * searchK);
            std::vector<float16_t> distances(nq * searchK);

            double startTime = Elapsed();
            indexFlat.search(nq, queries.data(), searchK, distances.data(), indices.data());
            double costTime = Elapsed() - startTime;
            printf ("[%.3f s] Compute recalls, qps = %.3f\n", Elapsed() - t0, (nq * 1.0 / costTime));

            // evaluate result by hand.
            int n1 = 0, n10 = 0, n100 = 0;
            for (size_t i = 0; i < nq; i++) {
                ascend::Index::idx_t gtNn = gt[i * k];
                for (size_t j = 0; j < k; j++) {
                    if (indices[i * k + j] == gtNn) {
                        if (j < 1) {
                            n1++;
                        }
                        if (j < 2) {
                            n10++;
                        }
                        if (j < 100) {
                            n100++;
                        }
                    }
                }
            }
            printf("R@1 = %.4f\n", n1 / static_cast<float>(nq));
            printf("R@10 = %.4f\n", n10 / static_cast<float>(nq));
            printf("R@100 = %.4f\n", n100 / static_cast<float>(nq));
        }
    }
}


TEST(IndexFlat, QPSComparisonDim256)
{
    FastSrand(0);
    double t0 = Elapsed();
    const int dim = 512;
    ascend::IndexFlatL2 indexFlat(dim);
    indexFlat.init();
    const int unit = 10000;
    const std::vector<int> baseSizes {25 * unit};
    const std::vector<int> ks {100, 1000, 5000};

    std::vector<float16_t> addVec(25 * unit * dim);
    std::for_each(addVec.begin(), addVec.end(), [] (float16_t& v) {
        v = 1.0 * FastRand() / FAST_RAND_MAX;
    });

    for (auto size : baseSizes) {
        ascend::AscendTensor<float16_t, 2> data(addVec.data(), {size, dim});
        for (int i = 0; i < 1; ++i) {
            printf("[%.6f s] Add %d vectors\n", Elapsed() - t0, size);
            indexFlat.addVectors(data);
            printf("[%.6f s] done\n", Elapsed() - t0);
        }

        std::vector<int> queryNumVec {1, 2, 4, 8, 16, 32, 64, 128, 1000};
        for (auto queryNum : queryNumVec) {
            ascend::AscendTensor<float16_t, 2> queries(addVec.data(), {queryNum, dim});
            printf("[%.6f s] --- query batch size = %d ---\n", Elapsed() - t0, queryNum);
            for (auto k : ks) {
                std::vector<float16_t> dist(queryNum * k);
                std::vector<ascend::Index::idx_t> ids(queryNum * k);
                double startTime = Elapsed() - t0;
                printf("[%.6f s] queryNum = %d, k = %d\n", startTime, queryNum, k);
                indexFlat.search(queryNum, queries.data(), k, dist.data(), ids.data());
                double endTime = Elapsed() - t0;
                printf("[%.6f s] done, QPS = %.1f\n", endTime, queryNum * 1.0 / (endTime - startTime));
            }
        }

        printf("[%.6f s] reset index\n", Elapsed() - t0);
        indexFlat.reset();
        printf("[%.6f s] done\n\n", Elapsed() - t0);
    }
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    ascend::DeviceScope deviceScope;
    return RUN_ALL_TESTS();
}
