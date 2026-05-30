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

// 需要生成aicpu算子+sq8(-d 256)

#include <faiss/ascend/AscendCloner.h>
#include <faiss/ascend/AscendIndexSQ.h>
#include <faiss/ascend/AscendMultiIndexSearch.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/index_io.h>
#include <gtest/gtest.h>
#include <sys/time.h>

#include <cfloat>
#include <numeric>
#include <random>

namespace
{
const auto METRIC_TYPE = faiss::METRIC_INNER_PRODUCT;
const auto DIM = 256;
const int NTOTAL = 1000000;
const std::vector<int> DEVICE_IDS = {0};
unsigned int g_seed;
const int FAST_RAND_MAX = 0x7FFF;
const int MILLI_SECOND = 1000;
inline void FastSrand(int seed) { g_seed = seed; }

inline int FastRand(void)
{
    const int mutipliyNum = 214013;
    const int addNum = 2531011;
    const int rshiftNum = 16;
    g_seed = (mutipliyNum * g_seed + addNum);
    return (g_seed >> rshiftNum) & FAST_RAND_MAX;
}

inline double GetMillisecs()
{
    struct timeval tv = {0, 0};
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

inline void AssertEqual(std::vector<uint8_t> &gt, std::vector<uint8_t> &data)
{
    ASSERT_EQ(gt.size(), data.size());
    for (size_t i = 0; i < gt.size(); i++)
    {
        ASSERT_EQ(gt[i], data[i]);
    }
}

void Norm(float *data, size_t n, size_t dim)
{
#pragma omp parallel for if (n > 1)
    for (size_t i = 0; i < n; ++i)
    {
        float l2norm = 0.0;
        for (size_t j = 0; j < dim; ++j)
        {
            l2norm += data[i * dim + j] * data[i * dim + j];
        }
        l2norm = std::sqrt(l2norm);
        if (fabs(l2norm) < FLT_EPSILON)
        {
            std::cerr << "Error: Invalid l2norm value." << std::endl;
        }
        for (size_t j = 0; j < dim; ++j)
        {
            data[i * dim + j] = data[i * dim + j] / l2norm;
        }
    }
}

void MultiSearch(int indexNum, std::vector<std::vector<float>> &data,
                 std::vector<faiss::ascend::AscendIndex *> &indexes, std::vector<int> &searchNum)
{
    int k = 5;
    int loopTimes = 100;

    const int warmupNum = 15;
    std::vector<float> distWM(warmupNum * k, 0);
    std::vector<faiss::idx_t> labelWM(warmupNum * k, 0);

    for (int i = 0; i < indexNum; i++)
    {
        indexes[i]->search(warmupNum, data[0].data(), k, distWM.data(), labelWM.data());
    }

    for (size_t n = 0; n < searchNum.size(); n++)
    {
        std::vector<float> dist(searchNum[n] * k, 0);
        std::vector<faiss::idx_t> label(searchNum[n] * k, 0);

        double ts = GetMillisecs();
        for (int l = 0; l < loopTimes; l++)
        {
            for (int i = 0; i < indexNum; i++)
            {
                indexes[i]->search(searchNum[n], data[0].data(), k, dist.data(), label.data());
            }
        }
        double te = GetMillisecs();
        printf("multi search: false, index num: %d, k: %d, base: %d, dim: %d, search num: %2d, QPS: %9.4f \n", indexNum,
               k, NTOTAL, DIM, searchNum[n], MILLI_SECOND * searchNum[n] * loopTimes / (te - ts));
    }

    // multiSearch
    std::vector<float> distWU(indexNum * warmupNum * k, 0);
    std::vector<faiss::idx_t> labelWU(indexNum * warmupNum * k, 0);
    Search(indexes, warmupNum, data[0].data(), k, distWU.data(), labelWU.data(), false);

    for (size_t n = 0; n < searchNum.size(); n++)
    {
        std::vector<float> dist(indexNum * searchNum[n] * k, 0);
        std::vector<faiss::idx_t> label(indexNum * searchNum[n] * k, 0);

        double ts = GetMillisecs();
        for (int l = 0; l < loopTimes; l++)
        {
            Search(indexes, searchNum[n], data[0].data(), k, dist.data(), label.data(), false);
        }
        double te = GetMillisecs();
        printf("multi search: true, index num: %d, k: %d, base: %d, dim: %d, search num: %2d, QPS: %9.4f \n", indexNum,
               k, NTOTAL, DIM, searchNum[n], MILLI_SECOND * searchNum[n] * loopTimes / (te - ts));
    }
}

void MutiOneIndex(int indexNum, std::vector<std::vector<float>> &data,
                  std::vector<faiss::ascend::AscendIndex *> &indexes, std::vector<int> &searchNum)
{
    int topK = 5;
    int loopTime = 100;

    const int warmupNumber = 15;
    std::vector<float> distWU(indexNum * warmupNumber * topK, 0);
    std::vector<faiss::idx_t> labelWU(indexNum * warmupNumber * topK, 0);
    Search(indexes, warmupNumber, data[0].data(), topK, distWU.data(), labelWU.data(), false);

    for (size_t n = 0; n < searchNum.size(); n++)
    {
        std::vector<float> dist(searchNum[n] * topK, 0);
        std::vector<faiss::idx_t> label(searchNum[n] * topK, 0);

        double timeloop = 0.0;
        for (int l = 0; l < loopTime; l++)
        {
            for (int i = 0; i < indexNum; i++)
            {
                std::vector<faiss::ascend::AscendIndex *> index_one;
                index_one.emplace_back(indexes[i]);
                double ts = GetMillisecs();
                Search(index_one, searchNum[n], data[0].data(), topK, dist.data(), label.data(), false);
                double te = GetMillisecs();
                timeloop += (te - ts);
            }
        }

        printf(
            "multi search loop one index, index num: %d, topK: %d, base: %d, dim: %d, "
            "search num: %2d, QPS: %9.4f \n",
            indexNum, topK, NTOTAL, DIM, searchNum[n], MILLI_SECOND * searchNum[n] * loopTime / timeloop);
    }
}

TEST(TestAscendIndexSQ, TwentyIndexQPS)
{
    size_t ntotal = 300000;
    int indexNum = 20;
    std::vector<size_t> ntotals(indexNum, 36842);
    ntotals[0] = ntotal;

    std::vector<int> searchNum = {1, 2, 4, 8, 16};

    size_t maxSize = ntotal * DIM;
    std::vector<std::vector<float>> data(indexNum, std::vector<float>(maxSize));
    for (int i = 0; i < indexNum; ++i)
    {
        for (size_t j = 0; j < maxSize; j++)
        {
            data[i][j] = 1.0 * FastRand() / FAST_RAND_MAX;
        }
        Norm(data[i].data(), ntotal, DIM);
    }

    std::vector<faiss::ascend::AscendIndex *> indexes;
    faiss::ascend::AscendIndexSQConfig conf(DEVICE_IDS, 1024 * 1024 * 1024);

    for (int i = 0; i < indexNum; i++)
    {
        auto index =
            new faiss::ascend::AscendIndexSQ(DIM, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE, conf);
        ASSERT_FALSE(index == nullptr);
        indexes.emplace_back(index);
    }

    for (int j = 0; j < indexNum; j++)
    {
        const auto index = indexes[j];
        index->reset();
        for (auto deviceId : conf.deviceList)
        {
            int len = dynamic_cast<faiss::ascend::AscendIndexSQ *>(index)->getBaseSize(deviceId);
            ASSERT_EQ(len, 0);
        }
    }

    for (int i = 0; i < indexNum; ++i)
    {
        const auto index = indexes[i];
        index->train(ntotals[i], data[i].data());
        index->add(ntotals[i], data[i].data());

        {
            size_t getTotal = 0;
            for (auto deviceId : conf.deviceList)
            {
                size_t tmpTotal = dynamic_cast<faiss::ascend::AscendIndexSQ *>(index)->getBaseSize(deviceId);
                getTotal += tmpTotal;
            }
            ASSERT_EQ(getTotal, ntotals[i]);
        }
    }

    // multiSearch
    MultiSearch(indexNum, data, indexes, searchNum);

    // multiSearch loop one index
    MutiOneIndex(indexNum, data, indexes, searchNum);

    for (int i = 0; i < indexNum; i++)
    {
        delete indexes[i];
    }
}
}  // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
