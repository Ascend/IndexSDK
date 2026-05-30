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

// 需要生成aicpu算子+flat算子(-d 512)

#include <faiss/IndexFlat.h>
#include <faiss/ascend/AscendCloner.h>
#include <faiss/ascend/AscendIndexFlat.h>
#include <faiss/index_io.h>
#include <gtest/gtest.h>
#include <sys/time.h>

#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>

namespace
{
unsigned int g_seed;
const int FAST_RAND_MAX = 0x7FFF;
const int RECMAP_KEY_1 = 1;
const int RECMAP_KEY_10 = 10;
const int RECMAP_KEY_100 = 100;
const int MILLI_SECOND = 1000;
using recallMap = std::unordered_map<int, float>;
inline double GetMillisecs()
{
    struct timeval tv = {0, 0};
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

inline void AssertEqual(const std::vector<float> &gt, const std::vector<float> &data)
{
    const float epson = 1e-3;
    ASSERT_EQ(gt.size(), data.size());
    for (size_t i = 0; i < gt.size(); i++)
    {
        ASSERT_TRUE(fabs(gt[i] - data[i]) <= epson)
            << "i: " << i << " gt: " << gt[i] << " data: " << data[i] << std::endl;
    }
}

void computeRecall(recallMap &recMap, int j)
{
    recMap[RECMAP_KEY_100]++;
    switch (j)
    {
        case 0:
            recMap[RECMAP_KEY_1]++;
            recMap[RECMAP_KEY_10]++;
            break;
        case 1 ... 9:  // case 1到9
            recMap[RECMAP_KEY_10]++;
            break;
        default:
            break;
    }
}

recallMap calRecall(std::vector<faiss::idx_t> label, int64_t *gt, int queryNum)
{
    recallMap map;
    map[RECMAP_KEY_1] = 0;
    map[RECMAP_KEY_10] = 0;
    map[RECMAP_KEY_100] = 0;
    if (queryNum == 0)
    {
        std::cerr << "Error: Invalid queryNum value." << std::endl;
        return map;
    }
    int k = label.size() / queryNum;

    for (int i = 0; i < queryNum; i++)
    {
        std::set<int> labelSet(label.begin() + i * k, label.begin() + i * k + k);
        if (labelSet.size() != static_cast<size_t>(k))
        {
            printf("current query have duplicated labels!!! \n");
        }
        for (int j = 0; j < k; j++)
        {
            if (gt[i * k] == label[i * k + j])
            {
                computeRecall(map, j);
                break;
            }
        }
    }
    map[RECMAP_KEY_1] = map[RECMAP_KEY_1] / queryNum * 100;  // recMap[1]的百分比 这里的100代表的是百分比的计算因子
    map[RECMAP_KEY_10] = map[RECMAP_KEY_10] / queryNum * 100;  // recMap[10]的百分比 这里的100代表的是百分比的计算因子
    map[RECMAP_KEY_100] =
        map[RECMAP_KEY_100] / queryNum * 100;  // recMap[100]的百分比 这里的100代表的是百分比的计算因子
    return map;
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
        }
        for (int j = 0; j < dim; ++j)
        {
            data[i * dim + j] = data[i * dim + j] / l2norm;
        }
    }
}

TEST(TestAscendIndexFlat, QPS)
{
    int dim = 512;
    size_t ntotal = 1000000;
    size_t maxSize = ntotal * dim;
    try
    {
        std::vector<float> data(maxSize);
        for (size_t i = 0; i < maxSize; i++)
        {
            data[i] = 1.0 * FastRand() / FAST_RAND_MAX;
        }

        faiss::ascend::AscendIndexFlatConfig conf({0}, 1024 * 1024 * 1500);
        faiss::ascend::AscendIndexFlat index(dim, faiss::METRIC_L2, conf);
        index.verbose = true;

        // 标准化
        Norm(data.data(), ntotal, dim);

        index.add(ntotal, data.data());
        {
            size_t getTotal = 0;
            for (size_t i = 0; i < conf.deviceList.size(); i++)
            {
                size_t tmpTotal = index.getBaseSize(conf.deviceList[i]);
                getTotal += tmpTotal;
            }
            EXPECT_EQ(getTotal, ntotal);
        }

        int warmUpTimes = 10;
        std::vector<float> distw(127 * 10, 0);
        std::vector<faiss::idx_t> labelw(127 * 10, 0);
        for (int i = 0; i < warmUpTimes; i++)
        {
            index.search(127, data.data(), 10, distw.data(), labelw.data());
        }

        std::vector<int> searchNum = {8, 16, 32, 64, 128, 256};
        for (size_t n = 0; n < searchNum.size(); n++)
        {
            int k = 10;
            int loopTimes = 100;
            std::vector<float> dist(searchNum[n] * k, 0);
            std::vector<faiss::idx_t> label(searchNum[n] * k, 0);
            double ts = GetMillisecs();
            for (int i = 0; i < loopTimes; i++)
            {
                index.search(searchNum[n], data.data(), k, dist.data(), label.data());
            }
            double te = GetMillisecs();
            printf("case[%zu]: base:%zu, dim:%d, search num:%d, QPS:%.4f\n", n, ntotal, dim, searchNum[n],
                   MILLI_SECOND * searchNum[n] * loopTimes / (te - ts));
        }
    }
    catch (std::exception &e)
    {
        printf("%s\n", e.what());
    }
}

TEST(TestAscendIndexFlat, Acc)
{
    int dim = 512;
    size_t ntotal = 1000000;
    size_t maxSize = ntotal * dim;
    faiss::MetricType type = faiss::METRIC_L2;
    int topk = 100;
    int queryNum = 8;
    printf("generate data\n");
    std::vector<float> data(maxSize);
    try
    {
        for (size_t i = 0; i < maxSize; i++)
        {
            data[i] = 1.0 * FastRand() / FAST_RAND_MAX;
        }
        faiss::ascend::AscendIndexFlatConfig conf({0}, 1024 * 1024 * 1500);
        faiss::ascend::AscendIndexFlat index(dim, faiss::METRIC_L2, conf);
        index.verbose = true;

        // 标准化
        Norm(data.data(), ntotal, dim);

        index.add(ntotal, data.data());
        printf("start search by npu\n");
        std::vector<float> dist(queryNum * topk, 0);
        std::vector<faiss::idx_t> label(queryNum * topk, 0);
        index.search(queryNum, data.data(), topk, dist.data(), label.data());

        printf("start add by cpu\n");
        faiss::IndexFlat faissIndex(dim, type);
        faissIndex.add(ntotal, data.data());
        std::vector<float> cpuDist(queryNum * topk, 0);
        std::vector<faiss::idx_t> cpuLabel(queryNum * topk, 0);
        printf("start search by cpu\n");
        faissIndex.search(queryNum, data.data(), topk, cpuDist.data(), cpuLabel.data());
        recallMap top = calRecall(label, cpuLabel.data(), queryNum);
        printf("Recall %d: @1 = %.2f, @10 = %.2f, @100 = %.2f \n", topk, top[RECMAP_KEY_1], top[RECMAP_KEY_10],
               top[RECMAP_KEY_100]);
    }
    catch (std::exception &e)
    {
        printf("%s\n", e.what());
    }
}

}  // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
