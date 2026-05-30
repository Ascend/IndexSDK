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

// 需要生成aicpu算子+int8flat算子(-d 512)

#include <faiss/Clustering.h>
#include <faiss/ascend/AscendCloner.h>
#include <faiss/ascend/AscendIndexInt8Flat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/index_io.h>
#include <gtest/gtest.h>
#include <sys/time.h>

#include <algorithm>
#include <iostream>
#include <random>

namespace
{
using recallMap = std::unordered_map<int, float>;
const int RECMAP_KEY_1 = 1;
const int RECMAP_KEY_10 = 10;
const int RECMAP_KEY_100 = 100;
const int MILLI_SECOND = 1000;
void GenerateCodes(int8_t *codes, size_t total, size_t dim, int seed = -1)
{
    int maxValue = 255;
    int offset = 128;
    auto timeSeed = time(nullptr);
    if (timeSeed < 0)
    {
        std::cerr << "Error: Invalid seed value." << std::endl;
        return;
    }
    std::default_random_engine e((seed > 0) ? seed : timeSeed);
    std::uniform_real_distribution<float> rCode(0.0f, 1.0f);
    for (size_t i = 0; i < total; i++)
    {
        for (size_t j = 0; j < dim; j++)
        {
            // uint8's max value is 255
            codes[i * dim + j] = static_cast<int8_t>(maxValue * rCode(e) - offset);
        }
    }
}

inline double GetMillisecs()
{
    struct timeval tv
    {
        0, 0
    };
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

inline void AssertInt8Equal(size_t count, const int8_t *gt, const int8_t *data)
{
    for (size_t i = 0; i < count; i++)
    {
        ASSERT_TRUE(gt[i] == data[i]) << "i: " << i << " gt: " << int(gt[i]) << " data: " << int(data[i]) << std::endl;
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

template <class T>
recallMap calRecall(std::vector<T> label, int64_t *gt, int queryNum)
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

TEST(TestAscendIndexInt8Flat, QPS)
{
    int dim = 512;
    size_t ntotal = 1000000;
    std::vector<int> searchNum = {8, 16, 32, 64, 128, 256};
    try
    {
        faiss::ascend::AscendIndexInt8FlatConfig conf({0}, 1024 * 1024 * 1024);
        faiss::ascend::AscendIndexInt8Flat index(dim, faiss::METRIC_L2, conf);
        index.verbose = true;

        printf("generate data\n");
        std::vector<int8_t> base(ntotal * dim);
        GenerateCodes(base.data(), ntotal, dim);

        printf("add data\n");
        index.add(ntotal, base.data());
        int warmUpTimes = 10;
        std::vector<float> distw(127 * 10, 0);
        std::vector<faiss::idx_t> labelw(127 * 10, 0);
        for (int i = 0; i < warmUpTimes; i++)
        {
            index.search(127, base.data(), 10, distw.data(), labelw.data());
        }

        for (size_t n = 0; n < searchNum.size(); n++)
        {
            int k = 128;
            int loopTimes = 10;
            std::vector<float> dist(searchNum[n] * k, 0);
            std::vector<faiss::idx_t> label(searchNum[n] * k, 0);
            double ts = GetMillisecs();
            for (int l = 0; l < loopTimes; l++)
            {
                index.search(searchNum[n], base.data(), k, dist.data(), label.data());
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

TEST(TestAscendIndexInt8Flat, Acc)
{
    size_t dim = 512;
    size_t ntotal = 1000000;
    int searchNum = 8;
    try
    {
        faiss::ascend::AscendIndexInt8FlatConfig conf({0}, 1024 * 1024 * 1024);
        faiss::ascend::AscendIndexInt8Flat index(dim, faiss::METRIC_L2, conf);
        index.verbose = true;

        printf("generate data\n");
        std::vector<int8_t> base(ntotal * dim);
        GenerateCodes(base.data(), ntotal, dim);

        printf("add data\n");
        index.add(ntotal, base.data());
        printf("add finish\n");
        size_t k = 100;
        std::vector<float> dist(searchNum * k, 0);
        std::vector<faiss::idx_t> label(searchNum * k, 0);
        printf("search start\n");
        index.search(searchNum, base.data(), k, dist.data(), label.data());
        printf("search finish\n");
        printf("search compute distance by cpu\n");
        std::vector<faiss::idx_t> gtLabel;
        for (int q = 0; q < searchNum; q++)
        {
            std::vector<float> cpuDist(ntotal, 0);
            for (size_t i = 0; i < ntotal; i++)
            {
                int sum = 0;
                for (size_t d = 0; d < dim; d++)
                {
                    sum += (base[q * dim + d] - base[i * dim + d]) * (base[q * dim + d] - base[i * dim + d]);
                }
                cpuDist[i] = sqrt(sum);
            }
            std::vector<std::pair<int, float>> cpuRes;
            for (size_t i = 0; i < ntotal; ++i)
            {
                cpuRes.push_back({i, cpuDist[i]});
            }
            std::sort(cpuRes.begin(), cpuRes.end(),
                      [](std::pair<int, float> a, std::pair<int, float> b) -> bool { return a.second < b.second; });
            for (size_t d = 0; d < k; ++d)
            {
                gtLabel.push_back(cpuRes[d].first);
            }
        }
        recallMap top = calRecall(label, gtLabel.data(), searchNum);

        printf("Recall %zu: @1 = %.2f, @10 = %.2f, @100 = %.2f \n", k, top[RECMAP_KEY_1], top[RECMAP_KEY_10],
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
