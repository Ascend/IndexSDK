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

#include <faiss/ascend/AscendCloner.h>
#include <faiss/ascend/AscendIndexInt8Flat.h>
#include <faiss/index_io.h>
#include <gtest/gtest.h>
#include <sys/time.h>

#include <iostream>
#include <random>

namespace
{
using recallMap = std::unordered_map<int, float>;
const int RECMAP_KEY_1 = 1;
const int RECMAP_KEY_10 = 10;
const int RECMAP_KEY_100 = 100;
const int MILLI_SECOND = 1000;
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

inline void sqEncode(const faiss::ScalarQuantizer sq, const float *base, int8_t *baseInt8, const size_t n)
{
    const int offset = 128;
    std::vector<uint8_t> queryUint8(n * sq.code_size);
    sq.compute_codes(base, queryUint8.data(), n);
    for (size_t i = 0; i < n * sq.code_size; i++)
    {
        baseInt8[i] = queryUint8[i] - offset;
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
    recallMap Map;
    Map[RECMAP_KEY_1] = 0;
    Map[RECMAP_KEY_10] = 0;
    Map[RECMAP_KEY_100] = 0;
    if (queryNum == 0)
    {
        std::cerr << "Error: Invalid queryNum value." << std::endl;
        return Map;
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
                computeRecall(Map, j);
                break;
            }
        }
    }
    Map[RECMAP_KEY_1] = Map[RECMAP_KEY_1] / queryNum * 100;  // recMap[1]的百分比 这里的100代表的是百分比的计算因子
    Map[RECMAP_KEY_10] = Map[RECMAP_KEY_10] / queryNum * 100;  // recMap[10]的百分比 这里的100代表的是百分比的计算因子
    Map[RECMAP_KEY_100] =
        Map[RECMAP_KEY_100] / queryNum * 100;  // recMap[100]的百分比 这里的100代表的是百分比的计算因子
    return Map;
}

TEST(TestAscendIndexInt8Flat, QPS)
{
    int dim = 512;
    size_t ntotal = 1000000;
    std::vector<int> searchNum = {8, 16, 32, 64, 128, 256};
    try
    {
        faiss::ascend::AscendIndexInt8FlatConfig conf({0}, 1024 * 1024 * 1024);
        faiss::ascend::AscendIndexInt8Flat index(dim, faiss::METRIC_INNER_PRODUCT, conf);
        index.verbose = true;

        printf("start generate data\n");
        std::vector<int8_t> base(ntotal * dim);
        std::vector<float> baseFp(ntotal * dim);
        for (size_t i = 0; i < ntotal * dim; i++)
        {
            baseFp[i] = drand48();
        }
        printf("generate data finished\n");
        faiss::ScalarQuantizer sq = faiss::ScalarQuantizer(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit);
        sq.train(ntotal, baseFp.data());
        sqEncode(sq, baseFp.data(), base.data(), ntotal);
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
            int k = 100;
            int loopTimes = 10;

            std::vector<float> dist(searchNum[n] * k, 0);
            std::vector<faiss::idx_t> label(searchNum[n] * k, 0);
            double ts = GetMillisecs();
            // start to quantize query data
            std::vector<int8_t> query(searchNum[n] * dim);
            sqEncode(sq, baseFp.data(), query.data(), searchNum[n]);

            for (int l = 0; l < loopTimes; l++)
            {
                index.search(searchNum[n], query.data(), k, dist.data(), label.data());
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

TEST(TestAscendIndexInt8Flat, Recall)
{
    int dim = 512;
    size_t ntotal = 1000000;
    int searchNum = 8;
    try
    {
        faiss::ascend::AscendIndexInt8FlatConfig conf({0}, 1024 * 1024 * 1024);
        faiss::ascend::AscendIndexInt8Flat index(dim, faiss::METRIC_INNER_PRODUCT, conf);
        index.verbose = true;

        printf("start generate data\n");
        std::vector<int8_t> base(ntotal * dim);
        std::vector<float> baseFp(ntotal * dim);
        for (size_t i = 0; i < ntotal * dim; i++)
        {
            baseFp[i] = drand48();
        }
        printf("generate data finished\n");
        faiss::ScalarQuantizer sq = faiss::ScalarQuantizer(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit);
        sq.train(ntotal, baseFp.data());
        sqEncode(sq, baseFp.data(), base.data(), ntotal);
        printf("add data\n");
        index.add(ntotal, base.data());

        int k = 100;
        std::vector<faiss::idx_t> gt(searchNum * k, 0);
        for (int i = 0; i < searchNum; i++)
        {
            gt[i * k] = i;
        }
        std::vector<float> dist(searchNum * k, 0);
        std::vector<faiss::idx_t> label(searchNum * k, 0);
        // start to quantize query data
        std::vector<int8_t> query(searchNum * dim);
        sqEncode(sq, baseFp.data(), query.data(), searchNum);
        index.search(searchNum, query.data(), k, dist.data(), label.data());
        recallMap Top = calRecall(label, gt.data(), searchNum);
        printf("TOPK %d: t1 = %.2f, t10 = %.2f, t100 = %.2f\n", k, Top[1], Top[10], Top[100]);
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
