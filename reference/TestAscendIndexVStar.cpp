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

#include <faiss/ascend/AscendIndexVStar.h>
#include <gtest/gtest.h>
#include <sys/time.h>

#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>

using namespace faiss::ascend;
namespace
{
const int MILLI_SECOND = 1000;
inline double GetMillisecs()
{
    struct timeval tv = {0, 0};
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

void Generate(size_t ntotal, std::vector<float> &data, int seed = 5678)
{
    int maxValue = 255;
    int offset = 128;
    std::default_random_engine e(seed);
    std::uniform_real_distribution<float> rCode(0.0f, 1.0f);
    data.resize(ntotal);
    for (size_t i = 0; i < ntotal; ++i)
    {
        data[i] = static_cast<float>(maxValue * rCode(e) - offset);
    }
}

void Norm(std::vector<float> &data, int dim)
{
    float square = 0.0;
    int nTotal = (dim == 0) ? 0 : static_cast<int>(data.size() / dim);
    for (int i = 0; i < nTotal; ++i)
    {
        square = 0.0;
        for (int j = 0; j < dim; ++j)
        {
            square += pow(data[i * dim + j], 2);  // 2是先求平方，后续开根
        }
        square = sqrt(square);
        if (fabs(square) < FLT_EPSILON)
        {
            std::cerr << "Error: Invalid square value." << std::endl;
            return;
        }
        for (int j = 0; j < dim; ++j)
        {
            data[static_cast<size_t>(i) * dim + j] /= square;
        }
    }
}

/**
 * AKMode需要提前生成算子和码本
 * 码本和算子参数根据实际情况调整, dim nlistL1 subDimL1 要与创建的索引一致
 * 算子：python3 vstar_generate_models.py --dim 1024 --nlistL1 256 --subDimL1 128
 * 码本：python3 vstar_train_codebook.py --dataPath {实际base数据路径} --dim 1024 --codebookPath {实际码本输出路径}
 --nlistL1 256 --subDimL1 128 --device 0
 */
TEST(TestAscendIndexVstar, Test_Search_Func)
{
    int dim = 1024;
    size_t ntotal = 1e5;
    int nlist = 256;
    int subSpaceDim = 128;
    std::vector<int> devices = {0};
    bool verbose = false;
    try
    {
        AscendIndexVstarInitParams aParams(dim, subSpaceDim, nlist, devices, verbose);
        auto index = std::make_shared<AscendIndexVStar>(aParams);

        // 添加码本 需要提前生成好码本路径
        std::string codebook = "/home/work/codebook_256_1024_128/codebook_l1_l2.bin";
        auto ret = index->AddCodeBooksByPath(codebook);
        EXPECT_EQ(ret, 0);

        // 生成base底库数据
        std::vector<float> data(ntotal);
        Generate(ntotal * dim, data);
        // 标准化
        Norm(data, dim);

        // add底库
        index->Add(data);
        size_t total = 0;
        index->GetNTotal(total);
        EXPECT_EQ(total, ntotal);

        // search检索
        int topk = 100;
        int warmUpTimes = 10;
        size_t nq = 9000;
        std::vector<float> distsWarm(nq * topk);
        std::vector<int64_t> labelsWarm(nq * topk);

        // warm up
        for (int i = 0; i < warmUpTimes; ++i)
        {
            AscendIndexSearchParams searchParamsWarm{100, data, topk, distsWarm, labelsWarm};
            index->Search(searchParamsWarm);
        }

        // search
        std::vector<size_t> searchNum = {1, 8, 16, 32, 64, 128, 256};
        int loopTimes = 100;
        for (auto n : searchNum)
        {
            std::vector<float> queryData(data.begin(), data.begin() + n * dim);
            std::vector<float> dists(n * topk, 0);
            std::vector<int64_t> labels(n * topk, 0);
            double ts = GetMillisecs();
            for (int i = 0; i < loopTimes; ++i)
            {
                AscendIndexSearchParams searchParams{n, queryData, topk, dists, labels};
                index->Search(searchParams);
            }
            double te = GetMillisecs();
            printf("base:%zu, dim:%d, search num:%zu, QPS:%.4f\n", ntotal, dim, n,
                   MILLI_SECOND * n * loopTimes / (te - ts));
        }
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
