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

// python3 aicpu_generate_model.py -t npu_type
// python3 binary_flat_generate_model.py -d 256

#include <faiss/Clustering.h>
#include <faiss/ascend/AscendCloner.h>
#include <faiss/ascend/AscendIndexBinaryFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/index_io.h>
#include <gtest/gtest.h>
#include <sys/time.h>

#include <random>

namespace
{
const int BITS = 8;
const int SEED = 1;
const int MILLI_SECOND = 1000;
std::independent_bits_engine<std::mt19937, BITS, uint8_t> engine(SEED);

void FeatureGenerator(std::vector<uint8_t> &features)
{
    size_t n = features.size();
    for (size_t i = 0; i < n; ++i)
    {
        features[i] = engine();
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

TEST(TestAscendIndexBinaryFlat, QPS)
{
    int dim = 256;
    size_t ntotal = 1000000;
    std::vector<int> searchNum = {8, 16, 32, 64, 128, 256};

    faiss::ascend::AscendIndexBinaryFlatConfig conf({0});
    faiss::ascend::AscendIndexBinaryFlat index(dim, conf);
    index.verbose = true;

    printf("generate data\n");
    std::vector<uint8_t> base(ntotal * index.code_size, 0);
    FeatureGenerator(base);
    try
    {
        printf("add data\n");
        index.add(ntotal, base.data());
        int warmUpTimes = 10;
        std::vector<int32_t> distw(127 * 10, 0);
        std::vector<faiss::idx_t> labelw(127 * 10, 0);
        for (int i = 0; i < warmUpTimes; i++)
        {
            index.search(127, base.data(), 10, distw.data(), labelw.data());
        }

        for (size_t n = 0; n < searchNum.size(); n++)
        {
            int k = 128;
            int loopTimes = 10;
            std::vector<int32_t> dist(searchNum[n] * k, 0);
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
}  // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
