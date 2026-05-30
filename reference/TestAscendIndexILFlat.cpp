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

#include <faiss/ascend/AscendCloner.h>
#include <faiss/ascend/AscendIndexILFlat.h>
#include <faiss/index_io.h>
#include <gtest/gtest.h>
#include <sys/time.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>

#include "acl/acl.h"
namespace
{
unsigned int g_seed;
const int FAST_RAND_MAX = 0x7FFF;
const int MILLI_SECOND = 1000;
const int TABLELEN_REDUNDANCY = 48;

inline double GetMillisecs()
{
    struct timeval tv = {0, 0};
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

// Compute a pseudorandom integer.
// Output value in range [0, 32767]
inline int FastRand(void)
{
    const int mutipliyNum = 214013;
    const int ntotalum = 2531011;
    const int rshiftNum = 16;
    g_seed = (mutipliyNum * g_seed + ntotalum);
    return (g_seed >> rshiftNum) & FAST_RAND_MAX;
}
static void CreateMappingTable(float *table, unsigned int tableLen)
{
    for (unsigned int i = 0; i < tableLen; i++)
    {
        // 创建映射表，score分值加2， 保证在2.0~3.0之间
        *(table + i) = i * 1.0;
    }
}

static void CreateNormVectorFloat(std::vector<float> &normVec, size_t addn, size_t dim)
{
    std::vector<float> normBase(addn);
    for (size_t i = 0; i < addn * dim; i++)
    {
        normVec[i] = 1.0 * FastRand() / FAST_RAND_MAX;
        size_t baseIdx = i / dim;

        normBase[baseIdx] += normVec[i] * normVec[i];
        if ((i + 1) % dim == 0)
        {
            normBase[baseIdx] = sqrt(normBase[baseIdx]);
        }
    }
    // normalize
    for (size_t i = 0; i < addn * dim; i++)
    {
        normVec[i] /= normBase[i / dim];
    }
}

TEST(TestAscendIndexFlat, QPS)
{
    size_t dim = 512;
    size_t ntotal = 1000000;
    size_t capacity = 12000000;
    int queryN = 10;
    int topk = 10;
    try
    {
        const int64_t resourceSize = 1 * 1024 * 1024 * 1024;
        faiss::ascend::AscendIndexILFlat index;
        std::vector<int> deviceList = {0};
        auto metricType = faiss::MetricType::METRIC_INNER_PRODUCT;
        auto ret = index.Init(dim, capacity, metricType, deviceList, resourceSize);
        EXPECT_EQ(ret, 0);

        std::vector<float> addVec(ntotal * dim);
        std::vector<uint32_t> ids(ntotal);
        std::iota(ids.begin(), ids.end(), 0);
        for (size_t i = 0; i < addVec.size(); i++)
        {
            addVec[i] = drand48();
        }

        CreateNormVectorFloat(addVec, ntotal, dim);

        std::vector<uint16_t> addVecFp16(addVec.size());
        std::transform(addVec.begin(), addVec.end(), addVecFp16.begin(),
                       [](float tmp) { return aclFloatToFloat16(tmp); });

        ret = index.AddFeatures(ntotal, addVecFp16.data());
        EXPECT_EQ(ret, 0);

        std::vector<uint16_t> queries;
        queries.assign(addVecFp16.begin(), addVecFp16.begin() + queryN * dim);

        std::vector<uint32_t> idxs(queryN * topk);
        std::vector<float> distances(queryN * topk);
        unsigned int tableLen = 10000;
        std::vector<float> table(tableLen + TABLELEN_REDUNDANCY);
        CreateMappingTable(table.data(), tableLen + TABLELEN_REDUNDANCY);
        std::vector<int> num(queryN);

        ret = index.Search(queryN, queries.data(), topk, idxs.data(), distances.data(), tableLen, table.data());
        EXPECT_EQ(ret, 0);

        double start = GetMillisecs();
        int loopTimes = 2;
        for (int i = 0; i < loopTimes; i++)
        {
            ret = index.Search(queryN, queries.data(), topk, idxs.data(), distances.data(), tableLen, table.data());
            EXPECT_EQ(ret, 0);
        }
        double end = GetMillisecs();
        printf("base:%zu, dim:%zu, search num:%d, time:%.4f QPS:%.4f\n", ntotal, dim, queryN,
               (end - start) / queryN / loopTimes, MILLI_SECOND * queryN * loopTimes / (end - start));
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
