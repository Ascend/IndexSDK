/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/ascend/AscendIndexIVFRaBitQ.h>
#include <gtest/gtest.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <set>
#include <unordered_map>
#include <vector>

namespace
{
constexpr int kDim = 128;
constexpr faiss::idx_t kNtotal = 200000;
constexpr int kNprobe = 64;
constexpr int kTopK = 100;
constexpr int kQueryNum = 8;
constexpr int kCpSeed = 1234;
constexpr int kDatasetSeed = 5678;
constexpr int kTrainMultiplier = 40;
constexpr float kRecallMin = 0.95f;

const int RECMAP_KEY_1 = 1;
const int RECMAP_KEY_10 = 10;
const int RECMAP_KEY_100 = 100;

using recallMap = std::unordered_map<int, float>;

void Norm(float* data, size_t n, int dim)
{
#pragma omp parallel for if (n > 100)
    for (size_t i = 0; i < n; ++i)
    {
        float l2norm = 0.0f;
        for (int j = 0; j < dim; ++j)
        {
            l2norm += data[i * dim + j] * data[i * dim + j];
        }
        l2norm = std::sqrt(l2norm);
        if (std::fabs(l2norm) < FLT_EPSILON)
        {
            for (int j = 0; j < dim; ++j)
            {
                data[i * dim + j] = 1.0f / std::sqrt(static_cast<float>(dim));
            }
        }
        else
        {
            for (int j = 0; j < dim; ++j)
            {
                data[i * dim + j] = data[i * dim + j] / l2norm;
            }
        }
    }
}

void generateData(float* data, faiss::idx_t ntotal, int dim, int seed)
{
    std::mt19937 gen(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (faiss::idx_t i = 0; i < ntotal * dim; ++i)
    {
        data[i] = dis(gen);
    }
    Norm(data, static_cast<size_t>(ntotal), dim);
}

void computeRecall(recallMap& recMap, int j)
{
    recMap[RECMAP_KEY_100]++;
    switch (j)
    {
        case 0:
            recMap[RECMAP_KEY_1]++;
            recMap[RECMAP_KEY_10]++;
            break;
        case 1 ... 9:
            recMap[RECMAP_KEY_10]++;
            break;
        default:
            break;
    }
}

recallMap calRecall(const std::vector<faiss::idx_t>& npuLabel, const faiss::idx_t* cpuLabel, int queryNum, int k)
{
    recallMap map;
    map[RECMAP_KEY_1] = 0;
    map[RECMAP_KEY_10] = 0;
    map[RECMAP_KEY_100] = 0;
    if (queryNum <= 0)
    {
        return map;
    }

    for (int i = 0; i < queryNum; ++i)
    {
        std::set<faiss::idx_t> labelSet(npuLabel.begin() + i * k, npuLabel.begin() + i * k + k);
        for (int j = 0; j < k; ++j)
        {
            if (cpuLabel[i * k] == npuLabel[static_cast<size_t>(i * k + j)])
            {
                computeRecall(map, j);
                break;
            }
        }
        (void)labelSet;
    }

    map[RECMAP_KEY_1] = map[RECMAP_KEY_1] / queryNum * 100;
    map[RECMAP_KEY_10] = map[RECMAP_KEY_10] / queryNum * 100;
    map[RECMAP_KEY_100] = map[RECMAP_KEY_100] / queryNum * 100;
    return map;
}

void configureCpuIndex(faiss::IndexIVFRaBitQ& index)
{
    index.nprobe = kNprobe;
    index.cp.niter = 10;
    index.cp.min_points_per_centroid = 39;
    index.cp.max_points_per_centroid = 256;
    index.cp.seed = kCpSeed;
    index.cp.spherical = true;
    index.by_residual = true;
    index.qb = 0;
    index.verbose = true;
}

faiss::ascend::AscendIndexIVFRaBitQConfig makeAscendConfig()
{
    const int64_t resourceSize = static_cast<int64_t>(2048) * 1024 * 1024;
    faiss::ascend::AscendIndexIVFRaBitQConfig conf({0}, false, false, 12345, 2.0f, resourceSize);
    conf.useKmeansPP = false;
    conf.cp.niter = 10;
    conf.cp.min_points_per_centroid = 39;
    conf.cp.max_points_per_centroid = 256;
    conf.cp.seed = kCpSeed;
    conf.cp.spherical = true;
    return conf;
}

void runBoundaryRecallTest(int nlist, const char* caseName)
{
    const faiss::idx_t trainNum = kNtotal > static_cast<faiss::idx_t>(nlist) * kTrainMultiplier
                                      ? static_cast<faiss::idx_t>(nlist) * kTrainMultiplier
                                      : kNtotal;

    std::vector<float> data(static_cast<size_t>(kNtotal * kDim));
    generateData(data.data(), kNtotal, kDim, kDatasetSeed);

    faiss::IndexFlatL2 quantizer(kDim);
    faiss::IndexIVFRaBitQ cpuIndex(&quantizer, kDim, nlist, faiss::METRIC_L2);
    configureCpuIndex(cpuIndex);

    printf("[%s] train on CPU, trainNum=%ld, nlist=%d\n", caseName, static_cast<long>(trainNum), nlist);
    cpuIndex.train(trainNum, data.data());
    cpuIndex.add(kNtotal, data.data());

    faiss::ascend::AscendIndexIVFRaBitQConfig conf = makeAscendConfig();
    faiss::ascend::AscendIndexIVFRaBitQ npuIndex(kDim, faiss::METRIC_L2, nlist, conf);
    npuIndex.verbose = true;
    npuIndex.setNumProbes(kNprobe);
    npuIndex.copyFrom(&cpuIndex);

    std::vector<float> cpuDist(static_cast<size_t>(kQueryNum * kTopK), 0.0f);
    std::vector<faiss::idx_t> cpuLabel(static_cast<size_t>(kQueryNum * kTopK), 0);
    std::vector<float> npuDist(static_cast<size_t>(kQueryNum * kTopK), 0.0f);
    std::vector<faiss::idx_t> npuLabel(static_cast<size_t>(kQueryNum * kTopK), 0);

    cpuIndex.search(kQueryNum, data.data(), kTopK, cpuDist.data(), cpuLabel.data());
    npuIndex.search(kQueryNum, data.data(), kTopK, npuDist.data(), npuLabel.data());

    ASSERT_GT(kQueryNum, 0);
    recallMap recall = calRecall(npuLabel, cpuLabel.data(), kQueryNum, kTopK);
    printf("[%s] recall@%d vs CPU: @1=%.2f, @10=%.2f, @100=%.2f\n", caseName, kTopK, recall[RECMAP_KEY_1],
           recall[RECMAP_KEY_10], recall[RECMAP_KEY_100]);

    EXPECT_GE(recall[RECMAP_KEY_100], kRecallMin * 100.0f)
        << caseName << ": NPU vs CPU recall@" << kTopK << " below " << kRecallMin;
}

TEST(TestAscendIndexIVFRaBitQBoundary, L1DistCodesTile16384) { runBoundaryRecallTest(16384, "L1DistCodesTile16384"); }

TEST(TestAscendIndexIVFRaBitQBoundary, L1DistCodesTile10048) { runBoundaryRecallTest(10048, "L1DistCodesTile10048"); }

TEST(TestAscendIndexIVFRaBitQBoundary, L1DistCodesTile8192Control)
{
    runBoundaryRecallTest(8192, "L1DistCodesTile8192Control");
}

}  // namespace

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
