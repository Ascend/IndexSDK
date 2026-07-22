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

// 需要生成aicpu算子+ivfflat算子+ivfpq算子(-d 128 -c 1024)

#include <faiss/ascend/AscendIndexIVFPQ.h>

#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

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

int main()
{
    size_t dim = 128;
    size_t ntotal = 10000000;
    int ncentroids = 262144;
    int nprobe = 1024;
    int M = 4;
    int nbits = 8;

    printf("generate data\n");
    std::vector<float> data(dim * ntotal);
    for (size_t i = 0; i < data.size(); i++)
    {
        data[i] = drand48();
    }
    Norm(data.data(), ntotal, dim);
    std::vector<int64_t> ids(ntotal);
    for (size_t i = 0; i < ids.size(); i++)
    {
        ids[i] = i;
    }

    faiss::ascend::AscendIndexIVFPQ *index = nullptr;
    try
    {
        int64_t resourceSize = static_cast<int64_t>(8192) * 1024 * 1024;
        faiss::ascend::AscendIndexIVFPQConfig conf({0, 1, 5, 7}, resourceSize);
        conf.cp.niter = 3;
        conf.useKmeansPP = true;
        // Optional training tuning (defaults preserve legacy behavior):
        conf.trainSamplesPerList = 40;
        conf.maxTrainSamples = 300000;
        conf.pqNiter = 2;
        // For large nlist / large training samples on multi-device setups,
        // enable distributed coarse clustering (fp16, split across all devices):
        // conf = faiss::ascend::AscendIndexIVFPQConfig({0, 1, 2, 3}, resourceSize);
        conf.useDistributedCoarse = true;
        printf("create index\n");
        index = new faiss::ascend::AscendIndexIVFPQ(dim, faiss::METRIC_L2, ncentroids, M, nbits, conf);
        index->verbose = true;
        index->setNumProbes(nprobe);

        printf("start train\n");
        auto train_start = std::chrono::high_resolution_clock::now();
        index->train(ntotal, data.data());
        auto train_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::high_resolution_clock::now() - train_start)
                            .count();
        printf("train finished, elapsed=%lld ms\n", static_cast<long long>(train_ms));
        printf("start add\n");
        index->add_with_ids(ntotal, data.data(), ids.data());

        size_t n = 10;
        size_t k = 10;
        std::vector<float> dist(n * k, 0.0);
        std::vector<faiss::idx_t> label(n * k, 0);
        printf("start search\n");
        index->search(n, data.data(), k, dist.data(), label.data());
    }
    catch (std::exception &e)
    {
        printf("exceptin caught: %s\n", e.what());
        delete index;
        return -1;
    }
    delete index;
    printf("search success\n");
    return 0;
}
