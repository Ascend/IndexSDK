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

// 需要生成aicpu算子+ivfflat算子(-d 128 -c 1024)

#include <faiss/ascend/AscendIndexIVFFlat.h>

#include <cfloat>
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
    size_t ntotal = 1000000;
    int ncentroids = 1024;
    int nprobe = 32;

    printf("generate data\n");
    std::vector<float> data(dim * ntotal);
    for (size_t i = 0; i < data.size(); i++)
    {
        data[i] = drand48();
    }
    Norm(data.data(), ntotal, dim);

    faiss::ascend::AscendIndexIVFFlat *index = nullptr;
    try
    {
        faiss::ascend::AscendIndexIVFFlatConfig conf{{0}};
        printf("create index\n");
        index = new faiss::ascend::AscendIndexIVFFlat(dim, faiss::METRIC_INNER_PRODUCT, ncentroids, conf);
        index->verbose = true;
        index->setNumProbes(nprobe);

        printf("start train\n");
        index->train(ntotal, data.data());
        printf("start add\n");
        index->add(ntotal, data.data());

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
