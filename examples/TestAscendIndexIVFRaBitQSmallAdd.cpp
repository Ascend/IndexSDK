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

// IVFRaBitQ 小批量反复 add_with_ids 样例（验证 DeviceMemArena 容量预留修复，见 #135 / !349）
// 需要生成 aicpu + ivfflat + ivfrabitq 算子（建议 -d 128）
//
// 用法:
//   ./TestAscendIndexIVFRaBitQSmallAdd [ntotal] [nlist] [batch] [nprobe]
// 默认: ntotal=100000 nlist=1024 batch=64 nprobe=32
// 可选环境变量: ASCENDFAISS_MEM_DEBUG=1 打印 arena / HBM 调试日志

#include <faiss/ascend/AscendIndexIVFRaBitQ.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

namespace
{
void Norm(float *data, size_t n, size_t dim)
{
#pragma omp parallel for if (n > 1)
    for (size_t i = 0; i < n; ++i)
    {
        float l2norm = 0.0f;
        for (size_t j = 0; j < dim; ++j)
        {
            l2norm += data[i * dim + j] * data[i * dim + j];
        }
        l2norm = std::sqrt(l2norm);
        if (std::fabs(l2norm) < FLT_EPSILON)
        {
            std::fprintf(stderr, "Error: Invalid l2norm at vector %zu\n", i);
            continue;
        }
        for (size_t j = 0; j < dim; ++j)
        {
            data[i * dim + j] /= l2norm;
        }
    }
}

size_t ParseSizeArg(const char *arg, size_t fallback)
{
    if (arg == nullptr || arg[0] == '\0')
    {
        return fallback;
    }
    char *end = nullptr;
    const unsigned long long v = std::strtoull(arg, &end, 10);
    if (end == arg || v == 0ULL)
    {
        return fallback;
    }
    return static_cast<size_t>(v);
}

int ParseIntArg(const char *arg, int fallback)
{
    if (arg == nullptr || arg[0] == '\0')
    {
        return fallback;
    }
    char *end = nullptr;
    const long v = std::strtol(arg, &end, 10);
    if (end == arg || v <= 0)
    {
        return fallback;
    }
    return static_cast<int>(v);
}
}  // namespace

int main(int argc, char **argv)
{
    constexpr size_t kDim = 128;
    const size_t ntotal = ParseSizeArg(argc > 1 ? argv[1] : nullptr, 100000);
    const int nlist = ParseIntArg(argc > 2 ? argv[2] : nullptr, 1024);
    const size_t batch = ParseSizeArg(argc > 3 ? argv[3] : nullptr, 64);
    const int nprobe = ParseIntArg(argc > 4 ? argv[4] : nullptr, 32);

    if (batch > ntotal)
    {
        std::fprintf(stderr, "batch(%zu) must be <= ntotal(%zu)\n", batch, ntotal);
        return -1;
    }

    std::printf("IVFRaBitQ small-batch add_with_ids sample\n");
    std::printf("  dim=%zu ntotal=%zu nlist=%d batch=%zu nprobe=%d\n", kDim, ntotal, nlist, batch, nprobe);
    if (std::getenv("ASCENDFAISS_MEM_DEBUG") != nullptr)
    {
        std::printf("  ASCENDFAISS_MEM_DEBUG is set\n");
    }

    std::printf("generate data\n");
    std::vector<float> data(kDim * ntotal);
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i] = static_cast<float>(drand48());
    }
    Norm(data.data(), ntotal, kDim);

    std::vector<faiss::idx_t> ids(ntotal);
    for (size_t i = 0; i < ids.size(); ++i)
    {
        ids[i] = static_cast<faiss::idx_t>(i);
    }

    faiss::ascend::AscendIndexIVFRaBitQ *index = nullptr;
    try
    {
        std::vector<int> device{0};
        const int64_t resourceSize = static_cast<int64_t>(2048) * 1024 * 1024;
        faiss::ascend::AscendIndexIVFRaBitQConfig conf(device, resourceSize);
        conf.useKmeansPP = true;

        std::printf("create index\n");
        index = new faiss::ascend::AscendIndexIVFRaBitQ(kDim, faiss::MetricType::METRIC_L2, nlist, conf);
        index->verbose = true;
        index->setNumProbes(nprobe);

        const size_t trainNum = std::min(ntotal, static_cast<size_t>(nlist) * 40ULL);
        std::printf("start train trainNum=%zu\n", trainNum);
        index->train(static_cast<faiss::idx_t>(trainNum), data.data());

        std::printf("start small-batch add_with_ids\n");
        size_t added = 0;
        size_t rounds = 0;
        while (added < ntotal)
        {
            const size_t cur = std::min(batch, ntotal - added);
            index->add_with_ids(static_cast<faiss::idx_t>(cur), data.data() + added * kDim, ids.data() + added);
            added += cur;
            ++rounds;
            if (rounds % 50 == 0 || added == ntotal)
            {
                std::printf("  progress added=%zu/%zu rounds=%zu ntotal(index)=%ld\n", added, ntotal, rounds,
                            static_cast<long>(index->ntotal));
            }
        }

        if (static_cast<size_t>(index->ntotal) != ntotal)
        {
            std::fprintf(stderr, "ntotal mismatch: expect %zu got %ld\n", ntotal, static_cast<long>(index->ntotal));
            delete index;
            return -1;
        }

        constexpr size_t kQuery = 10;
        constexpr size_t kTopK = 10;
        std::vector<float> dist(kQuery * kTopK, 0.0f);
        std::vector<faiss::idx_t> label(kQuery * kTopK, 0);
        std::printf("start search\n");
        index->search(static_cast<faiss::idx_t>(kQuery), data.data(), static_cast<faiss::idx_t>(kTopK), dist.data(),
                      label.data());

        std::printf("labels[0]:");
        for (size_t i = 0; i < kTopK; ++i)
        {
            std::printf(" %ld", static_cast<long>(label[i]));
        }
        std::printf("\n");
    }
    catch (std::exception &e)
    {
        std::printf("exception caught: %s\n", e.what());
        delete index;
        return -1;
    }

    delete index;
    std::printf("small-batch add_with_ids success (rounds with batch=%zu)\n", batch);
    return 0;
}
