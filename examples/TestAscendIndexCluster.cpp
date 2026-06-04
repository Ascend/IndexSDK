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
// python3 flat_generate_model.py -d 256 -t npu_type

#include <faiss/ascend/AscendIndexCluster.h>

#include <iostream>
#include <random>

void Genarate(std::vector<float> &addnVec, std::vector<uint32_t> &ids, int ntotal, int dim)
{
    int maxValue = 255;
    int offset = 128;
    std::vector<float> normBase(ntotal);
    auto seed = time(nullptr);
    if (seed < 0)
    {
        std::cerr << "Error: Invalid seed value." << std::endl;
        return;
    }
    std::default_random_engine e(seed);
    std::uniform_real_distribution<float> rCode(0.0f, 1.0f);

    if (dim == 0)
    {
        std::cerr << "Error: Invalid dim value." << std::endl;
        return;
    }
    for (uint32_t i = 0; i < ntotal * static_cast<uint32_t>(dim); i++)
    {
        addnVec[i] = static_cast<int8_t>(maxValue * rCode(e) - offset);
        size_t baseIdx = i / dim;
        normBase[baseIdx] += addnVec[i] * addnVec[i];
        if ((i + 1) % dim == 0)
        {
            normBase[baseIdx] = sqrt(normBase[baseIdx]);
        }
    }

    for (uint32_t i = 0; i < ntotal * static_cast<uint32_t>(dim); i++)
    {
        addnVec[i] /= normBase[i / dim];
    }

    std::iota(ids.begin(), ids.end(), 0);
}

int main(int argc, char **argv)
{
    int dim = 256;
    int ntotal = 100000;
    int capacity = 1200000;
    int64_t resourceSize = static_cast<int64_t>(2) * static_cast<int64_t>(1024 * 1024 * 1024);
    auto meticType = faiss::MetricType::METRIC_INNER_PRODUCT;
    faiss::ascend::AscendIndexCluster index;
    std::vector<int> deciveList = {0};
    auto ret = index.Init(dim, capacity, meticType, deciveList, resourceSize);
    if (ret != 0)
    {
        printf("[ERROR] Init fail ret = %d \r\n", ret);
        index.Finalize();
        return 1;
    }

    std::vector<float> addVec(static_cast<int64_t>(ntotal) * static_cast<int64_t>(dim));
    std::vector<uint32_t> ids(ntotal);
    Genarate(addVec, ids, ntotal, dim);

    ret = index.AddFeatures(ntotal, addVec.data(), ids.data());
    if (ret != 0)
    {
        printf("[ERROR] AddFeatures fail ret = %d \r\n", ret);
        index.Finalize();
        return 1;
    }

    uint32_t nq = 128;
    uint32_t start = 0;
    uint32_t codeStartIdx = 0;
    uint32_t codeNum = 1000;
    float threshold = 0.75;
    try
    {
        std::vector<uint32_t> queryIdArr(nq);
        std::iota(queryIdArr.begin(), queryIdArr.end(), start);

        std::vector<std::vector<float>> resDist(nq);
        std::vector<std::vector<uint32_t>> resIdx(nq);

        ret = index.ComputeDistanceByThreshold(queryIdArr, codeStartIdx, codeNum, threshold, true, resDist, resIdx);
        if (ret != 0)
        {
            printf("[ERROR] ComputeDistanceByThreshold fail ret = %d \r\n", ret);
            index.Finalize();
            return 1;
        }

        for (uint32_t i = 0; i < nq; i++)
        {
            uint32_t len = resDist[i].size();
            printf("queryFeature(%d/%d), %u feature dist greater than the threshold:\r\n", i, nq, len);
            for (uint32_t j = 0; j < len; j++)
            {
                printf("   id: %u, dist: %.4lf\r\n", resIdx[i][j], resDist[i][j]);
            }
        }

        index.Finalize();
    }
    catch (std::exception &e)
    {
        index.Finalize();
        printf("%s\n", e.what());
    }
}
