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

 
#include <numeric>
#include <cmath>
#include <random>
#include <cstring>
#include <sys/time.h>
#include <faiss/index_io.h>
#include <cstdlib>
#include <gtest/gtest.h>
#include "Common.h"
#include "faiss/ascend/custom/AscendIndexFlatAT.h"
#include "faiss/ascend/AscendCloner.h"
#include "faiss/IndexFlat.h"
#include "faiss/impl/AuxIndexStructures.h"

namespace ascend {
constexpr int DIM = 64;
constexpr int K = 1;
constexpr size_t BASE_SIZE = 8192;
const std::vector<int> DEVICES = { 1, 3 };

TEST(TestAscendIndexUTFlatAT, All)
{
    std::vector<float> data(DIM * BASE_SIZE);
    FeatureGenerator(data);

    faiss::ascend::AscendIndexFlatATConfig conf(DEVICES);
    faiss::ascend::AscendIndexFlatAT index(DIM, BASE_SIZE, conf);

    index.add(BASE_SIZE, data.data());
    EXPECT_EQ(static_cast<size_t>(index.ntotal), BASE_SIZE);

    {
        int batch = 32;
        for (size_t i = BASE_SIZE - batch; i < BASE_SIZE; i += batch) {
            std::vector<float> dist(K * batch, 0);
            std::vector<faiss::idx_t> label(K * batch, 0);
            index.search(batch, data.data() + i * DIM, K, dist.data(), label.data());
            faiss::idx_t assign;
            index.assign(1, data.data() + i * DIM, &assign);
        }
    }
}

}