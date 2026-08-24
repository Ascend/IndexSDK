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
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "AscendIndexTS.h"

namespace
{
using faiss::ascend::AlgorithmType;
using faiss::ascend::AscendIndexTS;
using faiss::ascend::AttrFilter;
using faiss::ascend::FeatureAttr;

constexpr uint32_t BASE_COUNT = 128;
constexpr uint32_t TOKEN_NUM = 2500;

uint32_t GetDeviceId()
{
    const char *value = std::getenv("MX_INDEX_DEVICE");
    return value == nullptr ? 0 : static_cast<uint32_t>(std::stoul(value));
}

std::vector<uint8_t> AllTokens()
{
    std::vector<uint8_t> bits((TOKEN_NUM + 7) / 8, 0xff);
    bits.back() = static_cast<uint8_t>((1U << (TOKEN_NUM % 8)) - 1U);
    return bits;
}

struct IndexData
{
    explicit IndexData(uint32_t dim)
    {
        base.resize(static_cast<size_t>(BASE_COUNT) * dim);
        std::vector<int64_t> labels(BASE_COUNT);
        std::vector<FeatureAttr> attrs(BASE_COUNT);
        for (uint32_t row = 0; row < BASE_COUNT; ++row)
        {
            labels[row] = row;
            attrs[row] = {static_cast<int32_t>(row % 2), row % TOKEN_NUM};
            for (uint32_t col = 0; col < dim; ++col)
            {
                base[static_cast<size_t>(row) * dim + col] = static_cast<int8_t>((row * 31 + col * 17) % 127 - 63);
            }
        }
        EXPECT_EQ(index.Init(GetDeviceId(), dim, TOKEN_NUM, AlgorithmType::FLAT_L2_INT8), 0);
        EXPECT_EQ(index.AddFeature(BASE_COUNT, base.data(), attrs.data(), labels.data()), 0);
    }

    AscendIndexTS index;
    std::vector<int8_t> base;
};

void SearchAndCheck(IndexData &data, uint32_t dim, uint32_t batch, uint32_t topk, bool checkCpu = false,
                    bool shareAttrFilter = false)
{
    std::vector<int8_t> queries(static_cast<size_t>(batch) * dim);
    for (uint32_t query = 0; query < batch; ++query)
    {
        const uint32_t source = 2 * (query % (BASE_COUNT / 2));
        std::copy_n(data.base.data() + static_cast<size_t>(source) * dim, dim,
                    queries.data() + static_cast<size_t>(query) * dim);
    }
    auto tokenBits = AllTokens();
    AttrFilter filter{0, 0, tokenBits.data(), static_cast<uint32_t>(tokenBits.size())};
    std::vector<AttrFilter> filters(batch, filter);
    std::vector<int64_t> labels(static_cast<size_t>(batch) * topk, -1);
    std::vector<float> distances(static_cast<size_t>(batch) * topk, -1.0F);
    std::vector<uint32_t> validNums(batch, 0);
    const AttrFilter *searchFilters = shareAttrFilter ? &filter : filters.data();
    ASSERT_EQ(data.index.Search(batch, queries.data(), searchFilters, shareAttrFilter, topk, labels.data(),
                                distances.data(), validNums.data(), true),
              0);
    for (uint32_t query = 0; query < batch; ++query)
    {
        const size_t offset = static_cast<size_t>(query) * topk;
        EXPECT_EQ(labels[offset], 2 * (query % (BASE_COUNT / 2)));
        EXPECT_FLOAT_EQ(distances[offset], 0.0F);
        EXPECT_EQ(validNums[query], std::min(topk, BASE_COUNT / 2));
        if (checkCpu)
        {
            for (uint32_t rank = 0; rank < validNums[query]; ++rank)
            {
                float squared = 0;
                for (uint32_t col = 0; col < dim; ++col)
                {
                    const float delta = queries[static_cast<size_t>(query) * dim + col] -
                                        data.base[static_cast<size_t>(labels[offset + rank]) * dim + col];
                    squared += delta * delta;
                }
                EXPECT_NEAR(distances[offset + rank], std::sqrt(squared), 1.0F);
                EXPECT_EQ(labels[offset + rank] % 2, 0);
            }
        }
    }
}

TEST(TestAscendIndexTSA2A3, Int8L2AcceptanceMatrix)
{
    if (std::getenv("MX_INDEX_MODELPATH") == nullptr)
    {
        GTEST_SKIP() << "MX_INDEX_MODELPATH is required";
    }
    for (uint32_t dim : {64U, 128U, 256U, 384U, 512U, 768U, 1024U})
    {
        IndexData data(dim);
        SearchAndCheck(data, dim, 4, 5, dim == 256);
    }
    IndexData data(256);
    for (uint32_t batch : {1U, 2U, 4U, 6U, 8U, 12U, 16U, 18U, 24U, 32U, 36U, 48U, 64U, 128U, 256U, 10240U})
    {
        SearchAndCheck(data, 256, batch, 10);
    }
    SearchAndCheck(data, 256, 64, 10, false, true);
    for (uint32_t topk : {1U, 10U, 100U, 1000U, 10000U, 100000U})
    {
        SearchAndCheck(data, 256, 1, topk);
    }
}
}  // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
