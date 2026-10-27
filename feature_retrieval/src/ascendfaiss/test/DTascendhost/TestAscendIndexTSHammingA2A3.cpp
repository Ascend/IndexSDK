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
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "AscendIndexTS.h"

namespace
{
constexpr uint32_t DIM = 256;
constexpr uint32_t COUNT = 16;
constexpr uint32_t TOPK = 4;
constexpr uint32_t TOKEN_NUM = 2500;
constexpr uint64_t RESOURCE_SIZE = 2ULL * 1024 * 1024 * 1024;

uint32_t GetDeviceId()
{
    const char *device = std::getenv("MX_INDEX_DEVICE");
    if (device == nullptr)
    {
        return 0;
    }
    char *end = nullptr;
    errno = 0;
    const unsigned long value = std::strtoul(device, &end, 10);
    if (errno != 0 || end == device || *end != '\0' || value > std::numeric_limits<uint32_t>::max())
    {
        return 0;
    }
    return static_cast<uint32_t>(value);
}

std::vector<uint8_t> GenerateBase(uint32_t codeSize, uint32_t count = COUNT)
{
    std::mt19937 generator(42);
    std::uniform_int_distribution<uint16_t> distribution(0, 255);
    std::vector<uint8_t> base(count * codeSize);
    std::generate(base.begin(), base.end(), [&]() { return static_cast<uint8_t>(distribution(generator)); });
    return base;
}

uint32_t HammingDistance(const std::vector<uint8_t> &base, uint32_t codeSize, uint32_t lhs, uint32_t rhs)
{
    uint32_t distance = 0;
    for (uint32_t i = 0; i < codeSize; ++i)
    {
        const uint8_t value = base[lhs * codeSize + i] ^ base[rhs * codeSize + i];
        distance += static_cast<uint32_t>(__builtin_popcount(static_cast<unsigned int>(value)));
    }
    return distance;
}

TEST(TestAscendIndexTSHammingA2A3, MatchesCpuAndAppliesFilters)
{
    if (std::getenv("MX_INDEX_MODELPATH") == nullptr)
    {
        GTEST_SKIP() << "MX_INDEX_MODELPATH is required";
    }
    constexpr uint32_t codeSize = DIM / 8;
    std::vector<uint8_t> base = GenerateBase(codeSize);
    std::vector<int64_t> ids(COUNT);
    std::vector<faiss::ascend::FeatureAttr> attrs(COUNT);
    for (uint32_t i = 0; i < COUNT; ++i)
    {
        attrs[i] = {static_cast<int32_t>(i % 2), i % 8};
    }
    std::iota(ids.begin(), ids.end(), 0);

    faiss::ascend::AscendIndexTS index;
    ASSERT_EQ(index.Init(GetDeviceId(), DIM, TOKEN_NUM, faiss::ascend::AlgorithmType::FLAT_HAMMING), 0);
    ASSERT_EQ(index.AddFeature(COUNT, base.data(), attrs.data(), ids.data()), 0);

    std::vector<uint8_t> tokenBits((TOKEN_NUM + 7) / 8, 0xff);
    faiss::ascend::AttrFilter filter{0, 0, tokenBits.data(), static_cast<uint32_t>(tokenBits.size())};
    std::vector<int64_t> labels(TOPK);
    std::vector<float> distances(TOPK);
    uint32_t validNum = 0;
    ASSERT_EQ(index.Search(1, base.data(), &filter, true, TOPK, labels.data(), distances.data(), &validNum, true), 0);
    ASSERT_EQ(validNum, TOPK);
    for (uint32_t i = 0; i < TOPK; ++i)
    {
        ASSERT_GE(labels[i], 0);
        EXPECT_EQ(attrs[labels[i]].time, 0);
        EXPECT_FLOAT_EQ(distances[i], HammingDistance(base, codeSize, 0, static_cast<uint32_t>(labels[i])));
    }

    const int64_t deleted = 0;
    ASSERT_EQ(index.FastDeleteFeatureByIndice(1, &deleted), 0);
    filter.timesEnd = 1;
    ASSERT_EQ(index.Search(1, base.data(), &filter, true, TOPK, labels.data(), distances.data(), &validNum, false), 0);
    EXPECT_EQ(std::find(labels.begin(), labels.end(), deleted), labels.end());
}

TEST(TestAscendIndexTSHammingA2A3, PreservesLastVectorWithUnalignedTail)
{
    if (std::getenv("MX_INDEX_MODELPATH") == nullptr)
    {
        GTEST_SKIP() << "MX_INDEX_MODELPATH is required";
    }
    constexpr uint32_t codeSize = DIM / 8;
    for (uint32_t count : {1U, 17U, 31U, 33U})
    {
        SCOPED_TRACE(count);
        std::vector<uint8_t> base(count * codeSize, 0);
        std::vector<int64_t> ids(count);
        std::vector<faiss::ascend::FeatureAttr> attrs(count);
        std::iota(ids.begin(), ids.end(), 0);
        for (uint32_t i = 0; i < count; ++i)
        {
            base[i * codeSize + i / 8] = static_cast<uint8_t>(1U << (i % 8));
        }
        faiss::ascend::AscendIndexTS index;
        ASSERT_EQ(index.Init(GetDeviceId(), DIM, TOKEN_NUM, faiss::ascend::AlgorithmType::FLAT_HAMMING), 0);
        ASSERT_EQ(index.AddFeature(count, base.data(), attrs.data(), ids.data()), 0);
        std::vector<uint8_t> tokenBits((TOKEN_NUM + 7) / 8, 0xff);
        faiss::ascend::AttrFilter filter{0, 0, tokenBits.data(), static_cast<uint32_t>(tokenBits.size())};
        int64_t label = -1;
        float distance = -1;
        uint32_t validNum = 0;
        ASSERT_EQ(
            index.Search(1, base.data() + (count - 1) * codeSize, &filter, true, 1, &label, &distance, &validNum, true),
            0);
        EXPECT_EQ(validNum, 1U);
        EXPECT_EQ(label, static_cast<int64_t>(count - 1));
        EXPECT_FLOAT_EQ(distance, 0);
    }
}

TEST(TestAscendIndexTSHammingA2A3, SmallBatchMatchesFullCpuTopKAcrossTiles)
{
    if (std::getenv("MX_INDEX_MODELPATH") == nullptr)
    {
        GTEST_SKIP() << "MX_INDEX_MODELPATH is required";
    }
    constexpr uint32_t count = 4097;
    constexpr uint32_t batch = 8;
    constexpr uint32_t topk = 200;
    constexpr uint32_t codeSize = DIM / 8;
    auto base = GenerateBase(codeSize, count);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<faiss::ascend::FeatureAttr> attrs(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        attrs[i] = {static_cast<int32_t>(i % 2), 0};
    }
    faiss::ascend::AscendIndexTS index;
    ASSERT_EQ(index.Init(GetDeviceId(), DIM, TOKEN_NUM, faiss::ascend::AlgorithmType::FLAT_HAMMING), 0);
    ASSERT_EQ(index.AddFeature(count, base.data(), attrs.data(), ids.data()), 0);
    std::vector<uint8_t> tokenBits((TOKEN_NUM + 7) / 8, 0xff);
    std::vector<faiss::ascend::AttrFilter> filters(batch);
    std::vector<int64_t> labels(batch * topk);
    std::vector<float> distances(batch * topk);
    std::vector<uint32_t> valid(batch);
    for (bool shared : {false, true})
    {
        SCOPED_TRACE(shared);
        for (uint32_t q = 0; q < batch; ++q)
        {
            const int32_t time = shared ? 0 : static_cast<int32_t>(q % 2);
            filters[q] = {time, time, tokenBits.data(), static_cast<uint32_t>(tokenBits.size())};
        }
        ASSERT_EQ(index.Search(batch, base.data(), filters.data(), shared, topk, labels.data(), distances.data(),
                               valid.data(), true),
                  0);
        for (uint32_t q = 0; q < batch; ++q)
        {
            SCOPED_TRACE(q);
            std::vector<uint32_t> expected;
            for (uint32_t i = 0; i < count; ++i)
            {
                if (attrs[i].time == filters[q].timesStart)
                {
                    expected.push_back(HammingDistance(base, codeSize, q, i));
                }
            }
            std::sort(expected.begin(), expected.end());
            ASSERT_EQ(valid[q], topk);
            std::vector<bool> seen(count, false);
            for (uint32_t k = 0; k < topk; ++k)
            {
                const auto label = labels[q * topk + k];
                ASSERT_GE(label, 0);
                ASSERT_LT(label, count);
                EXPECT_FALSE(seen[label]);
                seen[label] = true;
                EXPECT_EQ(attrs[label].time, filters[q].timesStart);
                EXPECT_FLOAT_EQ(distances[q * topk + k], expected[k]);
                EXPECT_FLOAT_EQ(distances[q * topk + k], HammingDistance(base, codeSize, q, label));
            }
        }
    }
}

TEST(TestAscendIndexTSHammingA2A3, AppliesExtraValAndFastDeleteTogether)
{
    if (std::getenv("MX_INDEX_MODELPATH") == nullptr)
    {
        GTEST_SKIP() << "MX_INDEX_MODELPATH is required";
    }
    constexpr uint32_t codeSize = DIM / 8;
    std::vector<uint8_t> base = GenerateBase(codeSize);
    std::vector<int64_t> ids(COUNT);
    std::vector<faiss::ascend::FeatureAttr> attrs(COUNT);
    std::vector<faiss::ascend::ExtraValAttr> extraAttrs(COUNT);
    for (uint32_t i = 0; i < COUNT; ++i)
    {
        attrs[i] = {0, i % 8};
        extraAttrs[i].val = static_cast<int16_t>(1U << (i % TOPK));
    }
    std::iota(ids.begin(), ids.end(), 0);

    faiss::ascend::AscendIndexTS index;
    ASSERT_EQ(index.InitWithExtraVal(GetDeviceId(), DIM, TOKEN_NUM, RESOURCE_SIZE,
                                     faiss::ascend::AlgorithmType::FLAT_HAMMING),
              0);
    ASSERT_EQ(index.AddWithExtraVal(COUNT, base.data(), attrs.data(), ids.data(), extraAttrs.data()), 0);

    std::vector<uint8_t> tokenBits((TOKEN_NUM + 7) / 8, 0xff);
    faiss::ascend::AttrFilter filter{0, 0, tokenBits.data(), static_cast<uint32_t>(tokenBits.size())};
    faiss::ascend::ExtraValFilter extraFilter{1, 0};
    std::vector<int64_t> labels(TOPK);
    std::vector<float> distances(TOPK);
    uint32_t validNum = 0;
    ASSERT_EQ(index.SearchWithExtraVal(1, base.data(), &filter, false, TOPK, labels.data(), distances.data(), &validNum,
                                       &extraFilter, false),
              0);
    ASSERT_EQ(validNum, TOPK);
    for (auto label : labels)
    {
        ASSERT_GE(label, 0);
        EXPECT_NE(extraAttrs[label].val & extraFilter.filterVal, 0);
    }

    const int64_t deleted = 0;
    ASSERT_EQ(index.FastDeleteFeatureByIndice(1, &deleted), 0);
    ASSERT_EQ(index.SearchWithExtraVal(1, base.data(), &filter, false, TOPK, labels.data(), distances.data(), &validNum,
                                       &extraFilter, false),
              0);
    EXPECT_EQ(std::find(labels.begin(), labels.end(), deleted), labels.end());
}
}  // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
