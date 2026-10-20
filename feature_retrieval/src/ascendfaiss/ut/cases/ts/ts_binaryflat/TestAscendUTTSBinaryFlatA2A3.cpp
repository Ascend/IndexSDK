/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
 *
 * IndexSDK is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of the License at:
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

#include <mockcpp/mockcpp.hpp>
#include <numeric>
#include <string>
#include <vector>

#include "AscendIndexTS.h"
#include "acl.h"
#include "common/utils/SocUtils.h"

namespace
{
constexpr uint32_t DIM = 256;
constexpr uint32_t CODE_SIZE = DIM / 8;
constexpr uint32_t TOKEN_NUM = 2500;
constexpr uint32_t BASE_SIZE = 16;
constexpr uint32_t BATCH = 2;
constexpr uint32_t TOPK = 4;

class TestTSBinaryFlatA2A3 : public testing::TestWithParam<const char *>
{
   protected:
    void SetUp() override
    {
        socName = GetParam();
        MOCKER_CPP(&aclrtGetSocName).stubs().will(mockcpp::returnValue(socName.c_str()));
        faiss::ascend::SocUtils::GetInstance().Init();
    }

    void TearDown() override { mockcpp::GlobalMockObject::verify(); }

    void AddBase(faiss::ascend::AscendIndexTS &index)
    {
        std::iota(labels.begin(), labels.end(), 0);
        for (uint32_t i = 0; i < BASE_SIZE; ++i)
        {
            features[i * CODE_SIZE] = static_cast<uint8_t>(i);
            attrs[i] = {static_cast<int32_t>(i), i % TOKEN_NUM};
        }
        ASSERT_EQ(index.AddFeature(BASE_SIZE, features.data(), attrs.data(), labels.data()), 0);
    }

    std::string socName;
    std::vector<uint8_t> features = std::vector<uint8_t>(BASE_SIZE * CODE_SIZE, 0);
    std::vector<int64_t> labels = std::vector<int64_t>(BASE_SIZE);
    std::vector<faiss::ascend::FeatureAttr> attrs = std::vector<faiss::ascend::FeatureAttr>(BASE_SIZE);
};

TEST_P(TestTSBinaryFlatA2A3, InitializesHamming)
{
    faiss::ascend::AscendIndexTS index;
    EXPECT_EQ(index.Init(0, DIM, TOKEN_NUM, faiss::ascend::AlgorithmType::FLAT_HAMMING), 0);
}

TEST_P(TestTSBinaryFlatA2A3, SearchesWithFullTokenFilter)
{
    faiss::ascend::AscendIndexTS index;
    ASSERT_EQ(index.Init(0, DIM, TOKEN_NUM, faiss::ascend::AlgorithmType::FLAT_HAMMING), 0);
    AddBase(index);

    std::vector<uint8_t> tokenBits((TOKEN_NUM + 7) / 8, 0xff);
    tokenBits.back() = static_cast<uint8_t>((1U << (TOKEN_NUM % 8)) - 1U);
    faiss::ascend::AttrFilter filter{0, 0, tokenBits.data(), static_cast<uint32_t>(tokenBits.size())};
    std::vector<faiss::ascend::AttrFilter> filters(BATCH, filter);
    std::vector<int64_t> resultLabels(BATCH * TOPK);
    std::vector<float> distances(BATCH * TOPK);
    std::vector<uint32_t> validNums(BATCH);

    EXPECT_EQ(index.Search(BATCH, features.data(), filters.data(), false, TOPK, resultLabels.data(), distances.data(),
                           validNums.data(), false),
              0);
}

TEST_P(TestTSBinaryFlatA2A3, SearchesWithPartialTokenFilter)
{
    faiss::ascend::AscendIndexTS index;
    ASSERT_EQ(index.Init(0, DIM, TOKEN_NUM, faiss::ascend::AlgorithmType::FLAT_HAMMING), 0);
    AddBase(index);

    std::vector<uint8_t> firstTokenBits((TOKEN_NUM + 7) / 8, 0xff);
    std::vector<uint8_t> secondTokenBits = firstTokenBits;
    secondTokenBits[0] = 0x0f;
    std::vector<faiss::ascend::AttrFilter> filters{
        {0, 0, firstTokenBits.data(), static_cast<uint32_t>(firstTokenBits.size())},
        {0, 0, secondTokenBits.data(), static_cast<uint32_t>(secondTokenBits.size())}};
    std::vector<int64_t> resultLabels(BATCH * TOPK);
    std::vector<float> distances(BATCH * TOPK);
    std::vector<uint32_t> validNums(BATCH);

    EXPECT_EQ(index.Search(BATCH, features.data(), filters.data(), false, TOPK, resultLabels.data(), distances.data(),
                           validNums.data(), false),
              0);
}

TEST_P(TestTSBinaryFlatA2A3, SearchesWithBaseMaskAfterDelete)
{
    faiss::ascend::AscendIndexTS index;
    ASSERT_EQ(index.Init(0, DIM, TOKEN_NUM, faiss::ascend::AlgorithmType::FLAT_HAMMING), 0);
    AddBase(index);
    const int64_t deletedIndex = 0;
    ASSERT_EQ(index.FastDeleteFeatureByIndice(1, &deletedIndex), 0);

    std::vector<uint8_t> tokenBits((TOKEN_NUM + 7) / 8, 0xff);
    faiss::ascend::AttrFilter filter{0, 0, tokenBits.data(), static_cast<uint32_t>(tokenBits.size())};
    std::vector<int64_t> resultLabels(TOPK);
    std::vector<float> distances(TOPK);
    uint32_t validNum = 0;

    EXPECT_EQ(
        index.Search(1, features.data(), &filter, true, TOPK, resultLabels.data(), distances.data(), &validNum, false),
        0);
}

INSTANTIATE_TEST_SUITE_P(AtlasA2A3, TestTSBinaryFlatA2A3, testing::Values("Ascend910B2", "Ascend910_9382"));
}  // namespace
