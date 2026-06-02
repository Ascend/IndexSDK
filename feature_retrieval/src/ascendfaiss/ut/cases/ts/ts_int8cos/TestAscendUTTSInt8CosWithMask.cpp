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

#include <gtest/gtest.h>

#include <mockcpp/mockcpp.hpp>

#include "AscendIndexTS.h"
#include "acl.h"
#include "common/utils/SocUtils.h"
#include "faiss/impl/AuxIndexStructures.h"
#include "faiss/impl/IDSelector.h"
#include "fp16.h"
#include "ut/Common.h"

using namespace testing;
using namespace std;
namespace
{

struct Check910BItem
{
    size_t dim;
    uint32_t ntotal;
    faiss::MetricType metricType;
    std::string str;
};

class TestAscendIndexTSUT910B : public TestWithParam<Check910BItem>
{
};

const Check910BItem ITEMS910B[] = {{64, 1000, faiss::METRIC_INNER_PRODUCT, "Ascend910B1"},
                                   {128, 30000, faiss::METRIC_INNER_PRODUCT, "Ascend910B2"},
                                   {256, 1000, faiss::METRIC_L2, "Ascend910B3"},
                                   {384, 30000, faiss::METRIC_L2, "Ascend910B4"}};

void TestSearch(uint32_t ntotal, faiss::MetricType metricType, int dim, std::string socName)
{
    // 打桩socName为910B*
    MOCKER_CPP(&aclrtGetSocName).stubs().will(returnValue(socName.c_str()));

    EXPECT_TRUE(faiss::ascend::SocUtils::GetInstance().IsAscend910B());
    // 910B为ND格式
    EXPECT_FALSE(faiss::ascend::SocUtils::GetInstance().IsZZCodeFormat());
    EXPECT_EQ(faiss::ascend::SocUtils::CodeFormatType::FORMAT_TYPE_ND,
              faiss::ascend::SocUtils::GetInstance().GetCodeFormatType());

    std::vector<int> queryNums = {1};
    std::vector<int> topks = {10};
    int tokenNum = 2500;
    faiss::ascend::AscendIndexTS* tsIndex = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex->Init(0, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(ret, 0);

    printf("[add -----------]\n");
    std::vector<int8_t> features(ntotal * dim);
    ascend::FeatureGenerator(features);

    std::vector<int64_t> labels;
    for (int64_t j = 0; j < ntotal; ++j)
    {
        labels.emplace_back(j);
    }
    std::vector<faiss::ascend::FeatureAttr> attrs(ntotal);
    ascend::FeatureAttrGenerator(attrs);
    auto res = tsIndex->AddFeature(ntotal, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(res, 0);

    for (auto k : topks)
    {
        int loopTimes = 1;
        for (auto queryNum : queryNums)
        {
            std::vector<float> distances(queryNum * k, -1);
            std::vector<int64_t> labelRes(queryNum * k, -1);
            std::vector<uint32_t> validnum(queryNum, 1);
            uint32_t size = queryNum * dim;
            std::vector<int8_t> querys(size);
            querys.assign(features.begin(), features.begin() + size);

            uint32_t setlen = (uint32_t)((tokenNum + 7) / 8);
            std::vector<uint8_t> bitSet(setlen, 0);

            // 00000111   -> 0,1,2
            bitSet[0] = 0x1 << 0 | 0x1 << 1 | 0x1 << 2 | 0x1 << 3;

            faiss::ascend::AttrFilter filter{};
            filter.timesStart = 0;
            filter.timesEnd = 3;
            filter.tokenBitSet = bitSet.data();
            filter.tokenBitSetLen = setlen;

            std::vector<faiss::ascend::AttrFilter> queryFilters(queryNum, filter);
            for (int i = 0; i < loopTimes; i++)
            {
                tsIndex->Search(queryNum, querys.data(), queryFilters.data(), false, k, labelRes.data(),
                                distances.data(), validnum.data());
            }
        }
    }
    delete tsIndex;

    // mockcpp 需要显示调用该函数来恢复打桩
    GlobalMockObject::verify();
}

void TestSearchWithExtraMask(uint32_t ntotal, faiss::MetricType metricType, int dim, std::string socName)
{
    // 打桩socName为910B*
    MOCKER_CPP(&aclrtGetSocName).stubs().will(returnValue(socName.c_str()));

    EXPECT_TRUE(faiss::ascend::SocUtils::GetInstance().IsAscend910B());
    // 910B为ND格式
    EXPECT_FALSE(faiss::ascend::SocUtils::GetInstance().IsZZCodeFormat());
    EXPECT_EQ(faiss::ascend::SocUtils::CodeFormatType::FORMAT_TYPE_ND,
              faiss::ascend::SocUtils::GetInstance().GetCodeFormatType());

    std::vector<int> queryNums = {1};
    std::vector<int> topks = {10};
    int tokenNum = 2500;
    faiss::ascend::AscendIndexTS* tsIndex = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex->Init(0, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(ret, 0);

    printf("[add -----------]\n");
    std::vector<int8_t> features(ntotal * dim);
    ascend::FeatureGenerator(features);

    std::vector<int64_t> labels;
    for (int64_t j = 0; j < ntotal; ++j)
    {
        labels.emplace_back(j);
    }
    std::vector<faiss::ascend::FeatureAttr> attrs(ntotal);
    ascend::FeatureAttrGenerator(attrs);
    auto res = tsIndex->AddFeature(ntotal, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(res, 0);

    for (auto k : topks)
    {
        for (auto queryNum : queryNums)
        {
            std::vector<float> distances(queryNum * k, -1);
            std::vector<int64_t> labelRes(queryNum * k, -1);
            std::vector<uint32_t> validnum(queryNum, 1);
            uint32_t size = queryNum * dim;
            std::vector<uint8_t> querys(size);
            querys.assign(features.begin(), features.begin() + size);

            uint32_t setlen = (uint32_t)((tokenNum + 7) / 8);
            std::vector<uint8_t> bitSet(setlen, ~0);

            // 00000111   -> 0,1,2
            // bitSet[0] = 0x1 << 0 | 0x1 << 1 | 0x1 << 2;

            faiss::ascend::AttrFilter filter{};
            filter.timesStart = 0;
            filter.timesEnd = 3;
            filter.tokenBitSet = bitSet.data();
            filter.tokenBitSetLen = setlen;

            int extraMaskLen = 12;
            std::vector<uint8_t> extraMask(queryNum * extraMaskLen, 0);
            int ind = 1;
            for (int i = 0; i < queryNum; i++)
            {
                for (int j = 0; j < extraMaskLen; j++)
                {
                    extraMask[i * extraMaskLen + j] = 0x1 << (ind % 8);
                    ind++;
                }
            }
            std::vector<faiss::ascend::AttrFilter> queryFilters(queryNum, filter);
            ret = tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), false, k, extraMask.data(),
                                               extraMaskLen, false, labelRes.data(), distances.data(), validnum.data());
            EXPECT_EQ(ret, 0);
        }
    }

    delete tsIndex;

    // mockcpp 需要显示调用该函数来恢复打桩
    GlobalMockObject::verify();
}

TEST_P(TestAscendIndexTSUT910B, Search)
{
    Check910BItem item = GetParam();
    int dim = item.dim;
    uint32_t ntotal = item.ntotal;
    faiss::MetricType metricType = item.metricType;
    std::string socName = item.str;

    TestSearch(ntotal, metricType, dim, socName);
    TestSearchWithExtraMask(ntotal, metricType, dim, socName);
}

INSTANTIATE_TEST_CASE_P(Int8FlatCheckGroup, TestAscendIndexTSUT910B, ::testing::ValuesIn(ITEMS910B));
}  // namespace
