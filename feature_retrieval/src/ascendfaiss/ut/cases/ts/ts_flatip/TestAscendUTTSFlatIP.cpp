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

class TestAscendIndexTSFlatIPUT910B : public TestWithParam<Check910BItem>
{
};

const Check910BItem ITEMS910B[] = {{64, 1000, faiss::METRIC_INNER_PRODUCT, "Ascend910B4"},
                                   {128, 30000, faiss::METRIC_INNER_PRODUCT, "Ascend910B3"}};

void TestSearch(uint32_t ntotal, faiss::MetricType metricType, int dim, std::string socName)
{
    // 打桩socName为910B*
    MOCKER_CPP(&aclrtGetSocName).stubs().will(returnValue(socName.c_str()));

    EXPECT_TRUE(faiss::ascend::SocUtils::GetInstance().IsAscend910B());
    // 910B为ND格式
    EXPECT_FALSE(faiss::ascend::SocUtils::GetInstance().IsZZCodeFormat());
    EXPECT_EQ(faiss::ascend::SocUtils::CodeFormatType::FORMAT_TYPE_ND,
              faiss::ascend::SocUtils::GetInstance().GetCodeFormatType());

    int tokenNum = 2500;
    int k = 10;
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto res = tsIndex->Init(0, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_IP_FP16);
    EXPECT_EQ(res, 0);
    std::vector<float> features(ntotal * dim);
    ascend::FeatureGenerator(features);
    std::vector<int64_t> labels;
    for (size_t i = 0; i < ntotal; ++i)
    {
        labels.push_back(i);
    }
    std::vector<faiss::ascend::FeatureAttr> attrs(ntotal);
    ascend::FeatureAttrGenerator(attrs);
    res = tsIndex->AddFeature(ntotal, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(res, 0);

    int queryNum = 4;
    std::vector<float> distances(queryNum * k, -1);
    std::vector<int64_t> labelRes(queryNum * k, 10);
    std::vector<uint32_t> validnum(queryNum, 0);
    uint32_t size = queryNum * dim;
    std::vector<float> querys(size);
    querys.assign(features.begin(), features.begin() + size);

    uint32_t setlen = static_cast<uint32_t>(((tokenNum + 7) / 8));
    std::vector<uint8_t> bitSet(setlen, 0);
    bitSet[0] = 0x1 << 0 | 0x1 << 1 | 0x1 << 2 | 0x1 << 3;
    faiss::ascend::AttrFilter filter{};
    filter.timesStart = 0;
    filter.timesEnd = 3;
    filter.tokenBitSet = bitSet.data();
    filter.tokenBitSetLen = setlen;

    std::vector<faiss::ascend::AttrFilter> queryFilters(queryNum, filter);

    res = tsIndex->Search(queryNum, querys.data(), queryFilters.data(), false, k, labelRes.data(), distances.data(),
                          validnum.data());
    EXPECT_EQ(res, 0);

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

    int tokenNum = 2500;
    int k = 10;
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto res = tsIndex->Init(0, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_IP_FP16);
    EXPECT_EQ(res, 0);
    std::vector<float> features(ntotal * dim);
    ascend::FeatureGenerator(features);
    std::vector<int64_t> labels;
    for (size_t i = 0; i < ntotal; ++i)
    {
        labels.push_back(i);
    }
    std::vector<faiss::ascend::FeatureAttr> attrs(ntotal);
    ascend::FeatureAttrGenerator(attrs);
    res = tsIndex->AddFeature(ntotal, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(res, 0);

    int queryNum = 4;
    std::vector<float> distances(queryNum * k, -1);
    std::vector<int64_t> labelRes(queryNum * k, 10);
    std::vector<uint32_t> validnum(queryNum, 0);
    uint32_t size = queryNum * dim;
    std::vector<float> querys(size);
    querys.assign(features.begin(), features.begin() + size);

    uint32_t setlen = static_cast<uint32_t>(((tokenNum + 7) / 8));
    std::vector<uint8_t> bitSet(setlen, 0);
    bitSet[0] = 0x1 << 0 | 0x1 << 1 | 0x1 << 2 | 0x1 << 3;
    faiss::ascend::AttrFilter filter{};
    filter.timesStart = 0;
    filter.timesEnd = 3;
    filter.tokenBitSet = bitSet.data();
    filter.tokenBitSetLen = setlen;

    std::vector<faiss::ascend::AttrFilter> queryFilters(queryNum, filter);

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
    res = tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), false, k, extraMask.data(),
                                       extraMaskLen, false, labelRes.data(), distances.data(), validnum.data());
    EXPECT_EQ(res, 0);
    delete tsIndex;

    // mockcpp 需要显示调用该函数来恢复打桩
    GlobalMockObject::verify();
}

TEST_P(TestAscendIndexTSFlatIPUT910B, Search)
{
    Check910BItem item = GetParam();
    int dim = item.dim;
    uint32_t ntotal = item.ntotal;
    faiss::MetricType metricType = item.metricType;
    std::string socName = item.str;

    TestSearch(ntotal, metricType, dim, socName);
    TestSearchWithExtraMask(ntotal, metricType, dim, socName);
}

INSTANTIATE_TEST_CASE_P(FlatIPCheckGroup, TestAscendIndexTSFlatIPUT910B, ::testing::ValuesIn(ITEMS910B));
}  // namespace
