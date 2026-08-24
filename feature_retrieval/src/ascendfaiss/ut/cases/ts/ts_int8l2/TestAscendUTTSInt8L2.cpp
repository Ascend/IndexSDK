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
#include <cstdlib>
#include <mockcpp/mockcpp.hpp>

#include "AscendIndexTS.h"
#include "ErrorCode.h"
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
[[maybe_unused]] const bool DISABLE_ACL_FINALIZE = []()
{
    setenv("MX_INDEX_FINALIZE", "0", 1);
    return true;
}();

struct Check910BItem
{
    uint32_t dim;
    uint32_t ntotal;
    std::string socName;
    bool shareAttrFilter;
};

class TestAscendIndexTSInt8L2UT910B : public TestWithParam<Check910BItem>
{
};

const Check910BItem ITEMS910B[] = {{128, 1000, "Ascend910B2", true}, {256, 30000, "Ascend910_9382", false}};

std::vector<uint8_t> BuildAllTokenBitSet(uint32_t tokenNum)
{
    std::vector<uint8_t> bitSet((tokenNum + 7) / 8, 0xff);
    uint32_t validBits = tokenNum % 8;
    if (validBits != 0)
    {
        bitSet.back() = static_cast<uint8_t>((1U << validBits) - 1U);
    }
    return bitSet;
}

void TestSearch(uint32_t ntotal, uint32_t dim, const std::string &socName, bool shareAttrFilter)
{
    MOCKER_CPP(&aclrtGetSocName).stubs().will(returnValue(socName.c_str()));
    faiss::ascend::SocUtils::GetInstance().Init();

    EXPECT_TRUE(faiss::ascend::SocUtils::GetInstance().IsAscend910B());
    EXPECT_FALSE(faiss::ascend::SocUtils::GetInstance().IsZZCodeFormat());
    EXPECT_EQ(faiss::ascend::SocUtils::CodeFormatType::FORMAT_TYPE_ND,
              faiss::ascend::SocUtils::GetInstance().GetCodeFormatType());

    constexpr uint32_t deviceId = 0;
    constexpr uint32_t tokenNum = 2500;
    constexpr uint32_t queryNum = 4;
    constexpr uint32_t topk = 10;

    faiss::ascend::AscendIndexTS tsIndex;
    auto ret = tsIndex.Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_L2_INT8);
    EXPECT_EQ(ret, 0);

    std::vector<int8_t> features(static_cast<size_t>(ntotal) * dim);
    ascend::FeatureGenerator(features);
    std::vector<int64_t> labels;
    for (uint32_t i = 0; i < ntotal; ++i)
    {
        labels.emplace_back(i);
    }
    std::vector<faiss::ascend::FeatureAttr> attrs(ntotal);
    ascend::FeatureAttrGenerator(attrs);
    ret = tsIndex.AddFeature(ntotal, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(ret, 0);

    std::vector<int8_t> queries(queryNum * dim);
    queries.assign(features.begin(), features.begin() + queries.size());
    auto tokenBitSet = BuildAllTokenBitSet(tokenNum);
    faiss::ascend::AttrFilter filter{};
    filter.timesStart = 0;
    filter.timesEnd = 3;
    filter.tokenBitSet = tokenBitSet.data();
    filter.tokenBitSetLen = static_cast<uint32_t>(tokenBitSet.size());
    std::vector<faiss::ascend::AttrFilter> queryFilters(queryNum, filter);

    std::vector<float> distances(queryNum * topk, -1);
    std::vector<int64_t> labelRes(queryNum * topk, -1);
    std::vector<uint32_t> validnum(queryNum, 0);
    ret = tsIndex.Search(queryNum, queries.data(), queryFilters.data(), shareAttrFilter, topk, labelRes.data(),
                         distances.data(), validnum.data());
    EXPECT_EQ(ret, 0);

    GlobalMockObject::verify();
    faiss::ascend::SocUtils::GetInstance().Init();
}

TEST_P(TestAscendIndexTSInt8L2UT910B, Search)
{
    Check910BItem item = GetParam();
    TestSearch(item.ntotal, item.dim, item.socName, item.shareAttrFilter);
}

INSTANTIATE_TEST_CASE_P(Int8L2CheckGroup, TestAscendIndexTSInt8L2UT910B, ::testing::ValuesIn(ITEMS910B));
}  // namespace
