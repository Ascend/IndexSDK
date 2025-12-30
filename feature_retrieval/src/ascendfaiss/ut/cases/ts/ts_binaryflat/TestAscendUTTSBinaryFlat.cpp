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
#include "common/utils/SocUtils.h"
#include "ut/Common.h"
#include "AscendIndexTS.h"
#include "faiss/impl/AuxIndexStructures.h"
#include "faiss/impl/IDSelector.h"
#include "acl.h"
#include "fp16.h"

using namespace testing;
using namespace std;
namespace {
const int64_t MAX_N = 1e9;
const int64_t MAX_K = 4096;
struct CheckInitItem {
    uint32_t deviceId;
    uint32_t tokenNum;
};

struct CheckGetItem {
    int64_t count;
    int64_t *labels;
    faiss::ascend::ExtraValAttr *extraVal;
};

struct CheckAddItem {
    void *features;
    faiss::ascend::FeatureAttr *attributes;
    int64_t *labels;
};

class TestTSBinaryFlatInitUT : public TestWithParam<CheckInitItem> {
};

class TestTSBinaryFlatGetUT : public TestWithParam<CheckGetItem> {
};

class TestTSBinaryFlatAddUT : public TestWithParam<CheckAddItem> {
};

enum class ErrorCode {
    APP_ERR_OK = 0,
    APP_ERR_INVALID_PARAM = 2001,
    APP_ERR_ILLEGAL_OPERATION = 2009
};

const int BINARY_BYTE_SIZE = 8;
const int DEVICE = 0;
const int DIM = 256;
const int BASE_SIZE = 100;
const int RANGEMIN = 0;
const int RANGEMAX = 4;
const uint32_t DEFAULT_TOKEN = 2500;
const int64_t DEFAULT_MEM = 0x60000000;
std::vector<int64_t> g_label(BASE_SIZE);
std::vector<uint8_t> g_features(BASE_SIZE * DIM / BINARY_BYTE_SIZE);
std::vector<faiss::ascend::ExtraValAttr> g_extraVal(g_label.size());
std::vector<faiss::ascend::FeatureAttr> g_featureAttr(g_label.size());
const CheckInitItem INITITEMS[] = {
    { 10000, DEFAULT_TOKEN },
    { DEVICE, 300001 }
};

const CheckGetItem GETITEMS[] = {
    { 0, g_label.data(), g_extraVal.data() },
    { BASE_SIZE, nullptr, g_extraVal.data() },
    { BASE_SIZE, g_label.data(), nullptr }
};

const CheckAddItem ADDITEMS[] = {
    { nullptr, g_featureAttr.data(),  g_label.data() },
    { g_features.data(), nullptr,  g_label.data() },
    { g_features.data(), g_featureAttr.data(), nullptr}
};

TEST_P(TestTSBinaryFlatInitUT, test_init_invalid_input)
{
    CheckInitItem item = GetParam();
    uint32_t device = item.deviceId;
    uint32_t tokenNum = item.tokenNum;
    faiss::ascend::AscendIndexTS tsIndex;
    auto ret = tsIndex.Init(device, DIM, tokenNum);
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_INVALID_PARAM));
}

TEST_P(TestTSBinaryFlatGetUT, test_get_invalid_input)
{
    CheckGetItem item = GetParam();
    int64_t count = item.count;
    int64_t *labelsGet = item.labels;
    faiss::ascend::ExtraValAttr *extraVal = item.extraVal;

    std::vector<uint8_t> base(BASE_SIZE * DIM / 8);
    std::vector<int64_t> labels(BASE_SIZE);
    std::iota(labels.begin(), labels.end(), 0);
    std::vector<faiss::ascend::FeatureAttr> attrs(BASE_SIZE);

    faiss::ascend::AscendIndexTS tsIndex;
    auto ret = tsIndex.InitWithExtraVal(DEVICE, DIM, DEFAULT_TOKEN, DEFAULT_MEM,
        faiss::ascend::AlgorithmType::FLAT_HAMMING);
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));
    ret = tsIndex.AddFeature(BASE_SIZE, base.data(), attrs.data(), labels.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));

    ret = tsIndex.GetExtraValAttrByLabel(count, labelsGet, extraVal);
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_INVALID_PARAM));
}

TEST_P(TestTSBinaryFlatAddUT, test_add_invalid_input)
{
    CheckAddItem item = GetParam();
    int64_t *labelsAdd = item.labels;
    faiss::ascend::FeatureAttr *attrs = item.attributes;
    void *features = item.features;

    faiss::ascend::AscendIndexTS tsIndex;
    auto ret = tsIndex.InitWithExtraVal(DEVICE, DIM, DEFAULT_TOKEN, DEFAULT_MEM,
        faiss::ascend::AlgorithmType::FLAT_HAMMING);
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));
    ret = tsIndex.AddFeature(BASE_SIZE, features, attrs, labelsAdd);
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_ILLEGAL_OPERATION));
}

bool IsExtraAttrSame(const faiss::ascend::ExtraValAttr &lAttr, const faiss::ascend::ExtraValAttr &rAttr)
{
    return (lAttr.val == rAttr.val);
}
TEST(TestAscendIndexTSUT, GetExtraValAttrByLabel)
{
    int64_t ntotal = 100;
    uint32_t addNum = 1;
    uint32_t deviceId = 0;
    uint64_t resources = 1024 * 1024 * 1024;
    cout << "***** start create index *****" << endl;
    faiss::ascend::AscendIndexTS tsIndex;
    auto ret = tsIndex.InitWithExtraVal(deviceId, DIM, DEFAULT_TOKEN, resources,
        faiss::ascend::AlgorithmType::FLAT_HAMMING);
    ASSERT_EQ(ret, 0);
    cout << "***** end create index *****" << endl;

    std::vector<uint8_t> features(ntotal * DIM / 8);
    printf("[add -----------]\n");
    ascend::FeatureGenerator(features);
    std::vector<faiss::ascend::FeatureAttr> attrs(ntotal);
    std::vector<faiss::ascend::ExtraValAttr> valAttrs(ntotal);
    ascend::FeatureAttrGenerator(attrs);
    ascend::ExtraValAttrGenerator(valAttrs);
    auto res = tsIndex.SetSaveHostMemory();
    EXPECT_EQ(res, 0);
    for (size_t i = 0; i < addNum; i++) {
        std::vector<int64_t> labels;
        for (int64_t j = 0; j < ntotal; ++j) {
            labels.emplace_back(j + i * ntotal);
        }
        tsIndex.AddWithExtraVal(ntotal, features.data(), attrs.data(), labels.data(), valAttrs.data());
    }

    std::vector<int64_t> labels;
    for (int64_t i = ntotal * addNum - 1; i >= 0; i -= 262144) {
        labels.emplace_back(i);
    }
    std::vector<faiss::ascend::ExtraValAttr> attrsOut(labels.size());

    ret = tsIndex.GetExtraValAttrByLabel(labels.size(), labels.data(), attrsOut.data());

    ASSERT_EQ(ret, 0);
    for (size_t ilb = 0; ilb < labels.size(); ilb++) {
        auto lb = labels[ilb];
        int64_t lb100 = lb % ntotal;
        ASSERT_TRUE(IsExtraAttrSame(valAttrs[lb100], attrsOut[ilb]));
    }
}

TEST(TestAscendIndexUTTS, GetBaseByRange)
{
    size_t ntotal = 100;

    using namespace std;
    cout << "***** start Init *****" << endl;
    faiss::ascend::AscendIndexTS *tsIndex0 = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex0->Init(0, DIM, DEFAULT_TOKEN, faiss::ascend::AlgorithmType::FLAT_HAMMING);
    EXPECT_EQ(ret, 0);
    cout << "***** start Init  1*****" << endl;
    faiss::ascend::AscendIndexTS *tsIndex1 = new faiss::ascend::AscendIndexTS();
    ret = tsIndex1->Init(0, DIM, DEFAULT_TOKEN, faiss::ascend::AlgorithmType::FLAT_HAMMING);
    EXPECT_EQ(ret, 0);
    cout << "***** end Init *****" << endl;

    cout << "***** start add *****" << endl;
    std::vector<uint8_t> features(ntotal * DIM / 8);
    ascend::FeatureGenerator(features);
    auto res = tsIndex0->SetSaveHostMemory();
    EXPECT_EQ(res, 0);
    res = tsIndex1->SetSaveHostMemory();
    EXPECT_EQ(res, 0);

    std::vector<int64_t> labels;
    for (int64_t j = 0; j < ntotal; ++j) {
        labels.emplace_back(j);
    }
    std::vector<faiss::ascend::FeatureAttr> attrs(ntotal);
    ascend::FeatureAttrGenerator(attrs);

    res = tsIndex0->AddFeature(ntotal, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(res, 0);
    
    int64_t validNum = 0;
    tsIndex0->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal);

    int needAddNum = validNum / 2;
    std::vector<uint8_t> retBase(needAddNum * DIM / 8);
    std::vector<int64_t> retLabels(needAddNum);
    std::vector<faiss::ascend::FeatureAttr> retAttrs(needAddNum);
    ret = tsIndex0->GetBaseByRange(needAddNum, needAddNum, retLabels.data(), retBase.data(), retAttrs.data());
    EXPECT_EQ(ret, 0);

    ret = tsIndex1->AddFeature(needAddNum, retBase.data(), retAttrs.data(), retLabels.data());
    EXPECT_EQ(ret, 0);

    ret = tsIndex0->DeleteFeatureByLabel(needAddNum, retLabels.data());
    EXPECT_EQ(ret, 0);

#pragma omp parallel for if (ntotal > 10)
    for (int i = 0; i < needAddNum * DIM / 8; i++) {
        EXPECT_EQ(features[i + needAddNum * DIM / 8], retBase[i]);
        if (features[i + needAddNum * DIM / 8]!= retBase[i]) {
            printf("%u--%u \n", features[i + needAddNum * DIM/ 8], retBase[i]);
        }
    }
    delete tsIndex0;
    delete tsIndex1;
}


INSTANTIATE_TEST_CASE_P(BinaryFlatCheckGroup, TestTSBinaryFlatInitUT, ::testing::ValuesIn(INITITEMS));
INSTANTIATE_TEST_CASE_P(BinaryFlatCheckGroup, TestTSBinaryFlatGetUT, ::testing::ValuesIn(GETITEMS));
INSTANTIATE_TEST_CASE_P(BinaryFlatCheckGroup, TestTSBinaryFlatAddUT, ::testing::ValuesIn(ADDITEMS));
}