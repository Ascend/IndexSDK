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


#include <cmath>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <numeric>
#include <sys/time.h>
#include "Common.h"
#include "AscendIndexTS.h"
#include "TSBase.h"
#include "ErrorCode.h"

using namespace std;

namespace ascend {
constexpr int DIM = 256;
constexpr int TOKENNUM = 2500;
constexpr int K = 1;
constexpr size_t BASE_SIZE = 8192;
const std::vector<int> DEVICES = {0};

TEST(TestAscendIndexUTTS, Init)
{
    shared_ptr<faiss::ascend::AscendIndexTS> tsIndex = make_shared<faiss::ascend::AscendIndexTS>();
    auto res = tsIndex->Init(0, 64, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_HAMMING);
    EXPECT_EQ(res, ::ascend::APP_ERR_INNER_ERROR);
    res = tsIndex->Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_HAMMING);
    EXPECT_EQ(res, 0);
    tsIndex = make_shared<faiss::ascend::AscendIndexTS>();
    res = tsIndex->Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(res, 0);
    res = tsIndex->Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(res, ::ascend::APP_ERR_ILLEGAL_OPERATION);
    tsIndex = make_shared<faiss::ascend::AscendIndexTS>();
    res = tsIndex->Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_IP_FP16);
    EXPECT_EQ(res, 0);
    tsIndex = make_shared<faiss::ascend::AscendIndexTS>();
    res = tsIndex->Init(0, DIM, TOKENNUM, static_cast<faiss::ascend::AlgorithmType>(5));
    EXPECT_EQ(res, ::ascend::APP_ERR_NOT_IMPLEMENT);
    tsIndex = make_shared<faiss::ascend::AscendIndexTS>();
    res = tsIndex->Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_L2_INT8);
    EXPECT_EQ(res, 0);
    res = tsIndex->Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_L2_INT8);
    EXPECT_EQ(res, ::ascend::APP_ERR_ILLEGAL_OPERATION);
}

TEST(TestAscendIndexUTTS, Add)
{
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto res = tsIndex->Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_HAMMING);
    EXPECT_EQ(res, 0);
    std::vector<uint8_t> features(BASE_SIZE * DIM / BITELN);
    FeatureGenerator(features);
    std::vector<int64_t> labels;
    for (size_t i = 0; i < BASE_SIZE; ++i) {
        labels.push_back(i);
    }
    std::vector<faiss::ascend::FeatureAttr> attrs(BASE_SIZE);
    FeatureAttrGenerator(attrs);
    res = tsIndex->AddFeature(BASE_SIZE, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(res, 0);

    std::vector<uint8_t> getBase(BASE_SIZE * DIM);
    res = tsIndex->GetFeatureByLabel(BASE_SIZE, labels.data(), getBase.data());
    EXPECT_EQ(res, 0);

    int64_t validNum = 0;
    tsIndex->GetFeatureNum(&validNum);
    EXPECT_EQ(static_cast<size_t>(validNum), BASE_SIZE);
    // label not uinque
    res = tsIndex->AddFeature(BASE_SIZE, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(res, ::ascend::APP_ERR_INVALID_PARAM);
    delete tsIndex;
}

TEST(TestAscendIndexUTTS, AddCos)
{
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto res = tsIndex->Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(res, 0);
    std::vector<int8_t> features(BASE_SIZE * DIM);
    FeatureGenerator(features);
    std::vector<int64_t> labels;
    for (size_t i = 0; i < BASE_SIZE; ++i) {
        labels.push_back(i);
    }
    std::vector<faiss::ascend::FeatureAttr> attrs(BASE_SIZE);
    FeatureAttrGenerator(attrs);
    res = tsIndex->AddFeature(BASE_SIZE, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(res, 0);

    std::vector<int8_t> getBase(BASE_SIZE * DIM);
    res = tsIndex->GetFeatureByLabel(BASE_SIZE, labels.data(), getBase.data());
    EXPECT_EQ(res, 0);

    int64_t validNum = 0;
    tsIndex->GetFeatureNum(&validNum);
    EXPECT_EQ(static_cast<size_t>(validNum), BASE_SIZE);

    // label not uinque
    res = tsIndex->AddFeature(BASE_SIZE, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(res, ::ascend::APP_ERR_INVALID_PARAM);
    delete tsIndex;
}

TEST(TestAscendIndexUTTS, AddL2)
{
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto res = tsIndex->Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_L2_INT8);
    EXPECT_EQ(res, 0);
    std::vector<int8_t> features(BASE_SIZE * DIM);
    FeatureGenerator(features);
    std::vector<int64_t> labels;
    for (size_t i = 0; i < BASE_SIZE; ++i) {
        labels.push_back(i);
    }
    std::vector<faiss::ascend::FeatureAttr> attrs(BASE_SIZE);
    FeatureAttrGenerator(attrs);
    res = tsIndex->AddFeature(BASE_SIZE, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(res, 0);

    std::vector<int8_t> getBase(BASE_SIZE * DIM);
    res = tsIndex->GetFeatureByLabel(BASE_SIZE, labels.data(), getBase.data());
    EXPECT_EQ(res, 0);

    int64_t validNum = 0;
    tsIndex->GetFeatureNum(&validNum);
    EXPECT_EQ(static_cast<size_t>(validNum), BASE_SIZE);

    // label not uinque
    res = tsIndex->AddFeature(BASE_SIZE, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(res, ::ascend::APP_ERR_INVALID_PARAM);
    delete tsIndex;
}

TEST(TestAscendIndexUTTS, AddFlatIP)
{
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto res = tsIndex->Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_IP_FP16);
    EXPECT_EQ(res, 0);
    std::vector<float> features(BASE_SIZE * DIM);
    printf("[---add -----------]\n");
    FeatureGenerator(features);
    std::vector<int64_t> labels;
    for (size_t i = 0; i < BASE_SIZE; ++i) {
        labels.push_back(i);
    }
    std::vector<faiss::ascend::FeatureAttr> attrs(BASE_SIZE);
    FeatureAttrGenerator(attrs);
    res = tsIndex->AddFeature(BASE_SIZE, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(res, 0);

    std::vector<float> getBase(BASE_SIZE * DIM);
    res = tsIndex->GetFeatureByLabel(BASE_SIZE, labels.data(), getBase.data());
    EXPECT_EQ(res, 0);

    int64_t validNum = 0;
    tsIndex->GetFeatureNum(&validNum);
    EXPECT_EQ(static_cast<size_t>(validNum), BASE_SIZE);

    // label not uinque
    res = tsIndex->AddFeature(BASE_SIZE, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(res, ::ascend::APP_ERR_INVALID_PARAM);
    delete tsIndex;
}

TEST(TestAscendIndexUTTS, Search)
{
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto res = tsIndex->Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_HAMMING);
    EXPECT_EQ(res, 0);
    std::vector<uint8_t> features(BASE_SIZE * DIM / BITELN);
    FeatureGenerator(features);
    std::vector<int64_t> labels;
    for (size_t i = 0; i < BASE_SIZE; ++i) {
        labels.push_back(i);
    }
    std::vector<faiss::ascend::FeatureAttr> attrs(BASE_SIZE);
    FeatureAttrGenerator(attrs);
    res = tsIndex->AddFeature(BASE_SIZE, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(res, 0);

    int queryNum = 4;
    std::vector<float> distances(queryNum * K, -1);
    std::vector<int64_t> labelRes(queryNum * K, 10);
    std::vector<uint32_t> validnum(queryNum, 0);
    uint32_t size = queryNum * DIM / BITELN;
    std::vector<uint8_t> querys(size);
    querys.assign(features.begin(), features.begin() + size);

    uint32_t setlen = static_cast<uint32_t>(((TOKENNUM + 7) / BITELN));
    std::vector<uint8_t> bitSet(setlen, 0);
    bitSet[0] = 0x1 << 0 | 0x1 << 1 | 0x1 << 2 | 0x1 << 3;
    faiss::ascend::AttrFilter filter {};
    filter.timesStart = 0;
    filter.timesEnd = 3;
    filter.tokenBitSet = bitSet.data();
    filter.tokenBitSetLen = setlen;

    std::vector<faiss::ascend::AttrFilter> queryFilters(queryNum, filter);

    res = tsIndex->Search(queryNum, querys.data(), queryFilters.data(), false, K, labelRes.data(), distances.data(),
                          validnum.data());
    EXPECT_EQ(res, 0);
    res = tsIndex->Search(queryNum, querys.data(), queryFilters.data(), true, K, labelRes.data(), distances.data(),
                          validnum.data());
    EXPECT_EQ(res, 0);

    std::vector<uint32_t> delToken {0, 1};
    res = tsIndex->DeleteFeatureByToken(2, delToken.data());
    EXPECT_EQ(res, 0);
    int delCount = 100;
    std::vector<int64_t> delLabel(delCount);
    delLabel.assign(labels.begin(), labels.begin() + delCount);
    res = tsIndex->DeleteFeatureByLabel(delCount, delLabel.data());
    EXPECT_EQ(res, 0);
    delete tsIndex;
}

TEST(TestAscendIndexUTTS, SearchCos)
{
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto res = tsIndex->Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(res, 0);
    std::vector<int8_t> features(BASE_SIZE * DIM);
    FeatureGenerator(features);
    std::vector<int64_t> labels;
    for (size_t i = 0; i < BASE_SIZE; ++i) {
        labels.push_back(i);
    }
    std::vector<faiss::ascend::FeatureAttr> attrs(BASE_SIZE);
    FeatureAttrGenerator(attrs);
    res = tsIndex->AddFeature(BASE_SIZE, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(res, 0);

    int queryNum = 4;
    std::vector<float> distances(queryNum * K, -1);
    std::vector<int64_t> labelRes(queryNum * K, 10);
    std::vector<uint32_t> validnum(queryNum, 0);
    uint32_t size = queryNum * DIM;
    std::vector<int8_t> querys(size);
    querys.assign(features.begin(), features.begin() + size);

    uint32_t setlen = static_cast<uint32_t>(((TOKENNUM + 7) / BITELN));
    std::vector<uint8_t> bitSet(setlen, 0);
    bitSet[0] = 0x1 << 0 | 0x1 << 1 | 0x1 << 2 | 0x1 << 3;
    faiss::ascend::AttrFilter filter {};
    filter.timesStart = 0;
    filter.timesEnd = 3;
    filter.tokenBitSet = bitSet.data();
    filter.tokenBitSetLen = setlen;

    std::vector<faiss::ascend::AttrFilter> queryFilters(queryNum, filter);

    res = tsIndex->Search(queryNum, querys.data(), queryFilters.data(), false, K, labelRes.data(), distances.data(),
                          validnum.data());
    EXPECT_EQ(res, 0);
    res = tsIndex->Search(queryNum, querys.data(), queryFilters.data(), true, K, labelRes.data(), distances.data(),
                          validnum.data());
    EXPECT_EQ(res, 0);

    std::vector<uint32_t> delToken {0, 1};
    res = tsIndex->DeleteFeatureByToken(2, delToken.data());
    EXPECT_EQ(res, 0);
    int delCount = 100;
    std::vector<int64_t> delLabel(delCount);
    delLabel.assign(labels.begin(), labels.begin() + delCount);
    res = tsIndex->DeleteFeatureByLabel(delCount, delLabel.data());
    EXPECT_EQ(res, 0);
    delete tsIndex;
}

TEST(TestAscendIndexUTTS, SearchL2)
{
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto res = tsIndex->Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_L2_INT8);
    EXPECT_EQ(res, 0);
    std::vector<int8_t> features(BASE_SIZE * DIM);
    FeatureGenerator(features);
    std::vector<int64_t> labels;
    for (size_t i = 0; i < BASE_SIZE; ++i) {
        labels.push_back(i);
    }
    std::vector<faiss::ascend::FeatureAttr> attrs(BASE_SIZE);
    FeatureAttrGenerator(attrs);
    res = tsIndex->AddFeature(BASE_SIZE, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(res, 0);

    int queryNum = 4;
    std::vector<float> distances(queryNum * K, -1);
    std::vector<int64_t> labelRes(queryNum * K, 10);
    std::vector<uint32_t> validnum(queryNum, 0);
    uint32_t size = queryNum * DIM;
    std::vector<int8_t> querys(size);
    querys.assign(features.begin(), features.begin() + size);

    uint32_t setlen = static_cast<uint32_t>(((TOKENNUM + 7) / BITELN));
    std::vector<uint8_t> bitSet(setlen, 0);
    bitSet[0] = 0x1 << 0 | 0x1 << 1 | 0x1 << 2 | 0x1 << 3;
    faiss::ascend::AttrFilter filter {};
    filter.timesStart = 0;
    filter.timesEnd = 3;
    filter.tokenBitSet = bitSet.data();
    filter.tokenBitSetLen = setlen;

    std::vector<faiss::ascend::AttrFilter> queryFilters(queryNum, filter);

    res = tsIndex->Search(queryNum, querys.data(), queryFilters.data(), false, K, labelRes.data(), distances.data(),
                          validnum.data());
    EXPECT_EQ(res, 0);
    res = tsIndex->Search(queryNum, querys.data(), queryFilters.data(), true, K, labelRes.data(), distances.data(),
                          validnum.data());
    EXPECT_EQ(res, 0);

    std::vector<uint32_t> delToken {0, 1};
    res = tsIndex->DeleteFeatureByToken(2, delToken.data());
    EXPECT_EQ(res, 0);
    int delCount = 100;
    std::vector<int64_t> delLabel(delCount);
    delLabel.assign(labels.begin(), labels.begin() + delCount);
    res = tsIndex->DeleteFeatureByLabel(delCount, delLabel.data());
    EXPECT_EQ(res, 0);
    delete tsIndex;
}

TEST(TestAscendIndexUTTS, SearchFP16)
{
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto res = tsIndex->Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_IP_FP16);
    EXPECT_EQ(res, 0);
    std::vector<float> features(BASE_SIZE * DIM);
    FeatureGenerator(features);
    std::vector<int64_t> labels;
    for (size_t i = 0; i < BASE_SIZE; ++i) {
        labels.push_back(i);
    }
    std::vector<faiss::ascend::FeatureAttr> attrs(BASE_SIZE);
    FeatureAttrGenerator(attrs);
    res = tsIndex->AddFeature(BASE_SIZE, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(res, 0);

    int queryNum = 4;
    std::vector<float> distances(queryNum * K, -1);
    std::vector<int64_t> labelRes(queryNum * K, 10);
    std::vector<uint32_t> validnum(queryNum, 0);
    uint32_t size = queryNum * DIM;
    std::vector<float> querys(size);
    querys.assign(features.begin(), features.begin() + size);

    uint32_t setlen = static_cast<uint32_t>(((TOKENNUM + 7) / BITELN));
    std::vector<uint8_t> bitSet(setlen, 0);
    bitSet[0] = 0x1 << 0 | 0x1 << 1 | 0x1 << 2 | 0x1 << 3;
    faiss::ascend::AttrFilter filter {};
    filter.timesStart = 0;
    filter.timesEnd = 3;
    filter.tokenBitSet = bitSet.data();
    filter.tokenBitSetLen = setlen;

    std::vector<faiss::ascend::AttrFilter> queryFilters(queryNum, filter);

    res = tsIndex->Search(queryNum, querys.data(), queryFilters.data(), false, K, labelRes.data(), distances.data(),
                          validnum.data());
    EXPECT_EQ(res, 0);
    res = tsIndex->Search(queryNum, querys.data(), queryFilters.data(), true, K, labelRes.data(), distances.data(),
                          validnum.data());
    EXPECT_EQ(res, 0);

    std::vector<uint32_t> delToken {0, 1};
    res = tsIndex->DeleteFeatureByToken(2, delToken.data());
    EXPECT_EQ(res, 0);
    int delCount = 100;
    std::vector<int64_t> delLabel(delCount);
    delLabel.assign(labels.begin(), labels.begin() + delCount);
    res = tsIndex->DeleteFeatureByLabel(delCount, delLabel.data());
    EXPECT_EQ(res, 0);
    delete tsIndex;
}

TEST(TestAscendIndexUTTS, SearchWithExtraMask)
{
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto res = tsIndex->Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_HAMMING);
    EXPECT_EQ(res, 0);
    std::vector<uint8_t> features(BASE_SIZE * DIM / BITELN);
    FeatureGenerator(features);
    std::vector<int64_t> labels;
    for (size_t i = 0; i < BASE_SIZE; ++i) {
        labels.push_back(i);
    }
    std::vector<faiss::ascend::FeatureAttr> attrs(BASE_SIZE);
    FeatureAttrGenerator(attrs);
    res = tsIndex->AddFeature(BASE_SIZE, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(res, 0);

    int queryNum = 4;
    std::vector<float> distances(queryNum * K, -1);
    std::vector<int64_t> labelRes(queryNum * K, 10);
    std::vector<uint32_t> validnum(queryNum, 0);
    uint32_t size = queryNum * DIM / BITELN;
    std::vector<uint8_t> querys(size);
    querys.assign(features.begin(), features.begin() + size);

    uint32_t setlen = static_cast<uint32_t>(((TOKENNUM + 7) / BITELN));
    std::vector<uint8_t> bitSet(setlen, 0);
    bitSet[0] = 0x1 << 0 | 0x1 << 1 | 0x1 << 2 | 0x1 << 3;
    faiss::ascend::AttrFilter filter {};
    filter.timesStart = 0;
    filter.timesEnd = 3;
    filter.tokenBitSet = bitSet.data();
    filter.tokenBitSetLen = setlen;

    std::vector<faiss::ascend::AttrFilter> queryFilters(queryNum, filter);

    int extraMaskLen = 12;
    std::vector<uint8_t> extraMask(queryNum * extraMaskLen, 0);
    int ind = 1;
    for (int i = 0; i < queryNum; i++) {
        for (int j = 0; j < extraMaskLen; j++) {
            extraMask[i * extraMaskLen + j] = 0x1 << (ind % 8);
            ind++;
        }
    }
    res = tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), false, K, extraMask.data(),
                                             extraMaskLen, false, labelRes.data(), distances.data(), validnum.data());
    EXPECT_EQ(res, 0);
    res = tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), true, K, extraMask.data(),
                                             extraMaskLen, false, labelRes.data(), distances.data(), validnum.data());
    EXPECT_EQ(res, 0);
    delete tsIndex;
}

TEST(TestAscendIndexUTTS, SearchWithExtraMaskCos)
{
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto res = tsIndex->Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(res, 0);
    std::vector<int8_t> features(BASE_SIZE * DIM);
    FeatureGenerator(features);
    std::vector<int64_t> labels;
    for (size_t i = 0; i < BASE_SIZE; ++i) {
        labels.push_back(i);
    }
    std::vector<faiss::ascend::FeatureAttr> attrs(BASE_SIZE);
    FeatureAttrGenerator(attrs);
    res = tsIndex->AddFeature(BASE_SIZE, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(res, 0);

    int queryNum = 4;
    std::vector<float> distances(queryNum * K, -1);
    std::vector<int64_t> labelRes(queryNum * K, 10);
    std::vector<uint32_t> validnum(queryNum, 0);
    uint32_t size = queryNum * DIM;
    std::vector<int8_t> querys(size);
    querys.assign(features.begin(), features.begin() + size);

    uint32_t setlen = static_cast<uint32_t>(((TOKENNUM + 7) / BITELN));
    std::vector<uint8_t> bitSet(setlen, 0);
    bitSet[0] = 0x1 << 0 | 0x1 << 1 | 0x1 << 2 | 0x1 << 3;
    faiss::ascend::AttrFilter filter {};
    filter.timesStart = 0;
    filter.timesEnd = 3;
    filter.tokenBitSet = bitSet.data();
    filter.tokenBitSetLen = setlen;

    std::vector<faiss::ascend::AttrFilter> queryFilters(queryNum, filter);

    int extraMaskLen = 12;
    std::vector<uint8_t> extraMask(queryNum * extraMaskLen, 0);
    int ind = 1;
    for (int i = 0; i < queryNum; i++) {
        for (int j = 0; j < extraMaskLen; j++) {
            extraMask[i * extraMaskLen + j] = 0x1 << (ind % 8);
            ind++;
        }
    }
    res = tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), false, K, extraMask.data(),
                                             extraMaskLen, false, labelRes.data(), distances.data(), validnum.data());
    EXPECT_EQ(res, 0);
    res = tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), true, K, extraMask.data(),
                                             extraMaskLen, false, labelRes.data(), distances.data(), validnum.data());
    EXPECT_EQ(res, 0);

    delete tsIndex;
}

TEST(TestAscendIndexUTTS, SearchWithExtraMaskL2)
{
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto res = tsIndex->Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_L2_INT8);
    EXPECT_EQ(res, 0);
    std::vector<int8_t> features(BASE_SIZE * DIM);
    FeatureGenerator(features);
    std::vector<int64_t> labels;
    for (size_t i = 0; i < BASE_SIZE; ++i) {
        labels.push_back(i);
    }
    std::vector<faiss::ascend::FeatureAttr> attrs(BASE_SIZE);
    FeatureAttrGenerator(attrs);
    res = tsIndex->AddFeature(BASE_SIZE, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(res, 0);

    int queryNum = 4;
    std::vector<float> distances(queryNum * K, -1);
    std::vector<int64_t> labelRes(queryNum * K, 10);
    std::vector<uint32_t> validnum(queryNum, 0);
    uint32_t size = queryNum * DIM;
    std::vector<int8_t> querys(size);
    querys.assign(features.begin(), features.begin() + size);

    uint32_t setlen = static_cast<uint32_t>(((TOKENNUM + 7) / BITELN));
    std::vector<uint8_t> bitSet(setlen, 0);
    bitSet[0] = 0x1 << 0 | 0x1 << 1 | 0x1 << 2 | 0x1 << 3;
    faiss::ascend::AttrFilter filter {};
    filter.timesStart = 0;
    filter.timesEnd = 3;
    filter.tokenBitSet = bitSet.data();
    filter.tokenBitSetLen = setlen;

    std::vector<faiss::ascend::AttrFilter> queryFilters(queryNum, filter);

    int extraMaskLen = 12;
    std::vector<uint8_t> extraMask(queryNum * extraMaskLen, 0);
    int ind = 1;
    for (int i = 0; i < queryNum; i++) {
        for (int j = 0; j < extraMaskLen; j++) {
            extraMask[i * extraMaskLen + j] = 0x1 << (ind % 8);
            ind++;
        }
    }
    res = tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), false, K, extraMask.data(),
                                             extraMaskLen, false, labelRes.data(), distances.data(), validnum.data());
    EXPECT_EQ(res, 0);
    res = tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), true, K, extraMask.data(),
                                             extraMaskLen, false, labelRes.data(), distances.data(), validnum.data());
    EXPECT_EQ(res, 0);
    delete tsIndex;
}

TEST(TestAscendIndexUTTS, SearchWithExtraMaskFP16)
{
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto res = tsIndex->Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_IP_FP16);
    EXPECT_EQ(res, 0);
    std::vector<float> features(BASE_SIZE * DIM);
    FeatureGenerator(features);
    std::vector<int64_t> labels;
    for (size_t i = 0; i < BASE_SIZE; ++i) {
        labels.push_back(i);
    }
    std::vector<faiss::ascend::FeatureAttr> attrs(BASE_SIZE);
    FeatureAttrGenerator(attrs);
    res = tsIndex->AddFeature(BASE_SIZE, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(res, 0);

    int queryNum = 4;
    std::vector<float> distances(queryNum * K, -1);
    std::vector<int64_t> labelRes(queryNum * K, 10);
    std::vector<uint32_t> validnum(queryNum, 0);
    uint32_t size = queryNum * DIM;
    std::vector<float> querys(size);
    querys.assign(features.begin(), features.begin() + size);

    uint32_t setlen = static_cast<uint32_t>(((TOKENNUM + 7) / BITELN));
    std::vector<uint8_t> bitSet(setlen, 0);
    bitSet[0] = 0x1 << 0 | 0x1 << 1 | 0x1 << 2 | 0x1 << 3;
    faiss::ascend::AttrFilter filter {};
    filter.timesStart = 0;
    filter.timesEnd = 3;
    filter.tokenBitSet = bitSet.data();
    filter.tokenBitSetLen = setlen;

    std::vector<faiss::ascend::AttrFilter> queryFilters(queryNum, filter);

    int extraMaskLen = 12;
    std::vector<uint8_t> extraMask(queryNum * extraMaskLen, 0);
    int ind = 1;
    for (int i = 0; i < queryNum; i++) {
        for (int j = 0; j < extraMaskLen; j++) {
            extraMask[i * extraMaskLen + j] = 0x1 << (ind % 8);
            ind++;
        }
    }
    res = tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), false, K, extraMask.data(),
                                             extraMaskLen, false, labelRes.data(), distances.data(), validnum.data());
    EXPECT_EQ(res, 0);
    res = tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), true, K, extraMask.data(),
                                             extraMaskLen, false, labelRes.data(), distances.data(), validnum.data());
    EXPECT_EQ(res, 0);
    delete tsIndex;
}

TEST(TestAscendIndexUTTS, ExtraValSearch)
{
    uint32_t ntotal = 10000;
    uint32_t addNum = 1;
    uint32_t deviceId = 0;
    uint32_t dim = 256;
    uint32_t tokenNum = 300000;
    uint32_t queryNum = 1;
    uint32_t k = 10;
    uint64_t resources = 1024 * 1024 * 1024;
    faiss::ascend::AscendIndexTS tsIndex;
    auto ret = tsIndex.InitWithExtraVal(deviceId, dim, tokenNum, resources, faiss::ascend::AlgorithmType::FLAT_HAMMING);
    EXPECT_EQ(ret, 0);

    std::vector<uint8_t> features(ntotal * dim / 8);
    FeatureGenerator(features);
    std::vector<faiss::ascend::FeatureAttr> attrs(ntotal);
    std::vector<faiss::ascend::ExtraValAttr> valAttrs(ntotal);
    for (size_t i = 0; i < addNum; i++) {
        std::vector<int64_t> labels;

        for (int64_t j = 0; j < ntotal; ++j) {
            labels.emplace_back(j + i * ntotal);
        }
        FeatureAttrGenerator(attrs);
        ExtraValAttrGenerator(valAttrs);
        ret = tsIndex.AddWithExtraVal(ntotal, features.data(), attrs.data(), labels.data(), valAttrs.data());
        EXPECT_EQ(ret, 0);
    }

    std::vector<float> distances(queryNum * k, -1);
    std::vector<int64_t> labelRes(queryNum * k, 10);
    std::vector<uint32_t> validnum(queryNum, 0);
    uint32_t size = queryNum * dim / 8;
    std::vector<uint8_t> querys(size);
    querys.assign(features.begin(), features.begin() + size);

    uint32_t setlen = static_cast<uint32_t>(((tokenNum + 7) / 8));
    std::vector<uint8_t> bitSet(setlen, 255);
    faiss::ascend::AttrFilter filter {};
    filter.timesStart = 0;
    filter.timesEnd = 3;
    filter.tokenBitSet = bitSet.data();
    filter.tokenBitSetLen = setlen;

    faiss::ascend::ExtraValFilter valFilter {};
    valFilter.filterVal = 10;
    valFilter.matchVal = 1;
    
    std::vector<faiss::ascend::AttrFilter> queryFilters(queryNum, filter);
    std::vector< faiss::ascend::ExtraValFilter> queryValFilters(queryNum, valFilter);
    ret = tsIndex.SearchWithExtraVal(queryNum, querys.data(), queryFilters.data(), false, k, labelRes.data(),
        distances.data(), validnum.data(), queryValFilters.data());
    EXPECT_EQ(ret, 0);
}

TEST(TestAscendIndexUTTS, MixAdd)
{
    int ntotal = 100;
    int deviceId = 0;
    int dim = 512;
    int tokenNum = 2500;
    faiss::ascend::AscendIndexTS tsIndex;
    faiss::ascend::AscendIndexTS tsExtraValIndex;
    faiss::ascend::AscendIndexTS tsTwoIndex;
    faiss::ascend::AscendIndexTS tsTwoExtraValIndex;
    auto ret = tsIndex.Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_HAMMING);
    EXPECT_EQ(ret, 0);

    ret = tsExtraValIndex.Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_HAMMING);
    EXPECT_EQ(ret, 0);

    ret = tsTwoIndex.Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_HAMMING);
    EXPECT_EQ(ret, 0);

    ret = tsTwoExtraValIndex.Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_HAMMING);
    EXPECT_EQ(ret, 0);

    std::vector<uint8_t> features(ntotal * dim / 8);
    FeatureGenerator(features);

    std::vector<faiss::ascend::FeatureAttr> attrs(ntotal);
    std::vector<faiss::ascend::ExtraValAttr> valAttrs(ntotal);
    FeatureAttrGenerator(attrs);
    ExtraValAttrGenerator(valAttrs);
    std::vector<int64_t> labels(ntotal * 2);
    for (int64_t j = 0; j < ntotal * 2; ++j) {
        labels[j] = j;
    }

    std::vector<faiss::ascend::FeatureAttr> attr(ntotal);
    FeatureAttrGenerator(attr);
    // tsIndex对应第一次为AddFeature
    ret = tsIndex.AddFeature(ntotal, features.data(), attr.data(), labels.data() + ntotal);
    EXPECT_EQ(ret, 0);
    
    ret = tsIndex.AddWithExtraVal(ntotal, features.data(), attrs.data(), labels.data(), valAttrs.data());
    EXPECT_NE(ret, 0);

    // tsExtraValIndex对应第一次为AddWithExtraVal
    ret = tsExtraValIndex.AddWithExtraVal(ntotal, features.data(), attrs.data(), labels.data(), valAttrs.data());
    EXPECT_EQ(ret, 0);
    
    ret = tsExtraValIndex.AddFeature(ntotal, features.data(), attr.data(), labels.data() + ntotal);
    EXPECT_NE(ret, 0);

    // tsTwoIndex对应调用2次AddFeature
    ret = tsTwoIndex.AddFeature(ntotal, features.data(), attr.data(), labels.data() + ntotal);
    EXPECT_EQ(ret, 0);
    
    ret = tsTwoIndex.AddFeature(ntotal, features.data(), attr.data(), labels.data());
    EXPECT_EQ(ret, 0);
    
    // tsTwoExtraValIndex对应调用2次AddWithExtraVal
    ret = tsTwoExtraValIndex.AddWithExtraVal(ntotal, features.data(), attrs.data(),
        labels.data() + ntotal, valAttrs.data());
    EXPECT_EQ(ret, 0);

    ret = tsTwoExtraValIndex.AddWithExtraVal(ntotal, features.data(), attrs.data(), labels.data(), valAttrs.data());
    EXPECT_EQ(ret, 0);
}

TEST(TestAscendIndexUTTS, HMSetSaveHostMemory)
{
    faiss::ascend::AscendIndexTS tsIndex;
    auto res = tsIndex.Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_HAMMING);
    EXPECT_EQ(res, 0);
    std::vector<uint8_t> features(BASE_SIZE * DIM / BITELN);
    FeatureGenerator(features);
    std::vector<int64_t> labels;
    for (size_t i = 0; i < BASE_SIZE; ++i) {
        labels.push_back(i);
    }
    std::vector<faiss::ascend::FeatureAttr> attrs(BASE_SIZE);
    FeatureAttrGenerator(attrs);
    res = tsIndex.SetSaveHostMemory();
    EXPECT_EQ(res, 0);
    res = tsIndex.AddFeature(BASE_SIZE, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(res, 0);

    // SetSaveHostMemory在add之前设置
    res = tsIndex.SetSaveHostMemory();
    EXPECT_NE(res, 0);
    // SetSaveHostMemory后无法使用DeleteFeatureByToken
    std::vector<uint32_t> delToken {0, 1};
    res = tsIndex.DeleteFeatureByToken(2, delToken.data());
    EXPECT_NE(res, 0);
}

TEST(TestAscendIndexUTTS, Int8CosSetSaveHostMemory)
{
    faiss::ascend::AscendIndexTS tsIndex;
    auto res = tsIndex.Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(res, 0);
    std::vector<int8_t> features(BASE_SIZE * DIM);
    FeatureGenerator(features);
    std::vector<int64_t> labels;
    for (size_t i = 0; i < BASE_SIZE; ++i) {
        labels.push_back(i);
    }
    std::vector<faiss::ascend::FeatureAttr> attrs(BASE_SIZE);
    FeatureAttrGenerator(attrs);
    res = tsIndex.SetSaveHostMemory();
    EXPECT_EQ(res, 0);
    res = tsIndex.AddFeature(BASE_SIZE, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(res, 0);

    // SetSaveHostMemory在add之前设置
    res = tsIndex.SetSaveHostMemory();
    EXPECT_NE(res, 0);
    // SetSaveHostMemory后无法使用DeleteFeatureByToken
    std::vector<uint32_t> delToken {0, 1};
    res = tsIndex.DeleteFeatureByToken(2, delToken.data());
    EXPECT_NE(res, 0);
}

TEST(TestAscendIndexUTTS, GetFeatureAttrByLabel)
{
    std::vector<uint8_t> base(BASE_SIZE * DIM / 8);
    FeatureGenerator(base);
    std::vector<int64_t> labels(BASE_SIZE);
    std::iota(labels.begin(), labels.end(), 0);
    std::vector<faiss::ascend::FeatureAttr> attrs(BASE_SIZE);
    FeatureAttrGenerator(attrs);
    faiss::ascend::AscendIndexTS index;
    auto ret = index.Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_HAMMING);
    EXPECT_EQ(ret, 0);
    ret = index.AddFeature(BASE_SIZE, base.data(), attrs.data(), labels.data());
    EXPECT_EQ(ret, 0);
    int getAttrCnt = 10;
    std::vector<faiss::ascend::FeatureAttr> getAttr(getAttrCnt);
    printf("GetFeatureAttrByLabel  start\n");
    ret = index.GetFeatureAttrByLabel(getAttrCnt, labels.data(), getAttr.data());
    EXPECT_EQ(ret, 0);

    for (int i = 0; i < getAttr.size(); i++) {
        EXPECT_EQ(attrs[i].time, getAttr[i].time);
        EXPECT_EQ(attrs[i].tokenId, getAttr[i].tokenId);
    }
}

TEST(TestAscendIndexUTTS, GetFeatureByLabel)
{
    size_t baseSize = 100;
    std::vector<uint8_t> base(baseSize * DIM / 8);
    FeatureGenerator(base);
    std::vector<int64_t> label(baseSize);
    std::iota(label.begin(), label.end(), 0);
    std::vector<faiss::ascend::FeatureAttr> attrs(baseSize);
    FeatureAttrGenerator(attrs);
    faiss::ascend::AscendIndexTS index;
    auto ret = index.Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_HAMMING);
    EXPECT_EQ(ret, 0);
    auto res = index.SetSaveHostMemory();
    EXPECT_EQ(res, 0);
    ret = index.AddFeature(baseSize, base.data(), attrs.data(), label.data());
    EXPECT_EQ(ret, 0);
    std::vector<uint8_t> getBase(baseSize * DIM);

    ret = index.GetFeatureByLabel(baseSize, label.data(), getBase.data());
    EXPECT_EQ(ret, 0);

    std::vector<int64_t> labelOut(baseSize);
    std::iota(labelOut.begin(), labelOut.end(), baseSize);
    std::vector<uint8_t> getBaseOut(baseSize * DIM);
    // 无法获取不存在的label
    ret = index.GetFeatureByLabel(baseSize, labelOut.data(), getBaseOut.data());
    EXPECT_NE(ret, 0);
}

TEST(TestAscendIndexUTTS, GetBaseByRangeWithExtraVal)
{
    size_t ntotal = 100;
    uint32_t addNum = 1;

    using namespace std;
    cout << "***** start Init *****" << endl;
    faiss::ascend::AscendIndexTS *tsIndex0 = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex0->Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_HAMMING);
    EXPECT_EQ(ret, 0);
    cout << "***** end Init *****" << endl;

    cout << "***** start add *****" << endl;
    std::vector<uint8_t> features(ntotal * DIM / 8);
    
    FeatureGenerator(features);

    std::vector<int64_t> labels;
    for (int64_t j = 0; j < ntotal; ++j) {
        labels.emplace_back(j);
    }
    std::vector<faiss::ascend::FeatureAttr> attrs(ntotal);
    std::vector<faiss::ascend::ExtraValAttr> valAttrs(ntotal);
    FeatureAttrGenerator(attrs);
    ExtraValAttrGenerator(valAttrs);
    auto res = tsIndex0->SetSaveHostMemory();
    EXPECT_EQ(res, 0);
    res = tsIndex0->AddWithExtraVal(ntotal, features.data(), attrs.data(), labels.data(), valAttrs.data());
    EXPECT_EQ(res, 0);

    int64_t validNum = 0;
    tsIndex0->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal);

    int needAddNum = validNum / 2;
    std::vector<uint8_t> retBase(needAddNum * DIM / 8);
    std::vector<int64_t> retLabels(needAddNum);
    std::vector<faiss::ascend::FeatureAttr> retAttrs(needAddNum);
    std::vector<faiss::ascend::ExtraValAttr> retValAttrs(needAddNum);

    ret = tsIndex0->GetBaseByRangeWithExtraVal(needAddNum, needAddNum, retLabels.data(),
        retBase.data(), retAttrs.data(), retValAttrs.data());
    EXPECT_EQ(ret, 0);

#pragma omp parallel for if (ntotal > 100)
    for (int i = 0; i < needAddNum * DIM / 8; i++) {
        EXPECT_EQ(features[i + needAddNum * DIM / 8], retBase[i]);
    }

    for (int i = 0; i < needAddNum; i++) {
        EXPECT_EQ(retValAttrs[i].val, valAttrs[i + needAddNum].val);
    }
    delete tsIndex0;
}

bool IsExtraAttrSame(const faiss::ascend::ExtraValAttr &lAttr, const faiss::ascend::ExtraValAttr &rAttr)
{
    return (lAttr.val == rAttr.val);
}

TEST(TestAscendIndexUTTS, GetExtraValAttrByLabel)
{
    int64_t ntotal = 1000;
    uint32_t addNum = 1;

    uint64_t resources = 1024 * 1024 * 1024;
    cout << "***** start create index *****" << endl;
    faiss::ascend::AscendIndexTS tsIndex;
    auto ret = tsIndex.InitWithExtraVal(0, DIM, TOKENNUM, resources, faiss::ascend::AlgorithmType::FLAT_HAMMING);
    ASSERT_EQ(ret, 0);
    cout << "***** end create index *****" << endl;

    std::vector<uint8_t> features(ntotal * DIM / 8);
    printf("[add -----------]\n");
    FeatureGenerator(features);
    std::vector<faiss::ascend::FeatureAttr> attrs(ntotal);
    std::vector<faiss::ascend::ExtraValAttr> valAttrs(ntotal);
    FeatureAttrGenerator(attrs);
    ExtraValAttrGenerator(valAttrs);
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

void TestSearchWithExtraMaskInvalidParameter(int64_t queryNum, std::vector<float> &querys,
    std::vector<faiss::ascend::AttrFilter> &queryFilters, std::vector<uint8_t> &extraMask, int extraMaskLen,
    std::vector<uint16_t> &extraScore, std::vector<int64_t> &labelRes, std::vector<float> &distances,
    std::vector<uint32_t> &validnum, std::shared_ptr<faiss::ascend::AscendIndexTS> &tsIndex)
{
    auto res = tsIndex->SearchWithExtraMask(0, querys.data(), queryFilters.data(), false, K, extraMask.data(),
        extraMaskLen, false, extraScore.data(), labelRes.data(), distances.data(), validnum.data());
    EXPECT_EQ(res, ::ascend::APP_ERR_INVALID_PARAM);
    res = tsIndex->SearchWithExtraMask(10241, querys.data(), queryFilters.data(), false, K, extraMask.data(), // 10241
        extraMaskLen, false, extraScore.data(), labelRes.data(), distances.data(), validnum.data());
    EXPECT_EQ(res, ::ascend::APP_ERR_INVALID_PARAM);

    res = tsIndex->SearchWithExtraMask(queryNum, nullptr, queryFilters.data(), false, K, extraMask.data(),
        extraMaskLen, false, extraScore.data(), labelRes.data(), distances.data(), validnum.data());
    EXPECT_EQ(res, ::ascend::APP_ERR_INVALID_PARAM);

    res = tsIndex->SearchWithExtraMask(queryNum, querys.data(), nullptr, false, K, extraMask.data(),
        extraMaskLen, false, extraScore.data(), labelRes.data(), distances.data(), validnum.data());
    EXPECT_EQ(res, ::ascend::APP_ERR_INVALID_PARAM);

    res = tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), false, -1, extraMask.data(),
        extraMaskLen, false, extraScore.data(), labelRes.data(), distances.data(), validnum.data());
    EXPECT_EQ(res, ::ascend::APP_ERR_INVALID_PARAM);
    res = tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), false, 1e5+1, extraMask.data(),
        extraMaskLen, false, extraScore.data(), labelRes.data(), distances.data(), validnum.data());
    EXPECT_EQ(res, ::ascend::APP_ERR_INVALID_PARAM);

    res = tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), false, K, nullptr,
        extraMaskLen, false, extraScore.data(), labelRes.data(), distances.data(), validnum.data());
    EXPECT_EQ(res, ::ascend::APP_ERR_INVALID_PARAM);

    res = tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), false, K, extraMask.data(),
        2, false, extraScore.data(), labelRes.data(), distances.data(), validnum.data()); // 2 invalid
    EXPECT_EQ(res, ::ascend::APP_ERR_INVALID_PARAM);

    res = tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), false, K, extraMask.data(),
        extraMaskLen, false, nullptr, labelRes.data(), distances.data(), validnum.data());
    EXPECT_EQ(res, ::ascend::APP_ERR_INVALID_PARAM);

    res = tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), false, K, extraMask.data(),
        extraMaskLen, false, extraScore.data(), nullptr, distances.data(), validnum.data());
    EXPECT_EQ(res, ::ascend::APP_ERR_INVALID_PARAM);

    res = tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), false, K, extraMask.data(),
        extraMaskLen, false, extraScore.data(), labelRes.data(), nullptr, validnum.data());
    EXPECT_EQ(res, ::ascend::APP_ERR_INVALID_PARAM);
    res = tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), false, K, extraMask.data(),
        extraMaskLen, false, extraScore.data(), labelRes.data(), distances.data(), nullptr);
    EXPECT_EQ(res, ::ascend::APP_ERR_INVALID_PARAM);
}

void GenerateData(int64_t queryNum, int64_t baseSize, int64_t dim, int extraMaskLen,
                  std::vector<float> &features, std::vector<int64_t> &labels,
                  std::vector<faiss::ascend::FeatureAttr> &attrs,
                  std::vector<float> &querys, std::vector<uint8_t> &bitSet,
                  std::vector<faiss::ascend::AttrFilter> &queryFilters,
                  std::vector<uint8_t> &extraMask)
{
    features.resize(baseSize * dim);
    FeatureGenerator(features);

    labels.resize(baseSize);
    std::iota(labels.begin(), labels.end(), 0);

    attrs.resize(baseSize);
    FeatureAttrGenerator(attrs);

    uint32_t size = queryNum * dim;
    querys.assign(features.begin(), features.begin() + size);

    uint32_t setlen = static_cast<uint32_t>(((TOKENNUM + 8 - 1) / BITELN)); // 按照8对齐
    bitSet.resize(setlen, 0);
    bitSet[0] = ((0x1 << 0) | (0x1 << 1) | (0x1 << 2) | (0x1 << 3));
    faiss::ascend::AttrFilter filter {};
    filter.timesStart = 0;
    filter.timesEnd = 3;
    filter.tokenBitSet = bitSet.data();
    filter.tokenBitSetLen = setlen;

    queryFilters.resize(queryNum, filter);
    extraMask.resize(queryNum * extraMaskLen, 0);
    int ind = 1;
    for (int64_t i = 0; i < queryNum; i++) {
        for (int j = 0; j < extraMaskLen; j++) {
            extraMask[i * extraMaskLen + j] = 0x1 << (ind % 8);
            ind++;
        }
    }
}

TEST(TestAscendIndexUTTS, SearchWithExtraMaskAndExtraScore)
{
    int64_t baseSize = 1;
    int64_t queryNum = baseSize;
    int extraMaskLen = 1;
    std::vector<float> features;
    std::vector<int64_t> labels;
    std::vector<faiss::ascend::FeatureAttr> attrs;
    std::vector<float> querys;
    std::vector<uint8_t> bitSet;
    std::vector<faiss::ascend::AttrFilter> queryFilters;
    std::vector<uint8_t> extraMask;
    GenerateData(queryNum, baseSize, DIM, extraMaskLen, features, labels,
                 attrs, querys, bitSet, queryFilters, extraMask);

    std::shared_ptr<faiss::ascend::AscendIndexTS> tsIndex = std::make_shared<faiss::ascend::AscendIndexTS>();
    auto res = tsIndex->Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_IP_FP16);
    EXPECT_EQ(res, 0);
    res = tsIndex->AddFeature(baseSize, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(res, 0);

    std::vector<float> distances(queryNum * K, -1);
    std::vector<int64_t> labelRes(queryNum * K, 10);
    std::vector<uint32_t> validnum(queryNum, 0);
    std::vector<uint16_t> extraScore(queryNum * ((baseSize + 16 - 1) / 16 * 16)); // 16对齐

    TestSearchWithExtraMaskInvalidParameter(queryNum, querys, queryFilters, extraMask, extraMaskLen,
        extraScore, labelRes, distances, validnum, tsIndex);

    res = tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), false, K, extraMask.data(),
        extraMaskLen, false, extraScore.data(), labelRes.data(), distances.data(), validnum.data());
    EXPECT_EQ(res, 0);
}

// 测试所有索引递增且连续，没有新增索引的情况
TEST(TestAscendIndexUTTS, CheckIndicesShouldReturnOkWhenAllIndicesAreConsecutiveAndNoNewAdd)
{
    int64_t ntotal = 5;
    int64_t n = 3;
    int64_t indices[] = {1, 2, 3};
    int64_t replaceNum = 0;
    std::vector<std::pair<int64_t, int64_t>> segments;

    APP_ERROR result = TSBase::CheckIndices(ntotal, n, indices, replaceNum, segments);
    EXPECT_EQ(result, APP_ERR_OK);
    EXPECT_EQ(replaceNum, 3);
    EXPECT_EQ(segments.size(), 1);
    EXPECT_EQ(segments[0].first, 0);
    EXPECT_EQ(segments[0].second, 3);
}

// 测试部分索引递增且连续，有新增索引的情况
TEST(TestAscendIndexUTTS, CheckIndicesShouldReturnOkWhenSomeIndicesAreConsecutiveAndNewAddExists)
{
    int64_t ntotal = 5;
    int64_t n = 4;
    int64_t indices[] = {3, 4, 5, 6};
    int64_t replaceNum = 0;
    std::vector<std::pair<int64_t, int64_t>> segments;

    APP_ERROR result = TSBase::CheckIndices(ntotal, n, indices, replaceNum, segments);
    EXPECT_EQ(result, APP_ERR_OK);
    EXPECT_EQ(replaceNum, 2);
    EXPECT_EQ(segments.size(), 1);
    EXPECT_EQ(segments[0].first, 0);
    EXPECT_EQ(segments[0].second, 4);
}

// 测试n=1的情况
TEST(TestAscendIndexUTTS, CheckIndicesShouldReturnOkWhenNIsOne)
{
    int64_t ntotal = 5;
    int64_t n = 1;
    int64_t indices[] = {0};
    int64_t replaceNum = 0;
    std::vector<std::pair<int64_t, int64_t>> segments;

    APP_ERROR result = TSBase::CheckIndices(ntotal, n, indices, replaceNum, segments);
    EXPECT_EQ(result, APP_ERR_OK);
    EXPECT_EQ(replaceNum, 1);
    EXPECT_EQ(segments.size(), 1);
    EXPECT_EQ(segments[0].first, 0);
    EXPECT_EQ(segments[0].second, 1);
}

// 测试indices[0] = ntotal - 1的情况
TEST(TestAscendIndexUTTS, CheckIndicesShouldReturnOkWhenIndicesStartAtNtotalMinusOne)
{
    int64_t ntotal = 5;
    int64_t n = 2;
    int64_t indices[] = {4, 5};
    int64_t replaceNum = 0;
    std::vector<std::pair<int64_t, int64_t>> segments;
    APP_ERROR result = TSBase::CheckIndices(ntotal, n, indices, replaceNum, segments);
    EXPECT_EQ(result, APP_ERR_OK);
    EXPECT_EQ(replaceNum, 1);
    EXPECT_EQ(segments.size(), 1);
    EXPECT_EQ(segments[0].first, 0);
    EXPECT_EQ(segments[0].second, 2);
}

// 测试n=0的情况，应返回错误
TEST(TestAscendIndexUTTS, CheckIndicesShouldReturnErrorWhenNIsZero)
{
    int64_t ntotal = 5;
    int64_t n = 0;
    int64_t indices[] = {};
    int64_t replaceNum = 0;
    std::vector<std::pair<int64_t, int64_t>> segments;

    APP_ERROR result = TSBase::CheckIndices(ntotal, n, indices, replaceNum, segments);
    EXPECT_NE(result, APP_ERR_OK);
}

// 测试indices[0] < 0的情况，应返回错误
TEST(TestAscendIndexUTTS, CheckIndicesShouldReturnErrorWhenIndicesStartWithNegative)
{
    int64_t ntotal = 5;
    int64_t n = 2;
    int64_t indices[] = {-1, 0};
    int64_t replaceNum = 0;
    std::vector<std::pair<int64_t, int64_t>> segments;

    APP_ERROR result = TSBase::CheckIndices(ntotal, n, indices, replaceNum, segments);
    EXPECT_NE(result, APP_ERR_OK);
}

// 测试索引不递增的情况，应返回错误
TEST(TestAscendIndexUTTS, CheckIndicesShouldReturnErrorWhenIndicesAreNotAscending)
{
    int64_t ntotal = 5;
    int64_t n = 3;
    int64_t indices[] = {1, 3, 2};
    int64_t replaceNum = 0;
    std::vector<std::pair<int64_t, int64_t>> segments;

    APP_ERROR result = TSBase::CheckIndices(ntotal, n, indices, replaceNum, segments);
    EXPECT_NE(result, APP_ERR_OK);
}

// 测试索引不连续的情况，应返回错误
TEST(TestAscendIndexUTTS, CheckIndicesShouldReturnOkWhenIndicesAreNotConsecutive)
{
    int64_t ntotal = 5;
    int64_t n = 3;
    int64_t indices[] = {1, 2, 4};
    int64_t replaceNum = 0;
    std::vector<std::pair<int64_t, int64_t>> segments;

    APP_ERROR result = TSBase::CheckIndices(ntotal, n, indices, replaceNum, segments);
    EXPECT_EQ(result, APP_ERR_OK);
    EXPECT_EQ(replaceNum, 3);
    EXPECT_EQ(segments.size(), 2);
    EXPECT_EQ(segments[0].first, 0);
    EXPECT_EQ(segments[0].second, 2);
    EXPECT_EQ(segments[1].first, 2);
    EXPECT_EQ(segments[1].second, 1);
}

// 测试新增索引不从ntotal开始的情况，应返回错误
TEST(TestAscendIndexUTTS, CheckIndicesShouldReturnErrorWhenNewAddDoesNotStartAtNtotal)
{
    int64_t ntotal = 5;
    int64_t n = 2;
    int64_t indices[] = {5, 7};
    int64_t replaceNum = 0;
    std::vector<std::pair<int64_t, int64_t>> segments;

    APP_ERROR result = TSBase::CheckIndices(ntotal, n, indices, replaceNum, segments);
    EXPECT_NE(result, APP_ERR_OK);
}

// 测试replaceNum + newAddN != n的情况，应返回错误
TEST(TestAscendIndexUTTS, CheckIndicesShouldReturnErrorWhenReplaceNumPlusNewAddNNotEqualN)
{
    int64_t ntotal = 5;
    int64_t n = 3;
    int64_t indices[] = {3, 4, 6};
    int64_t replaceNum = 0;
    std::vector<std::pair<int64_t, int64_t>> segments;

    APP_ERROR result = TSBase::CheckIndices(ntotal, n, indices, replaceNum, segments);
    EXPECT_NE(result, APP_ERR_OK);
}

// 测试AddFeatureByIndice对入参的校验
TEST(TestAscendIndexUTTS, AddFeatureByIndiceShouldReturnErrorWhenAnyParamIsInvalid)
{
    int64_t count = 1;
    std::vector<float> features(count * DIM);
    std::vector<faiss::ascend::FeatureAttr> attributes(count);
    std::vector<int64_t> indices(count);
    std::vector<faiss::ascend::ExtraValAttr> extraVal(count);
    std::vector<uint8_t> customAttr(count);
    // 没有初始化就调用
    std::shared_ptr<faiss::ascend::AscendIndexTS> tsIndex = std::make_shared<faiss::ascend::AscendIndexTS>();
    auto res = tsIndex->AddFeatureByIndice(count, features.data(), attributes.data(), indices.data(),
        extraVal.data(), customAttr.data());
    EXPECT_EQ(res, APP_ERR_ILLEGAL_OPERATION);

    res = tsIndex->Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_IP_FP16);
    EXPECT_EQ(res, 0);

    // features is nullptr
    res = tsIndex->AddFeatureByIndice(count, nullptr, attributes.data(), indices.data(),
        extraVal.data(), customAttr.data());
    EXPECT_EQ(res, APP_ERR_INVALID_PARAM);

    // attributes is nullptr
    res = tsIndex->AddFeatureByIndice(count, features.data(), nullptr, indices.data(),
        extraVal.data(), customAttr.data());
    EXPECT_EQ(res, APP_ERR_INVALID_PARAM);

    // indices is nullptr
    res = tsIndex->AddFeatureByIndice(count, features.data(), attributes.data(), nullptr,
        extraVal.data(), customAttr.data());
    EXPECT_EQ(res, APP_ERR_INVALID_PARAM);

    // count is 0
    res = tsIndex->AddFeatureByIndice(0, features.data(), attributes.data(), indices.data(),
        extraVal.data(), customAttr.data());
    EXPECT_EQ(res, APP_ERR_INVALID_PARAM);

    // count is UPPER_LIMIT_FOR_ADD + 1
    res = tsIndex->AddFeatureByIndice(UPPER_LIMIT_FOR_ADD + 1, features.data(), attributes.data(), indices.data(),
        extraVal.data(), customAttr.data());
    EXPECT_EQ(res, APP_ERR_INVALID_PARAM);

    // indices[count - 1] = UPPER_LIMIT_FOR_NTOTAL
    indices[count - 1] = UPPER_LIMIT_FOR_NTOTAL;
    res = tsIndex->AddFeatureByIndice(count, features.data(), attributes.data(), indices.data(),
        extraVal.data(), customAttr.data());
    EXPECT_EQ(res, APP_ERR_INVALID_PARAM);

    indices[count - 1] = 0;
    res = tsIndex->AddFeatureByIndice(count, features.data(), attributes.data(), indices.data(),
        extraVal.data(), customAttr.data());
    EXPECT_EQ(res, 0);
}

// 测试GetFeatureByIndice对入参的校验
TEST(TestAscendIndexUTTS, GetFeatureByIndiceShouldReturnErrorWhenAnyParamIsInvalid)
{
    int64_t count = 1;
    std::vector<float> features(count * DIM);
    std::vector<faiss::ascend::FeatureAttr> attributes(count);
    std::vector<int64_t> indices(count);
    std::vector<int64_t> labels(count);
    std::vector<faiss::ascend::ExtraValAttr> extraVal(count);
    std::vector<uint8_t> customAttr(count);
    // 没有初始化就调用
    std::shared_ptr<faiss::ascend::AscendIndexTS> tsIndex = std::make_shared<faiss::ascend::AscendIndexTS>();
    auto res = tsIndex->GetFeatureByIndice(count, indices.data(), labels.data(), features.data(), attributes.data(),
        extraVal.data());
    EXPECT_EQ(res, APP_ERR_ILLEGAL_OPERATION);

    res = tsIndex->Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_IP_FP16);
    EXPECT_EQ(res, 0);

    // indices is nullptr
    res = tsIndex->GetFeatureByIndice(count, nullptr, labels.data(), features.data(), attributes.data(),
        extraVal.data());
    EXPECT_EQ(res, APP_ERR_INVALID_PARAM);

    // count is 0
    res = tsIndex->GetFeatureByIndice(0, indices.data(), labels.data(), features.data(), attributes.data(),
        extraVal.data());
    EXPECT_EQ(res, APP_ERR_INVALID_PARAM);

    // count is UPPER_LIMIT_FOR_ADD + 1
    res = tsIndex->GetFeatureByIndice(UPPER_LIMIT_FOR_ADD + 1, indices.data(), labels.data(), features.data(),
        attributes.data(), extraVal.data());
    EXPECT_EQ(res, APP_ERR_INVALID_PARAM);

    // 添加一条特征
    res = tsIndex->AddFeatureByIndice(count, features.data(), attributes.data(), indices.data(),
        extraVal.data(), customAttr.data());
    EXPECT_EQ(res, 0);

    // 获取labels、features等
    res = tsIndex->GetFeatureByIndice(count, indices.data(), labels.data(), features.data(),
        attributes.data(), extraVal.data());
    EXPECT_EQ(res, 0);

    // 仅调用，不获取lables、features等
    res = tsIndex->GetFeatureByIndice(count, indices.data(), nullptr, nullptr, nullptr, nullptr);
    EXPECT_EQ(res, 0);

    indices[0] = -1;
    res = tsIndex->GetFeatureByIndice(count, indices.data(), labels.data(), features.data(),
        attributes.data(), extraVal.data());
    EXPECT_EQ(res, APP_ERR_INVALID_PARAM);
}

// 测试GetFeatureByIndice对入参的校验
TEST(AscendIndexTSTest, FastDeleteFeatureByIndiceBatch)
{
    std::shared_ptr<faiss::ascend::AscendIndexTS> tsIndex = std::make_shared<faiss::ascend::AscendIndexTS>();
    auto ret = tsIndex->Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_IP_FP16);
    EXPECT_EQ(ret, 0);

    int64_t count = 5;
    std::vector<float> features(count * DIM);
    std::vector<faiss::ascend::FeatureAttr> attributes(count);
    std::vector<int64_t> indices = {0, 1, 2, 3, 4};
    std::vector<faiss::ascend::ExtraValAttr> extraVal(count);
    std::vector<uint8_t> customAttr(count);
    // 添加一些特征
    ret = tsIndex->AddFeatureByIndice(count, features.data(), attributes.data(), indices.data(),
        extraVal.data(), customAttr.data());
    EXPECT_EQ(ret, 0);

    // 测试正常删除特征
    count = 2;
    indices = {0, 1};
    ret = tsIndex->FastDeleteFeatureByIndice(count, indices.data());
    EXPECT_EQ(ret, APP_ERR_OK);

    // 测试边界条件：count为1
    count = 1;
    indices = {0};
    ret = tsIndex->FastDeleteFeatureByIndice(count, indices.data());
    EXPECT_EQ(ret, APP_ERR_OK);

    // 测试空指针情况
    count = 2;
    int64_t* indicesNull = nullptr;
    ret = tsIndex->FastDeleteFeatureByIndice(count, indicesNull);
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);

    // 测试count为负数
    count = -1;
    indices = {0};
    ret = tsIndex->FastDeleteFeatureByIndice(count, indices.data());
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);

    // 测试索引超出范围
    count = 6;
    ret = tsIndex->FastDeleteFeatureByIndice(count, indices.data());
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);

    // 测试异常处理
    count = 2;
    indices = {0, -1};
    ret = tsIndex->FastDeleteFeatureByIndice(count, indices.data());
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);
}

// 测试GetFeatureByIndice连续删除
TEST(AscendIndexTSTest, FastDeleteFeatureByIndiceRange)
{
    std::shared_ptr<faiss::ascend::AscendIndexTS> tsIndex = std::make_shared<faiss::ascend::AscendIndexTS>();
    auto ret = tsIndex->Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_IP_FP16);
    EXPECT_EQ(ret, 0);

    int64_t count = 5;
    std::vector<float> features(count * DIM);
    std::vector<faiss::ascend::FeatureAttr> attributes(count);
    std::vector<int64_t> indices = {0, 1, 2, 3, 4};
    std::vector<faiss::ascend::ExtraValAttr> extraVal(count);
    std::vector<uint8_t> customAttr(count);
    // 添加一些特征
    ret = tsIndex->AddFeatureByIndice(count, features.data(), attributes.data(), indices.data(),
        extraVal.data(), customAttr.data());
    EXPECT_EQ(ret, 0);

    int64_t validNum = 0;
    tsIndex->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, count);

    // 测试正常删除特征
    int64_t start = 0;
    count = 2;
    ret = tsIndex->FastDeleteFeatureByRange(start, count);
    EXPECT_EQ(ret, APP_ERR_OK);

    // 测试边界条件：start为0，count为最大值
    start = 0;
    count = validNum;
    ret = tsIndex->FastDeleteFeatureByRange(start, count);
    EXPECT_EQ(ret, APP_ERR_OK);

    // 测试边界条件：start为最大值减去count，确保start + count刚好小于validNum
    count = 2;
    start = validNum - count;
    ret = tsIndex->FastDeleteFeatureByRange(start, count);
    EXPECT_EQ(ret, APP_ERR_OK);

    // 测试start为负数
    start = -1;
    ret = tsIndex->FastDeleteFeatureByRange(start, count);
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);

    // 测试start等于或大于validNum
    start = validNum;
    ret = tsIndex->FastDeleteFeatureByRange(start, count);
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);

    // 测试count为负数
    start = 0;
    count = -1;
    ret = tsIndex->FastDeleteFeatureByRange(start, count);
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);

    count = validNum + 1;
    ret = tsIndex->FastDeleteFeatureByRange(start, count);
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);

    // 测试start + count等于或大于pImpl->getAttrTotal()
    start = validNum - 1;
    count = 2;
    ret = tsIndex->FastDeleteFeatureByRange(start, count);
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);
}

// 测试GetBaseMask
TEST(AscendIndexTSTest, GetBaseMask)
{
    std::shared_ptr<faiss::ascend::AscendIndexTS> tsIndex = std::make_shared<faiss::ascend::AscendIndexTS>();

    // 测试mask为空
    auto ret = tsIndex->GetBaseMask(1, nullptr);
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);

    // 测试没有初始化就删除
    std::vector<uint8_t> mask(1);
    ret = tsIndex->GetBaseMask(1, mask.data());
    EXPECT_EQ(ret, APP_ERR_ILLEGAL_OPERATION);

    ret = tsIndex->Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_IP_FP16);
    EXPECT_EQ(ret, 0);

    int64_t count = 5;
    std::vector<float> features(count * DIM);
    std::vector<faiss::ascend::FeatureAttr> attributes(count);
    std::vector<int64_t> indices = {0, 1, 2, 3, 4};
    std::vector<faiss::ascend::ExtraValAttr> extraVal(count);
    std::vector<uint8_t> customAttr(count);
    // 添加一些特征
    ret = tsIndex->AddFeatureByIndice(count, features.data(), attributes.data(), indices.data(),
        extraVal.data(), customAttr.data());
    EXPECT_EQ(ret, 0);

    // 替换一些
    ret = tsIndex->AddFeatureByIndice(count, features.data(), attributes.data(), indices.data(),
        extraVal.data(), customAttr.data());
    EXPECT_EQ(ret, 0);

    int64_t validNum = 0;
    tsIndex->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, count);

    int64_t maskLen = (validNum + 8 - 1) / 8;
    mask.resize(maskLen);
    ret = tsIndex->GetBaseMask(maskLen, mask.data());
    EXPECT_EQ(ret, 0);
}

TEST(TestAscendIndexUTTS, GetBaseByRangeWithExtraValTestParameter)
{
    std::shared_ptr<faiss::ascend::AscendIndexTS> tsIndex = std::make_shared<faiss::ascend::AscendIndexTS>();
    auto ret = tsIndex->Init(0, DIM, TOKENNUM, faiss::ascend::AlgorithmType::FLAT_IP_FP16);
    EXPECT_EQ(ret, 0);

    int64_t count = 5;
    std::vector<float> features(count * DIM);
    std::vector<faiss::ascend::FeatureAttr> attributes(count);
    std::vector<int64_t> indices = {0, 1, 2, 3, 4};
    std::vector<faiss::ascend::ExtraValAttr> extraVal(count);
    std::vector<uint8_t> customAttr(count);
    std::vector<int64_t> labels(count);
    // 添加一些特征
    ret = tsIndex->AddFeatureByIndice(count, features.data(), attributes.data(), indices.data(),
        extraVal.data(), customAttr.data());
    EXPECT_EQ(ret, 0);

    // 从0位置开始获取1条
    ret = tsIndex->GetBaseByRangeWithExtraVal(0, 1, labels.data(), features.data(), attributes.data(),
        extraVal.data());
    EXPECT_EQ(ret, 0);

    // 从0位置开始获取，count条
    ret = tsIndex->GetBaseByRangeWithExtraVal(0, count, labels.data(), features.data(), attributes.data(),
        extraVal.data());
    EXPECT_EQ(ret, 0);

    // 从1位置开始获取，count条
    ret = tsIndex->GetBaseByRangeWithExtraVal(1, count, labels.data(), features.data(), attributes.data(),
        extraVal.data());
    EXPECT_EQ(ret, APP_ERR_ILLEGAL_OPERATION);

    ret = tsIndex->GetFeatureAttrByLabel(0, labels.data(), attributes.data());
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);

    ret = tsIndex->GetFeatureAttrByLabel(1, labels.data(), attributes.data());
    EXPECT_EQ(ret, 0);
}

// 测试有量化的Init初始化参数校验
TEST(AscendIndexTSTest, TestInitWithQuantify)
{
    uint64_t resources = 1024 * 1024 * 1024;
    std::vector<float> scale(DIM, 1.0f);

    faiss::ascend::AscendIndexTS index;
    // deviceId 越界
    APP_ERROR ret = index.InitWithQuantify(1025, DIM, TOKENNUM, resources, scale.data()); // invalid id 1025
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);

    // tokenNum 越界
    ret = index.InitWithQuantify(0, DIM, 0, resources, scale.data());
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);

    // tokenNum 越界
    ret = index.InitWithQuantify(0, DIM, 3.0E5+1, resources, scale.data()); // invalid token 3E5+1
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);

    // customAttrLen 越界
    uint32_t customAttrLen = 33; // invalid len 33
    ret = index.InitWithQuantify(0, DIM, TOKENNUM, resources, scale.data(),
        faiss::ascend::AlgorithmType::FLAT_IP_FP16, customAttrLen, 0);
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);

    // customAttrBlockSize 越界
    customAttrLen = 32; // invalid len 33
    uint32_t customAttrBlockSize = 16777217; // invalid len 16777216
    ret = index.InitWithQuantify(0, DIM, TOKENNUM, resources, scale.data(),
        faiss::ascend::AlgorithmType::FLAT_IP_FP16, customAttrLen, customAttrBlockSize);
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);

    // customAttrLen 和 customAttrBlockSize 不对齐
    customAttrBlockSize = 2621445; // invalid len 2621445
    ret = index.InitWithQuantify(0, DIM, TOKENNUM, resources, scale.data(),
        faiss::ascend::AlgorithmType::FLAT_IP_FP16, customAttrLen, customAttrBlockSize);
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);

    // scale 为 nullptr
    ret = index.InitWithQuantify(0, DIM, TOKENNUM, resources, nullptr);
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);

    // scale 中有非法值（接近 0）
    scale.back() = 1e-7f;
    ret = index.InitWithQuantify(0, DIM, TOKENNUM, resources, scale.data());
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);

    // 不支持的算法类型
    scale.back() = 1.0f;
    ret = index.InitWithQuantify(0, DIM, TOKENNUM, resources, scale.data(),
        faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(ret, APP_ERR_NOT_IMPLEMENT);

    // 正常初始化
    ret = index.InitWithQuantify(0, DIM, TOKENNUM, resources, scale.data());
    EXPECT_EQ(ret, APP_ERR_OK);

    // 重复初始化
    ret = index.InitWithQuantify(0, DIM, TOKENNUM, resources, scale.data());
    EXPECT_EQ(ret, APP_ERR_ILLEGAL_OPERATION);
}

// 测试有量化的特征添加和获取功能
TEST(AscendIndexTSTest, TestAddWithQuantify)
{
    uint64_t resources = 1024 * 1024 * 1024;
    std::vector<float> scale(DIM, 1.0f);

    faiss::ascend::AscendIndexTS index;
    // 需要进行量化
    APP_ERROR ret = index.InitWithQuantify(0, DIM, TOKENNUM, resources, scale.data());
    EXPECT_EQ(ret, APP_ERR_OK);

    int64_t count = 10;
    std::vector<float> features(count * DIM);
    std::vector<faiss::ascend::FeatureAttr> attributes(count);
    std::vector<int64_t> indices(count);
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<faiss::ascend::ExtraValAttr> extraVal(count);
    std::vector<uint8_t> customAttr(count);

    // 添加底库需要量化
    ret = index.AddFeatureByIndice(count, features.data(), attributes.data(), indices.data(),
        extraVal.data(), customAttr.data());
    EXPECT_EQ(ret, APP_ERR_OK);

    // 从0位置开始获取，count条
    std::vector<int64_t> labels(count);
    ret = index.GetBaseByRangeWithExtraVal(0, count, labels.data(), features.data(), attributes.data(),
        extraVal.data());
    EXPECT_EQ(ret, APP_ERR_OK);

    // 获取0位置的特征信息
    ret = index.GetFeatureByIndice(1, indices.data(), labels.data(), features.data(), attributes.data(),
        extraVal.data());
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST(AscendIndexTSTest, TestScaleAndSearch)
{
    int extraMaskLen = 1;
    int k = 1;
    int64_t baseSize = 1;
    int64_t queryNum = baseSize;
    std::vector<float> features;
    std::vector<int64_t> labels;
    std::vector<faiss::ascend::FeatureAttr> attrs;
    std::vector<float> querys;
    std::vector<uint8_t> bitSet;
    std::vector<faiss::ascend::AttrFilter> queryFilters;
    std::vector<uint8_t> extraMask;
    GenerateData(queryNum, queryNum, DIM, extraMaskLen, features, labels,
                 attrs, querys, bitSet, queryFilters, extraMask);
    uint64_t resources = 1024 * 1024 * 1024;
    std::vector<float> scale(DIM, 1.0f);

    faiss::ascend::AscendIndexTS index;
    // 需要进行量化
    APP_ERROR ret = index.InitWithQuantify(0, DIM, TOKENNUM, resources, scale.data());
    EXPECT_EQ(ret, APP_ERR_OK);

    std::vector<int64_t> indices(baseSize);
    std::iota(indices.begin(), indices.end(), 0);

    // 添加底库需要量化
    ret = index.AddFeatureByIndice(baseSize, features.data(), attrs.data(), indices.data());
    EXPECT_EQ(ret, APP_ERR_OK);

    std::vector<float> distances(queryNum * k, -1);
    std::vector<int64_t> labelRes(queryNum * k, -1);
    std::vector<uint32_t> validnum(queryNum, 1);

    ret = index.Search(queryNum, querys.data(), queryFilters.data(), false, k, labelRes.data(),
        distances.data(), validnum.data(), false);
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST(AscendIndexTSTest, TestScaleInvalidAlgorithmType)
{
    uint64_t resources = 1024 * 1024 * 1024;
    std::vector<float> scale(DIM, 1.0f);

    faiss::ascend::AscendIndexTS index;
    // 需要进行量化
    APP_ERROR ret = index.InitWithQuantify(0, DIM, TOKENNUM, resources, scale.data(),
        faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(ret, APP_ERR_NOT_IMPLEMENT);
}

TEST(AscendIndexTSTest, TestScaleShareMask)
{
    int extraMaskLen = 1;
    int k = 1;
    int64_t baseSize = 1;
    int64_t queryNum = baseSize;
    std::vector<float> features;
    std::vector<int64_t> labels;
    std::vector<faiss::ascend::FeatureAttr> attrs;
    std::vector<float> querys;
    std::vector<uint8_t> bitSet;
    std::vector<faiss::ascend::AttrFilter> queryFilters;
    std::vector<uint8_t> extraMask;
    GenerateData(queryNum, queryNum, DIM, extraMaskLen, features, labels,
                 attrs, querys, bitSet, queryFilters, extraMask);
    uint64_t resources = 1024 * 1024 * 1024;
    std::vector<float> scale(DIM, 1.0f);

    faiss::ascend::AscendIndexTS index;
    // 需要进行量化
    APP_ERROR ret = index.InitWithQuantify(0, DIM, TOKENNUM, resources, scale.data());
    EXPECT_EQ(ret, APP_ERR_OK);

    std::vector<int64_t> indices(baseSize);
    std::iota(indices.begin(), indices.end(), 0);

    // 添加底库需要量化
    ret = index.AddFeatureByIndice(baseSize, features.data(), attrs.data(), indices.data());
    EXPECT_EQ(ret, APP_ERR_OK);

    std::vector<float> distances(queryNum * k, -1);
    std::vector<int64_t> labelRes(queryNum * k, -1);
    std::vector<uint32_t> validnum(queryNum, 1);

    ret = index.Search(queryNum, querys.data(), queryFilters.data(), true, k, labelRes.data(),
        distances.data(), validnum.data(), false);
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);
}

} // namespace ascend
