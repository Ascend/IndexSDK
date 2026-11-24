/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * IndexSDK is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
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
enum class ErrorCode { APP_ERR_OK = 0, APP_ERR_INVALID_PARAM = 2001, APP_ERR_ILLEGAL_OPERATION = 2009 };

const int BINARY_BYTE_SIZE = 8;
const int DEVICE = 0;
const int DIM = 256;
const int BASE_SIZE = 100;
const int RANGEMIN = 0;
const int RANGEMAX = 4;
const int SCORE_ALGIN = 16;
const uint32_t DEFAULT_TOKEN = 2500;
const int64_t DEFAULT_MEM = 0x60000000;
const int K = 1;
const uint32_t BLOCKSIZE = 262144;
const uint32_t ATTRLEN = 2;
const int BATCH = 1;
const int EXTRA_MASK_LEN = 10;
constexpr size_t KB = 1024;
constexpr size_t VALID_DEVICE_CAPACITY = KB * KB * KB;    // 1073741824
constexpr size_t VALID_DEVICE_BUFFER = 9 * 64 * KB * KB;  // 603979776
constexpr size_t VALID_HOST_CAPACITY = 20 * KB * KB * KB; // 21474836480

TEST(TestAscendIndexTSInt8Flat, SearchWithExtraMask)
{
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex->Init(0, DIM, DEFAULT_TOKEN, faiss::ascend::AlgorithmType::FLAT_COS_INT8,
        faiss::ascend::MemoryStrategy::PURE_DEVICE_MEMORY, ATTRLEN, BLOCKSIZE);
    EXPECT_EQ(ret, 0);

    std::vector<int8_t> features(BASE_SIZE * DIM);
    ascend::FeatureGenerator(features);

    std::vector<int64_t> labels;
    for (int64_t j = 0; j < BASE_SIZE; ++j) {
        labels.emplace_back(j);
    }
    std::vector<faiss::ascend::FeatureAttr> attrs(BASE_SIZE);
    ascend::FeatureAttrGenerator(attrs);

    std::vector<uint8_t> customAttrs(BASE_SIZE * ATTRLEN, 0);
    ascend::customAttrGenerator(customAttrs, ATTRLEN);

    ret = tsIndex->AddFeature(BASE_SIZE, features.data(), attrs.data(), labels.data(), customAttrs.data());
    EXPECT_EQ(ret, 0);

    uint8_t *customAttrBase = nullptr;
    ret = tsIndex->GetCustomAttrByBlockId(0, customAttrBase);
    EXPECT_EQ(ret, 0);

    std::vector<float> distances(BATCH * K, -1);
    std::vector<int64_t> labelRes(BATCH * K, -1);
    std::vector<uint32_t> validnum(BATCH, 0);
    std::vector<int8_t> querys(BATCH * DIM);
    querys.assign(features.begin(), features.begin() + BATCH * DIM);

    uint32_t setlen = ((DEFAULT_TOKEN + 7) / 8);
    std::vector<uint8_t> bitSet(setlen, 11);

    faiss::ascend::AttrFilter filter{};
    filter.timesStart = 0;
    filter.timesEnd = 3;
    filter.tokenBitSet = bitSet.data();
    filter.tokenBitSetLen = setlen;

    std::vector<faiss::ascend::AttrFilter> queryFilters(BATCH, filter);
    std::vector<uint8_t> extraMask(BATCH * EXTRA_MASK_LEN, 0);

    for (int j = 0; j < EXTRA_MASK_LEN; j++) {
        extraMask[j] = 0x1 << 1;
    }
    ret = tsIndex->SearchWithExtraMask(BATCH, querys.data(), queryFilters.data(), false, K, extraMask.data(),
        EXTRA_MASK_LEN, false, labelRes.data(), distances.data(), validnum.data());
    EXPECT_EQ(ret, 0);

    ret = tsIndex->SearchWithExtraMask(BATCH, querys.data(), queryFilters.data(), true, K, extraMask.data(),
        EXTRA_MASK_LEN, false, labelRes.data(), distances.data(), validnum.data());
    EXPECT_EQ(ret, 0);

    int deleteCount = 10;
    std::vector<int64_t> delLabel(deleteCount, 0);
    ret = tsIndex->DeleteFeatureByLabel(deleteCount, delLabel.data());
    EXPECT_EQ(ret, 0);

    delete tsIndex;
}

TEST(TestAscendIndexTSInt8Flat, SearchWithExtraMaskExtraScore)
{
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex->Init(0, DIM, DEFAULT_TOKEN, faiss::ascend::AlgorithmType::FLAT_COS_INT8,
        faiss::ascend::MemoryStrategy::PURE_DEVICE_MEMORY, ATTRLEN, BLOCKSIZE);
    std::vector<int8_t> features(BASE_SIZE * DIM);
    ascend::FeatureGenerator(features);
    std::vector<int64_t> labels;
    for (int64_t j = 0; j < BASE_SIZE; ++j) {
        labels.emplace_back(j);
    }
    std::vector<faiss::ascend::FeatureAttr> attrs(BASE_SIZE);
    ascend::FeatureAttrGenerator(attrs);
    std::vector<uint8_t> customAttrs(BASE_SIZE * ATTRLEN, 0);
    ascend::customAttrGenerator(customAttrs, ATTRLEN);
    ret = tsIndex->AddFeature(BASE_SIZE, features.data(), attrs.data(), labels.data(), customAttrs.data());
    EXPECT_EQ(ret, 0);

    std::vector<float> distances(BATCH * K, -1);
    std::vector<int64_t> labelRes(BATCH * K, -1);
    std::vector<uint32_t> validnum(BATCH, 0);
    std::vector<int8_t> querys(BATCH * DIM);
    querys.assign(features.begin(), features.begin() + BATCH * DIM);

    uint32_t setlen = ((DEFAULT_TOKEN + 7) / 8);
    std::vector<uint8_t> bitSet(setlen, 11);

    faiss::ascend::AttrFilter filter{};
    filter.timesStart = 0;
    filter.timesEnd = 3;
    filter.tokenBitSet = bitSet.data();
    filter.tokenBitSetLen = setlen;

    std::vector<faiss::ascend::AttrFilter> queryFilters(BATCH, filter);
    std::vector<uint8_t> extraMask(BATCH * EXTRA_MASK_LEN, 0);
    int extraScoreLen = BASE_SIZE % SCORE_ALGIN == 0 ? BASE_SIZE : (BASE_SIZE / SCORE_ALGIN + 1) * SCORE_ALGIN;
    std::vector<uint16_t> extraScore(BATCH * extraScoreLen);

    for (int j = 0; j < EXTRA_MASK_LEN; j++) {
        extraMask[j] = 0x1 << 1;
    }
    ret = tsIndex->SearchWithExtraMask(BATCH, querys.data(), queryFilters.data(), false, K, extraMask.data(),
        EXTRA_MASK_LEN, false, extraScore.data(), labelRes.data(), distances.data(), validnum.data());
    EXPECT_EQ(ret, 0);

    int deleteCount = 10;
    std::vector<int64_t> delLabel(deleteCount, 0);
    ret = tsIndex->DeleteFeatureByLabel(deleteCount, delLabel.data());
    EXPECT_EQ(ret, 0);

    delete tsIndex;
}

TEST(TestAscendIndexTSInt8Flat, GetBaseByRange)
{
    using namespace std;
    cout << "***** start Init *****" << endl;
    faiss::ascend::AscendIndexTS *tsIndex0 = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex0->Init(0, DIM, DEFAULT_TOKEN, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(ret, 0);
    cout << "***** end Init *****" << endl;

    cout << "***** start add *****" << endl;
    std::vector<int8_t> features(BASE_SIZE * DIM);
    ascend::FeatureGenerator(features);

    std::vector<int64_t> labels;
    for (int64_t j = 0; j < BASE_SIZE; ++j) {
        labels.emplace_back(j);
    }
    std::vector<faiss::ascend::FeatureAttr> attrs(BASE_SIZE);
    ascend::FeatureAttrGenerator(attrs);

    ret = tsIndex0->AddFeature(BASE_SIZE, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(ret, 0);

    int64_t validNum = 0;
    tsIndex0->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, BASE_SIZE);

    int needAddNum = validNum / 2;
    std::vector<int8_t> retBase(needAddNum * DIM);
    std::vector<int64_t> retLabels(needAddNum);
    std::vector<faiss::ascend::FeatureAttr> retAttrs(needAddNum);
    ret = tsIndex0->GetBaseByRange(needAddNum, needAddNum, retLabels.data(), retBase.data(), retAttrs.data());
    EXPECT_EQ(ret, 0);

    ret = tsIndex0->DeleteFeatureByLabel(needAddNum, retLabels.data());
    EXPECT_EQ(ret, 0);

    delete tsIndex0;
}

TEST(TestAscendIndexTSInt8Flat, GetFeatureAttrByLabel)
{
    uint32_t addNum = 1;

    cout << "***** start create index *****" << endl;
    faiss::ascend::AscendIndexTS tsIndex;
    auto ret = tsIndex.Init(0, DIM, DEFAULT_TOKEN, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    ASSERT_EQ(ret, 0);
    cout << "***** end create index *****" << endl;

    std::vector<int8_t> features(BASE_SIZE * DIM);
    cout << "***** add *****" << endl;
    ascend::FeatureGenerator(features);
    std::vector<faiss::ascend::FeatureAttr> attrs(BASE_SIZE);
    ascend::FeatureAttrGenerator(attrs);
    for (size_t i = 0; i < addNum; i++) {
        std::vector<int64_t> labels;
        for (int64_t j = 0; j < BASE_SIZE; ++j) {
            labels.emplace_back(j + i * BASE_SIZE);
        }
        tsIndex.AddFeature(BASE_SIZE, features.data(), attrs.data(), labels.data());
    }

    vector<int64_t> labels;
    for (int64_t i = BASE_SIZE * addNum - 1; i >= 0; i -= 262144) {
        labels.emplace_back(i);
    }
    vector<faiss::ascend::FeatureAttr> attrsOut(labels.size());
    ret = tsIndex.GetFeatureAttrByLabel(labels.size(), labels.data(), attrsOut.data());
    ASSERT_EQ(ret, 0);
}

bool IsExtraAttrSame(const faiss::ascend::ExtraValAttr &lAttr, const faiss::ascend::ExtraValAttr &rAttr)
{
    return (lAttr.val == rAttr.val);
}

TEST(TestAscendIndexTSInt8Flat, GetExtraValAttrByLabel)
{
    uint32_t addNum = 1;
    cout << "***** start create index *****" << endl;
    faiss::ascend::AscendIndexTS tsIndex;
    auto ret =
        tsIndex.InitWithExtraVal(0, DIM, DEFAULT_TOKEN, DEFAULT_MEM, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    ASSERT_EQ(ret, 0);
    cout << "***** end create index *****" << endl;

    std::vector<int8_t> features(BASE_SIZE * DIM);
    printf("[add -----------]\n");
    ascend::FeatureGenerator(features);
    std::vector<faiss::ascend::FeatureAttr> attrs(BASE_SIZE);
    std::vector<faiss::ascend::ExtraValAttr> valAttrs(BASE_SIZE);
    ascend::FeatureAttrGenerator(attrs);
    ascend::ExtraValAttrGenerator(valAttrs);
    auto res = tsIndex.SetSaveHostMemory();
    EXPECT_EQ(res, 0);
    for (size_t i = 0; i < addNum; i++) {
        std::vector<int64_t> labels;
        for (int64_t j = 0; j < BASE_SIZE; ++j) {
            labels.emplace_back(j + i * BASE_SIZE);
        }
        tsIndex.AddWithExtraVal(BASE_SIZE, features.data(), attrs.data(), labels.data(), valAttrs.data());
    }

    std::vector<int64_t> labels;
    for (int64_t i = BASE_SIZE * addNum - 1; i >= 0; i -= 262144) {
        labels.emplace_back(i);
    }
    std::vector<faiss::ascend::ExtraValAttr> attrsOut(labels.size());
    ret = tsIndex.GetExtraValAttrByLabel(labels.size(), labels.data(), attrsOut.data());

    ASSERT_EQ(ret, 0);
    for (size_t ilb = 0; ilb < labels.size(); ilb++) {
        auto lb = labels[ilb];
        int64_t lb100 = lb % BASE_SIZE;
        ASSERT_TRUE(IsExtraAttrSame(valAttrs[lb100], attrsOut[ilb]));
    }
}

TEST(TestAscendIndexTSInt8Flat, GetBaseByRangeWithExtraVal)
{
    uint32_t addNum = 10;
    using namespace std;
    cout << "***** start Init *****" << endl;
    faiss::ascend::AscendIndexTS *tsIndex0 = new faiss::ascend::AscendIndexTS();
    auto ret =
        tsIndex0->InitWithExtraVal(0, DIM, DEFAULT_TOKEN, DEFAULT_MEM, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(ret, 0);
    cout << "***** end Init *****" << endl;

    cout << "***** start add *****" << endl;
    std::vector<int8_t> features(BASE_SIZE * DIM);

    ascend::FeatureGenerator(features);

    std::vector<int64_t> labels;
    for (int64_t j = 0; j < BASE_SIZE; ++j) {
        labels.emplace_back(j);
    }
    std::vector<faiss::ascend::FeatureAttr> attrs(BASE_SIZE);
    std::vector<faiss::ascend::ExtraValAttr> valAttrs(BASE_SIZE);
    ascend::FeatureAttrGenerator(attrs);
    ascend::ExtraValAttrGenerator(valAttrs);
    printf("[AddWithExtraVal -----------]\n");
    ret = tsIndex0->AddWithExtraVal(BASE_SIZE, features.data(), attrs.data(), labels.data(), valAttrs.data());
    EXPECT_EQ(ret, 0);

    int64_t validNum = 0;
    tsIndex0->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, BASE_SIZE);
    printf("[GetFeatureNum -----------]\n");
    int needAddNum = validNum / 2;
    std::vector<int8_t> retBase(needAddNum * DIM);
    std::vector<int64_t> retLabels(needAddNum);
    std::vector<faiss::ascend::FeatureAttr> retAttrs(needAddNum);
    std::vector<faiss::ascend::ExtraValAttr> retValAttrs(needAddNum);

    ret = tsIndex0->GetBaseByRangeWithExtraVal(
        needAddNum, needAddNum, retLabels.data(), retBase.data(), retAttrs.data(), retValAttrs.data());
    EXPECT_EQ(ret, 0);

#pragma omp parallel for if (BASE_SIZE > 100)
    for (int i = 0; i < needAddNum * DIM; i++) {
        EXPECT_EQ(features[i + needAddNum * DIM], retBase[i]);
        if (features[i + needAddNum * DIM] != retBase[i]) {
            printf("%u--%u \n", features[i + needAddNum * DIM], retBase[i]);
        }
    }

    for (int i = 0; i < needAddNum; i++) {
        EXPECT_EQ(retValAttrs[i].val, valAttrs[i + needAddNum].val);
    }
    delete tsIndex0;
}

TEST(TestAscendIndexTSInt8Flat, SearchWithExtraVal)
{
    faiss::ascend::AscendIndexTS tsIndex;
    auto ret =
        tsIndex.InitWithExtraVal(0, DIM, DEFAULT_TOKEN, DEFAULT_MEM, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(ret, 0);

    std::vector<int8_t> features(BASE_SIZE * DIM);
    ascend::FeatureGenerator(features);
    std::vector<faiss::ascend::FeatureAttr> attrs(BASE_SIZE);
    std::vector<faiss::ascend::ExtraValAttr> valAttrs(BASE_SIZE);
    ascend::FeatureAttrGenerator(attrs);
    ascend::ExtraValAttrGenerator(valAttrs);
    std::vector<int64_t> labels;
    for (int64_t j = 0; j < BASE_SIZE; ++j) {
        labels.emplace_back(j);
    }
    ret = tsIndex.AddWithExtraVal(BASE_SIZE, features.data(), attrs.data(), labels.data(), valAttrs.data());
    EXPECT_EQ(ret, 0);

    std::vector<float> distances(BATCH * K, -1);
    std::vector<int64_t> labelRes(BATCH * K, 10);
    std::vector<uint32_t> validnum(BATCH, 0);
    uint32_t size = BATCH * DIM;
    std::vector<uint8_t> querys(size);
    querys.assign(features.begin(), features.begin() + size);

    uint32_t setlen = (DEFAULT_TOKEN + 7) / 8;
    std::vector<uint8_t> bitSet(setlen, 255);
    faiss::ascend::AttrFilter filter{};
    filter.timesStart = 0;
    filter.timesEnd = 3;
    filter.tokenBitSet = bitSet.data();
    filter.tokenBitSetLen = setlen;

    faiss::ascend::ExtraValFilter valFilter{};

    valFilter.filterVal = 1;
    valFilter.matchVal = 0;

    std::vector<faiss::ascend::AttrFilter> queryFilters(BATCH, filter);
    std::vector<faiss::ascend::ExtraValFilter> valFilters(BATCH, valFilter);

    ret = tsIndex.SearchWithExtraVal(BATCH, querys.data(), queryFilters.data(), false, K, labelRes.data(),
        distances.data(), validnum.data(), valFilters.data());
    EXPECT_EQ(ret, 0);
}

TEST(TestAscendIndexTSInt8Flat, AddByIndices)
{
    faiss::ascend::AscendIndexTS tsIndex;
    auto ret =
        tsIndex.InitWithExtraVal(0, DIM, DEFAULT_TOKEN, DEFAULT_MEM, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(ret, 0);

    std::vector<int8_t> features(BASE_SIZE * DIM);
    ascend::FeatureGenerator(features);
    std::vector<faiss::ascend::FeatureAttr> attrs(BASE_SIZE);
    std::vector<faiss::ascend::ExtraValAttr> valAttrs(BASE_SIZE);
    ascend::FeatureAttrGenerator(attrs);
    ascend::ExtraValAttrGenerator(valAttrs);
    std::vector<uint8_t> customAttr(BASE_SIZE);
    ascend::customAttrGenerator(customAttr, ATTRLEN);
    std::vector<int64_t> indices;
    for (int64_t j = 0; j < BASE_SIZE; ++j) {
        indices.emplace_back(j);
    }
    ret = tsIndex.AddFeatureByIndice(
        BASE_SIZE, features.data(), attrs.data(), indices.data(), valAttrs.data(), customAttr.data());

    EXPECT_EQ(ret, 0);
    std::vector<int8_t> getFeatures(DIM, -1);
    std::vector<int64_t> indice{ 1 };
    std::vector<faiss::ascend::FeatureAttr> getAttrs(1);
    std::vector<faiss::ascend::ExtraValAttr> getValAttrs(1);
    std::vector<int64_t> getLabel(1);
    ret = tsIndex.GetFeatureByIndice(
        1, indice.data(), getLabel.data(), getFeatures.data(), getAttrs.data(), getValAttrs.data());
    EXPECT_EQ(ret, 0);
    indice[0] = 2;
    std::vector<uint8_t> addCustomAttr(1);
    ret = tsIndex.AddFeatureByIndice(
        1, getFeatures.data(), getAttrs.data(), indice.data(), getValAttrs.data(), addCustomAttr.data());

    EXPECT_EQ(ret, 0);
    std::vector<int8_t> getFeatures1(DIM, -1);
    std::vector<faiss::ascend::FeatureAttr> getAttrs1(1);
    std::vector<faiss::ascend::ExtraValAttr> getValAttrs1(1);
    ret = tsIndex.GetFeatureByIndice(
        1, indice.data(), getLabel.data(), getFeatures1.data(), getAttrs1.data(), getValAttrs1.data());
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(getFeatures[0], getFeatures1[0]);
    EXPECT_EQ(getAttrs[0].tokenId, getAttrs1[0].tokenId);
    EXPECT_EQ(getValAttrs[0].val, getValAttrs1[0].val);
}

// 统计底库区间 [0, totalNum) 中每个待查 token 实际占用的条目数
std::vector<int64_t> CountTokensInBase(
    faiss::ascend::AscendIndexTS &index, int64_t totalNum, const std::vector<uint32_t> &tokens)
{
    std::vector<int64_t> labels(totalNum);
    std::vector<int8_t> baseVectors(static_cast<size_t>(totalNum) * DIM);
    std::vector<faiss::ascend::FeatureAttr> attrs(totalNum);
    auto ret =
        index.GetBaseByRange(0, static_cast<uint32_t>(totalNum), labels.data(), baseVectors.data(), attrs.data());
    EXPECT_EQ(ret, 0);
    std::vector<int64_t> counts(tokens.size(), 0);
    for (int64_t i = 0; i < totalNum; ++i) {
        for (size_t t = 0; t < tokens.size(); ++t) {
            if (attrs[i].tokenId == tokens[t]) {
                ++counts[t];
            }
        }
    }
    return counts;
}

// 覆盖写（AddFeatureByIndice）后，token 索引必须与底库属性严格一致：
// 位置 10~19 在 token C 与 token B 之间反复覆盖后，token C 只应剩位置 99 一个真实成员；
// 若覆盖写时未摘除旧 token 的条目，DeleteFeatureByToken(C) 会连带删掉 10~19 这 10 条 B 的数据。
TEST(TestAscendIndexTSInt8Flat, DeleteTokenAfterOverwriteDeletesExactCount)
{
    const uint32_t tokenA = 1;
    const uint32_t tokenB = 2;
    const uint32_t tokenC = 3;
    const int overwriteNum = 10;
    const int overwriteStart = 10;
    const int roundNum = 3;

    faiss::ascend::AscendIndexTS tsIndex;
    auto ret =
        tsIndex.InitWithExtraVal(0, DIM, DEFAULT_TOKEN, DEFAULT_MEM, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    ASSERT_EQ(ret, 0);

    std::vector<int8_t> features(BASE_SIZE * DIM);
    ascend::FeatureGenerator(features);
    std::vector<faiss::ascend::ExtraValAttr> valAttrs(BASE_SIZE);
    ascend::ExtraValAttrGenerator(valAttrs);

    std::vector<int64_t> labels(BASE_SIZE);
    std::vector<faiss::ascend::FeatureAttr> attrs(BASE_SIZE);
    for (int64_t i = 0; i < BASE_SIZE; ++i) {
        labels[i] = i;
        attrs[i].time = static_cast<int32_t>(i);
        attrs[i].tokenId = tokenB;
    }
    for (int i = 0; i < overwriteNum; ++i) {
        attrs[i].tokenId = tokenA; // 位置 0~9 属于 A
    }
    attrs[BASE_SIZE - 1].tokenId = tokenC; // 位置 99 属于 C
    ret = tsIndex.AddWithExtraVal(BASE_SIZE, features.data(), attrs.data(), labels.data(), valAttrs.data());
    ASSERT_EQ(ret, 0);

    int64_t validNum = 0;
    ASSERT_EQ(tsIndex.GetFeatureNum(&validNum), 0);
    ASSERT_EQ(validNum, BASE_SIZE);

    std::vector<int64_t> indices(overwriteNum);
    std::vector<faiss::ascend::FeatureAttr> attrC(overwriteNum);
    std::vector<faiss::ascend::FeatureAttr> attrB(overwriteNum);
    std::vector<faiss::ascend::ExtraValAttr> valOverwrite(overwriteNum);
    for (int i = 0; i < overwriteNum; ++i) {
        indices[i] = overwriteStart + i;
        attrC[i].time = i;
        attrC[i].tokenId = tokenC;
        attrB[i].time = i;
        attrB[i].tokenId = tokenB;
    }
    for (int round = 0; round < roundNum; ++round) {
        ret = tsIndex.AddFeatureByIndice(
            overwriteNum, features.data(), attrC.data(), indices.data(), valOverwrite.data(), nullptr);
        ASSERT_EQ(ret, 0);
        ret = tsIndex.AddFeatureByIndice(
            overwriteNum, features.data(), attrB.data(), indices.data(), valOverwrite.data(), nullptr);
        ASSERT_EQ(ret, 0);
    }
    // 覆盖写不新增条目
    ASSERT_EQ(tsIndex.GetFeatureNum(&validNum), 0);
    ASSERT_EQ(validNum, BASE_SIZE);

    // token C 只剩位置 99 一个真实成员，删除后应恰好少 1 条
    ret = tsIndex.DeleteFeatureByToken(1, &tokenC);
    ASSERT_EQ(ret, 0);
    ASSERT_EQ(tsIndex.GetFeatureNum(&validNum), 0);
    EXPECT_EQ(validNum, BASE_SIZE - 1);
    const std::vector<uint32_t> tokens{ tokenA, tokenB, tokenC };
    auto counts = CountTokensInBase(tsIndex, validNum, tokens);
    EXPECT_EQ(counts[0], overwriteNum);
    EXPECT_EQ(counts[1], BASE_SIZE - overwriteNum - 1);
    EXPECT_EQ(counts[2], 0);
}

// 整个 token 的位置被覆盖走之后，token 索引中该 token 必须整体消失：
// 位置 0~9 由 B 覆盖成 A 再覆盖回 B 后，A 已无任何真实成员；
// 若残留陈旧条目，DeleteFeatureByToken(A) 会误删 10 条 B 的数据。
TEST(TestAscendIndexTSInt8Flat, DeleteFullyOverwrittenTokenKeepsOtherTokens)
{
    const uint32_t tokenA = 1;
    const uint32_t tokenB = 2;
    const int overwriteNum = 10;
    const int roundNum = 3;

    faiss::ascend::AscendIndexTS tsIndex;
    auto ret =
        tsIndex.InitWithExtraVal(0, DIM, DEFAULT_TOKEN, DEFAULT_MEM, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    ASSERT_EQ(ret, 0);

    std::vector<int8_t> features(BASE_SIZE * DIM);
    ascend::FeatureGenerator(features);
    std::vector<faiss::ascend::ExtraValAttr> valAttrs(BASE_SIZE);
    ascend::ExtraValAttrGenerator(valAttrs);

    std::vector<int64_t> labels(BASE_SIZE);
    std::vector<faiss::ascend::FeatureAttr> attrs(BASE_SIZE);
    for (int64_t i = 0; i < BASE_SIZE; ++i) {
        labels[i] = i;
        attrs[i].time = static_cast<int32_t>(i);
        attrs[i].tokenId = tokenB;
    }
    ret = tsIndex.AddWithExtraVal(BASE_SIZE, features.data(), attrs.data(), labels.data(), valAttrs.data());
    ASSERT_EQ(ret, 0);

    int64_t validNum = 0;
    ASSERT_EQ(tsIndex.GetFeatureNum(&validNum), 0);
    ASSERT_EQ(validNum, BASE_SIZE);

    std::vector<int64_t> indices(overwriteNum);
    std::vector<faiss::ascend::FeatureAttr> attrA(overwriteNum);
    std::vector<faiss::ascend::FeatureAttr> attrB(overwriteNum);
    std::vector<faiss::ascend::ExtraValAttr> valOverwrite(overwriteNum);
    for (int i = 0; i < overwriteNum; ++i) {
        indices[i] = i;
        attrA[i].time = i;
        attrA[i].tokenId = tokenA;
        attrB[i].time = i;
        attrB[i].tokenId = tokenB;
    }
    for (int round = 0; round < roundNum; ++round) {
        ret = tsIndex.AddFeatureByIndice(
            overwriteNum, features.data(), attrA.data(), indices.data(), valOverwrite.data(), nullptr);
        ASSERT_EQ(ret, 0);
        ret = tsIndex.AddFeatureByIndice(
            overwriteNum, features.data(), attrB.data(), indices.data(), valOverwrite.data(), nullptr);
        ASSERT_EQ(ret, 0);
    }
    ASSERT_EQ(tsIndex.GetFeatureNum(&validNum), 0);
    ASSERT_EQ(validNum, BASE_SIZE);

    // token A 已无任何真实成员，删除应报错（tokenId 不存在）且不删除任何数据；
    // 关键是绝不能因为陈旧条目把 B 的数据一起删掉
    ret = tsIndex.DeleteFeatureByToken(1, &tokenA);
    EXPECT_NE(ret, 0);
    ASSERT_EQ(tsIndex.GetFeatureNum(&validNum), 0);
    EXPECT_EQ(validNum, BASE_SIZE);
    const std::vector<uint32_t> tokens{ tokenA, tokenB };
    auto counts = CountTokensInBase(tsIndex, validNum, tokens);
    EXPECT_EQ(counts[0], 0);
    EXPECT_EQ(counts[1], BASE_SIZE);
}

} // namespace
