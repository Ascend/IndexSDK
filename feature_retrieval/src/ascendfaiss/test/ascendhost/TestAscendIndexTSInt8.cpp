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

#include <bitset>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <faiss/ascend/AscendIndexTS.h>
#include <faiss/ascend/AscendIndexInt8Flat.h>
#include <functional>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <queue>
#include <random>
#include <sys/time.h>
#include <unistd.h>
#include <vector>
using namespace std;
namespace {
constexpr int BIT_LENGTH = 8;
constexpr int UINT8_GAP = 128;
using idx_t = int64_t;
using FeatureAttr = faiss::ascend::FeatureAttr;
using AttrFilter = faiss::ascend::AttrFilter;

std::independent_bits_engine<std::mt19937, BIT_LENGTH, uint8_t> engine(1);

void FeatureGenerator(std::vector<int8_t> &features)
{
    size_t n = features.size();
    for (size_t i = 0; i < n; ++i) {
        features[i] = engine() - UINT8_GAP;
    }
}

// 用例均默认为4取余
void FeatureAttrGenerator(std::vector<FeatureAttr> &attrs, int32_t power = 4)
{
    size_t n = attrs.size();
    for (size_t i = 0; i < n; ++i) {
        attrs[i].time = int32_t(i % power);  
        attrs[i].tokenId = int32_t(i % power);
    }
}

inline double GetMillisecs()
{
    struct timeval tv = {0, 0};
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}
} // end of namespace

TEST(TestAscendIndexTS_Int8, Init)
{
    uint32_t deviceId = 0;
    uint32_t dim = 512;
    uint32_t tokenNum = 2500;
    auto ts = GetMillisecs();
    cout << "***** start Init *****" << endl;
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    int res = tsIndex->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_L2_INT8);
    EXPECT_EQ(res, 0);
    auto te = GetMillisecs();
    printf("init cost %f ms\n", te - ts);
    delete tsIndex;
    cout << "***** end Init *****" << endl;
}

TEST(TestAscendIndexTS_Int8, add)
{
    idx_t ntotal = 1000000;
    uint32_t deviceId = 0;
    uint32_t dim = 512;
    uint32_t tokenNum = 2500;

    cout << "***** start Init *****" << endl;
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_L2_INT8);
    EXPECT_EQ(ret, 0);
    cout << "***** end Init *****" << endl;

    std::vector<int8_t> features(ntotal * dim);
    FeatureGenerator(features);
    std::vector<int64_t> labels;
    for (int i = 0; i < ntotal; ++i) {
        labels.push_back(i);
    }
    std::vector<FeatureAttr> attrs(ntotal);
    FeatureAttrGenerator(attrs);
    auto ts = GetMillisecs();

    cout << "***** start add *****" << endl;
    auto res = tsIndex->AddFeature(ntotal, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(res, 0);
    cout << "***** end add *****" << endl;

    auto te = GetMillisecs();
    printf("add %ld cost %f ms\n", ntotal, te - ts);
    int64_t validNum = 0;
    tsIndex->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal);

    delete tsIndex;
}


TEST(TestAscendIndexTS_Int8, GetBaseByRange)
{
    idx_t ntotal = 1000000;
    uint32_t addNum = 1;
    uint32_t deviceId = 0;
    uint32_t dim = 256;
    uint32_t tokenNum = 2500;

    cout << "***** start Init *****" << endl;
    faiss::ascend::AscendIndexTS *tsIndex0 = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex0->Init(0, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(ret, 0);
    cout << "***** start Init  1*****" << endl;
    faiss::ascend::AscendIndexTS *tsIndex1 = new faiss::ascend::AscendIndexTS();
    ret = tsIndex1->Init(1, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_L2_INT8);
    EXPECT_EQ(ret, 0);
    cout << "***** end Init *****" << endl;

    cout << "***** start add *****" << endl;
    std::vector<int8_t> features(ntotal * dim);
    FeatureGenerator(features);
    auto ts1 = GetMillisecs();
    for (size_t i = 0; i < addNum; i++) {
        std::vector<int64_t> labels;
        for (int64_t j = 0; j < ntotal; ++j) {
            labels.emplace_back(j + i * ntotal);
        }
        std::vector<FeatureAttr> attrs(ntotal);
        FeatureAttrGenerator(attrs);
        auto t1 = GetMillisecs();
        auto res = tsIndex0->AddFeature(ntotal, features.data(), attrs.data(), labels.data());
        EXPECT_EQ(res, 0);
        auto t2 = GetMillisecs();
        printf("add %ld cost %f ms\n", ntotal * i, t2 - t1);
    }
    auto ts2 = GetMillisecs();

    int64_t validNum = 0;
    tsIndex0->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal * addNum);

    int needAddNum = validNum / 2;
    std::vector<int8_t> retBase(needAddNum * dim);
    std::vector<int64_t> retLabels(needAddNum);
    std::vector<FeatureAttr> retAttrs(needAddNum);
    ret = tsIndex0->GetBaseByRange(needAddNum, needAddNum, retLabels.data(), retBase.data(), retAttrs.data());
    EXPECT_EQ(ret, 0);
    auto ts3 = GetMillisecs();

    ret = tsIndex1->AddFeature(needAddNum, retBase.data(), retAttrs.data(), retLabels.data());
    EXPECT_EQ(ret, 0);
    auto ts4 = GetMillisecs();
    ret = tsIndex0->DeleteFeatureByLabel(needAddNum, retLabels.data());
    EXPECT_EQ(ret, 0);
    auto ts5 = GetMillisecs();
    printf("GetBaseByRange %ld all cost [%f] ms, index0 add [%f] ms, index0 GetBaseByRange [%f] ms, index1 add [%f] "
        "ms, index0 del [%f] ms\n",
        ntotal * addNum, ts5 - ts1, ts2 - ts1, ts3 - ts2, ts4 - ts3, ts5 - ts4);
#pragma omp parallel for if (ntotal > 100)
    for (int i = 0; i < needAddNum * dim; i++) {
        EXPECT_EQ(features[i + needAddNum * dim], retBase[i]);
        if (features[i + needAddNum * dim]!= retBase[i]) {
            printf("%d--%d \n", features[i + needAddNum * dim], retBase[i]);
        }     
    }
    delete tsIndex0;
    delete tsIndex1;
}


TEST(TestAscendIndexTS_Int8, GetFeatureByLabel)
{
    int dim = 512;
    int maxTokenId = 2500;
    int ntotal = 100000;
    std::vector<int8_t> base(ntotal * dim);
    FeatureGenerator(base);
    std::vector<int64_t> label(ntotal);
    std::iota(label.begin(), label.end(), 0);
    std::vector<FeatureAttr> attrs(ntotal);
    FeatureAttrGenerator(attrs);

    cout << "***** start Init *****" << endl;
    auto *index = new faiss::ascend::AscendIndexTS();
    auto ret = index->Init(0, dim, maxTokenId, faiss::ascend::AlgorithmType::FLAT_L2_INT8);
    EXPECT_EQ(ret, 0);
    cout << "***** end Init *****" << endl;

    cout << "***** start add *****" << endl;
    ret = index->AddFeature(ntotal, base.data(), attrs.data(), label.data());
    EXPECT_EQ(ret, 0);
    cout << "***** end add *****" << endl;

    cout << "***** start GetFeatureByLabel *****" << endl;
    std::vector<int8_t> getBase(ntotal * dim);
    auto ts = GetMillisecs();
    ret = index->GetFeatureByLabel(ntotal, label.data(), getBase.data());
    auto te = GetMillisecs();
    printf("GetFeatureByLabel  cost  total %f ms \n", te - ts);
    EXPECT_EQ(ret, 0);

#pragma omp parallel for if (ntotal > 100)
    for (int i = 0; i < ntotal * dim; i++) {
        EXPECT_EQ(base[i], getBase[i]);
    }
    cout << "***** end GetFeatureByLabel *****" << endl;
}

TEST(TestAscendIndexTS_Int8, DeleteFeatureByLabel)
{
    int dim = 512;
    int maxTokenId = 2500;
    int ntotal = 10000;
    std::vector<int8_t> base(ntotal * dim);
    FeatureGenerator(base);
    std::vector<int64_t> label(ntotal);
    std::iota(label.begin(), label.end(), 0);
    std::vector<FeatureAttr> attrs(ntotal);
    FeatureAttrGenerator(attrs);

    cout << "***** start Init *****" << endl;
    auto *index = new faiss::ascend::AscendIndexTS();
    auto ret = index->Init(0, dim, maxTokenId, faiss::ascend::AlgorithmType::FLAT_L2_INT8);
    EXPECT_EQ(ret, 0);
    cout << "***** end Init *****" << endl;

    cout << "***** start add *****" << endl;
    ret = index->AddFeature(ntotal, base.data(), attrs.data(), label.data());
    EXPECT_EQ(ret, 0);
    cout << "***** end add *****" << endl;

    int64_t validNum = 0;
    index->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal);
    int delCount = 1000;
    std::vector<int64_t> delLabel(delCount);
    delLabel.assign(label.begin(), label.begin() + delCount);

    cout << "***** start DeleteFeatureByLabel *****" << endl;
    auto ts = GetMillisecs();
    index->DeleteFeatureByLabel(delCount, delLabel.data());
    auto te = GetMillisecs();
    printf("DeleteFeatureByLabel delete cost  total %f ms \n", te - ts);
    cout << "***** end DeleteFeatureByLabel *****" << endl;

    index->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal - delCount);
    // delete again
    index->DeleteFeatureByLabel(delCount, delLabel.data());
    index->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal - delCount);
    cout << "***** end DeleteFeatureByLabel again*****" << endl;
}

TEST(TestAscendIndexTS_Int8, DeleteFeatureByToken)
{
    int dim = 512;
    int maxTokenId = 2500;
    int ntotal = 10000;
    std::vector<int8_t> base(ntotal * dim);
    FeatureGenerator(base);
    std::vector<int64_t> label(ntotal);
    std::iota(label.begin(), label.end(), 0);
    std::vector<FeatureAttr> attrs(ntotal);
    FeatureAttrGenerator(attrs);

    cout << "***** start Init *****" << endl;
    auto *index = new faiss::ascend::AscendIndexTS();
    auto ret = index->Init(0, dim, maxTokenId, faiss::ascend::AlgorithmType::FLAT_L2_INT8);
    EXPECT_EQ(ret, 0);
    cout << "***** end Init *****" << endl;

    cout << "***** start add *****" << endl;
    ret = index->AddFeature(ntotal, base.data(), attrs.data(), label.data());
    EXPECT_EQ(ret, 0);
    cout << "***** end add *****" << endl;

    int64_t validNum = 0;
    index->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal);
    std::vector<uint32_t> delToken {0, 1};

    cout << "***** start DeleteFeatureByToken *****" << endl;
    auto ts = GetMillisecs();
    index->DeleteFeatureByToken(2, delToken.data());
    auto te = GetMillisecs();
    cout << "***** end DeleteFeatureByToken *****" << endl;

    printf("DeleteFeatureByToken delete cost  total %f ms \n", te - ts);
    index->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal / 2);
}

TEST(TestAscendIndexTS_Int8, SearchNoShareQPS)
{
    idx_t ntotal = 1000000;
    uint32_t addNum = 1;
    uint32_t deviceId = 0;
    uint32_t dim = 256;
    uint32_t tokenNum = 2500;
    std::vector<int> queryNums = {1, 2, 4, 8, 16, 32, 64, 128, 256}; 
    std::vector<int> topks = {100};
    
    cout << "***** start Init *****" << endl;
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_L2_INT8);
    EXPECT_EQ(ret, 0);
    cout << "***** end Init *****" << endl;

    cout << "***** start add *****" << endl;
    std::vector<int8_t> features(ntotal * dim);
    FeatureGenerator(features);
    for (size_t i = 0; i < addNum; i++) {
        std::vector<int64_t> labels;
        for (int64_t j = 0; j < ntotal; ++j) {
            labels.emplace_back(j + i * ntotal);
        }
        std::vector<FeatureAttr> attrs(ntotal);
        FeatureAttrGenerator(attrs);
        auto res = tsIndex->AddFeature(ntotal, features.data(), attrs.data(), labels.data());
        EXPECT_EQ(res, 0);
    }
    cout << "***** end add *****" << endl;

    cout << "***** start SearchNoShareQPS *****" << endl;
    long double ts {0.};
    long double te {0.};
    for (auto k : topks) {
        int warmupTimes = 3;
        int loopTimes = 2;
        for (auto queryNum : queryNums) {
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

            AttrFilter filter {};
            filter.timesStart = 0;
            filter.timesEnd = 3;
            filter.tokenBitSet = bitSet.data();
            filter.tokenBitSetLen = setlen;

            std::vector<AttrFilter> queryFilters(queryNum, filter);
            for (int i = 0; i < loopTimes + warmupTimes; i++) {
                if (i == warmupTimes) {
                    ts = GetMillisecs();
                }
                tsIndex->Search(queryNum, querys.data(), queryFilters.data(), false, k, labelRes.data(),
                                distances.data(), validnum.data());
            }
            te = GetMillisecs();

            printf("base: %ld,  dim: %d,  batch: %4d,  top%d,  QPS:%7.2Lf\n", ntotal * addNum, dim, queryNum, k,
                   (long double)1000.0 * queryNum * loopTimes / (te - ts));
        }
    }
    cout << "***** end SearchNoShareQPS *****" << endl;
    delete tsIndex;
}

TEST(TestAscendIndexTS_Int8, SearchShareQPS)
{
    idx_t ntotal = 1000000;
    uint32_t addNum = 30;
    uint32_t deviceId = 0;
    uint32_t dim = 512;
    uint32_t tokenNum = 2500;
    std::vector<int> queryNums = {1, 2, 4, 8, 16, 32, 64, 128}; // make sure the vector is in ascending order
    std::vector<int> topks = {10};

    cout << "***** start Init *****" << endl;
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_L2_INT8);
    EXPECT_EQ(ret, 0);
    cout << "***** end Init *****" << endl;

    cout << "***** start add *****" << endl;
    std::vector<int8_t> features(ntotal * dim);
    FeatureGenerator(features);
    for (size_t i = 0; i < addNum; i++) {
        std::vector<int64_t> labels;
        for (int64_t j = 0; j < ntotal; ++j) {
            labels.emplace_back(j + i * ntotal);
        }
        std::vector<FeatureAttr> attrs(ntotal);
        FeatureAttrGenerator(attrs);

        auto res = tsIndex->AddFeature(ntotal, features.data(), attrs.data(), labels.data());
        EXPECT_EQ(res, 0);
    }
    cout << "***** end add *****" << endl;

    cout << "***** start SearchShareQPS *****" << endl;
    long double ts {0.};
    long double te {0.};
    for (auto k : topks) {
        int warmupTimes = 3;
        int loopTimes = 2;
        for (auto queryNum : queryNums) {
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

            AttrFilter filter {};
            filter.timesStart = 0;
            filter.timesEnd = 3;
            filter.tokenBitSet = bitSet.data();
            filter.tokenBitSetLen = setlen;

            std::vector<AttrFilter> queryFilters(queryNum, filter);
            for (int i = 0; i < loopTimes + warmupTimes; i++) {
                if (i == warmupTimes) {
                    ts = GetMillisecs();
                }
                tsIndex->Search(queryNum, querys.data(), queryFilters.data(), true, k, labelRes.data(),
                                distances.data(), validnum.data());
            }
            te = GetMillisecs();

            printf("base: %ld,  dim: %d,  batch: %4d,  top%d,  QPS:%7.2Lf\n", ntotal * addNum, dim, queryNum, k,
                   (long double)1000.0 * queryNum * loopTimes / (te - ts));
        }
    }
    cout << "***** end SearchShareQPS *****" << endl;
    delete tsIndex;
}

TEST(TestAscendIndexTS_Int8, SearchShareWithExtraQPS)
{
    idx_t ntotal = 1000000;
    uint32_t addNum = 1;
    uint32_t deviceId = 0;
    uint32_t dim = 512;
    uint32_t tokenNum = 2500;
    std::vector<int> queryNums = {1, 2, 4, 8, 16, 32, 64, 128, 256}; // make sure the vector is in ascending order
    std::vector<int> topks = {100};

    cout << "***** start Init *****" << endl;
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_L2_INT8);
    EXPECT_EQ(ret, 0);
    cout << "***** end Init *****" << endl;

    cout << "***** start add *****" << endl;
    std::vector<int8_t> features(ntotal * dim);
    FeatureGenerator(features);
    for (uint32_t i = 0; i < addNum; i++) {
        std::vector<int64_t> labels;
        for (int64_t j = 0; j < ntotal; ++j) {
            labels.emplace_back(j + i * ntotal);
        }
        std::vector<FeatureAttr> attrs(ntotal);
        FeatureAttrGenerator(attrs);
        auto ts0 = GetMillisecs();
        auto res = tsIndex->AddFeature(ntotal, features.data(), attrs.data(), labels.data());
        EXPECT_EQ(res, 0);
        auto te0 = GetMillisecs();
        printf("add %ld cost %f ms\n", ntotal, te0 - ts0);
    }
    cout << "***** end add *****" << endl;
    
    cout << "***** start SearchShareWithExtraQPS *****" << endl;
    long double ts {0.};
    long double te {0.};
    for (auto k : topks) {
        int warmupTimes = 3;
        int loopTimes = 10;
        for (auto queryNum : queryNums) {
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

            AttrFilter filter {};
            filter.timesStart = 0;
            filter.timesEnd = 3;
            filter.tokenBitSet = bitSet.data();
            filter.tokenBitSetLen = setlen;

            int extraMaskLen = 12;
            std::vector<uint8_t> extraMask(queryNum * extraMaskLen, 0);
            int ind = 1;
            for (int i = 0; i < queryNum; i++) {
                for (int j = 0; j < extraMaskLen; j++) {
                    extraMask[i * extraMaskLen + j] = 0x1 << (ind % 8);
                    ind++;
                }
            }
            std::vector<AttrFilter> queryFilters(queryNum, filter);
            for (int i = 0; i < warmupTimes; i++) {
                tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), true, k, extraMask.data(),
                                             extraMaskLen, false, labelRes.data(), distances.data(), validnum.data());
            }
            ts = GetMillisecs();
            for (int i = 0; i < loopTimes; i++) {
                tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), true, k, extraMask.data(),
                                             extraMaskLen, false, labelRes.data(), distances.data(), validnum.data());
            }
            te = GetMillisecs();
            printf("base: %ld,  dim: %d,  batch: %4d,  top%d,  QPS:%7.2Lf\n", ntotal * addNum, dim, queryNum, k,
                   (long double)1000.0 * queryNum * loopTimes / (te - ts));
        }
    }
    cout << "***** end SearchShareWithExtraQPS *****" << endl;
    delete tsIndex;
}

TEST(TestAscendIndexTS_Int8L2, Precision)
{
    idx_t ntotal = 1000000;
    uint32_t addNum = 1;
    std::vector<int> deviceId { 0 };
    uint32_t device = 0;
    uint32_t dim = 512;
    std::vector<int> queryNums = { 5 }; // make sure the vector is in ascending order
    std::vector<int> topks = { 10 };
    int64_t resourceSize = 1 * static_cast<int64_t>(1024 * 1024 * 1024);
    uint32_t blockSize = 16384;
    cout << "***** start create Int8Flat index *****" << endl;
    faiss::ascend::AscendIndexInt8FlatConfig conf(deviceId, resourceSize, blockSize);
    faiss::ascend::AscendIndexInt8Flat index(dim, faiss::METRIC_L2, conf);
    index.verbose = true;
    cout << "***** end create Int8Flat index *****" << endl;

    uint32_t tokenNum = 2500;
    cout << "***** start create ts index *****" << endl;
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex->Init(device, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_L2_INT8);
    EXPECT_EQ(ret, 0);
    cout << "***** end create ts index *****" << endl;

    cout << "***** start add *****" << endl;
    std::vector<int8_t> features(ntotal * dim);
    FeatureGenerator(features);
    
    for (size_t i = 0; i < addNum; i++) {
        std::vector<int64_t> labels;

        for (int64_t j = 0; j < ntotal; ++j) {
            labels.emplace_back(j + i * ntotal);
        }
        std::vector<FeatureAttr> attrs(ntotal);
        FeatureAttrGenerator(attrs);
        index.add(ntotal, features.data());
        ret = tsIndex->AddFeature(ntotal, features.data(), attrs.data(), labels.data());
        ASSERT_EQ(ret, 0);
    }
    cout << "***** end add *****" << endl;

    cout << "***** start search *****" << endl;
    for (auto k : topks) {
        int warmupTimes = 0;
        int loopTimes = 1;
        for (auto queryNum : queryNums) {
            std::vector<float> distances1(queryNum * k, -1);
            std::vector<int64_t> labelRes1(queryNum * k, 10);
            uint32_t size = queryNum * dim;
            std::vector<int8_t> querys1(size);
            querys1.assign(features.begin(), features.begin() + size);

            cout << "***** Int8L2 search *****" << endl;
            for (int i = 0; i < loopTimes + warmupTimes; i++) {
                index.search(queryNum, querys1.data(), k, distances1.data(), labelRes1.data());
            }
            cout << "***** Int8L2 search end*****" << endl;

            uint32_t setlen = (uint32_t)((tokenNum + 7) / 8);
            std::vector<uint8_t> bitSet(setlen, 0);
            // 11111111 不过滤和int8L2的结果比较
            bitSet[0] = 0b11111111;
            AttrFilter filter {};
            filter.timesStart = 0;
            filter.timesEnd = 3;
            filter.tokenBitSet = bitSet.data();
            filter.tokenBitSetLen = setlen;

            std::vector<AttrFilter> queryFilters(queryNum, filter);

            std::vector<float> distances2(queryNum * k, -1);
            std::vector<int64_t> labelRes2(queryNum * k, 10);
            std::vector<uint32_t> validnum2(queryNum, 0);

            cout << "***** TSInt8L2 search *****" << endl;
            std::vector<uint8_t> querys2(size);
            querys2.assign(features.begin(), features.begin() + size);
            for (int i = 0; i < loopTimes + warmupTimes; i++) {
                tsIndex->Search(queryNum, querys1.data(), queryFilters.data(), false, k, labelRes2.data(),
                                distances2.data(), validnum2.data());
            }
            cout << "***** TSInt8L2 search end*****" << endl;
            cout << "***** end search *****" << endl;
            for (int i = 0; i < queryNum; i++) {
                for (int j = 0; j < k; j++) {
                    EXPECT_TRUE(labelRes1[i * k + j] == labelRes2[i * k + j]);
                    EXPECT_TRUE(distances1[i * k + j] == distances2[i * k + j]);
                }
            }
        }
    }
    cout << "***** check Precision success *****" << endl;
    delete tsIndex;
}

TEST(TestAscendIndexTS_Int8L2, PrecisionWithMask)
{
    idx_t ntotal = 1000000;
    uint32_t dim = 512;
    uint32_t tokenNum = 2500;
    int queryNums = 20;
    int topks = 10;
    cout << "***** start create index *****" << endl;
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex->Init(0, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_L2_INT8);
    ASSERT_EQ(ret, 0);
    cout << "***** end create index *****" << endl;

    cout << "***** start add *****" << endl;
    std::vector<int8_t> features(ntotal * dim);
    FeatureGenerator(features);

    std::vector<int64_t> labels(ntotal);
    for (size_t i = 0; i < labels.size(); i++) {
        labels[i] = i;
    }

    std::vector<FeatureAttr> attrs(ntotal);
    int power = 10;
    FeatureAttrGenerator(attrs, power);
    ret = tsIndex->AddFeature(ntotal, features.data(), attrs.data(), labels.data());
    ASSERT_EQ(ret, 0);

    cout << "***** end add *****" << endl;
    uint32_t size = queryNums * dim;
    std::vector<uint8_t> querys(size);
    querys.assign(features.begin(), features.begin() + size);
    cout << "***** start search *****" << endl;
    uint32_t setlen = static_cast<uint32_t>((tokenNum + 7) / 8);
    vector<uint8_t> bitSet(setlen, 0xff);
    AttrFilter filter;
    filter.timesStart = 0;
    filter.timesEnd = 9;
    filter.tokenBitSet = bitSet.data();
    filter.tokenBitSetLen = bitSet.size();

    vector<AttrFilter> queryFilters(queryNums, filter);
    vector<float> distances(queryNums * topks, -1);
    vector<int64_t> labelRes(queryNums * topks, -1);
    vector<uint32_t> validnum(queryNums, 1);
    tsIndex->Search(queryNums, querys.data(), queryFilters.data(), false, topks, labelRes.data(),
                                distances.data(), validnum.data());
    cout << "***** end search *****" << endl;

    cout << "***** start search with mask *****" << endl;
    // 前面生成的tokenId 是0/1/2/3.../9, 这里过滤成1/2/3..../9，第一个bit是11111110
    vector<uint8_t> tokenBitSet(setlen, 0xff);
    tokenBitSet[0] = 0b11111110;
    cout << "first bit[" << to_string(tokenBitSet[0]) << "] should be 254" << endl;
    filter.timesStart = 0;
    filter.timesEnd = 8;
    filter.tokenBitSet = tokenBitSet.data();
    filter.tokenBitSetLen = tokenBitSet.size();
    vector<AttrFilter> queryFiltersMask(queryNums, filter);
    vector<float> distancesMask(queryNums * topks, -1);
    vector<int64_t> labelResMask(queryNums * topks, -1);
    vector<uint32_t> validnumMask(queryNums, 1);

    tsIndex->Search(queryNums, querys.data(), queryFiltersMask.data(), false, topks, labelResMask.data(),
        distancesMask.data(), validnumMask.data());
    size_t validNumAll = 0;
    for (int iq = 0; iq < queryNums; iq++) {
        for (size_t ik = 0; ik < static_cast<size_t>(topks); ik++) {
            int64_t curGtLabel = labelRes[iq * topks + ik];
            int64_t idTemp = curGtLabel % power;
            // 0是因为被时间过滤，9是被token过滤
            if ((idTemp == 0) || (idTemp == 9)) {
                continue;
            }
            validNumAll++;
            auto it = find(labelResMask.begin() + iq * topks, labelResMask.begin() + (iq + 1) * topks, curGtLabel);

            if (it == labelResMask.begin() + (iq + 1) * topks) {
                if (distances[iq * topks + ik] == distancesMask[iq * topks + ik]) continue;
                printf("Error! Label[%d][%zu] = %ld not found!\r\n", iq, ik, curGtLabel);
                ASSERT_TRUE(false);
                return;
            }
        }
    }
    printf("check result success, validNumAll[%zu], acturalNum[%zu], filterPower[%lf]\r\n",
        validNumAll, labelRes.size(), static_cast<float>(validNumAll)/labelRes.size());
    cout << "***** end search with mask *****" << endl;
}

TEST(TestAscendIndexTS_Int8Cos, Acc)
{
    idx_t ntotal = 1000000;
    uint32_t addNum = 1;
    uint32_t deviceId = 0;
    uint32_t dim = 512;
    uint32_t tokenNum = 2500;
    std::vector<int> queryNums = {5}; // make sure the vector is in ascending order
    std::vector<int> topks = {10};
    
    cout << "***** start create index *****" << endl;
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(ret, 0);
    cout << "***** end create index *****" << endl;

    cout << "***** start add *****" << endl;
    std::vector<int8_t> features(ntotal * dim);
    FeatureGenerator(features);
    for (size_t i = 0; i < addNum; i++) {
        std::vector<int64_t> labels;

        for (int64_t j = 0; j < ntotal; ++j) {
            labels.emplace_back(j + i * ntotal);
        }
        std::vector<FeatureAttr> attrs(ntotal);
        FeatureAttrGenerator(attrs);
        auto ts0 = GetMillisecs();
        tsIndex->AddFeature(ntotal, features.data(), attrs.data(), labels.data());
        auto te0 = GetMillisecs();
        printf("add %ld cost %f ms\n", ntotal, te0 - ts0);
    }
    cout << "***** end add *****" << endl;

    for (auto k : topks) {
        int warmupTimes = 0;
        int loopTimes = 1;
        for (auto queryNum : queryNums) {
            std::vector<float> distances(queryNum * k, -1);
            std::vector<int64_t> labelRes(queryNum * k, 10);
            std::vector<uint32_t> validnum(queryNum, 0);
            uint32_t size = queryNum * dim;
            std::vector<uint8_t> querys(size);
            querys.assign(features.begin(), features.begin() + size);

            uint32_t setlen = (uint32_t)((tokenNum + 7) / 8);
            std::vector<uint8_t> bitSet(setlen, 0);

            // 00001111
            bitSet[0] = 0x1 << 0 | 0x1 << 1 | 0x1 << 2 | 0x1 << 3;
            AttrFilter filter {};
            filter.timesStart = 0;
            filter.timesEnd = 3;
            filter.tokenBitSet = bitSet.data();
            filter.tokenBitSetLen = setlen;

            std::vector<AttrFilter> queryFilters(queryNum, filter);
            for (int i = 0; i < loopTimes + warmupTimes; i++) {

                tsIndex->Search(queryNum, querys.data(), queryFilters.data(), false, k, labelRes.data(),
                                distances.data(), validnum.data());
            }
            for (size_t i = 0; i < queryNum; i++) {
                ASSERT_TRUE(labelRes[i * k] == i);
                ASSERT_TRUE(distances[i * k] > float(0.99) && distances[i * k] < float(1.01));
            }
            // 00000011   -> 0,1
            bitSet[0] = 0x1 << 0 | 0x1 << 1;

            filter.timesStart = 1;
            filter.timesEnd = 3;

            queryFilters.clear();
            queryFilters.insert(queryFilters.begin(), queryNum, filter);
            labelRes.clear();
            distances.clear();
            for (int i = 0; i < loopTimes + warmupTimes; i++) {
                tsIndex->Search(queryNum, querys.data(), queryFilters.data(), false, k, labelRes.data(),
                                distances.data(), validnum.data());
            }
            for (size_t i = 0; i < queryNum; i++) {
                if (i % 4 == 1) {
                    ASSERT_TRUE(labelRes[i * k] == i);
                    ASSERT_TRUE(distances[i * k] > float(0.99) && distances[i * k] < float(1.01));
                } else {
                    ASSERT_TRUE(labelRes[i * k] != i);
                    ASSERT_TRUE(distances[i * k] <= float(0.3));
                }
            }
        }
    }
    cout << "***** search success *****" << endl;
    delete tsIndex;
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
