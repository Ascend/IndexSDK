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
#include <functional>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <queue>
#include <random>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

namespace {
using idx_t = int64_t;
using FeatureAttr = faiss::ascend::FeatureAttr;
using AttrFilter = faiss::ascend::AttrFilter;

std::independent_bits_engine<std::mt19937, 8, uint8_t> engine(1);

void FeatureGenerator(std::vector<uint8_t> &features)
{
    size_t n = features.size();
    for (size_t i = 0; i < n; ++i) {
        features[i] = engine();
    }
}

void FeatureAttrGenerator(std::vector<FeatureAttr> &attrs)
{
    size_t n = attrs.size();
    for (size_t i = 0; i < n; ++i) {
        attrs[i].time = int32_t(i % 4);
        attrs[i].tokenId = int32_t(i % 4);
    }
}

inline double GetMillisecs()
{
    struct timeval tv = { 0, 0 };
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}
} // end of namespace

TEST(TestAscendIndexTS, Init)
{
    uint32_t deviceId = 0;
    uint32_t dim = 512;
    uint32_t tokenNum = 2500;
    auto ts = GetMillisecs();
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    int res = tsIndex->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_HAMMING);
    EXPECT_EQ(res, 0);
    auto te = GetMillisecs();
    printf("init cost %f ms\n", te - ts);
    delete tsIndex;
}

TEST(TestAscendIndexTS, add)
{
    idx_t ntotal = 1000000;
    uint32_t deviceId = 0;
    uint32_t dim = 512;
    uint32_t tokenNum = 2500;
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    tsIndex->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_HAMMING);

    std::vector<uint8_t> features(ntotal * dim / 8);
    printf("[---add -----------]\n");
    FeatureGenerator(features);
    std::vector<int64_t> labels;
    for (int i = 0; i < ntotal; ++i) {
        labels.push_back(i);
    }
    std::vector<FeatureAttr> attrs(ntotal);
    FeatureAttrGenerator(attrs);
    auto ts = GetMillisecs();
    auto res = tsIndex->AddFeature(ntotal, features.data(), attrs.data(), labels.data());
    EXPECT_EQ(res, 0);
    auto te = GetMillisecs();
    printf("add %ld cost %f ms\n", ntotal, te - ts);

    delete tsIndex;
}

TEST(TestAscendIndexTS, GetBaseByRange)
{
    idx_t ntotal = 1000;
    uint32_t addNum = 1;
    uint32_t deviceId = 0;
    uint32_t dim = 256;
    uint32_t tokenNum = 2500;

    using namespace std;
    cout << "***** start Init *****" << endl;
    faiss::ascend::AscendIndexTS *tsIndex0 = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex0->Init(0, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_HAMMING);
    EXPECT_EQ(ret, 0);
    cout << "***** start Init  1*****" << endl;
    faiss::ascend::AscendIndexTS *tsIndex1 = new faiss::ascend::AscendIndexTS();
    ret = tsIndex1->Init(0, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_HAMMING);
    EXPECT_EQ(ret, 0);
    cout << "***** end Init *****" << endl;

    cout << "***** start add *****" << endl;
    std::vector<uint8_t> features(ntotal * dim / 8);
    FeatureGenerator(features);
    auto ts1 = GetMillisecs();
    for (size_t i = 0; i < addNum; i++) {
        std::vector<int64_t> labels;
        for (int64_t j = 0; j < ntotal; ++j) {
            labels.emplace_back(j + i * ntotal);
        }
        std::vector<FeatureAttr> attrs(ntotal);
        FeatureAttrGenerator(attrs);
        auto res = tsIndex0->AddFeature(ntotal, features.data(), attrs.data(), labels.data());
        EXPECT_EQ(res, 0);
    }
    auto ts2 = GetMillisecs();

    int64_t validNum = 0;
    tsIndex0->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal * addNum);

    int needAddNum = validNum / 2;
    std::vector<uint8_t> retBase(needAddNum * dim / 8);
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
    for (int i = 0; i < needAddNum * dim / 8; i++) {
        EXPECT_EQ(features[i + needAddNum * dim / 8], retBase[i]);
        if (features[i + needAddNum * dim / 8]!= retBase[i]) {
            printf("%u--%u \n", features[i + needAddNum * dim/ 8], retBase[i]);
        }
    }
    delete tsIndex0;
    delete tsIndex1;
}

TEST(TestAscendIndexTS, GetFeatureByLabel)
{
    int dim = 512;
    int maxTokenId = 2500;
    int ntotal = 100000;
    std::vector<uint8_t> base(ntotal * dim / 8);
    FeatureGenerator(base);
    std::vector<int64_t> label(ntotal);
    std::iota(label.begin(), label.end(), 0);
    std::vector<FeatureAttr> attrs(ntotal);
    FeatureAttrGenerator(attrs);
    auto *index = new faiss::ascend::AscendIndexTS();
    auto ret = index->Init(0, dim, maxTokenId, faiss::ascend::AlgorithmType::FLAT_HAMMING);
    EXPECT_EQ(ret, 0);
    ret = index->AddFeature(ntotal, base.data(), attrs.data(), label.data());
    EXPECT_EQ(ret, 0);
    std::vector<uint8_t> getBase(ntotal * dim);
    auto ts = GetMillisecs();
    ret = index->GetFeatureByLabel(ntotal, label.data(), getBase.data());
    auto te = GetMillisecs();
    printf("GetFeatureByLabel  cost  total %f ms \n" ,te-ts);
    EXPECT_EQ(ret, 0);

#pragma omp parallel for if (ntotal > 100)
    for (int i = 0; i < ntotal * dim / 8; i++) {
        EXPECT_EQ(base[i], getBase[i]);
    }
}

TEST(TestAscendIndexTS, GetFeatureAttrByLabel)
{
    int dim = 512;
    int maxTokenId = 2500;
    size_t ntotal = 1000000;
    int addNum = 10;
    std::vector<uint8_t> base(ntotal * addNum * dim / 8);
    FeatureGenerator(base);
    std::vector<int64_t> labels(ntotal * addNum);
    std::iota(labels.begin(), labels.end(), 0);
    std::vector<FeatureAttr> attrs(ntotal * addNum);
    FeatureAttrGenerator(attrs);
    auto *index = new faiss::ascend::AscendIndexTS();
    auto ret = index->Init(0, dim, maxTokenId, faiss::ascend::AlgorithmType::FLAT_HAMMING);
    EXPECT_EQ(ret, 0);
    for (size_t i = 0; i < addNum; i++) {
        auto ts0 = GetMillisecs();
        ret = index->AddFeature(ntotal, base.data() + i * ntotal * dim / 8, attrs.data() + i * ntotal,
            labels.data() + i * ntotal);
        EXPECT_EQ(ret, 0);
        auto te0 = GetMillisecs();
        printf("add %ld cost %f ms\n", ntotal, te0 - ts0);
    }
    int getAttrCnt = 1000;
    std::vector<FeatureAttr> getAttr(getAttrCnt);
    printf("GetFeatureAttrByLabel  start\n");
    auto ts = GetMillisecs();
    ret = index->GetFeatureAttrByLabel(getAttrCnt, labels.data(), getAttr.data());
    EXPECT_EQ(ret, 0);
    auto te = GetMillisecs();
    printf("GetFeatureAttrByLabel[%d]  cost  total %f ms \n", getAttrCnt, te - ts);
    for (int i = 0; i < getAttr.size(); i++) {
        EXPECT_EQ(attrs[i].time, getAttr[i].time);
        EXPECT_EQ(attrs[i].tokenId, getAttr[i].tokenId);
    }

    std::vector<uint8_t> oneBase(1 * dim / 8);
    FeatureGenerator(oneBase);
    std::vector<int64_t> oneLabels(1, 555555555);
    std::vector<FeatureAttr> oneAttrs(1);
    oneAttrs[0].time = 10;
    oneAttrs[0].tokenId = 156;
    ret = index->AddFeature(1, oneBase.data(), oneAttrs.data(), oneLabels.data());
    EXPECT_EQ(ret, 0);

    std::vector<FeatureAttr> getOneAttr(1);
    ret = index->GetFeatureAttrByLabel(1, oneLabels.data(), getOneAttr.data());
    EXPECT_EQ(ret, 0);
    for (int i = 0; i < getOneAttr.size(); i++) {
        EXPECT_EQ(oneAttrs[i].time, getOneAttr[i].time);
        EXPECT_EQ(oneAttrs[i].tokenId, getOneAttr[i].tokenId);
    }
}

TEST(TestAscendIndexTS, DeleteFeatureByLabel)
{
    int dim = 512;
    int maxTokenId = 2500;
    int ntotal = 1000000;
    std::vector<uint8_t> base(ntotal * dim / 8);
    FeatureGenerator(base);
    std::vector<int64_t> label(ntotal);
    std::iota(label.begin(), label.end(), 0);
    std::vector<FeatureAttr> attrs(ntotal);
    FeatureAttrGenerator(attrs);
    auto *index = new faiss::ascend::AscendIndexTS();
    auto ret = index->Init(0, dim, maxTokenId, faiss::ascend::AlgorithmType::FLAT_HAMMING);
    EXPECT_EQ(ret, 0);
    ret = index->AddFeature(ntotal, base.data(), attrs.data(), label.data());
    EXPECT_EQ(ret, 0);
    int64_t validNum = 0;
    index->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal);
    int delCount = 10000;
    std::vector<int64_t> delLabel(delCount);

    // 将前 100000 个偶数存储在向量中      
    delLabel.assign(label.begin(), label.begin() + delCount);
    auto ts = GetMillisecs();
    ret = index->DeleteFeatureByLabel(delCount, delLabel.data());
    EXPECT_EQ(ret, 0);
    auto te = GetMillisecs();
    printf("DeleteFeatureByLabel delete [%d] cost  total %f ms \n" ,delCount, te-ts);
    
    index->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal - delCount);

    std::vector<uint8_t> getBase(delCount*dim / 8);
    ts = GetMillisecs();
    ret = index->GetFeatureByLabel(delCount, label.data() + ntotal-delCount, getBase.data());
    te = GetMillisecs();
    printf("GetFeatureByLabel  cost  total %f ms \n" ,te-ts);
    EXPECT_EQ(ret, 0);

#pragma omp parallel for if (ntotal > 100)
    for (int i = 0; i < delCount * dim / 8; i++) {
        EXPECT_EQ(base[i+(ntotal-delCount) * dim / 8], getBase[i]);
    }
}

TEST(TestAscendIndexTS, DeleteFeatureByToken)
{
    int dim = 512;
    int maxTokenId = 2500;
    int ntotal = 1000000;
    std::vector<uint8_t> base(ntotal * dim / 8);
    FeatureGenerator(base);
    std::vector<int64_t> label(ntotal);
    std::iota(label.begin(), label.end(), 0);
    std::vector<FeatureAttr> attrs(ntotal);
    FeatureAttrGenerator(attrs);
    auto *index = new faiss::ascend::AscendIndexTS();
    auto ret = index->Init(0, dim, maxTokenId, faiss::ascend::AlgorithmType::FLAT_HAMMING);
    EXPECT_EQ(ret, 0);
    ret = index->AddFeature(ntotal, base.data(), attrs.data(), label.data());
    EXPECT_EQ(ret, 0);
    int64_t validNum = 0;
    index->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal);
    std::vector<uint32_t> delToken{0, 1};
    auto ts = GetMillisecs();
    ret = index->DeleteFeatureByToken(2, delToken.data());
    EXPECT_EQ(ret, 0);
    auto te = GetMillisecs();
    printf("DeleteFeatureByToken delete cost  total %f ms \n" ,te-ts);
    index->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal / 2);
}

TEST(TestAscendIndexTS, Acc)
{
    idx_t ntotal = 1000000;
    uint32_t addNum = 1;
    uint32_t deviceId = 0;
    uint32_t dim = 512;
    uint32_t tokenNum = 2500;
    std::vector<int> queryNums = { 10 }; // make sure the vector is in ascending order
    std::vector<int> topks = { 10 };
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_HAMMING);
    EXPECT_EQ(ret, 0);

    std::vector<uint8_t> features(ntotal * dim / 8);
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
        int delCount = 300000;
        auto ts = GetMillisecs();
        tsIndex->DeleteFeatureByLabel(delCount, labels.data()+100000);
        auto te = GetMillisecs();
        printf("DeleteFeatureByLabel delete [%d] cost  total %f ms \n" ,delCount, te-ts);
    }

    for (auto k : topks) {
        int warmupTimes = 3;
        int loopTimes = 2;
        for (auto queryNum : queryNums) {
            std::vector<float> distances(queryNum * k, -1);
            std::vector<int64_t> labelRes(queryNum * k, 10);
            std::vector<uint32_t> validnum(queryNum, 0);
            uint32_t size = queryNum * dim / 8;
            std::vector<uint8_t> querys(size);
            querys.assign(features.begin(), features.begin() + size);

            uint32_t setlen = (uint32_t)(((tokenNum + 7) / 8));
            std::vector<uint8_t> bitSet(setlen, 0);
            bitSet[0] = 0x1 << 0 | 0x1 << 1 | 0x1 << 2 | 0x1 << 3;
            AttrFilter filter {};
            filter.timesStart = 0;
            filter.timesEnd = 3;
            filter.tokenBitSet = bitSet.data();
            filter.tokenBitSetLen = setlen;

            std::vector<AttrFilter> queryFilters(queryNum, filter);
            for (int i = 0; i < loopTimes + warmupTimes; i++) {
                if (i == warmupTimes) {
                    continue;
                }
                tsIndex->Search(queryNum, querys.data(), queryFilters.data(), false, k, labelRes.data(),
                                distances.data(), validnum.data());
            }
            for (size_t i = 0; i < queryNum; i++) {
                printf("TSdistances[%d]: ", i);
                for (size_t j = 0; j < k; j++) {
                    printf("%10f  ", distances[i * k + j]);
                }
                printf("\r\n");
                printf("TSlabelRes[%d]: ", i);
                for (size_t j = 0; j < k; j++) {
                    printf("%10d  ", labelRes[i * k + j]);
                }
                printf("\r\n");
            }
            for (size_t i = 0; i < queryNum; i++) {
                ASSERT_TRUE(labelRes[i * k] == i);
                ASSERT_TRUE(distances[i * k] == float(0));
            }
            // 00000111   -> 0,1
            bitSet[0] = 0x1 << 0 | 0x1 << 1;
            filter.timesStart = 1;
            filter.timesEnd = 3;

            queryFilters.clear();
            queryFilters.insert(queryFilters.begin(), queryNum, filter);
            for (int i = 0; i < loopTimes + warmupTimes; i++) {
                if (i == warmupTimes) {
                    continue;
                }
                tsIndex->Search(queryNum, querys.data(), queryFilters.data(), false, k, labelRes.data(),
                                distances.data(), validnum.data());
            }
            for (size_t i = 0; i < queryNum; i++) {
                printf("TSdistances[%d]: ", i);
                for (size_t j = 0; j < k; j++) {
                    printf("%10f  ", distances[i * k + j]);
                }
                printf("\r\n");
                printf("TSlabelRes[%d]: ", i);
                for (size_t j = 0; j < k; j++) {
                    printf("%10d  ", labelRes[i * k + j]);
                }
                printf("\r\n");
            }

            for (size_t i = 0; i < queryNum; i++) {
                if (i % 4 == 1) {
                    ASSERT_TRUE(labelRes[i * k] == i);
                    ASSERT_TRUE(distances[i * k] == float(0));
                } else {
                    ASSERT_TRUE(labelRes[i * k] != i);
                    ASSERT_TRUE(distances[i * k] >= float(0));
                }
            }
        }
    }
    delete tsIndex;
}

TEST(TestAscendIndexTS, SearchNoShareQPS)
{
    idx_t ntotal = 1000000;
    uint32_t addNum = 1;
    uint32_t deviceId = 0;
    uint32_t dim = 512;
    uint32_t tokenNum = 2500;
    std::vector<int> queryNums = { 1, 2, 4, 8, 16, 32, 64, 128, 256 }; // make sure the vector is in ascending order
    std::vector<int> topks = { 1024 };
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_HAMMING);
    EXPECT_EQ(ret, 0);

    std::vector<uint8_t> features(ntotal * dim / 8);
    printf("[add -----------]\n");
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

    long double ts { 0. };
    long double te { 0. };
    for (auto k : topks) {
        int warmupTimes = 3;
        int loopTimes = 2;
        for (auto queryNum : queryNums) {
            std::vector<float> distances(queryNum * k, -1);
            std::vector<int64_t> labelRes(queryNum * k, -1);
            std::vector<uint32_t> validnum(queryNum, 1);
            uint32_t size = queryNum * dim / 8;
            std::vector<uint8_t> querys(size);
            querys.assign(features.begin(), features.begin() + size);

            uint32_t setlen = (uint32_t)(((tokenNum + 7) / 8));
            std::vector<uint8_t> bitSet(setlen, 0);
            // 00000111   -> 0,1,2
            bitSet[0] = 0x1 << 0 | 0x1 << 1 | 0x1 << 2;

            AttrFilter filter {};
            filter.timesStart = 0;
            filter.timesEnd = 100;
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

    delete tsIndex;
}

TEST(TestAscendIndexTS, SearchShareQPS)
{
    idx_t ntotal = 1000000;
    uint32_t addNum = 1;
    uint32_t deviceId = 0;
    uint32_t dim = 512;
    uint32_t tokenNum = 2500;
    std::vector<int> queryNums = { 1, 2, 4, 8, 16, 32, 64, 128, 256 }; // make sure the vector is in ascending order
    std::vector<int> topks = { 1024 };
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_HAMMING);

    EXPECT_EQ(ret, 0);

    std::vector<uint8_t> features(ntotal * dim / 8);
    printf("[add -----------]\n");
    FeatureGenerator(features);
    for (size_t i = 0; i < 1; i++) {
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

    long double ts { 0. };
    long double te { 0. };
    for (auto k : topks) {
        int warmupTimes = 3;
        int loopTimes = 2;
        for (auto queryNum : queryNums) {
            std::vector<float> distances(queryNum * k, -1);
            std::vector<int64_t> labelRes(queryNum * k, -1);
            std::vector<uint32_t> validnum(queryNum, 1);
            uint32_t size = queryNum * dim / 8;
            std::vector<uint8_t> querys(size);
            querys.assign(features.begin(), features.begin() + size);

            uint32_t setlen = (uint32_t)(((tokenNum + 7) / 8));
            std::vector<uint8_t> bitSet(setlen, 0);
            // 00000111   -> 0,1,2
            bitSet[0] = 0x1 << 0 | 0x1 << 1 | 0x1 << 2;

            AttrFilter filter {};
            filter.timesStart = 0;
            filter.timesEnd = 100;
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

    delete tsIndex;
}

TEST(TestAscendIndexTS, HKQPS)
{
    idx_t ntotal = 1000000;
    uint32_t addNum = 250;
    uint32_t deviceId = 0;
    uint32_t dim = 512;
    uint32_t tokenNum = 2500;
    std::vector<int> queryNums = { 32 }; // make sure the vector is in ascending order
    std::vector<int> topks = { 20000 };
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_HAMMING);

    EXPECT_EQ(ret, 0);
    std::vector<uint8_t> features(ntotal * dim / 8);
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

    long double ts { 0. };
    long double te { 0. };
    for (auto k : topks) {
        int warmupTimes = 3;
        int loopTimes = 2;
        for (auto queryNum : queryNums) {
            std::vector<float> distances(queryNum * k, -1);
            std::vector<int64_t> labelRes(queryNum * k, -1);
            std::vector<uint32_t> validnum(queryNum, 1);
            uint32_t size = queryNum * dim / 8;
            std::vector<uint8_t> querys(size);
            querys.assign(features.begin(), features.begin() + size);

            uint32_t setlen = (uint32_t)(((tokenNum + 7) / 8));
            std::vector<uint8_t> bitSet(setlen, 0);
            // 00000111   -> 0,1,2
            bitSet[0] = 0x1 << 0 | 0x1 << 1 | 0x1 << 2;

            AttrFilter filter {};
            filter.timesStart = 0;
            filter.timesEnd = 100;
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

    delete tsIndex;
}

TEST(TestAscendIndexTS, YCQPS)
{
    idx_t ntotal = 1000000;
    uint32_t addNum = 200;
    uint32_t deviceId = 0;
    uint32_t dim = 512;
    uint32_t tokenNum = 2500;
    std::vector<int> queryNums = { 256 }; // make sure the vector is in ascending order
    std::vector<int> topks = { 1024 };
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_HAMMING);

    EXPECT_EQ(ret, 0);

    std::vector<uint8_t> features(ntotal * dim / 8);
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

    long double ts { 0. };
    long double te { 0. };
    for (auto k : topks) {
        int warmupTimes = 3;
        int loopTimes = 2;
        for (auto queryNum : queryNums) {
            std::vector<float> distances(queryNum * k, -1);
            std::vector<int64_t> labelRes(queryNum * k, -1);
            std::vector<uint32_t> validnum(queryNum, 1);
            uint32_t size = queryNum * dim / 8;
            std::vector<uint8_t> querys(size);
            querys.assign(features.begin(), features.begin() + size);

            uint32_t setlen = (uint32_t)(((tokenNum + 7) / 8));
            std::vector<uint8_t> bitSet(setlen, 0);
            // 00000111   -> 0,1,2
            bitSet[0] = 0x1 << 0 | 0x1 << 1 | 0x1 << 2;

            AttrFilter filter {};
            filter.timesStart = 0;
            filter.timesEnd = 100;
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
            printf("base: %ld,  dim: %d,  batch: %4d,  top%d,  QPS:%7.2Lf\n", ntotal * 200, dim, queryNum, k,
                   (long double)1000.0 * queryNum * loopTimes / (te - ts));
        }
    }

    delete tsIndex;
}


int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
