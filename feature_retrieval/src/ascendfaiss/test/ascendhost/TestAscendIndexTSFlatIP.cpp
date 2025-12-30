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


#include <numeric>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <random>
#include <algorithm>
#include <sys/time.h>
#include <gtest/gtest.h>
#include <faiss/index_io.h>
#include <faiss/IndexFlat.h>
#include <faiss/ascend/AscendIndexTS.h>
#include <faiss/ascend/AscendIndexFlat.h>
#include <thread>
#include "acl/acl.h"
#include <time.h>
#include <unistd.h>
#include <fstream>

namespace {
using idx_t = int64_t;
using FeatureAttr = faiss::ascend::FeatureAttr;
using AttrFilter = faiss::ascend::AttrFilter;

int32_t dim = 768;
int32_t deviceId = 0;
uint32_t tokenNum = 2500;
constexpr int MASK_ALIGN = 8;
uint32_t g_customAttrLen = 2;
uint32_t g_customAttrBlockSize = 262144;

template<typename T>
static void WriteTensorToFile(const std::string name, T* data, size_t num)
{
    std::ofstream of(name + ".bin", std::ios::out | std::ios::binary);
    of.write(reinterpret_cast<char *>(data), num * sizeof(T));
    of.close();
}

inline int GenRand(int maxValue)
{
    std::random_device rd;                                   // 获取随机数种子
    std::mt19937 gen(rd());                                  // 使用 mt19937 随机数生成器
    std::uniform_int_distribution<int> dis(0, maxValue - 1); // 定义均匀分布函数，范围为 [1, 100]

    int random_num = dis(gen); // 生成随机数
    return random_num;
}

inline double GetMillisecs()
{
    struct timeval tv = { 0, 0 };
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

void Norm(float *data, int64_t n, int64_t dim)
{
#pragma omp parallel for if (n > 1)
    for (int64_t i = 0; i < n; ++i) {
        float l2norm = 0;
        for (int64_t j = 0; j < dim; ++j) {
            l2norm += data[i * dim + j] * data[i * dim + j];
        }
        l2norm = sqrt(l2norm);

        for (int64_t j = 0; j < dim; ++j) {
            data[i * dim + j] = data[i * dim + j] / l2norm;
        }
    }
}

void FeatureGenerator(std::vector<float> &features)
{
    // 获取高精度时间点（纳秒精度）
    auto now = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    long seed = static_cast<long>(ns); // 转换为long（自动处理32/64位）
    srand48(seed); // 设置种子
    size_t n = features.size();
    for (size_t i = 0; i < n; ++i) {
        features[i] = drand48();
    }
}

void ScaleGenerator(std::vector<float> &scale, float min = -1.0, float max = 1.0)
{
    // 使用当前时间作为随机种子
    unsigned seed = static_cast<unsigned>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());

    std::mt19937 gen(seed);  // 使用梅森旋转算法生成器
    std::uniform_real_distribution<float> dist(min, max);  // 定义均匀分布

    // 生成随机数并填充到 scale 中
    for (size_t i = 0; i < scale.size(); ++i) {
        scale[i] = dist(gen);
    }
}

// 量化误差
float CalculateQuantizationError(float a, float scale)
{
    int8_t q = static_cast<int8_t>(a * scale);
    float b = static_cast<float>(q) / scale;
    return std::fabs(a - b);
}

void FeatureCustomAttr(std::vector<uint8_t> &customAttr)
{
    // 使用随机设备获取随机种子
    std::random_device rd;
    // 结合当前时间，增加随机性
    std::mt19937 gen(rd() ^ std::chrono::system_clock::now().time_since_epoch().count());
    // 定义分布范围为0到255
    std::uniform_int_distribution<uint8_t> dist(0, 255);

    // 生成随机数据
    size_t dataSize = customAttr.size();
    for (size_t i = 0; i < dataSize; ++i) {
        customAttr[i] = (dist(gen));
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

// index-add-getBaseSize-getbase-copyto
TEST(TestAscendIndexTSFlatIP, AddFeature)
{
    idx_t ntotal = 100000;
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto res = tsIndex->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_IP_FP16);
    EXPECT_EQ(res, 0);
    std::vector<float> features(ntotal * dim);
    printf("[---add -----------]\n");
    FeatureGenerator(features);
    std::vector<int64_t> labels;
    for (int i = 0; i < ntotal; ++i) {
        labels.push_back(i);
    }
    std::vector<FeatureAttr> attrs(ntotal);
    FeatureAttrGenerator(attrs);
    auto ts = GetMillisecs();
    res = tsIndex->AddFeature(ntotal, features.data(), attrs.data(), labels.data());
    auto te = GetMillisecs();
    printf("add %ld cost %f ms\n", ntotal, te - ts);
    EXPECT_EQ(res, 0);
    int64_t validNum = 0;
    tsIndex->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal);
    delete tsIndex;
}

// index-add-getBaseSize-getbase-copyto
TEST(TestAscendIndexTSFlatIP, Search)
{
    idx_t ntotal = 100000;
    std::vector<int> queryNums = { 100 }; // make sure the vector is in ascending order
    std::vector<int> topks = { 100 };
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto res = tsIndex->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_IP_FP16);
    EXPECT_EQ(res, 0);
    std::vector<float> features(ntotal * dim);
    printf("[---add -----------]\n");
    FeatureGenerator(features);
    std::vector<int64_t> labels;
    for (int i = 0; i < ntotal; ++i) {
        labels.push_back(i);
    }
    std::vector<FeatureAttr> attrs(ntotal);
    FeatureAttrGenerator(attrs);
    auto ts = GetMillisecs();
    res = tsIndex->AddFeature(ntotal, features.data(), attrs.data(), labels.data());
    auto te = GetMillisecs();
    printf("add %ld cost %f ms\n", ntotal, te - ts);
    EXPECT_EQ(res, 0);
    int64_t validNum = 0;
    tsIndex->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal);

    faiss::IndexFlatIP index(dim);
    index.add(ntotal, features.data());

    for (auto k : topks) {
        int warmupTimes = 0;
        int loopTimes = 1;
        for (auto queryNum : queryNums) {
            std::vector<float> distances(queryNum * k, -1);
            std::vector<int64_t> labelRes(queryNum * k, 10);
            std::vector<float> faissDistances(queryNum * k, -1);
            std::vector<int64_t> faissLabelRes(queryNum * k, 10);
            std::vector<uint32_t> validnum(queryNum, 0);
            uint32_t size = queryNum * dim;
            std::vector<float> querys(size);
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
                res = tsIndex->Search(queryNum, querys.data(), queryFilters.data(), false, k, labelRes.data(),
                    distances.data(), validnum.data());
                index.search(queryNum, querys.data(), k, faissDistances.data(), faissLabelRes.data());
                EXPECT_EQ(res, 0);
            }
            for (size_t i = 0; i < queryNum; i++) {
                ASSERT_TRUE(labelRes[i * k] == i);
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
                res = tsIndex->Search(queryNum, querys.data(), queryFilters.data(), false, k, labelRes.data(),
                    distances.data(), validnum.data());
                EXPECT_EQ(res, 0);
            }
            for (size_t i = 0; i < queryNum; i++) {
                if (i % 4 == 1) {
                    ASSERT_TRUE(labelRes[i * k] == i);
                } else {
                    ASSERT_TRUE(labelRes[i * k] != i);
                }
            }
        }
    }
    delete tsIndex;
}


TEST(TestAscendIndexTSFlatIP, Acc)
{
    idx_t ntotal = 10000;
    uint32_t addNum = 4;
    int queryNum = 10; // make sure the vector is in ascending order
    int k = 10;
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_IP_FP16);
    EXPECT_EQ(ret, 0);

    std::vector<float> features(ntotal * addNum * dim);
    printf("[---add -----------]\n");
    FeatureGenerator(features);

    Norm(features.data(), ntotal * addNum, dim);
    for (size_t i = 0; i < addNum; i++) {
        std::vector<int64_t> labels;

        for (int64_t j = 0; j < ntotal; ++j) {
            labels.emplace_back(j + i * ntotal);
        }
        std::vector<FeatureAttr> attrs(ntotal);
        FeatureAttrGenerator(attrs);
        auto ts0 = GetMillisecs();
        tsIndex->AddFeature(ntotal, features.data() + (i * ntotal * dim), attrs.data(), labels.data());
        auto te0 = GetMillisecs();
        printf("add %ld cost %f ms\n", ntotal, te0 - ts0);

        std::vector<float> getBase(100 * dim);
        auto ts = GetMillisecs();
        ret = tsIndex->GetFeatureByLabel(100, labels.data(), getBase.data());
        auto te = GetMillisecs();
        printf("GetFeatureByLabel  cost  total %f ms \n", te - ts);
        EXPECT_EQ(ret, 0);
        for (int j = 0; j < (100 - 1) * dim;) {
            ASSERT_TRUE(std::abs(features[j + i * ntotal * dim] - getBase[j]) < 0.001);
            j = j + dim;
        }
    }
    int64_t validNum = 0;
    tsIndex->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal * addNum);


    faiss::IndexFlatIP index(dim);
    index.add(ntotal, features.data());

    int warmupTimes = 2;
    int loopTimes = 10;


    std::vector<float> distances(queryNum * k, -1);
    std::vector<int64_t> labelRes(queryNum * k, 10);
    std::vector<float> faissDistances(queryNum * k, -1);
    std::vector<int64_t> faissLabelRes(queryNum * k, 10);
    std::vector<uint32_t> validnum(queryNum, 0);
    uint32_t size = queryNum * dim;
    std::vector<float> querys(size);
    querys.assign(features.begin(), features.begin() + size);

    uint32_t setlen = (uint32_t)((tokenNum + 7) / 8);
    std::vector<uint8_t> bitSet(setlen, 255);

    // 00001111
    bitSet[0] = 0x1 << 0 | 0x1 << 1 | 0x1 << 2 | 0x1 << 3;
    AttrFilter filter {};
    filter.timesStart = 0;
    filter.timesEnd = 2;
    filter.tokenBitSet = bitSet.data();
    filter.tokenBitSetLen = setlen;

    std::vector<AttrFilter> queryFilters(queryNum, filter);


    for (int i = 0; i < loopTimes + warmupTimes; i++) {
        auto res = tsIndex->Search(queryNum, querys.data(), queryFilters.data(), false, k, labelRes.data(),
            distances.data(), validnum.data());
        index.search(queryNum, querys.data(), k, faissDistances.data(), faissLabelRes.data());
        ASSERT_EQ(res, 0);
    }

    for (int i = 0; i < queryNum; i++) {
        printf("TSdistances[%d]: ", i);
        for (int j = 0; j < k; j++) {
            printf("%10f  ", distances[i * k + j]);
        }
        printf("\r\n");
        printf("tslabelRes[%d]: ", i);
        for (int j = 0; j < k; j++) {
            printf("%10ld  ", labelRes[i * k + j]);
        }

        printf("\r\n");
    }
    delete tsIndex;

    printf("ascend add start \n");
    int filterTotal = ntotal * addNum;
    std::vector<faiss::idx_t> ids(filterTotal);
    std::iota(ids.begin(), ids.end(), 0);

    auto it = ids.begin();
    auto featureIt = features.begin();
    while (it != ids.end()) {
        if ((*it % 4) == 3) {
            it = ids.erase(it);
            featureIt = features.erase(featureIt, featureIt + dim);
            filterTotal--;
        } else {
            it++;
            featureIt = featureIt + dim;
        }
    }
    ntotal = filterTotal / addNum;
    printf("ascend add final \n");

    faiss::ascend::AscendIndexFlatConfig conf({ deviceId });
    faiss::ascend::AscendIndexFlat ascendindex(dim, faiss::METRIC_INNER_PRODUCT, conf);

    ascendindex.add_with_ids(filterTotal, features.data(), ids.data());
    printf("ascend add final--- \n");

    std::vector<float> ascenddistances(queryNum * k, -1);
    std::vector<int64_t> ascendlabelRes(queryNum * k, 10);

    for (int i = 0; i < loopTimes + warmupTimes; i++) {
        ascendindex.search(queryNum, querys.data(), k, ascenddistances.data(), ascendlabelRes.data());
    }

    for (int i = 0; i < queryNum; i++) {
        printf("ascenddistances[%d]: ", i);
        for (int j = 0; j < k; j++) {
            printf("%10f  ", ascenddistances[i * k + j]);
        }
        printf("\r\n");
        printf("ascendlabelRes[%d]: ", i);
        for (int j = 0; j < k; j++) {
            printf("%10ld  ", ascendlabelRes[i * k + j]);
        }
        printf("\r\n");
    }

    for (int i = 0; i < queryNum; i++) {
        float lastdis = distances[i * k + (k - 1)];
        for (int j = 0; j < k; j++) {
            EXPECT_EQ(ascenddistances[i * k + j], distances[i * k + j]);
            if (distances[i * k + j] != lastdis) {
                auto result =
                    find(ascendlabelRes.begin() + i * k, ascendlabelRes.begin() + (i + 1) * k, labelRes[i * k + j]);
                ASSERT_FALSE(result == ascendlabelRes.begin() + (i + 1) * k);
            }
        }
    }
}

TEST(TestAscendIndexTSFlatIP, SearchNoShareQPS)
{
    idx_t ntotal = 1000000;
    uint32_t addNum = 10;
    std::vector<int> queryNums = { 128, 64, 48, 36, 32, 30, 24, 18, 16, 12, 8, 6, 4, 2, 1 };
    std::vector<int> topks = { 10 };
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_IP_FP16);
    EXPECT_EQ(ret, 0);

    std::vector<float> features(ntotal * dim);
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
        int loopTimes = 100;
        for (auto queryNum : queryNums) {
            std::vector<float> distances(queryNum * k, -1);
            std::vector<int64_t> labelRes(queryNum * k, -1);
            std::vector<uint32_t> validnum(queryNum, 1);
            uint32_t size = queryNum * dim;
            std::vector<float> querys(size);
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
                    distances.data(), validnum.data(), false);
            }
            te = GetMillisecs();

            printf("base: %ld,  dim: %d,  batch: %4d,  top%d,  QPS:%7.2Lf\n", ntotal * addNum, dim, queryNum, k,
                (long double)1000.0 * queryNum * loopTimes / (te - ts));
        }
    }
    delete tsIndex;
}

TEST(TestAscendIndexTSFlatIP, GetFeatureByLabel)
{
    int ntotal = 10000;
    std::vector<float> base(ntotal * dim);
    FeatureGenerator(base);
    std::vector<int64_t> label(ntotal);
    std::iota(label.begin(), label.end(), 0);
    std::vector<FeatureAttr> attrs(ntotal);
    FeatureAttrGenerator(attrs);
    auto *index = new faiss::ascend::AscendIndexTS();
    auto ret = index->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_IP_FP16);
    EXPECT_EQ(ret, 0);
    ret = index->AddFeature(ntotal, base.data(), attrs.data(), label.data());
    EXPECT_EQ(ret, 0);
    std::vector<float> getBase(ntotal * dim);
    auto ts = GetMillisecs();
    ret = index->GetFeatureByLabel(ntotal, label.data(), getBase.data());
    auto te = GetMillisecs();
    printf("GetFeatureByLabel  cost  total %f ms \n", te - ts);
    EXPECT_EQ(ret, 0);

    for (int i = 0; i < (ntotal - 1) * dim;) {
        ASSERT_TRUE(std::abs(base[i] - getBase[i]) < 0.001);
        i = i + dim;
    }
    delete index;
}

TEST(TestAscendIndexTSFlatIP, GetBaseByRange)
{
    idx_t ntotal = 100;
    uint32_t addNum = 1;
    uint32_t deviceId = 0;
    uint32_t dim = 256;
    uint32_t tokenNum = 2500;

    using namespace std;
    cout << "***** start Init *****" << endl;
    faiss::ascend::AscendIndexTS *tsIndex0 = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex0->Init(0, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_IP_FP16);
    EXPECT_EQ(ret, 0);
    cout << "***** start Init  1*****" << endl;
    faiss::ascend::AscendIndexTS *tsIndex1 = new faiss::ascend::AscendIndexTS();
    ret = tsIndex1->Init(0, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_IP_FP16);
    EXPECT_EQ(ret, 0);
    cout << "***** end Init *****" << endl;

    cout << "***** start add *****" << endl;
    std::vector<float> features(ntotal * dim);
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
    std::vector<float> retBase(needAddNum * dim);
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

    for (int i = 0; i < needAddNum * dim; i++) {
        ASSERT_TRUE(std::abs(features[i + needAddNum * dim] - retBase[i]) < 0.001);
        if (std::abs(features[i + needAddNum * dim] - retBase[i]) > 0.001) {
            printf("%f--%f \n", features[i + needAddNum * dim], retBase[i]);
        }
    }
    delete tsIndex0;
    delete tsIndex1;
}

TEST(TestAscendIndexTSFlatIP, GetFeatureAttrByLabel)
{
    size_t ntotal = 1000000;
    int addNum = 1;
    std::vector<float> base(ntotal * addNum * dim);
    FeatureGenerator(base);
    std::vector<int64_t> labels(ntotal * addNum);
    std::iota(labels.begin(), labels.end(), 0);
    std::vector<FeatureAttr> attrs(ntotal * addNum);
    FeatureAttrGenerator(attrs);
    auto *index = new faiss::ascend::AscendIndexTS();
    auto ret = index->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_IP_FP16);
    EXPECT_EQ(ret, 0);
    for (size_t i = 0; i < addNum; i++) {
        auto ts0 = GetMillisecs();
        ret = index->AddFeature(ntotal, base.data() + i * ntotal * dim / 8, attrs.data() + i * ntotal,
            labels.data() + i * ntotal);
        EXPECT_EQ(ret, 0);
        auto te0 = GetMillisecs();
        printf("add %ld cost %f ms\n", ntotal, te0 - ts0);
    }
    int getAttrCnt = 1000000;
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

    std::vector<float> oneBase(1 * dim);
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

TEST(TestAscendIndexTSFlatIP, DeleteFeatureByToken)
{
    int tokenNum = 2500;
    int ntotal = 20000;
    std::vector<float> base(ntotal * dim);
    FeatureGenerator(base);
    std::vector<int64_t> label(ntotal);
    std::iota(label.begin(), label.end(), 0);
    std::vector<FeatureAttr> attrs(ntotal);
    FeatureAttrGenerator(attrs);
    auto *index = new faiss::ascend::AscendIndexTS();
    auto ret = index->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_IP_FP16);
    EXPECT_EQ(ret, 0);
    ret = index->AddFeature(ntotal, base.data(), attrs.data(), label.data());
    EXPECT_EQ(ret, 0);
    int64_t validNum = 0;
    index->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal);
    std::vector<uint32_t> delToken { 0, 1 };
    auto ts = GetMillisecs();
    index->DeleteFeatureByToken(2, delToken.data());
    auto te = GetMillisecs();
    printf("DeleteFeatureByToken delete cost  total %f ms \n", te - ts);
    index->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal / 2);
}

TEST(TestAscendIndexTSFlatIP, DeleteFeatureByLabel)
{
    size_t ntotal = 100000;
    int tokenNum = 2500;
    std::vector<float> base(ntotal * dim);
    FeatureGenerator(base);
    std::vector<int64_t> label(ntotal);
    std::iota(label.begin(), label.end(), 0);
    std::vector<FeatureAttr> attrs(ntotal);
    FeatureAttrGenerator(attrs);
    auto *index = new faiss::ascend::AscendIndexTS();
    auto ret = index->Init(0, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_IP_FP16);
    EXPECT_EQ(ret, 0);
    ret = index->AddFeature(ntotal, base.data(), attrs.data(), label.data());
    EXPECT_EQ(ret, 0);
    int64_t validNum = 0;
    index->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal);
    int delCount = 1;
    std::vector<int64_t> delLabel(delCount);
    delLabel.assign(label.begin(), label.begin() + delCount);
    auto ts = GetMillisecs();
    index->DeleteFeatureByLabel(delCount, delLabel.data());
    auto te = GetMillisecs();
    printf("DeleteFeatureByLabel delete cost  total %f ms \n", te - ts);
    index->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal - delCount);
    // delete again
    index->DeleteFeatureByLabel(delCount, delLabel.data());
    index->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal - delCount);

    std::vector<float> getBase(delCount * dim);
    ts = GetMillisecs();
    ret = index->GetFeatureByLabel(delCount, label.data() + ntotal - delCount, getBase.data());
    te = GetMillisecs();
    printf("GetFeatureByLabel  cost  total %f ms \n", te - ts);
    EXPECT_EQ(ret, 0);

    std::vector<FeatureAttr> getAttr(delCount);
    printf("GetFeatureAttrByLabel  start\n");
    ts = GetMillisecs();
    ret = index->GetFeatureAttrByLabel(delCount, label.data() + ntotal - delCount, getAttr.data());
    EXPECT_EQ(ret, 0);
    te = GetMillisecs();
    printf("GetFeatureAttrByLabel[%d]  cost  total %f ms \n", delCount, te - ts);

    for (int i = 0; i < getAttr.size(); i++) {
        EXPECT_EQ(attrs[i + ntotal - delCount].time, getAttr[i].time);
        EXPECT_EQ(attrs[i + ntotal - delCount].tokenId, getAttr[i].tokenId);
    }

    for (int i = 0; i < delCount * dim; i++) {
        ASSERT_TRUE(std::abs(base[i + (ntotal - delCount) * dim] - getBase[i]) <= float(0.005));
    }
}

TEST(TestAscendIndexTSFlatIP, SearchShareWithExtraQPS)
{
    idx_t ntotal = 100000;
    uint32_t addNum = 10;
    std::vector<int> queryNums = { 1, 2, 4, 8, 16, 32, 64, 128, 256 }; // make sure the vector is in ascending order
    std::vector<int> topks = { 100 };
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    auto ret = tsIndex->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_IP_FP16);
    EXPECT_EQ(ret, 0);

    printf("[add -----------]\n");
    std::vector<float> features(ntotal * dim);
    FeatureGenerator(features);
    for (int i = 0; i < addNum; i++) {
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
        int loopTimes = 10;
        for (auto queryNum : queryNums) {
            std::vector<float> distances(queryNum * k, -1);
            std::vector<int64_t> labelRes(queryNum * k, -1);
            std::vector<uint32_t> validnum(queryNum, 1);
            uint32_t size = queryNum * dim;
            std::vector<float> querys(size);
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
                ret = tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), true, k,
                    extraMask.data(), extraMaskLen, false, labelRes.data(), distances.data(), validnum.data());
                EXPECT_EQ(ret, 0);
            }
            te = GetMillisecs();
            printf("base: %ld,  dim: %d,  batch: %4d,  top%d,  QPS:%7.2Lf\n", ntotal * addNum, dim, queryNum, k,
                (long double)1000.0 * queryNum * loopTimes / (te - ts));
        }
    }
    delete tsIndex;
}

TEST(TestAscendIndexTSFlatIP, ExtraMaskAcc)
{
    idx_t ntotal = 1000000;
    int queryNum = 11; // make sure the vector is in ascending order
    int k = 10;
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    tsIndex->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_IP_FP16);

    std::vector<float> features(ntotal * dim);
    printf("[---add -----------]\n");
    FeatureGenerator(features);
    Norm(features.data(), ntotal, dim);
    std::vector<int64_t> labels;
    for (int i = 0; i < ntotal; ++i) {
        labels.push_back(i);
    }
    std::vector<FeatureAttr> attrs(ntotal);
    FeatureAttrGenerator(attrs);
    auto ts = GetMillisecs();
    auto res = tsIndex->AddFeature(ntotal, features.data(), attrs.data(), labels.data());
    auto te = GetMillisecs();
    printf("add %ld cost %f ms\n", ntotal, te - ts);
    EXPECT_EQ(res, 0);
    int64_t validNum = 0;
    tsIndex->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal);

    faiss::IndexFlatIP index(dim);
    index.add(ntotal, features.data());

    int warmupTimes = 0;
    int loopTimes = 1;
    std::vector<float> distances(queryNum * k, -1);
    std::vector<int64_t> labelRes(queryNum * k, 10);
    std::vector<float> extramaskDistances(queryNum * k, -1);
    std::vector<int64_t> extramaskLabelRes(queryNum * k, 10);
    std::vector<uint32_t> validnum(queryNum, 0);
    uint32_t size = queryNum * dim;
    std::vector<float> querys(size);
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

    int extraMaskLen = ntotal / 8;
    std::vector<uint8_t> extraMask(queryNum * extraMaskLen, 255);
    std::vector<AttrFilter> queryFilters(queryNum, filter);
    for (int i = 0; i < loopTimes + warmupTimes; i++) {
        res = tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), false, k, extraMask.data(),
            extraMaskLen, false, labelRes.data(), distances.data(), validnum.data());
        EXPECT_EQ(res, 0);
    }
    for (int i = 0; i < queryNum; i++) {
        ASSERT_TRUE(labelRes[i * k] == i);
    }

    for (int i = 0; i < queryNum; i++) {
        printf("ascenddistances[%d]: ", i);
        for (int j = 0; j < k; j++) {
            printf("%10f  ", distances[i * k + j]);
        }
        printf("\r\n");
        printf("ascendlabelRes[%d]: ", i);
        for (int j = 0; j < k; j++) {
            printf("%10ld  ", labelRes[i * k + j]);
        }
        printf("\r\n");
    }

    for (int i = 0; i < queryNum; i++) {
        int ind = 0;
        for (int j = 0; j < extraMaskLen; j++) {
            for (int k = 0; k < 8; k++) {
                if (i == ind) {
                    extraMask[j + i * extraMaskLen] = extraMask[j + i * extraMaskLen] & ~(0x1 << (ind % 8));
                    ind++;
                    break;
                }
                ind++;
            }
        }
    }
    for (int i = 0; i < queryNum; i++) {
        int num = labelRes[i * k + 1];
        extraMask[num / 8 + i * extraMaskLen] = extraMask[num / 8 + i * extraMaskLen] & ~(0x1 << (num % 8));
    }

    for (int i = 0; i < loopTimes + warmupTimes; i++) {
        res = tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), false, k, extraMask.data(),
            extraMaskLen, false, extramaskLabelRes.data(), extramaskDistances.data(), validnum.data());
        EXPECT_EQ(res, 0);
    }

    for (int i = 0; i < queryNum; i++) {
        printf("extramaskDistances[%d]: ", i);
        for (int j = 0; j < k; j++) {
            printf("%10f  ", extramaskDistances[i * k + j]);
        }
        printf("\r\n");
        printf("extramaskLabelRes[%d]: ", i);
        for (int j = 0; j < k; j++) {
            printf("%10ld  ", extramaskLabelRes[i * k + j]);
        }
        printf("\r\n");
    }

    for (int i = 0; i < queryNum; i++) {
        for (int j = 0; j < k - 2; j++) {
            EXPECT_EQ(distances[i * k + j + 2], extramaskDistances[i * k + j]);
        }
    }
    delete tsIndex;
}

void PrintResult(int queryNum,
                 int k,
                 const std::vector<float> &distances,
                 const std::vector<int64_t> &labelRes,
                 const std::vector<uint32_t> &validnum = std::vector<uint32_t>())
{
    for (int i = 0; i < queryNum; i++) {
        printf("ascenddistances[%d]: ", i);
        for (int j = 0; j < k; j++) {
            printf("%10f  ", distances[i * k + j]);
        }
        printf("\r\n");
        printf("ascendlabelRes[%d]: ", i);
        for (int j = 0; j < k; j++) {
            printf("%10ld  ", labelRes[i * k + j]);
        }
        printf("\r\n");
        if (!validnum.empty()) {
            printf("validnum[%d]: %u", i, validnum[i]);
            printf("\r\n");
        }
    }
}

static void GenerateData(int64_t ntotal,
                         int64_t queryNum,
                         int64_t extraMaskLen,
                         std::vector<float> &features,
                         int64_t indiceStart,
                         std::vector<int64_t> &indices,
                         std::vector<FeatureAttr> &attrs,
                         std::vector<uint8_t> &extraMask,
                         std::vector<uint8_t> &bitSet,
                         std::vector<AttrFilter> &queryFilters)
{
    features.resize(ntotal * dim);
    FeatureGenerator(features);
    Norm(features.data(), ntotal, dim);

    indices.resize(ntotal);
    std::iota(indices.begin(), indices.end(), indiceStart);

    attrs.resize(ntotal);
    FeatureAttrGenerator(attrs);

    extraMask.resize(queryNum * extraMaskLen, 255); // 255 bit位全是1，不过滤

    uint32_t setLen = static_cast<uint32_t>((tokenNum + MASK_ALIGN - 1) / MASK_ALIGN);
    bitSet.resize(setLen, 0);
    // 00001111
    bitSet[0] = 0x1 << 0 | 0x1 << 1 | 0x1 << 2 | 0x1 << 3;
    AttrFilter filter {};
    filter.timesStart = 0;
    filter.timesEnd = 3;  // 3
    filter.tokenBitSet = bitSet.data();
    filter.tokenBitSetLen = setLen;
    queryFilters.resize(queryNum, filter);
}

TEST(TestAscendIndexTSFlatIP, ExtraMaskExtraScoreAcc)
{
    idx_t ntotal = 10000000;
    int queryNum = 111; // 111覆盖所有batch size
    int k = 10;

    std::vector<float> features;
    std::vector<int64_t> labels;
    std::vector<FeatureAttr> attrs;
    int extraMaskLen = (ntotal + 8 - 1) / 8; // 按照8对齐
    std::vector<uint8_t> extraMask;
    std::vector<uint8_t> bitSet;
    std::vector<AttrFilter> queryFilters;

    GenerateData(ntotal, queryNum, extraMaskLen, features, 0, labels, attrs, extraMask, bitSet, queryFilters);

    std::shared_ptr<faiss::ascend::AscendIndexTS> tsIndex;
    try {
        tsIndex = std::make_shared<faiss::ascend::AscendIndexTS>();
        tsIndex->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_IP_FP16);
        idx_t addPaged = 1000000;
        for (idx_t i = 0; i < ntotal; i += addPaged) {
            idx_t addCount = std::min(addPaged, ntotal - i);
            auto res = tsIndex->AddFeature(addCount, features.data() + i * dim, attrs.data() + i, labels.data() + i);
            EXPECT_EQ(res, 0);
        }
        int64_t validNum = 0;
        tsIndex->GetFeatureNum(&validNum);
        EXPECT_EQ(validNum, ntotal);

        std::vector<float> distances(static_cast<int64_t>(queryNum) * k, -1);
        std::vector<int64_t> labelRes(static_cast<int64_t>(queryNum) * k, 10); // 10, default value
        std::vector<uint32_t> validnum(static_cast<int64_t>(queryNum), 0);
        int64_t size = static_cast<int64_t>(queryNum) * dim;
        std::vector<float> querys(static_cast<int64_t>(queryNum) * dim);
        querys.assign(features.begin(), features.begin() + size);

        int64_t ntotalPad = (ntotal + 16 - 1) / 16 * 16; // 长度等于底库长度按照16对齐
        float extraScoreValue = 1.0;
        std::vector<float> extraScore(queryNum * ntotalPad, extraScoreValue);
        std::vector<uint16_t> extraScoreFp16(extraScore.size());
        std::transform(extraScore.begin(), extraScore.end(), extraScoreFp16.begin(),
            [] (float tmp) { return aclFloatToFloat16(tmp); });
        auto res = tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), false, k,
            extraMask.data(), extraMaskLen, false, extraScoreFp16.data(), labelRes.data(), distances.data(),
            validnum.data());
        EXPECT_EQ(res, 0);
        for (int i = 0; i < queryNum; i++) {
            EXPECT_TRUE(labelRes[i * k] == i);
        }
        PrintResult(queryNum, k, distances, labelRes);

        // save origin data for comparing with cpu
        std::string name = "features_ori";
        WriteTensorToFile(name, features.data(), features.size());
        name = "querys_ori";
        WriteTensorToFile(name, querys.data(), querys.size());
        name = "extra_score_fp16";
        WriteTensorToFile(name, extraScoreFp16.data(), extraScoreFp16.size());
        name = "dist_result";
        WriteTensorToFile(name, distances.data(), distances.size());
    } catch (std::exception &e) {
        printf("catch exception %s\n", e.what());
    }
}

static bool TestQPS(int warmupTimes, int loopTimes, idx_t ntotal, uint32_t k, const std::vector<int> &batchSize,
                    const std::vector<float> &querys, std::vector<AttrFilter> &queryFilters,
                    const std::vector<uint16_t> &extraScore, const std::vector<uint8_t> &extraMask,
                    uint64_t extraMaskLen, bool extraMaskIsAtDevice, std::vector<int64_t> &labels,
                    std::vector<float> &distances, std::vector<uint32_t> &validNums,
                    std::shared_ptr<faiss::ascend::AscendIndexTS> &tsIndex)
{
    for (size_t i = 0; i < batchSize.size(); i++) {
        for (int j = 0; j < warmupTimes; j++) {
            auto res = tsIndex->SearchWithExtraMask(batchSize[i], querys.data(), queryFilters.data(), false, k,
                extraMask.data(), extraMaskLen, false, labels.data(), distances.data(), validNums.data());
            if (res != 0) {
                return false;
            }
            res = tsIndex->SearchWithExtraMask(batchSize[i], querys.data(), queryFilters.data(),
                false, k, extraMask.data(), extraMaskLen, false, extraScore.data(), labels.data(),
                distances.data(), validNums.data());
            if (res != 0) {
                return false;
            }
        }
    }
    printf("non extra score\n");
    for (size_t i = 0; i < batchSize.size(); i++) {
        auto ts = GetMillisecs();
        for (int j = 0; j < loopTimes; j++) {
            auto ret = tsIndex->SearchWithExtraMask(batchSize[i], querys.data(), queryFilters.data(), false, k,
                extraMask.data(), extraMaskLen, false, labels.data(), distances.data(), validNums.data());
            if (ret != 0) {
                return false;
            }
        }
        auto te = GetMillisecs();
        printf("base: %ld,  dim: %d,  batch: %4d,  top%d, avg_time:%10.4f, QPS:%7.2f\n", ntotal, dim,
            batchSize[i], k, (te - ts) / loopTimes,
            static_cast<double>(1000.0) * batchSize[i] * loopTimes / (te - ts)); // 1000.0 ms
    }
    printf("with extra score\n");
    for (size_t i = 0; i < batchSize.size(); i++) {
        auto ts = GetMillisecs();
        for (int j = 0; j < loopTimes; j++) {
            auto ret = tsIndex->SearchWithExtraMask(batchSize[i], querys.data(), queryFilters.data(), false, k,
                extraMask.data(), extraMaskLen, false, extraScore.data(), labels.data(), distances.data(),
                validNums.data());
            if (ret != 0) {
                return false;
            }
        }
        auto te = GetMillisecs();
        printf("base: %ld,  dim: %d,  batch: %4d,  top%d, avg_time:%10.4f, QPS:%7.2f\n", ntotal, dim,
            batchSize[i], k, (te - ts) / loopTimes,
            static_cast<double>(1000.0) * batchSize[i] * loopTimes / (te - ts)); // 1000.0 ms
    }
    return true;
}

TEST(TestAscendIndexTSFlatIP, ExtraMaskExtraScoreQPS)
{
    idx_t ntotal = 20000000;
    int k = 10;

    std::vector<float> features;
    std::vector<int64_t> labels;
    std::vector<FeatureAttr> attrs;
    int extraMaskLen = (ntotal + 8 - 1) / 8; // 按照8对齐
    std::vector<uint8_t> extraMask;
    std::vector<uint8_t> bitSet;
    std::vector<AttrFilter> queryFilters;
    std::vector<int> batchSize = { 1, 2, 4, 8, 16, 32, 48, 64, 128, 256 };
    int queryNum = batchSize.back();

    GenerateData(ntotal, queryNum, extraMaskLen, features, 0, labels, attrs, extraMask, bitSet, queryFilters);

    uint64_t resourceSize = static_cast<uint64_t>(4) * 1024 * 1024 * 1024; // 最大4GB，如果默认的话设置为0x60000000（1.5G）
    std::shared_ptr<faiss::ascend::AscendIndexTS> tsIndex;
    try {
        tsIndex = std::make_shared<faiss::ascend::AscendIndexTS>();
        tsIndex->InitWithExtraVal(deviceId, dim, tokenNum, resourceSize, faiss::ascend::AlgorithmType::FLAT_IP_FP16);
        idx_t addPaged = 1000000;
        for (idx_t i = 0; i < ntotal; i += addPaged) {
            idx_t addCount = std::min(addPaged, ntotal - i);
            auto ts = GetMillisecs();
            auto res = tsIndex->AddFeature(addCount, features.data() + i * dim, attrs.data() + i, labels.data() + i);
            auto te = GetMillisecs();
            printf("add %ld cost %f ms\n", addCount, te - ts);
            EXPECT_EQ(res, 0);
        }
        int64_t validNum = 0;
        tsIndex->GetFeatureNum(&validNum);
        EXPECT_EQ(validNum, ntotal);

        std::vector<float> distances(static_cast<int64_t>(queryNum) * k, -1);
        std::vector<int64_t> labelRes(static_cast<int64_t>(queryNum) * k, 10); // 10, default value
        std::vector<uint32_t> validNums(static_cast<int64_t>(queryNum), 0);
        int64_t size = static_cast<int64_t>(queryNum) * dim;
        std::vector<float> querys(static_cast<int64_t>(queryNum) * dim);
        querys.assign(features.begin(), features.begin() + size);
        features.clear();

        int64_t ntotalPad = (ntotal + 16 - 1) / 16 * 16; // 长度等于底库长度按照16对齐
        float extraScoreValue = 1.0;
        std::vector<float> extraScore(queryNum * ntotalPad, extraScoreValue);
        std::vector<uint16_t> extraScoreFp16(extraScore.size());
        std::transform(extraScore.begin(), extraScore.end(), extraScoreFp16.begin(),
            [] (float tmp) { return aclFloatToFloat16(tmp); });
        extraScore.clear();

        int warmupTimes = 3;
        int loopTimes = 10;
        if (!TestQPS(warmupTimes, loopTimes, ntotal, k, batchSize, querys, queryFilters, extraScoreFp16,
                    extraMask, extraMaskLen, false, labelRes, distances, validNums, tsIndex)) {
            printf("TestQPS failed\n");
        }
    } catch (std::exception &e) {
        printf("catch exception %s\n", e.what());
    }
}

// 定义一个保存特征向量及属性的类型，用来和npu获取的数据进行对比
struct BaseAllFeature {
    int64_t ntotal {0};
    std::vector<float> features;
    std::vector<FeatureAttr> attrs;
    std::vector<uint8_t> customAttrs;
    std::vector<uint8_t> baseMask;
    std::vector<float> scale;
};

constexpr int64_t SINGLE_ADD_LIMIT = 1000000;

void TestAdd(const std::vector<int64_t> &indices,
    std::shared_ptr<faiss::ascend::AscendIndexTS> &tsIndex, BaseAllFeature &baseAllFeature)
{
    int64_t addNum = indices.size();
    ASSERT_GT(addNum, 0);
    std::vector<float> features(addNum * dim);
    FeatureGenerator(features);
    Norm(features.data(), addNum, dim);

    std::vector<FeatureAttr> attrs(addNum);
    FeatureAttrGenerator(attrs);

    std::vector<uint8_t> customAttr(addNum * g_customAttrLen);
    FeatureCustomAttr(customAttr);

    int64_t featureNumBak = 0;
    auto res = tsIndex->GetFeatureNum(&featureNumBak);
    EXPECT_EQ(res, 0);
    // 增加一些
    for (int64_t i = 0; i < addNum; i += SINGLE_ADD_LIMIT) {
        int64_t addCount = std::min(SINGLE_ADD_LIMIT, addNum - i);
        double ts = GetMillisecs();
        auto res = tsIndex->AddFeatureByIndice(
            addCount, features.data() + i * dim, attrs.data() + i, indices.data() + i, nullptr,
            customAttr.data() + i * g_customAttrLen);
        ASSERT_EQ(res, 0);
        double te = GetMillisecs();
        printf("add %ld cost %f ms\n", addCount, te - ts);
    }

    int64_t newFeatureNum = 0;
    res = tsIndex->GetFeatureNum(&newFeatureNum);
    ASSERT_EQ(res, 0);
    ASSERT_EQ(newFeatureNum - featureNumBak, indices[addNum - 1] - featureNumBak + 1);

    // 检查新增加的部分是不是全是1
    int64_t maskLength = (newFeatureNum + MASK_ALIGN - 1) / MASK_ALIGN;
    std::vector<uint8_t> baseMask(maskLength); // 按8对齐
    res = tsIndex->GetBaseMask(maskLength, baseMask.data());
    ASSERT_EQ(res, 0);
    for (int64_t i = featureNumBak; i < newFeatureNum; i++) {
        int64_t byteIndex = i / MASK_ALIGN;
        int64_t bitIndex = i % MASK_ALIGN;
        ASSERT_NE(baseMask[byteIndex] & (1 << bitIndex), 0);
    }

    baseAllFeature.ntotal = newFeatureNum;
    baseAllFeature.features.resize(baseAllFeature.ntotal * dim);
    for (int64_t i = 0; i < addNum; i++) {
        std::copy(features.data() + i * dim, features.data() + i * dim + dim,
            baseAllFeature.features.data() + indices[i] * dim);
    }

    baseAllFeature.attrs.resize(baseAllFeature.ntotal);
    for (int64_t i = 0; i < addNum; i++) {
        baseAllFeature.attrs[indices[i]] = attrs[i];
    }

    baseAllFeature.customAttrs.resize(baseAllFeature.ntotal * g_customAttrLen);
    for (int64_t i = 0; i < addNum; i++) {
        std::copy(customAttr.data() + i * g_customAttrLen, customAttr.data() + i * g_customAttrLen + g_customAttrLen,
            baseAllFeature.customAttrs.data() + indices[i] * g_customAttrLen);
    }

    baseAllFeature.baseMask = std::move(baseMask);
}

// 自定义属性单独接口获取，按照block的方式整块获取，且获取的地址是device侧的空间。需要拷贝到host后，再转换后和npu的进行对比
void TestCustomAttr(int64_t featureNum, std::shared_ptr<faiss::ascend::AscendIndexTS> &tsIndex,
    const BaseAllFeature &baseAllFeature)
{
    uint32_t customBlockNum = (static_cast<uint32_t>(featureNum) + g_customAttrBlockSize - 1) / g_customAttrBlockSize;
    uint32_t leftCustomAttrSize = static_cast<uint32_t>(featureNum) % g_customAttrBlockSize;
    // 如果余数为0，则最后一个block是填满的
    leftCustomAttrSize = (leftCustomAttrSize == 0) ? g_customAttrBlockSize : leftCustomAttrSize;
    uint8_t *deviceCustomAttr = nullptr;
    uint32_t blockDataSize = g_customAttrBlockSize * g_customAttrLen;
    for (uint32_t i = 0; i < customBlockNum; i++) {
        auto res = tsIndex->GetCustomAttrByBlockId(i, deviceCustomAttr);
        ASSERT_EQ(res, 0);
        std::vector<uint8_t> customAttr(blockDataSize);
        res = aclrtMemcpy(customAttr.data(), customAttr.size() * sizeof(uint8_t),
            deviceCustomAttr, blockDataSize * sizeof(uint8_t),
            ACL_MEMCPY_DEVICE_TO_HOST);
        ASSERT_EQ(res, 0);
        
        std::vector<uint8_t> customAttrRaw(blockDataSize);
        uint32_t validSize = (i == customBlockNum - 1) ? leftCustomAttrSize : g_customAttrBlockSize;
        for (uint32_t j = 0; j < validSize; j++) {
            for (uint32_t k = 0; k < g_customAttrLen; k++) {
                customAttrRaw[j * g_customAttrLen + k] = customAttr[k * g_customAttrBlockSize + j];
            }
        }

        for (uint32_t j = 0; j < validSize * g_customAttrLen; j++) {
            auto baseFeatureData = baseAllFeature.customAttrs[i * blockDataSize + j];
            if (baseFeatureData != customAttrRaw[j]) {
                printf("i:[%u], j:[%u], [%d] vs [%d] is different\n", i, j, baseFeatureData, customAttrRaw[j]);
            }
            ASSERT_EQ(baseFeatureData, customAttrRaw[j]);
        }
    }
}

// 对比第i条特征的第j维度的数据是否是符合误差要求的。
bool CompareFeatureData(int64_t i, int64_t j, const BaseAllFeature &baseAllFeature, const std::vector<float> &retBase,
    const std::vector<int64_t> &indices = std::vector<int64_t>())
{
    float gap = 0.0005; // 0.0005 f32 fp16 gap
    if (indices.empty()) {
        if (!baseAllFeature.scale.empty()) {
            // 计算量化误差
            gap = CalculateQuantizationError(baseAllFeature.features[i * dim + j], baseAllFeature.scale[j]);
        }
        // 比较特征数据是否相同
        if (fabs(baseAllFeature.features[i * dim + j] - retBase[i * dim + j]) > gap) {
            printf("i:[%ld], j:[%ld], [%f] vs [%f] is different, gap:%f\n", i, j,
                baseAllFeature.features[i * dim + j], retBase[i * dim + j], gap);
            if (!baseAllFeature.scale.empty()) {
                printf("scale[%ld]:%f\n", j, baseAllFeature.scale[j]);
            }
            return false;
        }
    } else {
        if (!baseAllFeature.scale.empty()) {
            // 计算量化误差
            gap = CalculateQuantizationError(baseAllFeature.features[indices[i] * dim + j], baseAllFeature.scale[j]);
        }
        // 比较特征数据是否相同
        if (fabs(baseAllFeature.features[indices[i] * dim + j] - retBase[i * dim + j]) > gap) {
            printf("i:[%ld], indices[%ld]:[%ld] j:[%ld], [%f] vs [%f] is different, gap:%f\n", i, i, indices[i], j,
                baseAllFeature.features[indices[i] * dim + j], retBase[i * dim + j], gap);
            if (!baseAllFeature.scale.empty()) {
                printf("scale[%ld]:%f\n", j, baseAllFeature.scale[j]);
            }
            return false;
        }
    }
    return true;
}

void TestFeatureByRange(int64_t featureNum, int64_t maskLen, std::shared_ptr<faiss::ascend::AscendIndexTS> &tsIndex,
    const BaseAllFeature &baseAllFeature)
{
    std::vector<float> retBase(featureNum * dim);
    std::vector<int64_t> retLabels(featureNum);
    std::vector<faiss::ascend::FeatureAttr> retAttrs(featureNum);
    std::vector<faiss::ascend::ExtraValAttr> retValAttrs(featureNum);
    auto res = tsIndex->GetBaseByRangeWithExtraVal(0, featureNum, retLabels.data(),
        retBase.data(), retAttrs.data(), retValAttrs.data());
    ASSERT_EQ(res, 0);

    std::vector<uint8_t> baseMask(maskLen); // 按8对齐
    res = tsIndex->GetBaseMask(maskLen, baseMask.data());
    ASSERT_EQ(res, 0);
    for (int64_t i = 0; i < featureNum; i++) {
        ASSERT_EQ(i, retLabels[i]);
        for (int64_t j = 0; j < dim; j++) {
            ASSERT_TRUE(CompareFeatureData(i, j, baseAllFeature, retBase));
        }
        ASSERT_EQ(baseAllFeature.attrs[i].time, retAttrs[i].time);
        ASSERT_EQ(baseAllFeature.attrs[i].tokenId, retAttrs[i].tokenId);
    }
    for (int64_t i = 0; i < maskLen; i++) {
        ASSERT_EQ(baseAllFeature.baseMask[i], baseMask[i]);
    }

    TestCustomAttr(featureNum, tsIndex, baseAllFeature);
}

void TestFeatureByIndice(int64_t featureNum, int64_t maskLen, std::shared_ptr<faiss::ascend::AscendIndexTS> &tsIndex,
    const BaseAllFeature &baseAllFeature)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine e(seed);
    std::vector<int64_t> labels(featureNum);
    std::shuffle(labels.begin(), labels.end(), e);

    // 随机选择底库对比
    int64_t compareNum = featureNum > 100 ? 100 : featureNum;
    std::vector<float> retBase(compareNum * dim);
    std::vector<int64_t> retLabels(compareNum);
    std::vector<faiss::ascend::FeatureAttr> retAttrs(compareNum);
    std::vector<faiss::ascend::ExtraValAttr> retValAttrs(compareNum);

    std::vector<int64_t> indices(labels.begin(), labels.begin() + compareNum);
    auto res = tsIndex->GetFeatureByIndice(compareNum, indices.data(), retLabels.data(),
        retBase.data(), retAttrs.data(), retValAttrs.data());
    ASSERT_EQ(res, 0);
    for (int64_t i = 0; i < compareNum; i++) {
        ASSERT_EQ(indices[i], retLabels[i]);
        for (int64_t j = 0; j < dim; j++) {
            ASSERT_TRUE(CompareFeatureData(i, j, baseAllFeature, retBase, indices));
        }
        ASSERT_EQ(baseAllFeature.attrs[indices[i]].time, retAttrs[i].time);
        ASSERT_EQ(baseAllFeature.attrs[indices[i]].tokenId, retAttrs[i].tokenId);
    }
}

void TestBaseAllFeature(std::shared_ptr<faiss::ascend::AscendIndexTS> &tsIndex, const BaseAllFeature &baseAllFeature)
{
    int64_t featureNum = 0;
    auto res = tsIndex->GetFeatureNum(&featureNum);
    ASSERT_EQ(res, 0);
    ASSERT_EQ(baseAllFeature.ntotal, featureNum);
    int64_t maskLen = (featureNum + MASK_ALIGN - 1) / MASK_ALIGN;
    ASSERT_EQ(static_cast<int64_t>(baseAllFeature.features.size()), featureNum * dim);
    ASSERT_EQ(static_cast<int64_t>(baseAllFeature.attrs.size()), featureNum);
    ASSERT_EQ(static_cast<int64_t>(baseAllFeature.baseMask.size()), maskLen);

    printf("compare by base feature by range\n");
    TestFeatureByRange(featureNum, maskLen, tsIndex, baseAllFeature);

    printf("compare by base feature by indices\n");
    TestFeatureByIndice(featureNum, maskLen, tsIndex, baseAllFeature);
}

void TestFastDeleteByIndices(int64_t deleteOffset, int64_t deleteNum, std::vector<int64_t> &deleteIndices,
    std::shared_ptr<faiss::ascend::AscendIndexTS> &tsIndex, BaseAllFeature &baseAllFeature)
{
    int64_t featureNumBak = 0;
    auto res = tsIndex->GetFeatureNum(&featureNumBak);
    ASSERT_EQ(res, 0);

    // 随机从deleteOffset开始删除deleteNum个
    deleteIndices.resize(deleteNum);
    std::iota(deleteIndices.begin(), deleteIndices.end(), deleteOffset);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine e(seed);
    std::shuffle(deleteIndices.begin(), deleteIndices.end(), e);
    res = tsIndex->FastDeleteFeatureByIndice(static_cast<int64_t>(deleteIndices.size()), deleteIndices.data());
    ASSERT_EQ(res, 0);

    // featureNum不变
    int64_t newFeatureNum = 0;
    res = tsIndex->GetFeatureNum(&newFeatureNum);
    ASSERT_EQ(res, 0);
    ASSERT_EQ(newFeatureNum, featureNumBak);

    // 检查baseMask删除部分是不是设置为0
    int64_t maskLength = (newFeatureNum + MASK_ALIGN - 1) / MASK_ALIGN;
    std::vector<uint8_t> baseMask(maskLength); // 按8对齐
    res = tsIndex->GetBaseMask(maskLength, baseMask.data());
    ASSERT_EQ(res, 0);

    std::unordered_set<int64_t> deleteInidcesSet(deleteIndices.begin(), deleteIndices.end());
    for (int64_t i = deleteOffset; i < newFeatureNum; i++) {
        int64_t byteIndex = i / MASK_ALIGN;
        int64_t bitIndex = i % MASK_ALIGN;
        // 被删除的位置对应的bit需要为0，否则为1
        if (deleteInidcesSet.count(i)) {
            ASSERT_EQ(baseMask[byteIndex] & (1 << bitIndex), 0);
        } else {
            ASSERT_NE(baseMask[byteIndex] & (1 << bitIndex), 0);
        }
    }

    baseAllFeature.baseMask = std::move(baseMask);
    TestBaseAllFeature(tsIndex, baseAllFeature);
}

void TestFastDeleteByRange(int64_t deleteOffset, int64_t deleteNum,
    std::shared_ptr<faiss::ascend::AscendIndexTS> &tsIndex, BaseAllFeature &baseAllFeature)
{
    int64_t featureNumBak = 0;
    auto res = tsIndex->GetFeatureNum(&featureNumBak);
    ASSERT_EQ(res, 0);

    // 从deleteOffset开始删除deleteNum个
    res = tsIndex->FastDeleteFeatureByRange(deleteOffset, deleteNum);
    ASSERT_EQ(res, 0);

    // 检查featureNum不变
    int64_t newFeatureNum = 0;
    res = tsIndex->GetFeatureNum(&newFeatureNum);
    ASSERT_EQ(res, 0);
    ASSERT_EQ(newFeatureNum, featureNumBak);

    // 检查baseMask删除部分是不是设置为0
    int64_t maskLength = (newFeatureNum + MASK_ALIGN - 1) / MASK_ALIGN;
    std::vector<uint8_t> baseMask(maskLength); // 按8对齐
    res = tsIndex->GetBaseMask(maskLength, baseMask.data());
    ASSERT_EQ(res, 0);
    for (int64_t i = deleteOffset; i < deleteOffset + deleteNum; i++) {
        int64_t byteIndex = i / MASK_ALIGN;
        int64_t bitIndex = i % MASK_ALIGN;
        ASSERT_EQ(baseMask[byteIndex] & (1 << bitIndex), 0);
    }

    baseAllFeature.baseMask = std::move(baseMask);
    TestBaseAllFeature(tsIndex, baseAllFeature);
}

void TestReplaceAdd(int64_t replaceNum, std::vector<int64_t> &replaceIndices, int64_t addNum, int64_t ntotal,
    std::shared_ptr<faiss::ascend::AscendIndexTS> &tsIndex, BaseAllFeature &baseAllFeature)
{
    // 替换和新增的一起
    std::vector<int64_t> indices(replaceNum + addNum);

    // 替换需要是升序的
    std::sort(replaceIndices.begin(), replaceIndices.begin() + replaceNum);

    // 替换部分在前面，新增部分在后面
    std::copy(replaceIndices.begin(), replaceIndices.begin() + replaceNum, indices.begin());

    // 新增部分的值需要从ntotal开始连续递增
    std::iota(indices.begin() + replaceNum, indices.end(), ntotal);
    ASSERT_EQ(indices.back(), ntotal + addNum - 1);

    TestAdd(indices, tsIndex, baseAllFeature);

    TestBaseAllFeature(tsIndex, baseAllFeature);
}

void TestAddAndGet(std::shared_ptr<faiss::ascend::AscendIndexTS> &tsIndex, BaseAllFeature &baseAllFeature)
{
    try {
        // 先增加addNum个
        int64_t replaceNum = 0;
        int64_t addNum = 1000000;
        std::vector<int64_t> replaceIndices; // 要替换的indice，初始没有需要替换的，为空
        printf("------------1 add num %ld\n", addNum);
        TestReplaceAdd(replaceNum, replaceIndices, addNum, baseAllFeature.ntotal, tsIndex, baseAllFeature);

        // 删除底库
        int64_t deleteOffset = 0;
        int64_t deleteNum = addNum;
        printf("------------2 delete form %ld to %ld\n", deleteOffset, deleteOffset + deleteNum);
        std::vector<int64_t> deleteIndices;
        TestFastDeleteByIndices(deleteOffset, deleteNum, deleteIndices, tsIndex, baseAllFeature);
        ASSERT_EQ(deleteIndices.size(), static_cast<size_t>(deleteNum));

        // 将上面删除的位置重新替换成新的，再额外添加addNum个
        replaceNum = deleteNum;
        printf("------------3 replace num %ld, add num %ld\n", replaceNum, addNum);
        TestReplaceAdd(replaceNum, deleteIndices, addNum, baseAllFeature.ntotal, tsIndex, baseAllFeature);

        // 循环执行，先删除一些老的，再替换并新增一些
        int64_t testStep = 10000;
        for (int i = 0; i < 100; i++) {
            // 删除一片连续的位置
            deleteOffset = i * testStep + 1;
            deleteNum = testStep + 1;
            printf("------------%d base total %ld, delete form %ld to %ld\n",
                i * 2 + 4, baseAllFeature.ntotal, deleteOffset, deleteOffset + deleteNum);
            TestFastDeleteByRange(deleteOffset, deleteNum, tsIndex, baseAllFeature);

            // 将上面删除的位置重新替换成新的，再额外添加deleteNum个
            deleteIndices.resize(deleteNum);
            std::iota(deleteIndices.begin(), deleteIndices.end(), deleteOffset);
            printf("------------%d base total %ld, replace %ld [%ld to %ld], add num %ld\n",
                i * 2 + 5, baseAllFeature.ntotal, deleteNum, deleteOffset, deleteOffset + deleteNum, testStep);
            TestReplaceAdd(deleteNum, deleteIndices, testStep, baseAllFeature.ntotal, tsIndex, baseAllFeature);
        }
    } catch (std::exception &e) {
        printf("catch exception %s\n", e.what());
    }
}

TEST(TestAscendIndexTSFlatIP, AddDeleteByIndice)
{
    try {
        auto tsIndex = std::make_shared<faiss::ascend::AscendIndexTS>();
        auto res = tsIndex->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_IP_FP16,
            faiss::ascend::MemoryStrategy::PURE_DEVICE_MEMORY, g_customAttrLen, g_customAttrBlockSize);
        ASSERT_EQ(res, 0);

        BaseAllFeature baseAllFeature;
        TestAddAndGet(tsIndex, baseAllFeature);
    } catch (std::exception &e) {
        printf("catch exception %s\n", e.what());
    }
}

TEST(TestAscendIndexTSFlatIP, AddQuantify)
{
    try {
        std::vector<float> scale(dim);
        ScaleGenerator(scale, 100, 200);
        BaseAllFeature baseAllFeature;
        baseAllFeature.scale = scale;

        // 最大4GB，如果默认的话设置为0x60000000(1.5G)
        uint64_t resourceSize = static_cast<uint64_t>(4) * 1024 * 1024 * 1024;
        auto tsIndex = std::make_shared<faiss::ascend::AscendIndexTS>();
        auto res = tsIndex->InitWithQuantify(deviceId, dim, tokenNum, resourceSize, scale.data(),
            faiss::ascend::AlgorithmType::FLAT_IP_FP16, g_customAttrLen, g_customAttrBlockSize);
        ASSERT_EQ(res, 0);

        TestAddAndGet(tsIndex, baseAllFeature);
    } catch (std::exception &e) {
        printf("catch exception %s\n", e.what());
    }
}

TEST(TestAscendIndexTSFlatIP, AddDeleteByIndiceAndSearch)
{
    try {
        auto tsIndex = std::make_shared<faiss::ascend::AscendIndexTS>();
        auto res = tsIndex->Init(deviceId, dim, tokenNum, faiss::ascend::AlgorithmType::FLAT_IP_FP16,
            faiss::ascend::MemoryStrategy::PURE_DEVICE_MEMORY);
        ASSERT_EQ(res, 0);

        int64_t addNum = 200000;
        std::vector<float> features(addNum * dim);
        FeatureGenerator(features);
        Norm(features.data(), addNum, dim);

        std::vector<FeatureAttr> attrs(addNum);
        FeatureAttrGenerator(attrs);

        int64_t featureNum = 0;
        res = tsIndex->GetFeatureNum(&featureNum);
        EXPECT_EQ(res, 0);
        EXPECT_EQ(featureNum, 0);

        std::vector<int64_t> indices(addNum);
        std::iota(indices.begin(), indices.end(), 0);
        EXPECT_EQ(res, 0);
        // 增加一些
        for (int64_t i = 0; i < addNum; i += SINGLE_ADD_LIMIT) {
            int64_t addCount = std::min(SINGLE_ADD_LIMIT, addNum - i);
            double ts = GetMillisecs();
            res = tsIndex->AddFeatureByIndice(
                addCount, features.data() + i * dim, attrs.data() + i, indices.data() + i, nullptr, nullptr);
            ASSERT_EQ(res, 0);
            double te = GetMillisecs();
            printf("add %ld cost %f ms\n", addCount, te - ts);
        }

        res = tsIndex->GetFeatureNum(&featureNum);
        EXPECT_EQ(res, 0);
        EXPECT_EQ(featureNum, addNum);

        int64_t queryNum = 1;
        std::vector<float> querys(queryNum * dim);
        querys.assign(features.begin() + dim, features.begin() + 2 * dim); // 第1条

        int64_t ntotalPad = (featureNum + 16 - 1) / 16 * 16; // 长度等于底库长度按照16对齐
        float extraScoreValue = 1.0;
        std::vector<float> extraScore(queryNum * ntotalPad, extraScoreValue);
        std::vector<uint16_t> extraScoreFp16(extraScore.size());
        std::transform(extraScore.begin(), extraScore.end(), extraScoreFp16.begin(),
            [] (float tmp) { return aclFloatToFloat16(tmp); });

        uint32_t setlen = (uint32_t)((tokenNum + 7) / 8);
        std::vector<uint8_t> bitSet(setlen, 0);

        // 00000010
        bitSet[0] = 0x02;
        AttrFilter filter {};
        filter.timesStart = 0;
        filter.timesEnd = 4;
        filter.tokenBitSet = bitSet.data();
        filter.tokenBitSetLen = setlen;

        std::vector<AttrFilter> queryFilters(queryNum, filter);
        int64_t extraMaskLen = (featureNum + 8 - 1) / 8;
        std::vector<uint8_t> extraMask(queryNum * extraMaskLen, 255);
        int k = 5;

        // 按照token过滤进行检索，找到位置1的数据
        std::vector<float> distances(static_cast<int64_t>(queryNum) * k, -1);
        std::vector<int64_t> labelRes(static_cast<int64_t>(queryNum) * k, 10); // 10, default value
        std::vector<uint32_t> validnum(static_cast<int64_t>(queryNum), 0);
        res = tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), false, k,
            extraMask.data(), extraMaskLen, false, extraScoreFp16.data(), labelRes.data(), distances.data(),
            validnum.data());
        EXPECT_EQ(res, 0);

        PrintResult(queryNum, k, distances, labelRes, validnum);

        // 删除位置1的数据
        std::vector<int64_t> deleteIndices(1);
        deleteIndices[0] = 1;
        res = tsIndex->FastDeleteFeatureByIndice(static_cast<int64_t>(deleteIndices.size()), deleteIndices.data());
        EXPECT_EQ(res, 0);

        // 还是检索位置1的，应该找不到
        res = tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), false, k,
            extraMask.data(), extraMaskLen, false, extraScoreFp16.data(), labelRes.data(), distances.data(),
            validnum.data());
        EXPECT_EQ(res, 0);
        PrintResult(queryNum, k, distances, labelRes, validnum);

        // 位置1上重新添加新的特征和token值
        std::vector<int64_t> newIndices(1);
        newIndices[0] = 1;

        addNum = 1;
        std::vector<float> newFeatures(addNum * dim);
        FeatureGenerator(newFeatures);
        Norm(newFeatures.data(), addNum, dim);
        std::vector<FeatureAttr> newAttrs(addNum);
        newAttrs[0].time = 1;
        newAttrs[0].tokenId = 20;

        res = tsIndex->AddFeatureByIndice(
                addNum, newFeatures.data(), newAttrs.data(), newIndices.data(), nullptr, nullptr);
        EXPECT_EQ(res, 0);

        // 把位置1的数据拿出来
        int64_t compareNum = 1;
        std::vector<float> retBase(compareNum * dim);
        std::vector<int64_t> retLabels(compareNum);
        std::vector<faiss::ascend::FeatureAttr> retAttrs(compareNum);
        std::vector<faiss::ascend::ExtraValAttr> retValAttrs(compareNum);

        res = tsIndex->GetFeatureByIndice(compareNum, newIndices.data(), retLabels.data(),
            retBase.data(), retAttrs.data(), retValAttrs.data());
        EXPECT_EQ(res, 0);
        printf("retLabels[0]: %ld\n", retLabels[0]);
        printf("retAttrs[0]: {time:%d tokenId:%u}\n", retAttrs[0].time, retAttrs[0].tokenId);

        // token = 20, 对应00010000, 00000000, 00000000
        bitSet[0] = 0; // 00000000
        bitSet[1] = 0; // 00000000
        bitSet[2] = 16; // 00010000
        filter.timesStart = 0;
        filter.timesEnd = 4;
        filter.tokenBitSet = bitSet.data();
        filter.tokenBitSetLen = setlen;
        // 还是检索位置1的上新的特征，token为100
        res = tsIndex->SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), false, k,
            extraMask.data(), extraMaskLen, false, extraScoreFp16.data(), labelRes.data(), distances.data(),
            validnum.data());
        EXPECT_EQ(res, 0);
        PrintResult(queryNum, k, distances, labelRes, validnum);
    } catch (std::exception &e) {
        printf("catch exception %s\n", e.what());
    }
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
