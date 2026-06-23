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

// 需要生成aicpu算子+int8flat算子(-d 512)+mask算子

#include <faiss/ascend/AscendIndexTS.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sys/time.h>
#include <unistd.h>

#include <bitset>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <functional>
#include <queue>
#include <random>
#include <vector>

namespace
{
using idx_t = int64_t;
using FeatureAttr = faiss::ascend::FeatureAttr;
using AttrFilter = faiss::ascend::AttrFilter;

const int BITS = 8;
const int SEED = 1;
const uint32_t TOKEN_NUM = 2500;
const int DEVICE_ID = 0;
const int OFFSET = 128;
const int MILLI_SECOND = 1000;
std::independent_bits_engine<std::mt19937, BITS, uint8_t> engine(SEED);

void FeatureGenerator(std::vector<int8_t> &features)
{
    size_t n = features.size();
    for (size_t i = 0; i < n; ++i)
    {
        features[i] = engine() - OFFSET;
    }
}

void FeatureAttrGenerator(std::vector<FeatureAttr> &attrs)
{
    size_t n = attrs.size();
    int power = 4;
    for (size_t i = 0; i < n; ++i)
    {
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

void InitAndAdd(faiss::ascend::AscendIndexTS &tsIndex, int ntotal, int addNum, int dim, std::vector<int8_t> &features)
{
    auto ret = tsIndex.Init(DEVICE_ID, dim, TOKEN_NUM, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(ret, 0);

    for (int i = 0; i < addNum; i++)
    {
        std::vector<int64_t> labels;
        for (int64_t j = 0; j < ntotal; ++j)
        {
            labels.emplace_back(j + i * ntotal);
        }
        std::vector<FeatureAttr> attrs(ntotal);
        FeatureAttrGenerator(attrs);
        auto ts0 = GetMillisecs();
        tsIndex.AddFeature(ntotal, features.data(), attrs.data(), labels.data());
        auto te0 = GetMillisecs();
        printf("add %d cost %f ms\n", ntotal, te0 - ts0);
    }
}

void CheckResult(int queryNum, int k, std::vector<float> &distances, std::vector<int64_t> &labelRes)
{
    for (int i = 0; i < queryNum; i++)
    {
        // 根据过滤条件，时间为0-3，在4个时间属性中过滤1来判断精度
        if (i % 4 == 1)
        {
            ASSERT_TRUE(labelRes[i * k] == i);
            ASSERT_TRUE(distances[i * k] > float(0.99) &&
                        distances[i * k] < float(1.01));  // 过滤掉的期望结果在(0.99,1.01)
        }
        else
        {
            ASSERT_TRUE(labelRes[i * k] != i);
            ASSERT_TRUE(distances[i * k] <= float(0.3));  // 对于期望dis值小于等于0.3
        }
    }
}

}  // end of namespace

TEST(TestAscendIndexTS_int8Cos, Init)
{
    uint32_t dim = 512;
    auto ts = GetMillisecs();
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    int res = tsIndex->Init(DEVICE_ID, dim, TOKEN_NUM, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(res, 0);
    auto te = GetMillisecs();
    printf("init cost %f ms\n", te - ts);
    delete tsIndex;
}

TEST(TestAscendIndexTS_int8Cos, add)
{
    idx_t ntotal = 1000000;
    uint32_t dim = 512;
    faiss::ascend::AscendIndexTS *tsIndex = new faiss::ascend::AscendIndexTS();
    tsIndex->Init(DEVICE_ID, dim, TOKEN_NUM, faiss::ascend::AlgorithmType::FLAT_COS_INT8);

    std::vector<int8_t> features(ntotal * dim);
    printf("[---add-----------]\n");
    FeatureGenerator(features);
    std::vector<int64_t> labels;
    for (int i = 0; i < ntotal; ++i)
    {
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

TEST(TestAscendIndexTS_int8Cos, GetFeatureByLabel)
{
    int dim = 512;
    int ntotal = 100000;
    std::vector<int8_t> base(ntotal * dim);
    FeatureGenerator(base);
    std::vector<int64_t> label(ntotal);
    std::iota(label.begin(), label.end(), 0);
    std::vector<FeatureAttr> attrs(ntotal);
    FeatureAttrGenerator(attrs);
    auto *index = new faiss::ascend::AscendIndexTS();
    auto ret = index->Init(DEVICE_ID, dim, TOKEN_NUM, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(ret, 0);
    ret = index->AddFeature(ntotal, base.data(), attrs.data(), label.data());
    EXPECT_EQ(ret, 0);
    std::vector<int8_t> getBase(ntotal * dim);
    auto ts = GetMillisecs();
    ret = index->GetFeatureByLabel(ntotal, label.data(), getBase.data());
    auto te = GetMillisecs();
    printf("GetFeatureByLabel cost total %f ms\n", te - ts);
    EXPECT_EQ(ret, 0);

#pragma omp parallel for if (ntotal > 100)
    for (int i = 0; i < ntotal * dim; i++)
    {
        EXPECT_EQ(base[i], getBase[i]);
    }
    delete index;
}

TEST(TestAscendIndexTS_int8Cos, DeleteFeatureByLabel)
{
    int dim = 512;
    int ntotal = 1000000;
    std::vector<int8_t> base(ntotal * dim);
    FeatureGenerator(base);
    std::vector<int64_t> label(ntotal);
    std::iota(label.begin(), label.end(), 0);
    std::vector<FeatureAttr> attrs(ntotal);
    FeatureAttrGenerator(attrs);
    auto *index = new faiss::ascend::AscendIndexTS();
    auto ret = index->Init(DEVICE_ID, dim, TOKEN_NUM, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(ret, 0);
    ret = index->AddFeature(ntotal, base.data(), attrs.data(), label.data());
    EXPECT_EQ(ret, 0);
    int64_t validNum = 0;
    index->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal);
    int delCount = 1000;
    std::vector<int64_t> delLabel(delCount);
    delLabel.assign(label.begin(), label.begin() + delCount);
    auto ts = GetMillisecs();
    index->DeleteFeatureByLabel(delCount, delLabel.data());
    auto te = GetMillisecs();
    printf("DeleteFeatureByLabel delete cost totoal %f ms\n", te - ts);
    index->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal - delCount);

    index->DeleteFeatureByLabel(delCount, delLabel.data());
    index->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal - delCount);
    delete index;
}

TEST(TestAscendIndexTS_int8Cos, DeleteFeatureByToken)
{
    int dim = 512;
    int ntotal = 1000000;
    std::vector<int8_t> base(ntotal * dim);
    FeatureGenerator(base);
    std::vector<int64_t> label(ntotal);
    std::iota(label.begin(), label.end(), 0);
    std::vector<FeatureAttr> attrs(ntotal);
    FeatureAttrGenerator(attrs);
    auto *index = new faiss::ascend::AscendIndexTS();
    auto ret = index->Init(DEVICE_ID, dim, TOKEN_NUM, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(ret, 0);
    ret = index->AddFeature(ntotal, base.data(), attrs.data(), label.data());
    EXPECT_EQ(ret, 0);
    int64_t validNum = 0;
    index->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal);
    std::vector<uint32_t> delToken{0, 1};
    auto ts = GetMillisecs();
    index->DeleteFeatureByToken(2, delToken.data());
    auto te = GetMillisecs();
    printf("DeleteFeatureByToken delete cost totoal %f ms\n", te - ts);
    index->GetFeatureNum(&validNum);
    EXPECT_EQ(validNum, ntotal / 2);
    delete index;
}

TEST(TestAscendIndexTS_int8Cos, Acc)
{
    idx_t ntotal = 1000000;
    uint32_t addNum = 1;
    uint32_t dim = 512;
    std::vector<int> queryNums = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    int k = 10;
    faiss::ascend::AscendIndexTS tsIndex;
    std::vector<int8_t> features(ntotal * dim);
    FeatureGenerator(features);
    InitAndAdd(tsIndex, ntotal, addNum, dim, features);

    int loopTimes = 2;
    for (auto queryNum : queryNums)
    {
        std::vector<float> distances(queryNum * k, -1);
        std::vector<int64_t> labelRes(queryNum * k, 10);
        std::vector<uint32_t> validnum(queryNum, 0);
        uint32_t size = queryNum * dim;
        std::vector<uint8_t> querys(size);
        querys.assign(features.begin(), features.begin() + size);

        uint32_t setlen = (uint32_t)(((TOKEN_NUM + 7) / 8));
        std::vector<uint8_t> bitSet(setlen, 0);
        bitSet[0] = 0x1 << 0 | 0x1 << 1 | 0x1 << 2 | 0x1 << 3;
        AttrFilter filter{};
        filter.timesStart = 0;
        filter.timesEnd = 3;
        filter.tokenBitSet = bitSet.data();
        filter.tokenBitSetLen = setlen;

        std::vector<AttrFilter> queryFilters(queryNum, filter);
        for (int i = 0; i < loopTimes; i++)
        {
            tsIndex.Search(queryNum, querys.data(), queryFilters.data(), false, k, labelRes.data(), distances.data(),
                           validnum.data());
        }
        for (int i = 0; i < queryNum; i++)
        {
            ASSERT_TRUE(labelRes[i * k] == i);
            ASSERT_TRUE(distances[i * k] > float(0.99) && distances[i * k] < float(1.01));
        }

        bitSet[0] = 0x1 << 0 | 0x1 << 1;
        filter.timesStart = 1;
        filter.timesEnd = 3;

        queryFilters.clear();
        queryFilters.insert(queryFilters.begin(), queryNum, filter);
        for (int i = 0; i < loopTimes; i++)
        {
            tsIndex.Search(queryNum, querys.data(), queryFilters.data(), false, k, labelRes.data(), distances.data(),
                           validnum.data());
        }
        CheckResult(queryNum, k, distances, labelRes);
    }
}

TEST(TestAscendIndexTS_int8Cos, SearchNoShareQPS)
{
    idx_t ntotal = 1000000;
    uint32_t addNum = 10;
    uint32_t dim = 512;
    std::vector<int> queryNums = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    int k = 10;
    faiss::ascend::AscendIndexTS tsIndex;
    std::vector<int8_t> features(ntotal * dim);
    FeatureGenerator(features);
    InitAndAdd(tsIndex, ntotal, addNum, dim, features);

    long double ts{0.};
    long double te{0.};

    int warmupTimes = 3;
    int loopTimes = 2;
    for (auto queryNum : queryNums)
    {
        std::vector<float> distances(queryNum * k, -1);
        std::vector<int64_t> labelRes(queryNum * k, -1);
        std::vector<uint32_t> validnum(queryNum, 0);
        uint32_t size = queryNum * dim;
        std::vector<uint8_t> querys(size);
        querys.assign(features.begin(), features.begin() + size);

        uint32_t setlen = (uint32_t)(((TOKEN_NUM + 7) / 8));
        std::vector<uint8_t> bitSet(setlen, 0);
        bitSet[0] = 0x1 << 0 | 0x1 << 1 | 0x1 << 2;

        AttrFilter filter{};
        filter.timesStart = 0;
        filter.timesEnd = 100;
        filter.tokenBitSet = bitSet.data();
        filter.tokenBitSetLen = setlen;

        std::vector<AttrFilter> queryFilters(queryNum, filter);
        for (int i = 0; i < loopTimes + warmupTimes; i++)
        {
            if (i == warmupTimes)
            {
                ts = GetMillisecs();
            }
            tsIndex.Search(queryNum, querys.data(), queryFilters.data(), false, k, labelRes.data(), distances.data(),
                           validnum.data());
        }
        te = GetMillisecs();

        printf("base: %ld, dim: %d, batch: %4d, top%d, QPS:%7.2Lf\n", ntotal * addNum, dim, queryNum, k,
               MILLI_SECOND * queryNum * loopTimes / (te - ts));
    }
}

TEST(TestAscendIndexTS_int8Cos, SearchShareQPS)
{
    idx_t ntotal = 1000000;
    uint32_t addNum = 10;
    uint32_t dim = 512;
    std::vector<int> queryNums = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    int k = 10;
    faiss::ascend::AscendIndexTS tsIndex;
    std::vector<int8_t> features(ntotal * dim);
    FeatureGenerator(features);
    InitAndAdd(tsIndex, ntotal, addNum, dim, features);

    long double ts{0.};
    long double te{0.};

    int warmupTimes = 3;
    int loopTimes = 2;
    for (auto queryNum : queryNums)
    {
        std::vector<float> distances(queryNum * k, -1);
        std::vector<int64_t> labelRes(queryNum * k, -1);
        std::vector<uint32_t> validnum(queryNum, 1);
        uint32_t size = queryNum * dim;
        std::vector<int8_t> querys(size);
        querys.assign(features.begin(), features.begin() + size);

        uint32_t setlen = (uint32_t)(((TOKEN_NUM + 7) / 8));
        std::vector<uint8_t> bitSet(setlen, 0);

        bitSet[0] = 0x1 << 0 | 0x1 << 1 | 0x1 << 2;

        AttrFilter filter{};
        filter.timesStart = 0;
        filter.timesEnd = 100;
        filter.tokenBitSet = bitSet.data();
        filter.tokenBitSetLen = setlen;

        std::vector<AttrFilter> queryFilters(queryNum, filter);
        for (int i = 0; i < loopTimes + warmupTimes; i++)
        {
            if (i == warmupTimes)
            {
                ts = GetMillisecs();
            }
            tsIndex.Search(queryNum, querys.data(), queryFilters.data(), true, k, labelRes.data(), distances.data(),
                           validnum.data());
        }
        te = GetMillisecs();
        printf("base: %ld, dim: %d, batch: %4d, top%d, QPS:%7.2Lf\n", ntotal * addNum, dim, queryNum, k,
               MILLI_SECOND * queryNum * loopTimes / (te - ts));
    }
}

TEST(TestAscendIndexTS_int8Cos, SearchShareWithExtraQPS)
{
    idx_t ntotal = 1000000;
    uint32_t addNum = 10;
    uint32_t dim = 512;
    std::vector<int> queryNums = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    int k = 10;
    faiss::ascend::AscendIndexTS tsIndex;
    std::vector<int8_t> features(ntotal * dim);
    FeatureGenerator(features);
    InitAndAdd(tsIndex, ntotal, addNum, dim, features);

    long double ts{0.};
    long double te{0.};

    int warmupTimes = 3;
    int loopTimes = 10;
    for (auto queryNum : queryNums)
    {
        std::vector<float> distances(queryNum * k, -1);
        std::vector<int64_t> labelRes(queryNum * k, -1);
        std::vector<uint32_t> validnum(queryNum, 1);
        uint32_t size = queryNum * dim;
        std::vector<int8_t> querys(size);
        querys.assign(features.begin(), features.begin() + size);

        uint32_t setlen = (uint32_t)(((TOKEN_NUM + 7) / 8));
        std::vector<uint8_t> bitSet(setlen, ~0);
        //  00000111  means index lable  0,1,2 has been chosed;
        // bitSet[0] = 0x1 << 0 | 0x1 << 1 | 0x1 << 2;

        AttrFilter filter{};
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

        std::vector<AttrFilter> queryFilters(queryNum, filter);
        for (int i = 0; i < loopTimes + warmupTimes; i++)
        {
            if (i == warmupTimes)
            {
                ts = GetMillisecs();
            }
            tsIndex.SearchWithExtraMask(queryNum, querys.data(), queryFilters.data(), true, k, extraMask.data(),
                                        extraMaskLen, false, labelRes.data(), distances.data(), validnum.data());
        }
        te = GetMillisecs();

        printf("base: %ld, dim: %d, batch: %4d, top%d, QPS:%7.2Lf\n", ntotal * addNum, dim, queryNum, k,
               MILLI_SECOND * queryNum * loopTimes / (te - ts));
    }
}

TEST(TestAscendIndexTS_int8Cos, Int8AccVal)
{
    idx_t ntotal = 10000;
    uint32_t addNum = 2;
    uint32_t deviceId = 0;
    uint32_t dim = 1024;
    uint32_t tokenNum = 300000;
    std::vector<int> queryNums = {1};
    std::vector<int> topks = {32};
    uint64_t resources = 1024 * 1024 * 1024;
    faiss::ascend::AscendIndexTS tsIndex;

    auto ret =
        tsIndex.InitWithExtraVal(deviceId, dim, tokenNum, resources, faiss::ascend::AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(ret, 0);

    std::vector<int8_t> features(ntotal * dim);
    FeatureGeneratorInt8(features);
    std::vector<FeatureAttr> attrs(addNum * ntotal);
    std::vector<ExtraValAttr> valAttrs(addNum * ntotal);
    FeatureAttrGenerator(attrs);
    ExtraValAttrGenerator(valAttrs);
    std::vector<int64_t> labels;
    for (size_t i = 0; i < addNum; i++)
    {
        labels.clear();
        for (int64_t j = 0; j < ntotal; ++j)
        {
            labels.emplace_back(j + i * ntotal);
        }
        ret = tsIndex.AddWithExtraVal(ntotal, features.data(), attrs.data() + i * ntotal, labels.data(),
                                      valAttrs.data() + i * ntotal);
        EXPECT_EQ(ret, 0);
    }
    for (auto k : topks)
    {
        int warmupTimes = 0;
        int loopTimes = 1;
        for (auto queryNum : queryNums)
        {
            std::vector<float> distances(queryNum * k, -1);
            std::vector<int64_t> labelRes(queryNum * k, 10);
            std::vector<uint32_t> validnum(queryNum, 0);
            uint32_t size = queryNum * dim;
            std::vector<uint8_t> querys(size);
            querys.assign(features.begin(), features.begin() + size);

            uint32_t setlen = (uint32_t)(((tokenNum + 7) / 8));
            std::vector<uint8_t> bitSet(setlen, 255);
            AttrFilter filter{};
            filter.timesStart = 0;
            filter.timesEnd = 3;
            filter.tokenBitSet = bitSet.data();
            filter.tokenBitSetLen = setlen;

            ExtraValFilter valFilter{};

            valFilter.filterVal = 99;
            valFilter.matchVal = 0;

            std::vector<AttrFilter> queryFilters(queryNum, filter);
            std::vector<ExtraValFilter> queryValFilters(queryNum, valFilter);

            ret = tsIndex.SearchWithExtraVal(queryNum, querys.data(), queryFilters.data(), false, k, labelRes.data(),
                                             distances.data(), validnum.data(), queryValFilters.data());
            EXPECT_EQ(ret, 0);

            bool flag = false;
            auto mask = 0;
            for (int i = 0; i < queryNum; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    auto cid = labelRes[j + i * k];
                    if (queryValFilters[i].matchVal == 0)
                    {
                        mask = ((queryValFilters[i].filterVal & valAttrs[cid].val) == queryValFilters[i].filterVal) ? 1
                                                                                                                    : 0;
                    }
                    else
                    {
                        mask = ((queryValFilters[i].filterVal & valAttrs[cid].val) > 0) ? 1 : 0;
                    }
                    if (mask == 0)
                    {
                        printf("cid = %ld \n", cid);
                        // mask 为0说明过滤后的结果中包含需要过滤的标签
                        std::cout << std::endl;
                        flag = true;
                    }
                }
            }
            if (flag)
            {
                ASSERT_TRUE(false);
            }
            else
            {
                printf("dim: %d model %d, batch %d, Acc successful\n", dim, valFilter.matchVal, queryNum);
            }
        }
    }
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
