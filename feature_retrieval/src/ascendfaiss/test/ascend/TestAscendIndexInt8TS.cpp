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


#include "faiss/ascend/custom/AscendIndexTS.h"
#include <limits.h>
#include <random>
#include <sys/time.h>
#include <gtest/gtest.h>

// using namespace ascend;
namespace ascend {
namespace {
const int BLOCKSIZE = 16384 * 16;
const int RESOURCE = -1;
const int DIM = 512;
const int TOPK = 100;
const int seed = 100;
const int DEVICE_ID = 0;
const uint32_t MAX_TOKEN_COUNT = 20000;
const uint32_t MAX_TOKEN_IDX = (MAX_TOKEN_COUNT + 7 ) / 8;
const int OTHER_LEN = 22;

inline void GenerateCodes(int8_t *codes, size_t total, size_t dim, int seed = -1)
{
    std::default_random_engine e((seed > 0) ? seed : time(nullptr));
    std::uniform_real_distribution<float> rCode(0.0f, 1.0f);
    for (size_t i = 0; i < total; i++) {
        for (size_t j = 0; j < dim; j++) {
            // uint8's max value is 255, int8's max value is 255 - 128
            codes[i * dim + j] = static_cast<int8_t>(255 * rCode(e) - 128);
        }
    }
}

inline double GetMillisecs()
{
    struct timeval tv {
        0, 0
    };
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

void GenerateUint8List(uint8_t* token, int bitId) {
    int tokenId = bitId / 8;
    int tokenbit = bitId % 8;
    *(token + tokenId) |= static_cast<uint8_t>(0x1 << tokenbit);
}

void GenerateOtherInfo(uint8_t* other) {
    for (int i = 0; i < OTHER_LEN; ++i) {
        *(other + i) = 0xff;
    }
}

void GenerateFilterTokenFromFeature(AttrFilter& filter, FeatureAttr* attributes, int cnt)
{
    FeatureAttr attri;
    // by default, query all feature vectors
    for (int i = 0; i < cnt; ++i) {
        attri = *(attributes + i);
        GenerateUint8List(filter.token, attri.token);
    }
}

void GenerateFeatAttrs(FeatureAttr* attributes, int cnt)
{
    uint64_t time = 5;
    for (int i = 0; i < cnt; ++i) {
        FeatureAttr attri;
        attri.time = time;
        attri.token = static_cast<uint16_t >(i % MAX_TOKEN_COUNT);
        GenerateOtherInfo(attri.other);
        *(attributes + i) = attri;
    }
}

void GenerateFilters(AttrFilter* filters, int filterCnt, FeatureAttr* attributes, int attriCnt)
{
    int CommonTimeStart = 0;
    int CommonTimeEnd = 10;
    for (int i = 0; i < filterCnt; ++i) {
        AttrFilter filter;
        filter.enableTime = true;
        filter.timesStart = CommonTimeStart;
        filter.timesEnd = CommonTimeEnd;
        filter.otherLen = 22;
        GenerateFilterTokenFromFeature(filter, attributes, attriCnt);
        // same way to generate other info of attributes and filter, make sure they're the same
        GenerateOtherInfo(filter.other);
        GenerateOtherInfo(filter.mask);
        *(filters + i) = filter;
    }
}

static void ComputeDistByCpu(const int8_t *query, const int8_t *base,
    const int dim, float &distance)
{
    float normQ = 0;
    float normB = 0;
    for (int i = 0; i < dim; i++) {
        distance += static_cast<float>(*(query + i)) * (*(base + i));
        normQ += pow(*(query + i), 2);
        normB += pow(*(base + i), 2);
    }
    distance /= (sqrt(normB) * sqrt(normQ));
}
}

TEST(TestAscendIndexInt8FlatTS, Init)
{
    ascend::AscendIndexTS index;
    auto ret = index.Init(0, 512, ascend::AlgorithmType::FLAT_COS_INT8);
    ASSERT_EQ(ret, APP_ERR_OK);
}

TEST(TestAscendIndexInt8FlatTS, AddnGetFeatures)
{
    int dim = DIM;
    int deviceId = 0;
    int ntotal = 1000000;
    std::vector<int8_t> base(ntotal * dim);
    GenerateCodes(base.data(), ntotal, dim);
    std::vector<uint64_t> label(ntotal);
    std::iota(label.begin(), label.end(), 0);
    // feature attributes
    std::vector<FeatureAttr> attributes(ntotal);
    GenerateFeatAttrs(attributes.data(), ntotal);
    ascend::AscendIndexTS index;
    auto ret = index.Init(deviceId, DIM, AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(ret, APP_ERR_OK);
    ret = index.AddFeature(base.data(), attributes.data(), label.data(), ntotal);
    EXPECT_EQ(ret, APP_ERR_OK);

    std::vector<int8_t> getBase(ntotal * dim);
    ret = index.GetFeatureByLabel(getBase.data(), label.data(), ntotal);
    EXPECT_EQ(ret, APP_ERR_OK);

#pragma omp parallel for if (ntotal > 100)
    for (int i = 0; i < ntotal; i++) {
        for (int j = 0; j < dim; j++) {
            EXPECT_EQ(base[i * dim + j], getBase[i * dim + j]);
        }
    }
}

TEST(TestAscendIndexInt8FlatTS, Search)
{
    std::vector<uint32_t> queryNList = {0, 1, 1, 1, 63, 64, 129, 102401, 102400, 102400};
    std::vector<uint32_t> topkList = {1, 0, 1025, 1024, 100, 100, 1, 1, 1, 132};
    std::vector<float> thresholdList = {0.1, 0.1, 0.1, 0.1, 0.1, 100, -1, 0.1, 0.1, 0.1};
    std::vector<int> errorCodes = {2001, 2001, 2001, 0, 0, 0, 0, 2001, 0, 2001};

    uint32_t dim = DIM;
    uint64_t ntotal = 100000;
    // 1. prepare base parameters
    std::vector<uint64_t> ids(ntotal);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<int8_t> base(ntotal * dim);
    GenerateCodes(base.data(), ntotal, dim, seed);
    std::vector<FeatureAttr> attributes(ntotal);
    GenerateFeatAttrs(attributes.data(), ntotal);

    // 2. initiate index and add vectors
    AscendIndexTS index;
    auto ret = index.Init(DEVICE_ID, DIM, AlgorithmType::FLAT_COS_INT8);
    EXPECT_EQ(ret, APP_ERR_OK);
    ret = index.AddFeature(base.data(), attributes.data(), ids.data(), ntotal);
    EXPECT_EQ(ret, APP_ERR_OK);
    
    // 3. prepare parameters and search
    // generate single filter
    AttrFilter filter;
    GenerateFilters(&filter, 1, attributes.data(), ntotal);
    for (uint32_t i = 0; i < queryNList.size(); i++) {
        std::vector<int8_t> queries(queryNList[i] * dim);
        GenerateCodes(queries.data(), queryNList[i], dim, seed);
        std::vector<uint32_t> validNums(queryNList[i]);
        std::vector<float> distances(queryNList[i] * topkList[i]);
        std::vector<uint64_t> labels(queryNList[i] * topkList[i]);
        ret = index.Search(queries.data(), &filter, queryNList[i], topkList[i], thresholdList[i], validNums.data(),
            labels.data(), distances.data());
        ASSERT_EQ(ret, errorCodes[i]);
        printf("Case %u passed !\n", i);
    }
    
    // 4. Evaluate results with cpu computation
    uint32_t queryN = 64;
    int topk = 100;
    float threshold = 0.01;
    std::vector<int8_t> queries(queryN * dim);
    GenerateCodes(queries.data(), queryN, dim, seed);
    std::vector<uint64_t> labels(queryN * topk);
    std::vector<float> distances(queryN * topk);
    std::vector<uint32_t> validNums(queryN);
    ret = index.Search(queries.data(), &filter, 1, topk, threshold, validNums.data(), labels.data(), distances.data());
    ASSERT_EQ(ret, APP_ERR_OK);

    float epsilon = 0.002;
    for (uint32_t i = 0; i < queryN; i++) {
        float distByCpu = 0;
        for (uint32_t j = 0; j < validNums[i]; j++) {
            uint64_t tmpLabel = labels[i * topk + j];
            float tmpDist = distances[i * topk + j];
            ComputeDistByCpu(queries.data() + i * dim, base.data() + tmpLabel * dim, dim, distByCpu);
            EXPECT_NEAR(tmpDist, distByCpu, epsilon);
        }
    }
}

TEST(TestAscendIndexInt8FlatTS, SearchHugeBase)
{
    // loop add units times of base vectors, ntotal accumulates to units * baseSize
    uint32_t units = 33;
    uint32_t baseSize = 2000000;
    uint32_t dim = DIM;

    // 1. Initiate index
    AscendIndexTS index;
    auto ret = index.Init(DEVICE_ID, DIM, AlgorithmType::FLAT_COS_INT8);
    ASSERT_EQ(ret, APP_ERR_OK);

    // 2. initialize base vectors and attributes, loop add unit times of same vectors, accumulate the ids
    std::vector<int8_t> base(baseSize * dim);
    GenerateCodes(base.data(), baseSize, dim, seed);
    std::vector<FeatureAttr> attributes(baseSize);
    GenerateFeatAttrs(attributes.data(), baseSize);
    std::vector<uint64_t> ids(baseSize);
    for (uint32_t i = 0; i < units; i++) {
        std::iota(ids.begin(), ids.end(), i * baseSize);
        double addStart = GetMillisecs();
        ret = index.AddFeature(base.data(), attributes.data(), ids.data(), baseSize);
        double addEnd = GetMillisecs();
        printf("--------------  Add %d features, from:%d to %d, periods:%fms\n",
            baseSize, i * baseSize, (i + 1) * baseSize,  addEnd - addStart);
        ASSERT_EQ(ret, APP_ERR_OK);
    }

    // 3. prepare parameters and search
    uint32_t queryN = 64;
    uint32_t topk = 100;
    float threshold = 0.001;
    std::vector<int8_t> queries(queryN * dim);
    GenerateCodes(queries.data(), queryN, dim, seed);
    std::vector<float> distances(queryN * topk);
    std::vector<uint64_t> labels(queryN * topk); 
    std::vector<uint32_t> validNums(queryN);

    // 3. search and evaluate the results
    AttrFilter filter;
    GenerateFilters(&filter, 1, attributes.data(), baseSize);
    ret = index.Search(queries.data(), &filter, queryN, topk, threshold, validNums.data(),
        labels.data(), distances.data());
    ASSERT_EQ(ret, APP_ERR_OK);
    float epsilon = 0.002;
    for (uint32_t i = 0; i < queryN; i++) {
        float distByCpu = 0;
        for (uint32_t j = 0; j < validNums[i]; j++) {
            uint64_t tmpLabel = labels[i * topk + j] % baseSize;
            float tmpDist = distances[i * topk + j];
            ComputeDistByCpu(queries.data() + i * dim, base.data() + tmpLabel * dim, dim, distByCpu);
            EXPECT_NEAR(tmpDist, distByCpu, epsilon);
        }
    }

    // 4. evaluate performance
    int loop = 10;
    double t0 = GetMillisecs();
    for (int i = 0; i < loop; i++) {
        ret = index.Search(queries.data(), &filter, queryN, topk, threshold, validNums.data(),
            labels.data(), distances.data());
        EXPECT_EQ(ret, APP_ERR_OK);
    }
    double t1 = GetMillisecs();
    printf("Base:%u, topk:%d, dim:%d, query batch:%d, average time: %f ms, QPS: %f\n", baseSize * units,
        topk, dim, queryN, (t1 - t0) / (loop * queryN), (queryN * loop * 1000) / (t1 - t0));

}

} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
