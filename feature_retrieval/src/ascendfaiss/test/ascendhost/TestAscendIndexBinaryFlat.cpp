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
#include <functional>
#include <random>
#include <vector>
#include <queue>
#include <sys/time.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <faiss/impl/FaissException.h>
#include <faiss/ascend/AscendIndexBinaryFlat.h>
#include <faiss/impl/AuxIndexStructures.h>

namespace {
using idx_t = faiss::idx_t;

constexpr idx_t MAX_TOPK = 1e5;
constexpr int PAGE_BLOCKS = 5;
constexpr int BLOCK_SIZE = 1024 * 256;

std::independent_bits_engine<std::mt19937, 8, uint8_t> engine(1);

void FeatureGenerator(std::vector<uint8_t> &features)
{
    size_t n = features.size();
    for (size_t i = 0; i < n; ++i) {
        features[i] = engine();
    }
}

template <class T> void Print2Dims(std::vector<T> &vec, int dim1, int dim2)
{
    for (int i = 0; i < dim1; ++i) {
        if (i != 0) {
            std::cout << std::endl;
        }
        for (int j = 0; j < dim2; ++j) {
            std::cout << (int)vec[dim2 * i + j] << " ";
        }
        std::cout << std::endl;
    }
}

void GetXFeature(std::vector<uint8_t> &features, int x, std::vector<uint8_t> &ret)
{
    int code_size = ret.size();
    int offset = x * code_size;
    for (int i = 0; i < code_size; ++i) {
        ret[i] = features[offset++];
    }
}

inline void AssertInt8Equal(size_t count, const uint8_t *gt, const uint8_t *data)
{
    for (size_t i = 0; i < count; i++) {
        ASSERT_TRUE(static_cast<unsigned int>(gt[i]) == static_cast<unsigned int>(data[i]))
            << "i: " << i << " gt: " << static_cast<unsigned int>(gt[i]) 
            << " data: " << static_cast<unsigned int>(data[i]) << std::endl;
    }
}

inline double GetMillisecs()
{
    struct timeval tv = { 0, 0 };
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

int HammingDistance(std::vector<uint8_t> &oneFeature, std::vector<uint8_t> &oneQuery)
{
    int ret = 0;
    size_t code_size = oneFeature.size();
    for (size_t idx = 0; idx < code_size; ++idx) {
        ret += std::bitset<8>(oneFeature[idx] ^ oneQuery[idx]).count();
    }
    return ret;
}
} // end of namespace

TEST(TestAscendIndexBinary, add)
{
    idx_t ntotal = 10000000;
    size_t addTotal = 1;
    int dims = 512;
    auto conf = faiss::ascend::AscendIndexBinaryFlatConfig({ 0 }, 1024 * 1024 * 1024);
    faiss::IndexBinary *index = new faiss::ascend::AscendIndexBinaryFlat(dims, conf);
    std::vector<uint8_t> features(ntotal * dims / 8);
    FeatureGenerator(features);
    auto ts = GetMillisecs();
    for (size_t i = 0; i < addTotal; ++i) {
        index->add(ntotal, features.data());
    }
    auto te = GetMillisecs();
    printf("add %ld cost %f ms\n", index->ntotal, te - ts);

    delete index;
}

TEST(TestAscendIndexBinary, indexBinaryIDMap)
{
    idx_t ntotal = 1000000;
    size_t addTotal = 1;
    int dims = 512;

    faiss::IndexBinaryFlat faissIndex(dims);
    faiss::IndexBinaryIDMap idIndex(&faissIndex);
    auto conf = faiss::ascend::AscendIndexBinaryFlatConfig({ 0 }, 1024 * 1024 * 1024);
    faiss::IndexBinary *index = new faiss::ascend::AscendIndexBinaryFlat(&idIndex, conf);
    std::vector<uint8_t> features(ntotal * dims / 8);
    
    FeatureGenerator(features);

    ASSERT_EQ(index->ntotal, 0);
    auto ts = GetMillisecs();
    for (size_t i = 0; i < addTotal; ++i) {
        index->add(ntotal, features.data());
    }
    ASSERT_EQ(index->ntotal, ntotal*addTotal);
    auto te = GetMillisecs();
    printf("add %ld cost %f ms\n", index->ntotal, te - ts);

    delete index;
}

TEST(TestAscendIndexBinaryFlat, removeRange)
{
    std::vector<int> ntotals = {
        1000000, /* 10, 101, 1024, 1025, 9999, 262145, BLOCK_SIZE * PAGE_BLOCKS + 1 */
    }; // make sure the vector is in ascending order
    int dims = 512;
    std::vector<int> queryNums = {
        1, /* 6, 9, 16, 32, 64, 128, 255, 256, 513 */
    }; // make sure the vector is in ascending order
    std::vector<int> topks = { 10, 3, 2, 1 };

    auto conf = faiss::ascend::AscendIndexBinaryFlatConfig({ 0 }, (size_t)4 * 1024 * 1024 * 1024);

    std::vector<uint8_t> features(ntotals.back() * dims / 8, 0);
    std::vector<uint8_t> queries(queryNums.back() * dims / 8, 0);

    FeatureGenerator(features);
    FeatureGenerator(queries);
    
    int delRangeMin = 1000;
    int delRangeMax = 10000;
    faiss::IDSelectorRange del(delRangeMin, delRangeMax);
    
    for (auto ntotal : ntotals) {
        faiss::IndexBinary *index = new faiss::ascend::AscendIndexBinaryFlat(dims, conf);
        faiss::IndexBinary *faissIndex = new faiss::IndexBinaryFlat(dims);
        index->add(ntotal, features.data());
        faissIndex->add(ntotal, features.data());

        size_t indexRmedCnt = index->remove_ids(del);
        size_t faissIndexRmedCnt = faissIndex->remove_ids(del);
        std::cout << "indexRmedCnt=" << indexRmedCnt << ", faissIndexRmedCnt=" << faissIndexRmedCnt << std::endl;
        ASSERT_EQ(indexRmedCnt, faissIndexRmedCnt) <<
                  "indexRmedCnt=" << indexRmedCnt << ", faissIndexRmedCnt=" << faissIndexRmedCnt;

        for (auto queryNum : queryNums) {
            for (auto k : topks) {
                if (k > ntotal-indexRmedCnt) {
                    continue;
                }
                std::vector<int32_t> distances(queryNum * k, -1);
                std::vector<idx_t> labels(queryNum * k, -1);
                std::vector<int32_t> faissDistances(queryNum * k, -1);
                std::vector<idx_t> faissLabels(queryNum * k, -1);
                index->search(queryNum, queries.data(), k, distances.data(), labels.data());
                faissIndex->search(queryNum, queries.data(), k, faissDistances.data(), faissLabels.data());
                int total = queryNum * k;
                for (int nq = 0; nq < queryNum; ++nq) {
                    for (int nk = 0; nk < k; ++nk) {
                        ASSERT_EQ(distances[nq * k + nk], faissDistances[nq * k + nk])
                                 << queryNum << ntotal-indexRmedCnt << k;
                    }
                }
            }
        }
        delete index;
        delete faissIndex;
    }
}

TEST(TestAscendIndexBinaryFlat, removeBatch)
{
    std::vector<int> ntotals = {
        1000000, /* 10, 101, 1024, 1025, 9999, 262145, BLOCK_SIZE * PAGE_BLOCKS + 1 */
    }; // make sure the vector is in ascending order
    int dims = 512;
    std::vector<int> queryNums = {
        1, /* 6, 9, 16, 32, 64, 128, 255, 256, 513 */
    }; // make sure the vector is in ascending order
    std::vector<int> topks = { 10, 3, 2, 1 };

    auto conf = faiss::ascend::AscendIndexBinaryFlatConfig({ 0 }, (size_t)4 * 1024 * 1024 * 1024);

    std::vector<uint8_t> features(ntotals.back() * dims / 8, 0);
    std::vector<uint8_t> queries(queryNums.back() * dims / 8, 0);

    FeatureGenerator(features);
    FeatureGenerator(queries);
    
    std::vector<faiss::idx_t> delBatchs = { 0, 1, 2, 4, 6, 9, };
    faiss::IDSelectorBatch del(delBatchs.size(), delBatchs.data());
    
    for (auto ntotal : ntotals) {
        faiss::IndexBinary *index = new faiss::ascend::AscendIndexBinaryFlat(dims, conf);
        faiss::IndexBinary *faissIndex = new faiss::IndexBinaryFlat(dims);
        index->add(ntotal, features.data());
        faissIndex->add(ntotal, features.data());

        size_t indexRmedCnt = index->remove_ids(del);
        size_t faissIndexRmedCnt = faissIndex->remove_ids(del);
        std::cout << "indexRmedCnt=" << indexRmedCnt << ", faissIndexRmedCnt=" << faissIndexRmedCnt << std::endl;
        ASSERT_EQ(indexRmedCnt, faissIndexRmedCnt) <<
                  "indexRmedCnt=" << indexRmedCnt << ", faissIndexRmedCnt=" << faissIndexRmedCnt;

        for (auto queryNum : queryNums) {
            for (auto k : topks) {
                if (k > ntotal-indexRmedCnt) {
                    continue;
                }
                std::vector<int32_t> distances(queryNum * k, -1);
                std::vector<idx_t> labels(queryNum * k, -1);
                std::vector<int32_t> faissDistances(queryNum * k, -1);
                std::vector<idx_t> faissLabels(queryNum * k, -1);
                index->search(queryNum, queries.data(), k, distances.data(), labels.data());
                faissIndex->search(queryNum, queries.data(), k, faissDistances.data(), faissLabels.data());
                int total = queryNum * k;
                for (int nq = 0; nq < queryNum; ++nq) {
                    for (int nk = 0; nk < k; ++nk) {
                        ASSERT_EQ(distances[nq * k + nk], faissDistances[nq * k + nk])
                                 << queryNum << ntotal-indexRmedCnt << k;
                    }
                }
            }
        }
        delete index;
        delete faissIndex;
    }
}

TEST(TestAscendIndexBinaryFlat, copyFrom)
{
    int dim = 512;
    int ntotal = 10000000;
    std::vector<uint8_t> data(dim/8 * ntotal);
    FeatureGenerator(data);

    faiss::IndexBinaryFlat *faissIndex = new faiss::IndexBinaryFlat(dim);
    faissIndex->add(ntotal, data.data());

    auto conf = faiss::ascend::AscendIndexBinaryFlatConfig({ 0 }, (size_t)4 * 1024 * 1024 * 1024);
    faiss::ascend::AscendIndexBinaryFlat *index = new faiss::ascend::AscendIndexBinaryFlat(dim, conf);
    index->copyFrom(faissIndex);
    
    faiss::IndexBinaryFlat *faissIndexCopyed = new faiss::IndexBinaryFlat(dim);
    index->copyTo(faissIndexCopyed);

    EXPECT_EQ(faissIndex->d, dim);
    EXPECT_EQ(faissIndex->ntotal, ntotal);
    EXPECT_EQ(faissIndex->code_size, dim/8);

    EXPECT_EQ(faissIndex->d, index->d);
    EXPECT_EQ(faissIndex->ntotal, index->ntotal);
    EXPECT_EQ(faissIndex->code_size, index->code_size);
    AssertInt8Equal(faissIndex->ntotal * faissIndex->code_size, data.data(), faissIndex->xb.data());
    AssertInt8Equal(faissIndexCopyed->ntotal * faissIndexCopyed->code_size,
        faissIndexCopyed->xb.data(), faissIndex->xb.data());
}

TEST(TestAscendIndexBinaryFlat, copyTo)
{
    int dim = 512;
    int ntotal = 10000000;
    std::vector<uint8_t> data(dim/8 * ntotal);
    FeatureGenerator(data);

    auto conf = faiss::ascend::AscendIndexBinaryFlatConfig({ 0 }, (size_t)4 * 1024 * 1024 * 1024);
    faiss::ascend::AscendIndexBinaryFlat *index = new faiss::ascend::AscendIndexBinaryFlat(dim, conf);
    index->add(ntotal, data.data());

    faiss::IndexBinaryFlat *faissIndex = new faiss::IndexBinaryFlat(dim);
    index->copyTo(faissIndex);

    EXPECT_EQ(index->d, dim);
    EXPECT_EQ(index->ntotal, ntotal);
    EXPECT_EQ(index->code_size, dim/8);

    EXPECT_EQ(faissIndex->d, index->d);
    EXPECT_EQ(faissIndex->ntotal, index->ntotal);
    EXPECT_EQ(faissIndex->code_size, index->code_size);
    AssertInt8Equal(faissIndex->ntotal*faissIndex->code_size, data.data(), faissIndex->xb.data());
}

TEST(TestAscendIndexBinaryFlat, reset)
{
    int dim = 512;
    int ntotal = 10000000;
    std::vector<uint8_t> data(dim/8 * ntotal);
    FeatureGenerator(data);

    auto conf = faiss::ascend::AscendIndexBinaryFlatConfig({ 0 }, (size_t)4 * 1024 * 1024 * 1024);
    faiss::ascend::AscendIndexBinaryFlat *index = new faiss::ascend::AscendIndexBinaryFlat(dim, conf);
    index->add(ntotal, data.data());
    index->reset();
    faiss::IndexBinaryFlat *faissIndexCopyed = new faiss::IndexBinaryFlat(dim);
    index->copyTo(faissIndexCopyed);

    faiss::IndexBinaryFlat *faissIndex = new faiss::IndexBinaryFlat(dim);
    faissIndex->add(ntotal, data.data());
    faissIndex->reset();

    EXPECT_EQ(faissIndex->d, index->d);
    EXPECT_EQ(faissIndex->ntotal, index->ntotal);
    EXPECT_EQ(faissIndex->code_size, index->code_size);

    EXPECT_EQ(faissIndex->d, faissIndexCopyed->d);
    EXPECT_EQ(faissIndex->ntotal, faissIndexCopyed->ntotal);
    EXPECT_EQ(faissIndex->code_size, faissIndexCopyed->code_size);
    AssertInt8Equal(faissIndex->ntotal*faissIndex->code_size, faissIndexCopyed->xb.data(), faissIndex->xb.data());
}

TEST(TestAscendIndexBinary, Acc)
{
    std::vector<int> ntotals = {
        1, 10, 101, 1024, 1025, 9999, 32769, 262145
    }; // make sure the vector is in ascending order
    int dims = 256;
    std::vector<int> queryNums = {
        1, 6, 9, 16, 32, 64, 128, 255, 256, 513
    }; // make sure the vector is in ascending order
    std::vector<int> topks = { 1024, 100, 10, 1 };

    auto conf = faiss::ascend::AscendIndexBinaryFlatConfig({ 0 }, (size_t)4 * 1024 * 1024 * 1024, true);

    std::vector<uint8_t> features(ntotals.back() * dims / 8, 0);
    std::vector<uint8_t> queries(queryNums.back() * dims / 8, 0);

    FeatureGenerator(features);
    FeatureGenerator(queries);

    std::random_device rd;
    std::mt19937 g(rd());
    std::map<int, std::map<int, int>> distanceBuffer;
    for (auto ntotal : ntotals) {
        faiss::IndexBinary *index = new faiss::ascend::AscendIndexBinaryFlat(dims, conf);
        faiss::IndexBinary *faissIndex =
            new faiss::IndexBinaryFlat(dims); // faiss::IndexBinaryFlat does not implement add_with_ids

        std::vector<idx_t> ids(ntotal);
        std::iota(ids.begin(), ids.end(), rd());
        std::shuffle(ids.begin(), ids.end(), g);
        std::map<idx_t, idx_t> invertedIds;
        for (int i = 0; i < ntotal; ++i) {
            invertedIds[ids[i]] = i;
        }
        index->add_with_ids(ntotal, features.data(), ids.data());
        faissIndex->add(ntotal, features.data());
        for (auto queryNum : queryNums) {
            for (auto k : topks) {
                if (k > ntotal) {
                    continue;
                }
                std::vector<int32_t> distances(queryNum * k, -1);
                std::vector<idx_t> labels(queryNum * k, -1);
                std::vector<int32_t> faissDistances(queryNum * k, -1);
                std::vector<idx_t> faissLabels(queryNum * k, -1);
                index->search(queryNum, queries.data(), k, distances.data(), labels.data());
                faissIndex->search(queryNum, queries.data(), k, faissDistances.data(), faissLabels.data());
                for (int nq = 0; nq < queryNum; ++nq) {
                    std::set<idx_t> labelSet;
                    for (int nk = 0; nk < k; ++nk) {
                        ASSERT_TRUE(labelSet.find(labels[nq * k + nk]) == labelSet.end());
                        labelSet.insert(labels[nq * k + nk]);
                        ASSERT_EQ(distances[nq * k + nk], faissDistances[nq * k + nk]);
                        if (labels[nq * k + nk] != ids[faissLabels[nq * k + nk]]) {
                            int ascendLabelDistance;
                            if (distanceBuffer.find(nq) == distanceBuffer.end() ||
                                distanceBuffer[nq].find(invertedIds[labels[nq * k + nk]]) == distanceBuffer[nq].end()) {
                                std::vector<uint8_t> oneQuery(dims / 8);
                                std::vector<uint8_t> ascendLabelFeature(dims / 8);
                                GetXFeature(queries, nq, oneQuery);
                                GetXFeature(features, invertedIds[labels[nq * k + nk]], ascendLabelFeature);
                                ascendLabelDistance = HammingDistance(ascendLabelFeature, oneQuery);
                                distanceBuffer[nq][invertedIds[labels[nq * k + nk]]] = ascendLabelDistance;
                            } else {
                                ascendLabelDistance = distanceBuffer[nq][invertedIds[labels[nq * k + nk]]];
                            }

                            int faissLabelDistance;
                            if (distanceBuffer.find(nq) == distanceBuffer.end() ||
                                distanceBuffer[nq].find(faissLabels[nq * k + nk]) == distanceBuffer[nq].end()) {
                                std::vector<uint8_t> oneQuery(dims / 8);
                                std::vector<uint8_t> faissLabelFeature(dims / 8);
                                GetXFeature(queries, nq, oneQuery);
                                GetXFeature(features, faissLabels[nq * k + nk], faissLabelFeature);
                                faissLabelDistance = HammingDistance(faissLabelFeature, oneQuery);
                                distanceBuffer[nq][faissLabels[nq * k + nk]] = ascendLabelDistance;
                            } else {
                                faissLabelDistance = distanceBuffer[nq][faissLabels[nq * k + nk]];
                            }
                            ASSERT_EQ(ascendLabelDistance, faissLabelDistance);
                        }
                    }
                }
            }
        }
        delete index;
        delete faissIndex;
    }
}

TEST(TestAscendIndexBinary, QPS)
{
    idx_t ntotal = 10000000;
    int dims = 512;
    std::vector<int> queryNums = { 256 };
    std::vector<int> topks = { 1024 };

    auto conf = faiss::ascend::AscendIndexBinaryFlatConfig({ 0 });
    faiss::IndexBinary *index = new faiss::ascend::AscendIndexBinaryFlat(dims, conf);

    std::vector<uint8_t> features(ntotal * index->code_size, 0);
    std::vector<uint8_t> queries(queryNums.back() * index->code_size, 0);
    FeatureGenerator(features);
    FeatureGenerator(queries);

    index->add(ntotal, features.data());

    long double ts { 0. };
    long double te { 0. };
    for (auto k : topks) {
        int warmupTimes = 10;
        int loopTimes = 100;
        for (auto queryNum : queryNums) {
            std::vector<int32_t> distances(queryNum * k, -1);
            std::vector<idx_t> labels(queryNum * k, -1);
            for (int i = 0; i < loopTimes + warmupTimes; i++) {
                if (i == warmupTimes) {
                    ts = GetMillisecs();
                }
                index->search(queryNum, queries.data(), k, distances.data(), labels.data());
            }
            te = GetMillisecs();
            printf("base: %ld,  dim: %d,  batch: %4d,  top%d,  QPS:%7.2Lf\n", ntotal, dims, queryNum, k,
                (long double)1000.0 * queryNum * loopTimes / (te - ts));
        }
    }

    delete index;
}

TEST(TestAscendIndexBinaryFloat, QPS)
{
    int64_t ntotal = 1000000;
    size_t addTotal = 200;
    int dims = 512;
    std::vector<int> queryNums = { 1, 2, 4, 8, 16, 32, 64, 128, 256 };
    std::vector<int> topks = { 1000 };
    
    std::vector<float> queries(queryNums.back() * dims, 0);
    for (size_t i = 0; i < queries.size(); i++) {
        queries[i] = drand48();
    }
    std::vector<uint8_t> featuresInt8(ntotal * dims / 8);
    FeatureGenerator(featuresInt8);

    auto conf = faiss::ascend::AscendIndexBinaryFlatConfig({ 0 }, 1024 * 1024 * 1024);
    faiss::ascend::AscendIndexBinaryFlat *index = new faiss::ascend::AscendIndexBinaryFlat(dims, conf);

    printf("add start %ld \n", ntotal * addTotal);
    for (size_t i = 0; i < addTotal; ++i) {
        index->add(ntotal, featuresInt8.data());
    }
    long double ts { 0. };
    long double te { 0. };
    for (auto k : topks) {
        int warmupTimes = 2;
        int loopTimes = 1;
        for (auto queryNum : queryNums) {
            std::vector<float> distances(queryNum * k, -1);
            std::vector<int64_t> labels(queryNum * k, -1);
            for (int i = 0; i < loopTimes + warmupTimes; i++) {
                if (i == warmupTimes) {
                    ts = GetMillisecs();
                }
                index->search(queryNum, queries.data(), k, distances.data(), labels.data());
            }
            te = GetMillisecs();
            printf("base: %ld,  dim: %d,  batch: %4d,  top%d,  QPS:%7.2Lf\n", ntotal * addTotal, dims, queryNum, k,
                1000.0 * queryNum * loopTimes / (te - ts));
        }
    }
    delete index; 
}

TEST(TestAscendIndexBinaryFloat, Acc)
{
    int64_t ntotal = 1000000;
    size_t addTotal = 1;
    int dims = 512;
    std::vector<int> queryNums = { 128 };
    int k = 1000;
    printf("dim:%d, batch:%d \n", dims, queryNums[0]);
    std::vector<float> queries(queryNums.back() * dims, 0);

    for (size_t i = 0; i < queries.size(); i++) {
        queries[i] = drand48();
    }
    std::vector<uint8_t> featuresInt8(ntotal * dims / 8);
    FeatureGenerator(featuresInt8);

    // 将uint8二进制转换为1.00 -1.00浮点 保证输入相同
    std::vector<float> featuresFloat;
    for (size_t i = 0; i < featuresInt8.size(); i++) {
        uint8_t res = featuresInt8[i];
        for (int j = 0; j < 8; j++) {
            if((res & 1) == 1) {
                featuresFloat.push_back(1.00);
            }
            else {
                featuresFloat.push_back(-1.00);
            }
            res = (res >> 1);
        }
    }

    for (auto queryNum : queryNums) {
        faiss::ascend::AscendIndexFlatConfig confFlat({ 0 });
        faiss::ascend::AscendIndexFlat indexFlat(dims, faiss::METRIC_INNER_PRODUCT, confFlat);
        indexFlat.add(ntotal, featuresFloat.data());

        std::vector<float> distancesFlat(queryNum * k, -1);
        std::vector<int64_t> labelsFlat(queryNum * k, -1);

        indexFlat.search(queryNum, queries.data(), k, distancesFlat.data(), labelsFlat.data());

        for (int i = 0; i < 10; i++) {
            printf("labelflat[%d] = %lu , distflat[%d] = %.4f \n", i, labelsFlat[i], i, distancesFlat[i]);
        }
        
        auto conf = faiss::ascend::AscendIndexBinaryFlatConfig({ 0 }, 1024 * 1024 * 1024);
        faiss::ascend::AscendIndexBinaryFlat *index = new faiss::ascend::AscendIndexBinaryFlat(dims, conf);

        printf("add start %ld \n", ntotal * addTotal);
        for (size_t i = 0; i < addTotal; ++i) {
            index->add(ntotal, featuresInt8.data());
        }
 
        std::vector<float> distances(queryNum * k, -1);
        std::vector<int64_t> labels(queryNum * k, -1);

        index->search(queryNum, queries.data(), k, distances.data(), labels.data());
        
        for (int i = 0; i < 10; i++) {
            printf("label[%d] = %lu , dist[%d] = %.4f \n", i, labels[i], i, distances[i]);
        }

        for (int i = 0; i < queryNum * k; i++) {
            if (distances[i] != distancesFlat[i] || labels[i] != labelsFlat[i]) {
                printf("Acc error! \n");
                break;
            }
        }
        delete index; 
    }
}


int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}