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

#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include "AscendIndexBinaryFlat.h"
#include "Common.h"
#include "ascenddaemon/impl/Index.h"
#include "ascendhost/include/impl/AscendIndexBinaryFlatImpl.h"
#include "faiss/impl/AuxIndexStructures.h"
#include "faiss/impl/IDSelector.h"


namespace ascend {

void StubSetRemoveFast(bool useRemoveFast)
{
    faiss::ascend::AscendIndexBinaryFlatImpl::isRemoveFast = useRemoveFast;
}

void BinaryFlatUint8Test(idx_t ntotal, int dims, idx_t queryNum, idx_t k)
{
    auto conf = faiss::ascend::AscendIndexBinaryFlatConfig({0});
    faiss::ascend::AscendIndexBinaryFlat *index = new faiss::ascend::AscendIndexBinaryFlat(dims, conf);
    std::vector<uint8_t> features(ntotal * dims / BITELN);
    FeatureGenerator(features);
    index->add(ntotal, features.data());
    EXPECT_EQ(static_cast<idx_t>(index->ntotal), ntotal);
    std::vector<int32_t> dist(k * queryNum, -1);
    std::vector<faiss::idx_t> label(k * queryNum, -1);
    index->search(queryNum, features.data(), k, dist.data(), label.data());

    faiss::IndexBinaryFlat *faissIndex = new faiss::IndexBinaryFlat(dims);
    faissIndex->add(ntotal, features.data());
    faiss::ascend::AscendIndexBinaryFlat *index2 = new faiss::ascend::AscendIndexBinaryFlat(dims, conf);
    index2->copyFrom(faissIndex);
    EXPECT_EQ(faissIndex->ntotal, index2->ntotal);
    faiss::IndexBinaryFlat *faissIndexCopyed = new faiss::IndexBinaryFlat(dims);
    index2->copyTo(faissIndexCopyed);
    EXPECT_EQ(faissIndexCopyed->ntotal, index2->ntotal);
    delete index2;
    faissIndex->reset();
    faiss::IndexBinaryIDMap idIndex(faissIndex);
    faiss::ascend::AscendIndexBinaryFlat indexbyId(&idIndex, conf);
    indexbyId.copyFrom(faissIndexCopyed);
    EXPECT_EQ(indexbyId.ntotal, faissIndexCopyed->ntotal);
    delete faissIndexCopyed;
    faiss::ascend::AscendIndexBinaryFlat indexByFaiss(faissIndex, conf);
    indexByFaiss.copyFrom(&idIndex);
    EXPECT_EQ(idIndex.ntotal, indexByFaiss.ntotal);
    faiss::IndexBinaryIDMap idIndexCopyed(faissIndex);
    indexByFaiss.copyTo(&idIndexCopyed);
    EXPECT_EQ(idIndexCopyed.ntotal, indexByFaiss.ntotal);
    delete faissIndex;
    int delRangeMin = 10;
    int delRangeMax = 100;
    faiss::IDSelectorRange del(delRangeMin, delRangeMax);
    size_t indexRmedCnt = index->remove_ids(del);
    EXPECT_EQ(indexRmedCnt, static_cast<size_t>(delRangeMax - delRangeMin));
    index->reset();
    EXPECT_EQ(index->ntotal, 0);
    std::vector<faiss::idx_t> ids(ntotal);
    std::iota(ids.begin(), ids.end(), 0);
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(ids), std::end(ids), rng);
    index->add_with_ids(ntotal, features.data(), ids.data());
    EXPECT_EQ(static_cast<idx_t>(index->ntotal), ntotal);
    delete index;
}

void BinaryFlatFloatTest()
{
    int64_t ntotal = 1000;
    int dims = 512;
    int queryNum = 1;
    int k = 10;
    int bitUint8 = 8;
    bool isUsedFloat = true;
    std::vector<int> topks = { 1000 };
    
    std::vector<float> queries(queryNum * dims, 0);
    for (size_t i = 0; i < queries.size(); i++) {
        queries[i] = drand48();
    }
    std::vector<uint8_t> featuresInt8(ntotal * dims / bitUint8);
    FeatureGenerator(featuresInt8);
 
    auto conf = faiss::ascend::AscendIndexBinaryFlatConfig({ 0 }, 1024 * 1024 * 1024);
    faiss::ascend::AscendIndexBinaryFlat index(dims, conf, isUsedFloat);
    index.add(ntotal, featuresInt8.data());

    std::vector<float> distances(queryNum * k, -1);
    std::vector<int64_t> labels(queryNum * k, -1);
    index.search(queryNum, queries.data(), k, distances.data(), labels.data());
}

void BinaryFlatUint8TestWithRemoveFast(idx_t ntotal, int dims, idx_t queryNum, idx_t k)
{
    MOCKER_CPP(&faiss::ascend::AscendIndexBinaryFlatImpl::setRemoveFast).expects(once())
                .will(invoke(StubSetRemoveFast));
    MOCKER_CPP(&faiss::ascend::AscendIndexBinaryFlatImpl::removeSingle).expects(atLeast(1));

    auto conf = faiss::ascend::AscendIndexBinaryFlatConfig({0});
    faiss::ascend::AscendIndexBinaryFlat::setRemoveFast(true);
    faiss::ascend::AscendIndexBinaryFlat index(dims, conf);
    std::vector<uint8_t> features(ntotal * dims / BITELN);
    FeatureGenerator(features);
    index.add(ntotal, features.data());
    EXPECT_EQ(static_cast<idx_t>(index.ntotal), ntotal);

    int delRangeMin = 10;
    int delRangeMax = 100;
    faiss::IDSelectorRange delRange(delRangeMin, delRangeMax);
    size_t indexRmedCnt = index.remove_ids(delRange);
    std::vector<faiss::idx_t> delVec = { 0, 1, 2, 4, 6, 9 };
    faiss::IDSelectorBatch delBatchs(delVec.size(), delVec.data());
    indexRmedCnt = index.remove_ids(delBatchs);
    index.reset();
    EXPECT_EQ(index.ntotal, 0);

    GlobalMockObject::verify();
}

TEST(TestAscendIndexUTBinaryFlat, all)
{
    idx_t ntotal = 10000;
    int dims = 512;
    idx_t queryNum = 1;
    idx_t k = 10;
    printf("Hanming\r\n");
    BinaryFlatUint8Test(ntotal, dims, queryNum, k);
    printf("Hanming with fast remove\r\n");
    BinaryFlatUint8TestWithRemoveFast(ntotal, dims, queryNum, k);
    printf("float\r\n");
    BinaryFlatFloatTest();
}

}