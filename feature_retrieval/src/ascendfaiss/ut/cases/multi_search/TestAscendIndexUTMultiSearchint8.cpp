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
#include <sys/time.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <mockcpp/mockcpp.hpp>
#include <faiss/ascend/AscendIndexInt8Flat.h>
#include "securec.h"
#include "ut/Common.h"
#include "acl_op.h"
#include "AscendMultiIndexSearch.h"

using namespace testing;
using namespace std;

namespace {

struct CheckInt8Item {
    uint32_t indexNum;
    faiss::MetricType metricType;
    int dim;
};

struct CheckInt8ParamItem {
    uint32_t indexNum;
    faiss::MetricType metricType;
    int dim;
    int searchNum;
    int k;
    std::string str;
};

struct CheckInt8PointerItem {
    int8_t *query;
    float *dist;
    faiss::idx_t *label;
    faiss::MetricType metricType;
    int pointerNum;
    std::string str;
};

struct CheckInt8IndexItem {
    faiss::MetricType metricType;
    int dim;
    std::vector<int> device;
    std::string str;
};

class TestAscendIndexUTMultiSearchInt8 : public TestWithParam<CheckInt8Item> {
};

class TestAscendIndexMultiSearchInt8Param : public TestWithParam<CheckInt8ParamItem> {
};

class TestAscendIndexMultiSearchInt8Pointer : public TestWithParam<CheckInt8PointerItem> {
};

class TestAscendIndexMultiSearchInt8Indexes : public TestWithParam<CheckInt8IndexItem> {
};

constexpr int DIM = 256;
constexpr int SEARCHNUM = 10;
constexpr int INDEXNUM = 2;
constexpr int K = 10;
constexpr size_t NTOTAL = 1000;
const int BLOCKSIZE = 16384 * 16;
const int64_t RESOURSESIZE = 1024 * 1024 * 1024;
std::vector<int8_t> g_query(DIM * SEARCHNUM);
std::vector<float> g_dist(INDEXNUM * SEARCHNUM * K);
std::vector<faiss::idx_t> g_label(INDEXNUM * SEARCHNUM * K);

const CheckInt8Item INT8ITEMS[] = {
    { 1, faiss::METRIC_L2 },
    { 10, faiss::METRIC_L2 }
};

const CheckInt8ParamItem PARAM[] = {
    { 0, faiss::METRIC_L2, 256, 2, 1, "size of indexes (0) must be > 0 and <= 10000." },
    { 10001, faiss::METRIC_L2, 256, 1, 1,  "size of indexes (10001) must be > 0 and <= 10000." },
    { 2, faiss::METRIC_L2, 256, 0, 10, "n must be > 0 and <= 1024" },
    { 2, faiss::METRIC_L2, 256, 1025, 10, "n must be > 0 and <= 1024" },
    { 2, faiss::METRIC_L2, 256, 2, 0, "k must be > 0 and <= 1024" },
    { 2, faiss::METRIC_L2, 256, 1, 1025, "k must be > 0 and <= 1024" }
};

const CheckInt8PointerItem POINTER[] = {
    { nullptr, g_dist.data(), g_label.data(), faiss::METRIC_INNER_PRODUCT, 0, "Invalid x: nullptr." },
    { g_query.data(), nullptr, g_label.data(), faiss::METRIC_INNER_PRODUCT, 0, "Invalid distances: nullptr." },
    { g_query.data(), g_dist.data(), nullptr, faiss::METRIC_INNER_PRODUCT, 0, "Invalid labels: nullptr." },
    { g_query.data(), g_dist.data(), g_label.data(), faiss::METRIC_INNER_PRODUCT, 1,
        "Invalid index 0 from given indexes: nullptr." },
    { nullptr, g_dist.data(), g_label.data(), faiss::METRIC_L2, 0, "Invalid x: nullptr." },
    { g_query.data(), nullptr, g_label.data(), faiss::METRIC_L2, 0, "Invalid distances: nullptr." },
    { g_query.data(), g_dist.data(), nullptr, faiss::METRIC_L2, 0, "Invalid labels: nullptr." },
    { g_query.data(), g_dist.data(), g_label.data(), faiss::METRIC_L2, 1,
        "Invalid index 0 from given indexes: nullptr.", }
};

const CheckInt8IndexItem INDEXES[] = {
    { faiss::METRIC_L2, 512, { 0 }, "the dim must be same." },
    { faiss::METRIC_L2, 256, { 1 }, "the deviceList must be same." },
    { faiss::METRIC_L2, 256, { 0, 1 }, "the number of deviceList (2) must be 1." }
};

void AscendInt8Indexs(uint32_t indexNum, faiss::MetricType metricType,
    vector<faiss::ascend::AscendIndexInt8*> &indexes, vector<shared_ptr<faiss::ascend::AscendIndexInt8Flat>> &indexRet)
{
    faiss::ascend::AscendIndexInt8FlatConfig config({ 0 }, RESOURSESIZE, BLOCKSIZE);
    auto index = std::make_shared<faiss::ascend::AscendIndexInt8Flat>(DIM, metricType, config);
    for (uint32_t i = 0; i < indexNum; i++) {
        indexRet.emplace_back(index);
        indexes.emplace_back(index.get());
    }
}

TEST_P(TestAscendIndexUTMultiSearchInt8, MultiInt8IndexSearch)
{
    CheckInt8Item item = GetParam();
    int indexNum = item.indexNum;
    faiss::MetricType metricType = item.metricType;

    std::vector<std::vector<int8_t>> data(indexNum, std::vector<int8_t>(DIM * NTOTAL));

    for (int i = 0; i < indexNum; i++) {
        ascend::FeatureGenerator<int8_t>(data[i]);
    }
    vector<faiss::ascend::AscendIndexInt8*> indexes;
    vector<shared_ptr<faiss::ascend::AscendIndexInt8Flat>> tmpIndex;
    AscendInt8Indexs(indexNum, metricType, indexes, tmpIndex);

    for (int i = 0; i < indexNum; ++i) {
        indexes[i]->add(NTOTAL, data[i].data());
    }

    std::vector<float> dist(indexNum * SEARCHNUM * K, 0);
    std::vector<faiss::idx_t> label(indexNum * SEARCHNUM * K, 0);
    std::vector<int8_t> query(DIM * SEARCHNUM);
    ascend::FeatureGenerator<int8_t>(query);
    Search(indexes, SEARCHNUM, query.data(), K, dist.data(), label.data(), false);

    std::vector<float> mergeDist(SEARCHNUM * K, 0);
    std::vector<faiss::idx_t> mergeLabel(SEARCHNUM * K, 0);
    Search(indexes, SEARCHNUM, query.data(), K, mergeDist.data(), mergeLabel.data(), true);
}

TEST_P(TestAscendIndexMultiSearchInt8Param, InvalidInputParam)
{
    CheckInt8ParamItem item = GetParam();
    int indexNum = item.indexNum;
    faiss::MetricType metricType = item.metricType;
    int dim = item.dim;
    int searchNum = item.searchNum;
    int k = item.k;
    std::string str = item.str;
    std::string msg;

    vector<faiss::ascend::AscendIndexInt8*> indexes;
    vector<shared_ptr<faiss::ascend::AscendIndexInt8Flat>> tmpIndex;
    AscendInt8Indexs(indexNum, metricType, indexes, tmpIndex);
    std::vector<int8_t> query(dim * searchNum);
    std::vector<float> dist(indexNum * searchNum * k, 0);
    std::vector<faiss::idx_t> label(indexNum * searchNum * k, 0);
    try {
        Search(indexes, searchNum, query.data(), k, dist.data(), label.data(), false);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find(str) != std::string::npos);
}

TEST_P(TestAscendIndexMultiSearchInt8Pointer, InvalidInputPointer)
{
    CheckInt8PointerItem item = GetParam();
    int8_t *query = item.query;
    float *dist = item.dist;
    faiss::idx_t *label = item.label;
    faiss::MetricType metricType = item.metricType;
    int pointerNum = item.pointerNum;
    std::string str = item.str;
    std::string msg;

    std::vector<faiss::ascend::AscendIndexInt8 *> indexes;
    faiss::ascend::AscendIndexInt8FlatConfig config({ 0 }, RESOURSESIZE, BLOCKSIZE);
    for (int i = 0; i < pointerNum; ++i) {
        indexes.push_back(nullptr);
    }
    for (int i = pointerNum; i < INDEXNUM; ++i) {
        auto index = new faiss::ascend::AscendIndexInt8Flat(DIM, metricType, config);
        ASSERT_FALSE(index == nullptr);
        indexes.push_back(index);
    }
    try {
        Search(indexes, SEARCHNUM, query, K, dist, label, false);
    } catch(std::exception &e) {
        msg = e.what();
    }
    for (auto index : indexes) {
        delete index;
    }
    EXPECT_TRUE(msg.find(str) != std::string::npos);
}

TEST_P(TestAscendIndexMultiSearchInt8Indexes, InvalidInputIndexes)
{
    CheckInt8IndexItem item = GetParam();
    faiss::MetricType metricType = item.metricType;
    int dim = item.dim;
    std::vector<int> device = item.device;
    std::string str = item.str;
    std::string msg;

    std::vector<faiss::ascend::AscendIndexInt8 *> indexes;
    faiss::ascend::AscendIndexInt8FlatConfig config({ 0 }, RESOURSESIZE, BLOCKSIZE);
    faiss::ascend::AscendIndexInt8FlatConfig indexConfig(device, RESOURSESIZE, BLOCKSIZE);
    for (int i = 0; i < 1; ++i) {
        auto index = new faiss::ascend::AscendIndexInt8Flat(dim, metricType, indexConfig);
        indexes.push_back(index);
    }
    for (int i = 1; i < INDEXNUM; ++i) {
        auto index = new faiss::ascend::AscendIndexInt8Flat(DIM, faiss::METRIC_L2, config);
        ASSERT_FALSE(index == nullptr);
        indexes.push_back(index);
    }

    try {
        Search(indexes, SEARCHNUM, g_query.data(), K, g_dist.data(), g_label.data(), false);
    } catch(std::exception &e) {
        msg = e.what();
    }
    for (auto index : indexes) {
        delete index;
    }
    EXPECT_TRUE(msg.find(str) != std::string::npos);
}

INSTANTIATE_TEST_CASE_P(MultiSearchInt8CheckGroup, TestAscendIndexUTMultiSearchInt8, ::testing::ValuesIn(INT8ITEMS));
INSTANTIATE_TEST_CASE_P(MultiSearchInt8CheckGroup, TestAscendIndexMultiSearchInt8Param, ::testing::ValuesIn(PARAM));
INSTANTIATE_TEST_CASE_P(MultiSearchInt8CheckGroup, TestAscendIndexMultiSearchInt8Pointer, ::testing::ValuesIn(POINTER));
INSTANTIATE_TEST_CASE_P(MultiSearchInt8CheckGroup, TestAscendIndexMultiSearchInt8Indexes, ::testing::ValuesIn(INDEXES));

}; // namespace