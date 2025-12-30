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
#include "AscendMultiIndexSearch.h"
#include "securec.h"
#include "acl_op.h"

using namespace testing;
using namespace std;

namespace {

struct CheckFlatItem {
    uint32_t indexNum;
    faiss::MetricType metricType;
    int dim;
};

struct CheckFlatParamItem {
    uint32_t indexNum;
    faiss::MetricType metricType;
    int dim;
    int searchNum;
    int k;
    std::string str;
};

struct CheckFlatPointerItem {
    float *query;
    float *dist;
    faiss::idx_t *label;
    faiss::MetricType metricType;
    int pointerNum;
    std::string str;
};

struct CheckFlatIndexItem {
    faiss::MetricType metricType;
    int dim;
    std::vector<int> device;
    std::string str;
};

class TestAscendIndexUTMultiSearchFlat : public TestWithParam<CheckFlatItem> {
};

class TestAscendIndexMultiSearchFlatParam : public TestWithParam<CheckFlatParamItem> {
};

class TestAscendIndexMultiSearchFlatPointer : public TestWithParam<CheckFlatPointerItem> {
};

class TestAscendIndexMultiSearchFlatIndexes : public TestWithParam<CheckFlatIndexItem> {
};

constexpr int DIM = 256;
constexpr int SEARCHNUM = 10;
constexpr int INDEXNUM = 2;
constexpr int K = 10;
constexpr size_t NTOTAL = 1000;
std::vector<float> g_query(DIM * SEARCHNUM);
std::vector<float> g_dist(INDEXNUM * SEARCHNUM * K);
std::vector<faiss::idx_t> g_label(INDEXNUM * SEARCHNUM * K);

const CheckFlatItem FLATITEMS[] = {
    { 1, faiss::METRIC_INNER_PRODUCT },
    { 10, faiss::METRIC_INNER_PRODUCT },
    { 1, faiss::METRIC_L2 },
    { 10, faiss::METRIC_L2 }
};

const CheckFlatParamItem PARAM[] = {
    { 0, faiss::METRIC_INNER_PRODUCT, 256, 1000, 10, "size of indexes (0) must be > 0 and <= 10000." },
    { 10001, faiss::METRIC_INNER_PRODUCT, 256, 1, 3,  "size of indexes (10001) must be > 0 and <= 10000." },
    { 2, faiss::METRIC_INNER_PRODUCT, 256, 0, 10, "n must be > 0 and <= 1024" },
    { 2, faiss::METRIC_INNER_PRODUCT, 256, 1025, 10, "n must be > 0 and <= 1024" },
    { 2, faiss::METRIC_INNER_PRODUCT, 256, 1000, 0, "k must be > 0 and <= 1024" },
    { 2, faiss::METRIC_INNER_PRODUCT, 256, 1000, 1025, "k must be > 0 and <= 1024" },
    { 0, faiss::METRIC_L2, 256, 1000, 10, "size of indexes (0) must be > 0 and <= 10000." },
    { 10001, faiss::METRIC_L2, 256, 2, 4,  "size of indexes (10001) must be > 0 and <= 10000." },
    { 2, faiss::METRIC_L2, 256, 0, 10, "n must be > 0 and <= 1024" },
    { 2, faiss::METRIC_L2, 256, 1025, 10, "n must be > 0 and <= 1024" },
    { 2, faiss::METRIC_L2, 256, 1000, 0, "k must be > 0 and <= 1024" },
    { 2, faiss::METRIC_L2, 256, 1000, 1025, "k must be > 0 and <= 1024" }
};

const CheckFlatPointerItem POINTER[] = {
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

const CheckFlatIndexItem INDEXES[] = {
    { faiss::METRIC_INNER_PRODUCT, 256, { 0 }, "the metric type must be same" },
    { faiss::METRIC_L2, 512, { 0 }, "the dim must be same." },
    { faiss::METRIC_L2, 256, { 1 }, "the deviceList must be same." },
    { faiss::METRIC_L2, 256, { 0, 1 }, "the number of deviceList (2) must be 1." }
};

void FeatureGenerator(std::vector<float> &features)
{
    size_t n = features.size();
    for (size_t i = 0; i < n; ++i) {
        features[i] = drand48();
    }
}

TEST_P(TestAscendIndexUTMultiSearchFlat, MultiFlatIndexSearch)
{
    CheckFlatItem item = GetParam();
    int indexNum = item.indexNum;
    faiss::MetricType metricType = item.metricType;

    std::vector<std::vector<float>> data(indexNum, std::vector<float>(DIM * NTOTAL));

    for (int i = 0; i < indexNum; i++) {
        FeatureGenerator(data[i]);
    }

    std::vector<faiss::ascend::AscendIndex *> indexes;
    faiss::ascend::AscendIndexFlatConfig config({ 0 });
    for (int i = 0; i < indexNum; ++i) {
        auto index = new faiss::ascend::AscendIndexFlat(DIM, metricType, config);
        ASSERT_FALSE(index == nullptr);
        indexes.push_back(index);
    }

    for (int i = 0; i < indexNum; ++i) {
        indexes[i]->add(NTOTAL, data[i].data());
    }

    std::vector<float> dist(indexNum * SEARCHNUM * K, 0);
    std::vector<faiss::idx_t> label(indexNum * SEARCHNUM * K, 0);
    std::vector<float> query(DIM * SEARCHNUM);
    FeatureGenerator(query);
    Search(indexes, SEARCHNUM, query.data(), K, dist.data(), label.data(), false);

    std::vector<float> mergeDist(SEARCHNUM * K, 0);
    std::vector<faiss::idx_t> mergeLabel(SEARCHNUM * K, 0);
    Search(indexes, SEARCHNUM, query.data(), K, mergeDist.data(), mergeLabel.data(), true);
    for (int i = 0; i < indexNum; ++i) {
        delete indexes[i];
    }
}

TEST_P(TestAscendIndexMultiSearchFlatParam, InvalidInputParam)
{
    CheckFlatParamItem item = GetParam();
    int indexNum = item.indexNum;
    faiss::MetricType metricType = item.metricType;
    int dim = item.dim;
    int searchNum = item.searchNum;
    int k = item.k;
    std::string str = item.str;
    std::string msg;

    std::vector<faiss::ascend::AscendIndex *> indexes(indexNum);
    std::vector<float> query(dim * searchNum);
    std::vector<float> dist(indexNum * searchNum * k, 0);
    std::vector<faiss::idx_t> label(indexNum * searchNum * k, 0);
    try {
        Search(indexes, searchNum, query.data(), k, dist.data(), label.data(), false);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find(str) != std::string::npos);
}

TEST_P(TestAscendIndexMultiSearchFlatPointer, InvalidInputPointer)
{
    CheckFlatPointerItem item = GetParam();
    float *query = item.query;
    float *dist = item.dist;
    faiss::idx_t *label = item.label;
    faiss::MetricType metricType = item.metricType;
    int pointerNum = item.pointerNum;
    std::string str = item.str;
    std::string msg;

    std::vector<faiss::ascend::AscendIndex *> indexes;
    faiss::ascend::AscendIndexFlatConfig config({ 0 });
    for (int i = 0; i < pointerNum; ++i) {
        indexes.push_back(nullptr);
    }
    for (int i = pointerNum; i < INDEXNUM; ++i) {
        auto index = new faiss::ascend::AscendIndexFlat(DIM, metricType, config);
        ASSERT_FALSE(index == nullptr);
        indexes.push_back(index);
    }
    try {
        Search(indexes, SEARCHNUM, query, K, dist, label, false);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find(str) != std::string::npos);
    for (int i = 0; i < INDEXNUM; ++i) {
        delete indexes[i];
    }
}

TEST_P(TestAscendIndexMultiSearchFlatIndexes, InvalidInputIndexes)
{
    CheckFlatIndexItem item = GetParam();
    faiss::MetricType metricType = item.metricType;
    int dim = item.dim;
    std::vector<int> device = item.device;
    std::string str = item.str;
    std::string msg;

    std::vector<faiss::ascend::AscendIndex *> indexes;
    faiss::ascend::AscendIndexFlatConfig config({ 0 });
    faiss::ascend::AscendIndexFlatConfig indexConfig(device);
    for (int i = 0; i < 1; ++i) {
        auto index = new faiss::ascend::AscendIndexFlat(dim, metricType, indexConfig);
        indexes.push_back(index);
    }
    for (int i = 1; i < INDEXNUM; ++i) {
        auto index = new faiss::ascend::AscendIndexFlat(DIM, faiss::METRIC_L2, config);
        ASSERT_FALSE(index == nullptr);
        indexes.push_back(index);
    }

    try {
        Search(indexes, SEARCHNUM, g_query.data(), K, g_dist.data(), g_label.data(), false);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find(str) != std::string::npos);
    for (int i = 0; i < INDEXNUM; ++i) {
        delete indexes[i];
    }
}

INSTANTIATE_TEST_CASE_P(MultiSearchFlatCheckGroup, TestAscendIndexUTMultiSearchFlat, ::testing::ValuesIn(FLATITEMS));
INSTANTIATE_TEST_CASE_P(MultiSearchFlatCheckGroup, TestAscendIndexMultiSearchFlatParam, ::testing::ValuesIn(PARAM));
INSTANTIATE_TEST_CASE_P(MultiSearchFlatCheckGroup, TestAscendIndexMultiSearchFlatPointer, ::testing::ValuesIn(POINTER));
INSTANTIATE_TEST_CASE_P(MultiSearchFlatCheckGroup, TestAscendIndexMultiSearchFlatIndexes, ::testing::ValuesIn(INDEXES));

}; // namespace