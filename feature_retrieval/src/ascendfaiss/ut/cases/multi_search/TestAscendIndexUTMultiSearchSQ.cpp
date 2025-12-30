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
#include "ut/Common.h"
#include "acl_op.h"

using namespace testing;
using namespace std;

namespace {

struct CheckItem {
    uint32_t indexNum;
    faiss::MetricType metricType;
    bool isMerge;
};

class TestAscendIndexUTMultiSearchSQ : public TestWithParam<CheckItem> {
};

const int NTOTAL = 100;
const int SQ_DIM = 64;

const CheckItem ITEMS[] = {
    { 1, faiss::METRIC_INNER_PRODUCT, false },
    { 1, faiss::METRIC_INNER_PRODUCT, true },
    { 10, faiss::METRIC_INNER_PRODUCT, false },
    { 10, faiss::METRIC_INNER_PRODUCT, true },
    { 1, faiss::METRIC_L2, false },
    { 1, faiss::METRIC_L2, true },
    { 10, faiss::METRIC_L2, false },
    { 10, faiss::METRIC_L2, true }
};

struct TestFilter {
    TestFilter()
    {
        memset_s(cameraIdMask, sizeof(cameraIdMask) / sizeof(cameraIdMask[0]),
            static_cast<uint8_t>(0xFF), 16);  // mask 长度为16
        timeRange[0] = 0;
        timeRange[1] = -1;
    }

    uint8_t cameraIdMask[16] {0};
    uint32_t timeRange[2] {0};
};

static void Release(vector<faiss::ascend::AscendIndex*> &indexes)
{
    for (size_t i = 0; i < indexes.size(); ++i) {
        if (indexes[i] != nullptr) {
            delete indexes[i];
        }
    }
}

void GenL2Data(vector<float> &data, int total, int dim)
{
#pragma omp parallel for if (total > 1)
    for (int i = 0; i < total; i++) {
        float l2data = 0;
        for (int j = 0; j < dim; j++) {
            l2data += data[i * dim + j] * data[i * dim + j];
        }
        l2data = sqrt(l2data);

        for (int j = 0; j < dim; j++) {
            data[i * dim + j] = data[i * dim + j] / l2data;
        }
    }
}

vector<float> GetData()
{
    EXPECT_TRUE(SQ_DIM > 0);
    EXPECT_TRUE(NTOTAL > 0);
    vector<float> data(SQ_DIM * NTOTAL);
    return data;
}

vector<faiss::ascend::AscendIndex*> GenAscendIndexs(uint32_t indexNum, faiss::MetricType metricType,
    bool filterable = true)
{
    vector<faiss::ascend::AscendIndex*> indexRet;
    for (uint32_t i = 0; i < indexNum; i++) {
        const uint32_t blockSize = 8 * 16384;
        const int64_t defaultMem = static_cast<int64_t>(128) * 1024 * 1024;
        const initializer_list<int> devices = { 0 };

        faiss::ascend::AscendIndexSQConfig config { devices, defaultMem, blockSize };
        config.filterable = filterable;
        faiss::ascend::AscendIndexSQ *index = new faiss::ascend::AscendIndexSQ(SQ_DIM,
            faiss::ScalarQuantizer::QuantizerType::QT_8bit, metricType, config);
        for (const auto &deviceId : config.deviceList) {
            int len = index->getBaseSize(deviceId);
            EXPECT_EQ(len, 0);
        }
        indexRet.emplace_back(index);
    }

    return indexRet;
}

TEST(TestAscendIndexUTMultiSearchSQErrPara, error_index_num_0)
{
    uint32_t indexNum = 0;
    faiss::idx_t searchNum = 2;
    faiss::idx_t k = 10;
    auto indexes = GenAscendIndexs(indexNum, faiss::METRIC_INNER_PRODUCT);
    EXPECT_EQ(indexes.size(), indexNum);
    try {
        auto data = ::GetData();
        std::vector<float> dist(indexNum * searchNum * k, 0);
        std::vector<faiss::idx_t> label(indexNum * searchNum * k, 0);
        faiss::ascend::Search(indexes, searchNum, data.data(), k, dist.data(), label.data(), false);
    } catch (std::exception &e) {
        bool isSame = strstr(e.what(), "size of indexes (0) must be > 0 and <= 10000.") != nullptr;
        EXPECT_TRUE(isSame);
    }
    Release(indexes);
}

TEST(TestAscendIndexUTMultiSearchSQErrPara, error_index_num_over)
{
    uint32_t indexNum = 10001;
    faiss::idx_t searchNum = 2;
    faiss::idx_t k = 10;
    vector<faiss::ascend::AscendIndex*> indexes(indexNum);
    EXPECT_EQ(indexes.size(), indexNum);
    try {
        auto data = ::GetData();
        std::vector<float> dist(indexNum * k * searchNum, 0);
        std::vector<faiss::idx_t> label(indexNum * k * searchNum, 0);
        faiss::ascend::Search(indexes, searchNum, data.data(), k, dist.data(), label.data(), false);
    } catch (std::exception &e) {
        bool isSame = strstr(e.what(), "size of indexes (10001) must be > 0 and <= 10000.") != nullptr;
        EXPECT_TRUE(isSame);
    }
    Release(indexes);
}

TEST(TestAscendIndexUTMultiSearchSQErrPara, error_search_num_0)
{
    uint32_t indexNum = 1;
    faiss::idx_t searchNum = 0;
    faiss::idx_t k = 10;
    auto indexes = GenAscendIndexs(indexNum, faiss::METRIC_INNER_PRODUCT);
    EXPECT_EQ(indexes.size(), indexNum);
    try {
        auto data = ::GetData();
        std::vector<float> dist(indexNum * k * searchNum, 0);
        std::vector<faiss::idx_t> label(indexNum * k * searchNum, 0);
        faiss::ascend::Search(indexes, searchNum, data.data(), k, dist.data(), label.data(), false);
    } catch (std::exception &e) {
        bool isSame = strstr(e.what(), "n must be > 0 and <= 1024") != nullptr;
        EXPECT_TRUE(isSame);
    }
    Release(indexes);
}

TEST(TestAscendIndexUTMultiSearchSQErrPara, error_search_num_over)
{
    uint32_t indexNum = 1;
    faiss::idx_t searchNum = 1025;
    faiss::idx_t k = 10;
    auto indexes = GenAscendIndexs(indexNum, faiss::METRIC_INNER_PRODUCT);
    EXPECT_EQ(indexes.size(), indexNum);
    try {
        auto data = ::GetData();
        std::vector<float> dist(indexNum * k * searchNum, 0);
        std::vector<faiss::idx_t> label(indexNum * k * searchNum, 0);
        faiss::ascend::Search(indexes, searchNum, data.data(), k, dist.data(), label.data(), false);
    } catch (std::exception &e) {
        bool isSame = strstr(e.what(), "n must be > 0 and <= 1024") != nullptr;
        EXPECT_TRUE(isSame);
    }
    Release(indexes);
}

TEST(TestAscendIndexUTMultiSearchSQErrPara, error_k_0)
{
    uint32_t indexNum = 1;
    faiss::idx_t searchNum = 10;
    faiss::idx_t k = 0;
    auto indexes = GenAscendIndexs(indexNum, faiss::METRIC_INNER_PRODUCT);
    EXPECT_EQ(indexes.size(), indexNum);
    try {
        auto data = ::GetData();
        std::vector<float> dist(indexNum * k * searchNum, 0);
        std::vector<faiss::idx_t> label(indexNum * k * searchNum, 0);
        faiss::ascend::Search(indexes, searchNum, data.data(), k, dist.data(), label.data(), false);
    } catch (std::exception &e) {
        bool isSame = strstr(e.what(), "k must be > 0 and <= 1024") != nullptr;
        EXPECT_TRUE(isSame);
    }
    Release(indexes);
}

TEST(TestAscendIndexUTMultiSearchSQErrPara, error_k_over)
{
    uint32_t indexNum = 1;
    faiss::idx_t searchNum = 10;
    faiss::idx_t k = 1025;
    auto indexes = GenAscendIndexs(indexNum, faiss::METRIC_INNER_PRODUCT);
    EXPECT_EQ(indexes.size(), indexNum);
    try {
        auto data = ::GetData();
        std::vector<float> dist(indexNum * k * searchNum, 0);
        std::vector<faiss::idx_t> label(indexNum * k * searchNum, 0);
        faiss::ascend::Search(indexes, searchNum, data.data(), k, dist.data(), label.data(), false);
    } catch (std::exception &e) {
        bool isSame = strstr(e.what(), "k must be > 0 and <= 1024") != nullptr;
        EXPECT_TRUE(isSame);
    }
    Release(indexes);
}

TEST(TestAscendIndexUTMultiSearchSQErrPara, error_querry_data_null)
{
    uint32_t indexNum = 1;
    faiss::idx_t searchNum = 10;
    faiss::idx_t k = 10;
    auto indexes = GenAscendIndexs(indexNum, faiss::METRIC_INNER_PRODUCT);
    EXPECT_EQ(indexes.size(), indexNum);
    try {
        std::vector<float> dist(indexNum * k * searchNum, 0);
        std::vector<faiss::idx_t> label(indexNum * k * searchNum, 0);
        faiss::ascend::Search(indexes, searchNum, nullptr, k, dist.data(), label.data(), false);
    } catch (std::exception &e) {
        bool isSame = strstr(e.what(), "Invalid x: nullptr.") != nullptr;
        EXPECT_TRUE(isSame);
    }
    Release(indexes);
}

TEST(TestAscendIndexUTMultiSearchSQErrPara, error_distance_data_null)
{
    uint32_t indexNum = 1;
    faiss::idx_t searchNum = 10;
    faiss::idx_t k = 10;
    auto indexes = GenAscendIndexs(indexNum, faiss::METRIC_INNER_PRODUCT);
    EXPECT_EQ(indexes.size(), indexNum);
    try {
        auto data = ::GetData();
        std::vector<faiss::idx_t> label(indexNum * k * searchNum, 0);
        faiss::ascend::Search(indexes, searchNum, data.data(), k, nullptr, label.data(), false);
    } catch (std::exception &e) {
        bool isSame = strstr(e.what(), "Invalid distances: nullptr.") != nullptr;
        EXPECT_TRUE(isSame);
    }
    Release(indexes);
}

TEST(TestAscendIndexUTMultiSearchSQErrPara, error_label_data_null)
{
    uint32_t indexNum = 1;
    faiss::idx_t searchNum = 10;
    faiss::idx_t k = 10;
    auto indexes = GenAscendIndexs(indexNum, faiss::METRIC_INNER_PRODUCT);
    EXPECT_EQ(indexes.size(), indexNum);
    try {
        auto data = ::GetData();
        std::vector<float> dist(indexNum * k * searchNum, 0);
        faiss::ascend::Search(indexes, searchNum, data.data(), k, dist.data(), nullptr, false);
    } catch (std::exception &e) {
        bool isSame = strstr(e.what(), "Invalid labels: nullptr.") != nullptr;
        EXPECT_TRUE(isSame);
    }
    Release(indexes);
}

TEST(TestAscendIndexUTMultiSearchSQErrPara, error_first_index_null)
{
    uint32_t indexNum = 1;
    faiss::idx_t searchNum = 10;
    faiss::idx_t k = 10;
    vector<faiss::ascend::AscendIndex*> indexes(indexNum, nullptr);
    EXPECT_EQ(indexes.size(), indexNum);
    try {
        auto data = ::GetData();
        std::vector<float> dist(indexNum * k * searchNum, 0);
        std::vector<faiss::idx_t> label(indexNum * k * searchNum, 0);
        faiss::ascend::Search(indexes, searchNum, data.data(), k, dist.data(), label.data(), false);
    } catch (std::exception &e) {
        bool isSame = strstr(e.what(), "Invalid index 0 from given indexes: nullptr.") != nullptr;
        EXPECT_TRUE(isSame);
    }
    Release(indexes);
}

TEST(TestAscendIndexUTMultiSearchSQErrPara, error_first_index_not_trained)
{
    uint32_t indexNum = 1;
    faiss::idx_t searchNum = 10;
    faiss::idx_t k = 10;
    auto data = ::GetData();
    auto indexes = GenAscendIndexs(indexNum, faiss::METRIC_INNER_PRODUCT);
    EXPECT_EQ(indexes.size(), indexNum);
    try {
        std::vector<float> dist(indexNum * k * searchNum, 0);
        std::vector<faiss::idx_t> label(indexNum * k * searchNum, 0);
        faiss::ascend::Search(indexes, searchNum, data.data(), k, dist.data(), label.data(), false);
    } catch (std::exception &e) {
        bool isSame = strstr(e.what(), "Index 0 not trained") != nullptr;
        EXPECT_TRUE(isSame);
    }
    Release(indexes);
}

TEST_F(TestAscendIndexUTMultiSearchSQ,
    SearchWithFilter_throw_exception_with_input_indexes_do_not_support_filterable)
{
    uint32_t indexNum = 2;
    faiss::idx_t searchNum = 1;
    faiss::idx_t k = 1;
    auto data = ::GetData();
    auto indexes = GenAscendIndexs(indexNum, faiss::METRIC_INNER_PRODUCT, false);
    EXPECT_EQ(indexes.size(), indexNum);

    for (auto index : indexes) {
        index->train(NTOTAL, data.data());
        index->add(NTOTAL, data.data());
    }

    std::string realMsg;
    std::vector<float> dist(indexNum * k * searchNum, 0);
    std::vector<faiss::idx_t> label(indexNum * k * searchNum, 0);
    std::vector<TestFilter> filter(searchNum);
    try {
        faiss::ascend::SearchWithFilter(indexes, searchNum, data.data(), k, dist.data(), label.data(), filter.data(),
            false);
    } catch (std::exception &e) {
        realMsg = e.what();
    }

    const std::string errMsg = "the index does not support filterable";
    EXPECT_TRUE(realMsg.find(realMsg) != std::string::npos);

    Release(indexes);
}

TEST_P(TestAscendIndexUTMultiSearchSQ, multi_index_search_with_single_filter)
{
    CheckItem item = GetParam();
    uint32_t indexNum = item.indexNum;
    faiss::MetricType metricType = item.metricType;
    bool isMerge = item.isMerge;

    auto indexs = GenAscendIndexs(indexNum, metricType);
    EXPECT_EQ(indexs.size(), indexNum);

    auto data = ::GetData();
    // filter
    TestFilter filter;
    filter.timeRange[0] = 0;
    filter.timeRange[1] = 0x7fffffff;
    int seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine e1(seed);
    uniform_int_distribution<int32_t> id(0, numeric_limits<int32_t>::max());
    uniform_int_distribution<uint8_t> searchCid(0, 127);

    vector<int64_t> ids(NTOTAL, 0);
    for (int i = 0; i < NTOTAL; i++) {
        ids[i] = (static_cast<int64_t>(searchCid(e1)) << 42) + (static_cast<int64_t>(id(e1)) << 10);
    }

    for (uint32_t j = 0; j < sizeof(filter.cameraIdMask); j++) {
        filter.cameraIdMask[j] = searchCid(e1);
    }

    for (size_t k = 0; k < indexs.size(); k++) {
        indexs[k]->train(NTOTAL, data.data());
        indexs[k]->add_with_ids(NTOTAL, data.data(), ids.data());
        EXPECT_EQ(indexs[k]->ntotal, NTOTAL);
    }

    faiss::idx_t searchNum = 1;
    faiss::idx_t k = 10;

    vector<float> distance(indexNum * searchNum * k, 0);
    vector<faiss::idx_t> labels(indexNum * searchNum * k, 0);

    SearchWithFilter(indexs, searchNum, data.data(), k, distance.data(), labels.data(), &filter, isMerge);
    EXPECT_EQ(distance.size(), indexNum * searchNum * k);
    EXPECT_EQ(labels.size(), indexNum * searchNum * k);

    for (size_t i = 0; i < indexs.size(); i ++) {
        if (indexs[i] != nullptr) {
            delete indexs[i];
            indexs[i] = nullptr;
        }
    }
}

TEST_P(TestAscendIndexUTMultiSearchSQ, multi_index_search_with_multi_filter)
{
    CheckItem item = GetParam();
    uint32_t indexNum = item.indexNum;
    faiss::MetricType metricType = item.metricType;
    bool isMerge = item.isMerge;

    auto indexs = GenAscendIndexs(indexNum, metricType);
    EXPECT_EQ(indexs.size(), indexNum);

    auto data = ::GetData();
    int seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine e1(seed);
    uniform_int_distribution<int32_t> id(0, numeric_limits<int32_t>::max());
    uniform_int_distribution<uint8_t> searchCid(0, 127);

    vector<int64_t> ids(NTOTAL, 0);
    for (int i = 0; i < NTOTAL; i++) {
        ids[i] = (static_cast<int64_t>(searchCid(e1)) << 42) + (static_cast<int64_t>(id(e1)) << 10);
    }

    for (size_t k = 0; k < indexs.size(); k++) {
        indexs[k]->train(NTOTAL, data.data());
        indexs[k]->add_with_ids(NTOTAL, data.data(), ids.data());
        EXPECT_EQ(indexs[k]->ntotal, NTOTAL);
    }

    faiss::idx_t searchNum = 2;
    faiss::idx_t k = 10;

    void *multiFilters[searchNum];
    TestFilter idFilters[indexNum * searchNum];
    for (int queryIdx = 0; queryIdx < searchNum; queryIdx++) {
        for (uint32_t indexIdx = 0; indexIdx < indexNum; indexIdx++) {
            TestFilter idFilter;
            idFilter.timeRange[0] = 0;
            idFilter.timeRange[1] = 0x7fffffff;
            for (int i = 0; i < 16; i++) {
                idFilter.cameraIdMask[i] = searchCid(e1);
            }
            idFilters[queryIdx * indexNum + indexIdx] = idFilter;
        }
        multiFilters[queryIdx] = &idFilters[queryIdx * indexNum];
    }

    vector<float> distance(indexNum * searchNum * k, 0);
    vector<faiss::idx_t> labels(indexNum * searchNum * k, 0);

    SearchWithFilter(indexs, searchNum, data.data(), k, distance.data(), labels.data(), multiFilters, isMerge);
    EXPECT_EQ(distance.size(), indexNum * searchNum * k);
    EXPECT_EQ(labels.size(), indexNum * searchNum * k);

    for (size_t i = 0; i < indexs.size(); i ++) {
        if (indexs[i] != nullptr) {
            delete indexs[i];
            indexs[i] = nullptr;
        }
    }
}

TEST_P(TestAscendIndexUTMultiSearchSQ, multi_index_search)
{
    CheckItem item = GetParam();
    uint32_t indexNum = item.indexNum;
    faiss::MetricType metricType = item.metricType;
    bool isMerge = item.isMerge;

    faiss::ascend::AscendIndexSQConfig conf({ 0 });

    auto data = ::GetData();
    
    auto indexes = GenAscendIndexs(indexNum, metricType);
    EXPECT_EQ(indexes.size(), indexNum);

    for (uint32_t i = 0; i < indexNum; ++i) {
        indexes[i]->train(NTOTAL, data.data());
        indexes[i]->add(NTOTAL, data.data());
        EXPECT_EQ(indexes[i]->ntotal, NTOTAL);
    }

    faiss::idx_t searchNum = 2;
    faiss::idx_t k = 10;

    std::vector<float> dist(indexNum * k * searchNum, 0);
    std::vector<faiss::idx_t> label(indexNum * k * searchNum, 0);
    faiss::ascend::Search(indexes, searchNum, data.data(), k, dist.data(), label.data(), isMerge);
    EXPECT_EQ(dist.size(), indexNum * searchNum * k);
    EXPECT_EQ(label.size(), indexNum * searchNum * k);

    Release(indexes);
}

INSTANTIATE_TEST_CASE_P(SearchWithFilterCheckGroup, TestAscendIndexUTMultiSearchSQ, ::testing::ValuesIn(ITEMS));

}; // namespace