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
#include <mockcpp/mockcpp.hpp>
#include <faiss/index_io.h>
#include "AscendCloner.h"
#include "AscendIndexInt8Flat.h"
#include "Common.h"
#include "ErrorCode.h"

using namespace testing;
using namespace std;
namespace {
struct CheckItem {
    int64_t resourceSize;
    faiss::ascend::Int8IndexMode indexMode;
};

class TestAscendIndexUTInt8 : public TestWithParam<CheckItem> {
};

const CheckItem INITITEMS[] = {
    { 1324 * 1024 * 1024, faiss::ascend::Int8IndexMode::PIPE_SEARCH_MODE },
    { 128 * 1024 * 1024, faiss::ascend::Int8IndexMode::DEFAULT_MODE }
};

TEST_P(TestAscendIndexUTInt8, all)
{
    CheckItem item = GetParam();
    int64_t resourceSize = item.resourceSize;
    faiss::ascend::Int8IndexMode indexMode = item.indexMode;

    int ntotal = 2000;
    int dim = 64;
    uint32_t blockSize = 16384 * 16;
    std::vector<int8_t> base(dim * ntotal);
    ascend::FeatureGenerator<int8_t>(base);

    faiss::ascend::AscendIndexInt8FlatConfig conf({ 0 }, resourceSize, blockSize, indexMode);
    faiss::ascend::AscendIndexInt8Flat index(dim, faiss::METRIC_L2, conf);
    index.verbose = true;
    index.add(ntotal, base.data());
    EXPECT_EQ(index.getNTotal(), ntotal);

    for (auto deviceId : conf.deviceList) {
        int len = index.getBaseSize(deviceId);
        ASSERT_EQ(len, ntotal / conf.deviceList.size());
        std::vector<int8_t> data(len * dim);
        index.getBase(deviceId, data);
        std::vector<faiss::idx_t> idxMap;
        index.getIdxMap(deviceId, idxMap);
    }
    size_t batch = 1;
    size_t k = 10;
    std::vector<float> dist(k * batch, 0);
    std::vector<faiss::idx_t> label(k * batch, 0);
    index.search(batch, base.data(), k, dist.data(), label.data());
    const int bit = 8;
    std::vector<uint8_t> mask(ceil(ntotal / bit), 0);
    index.search_with_masks(batch, base.data(), k, dist.data(), label.data(), static_cast<void*>(mask.data()));

    faiss::IndexScalarQuantizer cpuIndex(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit);
    index.copyTo(&cpuIndex);

    index.reset();

    std::vector<float> data(dim * ntotal);
    ascend::FeatureGenerator(data);
    faiss::IndexScalarQuantizer faissIndex(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit);
    faissIndex.train(ntotal, data.data());

    faiss::ascend::AscendIndexInt8Flat index1(&faissIndex, conf);
    faiss::ascend::AscendIndexInt8Flat index2(dim, faiss::METRIC_L2, conf);
    index2.copyFrom(&cpuIndex);
    index2.copyFrom(&faissIndex);

    faiss::IndexIDMap idIndex(&faissIndex);
    faiss::ascend::AscendIndexInt8Flat index3(&idIndex, conf);
    faiss::ascend::AscendIndexInt8Flat index4(dim, faiss::METRIC_L2, conf);
    index4.copyFrom(&idIndex);
    index.copyTo(&idIndex);

    GlobalMockObject::verify();
}

TEST(TestAscendIndexUTInt8, remove_ids)
{
    int ntotal = 2000;
    std::vector<int> deviceId { 0 };
    int dim = 64;
    int64_t resourceSize = 1 * static_cast<int64_t>(1024 * 1024 * 1024);
    uint32_t blockSize = 16384 * 16;
    faiss::ascend::AscendIndexInt8FlatConfig conf(deviceId, resourceSize, blockSize);
    faiss::ascend::AscendIndexInt8Flat index(dim, faiss::METRIC_L2, conf);
    index.verbose = true;

    std::vector<int8_t> features(ntotal * dim);
    ascend::FeatureGenerator<int8_t>(features);
    
    std::vector<int64_t> labels(ntotal);
    std::iota(labels.begin(), labels.end(), 0);

    int delRangeMin = 0;
    int delRangeMax = 49;
    index.add(ntotal, features.data());
    faiss::IDSelectorRange del(delRangeMin, delRangeMax);
    size_t rmCnt = 0;
    for (int i = 0; i < ntotal; i++) {
        rmCnt += del.is_member(labels[i]) ? 1 : 0;
    }
    size_t rmedCnt = index.remove_ids(del);
    ASSERT_EQ(rmedCnt, rmCnt);
}

TEST(TestAscendIndexUTInt8, setPageSize)
{
    int dim = 64;
    faiss::ascend::AscendIndexInt8FlatConfig conf;
    faiss::ascend::AscendIndexInt8Flat index(dim, faiss::METRIC_INNER_PRODUCT, conf);

    string msg;
    try {
        index.setPageSize(0);
    } catch (exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("pageBlockNum[0] should be in (0, 144]") != std::string::npos);

    try {
        index.setPageSize(145);
    } catch (exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("pageBlockNum[145] should be in (0, 144]") != std::string::npos);

    index.setPageSize(144);
    index.setPageSize(128);
    index.setPageSize(10);
}

TEST(TestAscendIndexUTInt8, CloneCPU2Ascend)
{
    int ntotal = 2000;
    std::vector<int> device { 0 };
    int dim = 64;
    faiss::ascend::AscendIndexInt8FlatConfig conf(device);
    faiss::ascend::AscendIndexInt8Flat index(dim, faiss::METRIC_L2, conf);
    index.verbose = true;

    std::vector<int8_t> features(ntotal * dim);
    ascend::FeatureGenerator<int8_t>(features);
    
    index.add(ntotal, features.data());

    const char *filename = "Int8.faiss";
    faiss::Index *cpuIndex = faiss::ascend::index_int8_ascend_to_cpu(&index);
    ASSERT_FALSE(cpuIndex == nullptr);
    faiss::ascend::AscendIndexInt8Flat *realIndex
        = dynamic_cast<faiss::ascend::AscendIndexInt8Flat*>(faiss::ascend::index_int8_cpu_to_ascend(device, cpuIndex));
    delete realIndex;
    delete cpuIndex;
}

INSTANTIATE_TEST_CASE_P(Int8CheckGroup, TestAscendIndexUTInt8, ::testing::ValuesIn(INITITEMS));
} // namespace