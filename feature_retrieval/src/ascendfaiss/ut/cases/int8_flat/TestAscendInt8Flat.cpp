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
#include "common/utils/SocUtils.h"
#include "ut/Common.h"
#include "faiss/ascend/AscendIndexInt8Flat.h"
#include "faiss/impl/AuxIndexStructures.h"
#include "faiss/impl/IDSelector.h"
#include "acl.h"
#include "fp16.h"

using namespace testing;
using namespace std;
namespace {
const int64_t MAX_N = 1e9;
const int64_t MAX_K = 4096;
struct CheckInitItem {
    size_t dim;
    std::vector<int> deviceList;
    faiss::MetricType metricType;
    int64_t resourceSize;
    uint32_t dBlockSize;
    faiss::ascend::Int8IndexMode dIndexMode;
    std::string str;
};

struct CheckCopyItem {
    size_t dim;
    int64_t ntotal;
    faiss::MetricType metricType;
    std::string str;
};
struct CheckItem {
    uint32_t ntotal;
    faiss::MetricType metricType;
};

struct AddCheckItem {
    uint32_t ntotal;
    faiss::MetricType metricType;
};

struct Check910BItem {
    size_t dim;
    uint32_t ntotal;
    faiss::MetricType metricType;
    std::string str;
};

class TestInt8FlatInitUT : public TestWithParam<CheckInitItem> {
};

class TestInt8FlatCopyUT : public TestWithParam<CheckCopyItem> {
};
class TestInt8FlatAddUT : public TestWithParam<AddCheckItem> {};
class TestAscendIndexInt8FlatUT : public TestWithParam<CheckItem> {
};

class TestAscendIndexInt8FlatUT910B : public TestWithParam<Check910BItem> {
};

const int DIM = 64;
std::vector<int> DEIVCE = { 0 };
const int RANGEMIN = 0;
const int RANGEMAX = 4;
const int64_t DEFAULT_MEM = 0x8000000;
const CheckInitItem INITITEMS[] = {
    { 192, { 0 }, faiss::METRIC_INNER_PRODUCT, DEFAULT_MEM, 262144, faiss::ascend::Int8IndexMode::DEFAULT_MODE,
        "Unsupported dims" },
    { 512, { 0 }, faiss::METRIC_L1, DEFAULT_MEM, 262144, faiss::ascend::Int8IndexMode::DEFAULT_MODE,
        "Unsupported metric type" },
    { 512, { 0 }, faiss::METRIC_L2, -1000, 262144, faiss::ascend::Int8IndexMode::DEFAULT_MODE,
        "resourceSize should be -1 or in range [0, 16GB]!"},
    { 512, { 0 }, faiss::METRIC_L2, DEFAULT_MEM, 1000, faiss::ascend::Int8IndexMode::DEFAULT_MODE,
        " Unsupported blockSize 1000! "},
    { 512, { 0 }, faiss::METRIC_L2, DEFAULT_MEM, 262144, faiss::ascend::Int8IndexMode::WITHOUT_NORM_MODE,
        "Unsupported dIndexMode WITHOUT_NORM_MODE"},
    { 512, { 0 }, faiss::METRIC_INNER_PRODUCT, DEFAULT_MEM, 262144, faiss::ascend::Int8IndexMode::PIPE_SEARCH_MODE,
        "Unsupported metric type, should be METRIC_L2"}
};

const CheckCopyItem COPYITEMS[] = {
    { 192, 1000, faiss::METRIC_INNER_PRODUCT, "Unsupported dims" },
    { 512, 1000, faiss::METRIC_L1, "Unsupported metric type" },
    { 192, 1000, faiss::METRIC_L2, "Unsupported dims" },
};

const CheckItem ITEMS[] = {
    { 1000, faiss::METRIC_INNER_PRODUCT },
    { 300000, faiss::METRIC_L2 }
};

const AddCheckItem ADDITEMS[] = {
    { 1000, faiss::METRIC_INNER_PRODUCT },
    { 1000, faiss::METRIC_L2 },
    { 524289, faiss::METRIC_INNER_PRODUCT }
};

const Check910BItem ITEMS910B[] = {
    { 64, 1000, faiss::METRIC_INNER_PRODUCT, "Ascend910B1" },
    { 128, 30000, faiss::METRIC_INNER_PRODUCT, "Ascend910B2" },
    { 256, 1000, faiss::METRIC_L2, "Ascend910B3" },
    { 384, 30000, faiss::METRIC_L2, "Ascend910B4" }
};

TEST_P(TestInt8FlatInitUT, Int8FlatInitInvalidInput)
{
    std::string msg;
    CheckInitItem item = GetParam();
    std::vector<int> deviceList = item.deviceList;
    faiss::MetricType metricType = item.metricType;
    size_t dim = item.dim;
    int64_t resourceSize = item.resourceSize;
    uint32_t blockSize = item.dBlockSize;
    faiss::ascend::Int8IndexMode indexMode = item.dIndexMode;

    std::string str = item.str;

    try {
        faiss::ascend::AscendIndexInt8FlatConfig config(deviceList, resourceSize, blockSize, indexMode);
        faiss::ascend::AscendIndexInt8Flat index(dim, metricType, config);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find(str) != std::string::npos);
}

TEST_P(TestInt8FlatCopyUT, Int8FlatCopyInvalidInput)
{
    std::string msg;
    CheckCopyItem item = GetParam();
    int64_t ntotal = item.ntotal;
    faiss::MetricType metricType = item.metricType;
    size_t dim = item.dim;
    std::string str = item.str;

    std::vector<float> data(dim * ntotal);
    faiss::IndexScalarQuantizer cpuIndex(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit, metricType);
    cpuIndex.train(ntotal, data.data());
    cpuIndex.add(ntotal, data.data());

    try {
        faiss::ascend::AscendIndexInt8FlatConfig config({ 0 });
        faiss::ascend::AscendIndexInt8Flat index(dim, metricType, config);
        index.copyFrom(&cpuIndex);
    } catch(std::exception &e) {
        msg = e.what();
    }

    EXPECT_TRUE(msg.find(str) != std::string::npos);
}

void TestAdd(const std::vector<int8_t>& data, uint32_t ntotal, faiss::MetricType metricType, int dim)
{
    std::initializer_list<int> devices = { 0 };
    faiss::ascend::AscendIndexInt8FlatConfig config(devices);
    faiss::ascend::AscendIndexInt8Flat index(dim, metricType, config);

    for (auto id : config.deviceList) {
        int len = index.getBaseSize(id);
        ASSERT_EQ(len, 0);
    }

    index.add(ntotal, data.data());
    EXPECT_EQ(index.getNTotal(), ntotal);

    int deviceCnt = config.deviceList.size();
    int totals = 0;
    for (int i = 0; i < deviceCnt; i++) {
        int tmpTotal = index.getBaseSize(config.deviceList[i]);
        totals += tmpTotal;
    }
    EXPECT_EQ(totals, ntotal);
}

TEST_P(TestInt8FlatAddUT, add)
{
    AddCheckItem item = GetParam();
    uint32_t ntotal = item.ntotal;
    faiss::MetricType metricType = item.metricType;
    int dim = 64;

    std::vector<int8_t> data(dim * ntotal, 1);
    TestAdd(data, ntotal, metricType, dim);
}

TEST_P(TestAscendIndexInt8FlatUT, Copy)
{
    CheckItem item = GetParam();
    uint32_t ntotal = item.ntotal;
    int dim = 64;

    faiss::ascend::AscendIndexInt8FlatConfig conf({ 0 });
    faiss::ascend::AscendIndexInt8Flat index(dim, faiss::METRIC_L2, conf);
    faiss::IndexScalarQuantizer cpuIndex(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit);
    index.copyTo(&cpuIndex);

    index.reset();

    std::vector<float> data(dim * ntotal, 1);
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
}

void TestAddWithIds(const std::vector<int8_t>& data, uint32_t ntotal, faiss::MetricType metricType, int dim)
{
    std::vector<faiss::idx_t> ids(ntotal, 0);
    std::iota(ids.begin(), ids.end(), 0);

    faiss::ascend::AscendIndexInt8FlatConfig conf({ 0 });
    faiss::ascend::AscendIndexInt8Flat index(dim, metricType, conf);

    for (auto device : conf.deviceList) {
        int total = index.getBaseSize(device);
        ASSERT_EQ(total, 0);
    }

    index.add_with_ids(ntotal, data.data(), ids.data());
    EXPECT_EQ(index.getNTotal(), ntotal);

    int totals = 0;
    for (size_t i = 0; i < conf.deviceList.size(); i++) {
        int tmpTotal = index.getBaseSize(conf.deviceList[i]);
        totals += tmpTotal;
    }
    EXPECT_EQ(totals, ntotal);
}

TEST_P(TestAscendIndexInt8FlatUT, AddWithIds)
{
    CheckItem item = GetParam();
    uint32_t ntotal = item.ntotal;
    faiss::MetricType metricType = item.metricType;
    int dim = 64;

    std::vector<int8_t> data(dim * ntotal);
    TestAddWithIds(data, ntotal, metricType, dim);
}


void TestAddRemoveSearch(const std::vector<int8_t>& data, uint32_t ntotal, faiss::MetricType metricType,
    int dim, std::string socName)
{
    // 打桩socName为910B*
    MOCKER_CPP(&aclrtGetSocName).stubs().will(returnValue(socName.c_str()));
    faiss::ascend::SocUtils::GetInstance().Init();

    faiss::ascend::AscendIndexInt8FlatConfig conf({ 0 });
    faiss::ascend::AscendIndexInt8Flat index(dim, metricType, conf);

    EXPECT_TRUE(faiss::ascend::SocUtils::GetInstance().IsAscend910B());
    // 910B为ND格式
    EXPECT_FALSE(faiss::ascend::SocUtils::GetInstance().IsZZCodeFormat());
    EXPECT_EQ(faiss::ascend::SocUtils::CodeFormatType::FORMAT_TYPE_ND,
        faiss::ascend::SocUtils::GetInstance().GetCodeFormatType());

    for (auto device : conf.deviceList) {
        int total = index.getBaseSize(device);
        ASSERT_EQ(total, 0);
    }

    std::vector<faiss::idx_t> ids(ntotal, 0);
    std::iota(ids.begin(), ids.end(), 0);
    index.add_with_ids(ntotal, data.data(), ids.data());
    EXPECT_EQ(index.getNTotal(), ntotal);

    size_t batch = 1;
    size_t k = 10;
    std::vector<float> dist(k * batch, 0);
    std::vector<faiss::idx_t> label(k * batch, 0);
    index.search(batch, data.data(), k, dist.data(), label.data());

    int totals = 0;
    for (size_t i = 0; i < conf.deviceList.size(); i++) {
        int tmpTotal = index.getBaseSize(conf.deviceList[i]);
        totals += tmpTotal;
    }
    EXPECT_EQ(totals, ntotal);

    size_t delRangeMin = 0;
    size_t delRangeMax = 10;
    faiss::IDSelectorRange del(delRangeMin, delRangeMax);
    // ascend index delete feature
    auto removeCnt = index.remove_ids(del);
    EXPECT_EQ(removeCnt, delRangeMax - delRangeMin);

    // 插入删除的特征
    index.add_with_ids(removeCnt, data.data() + delRangeMin * dim, ids.data() + delRangeMin);
    EXPECT_EQ(index.getNTotal(), ntotal);

    index.search(batch, data.data(), k, dist.data(), label.data());

    // mockcpp 需要显示调用该函数来恢复打桩
    GlobalMockObject::verify();
}

TEST_P(TestAscendIndexInt8FlatUT910B, AddRemoveSearch)
{
    Check910BItem item = GetParam();
    uint32_t ntotal = item.ntotal;
    faiss::MetricType metricType = item.metricType;
    std::string socName = item.str;

    int dim = 64;
    std::vector<int8_t> data(dim * ntotal);
    TestAddRemoveSearch(data, ntotal, metricType, dim, socName);
}

TEST_P(TestAscendIndexInt8FlatUT, removeRange)
{
    CheckItem item = GetParam();
    uint32_t ntotal = item.ntotal;
    faiss::MetricType metricType = item.metricType;

    std::vector<int8_t> data(DIM * ntotal);

    faiss::ascend::AscendIndexInt8FlatConfig conf({ 0 });
    faiss::ascend::AscendIndexInt8Flat *index = new faiss::ascend::AscendIndexInt8Flat(DIM, metricType, conf);

    index->add(ntotal, data.data());
    // define ids
    std::vector<faiss::idx_t> ids(ntotal);
    std::iota(ids.begin(), ids.end(), 0);

    // save pre-delete basedata and index of vector
    std::vector<int8_t> preXb;
    std::vector<faiss::idx_t> preIdxMap;
    for (auto device : DEIVCE) {
        size_t baseSize = index->getBaseSize(device);
        std::vector<int8_t> base(baseSize * DIM);
        index->getBase(device, base);
        preXb.insert(preXb.end(), base.begin(), base.end());

        std::vector<faiss::idx_t> idMap(baseSize);
        index->getIdxMap(device, idMap);
        preIdxMap.insert(preIdxMap.end(), idMap.begin(), idMap.end());
    }

    faiss::IDSelectorRange del(RANGEMIN, RANGEMAX);
    int rmNum = 0;
    for (int i = 0; i < ntotal; i++) {
        rmNum += del.is_member(ids[i]) ? 1 : 0;
    }

    size_t rmedCnt = index->remove_ids(del);
    ASSERT_EQ(rmedCnt, rmNum);
    ASSERT_EQ(index->getNTotal(), (ntotal - rmedCnt));

    int total = 0;
    for (auto device : DEIVCE) {
        total += index->getBaseSize(device);
    }
    EXPECT_EQ(total, (ntotal - rmedCnt));
    delete index;
}


void checkRemoveBatch(std::vector<int8_t>& preXb, faiss::IDSelectorBatch del,
    std::vector<faiss::idx_t>& preIdxMap, size_t rmedCnt, faiss::ascend::AscendIndexInt8Flat *index)
{
    std::vector<int8_t> xb;
    std::vector<faiss::idx_t> idxMap;
    for (auto device : DEIVCE) {
        size_t size = index->getBaseSize(device);
        std::vector<int8_t> base(size * DIM);
        index->getBase(device, base);
        xb.insert(xb.end(), base.begin(), base.end());

        std::vector<faiss::idx_t> idMap(size);
        index->getIdxMap(device, idMap);
        idxMap.insert(idxMap.end(), idMap.begin(), idMap.end());
    }
    EXPECT_EQ(preIdxMap.size(), idxMap.size() + rmedCnt);
    EXPECT_EQ(preXb.size(), xb.size() + rmedCnt * DIM);
    {
        int offset = 0;
        for (size_t i = 0; i < idxMap.size(); i++) {
            if (del.set.find(preIdxMap[i]) != del.set.end()) {
                // check ids
                EXPECT_EQ(preIdxMap[idxMap.size() + offset], idxMap[i]);
                // check vector
                for (int j = 0; j < DIM; j++) {
                    EXPECT_EQ(preXb[(idxMap.size() + offset) * DIM + j], xb[i * DIM + j]);
                }

                offset += 1;
            } else {
                int ptr = i * DIM;
                // check ids
                EXPECT_EQ(preIdxMap[i], idxMap[i]);
                // check vector
                for (int j = 0; j < DIM; j++) {
                    EXPECT_EQ(preXb[ptr + j], xb[ptr + j]);
                }
            }
        }
        EXPECT_EQ(offset, del.set.size());
    }
}

TEST_P(TestAscendIndexInt8FlatUT, removeBatch)
{
    CheckItem item = GetParam();
    uint32_t ntotal = item.ntotal;
    faiss::MetricType metricType = item.metricType;

    std::vector<faiss::idx_t> delBatchs = { 1, 8 };

    std::vector<int8_t> data(DIM * ntotal);

    faiss::ascend::AscendIndexInt8FlatConfig conf({ 0 });
    faiss::ascend::AscendIndexInt8Flat *index = new faiss::ascend::AscendIndexInt8Flat(DIM, metricType, conf);

    index->add(ntotal, data.data());
    // define ids
    std::vector<faiss::idx_t> ids(ntotal);
    std::iota(ids.begin(), ids.end(), 0);

    // save pre-delete basedata and index of vector
    std::vector<int8_t> preXb;
    std::vector<faiss::idx_t> preIdxMap;
    for (auto device : DEIVCE) {
        size_t size = index->getBaseSize(device);
        std::vector<int8_t> base(size * DIM);
        index->getBase(device, base);
        preXb.insert(preXb.end(), base.begin(), base.end());

        std::vector<faiss::idx_t> idMapPre(size);
        index->getIdxMap(device, idMapPre);
        preIdxMap.insert(preIdxMap.end(), idMapPre.begin(), idMapPre.end());
    }

    faiss::IDSelectorBatch del(delBatchs.size(), delBatchs.data());
    int rmNum = 0;
    for (int i = 0; i < ntotal; i++) {
        rmNum += del.is_member(ids[i]) ? 1 : 0;
    }

    size_t rmedCnt = index->remove_ids(del);
    ASSERT_EQ(rmedCnt, rmNum);
    ASSERT_EQ(index->getNTotal(), (ntotal - rmedCnt));

    int total = 0;
    for (auto device : DEIVCE) {
        total += index->getBaseSize(device);
    }
    EXPECT_EQ(total, (ntotal - rmedCnt));

    checkRemoveBatch(preXb, del, preIdxMap, rmedCnt, index);
    delete index;
}

void TestSearchWithMasksInvliadNK(faiss::ascend::AscendIndexInt8Flat &index,
                                  const std::vector<int8_t> &data,
                                  std::vector<float> &dist,
                                  std::vector<faiss::idx_t> &label,
                                  std::vector<uint8_t> &mask)
{
    int64_t batch = 128;
    int64_t k = 10;
    std::string msg = "";
    try {
        int64_t n = 0;
        index.search_with_masks(n, data.data(), k, dist.data(), label.data(), static_cast<void*>(mask.data()));
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("n must be > 0 and < ") != std::string::npos);

    try {
        int64_t n = MAX_N;
        index.search_with_masks(n, data.data(), k, dist.data(), label.data(), static_cast<void*>(mask.data()));
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("n must be > 0 and < ") != std::string::npos);

    try {
        int64_t topk = 0;
        index.search_with_masks(batch, data.data(), topk, dist.data(), label.data(), static_cast<void*>(mask.data()));
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("k must be > 0 and <= ") != std::string::npos);

    try {
        int64_t topk = MAX_K + 1;
        index.search_with_masks(batch, data.data(), topk, dist.data(), label.data(), static_cast<void*>(mask.data()));
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("k must be > 0 and <= ") != std::string::npos);
}

void TestSearchWithMasksNullptr(faiss::ascend::AscendIndexInt8Flat &index,
                                const std::vector<int8_t> &data,
                                std::vector<float> &dist,
                                std::vector<faiss::idx_t> &label,
                                std::vector<uint8_t> &mask)
{
    int64_t batch = 128;
    int64_t k = 10;
    std::string msg = "";
    try {
        const int8_t *query = nullptr;
        index.search_with_masks(batch, query, k, dist.data(), label.data(), static_cast<void*>(mask.data()));
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("x can not be nullptr.") != std::string::npos);

    try {
        index.search_with_masks(batch, data.data(), k, nullptr, label.data(), static_cast<void*>(mask.data()));
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("distances can not be nullptr.") != std::string::npos);

    try {
        index.search_with_masks(batch, data.data(), k, dist.data(), nullptr, static_cast<void*>(mask.data()));
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("labels can not be nullptr.") != std::string::npos);

    try {
        index.search_with_masks(batch, data.data(), k, dist.data(), label.data(), nullptr);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("mask can not be nullptr.") != std::string::npos);
}

TEST_P(TestAscendIndexInt8FlatUT, searchWithMasks)
{
    CheckItem item = GetParam();
    uint32_t ntotal = item.ntotal;
    faiss::MetricType metricType = item.metricType;

    int dim = 64;

    std::vector<int8_t> data(dim * ntotal);
    std::vector<faiss::idx_t> ids(ntotal, 0);
    std::iota(ids.begin(), ids.end(), 0);

    faiss::ascend::AscendIndexInt8FlatConfig conf({ 0 });
    faiss::ascend::AscendIndexInt8Flat index(dim, metricType, conf);

    index.add_with_ids(ntotal, data.data(), ids.data());
    EXPECT_EQ(index.getNTotal(), ntotal);

    int64_t batch = 128;
    int64_t k = 10;
    size_t maskLen = (ntotal + 8 - 1) / 8; // align 8
    std::vector<uint8_t> mask(static_cast<size_t>(batch) * maskLen);
    std::vector<float> dist(k * batch, 0);
    std::vector<faiss::idx_t> label(k * batch, 0);

    TestSearchWithMasksInvliadNK(index, data, dist, label, mask);
    TestSearchWithMasksNullptr(index, data, dist, label, mask);

    std::string msg = "";
    try {
        index.search_with_masks(batch, data.data(), k, dist.data(), label.data(), static_cast<void*>(mask.data()));
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.empty());
}

INSTANTIATE_TEST_CASE_P(Int8FlatCheckGroup, TestInt8FlatInitUT, ::testing::ValuesIn(INITITEMS));
INSTANTIATE_TEST_CASE_P(Int8FlatCheckGroup, TestInt8FlatCopyUT, ::testing::ValuesIn(COPYITEMS));
INSTANTIATE_TEST_CASE_P(Int8FlatCheckGroup, TestInt8FlatAddUT, ::testing::ValuesIn(ADDITEMS));
INSTANTIATE_TEST_CASE_P(Int8FlatCheckGroup, TestAscendIndexInt8FlatUT, ::testing::ValuesIn(ITEMS));
INSTANTIATE_TEST_CASE_P(Int8FlatCheckGroup, TestAscendIndexInt8FlatUT910B, ::testing::ValuesIn(ITEMS910B));
}