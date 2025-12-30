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
#include "common/utils/SocUtils.h"
#include "ut/Common.h"
#include "faiss/ascend/AscendCloner.h"
#include "faiss/IndexFlat.h"
#include "ascenddaemon/impl/Index.h"
#include "faiss/ascend/AscendIndexFlat.h"
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

class TestFlatInitUT : public TestWithParam<CheckInitItem> {
};

class TestFlatCopyUT : public TestWithParam<CheckCopyItem> {
};
class TestFlatAddUT : public TestWithParam<AddCheckItem> {};
class TestAscendIndexFlatUT : public TestWithParam<CheckItem> {
};

class TestAscendIndexFlatUT910B : public TestWithParam<CheckCopyItem> {
};

const int DIM = 64;
std::vector<int> DEIVCE = { 0 };
const float EP = 1e-3;
const int RANGEMIN = 0;
const int RANGEMAX = 4;
const CheckInitItem INITITEMS[] = {
    { 16, { 0 }, faiss::METRIC_INNER_PRODUCT, "Unsupported dims" },
    { 512, { 0 }, faiss::METRIC_L1, "Unsupported metric type" },
    { 16, { 0 }, faiss::METRIC_L2, "Unsupported dims" }
};

const CheckCopyItem COPYITEMS[] = {
    { 16, 1000, faiss::METRIC_INNER_PRODUCT, "Unsupported dims" },
    { 512, 1000, faiss::METRIC_L1, "Unsupported metric type" },
    { 16, 1000, faiss::METRIC_L2, "Unsupported dims" },
};

const CheckItem ITEMS[] = {
    { 1000, faiss::METRIC_INNER_PRODUCT },
    { 1000, faiss::METRIC_L2 }
};

const AddCheckItem ADDITEMS[] = {
    { 1000, faiss::METRIC_INNER_PRODUCT },
    { 1000, faiss::METRIC_L2 },
    { 524289, faiss::METRIC_INNER_PRODUCT }
};

const CheckCopyItem ITEMS910B[] = {
    { 64, 1000, faiss::METRIC_INNER_PRODUCT, "Ascend910B1" },
    { 128, 30000, faiss::METRIC_INNER_PRODUCT, "Ascend910B2" },
    { 256, 1000, faiss::METRIC_L2, "Ascend910B3" },
    { 384, 30000, faiss::METRIC_L2, "Ascend910B4" }
};

inline void AssertEqual(const std::vector<uint8_t> &gt, const std::vector<uint8_t> &data)
{
    const float ep = 1e-3;
    ASSERT_EQ(gt.size(), data.size());
    for (size_t i = 0; i < gt.size(); i++) {
        ASSERT_TRUE(fabs(gt[i] - data[i]) <= ep) << i << gt[i] << " data: " << data[i] << std::endl;
    }
}

TEST_P(TestFlatInitUT, FlatInitInvalidInput)
{
    std::string msg;
    CheckInitItem item = GetParam();
    std::vector<int> deviceList = item.deviceList;
    faiss::MetricType metricType = item.metricType;
    size_t dim = item.dim;
    std::string str = item.str;

    try {
        faiss::ascend::AscendIndexFlatConfig config(deviceList);
        faiss::ascend::AscendIndexFlat index(dim, metricType, config);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find(str) != std::string::npos);
}

TEST_P(TestFlatCopyUT, FlatCopyInvalidInput)
{
    std::string msg;
    CheckCopyItem item = GetParam();
    int64_t ntotal = item.ntotal;
    faiss::MetricType metricType = item.metricType;
    size_t dim = item.dim;
    std::string str = item.str;

    std::vector<float> data(dim * ntotal);
    faiss::IndexFlat cpuIndex(dim, metricType);
    faiss::IndexIDMap idIndex(&cpuIndex);
    cpuIndex.add(ntotal, data.data());

    try {
        faiss::ascend::AscendIndexFlatConfig config({ 0 });
        faiss::ascend::AscendIndexFlat index(&idIndex, config);
        index.copyFrom(&cpuIndex);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find(str) != std::string::npos);
}

template<typename T>
void TestAdd(const std::vector<T>& data, uint32_t ntotal, faiss::MetricType metricType, int dim)
{
    std::initializer_list<int> devices = { 0 };
    faiss::ascend::AscendIndexFlatConfig config(devices);
    faiss::ascend::AscendIndexFlat index(dim, metricType, config);

    for (auto id : config.deviceList) {
        int len = index.getBaseSize(id);
        ASSERT_EQ(len, 0);
    }

    index.add(ntotal, data.data());
    EXPECT_EQ(index.ntotal, ntotal);

    int deviceCnt = config.deviceList.size();
    int totals = 0;
    for (int i = 0; i < deviceCnt; i++) {
        int tmpTotal = index.getBaseSize(config.deviceList[i]);
        totals += tmpTotal;
    }
    EXPECT_EQ(totals, ntotal);
}

TEST_P(TestFlatAddUT, add)
{
    AddCheckItem item = GetParam();
    uint32_t ntotal = item.ntotal;
    faiss::MetricType metricType = item.metricType;
    int dim = 64;

    // float
    std::vector<float> data(dim * ntotal, 0.5);
    TestAdd(data, ntotal, metricType, dim);

    // uint16_t
    std::vector<uint16_t> dataFp16(dim * ntotal, 1);
    TestAdd(dataFp16, ntotal, metricType, dim);
}

TEST_P(TestAscendIndexFlatUT, CopyFrom)
{
    CheckItem item = GetParam();
    uint32_t ntotal = item.ntotal;
    int dim = 64;

    std::vector<float> data(dim * ntotal);
    faiss::IndexFlat cpuIndex(dim);
    faiss::IndexIDMap idIndex(&cpuIndex);
    cpuIndex.add(ntotal, data.data());

    faiss::ascend::AscendIndexFlatConfig conf({ 0 });
    faiss::ascend::AscendIndexFlatL2 indexL2(dim, conf);
    faiss::ascend::AscendIndexFlat index(&idIndex, conf);

    index.copyFrom(&idIndex);
    index.copyFrom(&cpuIndex);
    EXPECT_EQ(index.d, dim);
    EXPECT_EQ(index.ntotal, ntotal);

    indexL2.copyFrom(&cpuIndex);
    {
        int sizeAscend = 0;
        std::vector<float> xbAsend;
        for (auto deviceId : conf.deviceList) {
            size_t size = index.getBaseSize(deviceId);
            std::vector<float> base(size * dim);
            index.getBase(deviceId, reinterpret_cast<char *>(base.data()));
            xbAsend.insert(xbAsend.end(), base.begin(), base.end());
            sizeAscend += size;
        }
        ASSERT_EQ(ntotal, sizeAscend);
    }
}

TEST_P(TestAscendIndexFlatUT, CopyTo)
{
    CheckItem item = GetParam();
    uint32_t ntotal = item.ntotal;
    int dim = 64;

    std::vector<float> data(dim * ntotal);
    faiss::IndexFlatL2 cpuIndexL2(dim);
    faiss::IndexFlat cpuIndex(dim);
    faiss::IndexFlat cpuIndexCopy(dim);
    faiss::IndexIDMap idIndex(&cpuIndex);
    faiss::ascend::AscendIndexFlatConfig conf({ 0 });
    faiss::ascend::AscendIndexFlat index(&cpuIndex, conf);
    faiss::ascend::AscendIndexFlatL2 indexL2(&cpuIndexL2, conf);

    index.add(ntotal, data.data());
    indexL2.add(ntotal, data.data());

    index.copyTo(&cpuIndex);
    index.copyTo(&idIndex);
    indexL2.copyTo(&cpuIndexCopy);
    EXPECT_EQ(cpuIndex.d, dim);
    EXPECT_EQ(cpuIndex.ntotal, ntotal);

    faiss::IndexFlat flatCpu;
    faiss::IndexIDMap idMap(&flatCpu);
    index.copyTo(&idMap);
    EXPECT_EQ(index.d, idMap.d);
    EXPECT_EQ(index.ntotal, idMap.ntotal);

    int tmp = 0;
    std::vector<uint8_t> res;
    for (auto device : conf.deviceList) {
        size_t size = index.getBaseSize(device);
        std::vector<uint8_t> base(size * dim * sizeof(float));
        index.getBase(device, reinterpret_cast<char *>(base.data()));

        res.insert(res.end(), base.begin(), base.end());
        tmp += size;
    }

    EXPECT_EQ(tmp, ntotal);
    AssertEqual(res, cpuIndex.codes);

    int tmpTotal = 0;
    std::vector<uint8_t> codes;
    for (auto deviceId : conf.deviceList) {
        size_t size = indexL2.getBaseSize(deviceId);
        std::vector<uint8_t> base(size * dim * sizeof(float));
        indexL2.getBase(deviceId, reinterpret_cast<char *>(base.data()));

        codes.insert(codes.end(), base.begin(), base.end());
        tmpTotal += size;
    }

    EXPECT_EQ(tmpTotal, ntotal);
    AssertEqual(codes, cpuIndexCopy.codes);
}

template<typename T>
void TestAddWithIds(const std::vector<T>& data, uint32_t ntotal, faiss::MetricType metricType, int dim)
{
    std::vector<faiss::idx_t> ids(ntotal, 0);
    std::iota(ids.begin(), ids.end(), 0);

    faiss::ascend::AscendIndexFlatConfig conf({ 0 });
    faiss::ascend::AscendIndexFlat index(dim, metricType, conf);

    for (auto device : conf.deviceList) {
        int total = index.getBaseSize(device);
        ASSERT_EQ(total, 0);
    }

    index.add_with_ids(ntotal, data.data(), ids.data());
    EXPECT_EQ(index.ntotal, ntotal);

    int totals = 0;
    for (size_t i = 0; i < conf.deviceList.size(); i++) {
        int tmpTotal = index.getBaseSize(conf.deviceList[i]);
        totals += tmpTotal;
    }
    EXPECT_EQ(totals, ntotal);
}

TEST_P(TestAscendIndexFlatUT, AddWithIds)
{
    CheckItem item = GetParam();
    uint32_t ntotal = item.ntotal;
    faiss::MetricType metricType = item.metricType;
    int dim = 64;

    // float
    std::vector<float> data(dim * ntotal);
    TestAddWithIds(data, ntotal, metricType, dim);

    // uint16_t
    std::vector<uint16_t> dataFp16(dim * ntotal);
    std::transform(data.begin(), data.end(), dataFp16.begin(),
        [](float temp) { return faiss::ascend::fp16(temp).data; });
    TestAddWithIds(dataFp16, ntotal, metricType, dim);
}

template<typename T>
void TestAddRemoveSearch(const std::vector<T>& data, uint32_t ntotal, faiss::MetricType metricType,
    int dim, std::string socName)
{
    // 打桩socName为910B*
    MOCKER_CPP(&aclrtGetSocName).stubs().will(returnValue(socName.c_str()));
    faiss::ascend::SocUtils::GetInstance().Init();

    faiss::ascend::AscendIndexFlatConfig conf({ 0 });
    faiss::ascend::AscendIndexFlat index(dim, metricType, conf);

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
    EXPECT_EQ(index.ntotal, ntotal);

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
    EXPECT_EQ(index.ntotal, ntotal);

    index.search(batch, data.data(), k, dist.data(), label.data());

    const char *fileName = "flat.faiss";
    // save
    printf("Test cloneAndLoad For flat, result save to %s\n", fileName);
    faiss::Index *cpuIndex = faiss::ascend::index_ascend_to_cpu(&index);
    faiss::write_index(cpuIndex, fileName);
    delete cpuIndex;

    index.reset();
    // load
    faiss::Index *initIndex = faiss::read_index(fileName);
    faiss::ascend::AscendIndexFlat *realIndex =
        dynamic_cast<faiss::ascend::AscendIndexFlat *>(faiss::ascend::index_cpu_to_ascend({ 0 }, initIndex));
    delete initIndex;
    delete realIndex;

    // mockcpp 需要显示调用该函数来恢复打桩
    GlobalMockObject::verify();
}

TEST_P(TestAscendIndexFlatUT910B, AddRemoveSearch)
{
    CheckCopyItem item = GetParam();
    uint32_t ntotal = item.ntotal;
    faiss::MetricType metricType = item.metricType;
    int dim = item.dim;
    std::string socName = item.str;

    // float
    std::vector<float> data(dim * ntotal);
    TestAddRemoveSearch(data, ntotal, metricType, dim, socName);
    
    // uint16_t
    std::vector<uint16_t> dataFp16(dim * ntotal);
    std::transform(data.begin(), data.end(), dataFp16.begin(),
        [](float temp) { return faiss::ascend::fp16(temp).data; });
    TestAddRemoveSearch(dataFp16, ntotal, metricType, dim, socName);
}

void checkRemoveRange(std::vector<float>& preXb, std::vector<faiss::idx_t>& preIdxMap, int rmedCnt,
    int ntotal, faiss::ascend::AscendIndexFlat *index)
{
    std::vector<float> xb;
    std::vector<faiss::idx_t> idxMap;
    for (auto deviceId : DEIVCE) {
        size_t size = index->getBaseSize(deviceId);
        std::vector<float> base(size * DIM);
        index->getBase(deviceId, reinterpret_cast<char *>(base.data()));
        xb.insert(xb.end(), base.begin(), base.end());

        std::vector<faiss::idx_t> idMap(size);
        index->getIdxMap(deviceId, idMap);
        idxMap.insert(idxMap.end(), idMap.begin(), idMap.end());
    }
    EXPECT_EQ(preIdxMap.size(), idxMap.size() + rmedCnt);
    EXPECT_EQ(preXb.size(), xb.size() + rmedCnt * DIM);
    {
        int num = (ntotal + DEIVCE.size() - 1) / DEIVCE.size();

        // check idx
        for (int i = 0; i < num - rmedCnt; i++) {
            if ((preIdxMap[i] >= RANGEMIN) && (preIdxMap[i] < RANGEMAX)) {
                int fptrNum = (num - rmedCnt + i) * DIM;
                int bptrNum = i * DIM;
                // check vector
                for (int j = 0; j < DIM; j++) {
                    ASSERT_TRUE(fabs(preXb[fptrNum + j] - xb[bptrNum + j]) <= EP);
                }
            } else {
                int ptr = i * DIM;
                // check idx
                EXPECT_EQ(preIdxMap[i], idxMap[i]);
                // check vector
                for (int j = 0; j < DIM; j++) {
                    ASSERT_TRUE(fabs(preXb[ptr + j] - xb[ptr + j]) <= EP);
                }
            }
        }
    }
}

TEST_P(TestAscendIndexFlatUT, removeRange)
{
    CheckItem item = GetParam();
    uint32_t ntotal = item.ntotal;
    faiss::MetricType metricType = item.metricType;

    std::vector<float> data(DIM * ntotal);

    faiss::ascend::AscendIndexFlatConfig conf({ 0 });
    faiss::ascend::AscendIndexFlat *index = new faiss::ascend::AscendIndexFlat(DIM, metricType, conf);

    index->add(ntotal, data.data());
    // define ids
    std::vector<faiss::idx_t> ids(ntotal);
    std::iota(ids.begin(), ids.end(), 0);

    // save pre-delete basedata and index of vector
    std::vector<float> preXb;
    std::vector<faiss::idx_t> preIdxMap;
    for (auto device : DEIVCE) {
        size_t baseSize = index->getBaseSize(device);
        std::vector<float> base(baseSize * DIM);
        index->getBase(device, reinterpret_cast<char *>(base.data()));
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
    ASSERT_EQ(index->ntotal, (ntotal - rmedCnt));

    int total = 0;
    for (auto device : DEIVCE) {
        total += index->getBaseSize(device);
    }
    EXPECT_EQ(total, (ntotal - rmedCnt));

    checkRemoveRange(preXb, preIdxMap, rmedCnt, ntotal, index);
    delete index;
}


void checkRemoveBatch(std::vector<float>& preXb, faiss::IDSelectorBatch del,
    std::vector<faiss::idx_t>& preIdxMap, size_t rmedCnt, faiss::ascend::AscendIndexFlat *index)
{
    std::vector<float> xb;
    std::vector<faiss::idx_t> idxMap;
    for (auto device : DEIVCE) {
        size_t size = index->getBaseSize(device);
        std::vector<float> base(size * DIM);
        index->getBase(device, reinterpret_cast<char *>(base.data()));
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

void testRemoveBatch(CheckItem &item)
{
    uint32_t ntotal = item.ntotal;
    faiss::MetricType metricType = item.metricType;
    std::vector<faiss::idx_t> delBatchs = { 1, 8 };

    std::vector<float> data(DIM * ntotal);

    faiss::ascend::AscendIndexFlatConfig conf({ 0 });
    faiss::ascend::AscendIndexFlat *index = new faiss::ascend::AscendIndexFlat(DIM, metricType, conf);

    index->add(ntotal, data.data());
    // define ids
    std::vector<faiss::idx_t> ids(ntotal);
    std::iota(ids.begin(), ids.end(), 0);

    // save pre-delete basedata and index of vector
    std::vector<float> preXb;
    std::vector<faiss::idx_t> preIdxMap;
    for (auto device : DEIVCE) {
        size_t size = index->getBaseSize(device);
        std::vector<float> base(size * DIM);
        index->getBase(device, reinterpret_cast<char *>(base.data()));
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
    ASSERT_EQ(index->ntotal, (ntotal - rmedCnt));

    int total = 0;
    for (auto device : DEIVCE) {
        total += index->getBaseSize(device);
    }
    EXPECT_EQ(total, (ntotal - rmedCnt));

    checkRemoveBatch(preXb, del, preIdxMap, rmedCnt, index);
    delete index;
}

TEST_P(TestAscendIndexFlatUT, removeBatch)
{
    CheckItem item = GetParam();
    testRemoveBatch(item);
}

template<typename T>
void TestSearchWithMasksInvliadNK(faiss::ascend::AscendIndexFlat &index,
                                  const std::vector<T> &data,
                                  std::vector<float> &dist,
                                  std::vector<faiss::idx_t> &label,
                                  std::vector<uint8_t> &mask)
{
    int64_t batch = 128;
    int64_t k = 10;
    std::string msg = "";
    try {
        int64_t n = 0;
        index.search_with_masks(n, data.data(), k, dist.data(), label.data(), mask.data());
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("n must be > 0 and < ") != std::string::npos);

    try {
        int64_t n = MAX_N;
        index.search_with_masks(n, data.data(), k, dist.data(), label.data(), mask.data());
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("n must be > 0 and < ") != std::string::npos);

    try {
        int64_t topk = 0;
        index.search_with_masks(batch, data.data(), topk, dist.data(), label.data(), mask.data());
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("k must be > 0 and <= ") != std::string::npos);

    try {
        int64_t topk = MAX_K + 1;
        index.search_with_masks(batch, data.data(), topk, dist.data(), label.data(), mask.data());
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("k must be > 0 and <= ") != std::string::npos);
}

template<typename T>
void TestSearchWithMasksNullptr(faiss::ascend::AscendIndexFlat &index,
                                const std::vector<T> &data,
                                std::vector<float> &dist,
                                std::vector<faiss::idx_t> &label,
                                std::vector<uint8_t> &mask)
{
    int64_t batch = 128;
    int64_t k = 10;
    std::string msg = "";
    try {
        const float *query = nullptr;
        index.search_with_masks(batch, query, k, dist.data(), label.data(), mask.data());
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("x cannot be nullptr") != std::string::npos);

    try {
        index.search_with_masks(batch, data.data(), k, nullptr, label.data(), mask.data());
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("distance cannot be nullptr") != std::string::npos);

    try {
        index.search_with_masks(batch, data.data(), k, dist.data(), nullptr, mask.data());
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("labels cannot be nullptr") != std::string::npos);

    try {
        index.search_with_masks(batch, data.data(), k, dist.data(), label.data(), nullptr);
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.find("mask cannot be nullptr") != std::string::npos);
}

void testSearchWithmask(CheckItem &item)
{
    uint32_t ntotal = item.ntotal;
    faiss::MetricType metricType = item.metricType;

    int dim = 64;

    std::vector<float> data(dim * ntotal);
    std::vector<faiss::idx_t> ids(ntotal, 0);
    std::iota(ids.begin(), ids.end(), 0);

    faiss::ascend::AscendIndexFlatConfig conf({ 0 });
    faiss::ascend::AscendIndexFlat index(dim, metricType, conf);

    index.add_with_ids(ntotal, data.data(), ids.data());
    EXPECT_EQ(index.ntotal, ntotal);

    int64_t batch = 128;
    int64_t k = 10;
    size_t maskLen = (ntotal + 8 - 1) / 8; // align 8
    std::vector<uint8_t> mask(static_cast<size_t>(batch) * maskLen);
    std::vector<float> dist(k * batch, 0);
    std::vector<faiss::idx_t> label(k * batch, 0);

    // float
    TestSearchWithMasksInvliadNK(index, data, dist, label, mask);
    TestSearchWithMasksNullptr(index, data, dist, label, mask);

    // uint16_t
    std::vector<uint16_t> dataFp16(dim * ntotal);
    std::transform(data.begin(), data.end(), dataFp16.begin(),
        [](float temp) { return faiss::ascend::fp16(temp).data; });
    TestSearchWithMasksInvliadNK(index, dataFp16, dist, label, mask);
    TestSearchWithMasksNullptr(index, dataFp16, dist, label, mask);

    std::string msg = "";
    try {
        index.search_with_masks(batch, data.data(), k, dist.data(), label.data(), mask.data());
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.empty());

    msg = "";
    try {
        index.search_with_masks(batch, dataFp16.data(), k, dist.data(), label.data(), mask.data());
    } catch(std::exception &e) {
        msg = e.what();
    }
    EXPECT_TRUE(msg.empty());
}

TEST_P(TestAscendIndexFlatUT, searchWithMasks)
{
    CheckItem item = GetParam();
    testSearchWithmask(item);
}

TEST_P(TestAscendIndexFlatUT, onlineOp)
{
    CheckItem item = GetParam();
    uint32_t ntotal = item.ntotal;
    faiss::MetricType metricType = item.metricType;
    int dim = 64;
    size_t batch = 1;
    size_t k = 10;
    std::vector<float> dist(k * batch, 0);
    std::vector<faiss::idx_t> label(k * batch, 0);
    std::vector<float> data(dim * ntotal);
    MOCKER_CPP(&ascend::Index::isUseOnlineOp).stubs().will(returnValue(true));
    faiss::ascend::AscendIndexFlatConfig config({ 0 });
    faiss::ascend::AscendIndexFlat index1(dim, metricType, config);
    index1.add(ntotal, data.data());
    index1.search(batch, data.data(), k, dist.data(), label.data());
    GlobalMockObject::verify();
}

INSTANTIATE_TEST_CASE_P(FlatCheckGroup, TestFlatInitUT, ::testing::ValuesIn(INITITEMS));
INSTANTIATE_TEST_CASE_P(FlatCheckGroup, TestFlatCopyUT, ::testing::ValuesIn(COPYITEMS));
INSTANTIATE_TEST_CASE_P(FlatCheckGroup, TestFlatAddUT, ::testing::ValuesIn(ADDITEMS));
INSTANTIATE_TEST_CASE_P(FlatCheckGroup, TestAscendIndexFlatUT, ::testing::ValuesIn(ITEMS));
INSTANTIATE_TEST_CASE_P(FlatCheckGroup, TestAscendIndexFlatUT910B, ::testing::ValuesIn(ITEMS910B));
}