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


#include <numeric>
#include <random>
#include <algorithm>
#include <securec.h>
#include <faiss/index_io.h>
#include "gtest/gtest.h"
#include "AscendIndexSQ.h"
#include "AscendCloner.h"
#include "Common.h"
namespace ascend {
const auto TEST_METRIC_L2 = faiss::METRIC_L2;
const int32_t K_MAX_CAMERA_NUM = 128;
const int MASK_LEN = 8;

struct IDFilter {
    IDFilter()
    {
        memset_s(cameraIdMask, sizeof(cameraIdMask) / sizeof(cameraIdMask[0]),
            static_cast<uint8_t>(0), K_MAX_CAMERA_NUM / MASK_LEN);
        timeRange[0] = 0;
        timeRange[1] = -1;
    }

    // 一个IDFilter对象是可以涵盖处理所有cid in [0, 127] 共128个camera
    uint8_t cameraIdMask[K_MAX_CAMERA_NUM / MASK_LEN] = {0};
    uint32_t timeRange[2] = {0};
};

void GenL2Data(float *data, int num, int dim)
{
#pragma omp parallel for if (num > 1)
    for (int i = 0; i < num; ++i) {
        float l2norm = 0;
        for (int j = 0; j < dim; ++j) {
            l2norm += data[i * dim + j] * data[i * dim + j];
        }
        l2norm = sqrt(l2norm);

        for (int j = 0; j < dim; ++j) {
            data[i * dim + j] = data[i * dim + j] / l2norm;
        }
    }
}

inline void AssertEqual(std::vector<uint8_t> &lData, std::vector<uint8_t> &rData)
{
    ASSERT_EQ(lData.size(), rData.size());
    for (size_t i = 0; i < lData.size(); i++) {
        ASSERT_EQ(lData[i], rData[i]) << "i: " << i << " lData: " << lData[i] << " rData: " << rData[i] << std::endl;
    }
}

void CheckSQCodes(faiss::ascend::AscendIndexSQConfig &config, faiss::ascend::AscendIndexSQ &index,
    std::vector<float> &data, int dim, int ntotal)
{
    int deviceCnt = config.deviceList.size();
    std::vector<uint8_t> codes;
    int totals = 0;
    for (int i = 0; i < deviceCnt; i++) {
        int deviceTotal = index.getBaseSize(config.deviceList[i]);
        std::vector<uint8_t> base(deviceTotal * dim);
        index.getBase(config.deviceList[i], reinterpret_cast<char*>(base.data()));
        codes.insert(codes.end(), base.begin(), base.end());
        totals += deviceTotal;
    }
    EXPECT_EQ(totals, ntotal);

    index.reset();
    for (auto deviceId : config.deviceList) {
        int len = index.getBaseSize(deviceId);
        ASSERT_EQ(len, 0);
    }

    index.train(ntotal, data.data());
    index.add(ntotal, data.data());
    EXPECT_EQ(index.ntotal, ntotal);
    totals = 0;
    std::vector<uint8_t> baseData;
    for (int i = 0; i < deviceCnt; i++) {
        int deviceTotal = index.getBaseSize(config.deviceList[i]);
        std::vector<uint8_t> tmpBase(deviceTotal * dim);
        index.getBase(config.deviceList[i], reinterpret_cast<char*>(tmpBase.data()));
        baseData.insert(baseData.end(), tmpBase.begin(), tmpBase.end());
        totals += deviceTotal;
    }
    EXPECT_EQ(totals, ntotal);
    AssertEqual(codes, baseData);
}

TEST(TestAscendIndexUTSQ, All)
{
    int ntotal = 40000;
    const int sqDim = 64;
    const uint32_t blockSize = 8 * 16384;
    const int64_t defaultMem = static_cast<int64_t>(128 * 1024 * 1024);
    const std::initializer_list<int> devices = { 0 };

    std::vector<float> data(sqDim * ntotal);
    ascend::FeatureGenerator(data);
    GenL2Data(data.data(), ntotal, sqDim);

    faiss::ascend::AscendIndexSQConfig sqConfig {devices, defaultMem, blockSize};
    faiss::ascend::AscendIndexSQ index(sqDim, faiss::ScalarQuantizer::QuantizerType::QT_8bit, TEST_METRIC_L2, sqConfig);

    for (auto deviceId : sqConfig.deviceList) {
        int len = index.getBaseSize(deviceId);
        ASSERT_EQ(len, 0);
    }
    index.train(ntotal, data.data());
    index.add(ntotal, data.data());
    EXPECT_EQ(index.ntotal, ntotal);

    CheckSQCodes(sqConfig, index, data, sqDim, ntotal);

    int batch = 4;
    for (int i = ntotal - 40; i < ntotal; i += batch) {
        int k = 10;
        std::vector<float> dist(k * batch, 0);
        std::vector<faiss::idx_t> label(k * batch, 0);
        index.search(batch, data.data() + i * sqDim, k, dist.data(), label.data());
        faiss::idx_t assign;
        index.assign(1, data.data() + i * sqDim, &assign);
    }

    const char *fileName = "sq.faiss";
    printf("Test cloneAndLoad For SQ, result save to %s\n", fileName);
    faiss::Index *cpuIndex = faiss::ascend::index_ascend_to_cpu(&index);
    write_index(cpuIndex, fileName);
    delete cpuIndex;

    index.reset();
    // load
    faiss::Index *initIndex = faiss::read_index(fileName);
    faiss::ascend::AscendIndexSQ *realIndex =
        dynamic_cast<faiss::ascend::AscendIndexSQ *>(faiss::ascend::index_cpu_to_ascend(devices, initIndex));
    delete initIndex;
    delete realIndex;
}

TEST(TestAscendIndexUTSQ, CopyTo)
{
    int ntotal = 2500;
    int dim = 64;
    std::vector<float> data(dim * ntotal);
    ascend::FeatureGenerator(data);
    GenL2Data(data.data(), ntotal, dim);

    faiss::ascend::AscendIndexSQConfig conf({ 0 });
    faiss::ascend::AscendIndexSQ index(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit, TEST_METRIC_L2, conf);

    index.train(ntotal, data.data());
    index.add(ntotal, data.data());

    faiss::IndexScalarQuantizer toCpuIndex(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit);
    faiss::IndexIDMap idIndex(&toCpuIndex);
    index.copyTo(&toCpuIndex);
    index.copyTo(&idIndex);
    EXPECT_EQ(toCpuIndex.ntotal, ntotal);
    EXPECT_EQ(toCpuIndex.d, dim);

    {
        int total = 0;
        std::vector<uint8_t> code;
        for (auto deviceId : conf.deviceList) {
            size_t size = index.getBaseSize(deviceId);
            std::vector<uint8_t> base(size * dim);
            index.getBase(deviceId, reinterpret_cast<char *>(base.data()));

            code.insert(code.end(), base.begin(), base.end());
            total += size;
        }

        EXPECT_EQ(total, ntotal);
        AssertEqual(code, toCpuIndex.codes);
    }
}

TEST(TestAscendIndexUTSQ, CopyFrom)
{
    int  ntotal = 2000;
    int dim = 64;

    std::vector<float> data(dim * ntotal);
    ascend::FeatureGenerator(data);
    faiss::IndexScalarQuantizer cpuIndex(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit);
    faiss::IndexIDMap idIndex(&cpuIndex);
    cpuIndex.train(ntotal, data.data());
    cpuIndex.add(ntotal, data.data());

    faiss::ascend::AscendIndexSQConfig conf({ 0 });
    faiss::ascend::AscendIndexSQ index(&idIndex, conf);

    index.copyFrom(&idIndex);
    index.copyFrom(&cpuIndex);
    EXPECT_EQ(index.d, dim);
    EXPECT_EQ(index.ntotal, ntotal);

    {
        int sizeAscend = 0;
        for (auto deviceId : conf.deviceList) {
            size_t size = index.getBaseSize(deviceId);
            std::vector<float> base(size * dim);
            index.getBase(deviceId, reinterpret_cast<char *>(base.data()));
            sizeAscend += size;
        }
        ASSERT_EQ(ntotal, sizeAscend);
    }
}

static void TestSearchWithMasks(int dim)
{
    std::vector<int> searchNum = { 1 };
    faiss::ascend::AscendIndexSQConfig conf({ 0 });
    int topk = 10;
    faiss::ascend::AscendIndexSQ index(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::METRIC_L2, conf);
    index.verbose = true;
    int ntotal = 8;
    std::vector<float> base(ntotal * dim);
    ascend::FeatureGenerator(base);
    GenL2Data(base.data(), ntotal, dim);
    std::vector<float> queryData(base.begin(), base.begin() + searchNum.back() * dim);

    index.train(ntotal, base.data());
    index.add(ntotal, base.data());
    int maksSize = (ntotal + 7) / 8;
    for (size_t n = 0; n < searchNum.size(); n++) {
        std::vector<uint8_t> mask(maksSize * searchNum[n], 1);
        
        std::vector<float> dist(searchNum[n] * topk, 0);
        std::vector<faiss::idx_t> label(searchNum[n] * topk, 0);
        index.search_with_masks(searchNum[n], queryData.data(), topk, dist.data(), label.data(), mask.data());
    }
}

TEST(TestAscendIndexUTSQ, search_with_masks)
{
    TestSearchWithMasks(64);
    TestSearchWithMasks(768);
}

TEST(TestAscendIndexUTSQ, search_with_filter)
{
    int dim = 64;
    int searchNum = 1;
    faiss::ascend::AscendIndexSQConfig conf({ 0 });
    conf.filterable = true;
    faiss::ascend::AscendIndexSQ index(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::METRIC_L2, conf);
 
    int ntotal = 128;
    std::vector<float> data(dim * ntotal);
    ascend::FeatureGenerator(data);
    GenL2Data(data.data(), ntotal, dim);
 
    std::vector<int64_t> ids(ntotal, 0);
    int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine e1(seed);
    std::uniform_int_distribution<int32_t> id(0, std::numeric_limits<int32_t>::max());
    std::uniform_int_distribution<uint8_t> search_cid(0, std::numeric_limits<uint8_t>::max());
 
    for (int i = 0; i < ntotal; i++) {
        ids[i] = (static_cast<int64_t>(search_cid(e1)) << 42) + (static_cast<int64_t>(id(e1)) << 10);
    }

    IDFilter filters[searchNum];
    for (int i = 0; i < searchNum; i++) {
        // 不考虑时间
        filters[i].timeRange[0] = 0;
        filters[i].timeRange[1] = 0x7fffffff;
        for (int j = 0; j < 16; j++) {
            filters[i].cameraIdMask[j] = search_cid(e1);
        }
    }
 
    index.train(ntotal, data.data());
    index.add_with_ids(ntotal, data.data(), ids.data());
 
    int k = 128;
    std::vector<float> dist(k * searchNum, 0);
    std::vector<faiss::idx_t> label(k * searchNum, 0);
 
    index.search_with_filter(searchNum, data.data(), k, dist.data(), label.data(), &filters);
}

} // namespace ascend
