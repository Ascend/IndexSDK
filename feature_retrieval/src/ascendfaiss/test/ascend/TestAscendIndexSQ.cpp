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
#include <gtest/gtest.h>
#include <faiss/ascend/AscendIndexSQ.h>
#include <faiss/ascend/AscendCloner.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/ascend/AscendMultiIndexSearch.h>
#include <faiss/index_io.h>
#include <algorithm>

#include <sys/time.h>

namespace {
const int DIM = 64;
const std::vector<int> DEVICES = { 0 };
const auto METRIC_TYPE = faiss::METRIC_INNER_PRODUCT;

unsigned int g_seed;
const int FAST_RAND_MAX = 0x7FFF;

inline void FastSrand(int seed)
{
    g_seed = seed;
}

// Compute a pseudorandom integer.
// Output value in range [0, 32767]
inline int FastRand(void)
{
    const int mutipliyNum = 214013;
    const int addNum = 2531011;
    const int rshiftNum = 16;
    g_seed = (mutipliyNum * g_seed + addNum);
    return (g_seed >> rshiftNum) & FAST_RAND_MAX;
}

void Norm(float *data, int n, int dim)
{
#pragma omp parallel for if (n > 1)
    for (int i = 0; i < n; ++i) {
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

inline double GetMillisecs()
{
    struct timeval tv {
        0, 0
    };
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

inline void AssertEqual(std::vector<uint8_t> &gt, std::vector<uint8_t> &data)
{
    ASSERT_EQ(gt.size(), data.size());
    for (size_t i = 0; i < gt.size(); i++) {
        ASSERT_EQ(gt[i], data[i]) << "i: " << i << " gt: " << gt[i] << " data: " << data[i] << std::endl;
    }
}

struct IDFilter {
    IDFilter()
    {
        memset(camera_id_mask, static_cast<uint8_t>(0xFF), 128 / 8);
        time_range[0] = 0;
        time_range[1] = -1;
    }
 
    uint8_t camera_id_mask[128 / 8] = {0xFF};
    uint32_t time_range[2] = {0};
};

TEST(TestAscendIndexSQ, All)
{
    int ntotal = 1000000;

    std::vector<float> data(DIM * ntotal);
    for (int i = 0; i < DIM * ntotal; i++) {
        data[i] = drand48();
    }
    Norm(data.data(), ntotal, DIM);

    faiss::ascend::AscendIndexSQConfig conf(DEVICES);
    faiss::ascend::AscendIndexSQ index(DIM, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE, conf);

    for (auto deviceId : conf.deviceList) {
        int len = index.getBaseSize(deviceId);
        ASSERT_EQ(len, 0);
    }

    index.train(ntotal, data.data());
    index.add(ntotal, data.data());
    EXPECT_EQ(index.ntotal, ntotal);

    int deviceCnt = conf.deviceList.size();
    std::vector<uint8_t> codes;
    {
        int totals = 0;
        for (int i = 0; i < deviceCnt; i++) {
            int tmpTotal = index.getBaseSize(conf.deviceList[i]);
            std::vector<uint8_t> base(tmpTotal * DIM);
            index.getBase(conf.deviceList[i], (char *)base.data());
            codes.insert(codes.end(), base.begin(), base.end());
            totals += tmpTotal;
        }
        EXPECT_EQ(totals, ntotal);
    }

    index.reset();

    for (auto deviceId : conf.deviceList) {
        int len = index.getBaseSize(deviceId);
        ASSERT_EQ(len, 0);
    }

    index.train(ntotal, data.data());
    index.add(ntotal, data.data());
    {
        int getTotal = 0;
        std::vector<uint8_t> baseData;
        for (int i = 0; i < deviceCnt; i++) {
            int tmpTotal = index.getBaseSize(conf.deviceList[i]);
            std::vector<uint8_t> tmpBase(tmpTotal * DIM);
            index.getBase(conf.deviceList[i], (char *)tmpBase.data());
            baseData.insert(baseData.end(), tmpBase.begin(), tmpBase.end());
            getTotal += tmpTotal;
        }
        EXPECT_EQ(getTotal, ntotal);
        AssertEqual(codes, baseData);
    }

    {
        int batch = 4;
        for (int i = ntotal - 40; i < ntotal; i += batch) {
            int k = 1000;
            std::vector<float> dist(k * batch, 0);
            std::vector<faiss::idx_t> label(k * batch, 0);
            index.search(batch, data.data() + i * DIM, k, dist.data(), label.data());
            ASSERT_EQ(label[0], i);
            ASSERT_EQ(label[k], i + 1);
            ASSERT_EQ(label[k * 2], i + 2);
            ASSERT_EQ(label[k * 3], i + 3);
            faiss::idx_t assign;
            index.assign(1, data.data() + i * DIM, &assign);
            ASSERT_EQ(assign, i);
        }
    }
}

TEST(TestAscendIndexSQ, AddWithIds)
{
    int ntotal = 200000;

    std::vector<float> data(DIM * ntotal);
    for (int i = 0; i < DIM * ntotal; i++) {
        data[i] = drand48();
    }
    Norm(data.data(), ntotal, DIM);

    std::vector<int64_t> ids(ntotal);
    std::iota(ids.begin(), ids.end(), 0);

    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(ids), std::end(ids), rng);

    faiss::ascend::AscendIndexSQConfig conf(DEVICES);
    faiss::ascend::AscendIndexSQ index(DIM, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE, conf);

    for (auto deviceId : conf.deviceList) {
        int len = index.getBaseSize(deviceId);
        ASSERT_EQ(len, 0);
    }

    index.train(ntotal, data.data());
    index.add_with_ids(ntotal, data.data(), ids.data());
    EXPECT_EQ(index.ntotal, ntotal);

    int deviceCnt = conf.deviceList.size();
    {
        int totals = 0;
        for (int i = 0; i < deviceCnt; i++) {
            int tmpTotal = index.getBaseSize(conf.deviceList[i]);
            totals += tmpTotal;
        }
        EXPECT_EQ(totals, ntotal);
    }

    {
        for (int i = 135; i < 200; i++) {
            int k = 1000;
            std::vector<float> dist(k, 0);
            std::vector<faiss::idx_t> label(k, 0);
            index.search(1, data.data() + i * DIM, k, dist.data(), label.data());
            ASSERT_EQ(label[0], ids[i]);
            faiss::idx_t assign;
            index.assign(1, data.data() + i * DIM, &assign);
            ASSERT_EQ(assign, ids[i]);
        }
    }
}

TEST(TestAscendIndexSQ, CopyFrom)
{
    int ntotal = 250000;

    std::vector<float> data(DIM * ntotal);
    for (int i = 0; i < DIM * ntotal; i++) {
        data[i] = drand48();
    }
    Norm(data.data(), ntotal, DIM);

    faiss::IndexScalarQuantizer cpuIndex(DIM, faiss::ScalarQuantizer::QuantizerType::QT_8bit);
    cpuIndex.train(ntotal, data.data());
    cpuIndex.add(ntotal, data.data());

    faiss::ascend::AscendIndexSQConfig conf(DEVICES);
    faiss::ascend::AscendIndexSQ index(&cpuIndex, conf);

    EXPECT_EQ(index.d, DIM);
    EXPECT_EQ(index.ntotal, ntotal);

    // only make sure the format of basedata is same
    faiss::IndexScalarQuantizer cpuIndexRef(DIM, faiss::ScalarQuantizer::QuantizerType::QT_8bit);
    index.copyTo(&cpuIndexRef);

    {
        int sizeAscend = 0;
        std::vector<uint8_t> codesAsend;
        for (auto deviceId : conf.deviceList) {
            size_t size = index.getBaseSize(deviceId);
            std::vector<uint8_t> base(size * DIM);
            index.getBase(deviceId, (char *)base.data());
            codesAsend.insert(codesAsend.end(), base.begin(), base.end());
            sizeAscend += size;
        }
        ASSERT_EQ(ntotal, sizeAscend);
        AssertEqual(codesAsend, cpuIndexRef.codes);
    }
}

TEST(TestAscendIndexSQ, CopyTo)
{
    int ntotal = 250000;

    std::vector<float> data(DIM * ntotal);
    for (int i = 0; i < DIM * ntotal; i++) {
        data[i] = drand48();
    }
    Norm(data.data(), ntotal, DIM);

    faiss::ascend::AscendIndexSQConfig conf(DEVICES);
    faiss::ascend::AscendIndexSQ index(DIM, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE, conf);

    index.train(ntotal, data.data());
    index.add(ntotal, data.data());

    faiss::IndexScalarQuantizer cpuIndex(DIM, faiss::ScalarQuantizer::QuantizerType::QT_8bit);
    index.copyTo(&cpuIndex);

    EXPECT_EQ(cpuIndex.ntotal, ntotal);
    EXPECT_EQ(cpuIndex.d, DIM);

    {
        int tmpTotal = 0;
        std::vector<uint8_t> codes;
        for (auto deviceId : conf.deviceList) {
            size_t size = index.getBaseSize(deviceId);
            std::vector<uint8_t> base(size * DIM);
            index.getBase(deviceId, (char *)base.data());

            codes.insert(codes.end(), base.begin(), base.end());
            tmpTotal += size;
        }

        EXPECT_EQ(tmpTotal, ntotal);
        AssertEqual(codes, cpuIndex.codes);
    }
}

TEST(TestAscendIndexSQ, CloneAscend2CPU)
{
    int n = 250000;
    int ntotal = n * 10;

    srand48(1000);
    std::vector<float> data(DIM * n);
    for (int i = 0; i < DIM * n; i++) {
        data[i] = drand48();
    }
    Norm(data.data(), n, DIM);

    for (int i = 0; i < 20; i++) {
        printf("%.4f\t", data[i]);
    }
    // ascend index
    faiss::ascend::AscendIndexSQConfig conf(DEVICES);
    faiss::ascend::AscendIndexSQ ascendIndex(DIM, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE,
        conf);

    ascendIndex.train(n, data.data());
    // add ground truth
    ascendIndex.add(n, data.data());

    // add 2250w vector
    for (int i = 0; i < (ntotal / n - 1); i++) {
        std::vector<float> dataTmp(DIM * n);
        for (int j = 0; j < DIM * n; j++) {
            dataTmp[j] = drand48();
        }
        Norm(dataTmp.data(), n, DIM);

        ascendIndex.add(n, dataTmp.data());
        printf("add %d times of data.\n", i);
    }

    // write index with cpu index
    faiss::Index *cpuIndex = faiss::ascend::index_ascend_to_cpu(&ascendIndex);
    ASSERT_FALSE(cpuIndex == nullptr);
    const char *outfilename = "./SQIndex_2500000.faiss";
    write_index(cpuIndex, outfilename);
    printf("write ascendIndex to file ok!\n");

    int lenall = 0;
    for (auto deviceId : conf.deviceList) {
        lenall += ascendIndex.getBaseSize(deviceId);
    }

    EXPECT_EQ(lenall, ntotal);

    for (int i = 0; i < 10; i++) {
        int k = 1000;
        int idx = i * 5;
        std::vector<float> dist(k, 0);
        std::vector<faiss::idx_t> label(k, 0);
        double ts = GetMillisecs();
        for (int j = 0; j < 500; j++) {
            ascendIndex.search(1, data.data() + idx * DIM, k, dist.data(), label.data());
        }
        double te = GetMillisecs();
        printf("all %f, means %f.\n", te - ts, (te - ts) / 500);
        EXPECT_EQ(label[0], idx);
    }
    delete cpuIndex;
}

TEST(TestAscendIndexSQ, CloneCPU2Ascend)
{
    int ntotal = 2500000;
    srand48(1000);
    std::vector<float> data(DIM * ntotal);
    for (int i = 0; i < DIM * ntotal; i++) {
        data[i] = drand48();
    }
    Norm(data.data(), ntotal, DIM);

    for (int i = 0; i < 20; i++) {
        printf("%.4f\t", data[i]);
    }
    const char *indexfilename = "./SQIndex_2500000.faiss";
    faiss::Index *initIndex = faiss::read_index(indexfilename);
    ASSERT_FALSE(initIndex == nullptr);

    // ascend index
    std::vector<int> devices = { 0 };
    faiss::ascend::AscendIndexSQ *ascendIndex =
        dynamic_cast<faiss::ascend::AscendIndexSQ *>(faiss::ascend::index_cpu_to_ascend(devices, initIndex));
    ASSERT_FALSE(ascendIndex == nullptr);

    int lenall = 0;
    for (auto deviceId : devices) {
        lenall += ascendIndex->getBaseSize(deviceId);
    }

    EXPECT_EQ(lenall, ntotal);

    for (int i = 0; i < 1; i++) {
        int k = 1000;
        int idx = i * 5;
        std::vector<float> dist(k, 0);
        std::vector<faiss::idx_t> label(k, 0);
        double ts = GetMillisecs();
        for (int i = 0; i < 500; i++) {
            ascendIndex->search(1, data.data() + idx * DIM, k, dist.data(), label.data());
        }
        double te = GetMillisecs();
        printf("all %f, means %f.\n", te - ts, (te - ts) / 500);
        printf("ascend idx %d dist %f %f %f %f %f lable %lu %lu %lu %lu %lu.\n", idx, dist[0], dist[1], dist[2],
            dist[3], dist[4], label[0], label[1], label[2], label[3], label[4]);
        EXPECT_EQ(label[0], idx);
    }

    delete ascendIndex;
    delete initIndex;
}

TEST(TestAscendIndexSQ, removeRange)
{
    int ntotal = 200000;
    int delRangeMin = 0;
    int delRangeMax = 4;

    std::vector<float> data(DIM * ntotal);
    for (int i = 0; i < DIM * ntotal; i++) {
        data[i] = drand48();
    }
    Norm(data.data(), ntotal, DIM);

    faiss::ascend::AscendIndexSQConfig conf(DEVICES);
    faiss::ascend::AscendIndexSQ index(DIM, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE, conf);

    index.train(ntotal, data.data());
    index.add(ntotal, data.data());
    // define ids
    std::vector<int64_t> ids(ntotal);
    std::iota(ids.begin(), ids.end(), 0);

    // save pre-delete basedata and index of vector
    std::vector<uint8_t> codePre;
    std::vector<faiss::idx_t> idxMapPre;
    for (auto deviceId : conf.deviceList) {
        size_t size = index.getBaseSize(deviceId);
        std::vector<uint8_t> base(size * DIM);
        index.getBase(deviceId, (char *)base.data());
        codePre.insert(codePre.end(), base.begin(), base.end());

        std::vector<faiss::idx_t> idMap(size);
        index.getIdxMap(deviceId, idMap);
        idxMapPre.insert(idxMapPre.end(), idMap.begin(), idMap.end());
    }

    faiss::IDSelectorRange del(delRangeMin, delRangeMax);
    int rmCnt = 0;
    for (int i = 0; i < ntotal; i++) {
        rmCnt += del.is_member(ids[i]) ? 1 : 0;
    }

    size_t rmedCnt = index.remove_ids(del);
    ASSERT_EQ(rmedCnt, rmCnt);
    ASSERT_EQ(index.ntotal, (ntotal - rmedCnt));

    int tmpTotal = 0;
    for (auto deviceId : conf.deviceList) {
        tmpTotal += index.getBaseSize(deviceId);
    }
    EXPECT_EQ(tmpTotal, (ntotal - rmedCnt));

    std::vector<uint8_t> codes;
    std::vector<faiss::idx_t> idxMap;
    for (auto deviceId : conf.deviceList) {
        size_t size = index.getBaseSize(deviceId);
        std::vector<uint8_t> base(size * DIM);
        index.getBase(deviceId, (char *)base.data());
        codes.insert(codes.end(), base.begin(), base.end());

        std::vector<faiss::idx_t> idMap(size);
        index.getIdxMap(deviceId, idMap);
        idxMap.insert(idxMap.end(), idMap.begin(), idMap.end());
    }
    EXPECT_EQ(idxMapPre.size(), idxMap.size() + rmedCnt);
    EXPECT_EQ(codePre.size(), codes.size() + rmedCnt * DIM);
    {
        // check idx
        int offset = 0;
        for (size_t i = 0; i < idxMap.size(); i++) {
            if ((idxMapPre[i] >= delRangeMin) && (idxMapPre[i] < delRangeMax)) {
                // check idx
                EXPECT_EQ(idxMapPre[idxMap.size() + i], idxMap[i]);
                // check vector
                for (int j = 0; j < DIM; j++) {
                    EXPECT_EQ(codePre[(idxMap.size() + i) * DIM + j], codes[i * DIM + j]);
                }
                offset += 1;
            } else {
                int ptr = i * DIM;
                // check idx
                EXPECT_EQ(idxMapPre[i], idxMap[i]);
                // check vector
                for (int j = 0; j < DIM; j++) {
                    EXPECT_EQ(codePre[ptr + j], codes[ptr + j]);
                }
            }
        }
        EXPECT_EQ(offset, delRangeMax - delRangeMin);
    }
}

TEST(TestAscendIndexSQ, removeBatch)
{
    int ntotal = 200000;
    std::vector<faiss::idx_t> delBatchs = { 1, 23, 50, 10000 };

    std::vector<float> data(DIM * ntotal);
    for (int i = 0; i < DIM * ntotal; i++) {
        data[i] = drand48();
    }
    Norm(data.data(), ntotal, DIM);

    faiss::ascend::AscendIndexSQConfig conf(DEVICES);
    faiss::ascend::AscendIndexSQ index(DIM, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE, conf);

    index.train(ntotal, data.data());
    index.add(ntotal, data.data());
    // define ids
    std::vector<int64_t> ids(ntotal);
    std::iota(ids.begin(), ids.end(), 0);

    // save pre-delete basedata and index of vector
    std::vector<uint8_t> codesPre;
    std::vector<faiss::idx_t> idxMapPre;
    for (auto deviceId : conf.deviceList) {
        size_t size = index.getBaseSize(deviceId);
        std::vector<uint8_t> base(size * DIM);
        index.getBase(deviceId, (char *)base.data());
        codesPre.insert(codesPre.end(), base.begin(), base.end());

        std::vector<faiss::idx_t> idMap(size);
        index.getIdxMap(deviceId, idMap);
        idxMapPre.insert(idxMapPre.end(), idMap.begin(), idMap.end());
    }

    faiss::IDSelectorBatch del(delBatchs.size(), delBatchs.data());
    int rmCnt = 0;
    for (int i = 0; i < ntotal; i++) {
        rmCnt += del.is_member(ids[i]) ? 1 : 0;
    }

    size_t rmedCnt = index.remove_ids(del);
    ASSERT_EQ(rmedCnt, rmCnt);
    ASSERT_EQ(index.ntotal, (ntotal - rmedCnt));

    int tmpTotal = 0;
    for (auto deviceId : conf.deviceList) {
        tmpTotal += index.getBaseSize(deviceId);
    }
    EXPECT_EQ(tmpTotal, (ntotal - rmedCnt));

    std::vector<uint8_t> codes;
    std::vector<faiss::idx_t> idxMap;
    for (auto deviceId : conf.deviceList) {
        size_t size = index.getBaseSize(deviceId);
        std::vector<uint8_t> base(size * DIM);
        index.getBase(deviceId, (char *)base.data());
        codes.insert(codes.end(), base.begin(), base.end());

        std::vector<faiss::idx_t> idMap(size);
        index.getIdxMap(deviceId, idMap);
        idxMap.insert(idxMap.end(), idMap.begin(), idMap.end());
    }
    EXPECT_EQ(idxMapPre.size(), idxMap.size() + rmedCnt);
    EXPECT_EQ(codesPre.size(), codes.size() + rmedCnt * DIM);
    {
        int offset = 0;
        for (size_t i = 0; i < idxMap.size(); i++) {
            if (del.set.find(idxMapPre[i]) != del.set.end()) {
                // check ids
                EXPECT_EQ(idxMapPre[idxMap.size() + offset], idxMap[i]);
                // check vector
                for (int j = 0; j < DIM; j++) {
                    EXPECT_EQ(codesPre[(idxMap.size() + offset) * DIM + j], codes[i * DIM + j]);
                }

                offset += 1;
            } else {
                int ptr = i * DIM;
                // check ids
                EXPECT_EQ(idxMapPre[i], idxMap[i]);
                // check vector
                for (int j = 0; j < DIM; j++) {
                    EXPECT_EQ(codesPre[ptr + j], codes[ptr + j]);
                }
            }
        }
        EXPECT_EQ(offset, del.set.size());
    }
}

TEST(TestAscendIndexSQ, QPS)
{
    std::vector<size_t> ntotal = { 1000000 };
    std::vector<int> searchNum = { 1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 1000 };
    std::vector<float> benchmark = {285, 547, 633, 1684, 2682,
                                    3488, 4140, 3465, 3945, 3696, 4297};
    // Since the qps might have some tricky issue
    const float slack = 0.95;
    size_t maxSize = ntotal.back() * DIM;
    std::vector<float> data(maxSize);
    for (size_t i = 0; i < maxSize; i++) {
        data[i] = 1.0 * FastRand() / FAST_RAND_MAX;
    }

    faiss::ascend::AscendIndexSQConfig conf(DEVICES);
    faiss::ascend::AscendIndexSQ index(DIM, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE,
        conf);

    for (size_t i = 0; i < ntotal.size(); i++) {
        index.reset();
        for (auto deviceId : conf.deviceList) {
            int len = index.getBaseSize(deviceId);
            ASSERT_EQ(len, 0);
        }

        index.train(ntotal[i], data.data());
        index.add(ntotal[i], data.data());
        {
            int getTotal = 0;
            for (size_t k = 0; k < conf.deviceList.size(); k++) {
                int tmpTotal = index.getBaseSize(conf.deviceList[k]);
                getTotal += tmpTotal;
            }
            EXPECT_EQ(getTotal, ntotal[i]);
        }

        {
            for (size_t n = 0; n < searchNum.size(); n++) {
                int k = 100;
                int loopTimes = 100;
                std::vector<float> dist(searchNum[n] * k, 0);
                std::vector<faiss::idx_t> label(searchNum[n] * k, 0);
                double ts = GetMillisecs();
                for (int l = 0; l < loopTimes; l++) {
                    index.search(searchNum[n], data.data(), k, dist.data(), label.data());
                }
                double te = GetMillisecs();
                int cases = i * searchNum.size() + n;
                double qps = 1000 * searchNum[n] * loopTimes / (te - ts);
                printf("case[%d]: base:%zu, dim:%d, search num:%d, QPS:%.4f\n", cases, ntotal[i], DIM,
                    searchNum[n], qps);
                EXPECT_TRUE(qps > benchmark[n] * slack);
            }
        }
    }
}

TEST(TestAscendIndexSQ, CameraId)
{
    int dim = 64;
    int searchNum = 1;
    faiss::ascend::AscendIndexSQConfig conf({ 0 });
    conf.filterable = true;
    faiss::ascend::AscendIndexSQ index(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE, conf);
 
    int ntotal = 128;
    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }
    Norm(data.data(), ntotal, dim);
 
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
        IDFilter fliter;
    
        fliter.time_range[0] = 0;
        fliter.time_range[1] = 0x7fffffff;
        for (int j = 0; j < 16; j++) {
            fliter.camera_id_mask[j] = search_cid(e1);
        }
        filters[i] = fliter;
    }
 
    index.train(ntotal, data.data());
    index.add_with_ids(ntotal, data.data(), ids.data());
 
    int k = 128;
    std::vector<float> dist(k * searchNum, 0);
    std::vector<faiss::idx_t> label(k * searchNum, 0);
 
    index.search_with_filter(searchNum, data.data(), k, dist.data(), label.data(), &filters);
    bool flag = false;
    for (int i = 0; i < searchNum; i++) {
        for (int j = 0; j < k; j++) {
            auto cid = label[j + i * k];
            printf("cid = %ld \n", cid);
            if (cid == -1) continue;
            uint8_t camera_id = (cid >> 42) & 0x7F;
            printf("camera_id = %u \n", camera_id);
            auto mask = filters[i].camera_id_mask[camera_id / 8] & (1u << (camera_id % 8u));
            std::cout << "mask = "<< mask << std::endl;
            std::cout << std::endl;
            if (mask == 0) {
                // mask 为0说明过滤后的结果中包含需要过滤的标签
                printf("cid = %ld, camera_id = %u, camera_id_mask[camera_id / 8] = %u \n",
                    cid, camera_id, filters[i].camera_id_mask[camera_id / 8]);
                std::cout << std::endl;
                flag = true;
            }
        }
    }
    if (flag) {
        ASSERT_TRUE(false);
    }
}

TEST(TestAscendIndexSQ, MultiCameraId)
{
    int dim = 64;
    int searchNum = 2;
    int indexNum = 10;
    faiss::ascend::AscendIndexSQConfig conf({ 0 });
    conf.filterable = true;

    int ntotal = 128;
    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }
    Norm(data.data(), ntotal, dim);

    std::vector<int64_t> ids(ntotal, 0);
    int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine e1(seed);
    std::uniform_int_distribution<int32_t> id(0, std::numeric_limits<int32_t>::max());
    std::uniform_int_distribution<uint8_t> search_cid(0, 127);

    for (int i = 0; i < ntotal; i++) {
        ids[i] = (static_cast<int64_t>(search_cid(e1)) << 42) + (static_cast<int64_t>(id(e1)) << 10);
    }
    
    std::vector<faiss::ascend::AscendIndex *> indexes(indexNum, nullptr);
    for (int i = 0; i < indexNum; ++i) {
        indexes[i] =
            new faiss::ascend::AscendIndexSQ(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE, conf);

        indexes[i]->train(ntotal, data.data());
        indexes[i]->add_with_ids(ntotal, data.data(), ids.data());
    }

    void *multiFilters[searchNum];
    IDFilter idFilters[indexNum * searchNum];
    for (int queryIdx = 0; queryIdx < searchNum; queryIdx++) {
        for (int indexIdx = 0; indexIdx < indexNum; indexIdx++) {
            IDFilter idFilter;
            idFilter.time_range[0] = 0;
            idFilter.time_range[1] = 0x7fffffff;
            for (int i = 0; i < 16; i++) {
                idFilter.camera_id_mask[i] = search_cid(e1);
            }
            idFilters[queryIdx * indexNum + indexIdx] = idFilter;
        }
        multiFilters[queryIdx] = &idFilters[queryIdx * indexNum];
    }

    int k = 128;
    std::vector<float> dist(indexNum * k * searchNum, 0);
    std::vector<faiss::idx_t> label(indexNum * k * searchNum, 0);

    SearchWithFilter(indexes, searchNum, data.data(), k, dist.data(), label.data(), multiFilters, false);
    bool flag = false;
    for (int z = 0; z < indexNum; z++) {
        for (int i = 0; i < searchNum; i++) {
            for (int j = 0; j < k; j++) {
                auto cid = label[j + i * k + z * k * searchNum];
                
                if (cid == -1) continue;
                uint8_t camera_id = (cid >> 42) & 0x7F;
                printf("camera_id = %u \n", camera_id);
                auto mask = idFilters[z + i * indexNum].camera_id_mask[camera_id / 8] & (1u << (camera_id % 8u));
                std::cout << "mask = "<< mask << std::endl;
                std::cout << std::endl;
                if (mask == 0) {
                    // mask 为0说明过滤后的结果中包含需要过滤的标签
                    printf("cid = %ld, camera_id = %u, camera_id_mask[camera_id / 8] = %u \n",
                        cid, camera_id, idFilters[z + i * indexNum].camera_id_mask[camera_id / 8]);
                    std::cout << std::endl;
                    flag = true;
                }
            }
        }
    }
    if (flag) {
        ASSERT_TRUE(false);
    }
    for (int i = 0; i < indexNum; ++i) {
        delete indexes[i];
    }
}

inline bool FilterPass(int ntotal, faiss::idx_t cid, uint8_t *camera_id_mask, int queryIndex)
{
    uint8_t camera_id = (cid >> 42) & 0x7F;
    uint8_t val = camera_id_mask[camera_id / 8];
    int offset = camera_id % 8;
    return (val >> offset) % 2 != 0;
}
 
bool CompareResult(int ntotal, std::vector<float> &dist, std::vector<faiss::idx_t> &label,
    std::vector<float> &distWithMask, std::vector<faiss::idx_t> &labelWithMask,
    uint8_t* camera_id_mask, int queryNum, int topk)
{
    std::vector<std::vector<float>> filteredDist;
    std::vector<std::vector<faiss::idx_t>> filteredLabel;
    for(int i = 0; i < queryNum; i++) {
        std::vector<float> tmpDist;
        std::vector<faiss::idx_t> tmpLabel;
        for (int j = 0; j < topk; j++){
            if (FilterPass(ntotal, label[i * topk + j], camera_id_mask, i)) {
                tmpDist.push_back(dist[i * topk + j]);
                tmpLabel.push_back(label[i * topk + j]);
            }
        }
        filteredDist.push_back(tmpDist);
        filteredLabel.push_back(tmpLabel);
    }
 
    bool hasPrintOnce = false;
    for(int j = 0; j < queryNum; j++) {
        if (!hasPrintOnce && filteredLabel[j].size() > 0) {
            std::cout << "one mask query result's length is " << filteredLabel[j].size() << std::endl;
            hasPrintOnce = true;
        }
        
        for (size_t i = 0; i < filteredLabel[j].size(); i++) {
            auto iter = std::find(labelWithMask.begin() + j * topk,
                labelWithMask.begin() + (j + 1) * topk, filteredLabel[j][i]);
            if (iter == labelWithMask.begin() + (j + 1) * topk) {
                printf(" Label not found: filteredLabel[%d][%zu] = %zu, dist = %f\n", j, i,
                    filteredLabel[j][i], filteredDist[j][i]);
                return false;
            }
 
            if (*(distWithMask.begin() + (iter - labelWithMask.begin())) != filteredDist[j][i]) {
                printf(" Filtered Dist not equal %f --- %f\n",
                    *(distWithMask.begin() + (iter - labelWithMask.begin())), filteredDist[j][i]);
                return false;
            }
        }
    }
    return true;
}
 
void testMaskPrecision(faiss::MetricType metric)
{
    std::vector<int> searchNum = { 1 };
    faiss::ascend::AscendIndexSQConfig conf({ 0 });
    conf.filterable = true;
    int dim = 64;
    int ntotal = 300000; // over one block size 262144
    int topk = 100;
 
    std::vector<float> base(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        base[i] = drand48();
    }
    Norm(base.data(), ntotal, dim);
 
    faiss::ascend::AscendIndexSQ index(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit, metric, conf);
    std::vector<int64_t> ids(ntotal, 0);
 
    IDFilter fliter;
    fliter.time_range[0] = 0;
    fliter.time_range[1] = 0x7fffffff;
    int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine e1(seed);
    std::uniform_int_distribution<int32_t> id(0, std::numeric_limits<int32_t>::max());
    std::uniform_int_distribution<uint8_t> search_cid(0, std::numeric_limits<uint8_t>::max());
 
    for (int i = 0; i < ntotal; i++) {
        ids[i] = (static_cast<int64_t>(search_cid(e1)) << 42) + (static_cast<int64_t>(id(e1)) << 10);
    }
 
    for (int j = 0; j < 16; j++) {
        fliter.camera_id_mask[j] = search_cid(e1);
    }
 
    index.train(ntotal, base.data());
    index.add_with_ids(ntotal, base.data(), ids.data());
 
    for (size_t i = 0; i < searchNum.size(); i++) {
        std::vector<float> dist(searchNum[i] * topk, 0);
        std::vector<faiss::idx_t> label(searchNum[i] * topk, 0);
        double ts = GetMillisecs();
        index.search(searchNum[i], base.data(), topk, dist.data(), label.data());
 
        std::vector<float> distWithMask(searchNum[i] * topk, 0);
        std::vector<faiss::idx_t> labelWithMask(searchNum[i] * topk, 0);
        index.search_with_filter(searchNum[i], base.data(), topk, distWithMask.data(), labelWithMask.data(), &fliter);
        // search全量搜索结果中的label，根据设置的filter过滤后得到的labels，它们在search_with_filter过滤搜索结果中必然存在且距离值必然相等。
        EXPECT_TRUE(CompareResult(ntotal, dist, label, distWithMask, labelWithMask,
            fliter.camera_id_mask, searchNum[i], topk));
        double te = GetMillisecs();
        printf("case[%zu]: type:%d, base:%d, dim:%d, topk:%d, search num:%d, duration:%.4f\n",
            i, metric, ntotal, dim, topk, searchNum[i], 1000 * searchNum[i] / (te - ts));
    }
}
 
TEST(TestAscendIndexSQ, MaskPrecision)
{
    printf("test AscendIndexSQ search_with_masks, type: METRIC_INNER_PRODUCT\n");
    testMaskPrecision(faiss::METRIC_INNER_PRODUCT);
    printf("test AscendIndexSQ search_with_masks, type: METRIC_L2\n");
    testMaskPrecision(faiss::METRIC_L2);
}

} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}