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
#include <faiss/ascend/AscendMultiIndexSearch.h>
#include <faiss/ascend/AscendCloner.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/index_io.h>

#include <sys/time.h>

namespace {
const auto METRIC_TYPE = faiss::METRIC_INNER_PRODUCT;
const auto DIM = 64;
const std::initializer_list<int> DEVICE_IDS = { 0 };

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

TEST(TestAscendIndexSQSlim, All)
{
    int ntotal = 200000;

    std::vector<float> data(DIM * ntotal);
    for (int i = 0; i < DIM * ntotal; i++) {
        data[i] = drand48();
    }
    Norm(data.data(), ntotal, DIM);

    faiss::ascend::AscendIndexSQConfig conf(DEVICE_IDS);
    conf.slim = false;
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

TEST(TestAscendIndexSQSlim, AddWithIds)
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

    faiss::ascend::AscendIndexSQConfig conf(DEVICE_IDS);
    conf.slim = false;
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

TEST(TestAscendIndexSQSlim, CopyFrom)
{
    int ntotal = 250000;

    std::vector<float> data(DIM * ntotal);
    for (int i = 0; i < DIM * ntotal; i++) {
        data[i] = drand48();
    }
    Norm(data.data(), ntotal, DIM);

    faiss::IndexScalarQuantizer cpuIndex(DIM, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE);
    cpuIndex.train(ntotal, data.data());
    cpuIndex.add(ntotal, data.data());

    faiss::ascend::AscendIndexSQConfig conf(DEVICE_IDS);
    conf.slim = false;
    faiss::ascend::AscendIndexSQ index(&cpuIndex, conf);

    EXPECT_EQ(index.d, DIM);
    EXPECT_EQ(index.ntotal, ntotal);

    // only make sure the format of basedata is same
    faiss::IndexScalarQuantizer cpuIndexRef(DIM, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE);
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

TEST(TestAscendIndexSQSlim, CopyTo)
{
    int ntotal = 250000;

    std::vector<float> data(DIM * ntotal);
    for (int i = 0; i < DIM * ntotal; i++) {
        data[i] = drand48();
    }
    Norm(data.data(), ntotal, DIM);

    faiss::ascend::AscendIndexSQConfig conf(DEVICE_IDS);
    conf.slim = false;
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

TEST(TestAscendIndexSQSlim, CloneAscend2CPU)
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
    faiss::ascend::AscendIndexSQConfig conf(DEVICE_IDS);
    conf.slim = false;
    faiss::ascend::AscendIndexSQ ascendIndex(DIM, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE, conf);

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

TEST(TestAscendIndexSQSlim, CloneCPU2Ascend)
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
    const char *indexfilename = "./SQIndex_2500000.faiss";
    faiss::Index *initIndex = faiss::read_index(indexfilename);
    ASSERT_FALSE(initIndex == nullptr);

    // ascend index
    faiss::ascend::AscendClonerOptions options;
    options.slim = false;
    faiss::ascend::AscendIndexSQ *ascendIndex = dynamic_cast<faiss::ascend::AscendIndexSQ *>(
        faiss::ascend::index_cpu_to_ascend(DEVICE_IDS, initIndex, &options));
    ASSERT_FALSE(ascendIndex == nullptr);

    int lenall = 0;
    for (auto deviceId : DEVICE_IDS) {
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

TEST(TestAscendIndexSQSlim, RemoveRange)
{
    int ntotal = 200000;
    int delRangeMin = 0;
    int delRangeMax = 4;

    std::vector<float> data(DIM * ntotal);
    for (int i = 0; i < DIM * ntotal; i++) {
        data[i] = drand48();
    }
    Norm(data.data(), ntotal, DIM);

    faiss::ascend::AscendIndexSQConfig conf(DEVICE_IDS);
    conf.slim = false;
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

TEST(TestAscendIndexSQSlim, RemoveBatch)
{
    int ntotal = 200000;
    std::vector<faiss::idx_t> delBatchs = { 1, 23, 50, 10000 };

    std::vector<float> data(DIM * ntotal);
    for (int i = 0; i < DIM * ntotal; i++) {
        data[i] = drand48();
    }
    Norm(data.data(), ntotal, DIM);

    faiss::ascend::AscendIndexSQConfig conf(DEVICE_IDS);
    conf.slim = false;
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

TEST(TestAscendIndexSQSlim, MultiIndexNotMerge)
{
    int ntotal = 200000;
    int indexNum = 4;

    std::vector<float> data(DIM * ntotal);
    for (int i = 0; i < DIM * ntotal; i++) {
        data[i] = drand48();
    }
    Norm(data.data(), ntotal, DIM);

    faiss::ascend::AscendIndexSQConfig conf(DEVICE_IDS);
    conf.slim = false;

    faiss::ascend::AscendIndexSQ index0(DIM, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE, conf);
    faiss::ascend::AscendIndexSQ index1(DIM, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE, conf);
    faiss::ascend::AscendIndexSQ index2(DIM, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE, conf);
    faiss::ascend::AscendIndexSQ index3(DIM, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE, conf);
    const std::vector<faiss::ascend::AscendIndexSQ *> indexes({ &index0, &index1, &index2, &index3 });

    for (int i = 0; i < indexNum; ++i) {
        for (auto deviceId : conf.deviceList) {
            int len0 = indexes[i]->getBaseSize(deviceId);
            ASSERT_EQ(len0, 0);
        }

        indexes[i]->train(ntotal, data.data());
        indexes[i]->add(ntotal, data.data());
        EXPECT_EQ(index0.ntotal, ntotal);
    }

    {
        std::vector<faiss::ascend::AscendIndex *> indexes({ &index0, &index2, &index3 });

        int batch = 4;
        for (int i = ntotal - 40; i < ntotal; i += batch) {
            int k = 100;
            std::vector<float> dist(indexes.size() * k * batch, 0);
            std::vector<faiss::idx_t> label(indexes.size() * k * batch, 0);
            Search(indexes, batch, data.data() + i * DIM, k, dist.data(), label.data(), false);

            for (size_t idx = 0; idx < indexes.size(); ++idx) {
                ASSERT_EQ(label[idx * batch * k + 0], i);
                ASSERT_EQ(label[idx * batch * k + k], i + 1);
                ASSERT_EQ(label[idx * batch * k + k * 2], i + 2);
                ASSERT_EQ(label[idx * batch * k + k * 3], i + 3);
            }
        }
    }
}

TEST(TestAscendIndexSQSlim, MultiIndexMerge)
{
    size_t ntotal = 200000;
    int indexNum = 4;

    std::vector<std::vector<float>> data(indexNum, std::vector<float>(DIM * ntotal, 0));
    for (int i = 0; i < indexNum; ++i) {
        for (size_t j = 0; j < ntotal * DIM; ++j) {
            data[i][j] = drand48();
        }
    }

    std::vector<std::vector<int64_t>> ids(indexNum, std::vector<int64_t>(ntotal, 0));
    for (int i = 0; i < indexNum; ++i) {
        std::iota(ids[i].begin(), ids[i].end(), i * ntotal);
    }

    for (int i = 0; i < indexNum; ++i) {
        Norm(data[i].data(), ntotal, DIM);
    }

    faiss::ascend::AscendIndexSQConfig conf(DEVICE_IDS);
    conf.slim = false;

    faiss::ascend::AscendIndexSQ index0(DIM, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE, conf);
    faiss::ascend::AscendIndexSQ index1(DIM, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE, conf);
    faiss::ascend::AscendIndexSQ index2(DIM, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE, conf);
    faiss::ascend::AscendIndexSQ index3(DIM, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE, conf);
    const std::vector<faiss::ascend::AscendIndexSQ *> indexes({ &index0, &index1, &index2, &index3 });

    for (int i = 0; i < indexNum; ++i) {
        for (auto deviceId : conf.deviceList) {
            int len0 = indexes[i]->getBaseSize(deviceId);
            ASSERT_EQ(len0, 0);
        }

        indexes[i]->train(ntotal, data[i].data());
        indexes[i]->add_with_ids(ntotal, data[i].data(), ids[i].data());
        EXPECT_EQ(indexes[i]->ntotal, ntotal);
    }

    {
        std::vector<faiss::ascend::AscendIndex *> indexes({ &index0, &index2, &index3 });

        int batch = 4;
        for (size_t i = ntotal - 40; i < ntotal; i += batch) {
            int k = 100;
            std::vector<float> dist(k * batch, 0);
            std::vector<faiss::idx_t> label(k * batch, 0);
            Search(indexes, batch, data[0].data() + i * DIM, k, dist.data(), label.data(), true);

            ASSERT_EQ(label[0], i);
            ASSERT_EQ(label[k], i + 1);
            ASSERT_EQ(label[k * 2], i + 2);
            ASSERT_EQ(label[k * 3], i + 3);
        }
    }
}

TEST(TestAscendIndexSQSlim, Create3000Index)
{
    int ntotal = 3000;
    int indexNum = 1000;

    srand48(1000);
    std::vector<float> data(DIM * ntotal);
    for (int i = 0; i < DIM * ntotal; i++) {
        data[i] = drand48();
    }
    Norm(data.data(), ntotal, DIM);

    // ascend index
    faiss::ascend::AscendIndexSQConfig conf(DEVICE_IDS, true);
    faiss::ascend::AscendIndexSQ ascendIndex(DIM, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE, conf);

    ascendIndex.train(ntotal, data.data());
    // add ground truth
    ascendIndex.add(ntotal, data.data());

    // write index with cpu index
    faiss::Index *cpuIndex = faiss::ascend::index_ascend_to_cpu(&ascendIndex);

    // ascend index
    faiss::ascend::AscendClonerOptions options;
    options.slim = false;

    std::vector<faiss::ascend::AscendIndexSQ *> ascendIndexs;
    for (int i = 0; i < indexNum; ++i) {
        faiss::ascend::AscendIndexSQ *indexSQ = dynamic_cast<faiss::ascend::AscendIndexSQ *>(
            faiss::ascend::index_cpu_to_ascend(DEVICE_IDS, cpuIndex, &options));
        ASSERT_FALSE(indexSQ == nullptr);
        printf("create index %d\n", i);
        ascendIndexs.push_back(indexSQ);
    }

    for (int i = 0; i < 1; i++) {
        int k = 100;
        int idx = i * 5;
        std::vector<float> dist(k, 0);
        std::vector<faiss::idx_t> label(k, 0);

        for (int m = 0; m < 10; ++m) {
            double ts = GetMillisecs();
            for (int j = 0; j < 100; j++) {
                ascendIndexs[m]->search(1, data.data() + idx * DIM, k, dist.data(), label.data());
            }
            double te = GetMillisecs();
            printf("all %f, means %f.\n", te - ts, (te - ts) / 500);
            printf("ascend idx %d dist %f %f %f %f %f lable %lu %lu %lu %lu %lu.\n", idx, dist[0], dist[1], dist[2],
                dist[3], dist[4], label[0], label[1], label[2], label[3], label[4]);
            EXPECT_EQ(label[0], idx);
        }
    }

    for (auto indexSQ : ascendIndexs) {
        delete indexSQ;
    }
    delete cpuIndex;
}

TEST(TestAscendIndexSQSlim, MultiIndexQPS)
{
    size_t ntotal = 10000;
    std::vector<int> searchNum = { 1, 2, 4, 8, 16, 32, 64, 128 };
    int indexNum = 100;
    bool slim = true;

    size_t maxSize = ntotal * DIM;
    std::vector<std::vector<float>> data(indexNum, std::vector<float>(maxSize));
    for (int i = 0; i < indexNum; ++i) {
        for (size_t j = 0; j < maxSize; j++) {
            data[i][j] = 1.0 * FastRand() / FAST_RAND_MAX;
        }
        Norm(data[i].data(), ntotal, DIM);
    }

    std::vector<faiss::ascend::AscendIndex *> indexes;
    faiss::ascend::AscendIndexSQConfig conf(DEVICE_IDS, 256 * 1024 * 1024);
    conf.slim = false;

    for (int i = 0; i < indexNum; ++i) {
        auto index =
            new faiss::ascend::AscendIndexSQ(DIM, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE, conf);
        ASSERT_FALSE(index == nullptr);
        indexes.push_back(index);
    }

    for (int i = 0; i < indexNum; ++i) {
        const auto index = indexes[i];
        index->reset();
        for (auto deviceId : conf.deviceList) {
            int len = dynamic_cast<faiss::ascend::AscendIndexSQ *>(index)->getBaseSize(deviceId);
            ASSERT_EQ(len, 0);
        }
    }

    for (int i = 0; i < indexNum; ++i) {
        const auto index = indexes[i];
        index->train(ntotal, data[i].data());
        index->add(ntotal, data[i].data());

        {
            int getTotal = 0;
            for (size_t k = 0; k < conf.deviceList.size(); k++) {
                int tmpTotal = dynamic_cast<faiss::ascend::AscendIndexSQ *>(index)->getBaseSize(conf.deviceList[k]);
                getTotal += tmpTotal;
            }
            EXPECT_EQ(getTotal, ntotal);
        }
    }

    for (size_t n = 0; n < searchNum.size(); n++) {
        int k = 100;
        int loopTimes = 100;
        std::vector<float> dist(searchNum[n] * k, 0);
        std::vector<faiss::idx_t> label(searchNum[n] * k, 0);
        double ts = GetMillisecs();
        for (int l = 0; l < loopTimes; l++) {
            for (int i = 0; i < indexNum; ++i) {
                indexes[i]->search(searchNum[n], data[0].data(), k, dist.data(), label.data());
            }
        }
        double te = GetMillisecs();
        printf("multi search: false, index num: %d, base:%zu, slim:%d, dim:%d, search num:%d, QPS:%.4f\n", indexNum,
            ntotal, slim, DIM, searchNum[n], 1000 * searchNum[n] * loopTimes / (te - ts));
    }

    if (slim) {
        for (size_t n = 0; n < searchNum.size(); n++) {
            int k = 100;
            int loopTimes = 100;
            std::vector<float> dist(indexNum * searchNum[n] * k, 0);
            std::vector<faiss::idx_t> label(indexNum * searchNum[n] * k, 0);
            double ts = GetMillisecs();
            for (int l = 0; l < loopTimes; l++) {
                Search(indexes, searchNum[n], data[0].data(), k, dist.data(), label.data(), false);
            }
            double te = GetMillisecs();
            printf("multi search: true, index num: %d, base:%zu, slim:%d, dim:%d, search num:%d, QPS:%.4f\n", indexNum,
                ntotal, slim, DIM, searchNum[n], 1000 * searchNum[n] * loopTimes / (te - ts));
        }
    }

    for (int i = 0; i < indexNum; ++i) {
        delete indexes[i];
    }
}

TEST(TestAscendIndexSQSlim, OneIndexQPS)
{
    size_t ntotal = 1000000;
    int k = 100;
    bool slim = false;
    std::vector<int> searchNum = { 1, 2, 4, 8, 16, 32, 64, 128 };

    size_t maxSize = ntotal * DIM;
    std::vector<float> data(maxSize);
    for (size_t i = 0; i < maxSize; ++i) {
        data[i] = 1.0 * FastRand() / FAST_RAND_MAX;
    }
    Norm(data.data(), ntotal, DIM);

    faiss::ascend::AscendIndexSQConfig conf(DEVICE_IDS);
    conf.slim = false;

    faiss::ascend::AscendIndexSQ index(DIM, faiss::ScalarQuantizer::QuantizerType::QT_8bit, METRIC_TYPE, conf);

    {
        index.reset();
        for (auto deviceId : conf.deviceList) {
            int len = index.getBaseSize(deviceId);
            ASSERT_EQ(len, 0);
        }

        index.train(ntotal, data.data());
        index.add(ntotal, data.data());
    }

    {
        int getTotal = 0;
        for (size_t k = 0; k < conf.deviceList.size(); k++) {
            int tmpTotal = index.getBaseSize(conf.deviceList[k]);
            getTotal += tmpTotal;
        }
        EXPECT_EQ(getTotal, ntotal);
    }

    for (size_t n = 0; n < searchNum.size(); n++) {
        int loopTimes = 100;
        std::vector<float> dist(searchNum[n] * k, 0);
        std::vector<faiss::idx_t> label(searchNum[n] * k, 0);

        double ts = GetMillisecs();
        for (int l = 0; l < loopTimes; l++) {
            index.search(searchNum[n], data.data(), k, dist.data(), label.data());
        }
        double te = GetMillisecs();
        printf("base:%zu, slim=%d, dim:%d, search num:%d, QPS:%.4f\n", ntotal, slim, DIM, searchNum[n],
            1000 * searchNum[n] * loopTimes / (te - ts));
    }
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}