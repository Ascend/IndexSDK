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
#include <faiss/index_io.h>

#include <sys/time.h>

namespace {
const int DIM = 64;
const std::vector<int> DEVICES = { 0,1,2,3 };
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
    const char *outfilename = "./SQIndex_250000.faiss";
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
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}