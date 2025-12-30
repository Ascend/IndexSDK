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


#include <random>
#include <sys/time.h>
#include <gtest/gtest.h>
#include <faiss/ascend/AscendIndexInt8Flat.h>
#include <faiss/ascend/AscendCloner.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/Clustering.h>
#include <faiss/index_io.h>

namespace {
const int BLOCKSIZE = 16384 * 16;
const int RESOURCE = -1;

inline void GenerateCodes(int8_t *codes, int total, int dim, int seed = -1)
{
    std::default_random_engine e((seed > 0) ? seed : time(nullptr));
    std::uniform_real_distribution<float> rCode(0.0f, 1.0f);
    for (int i = 0; i < total; i++) {
        for (int j = 0; j < dim; j++) {
            // uint8's max value is 255, int8's max value is 255 - 128
            codes[i * dim + j] = static_cast<int8_t>(255 * rCode(e) - 128);
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

inline void AssertInt8Equal(size_t count, const int8_t *gt, const int8_t *data)
{
    for (size_t i = 0; i < count; i++) {
        ASSERT_TRUE(gt[i] == data[i]) << "i: " << i << " gt: " << int(gt[i]) << " data: " << int(data[i]) << std::endl;
    }
}

TEST(TestAscendIndexInt8Flat, CopyTo)
{
    int dim = 384;
    int ntotal = 250000;

    std::vector<int8_t> base(dim * ntotal);
    GenerateCodes(base.data(), ntotal, dim);

    faiss::ascend::AscendIndexInt8FlatConfig conf({ 0 }, RESOURCE, BLOCKSIZE);
    faiss::ascend::AscendIndexInt8Flat index(dim, faiss::METRIC_L2, conf);
    index.verbose = true;

    index.add(ntotal, base.data());

    faiss::IndexScalarQuantizer cpuIndex(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit);
    index.copyTo(&cpuIndex);

    EXPECT_EQ(cpuIndex.ntotal, ntotal);
    EXPECT_EQ(cpuIndex.d, dim);

    {
        int tmpTotal = 0;
        std::vector<int8_t> xb;
        for (auto deviceId : conf.deviceList) {
            size_t size = index.getBaseSize(deviceId);
            std::vector<int8_t> codes(size * dim);
            index.getBase(deviceId, codes);

            xb.insert(xb.end(), codes.begin(), codes.end());
            tmpTotal += size;
        }

        EXPECT_EQ(tmpTotal, ntotal);
        AssertInt8Equal(xb.size(), xb.data(), reinterpret_cast<int8_t *>(cpuIndex.codes.data()));
    }
}

TEST(TestAscendIndexInt8Flat, CopyFrom)
{
    int dim = 384;
    int ntotal = 250000;

    std::vector<int8_t> base(dim * ntotal);
    GenerateCodes(base.data(), ntotal, dim);

    faiss::ascend::AscendIndexInt8FlatConfig conf({ 0 }, RESOURCE, BLOCKSIZE);
    faiss::ascend::AscendIndexInt8Flat index(dim, faiss::METRIC_L2, conf);
    index.verbose = true;

    index.add(ntotal, base.data());
    EXPECT_EQ(index.ntotal, ntotal);

    faiss::IndexScalarQuantizer cpuIndex(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit);
    index.copyTo(&cpuIndex);

    EXPECT_EQ(cpuIndex.ntotal, ntotal);
    EXPECT_EQ(cpuIndex.d, dim);

    {
        int tmpTotal = 0;
        std::vector<int8_t> xb;
        for (auto deviceId : conf.deviceList) {
            size_t size = index.getBaseSize(deviceId);
            std::vector<int8_t> codes(size * dim);
            index.getBase(deviceId, codes);

            xb.insert(xb.end(), codes.begin(), codes.end());
            tmpTotal += size;
        }

        EXPECT_EQ(tmpTotal, ntotal);
        printf("compare xb and cpuIndex\n");
        AssertInt8Equal(xb.size(), xb.data(), reinterpret_cast<int8_t *>(cpuIndex.codes.data()));
    }

    faiss::ascend::AscendIndexInt8FlatConfig confNew({ 0 });
    faiss::ascend::AscendIndexInt8Flat indexNew(&cpuIndex, confNew);

    EXPECT_EQ(indexNew.d, dim);
    EXPECT_EQ(indexNew.ntotal, ntotal);

    {
        int sizeAscend = 0;
        std::vector<int8_t> xbAsend;
        for (auto deviceId : confNew.deviceList) {
            size_t size = indexNew.getBaseSize(deviceId);
            std::vector<int8_t> codes(size * dim);
            indexNew.getBase(deviceId, codes);
            xbAsend.insert(xbAsend.end(), codes.begin(), codes.end());
            sizeAscend += size;
        }
        ASSERT_EQ(ntotal, sizeAscend);
        ASSERT_EQ(xbAsend.size(), base.size());
        printf("compare xbAsend and base\n");
        AssertInt8Equal(xbAsend.size(), xbAsend.data(), reinterpret_cast<int8_t *>(cpuIndex.codes.data()));
    }
}

TEST(TestAscendIndexInt8Flat, CloneAscend2CPU)
{
    int dim = 384;
    int ntotal = 250000;
    int xbSize = ntotal * 4;

    int seed = 1000;
    std::vector<int8_t> base(dim * ntotal);
    GenerateCodes(base.data(), ntotal, dim, seed);

    for (int i = 0; i < 10; i++) {
        printf("%d\t", int(base[i]));
    }
    printf("\n");

    // ascend index
    faiss::ascend::AscendIndexInt8FlatConfig conf({ 0 }, RESOURCE, BLOCKSIZE);
    faiss::ascend::AscendIndexInt8Flat ascendIndex(dim, faiss::METRIC_L2, conf);

    // add ground truth
    ascendIndex.add(ntotal, base.data());

    // add 2250w vector
    for (int i = 0; i < (xbSize / ntotal - 1); i++) {
        std::vector<int8_t> dataTmp(dim * ntotal);
        GenerateCodes(dataTmp.data(), ntotal, dim);
        ascendIndex.add(ntotal, dataTmp.data());
        printf("add %d times of ntotal data.\n", i);
    }

    // write index with cpu index
    faiss::Index *cpuIndex = faiss::ascend::index_int8_ascend_to_cpu(&ascendIndex);
    ASSERT_FALSE(cpuIndex == nullptr);
    const char *outfilename = "./int8FlatIndex_250000.faiss";
    write_index(cpuIndex, outfilename);
    printf("write ascendIndex to file ok!\n");

    int lenall = 0;
    for (auto deviceId : conf.deviceList) {
        lenall += ascendIndex.getBaseSize(deviceId);
    }

    EXPECT_EQ(lenall, xbSize);

    for (int i = 0; i < 10; i++) {
        int k = 1000;
        int idx = i * 5;
        std::vector<float> dist(k, 0);
        std::vector<faiss::idx_t> label(k, 0);
        double ts = GetMillisecs();
        for (int j = 0; j < 500; j++) {
            ascendIndex.search(1, base.data() + idx * dim, k, dist.data(), label.data());
        }
        double te = GetMillisecs();
        printf("all %f, means %f.\n", te - ts, (te - ts) / 500);
        EXPECT_EQ(label[0], idx);
    }
    delete cpuIndex;
}

TEST(TestAscendIndexInt8Flat, CloneCPU2Ascend)
{
    int dim = 384;
    int ntotal = 250000;
    int xbSize = ntotal * 4;

    int seed = 1000;
    std::vector<int8_t> base(dim * ntotal);
    GenerateCodes(base.data(), ntotal, dim, seed);

    for (int i = 0; i < 10; i++) {
        printf("%d\t", int(base[i]));
    }
    printf("\n");

    const char *indexfilename = "./int8FlatIndex_250000.faiss";
    faiss::Index *initIndex = faiss::read_index(indexfilename);
    ASSERT_FALSE(initIndex == nullptr);

    // ascend index
    std::vector<int> devices = { 0 };
    faiss::ascend::AscendIndexInt8Flat *ascendIndex =
        dynamic_cast<faiss::ascend::AscendIndexInt8Flat *>(faiss::ascend::index_int8_cpu_to_ascend(devices, initIndex));
    ASSERT_FALSE(ascendIndex == nullptr);

    int lenall = 0;
    for (auto deviceId : devices) {
        lenall += ascendIndex->getBaseSize(deviceId);
    }

    EXPECT_EQ(lenall, xbSize);

    for (int i = 0; i < 10; i++) {
        int k = 1000;
        int idx = i * 5;
        std::vector<float> dist(k, 0);
        std::vector<faiss::idx_t> label(k, 0);
        double ts = GetMillisecs();
        for (int i = 0; i < 500; i++) {
            ascendIndex->search(1, base.data() + idx * dim, k, dist.data(), label.data());
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

TEST(TestAscendIndexInt8Flat, QPS)
{
    int dim = 384;
    size_t ntotal = 250000;
    std::vector<int> searchNum = { 1, 2, 4, 8, 16, 32, 64, 128 };
    std::vector<float> benchmark = {189, 462, 663, 1271,
                                    2386, 3372, 3343, 3544};
    // Since the qps might have some tricky issue
    const float slack = 0.95;
    faiss::ascend::AscendIndexInt8FlatConfig conf({ 0 }, RESOURCE, BLOCKSIZE);
    faiss::ascend::AscendIndexInt8Flat index(dim, faiss::METRIC_L2, conf);
    index.verbose = true;

    printf("generate data\n");
    std::vector<int8_t> base(ntotal * dim);
    GenerateCodes(base.data(), ntotal, dim);

    printf("add data\n");
    index.add(ntotal, base.data());
    {
        for (size_t n = 0; n < searchNum.size(); n++) {
            int k = 128;
            int loopTimes = 10;
            std::vector<float> dist(searchNum[n] * k, 0);
            std::vector<faiss::idx_t> label(searchNum[n] * k, 0);
            double ts = GetMillisecs();
            for (int l = 0; l < loopTimes; l++) {
                index.search(searchNum[n], base.data(), k, dist.data(), label.data());
            }
            double te = GetMillisecs();
            double qps = 1000 * searchNum[n] * loopTimes / (te - ts);
            printf("case[%zu]: base:%zu, dim:%d, search num:%d, QPS:%.4f\n", n, ntotal, dim, searchNum[n], qps);
            EXPECT_TRUE(qps > benchmark[n] * slack);
        }
    }
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
