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
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <random>
#include <algorithm>
#include <sys/time.h>
#include <gtest/gtest.h>
#include <faiss/index_io.h>
#include <faiss/IndexFlat.h>
#include <faiss/ascend/AscendIndexFlat.h>
#include <faiss/ascend/AscendCloner.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>

namespace {
inline double GetMillisecs()
{
    struct timeval tv = { 0, 0 };
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

inline void AssertEqual(const std::vector<uint8_t> &gt, const std::vector<uint8_t> &data)
{
    const float epson = 1e-3;
    ASSERT_EQ(gt.size(), data.size());
    for (size_t i = 0; i < gt.size(); i++) {
        ASSERT_TRUE(fabs(gt[i] - data[i]) <= epson) << "i: " << i << " gt: " << gt[i] << " data: " << data[i] <<
            std::endl;
    }
}

// index-add-getBaseSize-getbase-copyto
TEST(TestAscendIndexFlat, All)
{
    int dim = 512;
    int ntotal = 200000;
    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }
    faiss::ascend::AscendIndexFlatConfig conf({ 0, 1, 2, 3 });
    faiss::ascend::AscendIndexFlat index(dim, faiss::METRIC_L2, conf);
    for (auto deviceId : conf.deviceList) {
        int len = index.getBaseSize(deviceId);
        ASSERT_EQ(len, 0);
    }

    index.add(ntotal, data.data());
    EXPECT_EQ(index.ntotal, ntotal);

    int deviceCnt = conf.deviceList.size();
    std::vector<uint8_t> codes;
    {
        int totals = 0;
        for (int i = 0; i < deviceCnt; i++) {
            int tmpTotal = index.getBaseSize(conf.deviceList[i]);
            std::vector<uint8_t> base(tmpTotal * dim * sizeof(float));
            index.getBase(conf.deviceList[i], (char *)base.data());
            codes.insert(codes.end(), base.begin(), base.end());
            totals += tmpTotal;
        }
        EXPECT_EQ(totals, ntotal);
    }

    faiss::IndexFlatL2 cpuIndex(dim);
    index.copyTo(&cpuIndex);
    EXPECT_EQ(cpuIndex.ntotal, ntotal);
    EXPECT_EQ(cpuIndex.d, dim);
    AssertEqual(codes, cpuIndex.codes);

    {
        std::vector<uint8_t> codesAsend;
        for (auto deviceId : conf.deviceList) {
            size_t size = index.getBaseSize(deviceId);
            std::vector<uint8_t> base(size * dim * sizeof(float));
            index.getBase(deviceId, (char *)base.data());
            codesAsend.insert(codesAsend.end(), base.begin(), base.end());
        }
        AssertEqual(codesAsend, cpuIndex.codes);
    }
    index.reset();
}

// index-add_with_ids-getBaseSize
TEST(TestAscendIndexFlat, add_with_ids)
{
    int dim = 512;
    int ntotal = 250000;
    faiss::ascend::AscendIndexFlatConfig conf({ 0, 1, 2 });
    faiss::ascend::AscendIndexFlat index(dim, faiss::METRIC_L2, conf);

    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }
    for (auto deviceId : conf.deviceList) {
        int len = index.getBaseSize(deviceId);
        ASSERT_EQ(len, 0);
    }

    std::vector<faiss::idx_t> ids(ntotal);
    std::iota(ids.begin(), ids.end(), 0);
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(ids), std::end(ids), rng);

    index.add_with_ids(ntotal, data.data(), ids.data());
    EXPECT_EQ(index.ntotal, ntotal);

    int deviceCnt = conf.deviceList.size();
    std::vector<float> baseData;

    int getTotal = 0;
    for (int i = 0; i < deviceCnt; i++) {
        int tmpTotal = index.getBaseSize(conf.deviceList[i]);
        getTotal += tmpTotal;
    }
    EXPECT_EQ(getTotal, ntotal);

    int batch = 4;
    for (int i = 195; i < 200; i++) {
        int k = 100;
        std::vector<float> dist(k * batch, 0);
        std::vector<faiss::idx_t> label(k * batch, 0);
        index.search(batch, data.data() + i * dim, k, dist.data(), label.data());
        ASSERT_EQ(label[0], ids[i]);
        ASSERT_EQ(label[k], ids[i + 1]);
        ASSERT_EQ(label[k * 2], ids[i + 2]);
        ASSERT_EQ(label[k * 3], ids[i + 3]);
        faiss::idx_t assign;
        index.assign(1, data.data() + i * dim, &assign);
        ASSERT_EQ(assign, ids[i]);
    }
}

TEST(TestAscendIndexFlat, CloneAscend2CPU)
{
    int dim = 512;
    int ntotal = 250000;

    srand48(1000);
    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }
    for (int i = 0; i < 20; i++) {
        printf("%.4f\t", data[i]);
    }
    // ascend index
    faiss::ascend::AscendIndexFlatConfig conf({ 0, 1 });
    faiss::ascend::AscendIndexFlat ascendIndex(dim, faiss::METRIC_L2, conf);

    // add ground truth
    ascendIndex.add(ntotal, data.data());

    // write index with cpu index
    faiss::Index *cpuIndex = faiss::ascend::index_ascend_to_cpu(&ascendIndex);
    ASSERT_FALSE(cpuIndex == nullptr);
    const char *outfilename = "./flatIndex_2500000.faiss";
    write_index(cpuIndex, outfilename);
    printf("write ascendIndex to file ok!\n");

    int lenall = 0;
    for (auto deviceId : conf.deviceList) {
        lenall += ascendIndex.getBaseSize(deviceId);
    }

    EXPECT_EQ(lenall, ntotal);

    int k = 1000;
    int idx = 5;
    std::vector<float> dist(k, 0);
    std::vector<faiss::idx_t> label(k, 0);

    clock_t start, end;
    start = clock();
    ascendIndex.search(1, data.data() + idx * dim, k, dist.data(), label.data());
    end = clock();
    int endtime = (end - start) / CLOCKS_PER_SEC;
    std::cout << "searchTime:" << endtime << "s\n";

    EXPECT_EQ(label[0], idx);

    // removeRange and removebatch
    // define ids
    int delRangeMin = 0;
    int delRangeMax = 4;
    faiss::IDSelectorRange delRange(delRangeMin, delRangeMax);

    // define batch
    std::vector<faiss::idx_t> batchs = { 10, 23, 50, 10000 };
    faiss::IDSelectorBatch delBatch(batchs.size(), batchs.data());

    const float epson = 1e-3;
    std::vector<faiss::idx_t> ids(ntotal);
    std::iota(ids.begin(), ids.end(), 0);

    // save pre-delete basedata and index of vector
    std::vector<float> xbPre;
    std::vector<faiss::idx_t> idxMapPre;
    for (auto deviceId : conf.deviceList) {
        size_t size = ascendIndex.getBaseSize(deviceId);
        std::vector<float> base(size * dim);
        ascendIndex.getBase(deviceId, (char *)base.data());
        xbPre.insert(xbPre.end(), base.begin(), base.end());

        std::vector<faiss::idx_t> idMap(size);
        ascendIndex.getIdxMap(deviceId, idMap);
        idxMapPre.insert(idxMapPre.end(), idMap.begin(), idMap.end());
    }

    size_t rmCntRange = 0;
    size_t rmCntBatch = 0;
    for (int i = 0; i < ntotal; i++) {
        rmCntRange += delRange.is_member(ids[i]) ? 1 : 0;
        rmCntBatch += delBatch.is_member(ids[i]) ? 1 : 0;
    }

    size_t rmedCntRange = ascendIndex.remove_ids(delRange);
    ASSERT_EQ(rmedCntRange, rmCntRange);
    ASSERT_EQ(ascendIndex.ntotal, static_cast<int>(ntotal - rmedCntRange));

    size_t rmedCntBatch = ascendIndex.remove_ids(delBatch);
    ASSERT_EQ(rmedCntBatch, rmCntBatch);
    ASSERT_EQ(ascendIndex.ntotal, static_cast<int>(ntotal - rmedCntRange - rmedCntBatch));

    int tmpTotal = 0;
    for (auto deviceId : conf.deviceList) {
        tmpTotal += ascendIndex.getBaseSize(deviceId);
    }
    EXPECT_EQ(tmpTotal, static_cast<int>(ntotal - rmedCntRange - rmedCntBatch));

    std::vector<float> xb;
    std::vector<faiss::idx_t> idxMap;
    for (auto deviceId : conf.deviceList) {
        size_t size = ascendIndex.getBaseSize(deviceId);
        std::vector<float> base(size * dim);
        ascendIndex.getBase(deviceId, (char *)base.data());
        xb.insert(xb.end(), base.begin(), base.end());

        std::vector<faiss::idx_t> idMap(size);
        ascendIndex.getIdxMap(deviceId, idMap);
        idxMap.insert(idxMap.end(), idMap.begin(), idMap.end());
    }
    EXPECT_EQ(idxMapPre.size(), idxMap.size() + rmedCntRange + rmedCntBatch);
    EXPECT_EQ(xbPre.size(), xb.size() + (rmedCntRange + rmedCntBatch) * dim);
    {
        int ntotal0 = (ntotal + conf.deviceList.size() - 1) / conf.deviceList.size();
        // check idx
        for (size_t i = 0; i < ntotal0 - (rmedCntRange + rmedCntBatch); i++) {
            if (delBatch.set.find(idxMapPre[i]) != delBatch.set.end()) {
            } else if ((idxMapPre[i] >= delRangeMin) && (idxMapPre[i] < delRangeMax)) {
                int fptr = (ntotal0 - (rmedCntRange) + i) * dim;
                int bptr = i * dim;
                // check vector
                for (int j = 0; j < dim; j++) {
                    ASSERT_TRUE(fabs(xbPre[fptr + j] - xb[bptr + j]) <= epson);
                }
            } else {
                int ptr = i * dim;
                // check idx
                EXPECT_EQ(idxMapPre[i], idxMap[i]);
                // check vector
                for (int j = 0; j < dim; j++) {
                    ASSERT_TRUE(fabs(xbPre[ptr + j] - xb[ptr + j]) <= epson);
                }
            }
        }
    }
    delete cpuIndex;
}

TEST(TestAscendIndexFlat, CloneCPU2Ascend)
{
    int dim = 512;
    int ntotal = 250000;

    srand48(1000);
    std::vector<float> data(dim * ntotal);
    for (int i = 0; i < dim * ntotal; i++) {
        data[i] = drand48();
    }
    for (int i = 0; i < 20; i++) {
        printf("%.4f\t", data[i]);
    }
    const char *indexfilename = "./flatIndex_2500000.faiss";
    faiss::Index *initIndex = faiss::read_index(indexfilename);
    ASSERT_FALSE(initIndex == nullptr);

    // ascend index
    std::vector<int> devices = { 0 };
    faiss::ascend::AscendIndexFlat *ascendIndex =
        dynamic_cast<faiss::ascend::AscendIndexFlat *>(faiss::ascend::index_cpu_to_ascend(devices, initIndex));
    ASSERT_FALSE(ascendIndex == nullptr);

    int lenall = 0;
    for (auto deviceId : devices) {
        lenall += ascendIndex->getBaseSize(deviceId);
    }

    EXPECT_EQ(lenall, ntotal);

    int k = 1000;
    int idx = 5;
    std::vector<float> dist(k, 0);
    std::vector<faiss::idx_t> label(k, 0);
    clock_t start, end;
    start = clock();
    ascendIndex->search(1, data.data() + idx * dim, k, dist.data(), label.data());
    end = clock();
    int endtime = (end - start) / CLOCKS_PER_SEC;
    std::cout << "searchTime:" << endtime << "s\n";

    EXPECT_EQ(label[0], idx);

    delete ascendIndex;
    delete initIndex;
}

namespace {
unsigned int g_seed;
const int FAST_RAND_MAX = 0x7FFF;
}

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
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
