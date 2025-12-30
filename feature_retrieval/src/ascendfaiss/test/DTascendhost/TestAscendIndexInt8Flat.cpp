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
#include <algorithm>
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
    EXPECT_EQ(index.getNTotal(), ntotal);

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

    EXPECT_EQ(indexNew.getDim(), dim);
    EXPECT_EQ(indexNew.getNTotal(), ntotal);

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

    int k = 1000;
    int idx = 5;
    std::vector<float> dist(k, 0);
    std::vector<faiss::idx_t> label(k, 0);
    double ts = GetMillisecs();
    ascendIndex.search(1, base.data() + idx * dim, k, dist.data(), label.data());
    double te = GetMillisecs();
    printf("search time: %f.\n", te - ts);
    EXPECT_EQ(label[0], idx);

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


    int k = 1000;
    int idx = 5;
    std::vector<float> dist(k, 0);
    std::vector<faiss::idx_t> label(k, 0);
    double ts = GetMillisecs();
    ascendIndex->search(1, base.data() + idx * dim, k, dist.data(), label.data());
    double te = GetMillisecs();
    printf("search time: %f.\n", te - ts);
    printf("ascend idx %d dist %f %f %f %f %f lable %lu %lu %lu %lu %lu.\n", idx, dist[0], dist[1], dist[2], dist[3],
        dist[4], label[0], label[1], label[2], label[3], label[4]);
    EXPECT_EQ(label[0], idx);


    delete ascendIndex;
    delete initIndex;
}

inline bool FilterPass(int ntotal, faiss::idx_t idx, std::vector<uint8_t> &mask, int queryIndex)
{
    int maskSize = (ntotal + 7) / 8;
    size_t idxInMask = static_cast<size_t>(idx / 8) + static_cast<size_t>(queryIndex * maskSize);
    if (idxInMask >= mask.size()) {
        return false;
    }
    uint8_t val = mask[idxInMask];
    int offset = idx % 8;
    return (val >> offset) % 2 != 0;
}

bool CompareResult(int ntotal, std::vector<float> &dist, std::vector<faiss::idx_t> &label,
    std::vector<float> &distWithMask, std::vector<faiss::idx_t> &labelWithMask,
    std::vector<uint8_t> &mask, int queryNum, int topk)
{
    std::vector<std::vector<float>> filteredDist;
    std::vector<std::vector<faiss::idx_t>> filteredLabel;
    for(int i = 0; i < queryNum; i++) {
        std::vector<float> tmpDist;
        std::vector<faiss::idx_t> tmpLabel;
        for ( int j = 0; j < topk; j++){
            if (FilterPass(ntotal, label[i * topk + j], mask, i)) {
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
    std::vector<int> searchNum = { 1, 2, 3, 4, 32, 64, 128 };
    faiss::ascend::AscendIndexInt8FlatConfig conf({ 0 });
    int dim = 384;
    int ntotal = 300000; // over one block size 262144
    int topk = 100;

    std::vector<int8_t> base(static_cast<size_t>(ntotal) * static_cast<size_t>(dim));
    GenerateCodes(base.data(), ntotal, dim);
    faiss::ascend::AscendIndexInt8Flat index(dim, metric, conf);
    printf("add data\n");
    index.add(ntotal, base.data());
    for (size_t i = 0; i < searchNum.size(); i++) {
        std::vector<float> dist(searchNum[i] * topk, 0);
        std::vector<faiss::idx_t> label(searchNum[i] * topk, 0);
        double ts = GetMillisecs();
        index.search(searchNum[i], base.data(), topk, dist.data(), label.data());

        int maskSize = (ntotal + 7) / 8;
        std::vector<uint8_t> mask(searchNum[i] * maskSize, 0);
        for(int q = 0; q < searchNum[i]; q++) {
            for(int j = 0; j < maskSize; ++j) {
                if (j % 2 == 1) {
                    mask[j + q * maskSize] = 255;
                }
            }
        }

        std::vector<float> distWithMask(searchNum[i] * topk, 0);
        std::vector<faiss::idx_t> labelWithMask(searchNum[i] * topk, 0);
        index.search_with_masks(searchNum[i], base.data(), topk, distWithMask.data(), labelWithMask.data(), mask.data());
        // search全量搜索结果中的label，根据设置的masks过滤后得到的labels，它们在masks过滤搜索结果中必然存在且距离值必然相等。
        EXPECT_TRUE(CompareResult(ntotal, dist, label, distWithMask, labelWithMask, mask, searchNum[i], topk));
        double te = GetMillisecs();
        printf("case[%zu]: type:%d, base:%d, dim:%d, topk:%d, search num:%d, duration:%.4f\n",
            i, metric, ntotal, dim, topk, searchNum[i], 1000 * searchNum[i] / (te - ts));
    }
}

TEST(TestAscendIndexInt8Flat, MaskPrecision)
{
    printf("test AscendIndexInt8Flat search_with_masks, type: METRIC_INNER_PRODUCT\n");
    testMaskPrecision(faiss::METRIC_INNER_PRODUCT);
    printf("test AscendIndexInt8Flat search_with_masks, type: METRIC_L2\n");
    testMaskPrecision(faiss::METRIC_L2);
}

} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
