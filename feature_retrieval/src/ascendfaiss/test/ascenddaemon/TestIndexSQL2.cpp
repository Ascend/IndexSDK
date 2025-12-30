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


#include <arm_fp16.h>
#include <gtest/gtest.h>
#include <random>
#include <ctime>
#include <cmath>

#include <ascenddaemon/impl/AuxIndexStructures.h>
#include <ascenddaemon/utils/DeviceVector.h>
#include <ascenddaemon/impl/IndexSQL2.h>
#include <ascenddaemon/utils/AscendTensor.h>

namespace {
static double Elapsed()
{
    struct timeval tv {
        0, 0
    };
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

inline void GenerateCodes(uint8_t *codes, int total, int dim)
{
    std::default_random_engine e(time(nullptr));
    std::uniform_real_distribution<float> rCode(0.0f, 1.0f);
    for (int i = 0; i < total; i++) {
        for (int j = 0; j < dim; j++) {
            // uint8's max value is 255
            codes[i * dim + j] = static_cast<unsigned char>(255 * rCode(e));
        }
    }
}

inline void GeneratePreCompute(float *preCompute, uint8_t *codes, std::vector<float16_t> &vdiff,
    std::vector<float16_t> &vmin, int total, int dim)
{
    for (int i = 0; i < total; i++) {
        int offset = i * dim;
        preCompute[i] = 0.0f;
        for (int j = 0; j < dim; j++) {
            float val = (codes[offset + j] + 0.5f) * (float)vdiff[j] / 255.0f + (float)vmin[j];
            preCompute[i] += val * val;
        }
    }
}

inline void RebuilVecs(int n, int dim, float16_t *rvecs, uint8_t *codes, std::vector<float16_t> &vdiff,
    std::vector<float16_t> &vmin)
{
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < dim; ++j) {
            rvecs[i * dim + j] = static_cast<float16_t>((codes[i * dim + j] + 0.5f) * vdiff[j] / 255.0f + vmin[j]);
        }
    }
}

inline void GenerateVdiffAndVmin(int dim, std::vector<float16_t> &vdiff, std::vector<float16_t> &vmin)
{
    std::default_random_engine e(time(nullptr));
    std::uniform_real_distribution<float> rdiff(1.0f, 5.0f);
    std::uniform_real_distribution<float> rmin(1.0f, 2.0f);

    for (int i = 0; i < dim; i++) {
        vdiff[i] = static_cast<float16_t>(rdiff(e));
        vmin[i] = static_cast<float16_t>(rmin(e));
    }
}

inline int GetCodesSize(const std::vector<std::unique_ptr<ascend::DeviceVector<uint8_t>>> &codes)
{
    int ret = 0;
    if (!codes.empty()) {
        for (auto &elem : codes) {
            ret += elem->size();
        }
    }
    return ret;
}

inline int GetPreComputeSize(const std::vector<std::unique_ptr<ascend::DeviceVector<float>>> &preCompute)
{
    int ret = 0;
    if (!preCompute.empty()) {
        for (auto &elem : preCompute) {
            ret += elem->size();
        }
    }
    return ret;
}

TEST(IndexSQL2, Add)
{
    unsigned int seed = time(nullptr);
    srand(seed);
    // create Index
    const int dim = 128;
    ascend::IndexSQL2 indexSQ(dim, false);
    const int distComputeBatchSize = indexSQ.getBlockSize();
    const auto &codes = indexSQ.getCodes();
    const auto &preCompute = indexSQ.getPreCompute();

    // add 1, will resize
    int total = 0;
    int addNum0 = 1;

    // train
    std::vector<float16_t> vdiff(dim);
    std::vector<float16_t> vmin(dim);
    GenerateVdiffAndVmin(dim, vdiff, vmin);
    ascend::AscendTensor<float16_t, ascend::DIMS_1> trainedMin(vmin.data(), { dim });
    ascend::AscendTensor<float16_t, ascend::DIMS_1> trainedDiff(vdiff.data(), { dim });
    indexSQ.updateTrainedValue(trainedMin, trainedDiff);

    std::shared_ptr<uint8_t[]> code0(new uint8_t[addNum0 * dim]);
    std::shared_ptr<float[]> preCompute0(new float[addNum0]);
    uint8_t *pCode0 = code0.get();
    float *pPreCompute0 = preCompute0.get();
    GenerateCodes(pCode0, addNum0, dim);
    GeneratePreCompute(pPreCompute0, pCode0, vdiff, vmin, addNum0, dim);

    indexSQ.addVectors(addNum0, pCode0, pPreCompute0);
    total += addNum0;

    EXPECT_EQ(indexSQ.getSize(), total);
    EXPECT_EQ(GetCodesSize(codes), ascend::utils::divUp(total, distComputeBatchSize) * distComputeBatchSize * dim);
    EXPECT_EQ(GetPreComputeSize(preCompute), ascend::utils::divUp(total, distComputeBatchSize) * distComputeBatchSize);

    // add distComputeBatchSize, will resize
    int addNum1 = distComputeBatchSize;
    std::shared_ptr<uint8_t[]> code1(new uint8_t[addNum1 * dim]);
    std::shared_ptr<float[]> preCompute1(new float[addNum1]);
    uint8_t *pCode1 = code1.get();
    float *pPreCompute1 = preCompute1.get();
    GenerateCodes(pCode1, addNum1, dim);
    GeneratePreCompute(pPreCompute1, pCode1, vdiff, vmin, addNum1, dim);

    indexSQ.addVectors(addNum1, pCode1, pPreCompute1);
    total += addNum1;

    EXPECT_EQ(indexSQ.getSize(), total);
    EXPECT_EQ(GetCodesSize(codes), ascend::utils::divUp(total, distComputeBatchSize) * distComputeBatchSize * dim);
    EXPECT_EQ(GetPreComputeSize(preCompute), ascend::utils::divUp(total, distComputeBatchSize) * distComputeBatchSize);

    // add distComputeBatchSize - 1, will not resize
    int addNum2 = distComputeBatchSize - addNum0;
    std::shared_ptr<uint8_t[]> code2(new uint8_t[addNum2 * dim]);
    std::shared_ptr<float[]> preCompute2(new float[addNum2]);
    uint8_t *pCode2 = code2.get();
    float *pPreCompute2 = preCompute2.get();
    GenerateCodes(pCode2, addNum2, dim);
    GeneratePreCompute(pPreCompute2, pCode2, vdiff, vmin, addNum2, dim);

    indexSQ.addVectors(addNum2, pCode2, pPreCompute2);
    total += addNum2;

    EXPECT_EQ(indexSQ.getSize(), total);
    EXPECT_EQ(GetCodesSize(codes), ascend::utils::divUp(total, distComputeBatchSize) * distComputeBatchSize * dim);
    EXPECT_EQ(GetPreComputeSize(preCompute), ascend::utils::divUp(total, distComputeBatchSize) * distComputeBatchSize);

    // reset
    indexSQ.reset();
    EXPECT_EQ(indexSQ.getSize(), 0);
    EXPECT_EQ(GetCodesSize(codes), 0);
    EXPECT_EQ(GetPreComputeSize(preCompute), 0);
}

TEST(IndexSQL2, AddBatch)
{
    unsigned int seed = time(nullptr);
    srand(seed);
    // create Index
    const int dim = 128;
    ascend::IndexSQL2 indexSQ(dim, false);
    const int distComputeBatchSize = indexSQ.getBlockSize();
    const auto &codes = indexSQ.getCodes();
    const auto &preCompute = indexSQ.getPreCompute();

    const int total = 1500000;
    const int batch = distComputeBatchSize - 1;

    // train
    std::vector<float16_t> vdiff(dim);
    std::vector<float16_t> vmin(dim);
    GenerateVdiffAndVmin(dim, vdiff, vmin);
    ascend::AscendTensor<float16_t, ascend::DIMS_1> trainedMin(vmin.data(), { dim });
    ascend::AscendTensor<float16_t, ascend::DIMS_1> trainedDiff(vdiff.data(), { dim });
    indexSQ.updateTrainedValue(trainedMin, trainedDiff);

    for (int i = 0; i < ascend::utils::divUp(total, batch); i++) {
        int addNum = std::min(batch, total - i * batch);

        std::shared_ptr<uint8_t[]> code0(new uint8_t[addNum * dim]);
        std::shared_ptr<float[]> preCompute0(new float[addNum]);
        uint8_t *pCode0 = code0.get();
        float *pPreCompute0 = preCompute0.get();
        GenerateCodes(pCode0, addNum, dim);
        GeneratePreCompute(pPreCompute0, pCode0, vdiff, vmin, addNum, dim);

        indexSQ.addVectors(addNum, pCode0, pPreCompute0);
        int expectSize = i * batch + addNum;

        ASSERT_EQ(indexSQ.getSize(), expectSize);
        EXPECT_EQ(GetCodesSize(codes),
            ascend::utils::divUp(expectSize, distComputeBatchSize) * distComputeBatchSize * dim);
        EXPECT_EQ(GetPreComputeSize(preCompute),
            ascend::utils::divUp(expectSize, distComputeBatchSize) * distComputeBatchSize);
    }

    // reset
    indexSQ.reset();
}

TEST(IndexSQL2, Get)
{
    unsigned int seed = time(nullptr);
    srand(seed);
    const int dim = 128;
    ascend::IndexSQL2 indexSQ(dim, false);
    const int distComputeBatchSize = indexSQ.getBlockSize();

    // train
    std::vector<float16_t> vdiff(dim);
    std::vector<float16_t> vmin(dim);
    GenerateVdiffAndVmin(dim, vdiff, vmin);
    ascend::AscendTensor<float16_t, ascend::DIMS_1> trainedMin(vmin.data(), { dim });
    ascend::AscendTensor<float16_t, ascend::DIMS_1> trainedDiff(vdiff.data(), { dim });
    indexSQ.updateTrainedValue(trainedMin, trainedDiff);

    // get empty base
    std::vector<uint8_t> result;
    indexSQ.getVectors(0, distComputeBatchSize, result);
    ASSERT_EQ(result.size(), 0);

    int remain = 1;
    int addNum = distComputeBatchSize + remain;

    std::shared_ptr<uint8_t[]> code0(new uint8_t[addNum * dim]);
    std::shared_ptr<float[]> preCompute0(new float[addNum]);
    uint8_t *pCode0 = code0.get();
    float *pPreCompute0 = preCompute0.get();
    GenerateCodes(pCode0, addNum, dim);
    GeneratePreCompute(pPreCompute0, pCode0, vdiff, vmin, addNum, dim);
    indexSQ.addVectors(addNum, code0.get(), pPreCompute0);

    // get all
    result.clear();
    indexSQ.getVectors(0, addNum, result);
    ASSERT_EQ(result.size(), addNum * dim);
    for (int i = 0; i < addNum * dim; i++) {
        if (result[i] != *(pCode0 + i)) {
            std::cout << "not equal";
        }
        ASSERT_FLOAT_EQ(result[i], *(pCode0 + i));
    }

    // get more than add num
    result.clear();
    indexSQ.getVectors(0, addNum + 1, result);
    ASSERT_EQ(result.size(), addNum * dim);
    for (int i = 0; i < addNum * dim; i++) {
        ASSERT_FLOAT_EQ(result[i], *(pCode0 + i));
    }

    // if offset >= total
    result.clear();
    indexSQ.getVectors(addNum, distComputeBatchSize, result);
    ASSERT_EQ(result.size(), 0);

    // if offset + num > total
    result.clear();
    indexSQ.getVectors(distComputeBatchSize, distComputeBatchSize, result);
    ASSERT_EQ(result.size(), remain * dim);
    for (int i = 0; i < remain * dim; i++) {
        ASSERT_FLOAT_EQ(result[i], *(pCode0 + distComputeBatchSize * dim + i));
    }
}

TEST(IndexSQL2, GetLimit)
{
    unsigned int seed = time(nullptr);
    srand(seed);
    const int dim = 128;
    ascend::IndexSQL2 indexSQ(dim, false);

    // train
    std::vector<float16_t> vdiff(dim);
    std::vector<float16_t> vmin(dim);
    GenerateVdiffAndVmin(dim, vdiff, vmin);
    ascend::AscendTensor<float16_t, ascend::DIMS_1> trainedMin(vmin.data(), { dim });
    ascend::AscendTensor<float16_t, ascend::DIMS_1> trainedDiff(vdiff.data(), { dim });
    indexSQ.updateTrainedValue(trainedMin, trainedDiff);

    const int addNum = 2000000;
    std::shared_ptr<uint8_t[]> code0(new uint8_t[addNum * dim]);
    std::shared_ptr<float[]> preCompute0(new float[addNum]);
    uint8_t *pCode0 = code0.get();
    float *pPreCompute0 = preCompute0.get();
    GenerateCodes(pCode0, addNum, dim);
    GeneratePreCompute(pPreCompute0, pCode0, vdiff, vmin, addNum, dim);
    indexSQ.addVectors(addNum, pCode0, pPreCompute0);

    // get all
    const int getPageSize = 16384;
    std::vector<uint8_t> result;
    int offset = 0;
    for (int i = 0; i < ascend::utils::divUp(addNum, getPageSize); i++) {
        int num = std::min(getPageSize, addNum - i * getPageSize);
        indexSQ.getVectors(offset, getPageSize, result);
        ASSERT_EQ(result.size(), num * dim);
        for (int j = 0; j < num * dim; j++) {
            ASSERT_FLOAT_EQ(result[i], *(pCode0 + offset * dim + i)) << "(" << i << "," << j << ")";
        }

        result.clear();
        offset += num;
    }
}

TEST(IndexSQL2, SearchBasic)
{
    unsigned int seed = time(nullptr);
    srand(seed);
    // Create Index
    const int dim = 128;
    ascend::IndexSQL2 indexSQ(dim, false);
    const int distComputeBatchSize = indexSQ.getBlockSize();

    // train
    std::vector<float16_t> vdiff(dim);
    std::vector<float16_t> vmin(dim);
    GenerateVdiffAndVmin(dim, vdiff, vmin);
    ascend::AscendTensor<float16_t, ascend::DIMS_1> trainedMin(vmin.data(), { dim });
    ascend::AscendTensor<float16_t, ascend::DIMS_1> trainedDiff(vdiff.data(), { dim });
    indexSQ.updateTrainedValue(trainedMin, trainedDiff);

    // Add
    int addNum0 = distComputeBatchSize + 1;
    std::shared_ptr<uint8_t[]> code0(new uint8_t[addNum0 * dim]);
    std::shared_ptr<float[]> preCompute0(new float[addNum0]);
    uint8_t *pCode0 = code0.get();
    float *pPreCompute0 = preCompute0.get();
    GenerateCodes(pCode0, addNum0, dim);
    GeneratePreCompute(pPreCompute0, pCode0, vdiff, vmin, addNum0, dim);
    indexSQ.addVectors(addNum0, pCode0, pPreCompute0);

    // Search
    const int nqMax = 1000;
    const int k = 1000;
    std::vector<int> queryNum { 1, 10, 31, 100, 1000 };
    std::shared_ptr<float16_t[]> queries(new float16_t[nqMax * dim]);
    float16_t *pQueries = queries.get();
    RebuilVecs(nqMax, dim, pQueries, pCode0, vdiff, vmin);

    for (auto nq : queryNum) {
        std::vector<float16_t> distances(nq * k);
        std::vector<ascend::Index::idx_t> labels(nq * k);

        try {
            indexSQ.search(nq, pQueries, k, distances.data(), labels.data());
        } catch (std::exception e) {
            std::cout << e.what() << std::endl;
        }

        for (int i = 0; i < nq; i++) {
            if (labels[i * k] != ascend::Index::idx_t(i)) {
                std::cout << "not equal";
            }
            EXPECT_EQ(labels[i * k], i);
        }
    }
}

TEST(IndexSQL2, SearchLimit)
{
    unsigned int seed = time(nullptr);
    srand(seed);
    // Create Index
    const int dim = 128;
    ascend::IndexSQL2 indexSQ(dim, false);

    // train
    std::vector<float16_t> vdiff(dim);
    std::vector<float16_t> vmin(dim);
    GenerateVdiffAndVmin(dim, vdiff, vmin);
    ascend::AscendTensor<float16_t, ascend::DIMS_1> trainedMin(vmin.data(), { dim });
    ascend::AscendTensor<float16_t, ascend::DIMS_1> trainedDiff(vdiff.data(), { dim });
    indexSQ.updateTrainedValue(trainedMin, trainedDiff);

    // Add
    const int maxBaseSize = 2000000;
    std::shared_ptr<uint8_t[]> code0(new uint8_t[maxBaseSize * dim]);
    std::shared_ptr<float[]> preCompute0(new float[maxBaseSize]);
    uint8_t *pCode0 = code0.get();
    float *pPreCompute0 = preCompute0.get();
    GenerateCodes(pCode0, maxBaseSize, dim);
    GeneratePreCompute(pPreCompute0, pCode0, vdiff, vmin, maxBaseSize, dim);

    const int nq = 1000;
    const int k = 100;
    const std::vector<int> baseSizes { 10000, 50000, 100000, 500000, 1000000, 1500000, maxBaseSize };
    for (auto baseSize : baseSizes) {
        std::cout << "Test base size " << baseSize << "\n";
        indexSQ.addVectors(baseSize, pCode0, pPreCompute0);

        const int offset = baseSize / 3;
        uint8_t *queriesCodes = pCode0 + offset * dim;
        std::shared_ptr<float16_t[]> queries(new float16_t[nq * dim]);
        float16_t *pQueries = queries.get();
        RebuilVecs(nq, dim, pQueries, queriesCodes, vdiff, vmin);

        std::vector<float16_t> distances(nq * k);
        std::vector<ascend::Index::idx_t> labels(nq * k);
        indexSQ.search(nq, pQueries, k, distances.data(), labels.data());

        for (int i = 0; i < nq; i++) {
            if (labels[i * k] != ascend::Index::idx_t(offset + i)) {
                std::cout << "not equal";
            }
            ASSERT_EQ(labels[i * k], offset + i);
        }
        indexSQ.reset();
    }
}

TEST(IndexSQL2, RemoveRangeBasic)
{
    unsigned int seed = time(nullptr);
    srand(seed);
    // Create Index
    const int dim = 128;
    ascend::IndexSQL2 indexSQ(dim, false);
    const int distComputeBatchSize = indexSQ.getBlockSize();
    const auto &codes = indexSQ.getCodes();
    const auto &preCompute = indexSQ.getPreCompute();
    int total;

    // train
    std::vector<float16_t> vdiff(dim);
    std::vector<float16_t> vmin(dim);
    GenerateVdiffAndVmin(dim, vdiff, vmin);
    ascend::AscendTensor<float16_t, ascend::DIMS_1> trainedMin(vmin.data(), { dim });
    ascend::AscendTensor<float16_t, ascend::DIMS_1> trainedDiff(vdiff.data(), { dim });
    indexSQ.updateTrainedValue(trainedMin, trainedDiff);

    // add
    int addNum0 = distComputeBatchSize + 2;
    std::shared_ptr<uint8_t[]> code0(new uint8_t[addNum0 * dim]);
    std::shared_ptr<float[]> preCompute0(new float[addNum0]);
    uint8_t *pCode0 = code0.get();
    float *pPreCompute0 = preCompute0.get();
    GenerateCodes(pCode0, addNum0, dim);
    GeneratePreCompute(pPreCompute0, pCode0, vdiff, vmin, addNum0, dim);

    total = addNum0;
    indexSQ.addVectors(addNum0, pCode0, pPreCompute0);
    ASSERT_EQ(indexSQ.getSize(), total);

    // remove first
    ascend::IDSelectorRange removeRange(0, 1);
    indexSQ.removeIds(removeRange);
    total -= 1;
    ASSERT_EQ(indexSQ.getSize(), total);
    ASSERT_EQ(GetPreComputeSize(preCompute), ascend::utils::divUp(total, distComputeBatchSize) * distComputeBatchSize);
    ASSERT_EQ(GetCodesSize(codes), ascend::utils::divUp(total, distComputeBatchSize) * distComputeBatchSize * dim);

    // [1, 10) -> [0, 9)
    int nq = 10;
    int k = 100;
    std::shared_ptr<float16_t[]> queries(new float16_t[nq * dim]);
    float16_t *pQueries = queries.get();
    RebuilVecs(nq, dim, pQueries, pCode0, vdiff, vmin);

    std::vector<float16_t> distances(nq * k);
    std::vector<ascend::Index::idx_t> labels(nq * k);
    indexSQ.search(nq, pQueries, k, distances.data(), labels.data());
    for (int i = 1; i < nq; i++) {
        // move forward by 1
        EXPECT_EQ(labels[i * k], i);
    }

    // remove last
    removeRange = ascend::IDSelectorRange(total - 1, total);
    indexSQ.removeIds(removeRange);
    total -= 1;
    ASSERT_EQ(indexSQ.getSize(), total);
    ASSERT_EQ(GetPreComputeSize(preCompute), ascend::utils::divUp(total, distComputeBatchSize) * distComputeBatchSize);
    ASSERT_EQ(GetCodesSize(codes), ascend::utils::divUp(total, distComputeBatchSize) * distComputeBatchSize * dim);

    uint8_t *queriesCode = pCode0 + (total - nq) * dim;
    RebuilVecs(nq, dim, pQueries, queriesCode, vdiff, vmin);
    indexSQ.search(nq, pQueries, k, distances.data(), labels.data());
    for (int i = 0; i < nq - 1; i++) {
        if (labels[i * k] != ascend::Index::idx_t(total - nq + i)) {
            std::cout << "error" << std::endl;
        }
        EXPECT_EQ(labels[i * k], total - nq + i);
    }

    // remove [5, 10), which is original [6, 11)
    const int imin = 5;
    const int imax = 10;
    removeRange = ascend::IDSelectorRange(imin, imax);
    indexSQ.removeIds(removeRange);
    total -= (imax - imin);
    ASSERT_EQ(indexSQ.getSize(), total);
    ASSERT_EQ(GetPreComputeSize(preCompute), ascend::utils::divUp(total, distComputeBatchSize) * distComputeBatchSize);
    ASSERT_EQ(GetCodesSize(codes), ascend::utils::divUp(total, distComputeBatchSize) * distComputeBatchSize * dim);

    nq = 20;
    distances.resize(nq * k);
    labels.resize(nq * k);
    std::shared_ptr<float16_t[]> queries0(new float16_t[nq * dim]);
    float16_t *pQueries0 = queries0.get();
    RebuilVecs(nq, dim, pQueries0, pCode0, vdiff, vmin);
    indexSQ.search(nq, pQueries0, k, distances.data(), labels.data());
    for (int i = 1; i < nq; i++) {
        if (i < 5) {
            EXPECT_EQ(labels[i * k], i);
        } else if (i >= 10) {
            EXPECT_EQ(labels[i * k], i);
        }
    }
}

TEST(IndexSQL2, QPSComparisonDim256)
{
    unsigned int seed = time(nullptr);
    srand(seed);
    double t0 = Elapsed();
    const int dim = 512;
    ascend::IndexSQL2 indexSQ(dim, false);

    // train
    std::vector<float16_t> vdiff(dim);
    std::vector<float16_t> vmin(dim);
    GenerateVdiffAndVmin(dim, vdiff, vmin);
    ascend::AscendTensor<float16_t, ascend::DIMS_1> trainedMin(vmin.data(), { dim });
    ascend::AscendTensor<float16_t, ascend::DIMS_1> trainedDiff(vdiff.data(), { dim });
    indexSQ.updateTrainedValue(trainedMin, trainedDiff);

    const int unit = 10000;
    const std::vector<int> baseSizes { 25 * unit };
    const std::vector<int> ks { 100, 1000 };

    int addNum0 = 25 * unit;
    std::shared_ptr<uint8_t[]> code0(new uint8_t[addNum0 * dim]);
    std::shared_ptr<float[]> preCompute0(new float[addNum0]);
    uint8_t *pCode0 = code0.get();
    float *pPreCompute0 = preCompute0.get();
    GenerateCodes(pCode0, addNum0, dim);
    GeneratePreCompute(pPreCompute0, pCode0, vdiff, vmin, addNum0, dim);

    for (auto size : baseSizes) {
        for (int i = 0; i < 1; ++i) {
            printf("[%.6f s] Add %d vectors\n", Elapsed() - t0, size);
            indexSQ.addVectors(addNum0, pCode0, pPreCompute0);
            printf("[%.6f s] done\n", Elapsed() - t0);
        }

        std::vector<int> queryNumVec { 1, 2, 4, 8, 16, 32, 48, 96, 100, 1000 };
        for (auto queryNum : queryNumVec) {
            float16_t *queries = new float16_t[queryNum * dim];
            RebuilVecs(queryNum, dim, queries, pCode0, vdiff, vmin);

            printf("[%.6f s] --- query batch size = %d ---\n", Elapsed() - t0, queryNum);
            for (auto k : ks) {
                std::vector<float16_t> dist(queryNum * k);
                std::vector<ascend::Index::idx_t> ids(queryNum * k);
                double startTime = Elapsed() - t0;
                printf("[%.6f s] queryNum = %d, k = %d\n", startTime, queryNum, k);
                indexSQ.search(queryNum, queries, k, dist.data(), ids.data());
                double endTime = Elapsed() - t0;
                printf("[%.6f s] done, QPS = %.1f\n", endTime, queryNum * 1.0 / (endTime - startTime));
            }
        }

        printf("[%.6f s] reset index\n", Elapsed() - t0);
        indexSQ.reset();
        printf("[%.6f s] done\n\n", Elapsed() - t0);
    }
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    ascend::DeviceScope deviceScope;
    return RUN_ALL_TESTS();
}