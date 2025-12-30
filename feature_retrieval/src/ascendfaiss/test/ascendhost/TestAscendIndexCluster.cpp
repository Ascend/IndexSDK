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

#include <algorithm>
#include <math.h>
#include <numeric>
#include <random>
#include <iostream>
#include <sys/time.h>
#include <vector>
#include <gtest/gtest.h>
#include "faiss/ascend/AscendIndexCluster.h"

namespace {
constexpr int DIM = 512;
constexpr int BASE_SIZE = 1000000;
constexpr int CAP = 6000000;
constexpr uint32_t CODE_NUM = 1000000;
constexpr float THRESHOLD = 0.78;
constexpr int QUERY_NUM = 64;
constexpr uint32_t CODE_START_IDX = 0;
constexpr uint32_t QUERY_START_IDX = 0;
constexpr int FAST_RAND_MAX = 0x7FFF;
unsigned int g_seed = 10;

inline double GetMillisecs()
{
    struct timeval tv = {0, 0};
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

inline int FastRand(void)
{
    constexpr int mutipliyNum = 214013;
    constexpr int addNum = 2531011;
    constexpr int rshiftNum = 16;
    g_seed = (mutipliyNum * g_seed + addNum);
    return (g_seed >> rshiftNum) & FAST_RAND_MAX;
}

static void CreateNormVector(std::vector<float> &normVec, size_t addn, int dim)
{
    std::vector<float> normBase(addn);
    std::default_random_engine e(time(nullptr));
    std::uniform_real_distribution<float> rCode(0.0f, 1.0f);
    for (size_t i = 0; i < addn * dim; i++) {
        normVec[i] = static_cast<int8_t>(255 * rCode(e) - 128);
        size_t baseIdx = i / dim;

        normBase[baseIdx] += normVec[i] * normVec[i];
        if ((i + 1) % dim == 0) {
            normBase[baseIdx] = sqrt(normBase[baseIdx]);
        }
    }
    // normalize
    for (size_t i = 0; i < addn * dim; i++) {
        normVec[i] /= normBase[i / dim];
    }
}

static void ComputeDistByCpuImpl(const float *query, const float *base, const int dim, float &distance)
{
    for (int i = 0; i < dim; i++) {
        distance += static_cast<float>(*(query + i)) * (*(base + i));
    }
}

void ComputByCpuAndCompare(std::vector<uint32_t> &queryIdxArr, std::vector<std::vector<float>> &resDistArr,
    std::vector<std::vector<uint32_t>> &resIdxArr, std::vector<float> &addVec, int dim)
{
    int queryNum = queryIdxArr.size();
    for (auto i = 0; i < queryNum; i++) {
        uint32_t queryIdx = queryIdxArr[i];
        uint32_t len = resIdxArr[i].size();
        for (uint32_t j = 0; j < len; j++) {
            float distByCpu = 0.0;
            uint32_t baseOffset = resIdxArr[i][j];
            ComputeDistByCpuImpl(addVec.data() + queryIdx * dim, addVec.data() + baseOffset * dim, dim, distByCpu);
            EXPECT_NEAR(distByCpu, resDistArr[i][j], 0.001);
        }
    }
}

void ComputByCpuAndCompareForAdd(std::vector<uint32_t> &queryIdxArr, std::vector<std::vector<float>> &resDistArr,
    std::vector<std::vector<uint32_t>> &resIdxArr, std::vector<float> &addVec, std::vector<float> &addVec2,
    uint32_t addn, int dim)
{
    int queryNum = queryIdxArr.size();
    for (auto i = 0; i < queryNum; i++) {
        uint32_t queryIdx = queryIdxArr[i];
        uint32_t len = resIdxArr[i].size();
        for (uint32_t j = 0; j < len; j++) {
            float distByCpu = 0.0;
            uint32_t baseOffset = resIdxArr[i][j];
            if (baseOffset >= addn) {
                ComputeDistByCpuImpl(addVec2.data() + (queryIdx - addn) * dim,
                    addVec2.data() + (baseOffset - addn) * dim,
                    dim,
                    distByCpu);
            } else {
                ComputeDistByCpuImpl(addVec.data() + queryIdx * dim, addVec.data() + baseOffset * dim, dim, distByCpu);
            }
            EXPECT_NEAR(distByCpu, resDistArr[i][j], 0.001);
        }
    }
}

void initVectors(int nq, std::vector<uint32_t> &queryIdxArr, std::vector<std::vector<float>> &resDistArr,
    std::vector<std::vector<uint32_t>> &resIdxArr)
{
    queryIdxArr.clear();
    resDistArr.clear();
    resIdxArr.clear();
    queryIdxArr.resize(nq);
    resDistArr.resize(nq);
    resIdxArr.resize(nq);
}

void initIndex(faiss::ascend::AscendIndexCluster &index, int dim, int capacity, int resourceSize)
{
    auto metricType = faiss::MetricType::METRIC_INNER_PRODUCT;
    std::vector<int> deviceList = {0};
    auto ret = index.Init(dim, capacity, metricType, deviceList, resourceSize);
    EXPECT_EQ(ret, 0);
}

void AddFeaturesToIndex(faiss::ascend::AscendIndexCluster &index, uint32_t addn, int dim, std::vector<float> &addVec,
    std::vector<uint32_t> &ids, int addStartidx)
{
    std::iota(ids.begin(), ids.end(), addStartidx);
    CreateNormVector(addVec, addn, dim);

    auto ret = index.AddFeatures(addn, addVec.data(), ids.data());
    EXPECT_EQ(ret, 0);
}

void initQueryArr(int nq, int addn, std::vector<uint32_t> &queryIdxArr)
{
    std::default_random_engine e(time(nullptr));
    std::uniform_real_distribution<float> rCode(0.0f, 1.0f);
    for (int i = 0; i < nq; i++) {
        queryIdxArr[i] = static_cast<uint32_t>(addn * rCode(e));
    }
}
}  // namespace

class ClusterParam {
public:
    ClusterParam() = default;

    ~ClusterParam();

    void init(int queryNum);
    uint32_t addn{BASE_SIZE};
    int dim{DIM};
    int capacity{CAP};
    int resourceSize{-1};
    int queryNum{QUERY_NUM};
    uint32_t codeStartIdx{CODE_START_IDX};
    uint32_t start{QUERY_START_IDX};
    uint32_t codeNum{CODE_NUM};
    float threashold{THRESHOLD};

    std::vector<std::vector<float>> resDistArr;
    std::vector<std::vector<uint32_t>> resIdxArr;
    std::vector<uint32_t> queryIdxArr;
};

ClusterParam::~ClusterParam() {}
void ClusterParam::init(int queryNum) 
{
    resDistArr.resize(queryNum);
    resIdxArr.resize(queryNum);
    queryIdxArr.resize(queryNum);
}

TEST(AscendIndexCluster, AddFeatures)
{
    ClusterParam param;
    param.init(QUERY_NUM);
    param.codeNum = CODE_NUM * 2;
    bool aboveFilter = true;

    faiss::ascend::AscendIndexCluster index;
    std::vector<float> addVec(param.addn * param.dim);
    std::vector<uint32_t> ids(param.addn);
    std::vector<float> addVec2(param.addn * param.dim);
    std::vector<uint32_t> ids2(param.addn);

    initIndex(index, param.dim, param.capacity, param.resourceSize);
    AddFeaturesToIndex(index, param.addn, param.dim, addVec, ids, 0);
    AddFeaturesToIndex(index, param.addn, param.dim, addVec2, ids2, param.addn);
    initQueryArr(param.queryNum, param.addn * 2, param.queryIdxArr);

    auto ret = index.ComputeDistanceByThreshold(
        param.queryIdxArr, param.codeStartIdx, param.codeNum, param.threashold, aboveFilter, param.resDistArr, param.resIdxArr);
    EXPECT_EQ(ret, 0);
    ComputByCpuAndCompareForAdd(param.queryIdxArr, param.resDistArr, param.resIdxArr, addVec, addVec2, param.addn, param.dim);

    // 查询第一次添加末尾和第二次添加头部的这些向量
    param.start = param.addn - 100;
    param.queryNum = 256;
    initVectors(param.queryNum, param.queryIdxArr, param.resDistArr, param.resIdxArr);
    std::iota(param.queryIdxArr.begin(), param.queryIdxArr.end(), param.start);
    // 最后一个底库中的向量
    param.queryIdxArr[param.queryNum - 1] = param.addn - 1;
    param.codeStartIdx = param.addn - 99;
    param.codeNum = param.addn;

    printf("param.queryNum(%d),start(%d),threashold(%.4lf)\r\n", param.queryNum, param.start, param.threashold);
    ret = index.ComputeDistanceByThreshold(
        param.queryIdxArr, param.codeStartIdx, param.codeNum, param.threashold, aboveFilter, param.resDistArr, param.resIdxArr);
    EXPECT_EQ(ret, 0);
    ComputByCpuAndCompareForAdd(param.queryIdxArr, param.resDistArr, param.resIdxArr, addVec, addVec2, param.addn, param.dim);

    index.Finalize();
    return;
}

void LowThresholdTest(faiss::ascend::AscendIndexCluster &index, ClusterParam &param, std::vector<float> &addVec)
{
    // 覆盖低threshold场景
    bool aboveFilter = true;
    initVectors(param.queryNum, param.queryIdxArr, param.resDistArr, param.resIdxArr);
    initQueryArr(param.queryNum, param.addn, param.queryIdxArr);
    param.threashold = 0.18;

    printf("param.threashold(%.4lf)\r\n", param.threashold);
    auto ret = index.ComputeDistanceByThreshold(
        param.queryIdxArr, param.codeStartIdx, param.codeNum, param.threashold, aboveFilter, param.resDistArr, param.resIdxArr);
    EXPECT_EQ(ret, 0);
    ComputByCpuAndCompare(param.queryIdxArr, param.resDistArr, param.resIdxArr, addVec, param.dim);
}

void SixteenAlignTest(faiss::ascend::AscendIndexCluster &index, ClusterParam &param, std::vector<float> &addVec)
{
    // 覆盖起始位置和查询数量都不满足16对齐场景
    bool aboveFilter = true;
    param.start = 15;
    param.queryNum = 15;
    initVectors(param.queryNum, param.queryIdxArr, param.resDistArr, param.resIdxArr);
    std::iota(param.queryIdxArr.begin(), param.queryIdxArr.end(), param.start);
    param.codeStartIdx = 15;
    param.codeNum = 15;
    param.threashold = 0.78;

    printf("param.queryNum(%d),start(%d),param.threashold(%.4lf)\r\n", param.queryNum, param.start, param.threashold);
    auto ret = index.ComputeDistanceByThreshold(
        param.queryIdxArr, param.codeStartIdx, param.codeNum, param.threashold, aboveFilter, param.resDistArr, param.resIdxArr);
    EXPECT_EQ(ret, 0);
    ComputByCpuAndCompare(param.queryIdxArr, param.resDistArr, param.resIdxArr, addVec, param.dim);
}

void QueryOneTest(faiss::ascend::AscendIndexCluster &index, ClusterParam &param, std::vector<float> &addVec)
{
    // 覆盖查询数量为1的情况
    bool aboveFilter = true;
    param.start = 17;
    param.queryNum = 1;
    initVectors(param.queryNum, param.queryIdxArr, param.resDistArr, param.resIdxArr);
    std::iota(param.queryIdxArr.begin(), param.queryIdxArr.end(), param.start);
    param.codeStartIdx = 14;
    param.codeNum = 5;
    param.threashold = 0.78;

    printf("param.queryNum(%d),start(%d),param.threashold(%.4lf)\r\n", param.queryNum, param.start, param.threashold);
    auto ret = index.ComputeDistanceByThreshold(
        param.queryIdxArr, param.codeStartIdx, param.codeNum, param.threashold, aboveFilter, param.resDistArr, param.resIdxArr);
    EXPECT_EQ(ret, 0);
    ComputByCpuAndCompare(param.queryIdxArr, param.resDistArr, param.resIdxArr, addVec, param.dim);
}

void ReturnZeroTest(faiss::ascend::AscendIndexCluster &index, ClusterParam &param, std::vector<float> &addVec)
{
    // 覆盖查询返回距离值满足阈值条数为0场景
    bool aboveFilter = true;
    param.start = 17;
    param.queryNum = 5;
    initVectors(param.queryNum, param.queryIdxArr, param.resDistArr, param.resIdxArr);
    std::iota(param.queryIdxArr.begin(), param.queryIdxArr.end(), param.start);
    param.codeStartIdx = 14;
    param.codeNum = 5;
    param.threashold = 0.78;

    printf("param.queryNum(%d),start(%d),param.threashold(%.4lf)\r\n", param.queryNum, param.start, param.threashold);
    auto ret = index.ComputeDistanceByThreshold(
        param.queryIdxArr, param.codeStartIdx, param.codeNum, param.threashold, aboveFilter, param.resDistArr, param.resIdxArr);
    EXPECT_EQ(ret, 0);
    ComputByCpuAndCompare(param.queryIdxArr, param.resDistArr, param.resIdxArr, addVec, param.dim);
}

void OverBlockSizeTest(faiss::ascend::AscendIndexCluster &index, ClusterParam &param, std::vector<float> &addVec)
{
    // 覆盖查询数量超过一个blocksize的情况
    bool aboveFilter = true;
    param.start = 0;
    param.queryNum = 256;
    initVectors(param.queryNum, param.queryIdxArr, param.resDistArr, param.resIdxArr);
    std::iota(param.queryIdxArr.begin(), param.queryIdxArr.end(), param.start);
    param.codeStartIdx = 0;
    param.codeNum = 262155;
    param.threashold = 0.78;

    printf("param.queryNum(%d),start(%d),param.threashold(%.4lf),param.codeNum(%d)\r\n", param.queryNum, param.start, param.threashold, param.codeNum);
    auto ret = index.ComputeDistanceByThreshold(
        param.queryIdxArr, param.codeStartIdx, param.codeNum, param.threashold, aboveFilter, param.resDistArr, param.resIdxArr);
    EXPECT_EQ(ret, 0);
    ComputByCpuAndCompare(param.queryIdxArr, param.resDistArr, param.resIdxArr, addVec, param.dim);
}

TEST(AscendIndexCluster, ComputeDistanceByThreshold)
{
    ClusterParam param;
    param.init(QUERY_NUM);
    bool aboveFilter = true;

    faiss::ascend::AscendIndexCluster index;

    std::vector<float> addVec(param.addn * param.dim);
    std::vector<uint32_t> ids(param.addn);
    initIndex(index, param.dim, param.capacity, param.resourceSize);
    AddFeaturesToIndex(index, param.addn, param.dim, addVec, ids, 0);
    initQueryArr(param.queryNum, param.addn, param.queryIdxArr);

    double timeStart = GetMillisecs();
    auto ret = index.ComputeDistanceByThreshold(
        param.queryIdxArr, param.codeStartIdx, param.codeNum, param.threashold, aboveFilter, param.resDistArr, param.resIdxArr);
    double timeEnd = GetMillisecs();
    EXPECT_EQ(ret, 0);
    printf("compute(%d) use tiem %lf, qps %.4lf, per query %.4lf\r\n",
        param.queryNum,
        timeEnd - timeStart,
        1000 * param.queryNum / (timeEnd - timeStart),
        (timeEnd - timeStart) / param.queryNum);

    ComputByCpuAndCompare(param.queryIdxArr, param.resDistArr, param.resIdxArr, addVec, param.dim);
    LowThresholdTest(index, param, addVec);
    SixteenAlignTest(index, param, addVec);
    QueryOneTest(index, param, addVec);
    ReturnZeroTest(index, param, addVec);
    OverBlockSizeTest(index, param, addVec);

    index.Finalize();
    return;
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}