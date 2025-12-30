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
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sys/time.h>
#include <vector>

#include "acl/acl.h"
#include <arm_fp16.h>
#include <gtest/gtest.h>

#include "ascenddaemon/impl_device/IndexILFlat.h"
#include "common/threadpool/AscendThreadPool.h"

namespace {
unsigned int g_seed = 10;
const int FAST_RAND_MAX = 0x7FFF;
const int DIM = 256;
const int BASE_SIZE = 1000000;
const int CAP = 1000000;
const unsigned int TABLELEN_REDUNDANCY = 48;

static double Elapsed()
{
    struct timeval tv {
        0
    };
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

// Used to seed the generator.
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

static void CreateNormVector(std::vector<float16_t> &normVec, size_t addn, int dim)
{
    std::vector<float> normBase(addn);
    for (size_t i = 0; i < addn * dim; i++) {
        normVec[i] = 1.0 * FastRand() / FAST_RAND_MAX;
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

static void CreateMappingTable(float *table, unsigned int tableLen)
{
    for (unsigned int i = 0; i < tableLen; i++) {
        // 创建映射表，score分值加2， 保证在2.0~3.0之间
        *(table + i) = i * 1.0;
    }
}

static void ComputeDistByCpu(const float16_t *query, const float16_t *base,
    const int dim, float &distance)
{
    for (int i = 0; i < dim; i++) {
        distance += static_cast<float>(*(query + i)) * (*(base + i));
    }
}

TEST(IndexILFlat, AddFeatures)
{
    srand(0);
    double t0 = Elapsed();
    // create index
    const size_t addn = BASE_SIZE;
    const int dim = DIM;
    const int unit = 12;
    const int capacity = CAP * unit;
    const int resourceSize = -1;
    auto metricType = ascend::AscendMetricType::ASCEND_METRIC_INNER_PRODUCT;

    ascend::IndexILFlat indexFlat;
    auto ret = indexFlat.Init(dim, capacity, metricType, resourceSize);
    if (ret) {
        printf("Index initialization failed, error code: %d\n", ret);
        return;
    }
    // Test add with ordered ids
    std::vector<float16_t> addVec(addn * dim);
    std::vector<ascend::idx_t> ids(addn);
    std::iota(ids.begin(), ids.end(), 0);
    CreateNormVector(addVec, addn, dim);
    ret = indexFlat.AddFeatures(addn, addVec.data(), ids.data());
    if (ret) {
        printf("Add 1 million features failed, error code: %d\n", ret);
        indexFlat.Finalize();
        return;
    }
    EXPECT_EQ(static_cast<size_t>(indexFlat.GetNTotal()), addn);

    // evaluate performance, repeat 10 times and record the mean time
    int repeatTimes = 10;
    double start = Elapsed();
    printf("[%.6f ms], Speed measurement start, addn=%ld, dim=%d\n", start - t0, addn, dim);
    for (int i = 0; i < repeatTimes; i++) {
        indexFlat.AddFeatures(addn, addVec.data(), ids.data());
    }
    double end = Elapsed();
    printf("[%.6f ms], repeated execution times: %d, average costs per add: [%.6f ms]\n",
        end - t0, repeatTimes, (end - start) / (addn * repeatTimes));

    printf("----------------Test add to %d features---------------- \n", CAP * unit);
    for (int i = 0; i < unit; i++)
    {
        std::iota(ids.begin(), ids.end(), i * addn);
        ret = indexFlat.AddFeatures(addn, addVec.data(), ids.data());
        if (ret) {
            printf("Add features failed at %d millions, error code: %d\n", i, ret);
            indexFlat.Finalize();
            return;
        }
    }
    EXPECT_EQ(static_cast<size_t>(indexFlat.GetNTotal()), addn * unit);

    // ----------------------------------------------------------
    // -              multi threading add                       -
    // ----------------------------------------------------------
    auto addFunctor = [&](ascend::IndexILFlat *index, size_t n, int i, float16_t *features, ascend::idx_t *indices) {
        aclrtSetDevice(0);
        auto ret = index->AddFeatures(n, features, indices);
        if (ret) {
            printf("add features failed, error code: %d\n", ret);
        }
    };

    int threadnumber = unit;
    int eatchNumber = addn;
    std::vector<std::future<void>> functorRet;
    AscendThreadPool pool(threadnumber);
    start = Elapsed();
    std::vector<ascend::idx_t> multiIds(addn * threadnumber);
    std::iota(multiIds.begin(), multiIds.end(), 0);
    for (int i = 0; i < threadnumber; i++) {
        functorRet.emplace_back(pool.Enqueue(
            addFunctor, &indexFlat, eatchNumber, i, addVec.data(), multiIds.data() + i * eatchNumber));
    }
    int seartchWait = 0;
    try {
        for (std::future<void> &ret : functorRet) {
            seartchWait++;
            ret.get();
        }
    } catch (std::exception &e) {
        for_each(functorRet.begin() + seartchWait, functorRet.end(), [](std::future<void> &ret) { ret.wait(); });
        printf("wait for add future failed.\n");
    }
    end = Elapsed();
    printf("[%.6f ms], thread nums: %d , average costs per add: [%.6f ms]\n",
        end - t0, threadnumber, (end - start) / (addn * threadnumber));
    EXPECT_EQ(indexFlat.GetNTotal(), eatchNumber * threadnumber);

    // 把最后一个线程add进去的数据读出来，跟本地的数据对比
    std::vector<float16_t> getVec(addn * dim);
    ret = indexFlat.GetFeatures(addn, getVec.data(), multiIds.data() + (threadnumber - 1) * addn);
    if (ret) {
        printf("Test get features failed, error code:%d\n", ret);
        indexFlat.Finalize();
        return;
    }
    for (size_t i = 0; i < addn * dim; i++) {
        ASSERT_FLOAT_EQ(*(getVec.data() + i), *(addVec.data() + i));
    }

    indexFlat.Finalize();
}

TEST(IndexILFlat, GetFeatures)
{
    srand(0);
    double t0 = Elapsed();
    // create index
    const size_t addn = BASE_SIZE;
    size_t getn = 400;
    const int dim = DIM;
    const int capacity = CAP;
    const int resourceSize = -1;
    auto metricType = ascend::AscendMetricType::ASCEND_METRIC_INNER_PRODUCT;

    ascend::IndexILFlat indexFlat;

    auto ret = indexFlat.Init(dim, capacity, metricType, resourceSize);
    if (ret) {
        printf("Index initialization failed, error code: %d\n", ret);
        return;
    }

    // add vectors
    std::vector<float16_t> addVec(addn * dim);
    std::vector<ascend::idx_t> ids(addn);
    std::iota(ids.begin(), ids.end(), 0);
    CreateNormVector(addVec, addn, dim);
    ret = indexFlat.AddFeatures(addn, addVec.data(), ids.data());
    if (ret) {
        printf("Add features failed, error code: %d\n", ret);
        indexFlat.Finalize();
        return;
    }
    // test GetFeatures
    std::vector<float16_t> getVec(getn * dim);
    ret = indexFlat.GetFeatures(getn, getVec.data(), ids.data());
    if (ret) {
        printf("Test get features failed, error code:%d\n", ret);
        indexFlat.Finalize();
        return;
    }
    for (size_t i = 0; i < getn * dim; i++) {
        ASSERT_FLOAT_EQ(*(getVec.data() + i), *(addVec.data() + i));
    }

    // evaluate performance, repeat 10 times and record the mean time
    int repeatTimes = 10;
    double start = Elapsed();
    printf("[%.6f ms], Speed measurement start, getn=%ld, baseNum=%ld, dim=%d\n", start - t0, getn, addn, dim);
    for (int i = 0; i < repeatTimes; i++) {
        indexFlat.GetFeatures(getn, getVec.data(), ids.data());
    }
    double end = Elapsed();
    printf("[%.6f ms], repeated execution times: %d, average costs per get: [%.6f ms]\n",
        end - t0, repeatTimes, (end - start) / (getn * repeatTimes));

    // ----------------------------------------------------------
    // -              multi threading get                       -
    // ----------------------------------------------------------
    auto getFunctor = [&](ascend::IndexILFlat *index, size_t n, int i, float16_t *features, ascend::idx_t *indices) {
        aclrtSetDevice(0);
        auto ret = index->GetFeatures(n, features, indices);
        if (ret) {
            printf("get features failed, error code: %d\n", ret);
        }
    };

    int threadnumber = repeatTimes;
    int eatchNumber = addn / threadnumber;
    std::vector<std::future<void>> functorRet;
    AscendThreadPool pool(threadnumber);
    std::vector<float16_t> getFeatureData(eatchNumber * DIM * threadnumber);
    start = Elapsed();
    for (int i = 0; i < threadnumber; i++) {
        functorRet.emplace_back(pool.Enqueue(getFunctor, &indexFlat, eatchNumber, i, getFeatureData.data() + i * eatchNumber * DIM, ids.data() + i * eatchNumber));
    }
    int seartchWait = 0;
    try {
        for (std::future<void> &ret : functorRet) {
            seartchWait++;
            ret.get();
        }
    } catch (std::exception &e) {
        for_each(functorRet.begin() + seartchWait, functorRet.end(), [](std::future<void> &ret) { ret.wait(); });
        printf("wait for get future failed.\n");
    }
    end = Elapsed();
    printf("[%.6f ms], thread nums: %d, average costs per get: [%.6f ms]\n",
        end - t0, threadnumber, (end - start) / (eatchNumber * threadnumber));

    for (size_t i = 0; i < addn * dim; i++) {
        ASSERT_FLOAT_EQ(*(getFeatureData.data() + i), *(addVec.data() + i));
    }

    indexFlat.Finalize();
}

TEST(IndexILFlat, RemoveFeatures)
{
    srand(0);
    double t0 = Elapsed();
    // create index
    const size_t addn = BASE_SIZE;
    const int removeN = 400;
    const int dim = DIM;
    const int capacity = CAP;
    const int resourceSize = -1;
    auto metricType = ascend::AscendMetricType::ASCEND_METRIC_INNER_PRODUCT;

    ascend::IndexILFlat indexFlat;

    auto ret = indexFlat.Init(dim, capacity, metricType, resourceSize);
    if (ret) {
        printf("Index initialization failed, error code: %d\n", ret);
        return;
    }

    // add vectors
    std::vector<float16_t> addVec(addn * dim);
    std::vector<ascend::idx_t> ids(addn);
    std::iota(ids.begin(), ids.end(), 0);
    CreateNormVector(addVec, addn, dim);
    ret = indexFlat.AddFeatures(addn, addVec.data(), ids.data());
    if (ret) {
        printf("Add features failed, error code: %d\n", ret);
        indexFlat.Finalize();
        return;
    }
    std::vector<ascend::idx_t> removeIds(removeN);
    std::iota(removeIds.begin(), removeIds.end(), addn - removeN);
    ret = indexFlat.RemoveFeatures(removeN, ids.data());
    if (ret) {
        printf("Remove features failed ,error code: %d\n", ret);
        indexFlat.Finalize();
        return;
    }
    // Test SetNToTal
    indexFlat.SetNTotal(addn - removeN);
    ASSERT_EQ(static_cast<unsigned int>(indexFlat.GetNTotal()), addn - removeN);
    std::vector<float16_t> getVec(removeN * dim);
    indexFlat.GetFeatures(removeN, getVec.data(), ids.data());
    for (size_t i = 0; i < removeN * dim; i++) {
        ASSERT_FLOAT_EQ(*(getVec.data() + i), 0);
    }

    // evaluate performance, repeat 10 times and record the mean time
    int repeatTimes = 10;
    double start = Elapsed();
    printf("[%.6f ms], Speed measurement start, removen=%d, baseNum=%ld, dim=%d\n", start - t0, removeN, addn, dim);
    for (int i = 0; i < repeatTimes; i++) {
        indexFlat.RemoveFeatures(removeN, ids.data() + i * removeN);
    }
    double end = Elapsed();
    printf("[%.6f ms], repeated execution times: %d, average costs per remove: [%.6f ms]\n",
        end - t0, repeatTimes, (end - start) / (removeN * repeatTimes));

    // ----------------------------------------------------------
    // -              multi threading remove                    -
    // ----------------------------------------------------------

    auto removeFunctor = [&](ascend::IndexILFlat *index, size_t n, ascend::idx_t *indices) {
        aclrtSetDevice(0);
        auto ret = index->RemoveFeatures(n, indices);
        if (ret) {
            printf("remove features failed, error code: %d\n", ret);
        }
    };
    int threadnumber = repeatTimes;
    int eatchNumber = removeN;
    std::vector<std::future<void>> functorRet;
    AscendThreadPool pool(threadnumber);
    start = Elapsed();
    for (int i = 0; i < threadnumber; i++) {
        functorRet.emplace_back(pool.Enqueue( removeFunctor, &indexFlat, eatchNumber, ids.data() + i * eatchNumber));
    }
    int seartchWait = 0;
    try {
        for (std::future<void> &ret : functorRet) {
            seartchWait++;
            ret.get();
        }
    } catch (std::exception &e) {
        for_each(functorRet.begin() + seartchWait, functorRet.end(), [](std::future<void> &ret) { ret.wait(); });
        printf("wait for remove future failed.\n");
    }
    end = Elapsed();
    printf("[%.6f ms], thread nums: %d, average costs per remove: [%.6f ms]\n",
        end - t0, threadnumber, (end - start) / (removeN * repeatTimes));

    int removeNum = threadnumber * eatchNumber;
    std::vector<float16_t> getVecMulti(removeNum * dim);
    ret = indexFlat.GetFeatures(removeNum, getVecMulti.data(), ids.data());
    if (ret) {
        printf("Test get features failed, error code:%d\n", ret);
        indexFlat.Finalize();
        return;
    }
    for (int i = 0; i < removeNum * dim; i++) {
        ASSERT_FLOAT_EQ(*(getVecMulti.data() + i), 0);
    }

    indexFlat.Finalize();
}

TEST(IndexILFlat, ComputeDistance)
{
    srand(0);
    double t0 = Elapsed();
    const int queryN = 256;
    const size_t addn = BASE_SIZE;
    const int dim = DIM;
    const int capacity = CAP;
    const int resourceSize = -1;
    auto metricType = ascend::AscendMetricType::ASCEND_METRIC_INNER_PRODUCT;

    // create index
    ascend::IndexILFlat indexFlat;
    auto ret = indexFlat.Init(dim, capacity, metricType, resourceSize);
    if (ret) {
        printf("Index initialization failed, error code: %d\n", ret);
        return;
    }

    // add normalized vectors
    std::vector<float16_t> addVec(addn * dim);
    std::vector<ascend::idx_t> ids(addn);
    std::iota(ids.begin(), ids.end(), 0);
    CreateNormVector(addVec, addn, dim);
    ret = indexFlat.AddFeatures(addn, addVec.data(), ids.data());
    if (ret) {
        printf("Add features failed, error code: %d\n", ret);
        indexFlat.Finalize();
        return;
    }
    int ntotal = indexFlat.GetNTotal();

    std::vector<float16_t> queries;
    queries.assign(addVec.begin(), addVec.begin() + queryN * dim);
    // 大块内存直接采用aclrtMalloc申请真实物理内存，优化接口处理性能
    void* distancePtr = nullptr;
    int ntotalPad = (ntotal + 15) / 16 * 16;
    size_t distMemSize = queryN * ntotalPad * sizeof(float);
    if (aclrtMalloc(&distancePtr, distMemSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS) {
        printf("AclrtMalloc failed for distMemSize! \n");
        indexFlat.Finalize();
        return;
    }
    float *distances = static_cast<float *>(distancePtr);
    ret = indexFlat.ComputeDistance(queryN, queries.data(), distances);
    if (ret) {
        printf("compute distance failed, error code:%d\n", ret);
        indexFlat.Finalize();
        return;
    }
    // compare the results with those computed by cpu, random choose 10% data of ntotal
    for (auto i = 0; i < queryN; i++) {
#pragma omp parallel for
        for (int j = 0; j < ntotal; j++) {
            float distByCpu = 0;
            ComputeDistByCpu(queries.data() + i * dim, addVec.data() + j * dim, dim, distByCpu);
            EXPECT_NEAR(*(distances + i * ntotalPad + j), distByCpu, 0.001);
        }
    }

    // evaluate performance, repeat 10 times and record the mean time
    int repeatTimes = 10;
    double start = Elapsed();
    printf("[%.6f ms], Speed measurement start, queryN: %d, base num=%ld, dim=%d\n",
        start - t0, queryN, addn, dim);
    for (int i = 0; i < repeatTimes; i++) {
        indexFlat.ComputeDistance(queryN, queries.data(), distances);
    }
    double end = Elapsed();
    printf("[%.6f ms], repeated execution times: %d, average costs per query: [%.6f ms]\n",
        end - t0, repeatTimes, (end - start) / (queryN * repeatTimes));

    (void) aclrtFree((static_cast<void *>(distances)));
    distances = nullptr;
    distancePtr = nullptr;
    indexFlat.Finalize();
}

TEST(IndexILFlat, ComputeDistanceWithTable)
{
    srand(0);
    double t0 = Elapsed();
    const int queryN = 256;
    const size_t addn = BASE_SIZE;
    const int dim = DIM;
    const int capacity = CAP;
    const int resourceSize = -1;
    auto metricType = ascend::AscendMetricType::ASCEND_METRIC_INNER_PRODUCT;

    ascend::IndexILFlat indexFlat;
    auto ret = indexFlat.Init(dim, capacity, metricType, resourceSize);
    if (ret) {
        printf("Index initialization failed, error code: %d\n", ret);
        return;
    }

    // add normalized vectors
    std::vector<float16_t> addVec(addn * dim);
    std::vector<ascend::idx_t> ids(addn);
    std::iota(ids.begin(), ids.end(), 0);
    CreateNormVector(addVec, addn, dim);
    ret = indexFlat.AddFeatures(addn, addVec.data(), ids.data());
    if (ret) {
        printf("Add features failed, error code: %d\n", ret);
        indexFlat.Finalize();
        return;
    }
    std::vector<float16_t> queries;
    queries.assign(addVec.begin(), addVec.begin() + queryN * dim);

    // 大块内存直接采用aclrtMalloc申请真实物理内存，优化接口处理性能
    void* distancePtr = nullptr;
    int ntotal = indexFlat.GetNTotal();
    int paddedntotal = (ntotal + 15) / 16 * 16;
    size_t distMemSize = queryN * paddedntotal * sizeof(float);
    if (aclrtMalloc(&distancePtr, distMemSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS) {
        printf("AclrtMalloc failed for distMemSize! \n");
        indexFlat.Finalize();
        return;
    }
    float *distances = static_cast<float *>(distancePtr);

    // test with valid table mapping, initialize table with length 10000,
    // the real space is 10 larger in case of overflow
    unsigned int tableLen = 10000;
    std::vector<float> table(tableLen + TABLELEN_REDUNDANCY);
    CreateMappingTable(table.data(), tableLen + TABLELEN_REDUNDANCY);
    ret = indexFlat.ComputeDistance(queryN, queries.data(), distances, tableLen, table.data());
    if (ret) {
        printf("Compute distance with valid table failed, error code:%d\n", ret);
        indexFlat.Finalize();
        return;
    }
    // compare the results with those computed by cpu, random choose 10% data of ntotal
    for (auto i = 0; i < queryN; i++) {
#pragma omp parallel for
        for (int j = 0; j < ntotal; j++) {
            float distByCpu = 0;
            ComputeDistByCpu(queries.data() + i * dim, addVec.data() + j * dim, dim, distByCpu);
            int tableIndex = static_cast<int>((distByCpu + 1) / 2 * tableLen + 0.5);
            EXPECT_NEAR(*(distances + i * paddedntotal + j), table[tableIndex], 1.0);
        }
    }

    // evaluate performance, repeat 10 times and record the mean time
    int repeatTimes = 10;
    double start = Elapsed();
    printf("[%.6f ms], Speed measurement start, queryN: %d, base num=%ld, dim=%d, tableLen: %d\n",
        start - t0, queryN, addn, dim, tableLen);
    for (int i = 0; i < repeatTimes; i++) {
        indexFlat.ComputeDistance(queryN, queries.data(), distances, tableLen, table.data());
    }
    double end = Elapsed();
    printf("[%.6f ms], repeated execution times: %d, average costs per query: [%.6f ms]\n",
        end - t0, repeatTimes, (end - start) / (queryN * repeatTimes));
    (void) aclrtFree((static_cast<void *>(distances)));
    distances = nullptr;
    distancePtr = nullptr;
    indexFlat.Finalize();
}

TEST(IndexILFlat, ComputeDistanceByThreshold)
{
    double t0 = Elapsed();
    // create index
    const int queryN = 256;
    const size_t addn = BASE_SIZE;
    const int dim = DIM;
    float threshold = 0.768;
    const int capacity = CAP;
    const int resourceSize = 1024 * 1024 * 1024;
    auto metricType = ascend::AscendMetricType::ASCEND_METRIC_INNER_PRODUCT;

    ascend::IndexILFlat indexFlat;
    auto ret = indexFlat.Init(dim, capacity, metricType, resourceSize);
    if (ret) {
        printf("Index initialization failed, error code: %d\n", ret);
        return;
    }

    // add normalized vectors
    std::vector<float16_t> addVec(addn * dim);
    std::vector<ascend::idx_t> ids(addn);
    std::iota(ids.begin(), ids.end(), 0);
    CreateNormVector(addVec, addn, dim);
    ret = indexFlat.AddFeatures(addn, addVec.data(), ids.data());
    if (ret) {
        printf("Add features failed, error code: %d\n", ret);
        indexFlat.Finalize();
        return;
    }
    int ntotal = indexFlat.GetNTotal();
    std::vector<float16_t> queries;
    queries.assign(addVec.begin(), addVec.begin() + queryN * dim);
    std::vector<int> num(queryN);
    int ntotalPad = (ntotal + 15) / 16 * 16;
    void* distancePtr = nullptr;
    void* indicePtr = nullptr;

    // 大块内存直接采用aclrtMalloc申请真实物理内存，优化接口处理性能
    size_t distMemSize = queryN * ntotalPad * sizeof(float);
    if (aclrtMalloc(&distancePtr, distMemSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS) {
        printf("AclrtMalloc failed for distMemSize! \n");
        indexFlat.Finalize();
        return;
    }
    size_t indiceMemSize = queryN * ntotalPad * sizeof(ascend::idx_t);
    if (aclrtMalloc(&indicePtr, indiceMemSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS) {
        printf("AclrtMalloc failed for distMemSize! \n");
        indexFlat.Finalize();
        return;
    }
    float *distances = static_cast<float *>(distancePtr);
    ascend::idx_t *indice = static_cast<ascend::idx_t *>(indicePtr);
    ret = indexFlat.ComputeDistanceByThreshold(queryN, queries.data(), threshold, num.data(), indice,
        distances);
    if (ret) {
        printf("ComputeDistanceByThreshold failed, error code: %d.\n", ret);
        indexFlat.Finalize();
        return;
    }

    int totalNum = 0;
    // compute distance by Cpu and compare the result
#pragma omp parallel for
    for (auto i = 0; i < queryN; i++) {
        totalNum += num[i];
        for (int j = 0; j < num[i]; j++) {
            float distByCpu = 0;
            ascend::idx_t idx = indice[i * ntotalPad + j];
            ComputeDistByCpu(queries.data() + i * dim, addVec.data() + idx * dim, dim, distByCpu);
            EXPECT_NEAR(*(distances + i * ntotalPad + j), distByCpu, 0.001);
            EXPECT_GE(*(distances + i * ntotalPad + j), threshold);
        }
    }
    printf("The ratio of filtered num is:%f\n", totalNum * 1.0 / (queryN * ntotal));

    // evaluate performance, repeat 10 times and record the mean time
    int repeatTimes = 10;
    double start = Elapsed();
    printf("[%.6f ms], Speed measurement start, queryN: %d, threshold: %.6f, base num=%ld, dim=%d\n",
        start - t0, queryN, threshold, addn, dim);
    for (int i = 0; i < repeatTimes; i++) {
        indexFlat.ComputeDistanceByThreshold(queryN, queries.data(), threshold, num.data(), indice,
            distances);
    }
    double end = Elapsed();
    printf("[%.6f ms], repeated execution times: %d, average costs per query: [%.6f ms]\n",
        end - t0, repeatTimes, (end - start) / (queryN * repeatTimes));

    aclrtFree((static_cast<void *>(distances)));
    aclrtFree((static_cast<void *>(indice)));
    distances = nullptr;
    indice = nullptr;
    indexFlat.Finalize();
}

TEST(IndexILFlat, ComputeDistanceByThresholdWithTable)
{
    double t0 = Elapsed();
    // create index
    const int queryN = 256;
    const size_t addn = BASE_SIZE;
    const int dim = DIM;
    const int capacity = CAP;
    const int resourceSize = 1024 * 1024 * 1024;
    auto metricType = ascend::AscendMetricType::ASCEND_METRIC_INNER_PRODUCT;

    ascend::IndexILFlat indexFlat;
    auto ret = indexFlat.Init(dim, capacity, metricType, resourceSize);
    if (ret) {
        printf("Index initialization failed, error code: %d\n", ret);
        return;
    }

    // add normalized vectors
    std::vector<float16_t> addVec(addn * dim);
    std::vector<ascend::idx_t> ids(addn);
    std::iota(ids.begin(), ids.end(), 0);
    CreateNormVector(addVec, addn, dim);
    ret = indexFlat.AddFeatures(addn, addVec.data(), ids.data());
    if (ret) {
        printf("Add features failed, error code: %d\n", ret);
        indexFlat.Finalize();
        return;
    }
    int ntotal = indexFlat.GetNTotal();
    std::vector<float16_t> queries;
    queries.assign(addVec.begin(), addVec.begin() + queryN * dim);
    float threshold = 8840;
    std::vector<int> num(queryN);

    // 大块内存直接采用aclrtMalloc申请真实物理内存，优化接口处理性能
    int ntotalPad = (ntotal + 15) / 16 * 16;
    void* distancePtr = nullptr;
    void* indicePtr = nullptr;
    size_t distMemSize = queryN * ntotalPad * sizeof(float);
    if (aclrtMalloc(&distancePtr, distMemSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS) {
        printf("AclrtMalloc failed for distMemSize! \n");
        indexFlat.Finalize();
        return;
    }
    size_t indiceMemSize = queryN * ntotalPad * sizeof(ascend::idx_t);
    if (aclrtMalloc(&indicePtr, indiceMemSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS) {
        printf("AclrtMalloc failed for distMemSize! \n");
        indexFlat.Finalize();
        return;
    }
    float *distances = static_cast<float *>(distancePtr);
    ascend::idx_t *indice = static_cast<ascend::idx_t *>(indicePtr);

    // test with valid table mapping, initialize table with length 10000,
    // the real space is 48 larger in case of overflow
    unsigned int tableLen = 10000;
    std::vector<float> table(tableLen + TABLELEN_REDUNDANCY);
    CreateMappingTable(table.data(), tableLen + TABLELEN_REDUNDANCY);
    ret = indexFlat.ComputeDistanceByThreshold(queryN, queries.data(), threshold, num.data(), indice,
        distances, tableLen, table.data());
    if (ret) {
        printf("ComputeDistanceByThreshold failed, error code: %d.\n", ret);
        indexFlat.Finalize();
        return;
    }

    int totalNum = 0;
    // compute distance by Cpu and compare the result
#pragma omp parallel for
    for (auto i = 0; i < queryN; i++) {
        totalNum += num[i];
        for (int j = 0; j < num[i]; j++) {
            float distByCpu = 0;
            ascend::idx_t index = indice[i * ntotalPad + j];
            ComputeDistByCpu(queries.data() + i * dim, addVec.data() + index * dim, dim, distByCpu);
            unsigned int tableIndex = static_cast<int>((distByCpu + 1) / 2 * tableLen + 0.5);
            // due to fp16 precision error, the max index difference is 1 from those computed by cpu
            EXPECT_NEAR(*(distances + i * ntotalPad + j), table[tableIndex], 1);
            EXPECT_GE(*(distances + i * ntotalPad + j), threshold);
        }
    }
    printf("The ratio of filtered num is:%f\n", totalNum * 1.0 / (queryN * ntotal));

    // evaluate performance, repeat 10 times and record the mean time
    int repeatTimes = 10;
    double start = Elapsed();
    printf("[%.6f ms], Speed measurement start, queryN: %d, threshold: %.6f, base num=%ld, dim=%d, tableLen: %d\n",
        start - t0, queryN, threshold, addn, dim, tableLen);
    for (int i = 0; i < repeatTimes; i++) {
        indexFlat.ComputeDistanceByThreshold(queryN, queries.data(), threshold, num.data(), indice,
            distances, tableLen, table.data());
    }
    double end = Elapsed();
    printf("[%.6f ms], repeated execution times: %d, average costs per query: [%.6f ms]\n",
        end - t0, repeatTimes, (end - start) / (queryN * repeatTimes));

    aclrtFree((static_cast<void *>(distances)));
    aclrtFree((static_cast<void *>(indice)));
    distances = nullptr;
    indice = nullptr;
    indexFlat.Finalize();
}

TEST(IndexILFlat, Search)
{
    srand(0);
    double t0 = Elapsed();
    // create index
    const int queryN = 256;
    const int topk = 200;
    const size_t addn = BASE_SIZE;
    const int dim = DIM;
    const int capacity = CAP;
    const int resourceSize = -1;
    auto metricType = ascend::AscendMetricType::ASCEND_METRIC_INNER_PRODUCT;

    ascend::IndexILFlat indexFlat;

    auto ret = indexFlat.Init(dim, capacity, metricType, resourceSize);
    if (ret) {
        printf("Index initialization failed, error code: %d\n", ret);
        return;
    }

    // add normalized vectors
    std::vector<float16_t> addVec(addn * dim);
    std::vector<ascend::idx_t> ids(addn);
    std::iota(ids.begin(), ids.end(), 0);
    CreateNormVector(addVec, addn, dim);
    ret = indexFlat.AddFeatures(addn, addVec.data(), ids.data());
    if (ret) {
        printf("Add features failed, error code: %d\n", ret);
        indexFlat.Finalize();
        return;
    }
    std::vector<float16_t> queries(queryN * dim);
    queries.assign(addVec.begin(), addVec.begin() + queryN * dim);
    std::vector<ascend::idx_t> idxs(queryN * topk);
    std::vector<float> distances(queryN * topk);
    ret = indexFlat.Search(queryN, queries.data(), topk, idxs.data(), distances.data());
    if (ret) {
        printf("Search failed, error code:%d\n", ret);
        indexFlat.Finalize();
        return;
    }
    for (auto i = 0; i < queryN; i++) {
        for (auto j = 0; j < topk; j++) {
            float distByCpu = 0;
            ascend::idx_t index = idxs[i * topk + j];
            ComputeDistByCpu(queries.data() + i * dim, addVec.data() + index * dim, dim, distByCpu);
            ASSERT_NEAR(*(distances.data() + i * topk + j), distByCpu, 0.001);
            ASSERT_NEAR(*(distances.data() + i * topk + j), 1, 0.4);
        }
    }

    // evaluate performance, repeat 10 times and record the mean time
    int repeatTimes = 10;
    double start = Elapsed();
    printf("[%.6f ms], Speed measurement start, queryN: %d, topk:%d, base num=%ld, dim=%d\n",
        start - t0, queryN, topk, addn, dim);
    for (int i = 0; i < repeatTimes; i++) {
        indexFlat.Search(queryN, queries.data(), topk, idxs.data(), distances.data());
    }
    double end = Elapsed();
    printf("[%.6f ms], repeated execution times: %d, average costs per query: [%.6f ms]\n",
        end - t0, repeatTimes, (end - start) / (queryN * repeatTimes));

    // multi thread search
    auto func_search = [&](ascend::IndexILFlat *index, int dim, int i, int topk, int queryNum, float16_t* query, ascend::idx_t *indices, float * dist) {
        aclrtSetDevice(0);
        ret = index->Search(queryNum, query, topk, indices, dist);
    };

    int threadnumber = repeatTimes;
    std::vector<std::future<void>> functorRet;
    AscendThreadPool pool(threadnumber);
    std::vector<ascend::idx_t> idxs_multi(queryN * topk * threadnumber);
    std::vector<float> distances_multi(queryN * topk * threadnumber);
    start = Elapsed();
    for (int i = 0; i < threadnumber; i++) {
        functorRet.emplace_back(
            pool.Enqueue(func_search, &indexFlat, dim, i, topk, queryN, queries.data(), idxs_multi.data() + queryN * topk* i, distances_multi.data() + queryN * topk* i));
    }
    int seartchWait = 0;
    try {
        for (std::future<void> &ret : functorRet) {
            seartchWait++;
            ret.get();
        }
    } catch (std::exception &e) {
        for_each(functorRet.begin() + seartchWait, functorRet.end(), [](std::future<void> &ret) { ret.wait(); });
        printf("wait for search future failed.\n");
    }

    end = Elapsed();
    printf("[%.6f ms], thread nums: %d, average costs per query: [%.6f ms]\n",
        end - t0, threadnumber, (end - start) / (queryN * threadnumber));

    for (int i = 0; i < queryN * topk * threadnumber; i++) {
        ASSERT_FLOAT_EQ(*(distances.data() + (i % (queryN * topk))), *(distances_multi.data() + i));
        ASSERT_EQ(*(idxs.data() + (i % (queryN * topk))), *(idxs_multi.data() + i));
    }

    indexFlat.Finalize();
}

TEST(IndexILFlat, SearchWithTable)
{
    srand(0);
    double t0 = Elapsed();
    // create index
    const int queryN = 256;
    const int topk = 200;
    const size_t addn = BASE_SIZE;
    const int dim = DIM;
    const int capacity = CAP;
    const int resourceSize = -1;
    auto metricType = ascend::AscendMetricType::ASCEND_METRIC_INNER_PRODUCT;

    ascend::IndexILFlat indexFlat;

    auto ret = indexFlat.Init(dim, capacity, metricType, resourceSize);
    if (ret) {
        printf("Index initialization failed, error code: %d\n", ret);
        return;
    }

    // add normalized vectors
    std::vector<float16_t> addVec(addn * dim);
    std::vector<ascend::idx_t> ids(addn);
    std::iota(ids.begin(), ids.end(), 0);
    CreateNormVector(addVec, addn, dim);
    ret = indexFlat.AddFeatures(addn, addVec.data(), ids.data());
    if (ret) {
        printf("Add features failed, error code: %d\n", ret);
        indexFlat.Finalize();
        return;
    }

    std::vector<float16_t> queries(queryN * dim);
    queries.assign(addVec.begin(), addVec.begin() + queryN * dim);
    std::vector<ascend::idx_t> idxs(queryN * topk);
    std::vector<float> distances(queryN * topk);

    // test with valid table mapping
    unsigned int tableLen = 10000;
    std::vector<float> table(tableLen + TABLELEN_REDUNDANCY);
    CreateMappingTable(table.data(), tableLen + TABLELEN_REDUNDANCY);
    ret = indexFlat.Search(queryN, queries.data(), topk, idxs.data(), distances.data(), tableLen, table.data());
    if (ret) {
        printf("Search with valid table failed ,error code: %d\n", ret);
        indexFlat.Finalize();
        return;
    }
    for (auto i = 0; i < queryN; i++) {
        for (auto j = 0; j < topk; j++) {
            float distByCpu = 0;
            ascend::idx_t index = idxs[i * topk + j];
            ComputeDistByCpu(queries.data() + i * dim, addVec.data() + index * dim, dim, distByCpu);
            int tableIndex = static_cast<int>((distByCpu + 1) / 2 * tableLen + 0.5);
            // due to fp16 precision error, the max index difference is 2 from those computed by cpu
            ASSERT_NEAR(*(distances.data() + i * topk + j), table[tableIndex], 2);
        }
    }

    // evaluate performance, repeat 10 times and record the mean time
    int repeatTimes = 10;
    double start = Elapsed();
    printf("[%.6f ms], Speed measurement start, queryN: %d, topk: %d, base num=%ld, dim=%d, tableLen: %d\n",
        start - t0, queryN, topk, addn, dim, tableLen);
    for (int i = 0; i < repeatTimes; i++) {
        indexFlat.Search(queryN, queries.data(), topk, idxs.data(), distances.data(), tableLen, table.data());
    }
    double end = Elapsed();
    printf("[%.6f ms], repeated execution times: %d, average costs per query: [%.6f ms]\n",
        end - t0, repeatTimes, (end - start) / (queryN * repeatTimes));

    // multi thread search with table
    auto func_searchTab = [&](ascend::IndexILFlat *index, int dim, int i, int topk, int queryNum, float16_t* query, ascend::idx_t *indices, float * dist, int tableLen, float* table) {
        aclrtSetDevice(0);
        ret = index->Search(queryNum, query, topk, indices, dist, tableLen, table);
    };

    int threadnumber = repeatTimes;
    std::vector<std::future<void>> functorRet;
    AscendThreadPool pool(threadnumber);
    std::vector<ascend::idx_t> idxsMulti(queryN * topk * threadnumber);
    std::vector<float> distancesMulti(queryN * topk * threadnumber);
    start = Elapsed();
    for (int i = 0; i < threadnumber; i++) {
        functorRet.emplace_back(pool.Enqueue(
                func_searchTab, &indexFlat, dim, i, topk, queryN, queries.data(), idxsMulti.data() + queryN * topk* i, distancesMulti.data() + queryN * topk* i, tableLen, table.data()));
    }
    int seartchWait = 0;
    try {
        for (std::future<void> &ret : functorRet) {
            seartchWait++;
            ret.get();
        }
    } catch (std::exception &e) {
        for_each(functorRet.begin() + seartchWait, functorRet.end(), [](std::future<void> &ret) { ret.wait(); });
        printf("wait for search future failed.\n");
    }

    end = Elapsed();
    printf("[%.6f ms], thread nums: %d, average costs per query: [%.6f ms]\n",
        end - t0, threadnumber, (end - start) / (queryN * threadnumber));

    for (int i = 0; i < queryN * topk * threadnumber; i++) {
        ASSERT_FLOAT_EQ(*(distances.data() + (i % (queryN * topk))), *(distancesMulti.data() + i));
        ASSERT_EQ(*(idxs.data() + (i % (queryN * topk))), *(idxsMulti.data() + i));
    }

    indexFlat.Finalize();
}

TEST(IndexILFlat, SearchByThreshold)
{
    srand(0);
    double t0 = Elapsed();
    // create index
    const int topk = 200;
    const int queryN = 256;
    const size_t addn = BASE_SIZE;
    const int dim = DIM;
    const int capacity = CAP;
    const int resourceSize = -1;
    auto metricType = ascend::AscendMetricType::ASCEND_METRIC_INNER_PRODUCT;

    ascend::IndexILFlat indexFlat;

    auto ret = indexFlat.Init(dim, capacity, metricType, resourceSize);
    if (ret) {
        printf("Index initialization failed, error code: %d\n", ret);
        return;
    }

    // add normalized vectors
    std::vector<float16_t> addVec(addn * dim);
    std::vector<ascend::idx_t> ids(addn);
    std::iota(ids.begin(), ids.end(), 0);
    CreateNormVector(addVec, addn, dim);
    ret = indexFlat.AddFeatures(addn, addVec.data(), ids.data());
    if (ret) {
        printf("Add features failed, error code: %d\n", ret);
        indexFlat.Finalize();
        return;
    }
    // compute distance for whole base vectors.
    std::vector<float16_t> queries(queryN * dim);
    queries.assign(addVec.begin(), addVec.begin() + queryN * dim);
    std::vector<ascend::idx_t> idxs(queryN * topk);
    std::vector<float> distances(queryN * topk);
    float threshold = 0.768;
    std::vector<int> num(queryN);
    ret = indexFlat.SearchByThreshold(
        queryN, queries.data(), threshold, topk, num.data(), idxs.data(), distances.data());
    if (ret) {
        printf("Search by threshold failed, error code: %d\n", ret);
        indexFlat.Finalize();
        return;
    }
    for (auto i = 0; i < queryN; i++) {
        for (auto j = 0; j < num[i]; j++) {
            float distByCpu = 0;
            ascend::idx_t index = idxs[i * topk + j];
            ComputeDistByCpu(queries.data() + i * dim, addVec.data() + index * dim, dim, distByCpu);
            ASSERT_NEAR(*(distances.data() + i * topk + j), distByCpu, 0.001);
        }
    }
    // evaluate performance, repeat 10 times and record the mean time
    int repeatTimes = 10;
    double start = Elapsed();
    printf("[%.6f ms], Speed measurement start, queryN: %d, base num=%ld, dim=%d\n",
        start - t0, queryN, addn, dim);
    for (int i = 0; i < repeatTimes; i++) {
        indexFlat.SearchByThreshold(queryN, queries.data(), threshold, topk, num.data(), idxs.data(), distances.data());
    }
    double end = Elapsed();
    printf("[%.6f ms], repeated execution times: %d, average costs per query: [%.6f ms]\n",
        end - t0, repeatTimes, (end - start) / (queryN * repeatTimes));

    // multi thread search with treashold
    auto func_searchTsh = [&](ascend::IndexILFlat *index, int dim, int i, int topk, float threshold, int queryNum, float16_t* query,int * num, ascend::idx_t *indices, float * dist) {
        aclrtSetDevice(0);
        ret = index->SearchByThreshold(queryNum, query, threshold, topk, num, indices, dist);
    };

    int threadnumber = repeatTimes;
    std::vector<std::future<void>> functorRet;
    AscendThreadPool pool(threadnumber);
    std::vector<ascend::idx_t> idxsMulti(queryN * topk * threadnumber);
    std::vector<float> distancesMulti(queryN * topk * threadnumber);
    std::vector<int> numMulti(queryN * threadnumber);
    start = Elapsed();
    for (int i = 0; i < threadnumber; i++) {
        functorRet.emplace_back(
            pool.Enqueue(func_searchTsh, &indexFlat, dim, i, topk, threshold, queryN, queries.data(), numMulti.data() + queryN * i, idxsMulti.data() + queryN * topk* i, distancesMulti.data() + queryN * topk* i));
    }
    int seartchWait = 0;
    try {
        for (std::future<void> &ret : functorRet) {
            seartchWait++;
            ret.get();
        }
    } catch (std::exception &e) {
        for_each(functorRet.begin() + seartchWait, functorRet.end(), [](std::future<void> &ret) { ret.wait(); });
        printf("wait for search future failed.\n");
    }

    end = Elapsed();
    printf("[%.6f ms], thread nums: %d, average costs per query: [%.6f ms]\n",
        end - t0, threadnumber, (end - start) / (queryN * threadnumber));

    for (int i = 0; i < queryN * topk * threadnumber; i++) {
        ASSERT_FLOAT_EQ(*(distances.data() + (i % (queryN * topk))), *(distancesMulti.data() + i));
        ASSERT_EQ(*(idxs.data() + (i % (queryN * topk))), *(idxsMulti.data() + i));
    }

    for (int i = 0; i < queryN  * threadnumber; i++) {
        ASSERT_EQ(*(num.data() + (i % queryN )), *(numMulti.data() + i));
    }
    indexFlat.Finalize();
}

TEST(IndexILFlat, SearchByThresholdWithTable)
{
    srand(0);
    double t0 = Elapsed();
    // create index
    const size_t addn = BASE_SIZE;
    const int topk = 200;
    const int queryN = 256;
    const int dim = DIM;
    const int capacity = CAP;
    const int resourceSize = -1;
    auto metricType = ascend::AscendMetricType::ASCEND_METRIC_INNER_PRODUCT;

    ascend::IndexILFlat indexFlat;

    auto ret = indexFlat.Init(dim, capacity, metricType, resourceSize);
    if (ret) {
        printf("Index initialization failed, error code: %d\n", ret);
        return;
    }

    // add normalized vectors
    std::vector<float16_t> addVec(addn * dim);
    std::vector<ascend::idx_t> ids(addn);
    std::iota(ids.begin(), ids.end(), 0);
    CreateNormVector(addVec, addn, dim);
    ret = indexFlat.AddFeatures(addn, addVec.data(), ids.data());
    if (ret) {
        printf("Add features failed, error code: %d\n", ret);
        indexFlat.Finalize();
        return;
    }
    std::vector<float16_t> queries(queryN * dim);
    queries.assign(addVec.begin(), addVec.begin() + queryN * dim);
    std::vector<ascend::idx_t> idxs(queryN * topk);
    std::vector<float> distances(queryN * topk);
    float threshold = 2.1;
    std::vector<int> num(queryN);

    // test with valid table mapping, initialize table with length 10000,
    // the real space is 10 larger in case of overflow
    unsigned int tableLen = 10000;
    std::vector<float> table(tableLen + TABLELEN_REDUNDANCY);
    CreateMappingTable(table.data(), tableLen + TABLELEN_REDUNDANCY);
    

    ret = indexFlat.SearchByThreshold(
        queryN, queries.data(), threshold, topk, num.data(), idxs.data(), distances.data(), tableLen, table.data());
    if (ret) {
        printf("Search by threshold with valid table failed, error code:%d\n", ret);
        indexFlat.Finalize();
        return;
    }
    for (auto i = 0; i < queryN; i++) {
        for (auto j = 0; j < num[i]; j++) {
            float distByCpu = 0;
            ascend::idx_t index = idxs[i * topk + j];
            ComputeDistByCpu(queries.data() + i * dim, addVec.data() + index * dim, dim, distByCpu);
            int tableIndex = static_cast<int>((distByCpu + 1) / 2 * tableLen + 0.5);
            ASSERT_NEAR(*(distances.data() + i * topk + j), table[tableIndex], 2);
        }
    }

    // evaluate performance, repeat 10 times and record the mean time
    int repeatTimes = 10;
    double start = Elapsed();
    printf("[%.6f ms], Speed measurement start, queryN: %d, base num=%ld, dim=%d, tableLen: %d\n",
        start - t0, queryN, addn, dim, tableLen);
    for (int i = 0; i < repeatTimes; i++) {
        indexFlat.SearchByThreshold(queryN, queries.data(), threshold, topk, num.data(), idxs.data(), distances.data(),
            tableLen, table.data());
    }
    double end = Elapsed();
    printf("[%.6f ms], repeated execution times: %d, average costs per query: [%.6f ms]\n",
        end - t0, repeatTimes, (end - start) / (queryN * repeatTimes));

    // multi thread search with threshold with table
    auto func_searchTshTab = [&](ascend::IndexILFlat *index, int dim, int i, int topk, float threshold, int queryNum, float16_t* query,int * num, ascend::idx_t *indices, float * dist, int tableLen, float* table) {
        aclrtSetDevice(0);
        ret = index->SearchByThreshold(queryNum, query, threshold, topk, num, indices, dist, tableLen, table);
    };
    int threadnumber = repeatTimes;
    std::vector<std::future<void>> functorRet;
    AscendThreadPool pool(threadnumber);
    std::vector<ascend::idx_t> idxsMulti(queryN * topk * threadnumber);
    std::vector<float> distancesMulti(queryN * topk * threadnumber);
    std::vector<int> numMulti(queryN * threadnumber);
    start = Elapsed();
    for (int i = 0; i < threadnumber; i++) {
        functorRet.emplace_back(pool.Enqueue(
                func_searchTshTab, &indexFlat, dim, i, topk, threshold, queryN, queries.data(), numMulti.data() + i * queryN, idxsMulti.data() + queryN * topk* i, distancesMulti.data() + queryN * topk* i, tableLen, table.data()));
    }
    int seartchWait = 0;
    try {
        for (std::future<void> &ret : functorRet) {
            seartchWait++;
            ret.get();
        }
    } catch (std::exception &e) {
        for_each(functorRet.begin() + seartchWait, functorRet.end(), [](std::future<void> &ret) { ret.wait(); });
        printf("wait for search future failed.\n");
    }

    end = Elapsed();
    printf("[%.6f ms], thread nums: %d, average costs per query: [%.6f ms]\n",
        end - t0, threadnumber, (end - start) / (queryN * threadnumber));

    for (int i = 0; i < queryN * topk * threadnumber; i++) {
        ASSERT_FLOAT_EQ(*(distances.data() + (i % (queryN * topk))), *(distancesMulti.data() + i));
        ASSERT_EQ(*(idxs.data() + (i % (queryN * topk))), *(idxsMulti.data() + i));
    }
    for (int i = 0; i < queryN * threadnumber; i++) {
        ASSERT_EQ(*(num.data() + (i % queryN)), *(numMulti.data() + i));
    }


    indexFlat.Finalize();
}

TEST(IndexILFlat, ComputeDistanceByIdx)
{
    srand(0);
    double t0 = Elapsed();
    // create index
    int queryN = 256;
    const int maxNum = 40;
    const size_t addn = BASE_SIZE;
    const int dim = DIM;
    const int capacity = CAP;
    const int resourceSize = -1;
    auto metricType = ascend::AscendMetricType::ASCEND_METRIC_INNER_PRODUCT;

    ascend::IndexILFlat indexFlat;

    auto ret = indexFlat.Init(dim, capacity, metricType, resourceSize);
    if (ret) {
        printf("Index initialization failed, error code: %d\n", ret);
        return;
    }

    // add normalized vectors
    std::vector<float16_t> addVec(addn * dim);
    std::vector<ascend::idx_t> ids(addn);
    std::iota(ids.begin(), ids.end(), 0);
    CreateNormVector(addVec, addn, dim);
    ret = indexFlat.AddFeatures(addn, addVec.data(), ids.data());
    if (ret) {
        printf("Add Features failed, error code: %d\n", ret);
        indexFlat.Finalize();
        return;
    }
    int ntotal = indexFlat.GetNTotal();
    EXPECT_EQ(static_cast<size_t>(ntotal), addn);

    // compute distance for whole base vectors.
    std::vector<float16_t> queries;
    queries.assign(addVec.begin(), addVec.begin() + queryN * dim);
    // 每个query指定maxNum个比对索引
    std::vector<int> num(queryN, maxNum);
    std::vector<float> distances(queryN * maxNum);
    std::vector<ascend::idx_t> indice(queryN * maxNum);
    // 指定每个query要比对的底库indice
    for (int i = 0; i < queryN; i++) {
        for (int j = 0; j < maxNum; j++) {
            indice[i * maxNum + j] = (int)(1.0 * FastRand() / FAST_RAND_MAX * (addn - 1));
        }
    }
    ret = indexFlat.ComputeDistanceByIdx(queryN, queries.data(), num.data(), indice.data(), distances.data());
    if (ret) {
        printf("Compute distance by idx failed, error code:%d\n", ret);
        indexFlat.Finalize();
        return;
    }
    // calculate the distance by cpu and compare
    for (int i = 0; i < queryN; i++) {
        for (int j = 0; j < num[i]; j++) {
            float distByCpu = 0;
            uint64_t id = indice[i * maxNum + j];
            ComputeDistByCpu(queries.data() + i * dim, addVec.data() + id * dim, dim, distByCpu);
            ASSERT_NEAR(*(distances.data() + i * maxNum + j), distByCpu, 0.0001);
        }
    }

    // evaluate performance, repeat 10 times and record the mean time
    int repeatTimes = 10;
    double start = Elapsed();
    printf("[%.6f ms], Speed measurement start, queryN: %d, base num=%ld, dim=%d, each indice num:%d\n",
        start - t0, queryN, addn, dim, maxNum);
    for (int i = 0; i < repeatTimes; i++) {
        indexFlat.ComputeDistanceByIdx(queryN, queries.data(), num.data(), indice.data(), distances.data());
    }
    double end = Elapsed();
    printf("[%.6f ms], repeated execution times: %d, average costs per query: [%.6f ms]\n",
        end - t0, repeatTimes, (end - start) / (queryN * repeatTimes));

    indexFlat.Finalize();
}

TEST(IndexILFlat, ComputeDistanceByIdxWithTable)
{
    srand(0);
    double t0 = Elapsed();
    // create index
    int queryN = 256;
    const int maxNum = 40;
    const size_t addn = BASE_SIZE;
    const int dim = DIM;
    const int capacity = CAP;
    const int resourceSize = -1;
    auto metricType = ascend::AscendMetricType::ASCEND_METRIC_INNER_PRODUCT;

    ascend::IndexILFlat indexFlat;

    auto ret = indexFlat.Init(dim, capacity, metricType, resourceSize);
    if (ret) {
        printf("Index initialization failed, error code: %d\n", ret);
        return;
    }

    // add normalized vectors
    std::vector<float16_t> addVec(addn * dim);
    std::vector<ascend::idx_t> ids(addn);
    std::iota(ids.begin(), ids.end(), 0);
    CreateNormVector(addVec, addn, dim);
    ret = indexFlat.AddFeatures(addn, addVec.data(), ids.data());
    if (ret) {
        printf("Add Features failed, error code: %d\n", ret);
        indexFlat.Finalize();
        return;
    }
    int ntotal = indexFlat.GetNTotal();
    EXPECT_EQ(static_cast<size_t>(ntotal), addn);

    // compute distance for whole base vectors.
    std::vector<float16_t> queries;
    queries.assign(addVec.begin(), addVec.begin() + queryN * dim);
    // 每个query指定maxNum个比对索引
    std::vector<int> num(queryN, maxNum);
    std::vector<float> distances(queryN * maxNum);
    std::vector<ascend::idx_t> indice(queryN * maxNum);
    // 指定每个query要比对的底库indice
    for (int i = 0; i < queryN; i++) {
        for (int j = 0; j < maxNum; j++) {
            indice[i * maxNum + j] = (int)(1.0 * FastRand() / FAST_RAND_MAX * (addn - 1));
        }
    }
    // 初始化表长10000的float型scores表
    unsigned int tableLen = 10000;
    std::vector<float> table(tableLen + TABLELEN_REDUNDANCY);
    CreateMappingTable(table.data(), tableLen + TABLELEN_REDUNDANCY);

    ret = indexFlat.ComputeDistanceByIdx(queryN, queries.data(), num.data(), indice.data(), distances.data(),
        tableLen, table.data());
    if (ret) {
        printf("Compute DistanceByIdx with valid table failed, error code: %d\n", ret);
        indexFlat.Finalize();
        return;
    }
    for (auto i = 0; i < queryN; i++) {
        for (int j = 0; j < num[i]; j++) {
            float distByCpu = 0;
            uint64_t id = indice[i * maxNum + j];
            ComputeDistByCpu(queries.data() + i * dim, addVec.data() + id * dim, dim, distByCpu);
            int tableIndex = static_cast<int>((distByCpu + 1) / 2 * tableLen + 0.5);
            EXPECT_NEAR(*(distances.data() + i * maxNum + j), table[tableIndex], 1);
        }
    }

    // evaluate performance, repeat 10 times and record the mean time
    int repeatTimes = 10;
    double start = Elapsed();
    printf("[%.6f ms], Speed measurement start, queryN: %d, base num=%ld, dim=%d, each indice num: %d, tableLen: %d\n",
        start - t0, queryN, addn, dim, maxNum, tableLen);
    for (int i = 0; i < repeatTimes; i++) {
        indexFlat.ComputeDistanceByIdx(queryN, queries.data(), num.data(), indice.data(), distances.data(),
            tableLen, table.data());
    }
    double end = Elapsed();
    printf("[%.6f ms], repeated execution times: %d, average costs per query: [%.6f ms]\n",
        end - t0, repeatTimes, (end - start) / (queryN * repeatTimes));
    indexFlat.Finalize();
}

TEST(IndexILFlat, singleIndexMultiSearch)
{
    srand(0);
    // create index
    const int topk = 200;
    const int queryN = 256;
    const size_t addn = BASE_SIZE;
    const int dim = DIM;
    const int capacity = CAP;
    const int resourceSize = -1;
    auto metricType = ascend::AscendMetricType::ASCEND_METRIC_INNER_PRODUCT;

    ascend::IndexILFlat indexFlat;

    auto ret = indexFlat.Init(dim, capacity, metricType, resourceSize);
    if (ret) {
        printf("Index initialization failed, error code: %d\n", ret);
        return;
    }

    // add normalized vectors
    std::vector<float16_t> addVec(addn * dim);
    std::vector<ascend::idx_t> ids(addn);
    std::iota(ids.begin(), ids.end(), 0);
    CreateNormVector(addVec, addn, dim);
    ret = indexFlat.AddFeatures(addn, addVec.data(), ids.data());
    if (ret) {
        printf("Add features failed, error code: %d\n", ret);
        indexFlat.Finalize();
        return;
    }
    // compute distance for whole base vectors.
    std::vector<float16_t> queries(queryN * dim);
    queries.assign(addVec.begin(), addVec.begin() + queryN * dim);

    std::vector<ascend::idx_t> idxs(queryN * topk);
    std::vector<float> distances(queryN * topk);
    std::vector<ascend::idx_t> idxsTable(queryN * topk);
    std::vector<float> distancesTable(queryN * topk);
    std::vector<ascend::idx_t> idxsTh(queryN * topk);
    std::vector<float> distancesTh(queryN * topk);
    std::vector<ascend::idx_t> idxsThTable(queryN * topk);
    std::vector<float> distancesThTable(queryN * topk);

    float threshold = 0.768;
    std::vector<int> num(queryN);

    unsigned int tableLen = 10000;
    std::vector<float> table(tableLen + TABLELEN_REDUNDANCY);
    CreateMappingTable(table.data(), tableLen + TABLELEN_REDUNDANCY);

    ret = indexFlat.Search(queryN, queries.data(), topk, idxs.data(), distances.data());
    if (ret) {
        printf("Search failed, error code:%d\n", ret);
        indexFlat.Finalize();
        return;
    }

    ret = indexFlat.Search(queryN, queries.data(), topk, idxsTable.data(), distancesTable.data(), tableLen, table.data());
    if (ret) {
        printf("Search with valid table failed ,error code: %d\n", ret);
        indexFlat.Finalize();
        return;
    }

    ret = indexFlat.SearchByThreshold(
        queryN, queries.data(), threshold, topk, num.data(), idxsTh.data(), distancesTh.data());
    if (ret) {
        printf("Search by threshold failed, error code: %d\n", ret);
        indexFlat.Finalize();
        return;
    }
        ret = indexFlat.SearchByThreshold(
        queryN, queries.data(), threshold, topk, num.data(), idxsThTable.data(), distancesThTable.data(), tableLen, table.data());
    if (ret) {
        printf("Search by threshold with valid table failed, error code:%d\n", ret);
        indexFlat.Finalize();
        return;
    }

    // search
    auto func_search = [&](ascend::IndexILFlat *index, int dim, int i, int queryNum, float16_t* query, ascend::idx_t *indices, float * dist) {
        aclrtSetDevice(0);
        // printf("id: %d Search\n", i);
        int topk = 200;
        ret = index->Search(queryNum, query, topk, indices, dist);
        // printf("id: %d, Search ret: %d\n", i, ret);
    };
    auto func_searchTab = [&](ascend::IndexILFlat *index, int dim, int i, int queryNum, float16_t* query, ascend::idx_t *indices, float * dist, int tableLen, float* table) {
        aclrtSetDevice(0);
        // printf("id: %d Search table\n", i);
        int topk = 200;
        ret = index->Search(queryNum, query, topk, indices, dist, tableLen, table);
        // printf("id: %d, Search table ret: %d\n", i, ret);
    };
    auto func_searchTsh = [&](ascend::IndexILFlat *index, int dim, int i, int queryNum, float16_t* query,int * num, ascend::idx_t *indices, float * dist) {
        aclrtSetDevice(0);
        // printf("id: %d SearchByThreshold\n", i);
        float threshold = 0.768;
        int topk = 200;
        ret = index->SearchByThreshold(queryNum, query, threshold, topk, num, indices, dist);
        // printf("id: %d, SearchByThreshold ret: %d\n", i, ret);
    };
    auto func_searchTshTab = [&](ascend::IndexILFlat *index, int dim, int i, int queryNum, float16_t* query,int * num, ascend::idx_t *indices, float * dist, int tableLen, float* table) {
        aclrtSetDevice(0);
        // printf("id: %d SearchByThreshold table\n", i);
        float threshold = 0.768;
        int topk = 200;
        ret = index->SearchByThreshold(queryNum, query, threshold, topk, num, indices, dist, tableLen, table);
        // printf("id: %d, SearchByThreshold table ret: %d\n", i, ret);
    };
    auto getFunctor = [&](ascend::IndexILFlat *index, size_t n, int i, float16_t *features, ascend::idx_t *indices) {
        aclrtSetDevice(0);
        auto ret = index->GetFeatures(n, features, indices);
        if (ret) {
            printf("get features failed, error code: %d\n", ret);
        }
    };

    int threadnumber = 10;
    int eatchNumber = queryN;
    int eatchGetNumber = addn / threadnumber;
    std::vector<std::future<void>> functorRet;
    AscendThreadPool pool(threadnumber);

    std::vector<ascend::idx_t> idxsMulti(queryN * topk * threadnumber);
    std::vector<float> distancesMulti(queryN * topk * threadnumber);
    std::vector<ascend::idx_t> idxsTableMulti(queryN * topk * threadnumber);
    std::vector<float> distancesTableMulti(queryN * topk * threadnumber);
    std::vector<ascend::idx_t> idxsThMulti(queryN * topk * threadnumber);
    std::vector<float> distancesThMulti(queryN * topk * threadnumber);
    std::vector<ascend::idx_t> idxsThTableMulti(queryN * topk * threadnumber);
    std::vector<float> distancesThTableMulti(queryN * topk * threadnumber);
    std::vector<float16_t> getFeatureData(eatchGetNumber * DIM * threadnumber);

    for (int i = 0; i < threadnumber; i++) {
        functorRet.emplace_back(pool.Enqueue(
            func_search, &indexFlat, dim, i, queryN, queries.data(), idxsMulti.data() + queryN * topk* i, distancesMulti.data() + queryN * topk* i));
        functorRet.emplace_back(pool.Enqueue(
            func_searchTab, &indexFlat, dim, i, queryN, queries.data(), idxsTableMulti.data() + queryN * topk* i, distancesTableMulti.data() + queryN * topk* i, tableLen, table.data()));
        functorRet.emplace_back(pool.Enqueue(
            func_searchTsh, &indexFlat, dim, i, queryN, queries.data(), num.data(), idxsThMulti.data() + queryN * topk* i, distancesThMulti.data() + queryN * topk* i));
        functorRet.emplace_back(pool.Enqueue(
            func_searchTshTab, &indexFlat, dim, i, queryN, queries.data(), num.data(), idxsThTableMulti.data() + queryN * topk* i, distancesThTableMulti.data() + queryN * topk* i, tableLen, table.data()));
        functorRet.emplace_back(pool.Enqueue(getFunctor, &indexFlat, eatchGetNumber, i, getFeatureData.data() + i * eatchGetNumber * DIM, ids.data() + i * eatchGetNumber));
    }
    int seartchWait = 0;
    try {
        for (std::future<void> &ret : functorRet) {
            seartchWait++;
            ret.get();
        }
    } catch (std::exception &e) {
        for_each(functorRet.begin() + seartchWait, functorRet.end(), [](std::future<void> &ret) { ret.wait(); });
        printf("wait for search threshold table future failed.\n");
    }
    for (int i = 0; i < eatchNumber * DIM * threadnumber; i++) {
        ASSERT_FLOAT_EQ(*(getFeatureData.data() + i), *(addVec.data() + i));
    }

    for (int i = 0; i < queryN * topk * threadnumber; i++) {
        ASSERT_FLOAT_EQ(*(distances.data() + (i % (queryN * topk))), *(distancesMulti.data() + i));
        ASSERT_EQ(*(idxs.data() + (i % (queryN * topk))), *(idxsMulti.data() + i));

        ASSERT_FLOAT_EQ(*(distancesTable.data() + (i % (queryN * topk))), *(distancesTableMulti.data() + i));
        ASSERT_EQ(*(idxsTable.data() + (i % (queryN * topk))), *(idxsTableMulti.data() + i));

        ASSERT_FLOAT_EQ(*(distancesTh.data() + (i % (queryN * topk))), *(distancesThMulti.data() + i));
        ASSERT_EQ(*(idxsTh.data() + (i % (queryN * topk))), *(idxsThMulti.data() + i));

        ASSERT_FLOAT_EQ(*(distancesThTable.data() + (i % (queryN * topk))), *(distancesThTableMulti.data() + i));
        ASSERT_EQ(*(idxsThTable.data() + (i % (queryN * topk))), *(idxsThTableMulti.data() + i));
    }

    indexFlat.Finalize();
}

TEST(IndexILFlat, multiIndexMultiSearch)
{
    auto func_search = [&](ascend::IndexILFlat *index, int dim, int i, int topk, int queryNum, float16_t* query, ascend::idx_t *indices, float * dist) {
        aclrtSetDevice(0);
        int ret = index->Search(queryNum, query, topk, indices, dist);
        if (ret) {
            printf("Search failed, error code: %d\n", ret);
        }
    };

    auto getFunctor = [&](ascend::IndexILFlat *index, size_t n, int i, float16_t *features, ascend::idx_t *indices) {
        aclrtSetDevice(0);
        auto ret = index->GetFeatures(n, features, indices);
        if (ret) {
            printf("GetFeatures failed, error code: %d\n", ret);
        }
    };

    srand(0);
    int ret = 0;
    int indexNum = 5;
    int threadnumber = 10;
    std::vector<std::future<void>> functorRet;
    std::vector<ascend::IndexILFlat *> indexes;
    const int topk = 200;
    const int queryN = 256;
    const size_t addn = BASE_SIZE;
    const int dim = DIM;
    const int capacity = CAP;
    const int resourceSize = -1;
    auto metricType = ascend::AscendMetricType::ASCEND_METRIC_INNER_PRODUCT;
    AscendThreadPool pool(indexNum * threadnumber);
    aclrtSetDevice(0);
    for (int i = 0; i < indexNum; i++) {
        ascend::IndexILFlat *indexIL = new ascend::IndexILFlat();
        indexIL->Init(dim, capacity, metricType, resourceSize);
        indexes.push_back(indexIL);
    }
    std::vector<float16_t> addVec(BASE_SIZE * dim);
    std::vector<ascend::idx_t> ids(BASE_SIZE);
    std::iota(ids.begin(), ids.end(), 0);
    CreateNormVector(addVec, BASE_SIZE, dim);
    for (int i = 0; i < indexNum; i++) {
        ret = indexes[i]->AddFeatures(BASE_SIZE, addVec.data(), ids.data());
        if (ret) {
            printf("Add features failed, error code: %d\n", ret);
            for (int j = 0; j < indexNum; j++) {
                indexes[j]->Finalize();
            }
            return;
        }
        int n = indexes[i]->GetNTotal();
        ASSERT_EQ(static_cast<size_t>(n), addn);
    }

    int eatchNumber = 400;
    std::vector<float16_t> query(queryN * dim);
    CreateNormVector(query, queryN, dim);
    std::vector<int> num(queryN);
    std::vector<std::vector<float>> dist(indexNum, std::vector<float>(queryN * topk * threadnumber));
    std::vector<std::vector<ascend::idx_t>> indices(indexNum, std::vector<ascend::idx_t>(queryN * topk * threadnumber));
    std::vector<std::vector<float>> distMulti(indexNum, std::vector<float>(queryN * topk * threadnumber));
    std::vector<std::vector<ascend::idx_t>> indicesMulti(indexNum, std::vector<ascend::idx_t>(queryN * topk * threadnumber));
    std::vector<std::vector<float16_t>> getFeatureDataMulti(indexNum, std::vector<float16_t>(eatchNumber * DIM * threadnumber));
    // single thread search
    for (int j = 0; j < indexNum; j++) {
        ret = indexes[j]->Search(queryN, query.data(), topk, indices[j].data(), dist[j].data());
        if (ret) {
            printf("search features failed, error code: %d\n", ret);
            for (int j = 0; j < indexNum; j++) {
                indexes[j]->Finalize();
            }
            return;
        }
    }

    // multi thread
    {
        for (int j = 0; j < indexNum; j++) {
            for (int i = 0; i < threadnumber; i++) {
                functorRet.emplace_back(
                    pool.Enqueue(func_search, indexes[j], dim, i, topk, queryN, query.data(), indicesMulti[j].data() + queryN * topk * i , distMulti[j].data() + queryN * topk * i));
                functorRet.emplace_back(pool.Enqueue(getFunctor, indexes[j], eatchNumber, i, getFeatureDataMulti[j].data() + i * eatchNumber * DIM, ids.data() + i * eatchNumber));
            } 
        }
        int seartchWait = 0;
        try {
            for (std::future<void> &ret : functorRet) {
                seartchWait++;
                ret.get();
            }
        } catch (std::exception &e) {
            for_each(functorRet.begin() + seartchWait, functorRet.end(), [](std::future<void> &ret) { ret.wait(); });
            printf("wait for search future failed.\n");
        }
        functorRet.clear();
    }



    for (int j = 0; j < indexNum; j++) {
        for (int i = 0; i < eatchNumber * DIM * threadnumber; i++) {
            ASSERT_FLOAT_EQ(*(getFeatureDataMulti[j].data() + i), *(addVec.data() + i));
        }
    }

    for (int j = 0; j < indexNum; j++) {
        for (int i = 0; i < queryN * topk * threadnumber; i++) {
            ASSERT_FLOAT_EQ(*(dist[j].data() + (i % (queryN * topk))), *(distMulti[j].data() + i));
            ASSERT_EQ(*(indices[j].data() + (i % (queryN * topk))), *(indicesMulti[j].data() + i));
        }
    }
    for (int j = 0; j < indexNum; j++) {
        indexes[j]->Finalize();
    }
    
}

}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    aclrtSetDevice(0);
    int ret = RUN_ALL_TESTS();
    aclrtResetDevice(0);
    return ret;
}