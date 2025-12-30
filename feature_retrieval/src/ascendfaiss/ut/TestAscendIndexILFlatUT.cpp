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
#include <cmath>
#include <random>
#include <gtest/gtest.h>
#include <cstring>
#include <sys/time.h>
#include <faiss/index_io.h>
#include <cstdlib>
#include <mockcpp/mockcpp.hpp>
#include "faiss/ascendhost/include/index/AscendIndexILFlat.h"
#include "faiss/impl/AuxIndexStructures.h"
#include "acl.h"
#include "ErrorCode.h"
namespace ascend {
constexpr int DIM = 512;
constexpr size_t BASE_SIZE = 1000;
constexpr int CUBE_ALIGN = 16;
constexpr int CAP = 1200;
const std::vector<int> DEVICES = { 0 };
const unsigned int TABLELEN_REDUNDANCY = 48;
static int32_t StubAclrtSetDevice(int32_t)
{
    return 1;
}
TEST(TestAscendIndexUTILFlat, init_invalid_input)
{
    int dim = DIM;
    int capacity = CAP;
 
    auto metricType = faiss::MetricType::METRIC_INNER_PRODUCT;
    faiss::ascend::AscendIndexILFlat index;
    auto ret = index.Init(dim, capacity, metricType, DEVICES);
    EXPECT_EQ(ret, 0);
    index.Finalize();
 
    // setDevice报错
    MOCKER(aclrtSetDevice).stubs().will(invoke(StubAclrtSetDevice));
    ret = index.Init(dim, capacity, metricType, DEVICES);
    EXPECT_EQ(ret, APP_ERR_ACL_SET_DEVICE_FAILED);
    index.Finalize();
    // mockcpp 需要显示调用该函数来恢复打桩
    GlobalMockObject::verify();
 
    // 输入capacity超出范围
    int invalidCap = 12000001;
    ret = index.Init(dim, invalidCap, metricType, DEVICES);
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);
    index.Finalize();
 
    // 输入capacity为0
    ret = index.Init(dim, 0, metricType, DEVICES);
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);
    index.Finalize();
 
    // 输入dim * cap 大于最大申请内存大小
    ret = index.Init(dim + 1, capacity, metricType, DEVICES);
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);
    index.Finalize();
 
    // 输入type不是ip
    ret = index.Init(dim, capacity, faiss::MetricType::METRIC_L2, DEVICES);
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);
    index.Finalize();
 
    // 输入非法resourceSize
    int64_t resourceSize = static_cast<int64_t>(2) * static_cast<int64_t>(1024 * 1024 * 1024);
    ret = index.Init(dim, capacity, metricType, DEVICES, resourceSize * 3);
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);
    index.Finalize();
 
    ret = index.Init(dim, capacity, metricType, DEVICES, 0);
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);
    index.Finalize();
 
    // 输入多device
    ret = index.Init(dim, capacity, metricType, { 0, 1 });
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);
    index.Finalize();
}
 
TEST(TestAscendIndexUTILFlat, add_invalid_input)
{
    int addn = BASE_SIZE;
    int dim = DIM;
    int capacity = CAP;
 
    std::vector<float> addVec(addn * dim, 0);
 
    auto metricType = faiss::MetricType::METRIC_INNER_PRODUCT;
    faiss::ascend::AscendIndexILFlat index;
    auto ret = index.Init(dim, capacity, metricType, DEVICES);
    EXPECT_EQ(ret, 0);
 
    ret = index.AddFeatures(addn, addVec.data());
    EXPECT_EQ(ret, 0);
 
    // 添加底库条数大于最大值
    ret = index.AddFeatures(capacity + 1, addVec.data());
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);
 
    // 添加底库条数为0
    ret = index.AddFeatures(0, addVec.data());
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);
 
    index.Finalize();
}
 
TEST(TestAscendIndexUTILFlat, ComputeDistance)
{
    int queryN = 2;
    const size_t addn = 100;
    const int64_t resourceSize = 1 * 1024 * 1024 * 1024;
    faiss::ascend::AscendIndexILFlat index;
    std::vector<int> deviceList = { 0 };
    auto metricType = faiss::MetricType::METRIC_INNER_PRODUCT;
    auto ret = index.Init(DIM, CAP, metricType, deviceList, resourceSize);
    EXPECT_EQ(ret, 0);
 
    std::vector<uint16_t> addVecFp16(addn * DIM, 1);
    ret = index.AddFeatures(addn, addVecFp16.data());
    EXPECT_EQ(ret, 0);
 
    std::vector<uint16_t> queries;
    queries.assign(addVecFp16.begin(), addVecFp16.begin() + queryN * DIM);
    int nTotalPad = (addn + 15) / 16 * 16;
    std::vector<float> distances(queryN * nTotalPad);
 
    ret = index.ComputeDistance(queryN, queries.data(), distances.data());
    EXPECT_EQ(ret, 0);
 
    index.Finalize();
}
 
TEST(TestAscendIndexUTILFlat, ComputeDistanceFP32)
{
    int queryN = 2;
    const size_t addn = 100;
    const int64_t resourceSize = 1 * 1024 * 1024 * 1024;
    faiss::ascend::AscendIndexILFlat index;
    std::vector<int> deviceList = { 0 };
    auto metricType = faiss::MetricType::METRIC_INNER_PRODUCT;
    auto ret = index.Init(DIM, CAP, metricType, deviceList, resourceSize);
    EXPECT_EQ(ret, 0);
 
    std::vector<float> addVecFp32(addn * DIM, 1);
    ret = index.AddFeatures(addn, addVecFp32.data());
    EXPECT_EQ(ret, 0);
 
    std::vector<float> queries;
    queries.assign(addVecFp32.begin(), addVecFp32.begin() + queryN * DIM);
    int nTotalPad = (addn + 15) / 16 * 16;
    std::vector<float> distances(queryN * nTotalPad);
 
    ret = index.ComputeDistance(queryN, queries.data(), distances.data());
    EXPECT_EQ(ret, 0);
 
    index.Finalize();
}
 
TEST(TestAscendIndexUTILFlat, Search)
{
    int queryN = 2;
    const size_t addn = 100;
    const int64_t resourceSize = 1 * 1024 * 1024 * 1024;
    int topk = 200;
    faiss::ascend::AscendIndexILFlat index;
    std::vector<int> deviceList = { 0 };
    auto metricType = faiss::MetricType::METRIC_INNER_PRODUCT;
    auto ret = index.Init(DIM, CAP, metricType, deviceList, resourceSize);
    EXPECT_EQ(ret, 0);
 
    std::vector<uint16_t> addVecFp16(addn * DIM, 1);

    ret = index.AddFeatures(addn, addVecFp16.data());
    EXPECT_EQ(ret, 0);
 
    std::vector<uint16_t> queries;
    queries.assign(addVecFp16.begin(), addVecFp16.begin() + queryN * DIM);
    std::vector<uint32_t> idxs(queryN * topk);
    std::vector<float> distances(queryN * topk);
 
    // 初始化表长10000的float型scores表
    unsigned int tableLen = 10000;
    std::vector<float> table(tableLen + TABLELEN_REDUNDANCY, 1.0);
 
    ret = index.Search(queryN, queries.data(), topk, idxs.data(), distances.data(), tableLen, table.data());
    EXPECT_EQ(ret, 0);
 
    index.Finalize();
}
 
TEST(TestAscendIndexUTILFlat, SearchFP32)
{
    int queryN = 2;
    const size_t addn = 100;
    const int64_t resourceSize = 1 * 1024 * 1024 * 1024;
    int topk = 200;
    faiss::ascend::AscendIndexILFlat index;
    std::vector<int> deviceList = { 0 };
    auto metricType = faiss::MetricType::METRIC_INNER_PRODUCT;
    auto ret = index.Init(DIM, CAP, metricType, deviceList, resourceSize);
    EXPECT_EQ(ret, 0);
 
    std::vector<float> addVecFp32(addn * DIM, 1);
    ret = index.AddFeatures(addn, addVecFp32.data());
    EXPECT_EQ(ret, 0);
 
    std::vector<float> queries;
    queries.assign(addVecFp32.begin(), addVecFp32.begin() + queryN * DIM);
    std::vector<uint32_t> idxs(queryN * topk);
    std::vector<float> distances(queryN * topk);
 
    // 初始化表长10000的float型scores表
    unsigned int tableLen = 10000;
    std::vector<float> table(tableLen + TABLELEN_REDUNDANCY, 1.0);
 
    ret = index.Search(queryN, queries.data(), topk, idxs.data(), distances.data(), tableLen, table.data());
    EXPECT_EQ(ret, 0);
 
    index.Finalize();
}

TEST(TestAscendIndexUTILFlat, Updata)
{
    const size_t addn = 100;
    const int64_t resourceSize = 1 * 1024 * 1024 * 1024;
    faiss::ascend::AscendIndexILFlat index;
    std::vector<int> deviceList = { 0 };
    auto metricType = faiss::MetricType::METRIC_INNER_PRODUCT;
    auto ret = index.Init(DIM, CAP, metricType, deviceList, resourceSize);
    EXPECT_EQ(ret, 0);
 
    std::vector<uint16_t> addVecFp16(addn * DIM, 1);

    ret = index.AddFeatures(addn, addVecFp16.data());
    EXPECT_EQ(ret, 0);
    int updateN = 1;
    std::vector<uint32_t> ids{0};
    std::vector<uint16_t> UpdateDddVec(updateN * DIM);
    ret = index.UpdateFeatures(updateN, UpdateDddVec.data(), ids.data());
    EXPECT_EQ(ret, 0);
    index.Finalize();
}

TEST(TestAscendIndexUTILFlat, SearchByThresholdFP32)
{
    int queryN = 2;
    const size_t addn = 100;
    const int64_t resourceSize = 1 * 1024 * 1024 * 1024;
    int topk = 200;
    faiss::ascend::AscendIndexILFlat index;
    std::vector<int> deviceList = { 0 };
    auto metricType = faiss::MetricType::METRIC_INNER_PRODUCT;
    auto ret = index.Init(DIM, CAP, metricType, deviceList, resourceSize);
    EXPECT_EQ(ret, 0);
 
    std::vector<float> addVecFp32(addn * DIM, 1);
    ret = index.AddFeatures(addn, addVecFp32.data());
    EXPECT_EQ(ret, 0);
 
    std::vector<float> queries;
    queries.assign(addVecFp32.begin(), addVecFp32.begin() + queryN * DIM);
    std::vector<uint32_t> idxs(queryN * topk);
    std::vector<float> distances(queryN * topk);
 
    // 初始化表长10000的float型scores表
    unsigned int tableLen = 10000;
    std::vector<float> table(tableLen + TABLELEN_REDUNDANCY, 1.0);
    std::vector<int> num(queryN);
    float threshold = 0.768;
    ret = index.SearchByThreshold(queryN, queries.data(), threshold, topk, num.data(), idxs.data(), distances.data());
    EXPECT_EQ(ret, 0);
 
    index.Finalize();
}

TEST(TestAscendIndexUTILFlat, SearchByThresholdFP16)
{
    int queryN = 2;
    const size_t addn = 100;
    const int64_t resourceSize = 1 * 1024 * 1024 * 1024;
    int topk = 200;
    faiss::ascend::AscendIndexILFlat index;
    std::vector<int> deviceList = { 0 };
    auto metricType = faiss::MetricType::METRIC_INNER_PRODUCT;
    auto ret = index.Init(DIM, CAP, metricType, deviceList, resourceSize);
    EXPECT_EQ(ret, 0);
 
    std::vector<uint16_t> addVecFp16(addn * DIM, 1);

    ret = index.AddFeatures(addn, addVecFp16.data());
    EXPECT_EQ(ret, 0);
 
    std::vector<uint16_t> queries;
    queries.assign(addVecFp16.begin(), addVecFp16.begin() + queryN * DIM);
    std::vector<uint32_t> idxs(queryN * topk);
    std::vector<float> distances(queryN * topk);
 
    // 初始化表长10000的float型scores表
    unsigned int tableLen = 10000;
    std::vector<float> table(tableLen + TABLELEN_REDUNDANCY, 1.0);
    std::vector<int> num(queryN);
    float threshold = 0.768;
    ret = index.SearchByThreshold(queryN, queries.data(), threshold, topk, num.data(), idxs.data(), distances.data());
    EXPECT_EQ(ret, 0);
 
    index.Finalize();
}

TEST(TestAscendIndexUTILFlat, ComputeDistanceByIdx)
{
    int queryN = 2;
    const size_t addn = 100;
    const int maxNum = 100;
    const int64_t resourceSize = 1 * 1024 * 1024 * 1024;
    faiss::ascend::AscendIndexILFlat index;
    std::vector<int> deviceList = { 0 };
    auto metricType = faiss::MetricType::METRIC_INNER_PRODUCT;
    auto ret = index.Init(DIM, CAP, metricType, deviceList, resourceSize);
    EXPECT_EQ(ret, 0);
 
    std::vector<uint16_t> addVecFp16(addn * DIM, 1);
    ret = index.AddFeatures(addn, addVecFp16.data());
    EXPECT_EQ(ret, 0);
 
    std::vector<uint16_t> queries;
    queries.assign(addVecFp16.begin(), addVecFp16.begin() + queryN * DIM);
    std::vector<float> distances(queryN * maxNum);
    std::vector<uint32_t> indice(queryN * maxNum);
    std::vector<int> num(queryN, maxNum);
 
    ret = index.ComputeDistanceByIdx(queryN, queries.data(), num.data(), indice.data(), distances.data());
    EXPECT_EQ(ret, 0);
 
    index.Finalize();
}
 
TEST(TestAscendIndexUTILFlat, ComputeDistanceByIdxFP32)
{
    int queryN = 2;
    const size_t addn = 100;
    const int64_t resourceSize = 1 * 1024 * 1024 * 1024;
    const int maxNum = 100;
    faiss::ascend::AscendIndexILFlat index;
    std::vector<int> deviceList = { 0 };
    auto metricType = faiss::MetricType::METRIC_INNER_PRODUCT;
    auto ret = index.Init(DIM, CAP, metricType, deviceList, resourceSize);
    EXPECT_EQ(ret, 0);
 
    std::vector<float> addVecFp32(addn * DIM, 1);
    ret = index.AddFeatures(addn, addVecFp32.data());
    EXPECT_EQ(ret, 0);
 
    std::vector<float> queries;
    queries.assign(addVecFp32.begin(), addVecFp32.begin() + queryN * DIM);

    std::vector<float> distances(queryN * maxNum);
    std::vector<uint32_t> indice(queryN * maxNum);
    std::vector<int> num(queryN, maxNum);
    ret = index.ComputeDistanceByIdx(queryN, queries.data(), num.data(), indice.data(), distances.data());
    EXPECT_EQ(ret, 0);
 
    index.Finalize();
}

TEST(TestAscendIndexUTILFlat, ComputeDistanceByIdxFP32_Input_Device)
{
    int queryN = 2;
    const size_t addn = 100;
    const int64_t resourceSize = 1 * 1024 * 1024 * 1024;
    const int maxNum = 100;
    faiss::ascend::AscendIndexILFlat index;
    std::vector<int> deviceList = { 0 };
    auto metricType = faiss::MetricType::METRIC_INNER_PRODUCT;
    auto ret = index.Init(DIM, CAP, metricType, deviceList, resourceSize);
    EXPECT_EQ(ret, 0);
 
    std::vector<float> addVecFp32(addn * DIM, 1);
    ret = index.AddFeatures(addn, addVecFp32.data());
    EXPECT_EQ(ret, 0);
 
    std::vector<float> queriesHost;
    queriesHost.assign(addVecFp32.begin(), addVecFp32.begin() + queryN * DIM);

    void* queriesPtr = nullptr;
    size_t queriesMemSize = queryN * DIM * sizeof(float);
    ret = aclrtMalloc(&queriesPtr, queriesMemSize, ACL_MEM_MALLOC_HUGE_FIRST);
    EXPECT_EQ(ret, 0);
    float *queries = static_cast<float *>(queriesPtr);

    ret = aclrtMemcpy(queries, queryN * DIM * sizeof(float), queriesHost.data(),
        queriesHost.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    EXPECT_EQ(ret, 0);

    std::vector<float> distances(queryN * maxNum);
    std::vector<uint32_t> indice(queryN * maxNum);
    std::vector<int> num(queryN, maxNum);
    ret = index.ComputeDistanceByIdx(queryN, queries, num.data(), indice.data(), distances.data(),
        faiss::ascend::MEMORY_TYPE::INPUT_DEVICE_OUTPUT_HOST);
    EXPECT_EQ(ret, 0);

    void* distancesPtr = nullptr;
    size_t distancesSize = queryN * maxNum * sizeof(float);
    ret = aclrtMalloc(&distancesPtr, distancesSize, ACL_MEM_MALLOC_HUGE_FIRST);
    EXPECT_EQ(ret, 0);

    float *distancesDevice = static_cast<float *>(distancesPtr);
    ret = index.ComputeDistanceByIdx(queryN, queries, num.data(), indice.data(), distancesDevice,
        faiss::ascend::MEMORY_TYPE::INPUT_DEVICE_OUTPUT_DEVICE);
    EXPECT_EQ(ret, 0);
    aclrtFree(distancesPtr);
    aclrtFree(queriesPtr);
    index.Finalize();
}

TEST(TestAscendIndexUTILFlat, ComputeDistanceByIdxFP16_Input_Device)
{
    int queryN = 2;
    const size_t addn = 100;
    const int64_t resourceSize = 1 * 1024 * 1024 * 1024;
    const int maxNum = 100;
    faiss::ascend::AscendIndexILFlat index;
    std::vector<int> deviceList = { 0 };
    auto metricType = faiss::MetricType::METRIC_INNER_PRODUCT;
    auto ret = index.Init(DIM, CAP, metricType, deviceList, resourceSize);
    EXPECT_EQ(ret, 0);
 
    std::vector<uint16_t> addVecFp16(addn * DIM, 1);
    ret = index.AddFeatures(addn, addVecFp16.data());
    EXPECT_EQ(ret, 0);
 
    std::vector<uint16_t> queriesHost;
    queriesHost.assign(addVecFp16.begin(), addVecFp16.begin() + queryN * DIM);

    void* queriesPtr = nullptr;
    size_t queriesMemSize = queryN * DIM * sizeof(uint16_t);
    ret = aclrtMalloc(&queriesPtr, queriesMemSize, ACL_MEM_MALLOC_HUGE_FIRST);
    EXPECT_EQ(ret, 0);
    uint16_t *queries = static_cast<uint16_t *>(queriesPtr);

    ret = aclrtMemcpy(queries, queryN * DIM * sizeof(uint16_t), queriesHost.data(),
        queriesHost.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    EXPECT_EQ(ret, 0);

    std::vector<float> distances(queryN * maxNum);
    std::vector<uint32_t> indice(queryN * maxNum);
    std::vector<int> num(queryN, maxNum);
    ret = index.ComputeDistanceByIdx(queryN, queries, num.data(), indice.data(), distances.data(),
        faiss::ascend::MEMORY_TYPE::INPUT_DEVICE_OUTPUT_HOST);
    EXPECT_EQ(ret, 0);

    void* distancesPtr = nullptr;
    size_t distancesSize = queryN * maxNum * sizeof(float);
    ret = aclrtMalloc(&distancesPtr, distancesSize, ACL_MEM_MALLOC_HUGE_FIRST);
    EXPECT_EQ(ret, 0);

    float *distancesDevice = static_cast<float *>(distancesPtr);
    ret = index.ComputeDistanceByIdx(queryN, queries, num.data(), indice.data(), distancesDevice,
        faiss::ascend::MEMORY_TYPE::INPUT_DEVICE_OUTPUT_DEVICE);
    EXPECT_EQ(ret, 0);
    aclrtFree(distancesPtr);
    aclrtFree(queriesPtr);
    index.Finalize();
}

}