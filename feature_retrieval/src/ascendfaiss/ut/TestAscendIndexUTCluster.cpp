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
#include "faiss/ascendhost/include/index/AscendIndexCluster.h"
#include "faiss/impl/AuxIndexStructures.h"
#include "acl.h"
#include "ErrorCode.h"
namespace ascend {
constexpr int DIM = 512;
constexpr size_t BASE_SIZE = 1000;
constexpr int CUBE_ALIGN = 16;
constexpr int CAP = 1200;
const std::vector<int> DEVICES = { 0 };
constexpr int FAST_RAND_MAX = 0x7FFF;
const unsigned int TABLELEN_REDUNDANCY = 48;
unsigned int g_seed = 10;
static int32_t StubAclrtSetDevice(int32_t)
{
    return 1;
}
TEST(TestAscendIndexUTCluster, init_invalid_input)
{
    int dim = DIM;
    int capacity = CAP;

    auto metricType = faiss::MetricType::METRIC_INNER_PRODUCT;
    faiss::ascend::AscendIndexCluster index;
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

TEST(TestAscendIndexUTCluster, add_invalid_input)
{
    int addn = BASE_SIZE;
    int dim = DIM;
    int capacity = CAP;

    std::vector<float> addVec(addn * dim, 0);
    std::vector<uint32_t> ids(addn, 0);

    auto metricType = faiss::MetricType::METRIC_INNER_PRODUCT;
    faiss::ascend::AscendIndexCluster index;
    auto ret = index.Init(dim, capacity, metricType, DEVICES);
    EXPECT_EQ(ret, 0);

    ret = index.AddFeatures(addn, addVec.data(), ids.data());
    EXPECT_EQ(ret, 0);

    // add非16对齐的底库
    ret = index.AddFeatures(addn / CUBE_ALIGN * CUBE_ALIGN - 1, addVec.data(), ids.data());
    EXPECT_EQ(ret, 0);

    // setDevice报错
    MOCKER(aclrtSetDevice).stubs().will(invoke(StubAclrtSetDevice));
    ret = index.AddFeatures(addn, addVec.data(), ids.data());
    EXPECT_EQ(ret, APP_ERR_ACL_SET_DEVICE_FAILED);
    // mockcpp 需要显示调用该函数来恢复打桩
    GlobalMockObject::verify();

    // 添加底库条数大于最大值
    ret = index.AddFeatures(capacity + 1, addVec.data(), ids.data());
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);

    // 添加底库条数为0
    ret = index.AddFeatures(0, addVec.data(), ids.data());
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);

    // 添加底库数据指针为空
    ret = index.AddFeatures(addn, nullptr, ids.data());
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);

    // 添加底库label指针为空
    ret = index.AddFeatures(addn, addVec.data(), nullptr);
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);
    index.Finalize();
}

TEST(TestAscendIndexUTCluster, compute_invalid_input)
{
    std::vector<float> addVec(BASE_SIZE * DIM, 0);
    std::vector<uint32_t> ids(BASE_SIZE, 0);

    auto metricType = faiss::MetricType::METRIC_INNER_PRODUCT;
    faiss::ascend::AscendIndexCluster index;
    auto ret = index.Init(DIM, CAP, metricType, DEVICES);
    EXPECT_EQ(ret, 0);

    ret = index.AddFeatures(BASE_SIZE, addVec.data(), ids.data());
    EXPECT_EQ(ret, 0);

    uint32_t codeNum = BASE_SIZE / 100;
    uint32_t codeStartIdx = 0;

    int nq = 127;
    float threshold = 0.75;
    std::vector<uint32_t> queryIdxArr(nq, 0);
    std::vector<std::vector<float>> resDistArr(nq);
    std::vector<std::vector<uint32_t>> resIdxArr(nq);
    ret = index.ComputeDistanceByThreshold(
        queryIdxArr, codeStartIdx, codeNum, threshold, true, resDistArr, resIdxArr);
    EXPECT_EQ(ret, 0);

    // setDevice报错
    MOCKER(aclrtSetDevice).stubs().will(invoke(StubAclrtSetDevice));
    ret = index.ComputeDistanceByThreshold(
        queryIdxArr, codeStartIdx, codeNum, threshold, true, resDistArr, resIdxArr);
    EXPECT_EQ(ret, APP_ERR_ACL_SET_DEVICE_FAILED);
    // mockcpp 需要显示调用该函数来恢复打桩
    GlobalMockObject::verify();

    // queryNum传0
    std::vector<uint32_t> queryTemp;
    ret = index.ComputeDistanceByThreshold(
        queryTemp, codeStartIdx, codeNum, threshold, true, resDistArr, resIdxArr);
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);

    // codeNum
    ret = index.ComputeDistanceByThreshold(
        queryIdxArr, codeStartIdx, 0, threshold, true, resDistArr, resIdxArr);
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);

    // resDistArr长度跟queryNum不相等
    resDistArr.resize(0);
    ret = index.ComputeDistanceByThreshold(
        queryIdxArr, codeStartIdx, 0, threshold, true, resDistArr, resIdxArr);
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);

    // resIdxArr长度跟queryNum不相等
    resIdxArr.resize(nq - 1);
    ret = index.ComputeDistanceByThreshold(
        queryIdxArr, codeStartIdx, 0, threshold, true, resDistArr, resIdxArr);
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);

    index.Finalize();
}

TEST(TestAscendIndexUTCluster, All)
{
    int addn = BASE_SIZE;
    int dim = DIM;
    int capacity = CAP;
    uint32_t codeNum = BASE_SIZE / 1000;
    uint32_t codeStartIdx = 0;

    int nq = 127;
    bool aboveFilter = true;
    float threshold = 0.75;
    std::vector<float> addVec(addn * dim, 0);
    std::vector<uint32_t> queryIdxArr(nq, 0);
    std::vector<uint32_t> ids(addn, 0);

    auto metricType = faiss::MetricType::METRIC_INNER_PRODUCT;
    faiss::ascend::AscendIndexCluster index;
    auto ret = index.Init(dim, capacity, metricType, DEVICES);
    EXPECT_EQ(ret, 0);

    ret = index.AddFeatures(addn, addVec.data(), ids.data());
    EXPECT_EQ(ret, 0);

    std::vector<std::vector<float>> resDistArr(nq);
    std::vector<std::vector<uint32_t>> resIdxArr(nq);
    ret = index.ComputeDistanceByThreshold(
        queryIdxArr, codeStartIdx, codeNum, threshold, aboveFilter, resDistArr, resIdxArr);
    EXPECT_EQ(ret, 0);

    nq = 1;
    std::vector<uint32_t> queryIdxArr02(nq, 0);
    std::vector<std::vector<float>> resDistArr02(nq);
    std::vector<std::vector<uint32_t>> resIdxArr02(nq);
    ret = index.ComputeDistanceByThreshold(
        queryIdxArr02, codeStartIdx, codeNum, threshold, aboveFilter, resDistArr02, resIdxArr02);
    EXPECT_EQ(ret, 0);
    index.Finalize();
}

TEST(TestAscendIndexUTCluster, ComputeDistanceByIdx)
{
    int queryN = 2;
    const int maxNum = 10;
    const size_t addn = 100;
    const int64_t resourceSize = 1 * 1024 * 1024 * 1024;
    int dim = DIM;
    faiss::ascend::AscendIndexCluster index;
    std::vector<int> deviceList = { 0 };
    auto metricType = faiss::MetricType::METRIC_INNER_PRODUCT;
    auto ret = index.Init(dim, CAP, metricType, deviceList, resourceSize);
    EXPECT_EQ(ret, 0);

    std::vector<uint16_t> addVecFp16(addn * dim, 1);
    std::vector<int64_t> ids(addn);
    std::iota(ids.begin(), ids.end(), 0);
    ret = index.AddFeatures(addn, addVecFp16.data(), ids.data());
    EXPECT_EQ(ret, 0);

    std::vector<uint16_t> queries;
    queries.assign(addVecFp16.begin(), addVecFp16.begin() + queryN * dim);
    // 每个query指定maxNum个比对索引
    std::vector<int> num(queryN, maxNum);
    std::vector<float> distances(queryN * maxNum);
    std::vector<uint32_t> indice(queryN * maxNum, 1);
    // 初始化表长10000的float型scores表
    unsigned int tableLen = 10000;
    std::vector<float> table(tableLen + TABLELEN_REDUNDANCY);

    ret = index.ComputeDistanceByIdx(queryN, queries.data(), num.data(), indice.data(), distances.data());
    EXPECT_EQ(ret, 0);

    index.Finalize();
}

TEST(TestAscendIndexUTCluster, SearchByThreshold)
{
    int queryN = 2;
    const size_t addn = 100;
    const int64_t resourceSize = 1 * 1024 * 1024 * 1024;
    float threshold = 0.768;
    int topk = 200;
    int dim = DIM;
    faiss::ascend::AscendIndexCluster index;
    std::vector<int> deviceList = { 0 };
    auto metricType = faiss::MetricType::METRIC_INNER_PRODUCT;
    auto ret = index.Init(dim, CAP, metricType, deviceList, resourceSize);
    EXPECT_EQ(ret, 0);

    std::vector<uint16_t> addVecFp16(addn * dim, 1);
    std::vector<int64_t> ids(addn);
    std::iota(ids.begin(), ids.end(), 0);
    ret = index.AddFeatures(addn, addVecFp16.data(), ids.data());
    EXPECT_EQ(ret, 0);

    std::vector<uint16_t> queries;
    queries.assign(addVecFp16.begin(), addVecFp16.begin() + queryN * dim);
    // 每个query指定maxNum个比对索引
    std::vector<int> num(queryN);
    std::vector<int64_t> idxs(queryN * topk);
    std::vector<float> distances(queryN * topk);

    // 初始化表长10000的float型scores表
    unsigned int tableLen = 10000;
    std::vector<float> table(tableLen + TABLELEN_REDUNDANCY, 1.0);

    ret = index.SearchByThreshold(queryN, queries.data(), threshold, topk,
        num.data(), idxs.data(), distances.data(), tableLen, table.data());
    EXPECT_EQ(ret, 0);

    index.Finalize();
}

/*
 * Case Description: 测试正常情况
 * Preset Condition: 所有参数都有效，n在范围内，features和indices不为空，且所有索引值有效
 * Test Steps: 1.准备输入数据 2.调用AddFeatures方法
 * Expected Result: 返回APP_ERR_OK
 */
TEST(TestAscendIndexUTCluster, AddFeatures_ShouldReturnOk_WhenAllParamsAreValid)
{
    // Arrange
    int queryN = 2;
    const int64_t resourceSize = 1 * 1024 * 1024 * 1024;
    float threshold = 0.768;
    int dim = DIM;
    faiss::ascend::AscendIndexCluster index;
    std::vector<int> deviceList = { 0 };
    auto metricType = faiss::MetricType::METRIC_INNER_PRODUCT;
    auto ret = index.Init(dim, CAP, metricType, deviceList, resourceSize);
    EXPECT_EQ(ret, 0);

    int n = 5;
    std::vector<uint16_t> features(n * dim, 1);
    std::vector<int64_t> indices(n, 1);

    // Act
    APP_ERROR result = index.AddFeatures(n, features.data(), indices.data());

    // Assert
    EXPECT_EQ(result, APP_ERR_OK);
    index.Finalize();
}

/*
 * Case Description: 测试n=0的情况
 * Preset Condition: n=0
 * Test Steps: 1.设置n=0 2.调用AddFeatures方法
 * Expected Result: 返回APP_ERR_INVALID_PARAM
 */
TEST(TestAscendIndexUTCluster, AddFeatures_ShouldReturnInvalidParam_WhenNIsZero)
{
    // Arrange
    const int64_t resourceSize = 1 * 1024 * 1024 * 1024;
    int dim = DIM;
    faiss::ascend::AscendIndexCluster index;
    std::vector<int> deviceList = { 0 };
    auto metricType = faiss::MetricType::METRIC_INNER_PRODUCT;
    auto ret = index.Init(dim, CAP, metricType, deviceList, resourceSize);
    EXPECT_EQ(ret, 0);
    int n = 0;
    const uint16_t* features = nullptr;
    const int64_t* indices = nullptr;

    // Act
    APP_ERROR result = index.AddFeatures(n, features, indices);

    // Assert
    EXPECT_EQ(result, APP_ERR_INVALID_PARAM);
    index.Finalize();
}

/*
 * Case Description: 测试未初始化的情况
 * Preset Condition: 当isInitialized为false时
 * Test Steps: 1.设置isInitialized为false 2.调用SearchByThreshold方法
 * Expected Result: 返回APP_ERR_INVALID_PARAM
 */
TEST(TestAscendIndexUTCluster, SearchByThreshold_ShouldReturnInvalidParam_WhenNotInitialized)
{
    // Arrange
    const int64_t resourceSize = 1 * 1024 * 1024 * 1024;
    int dim = DIM;
    faiss::ascend::AscendIndexCluster index;
    std::vector<int> deviceList = { 0 };
    auto metricType = faiss::MetricType::METRIC_INNER_PRODUCT;
    auto ret = index.Init(dim, CAP, metricType, deviceList, resourceSize);
    EXPECT_EQ(ret, 0);

    int n = 5;
    std::vector<uint16_t> features(n * dim, 1);
    std::vector<int64_t> indices(n, 1);

    // Act
    APP_ERROR result = index.AddFeatures(n, features.data(), indices.data());
    EXPECT_EQ(result, 0);
    const uint16_t* queries = nullptr;
    float threshold = 0.0f;
    int topk = 1;
    int* num = nullptr;
    int64_t* labels = nullptr;
    float* distances = nullptr;
    unsigned int tableLen = 0;
    const float* table = nullptr;

    // Act
    result = index.SearchByThreshold(n, queries, threshold, topk, num, labels, distances, tableLen, table);
    // Assert
    EXPECT_EQ(result, APP_ERR_INVALID_PARAM);
    index.Finalize();
}

TEST(TestAscendIndexUTCluster, GetNTotal)
{
    const size_t addn = 100;
    const int64_t resourceSize = 1 * 1024 * 1024 * 1024;
    faiss::ascend::AscendIndexCluster index;
    std::vector<int> deviceList = { 0 };
    auto metricType = faiss::MetricType::METRIC_INNER_PRODUCT;
    auto ret = index.Init(DIM, CAP, metricType, deviceList, resourceSize);
    EXPECT_EQ(ret, 0);

    std::vector<uint16_t> addVecFp16(addn * DIM, 1);
    std::vector<int64_t> ids(addn);
    std::iota(ids.begin(), ids.end(), 0);
    ret = index.AddFeatures(addn, addVecFp16.data(), ids.data());
    EXPECT_EQ(ret, 0);
    int num = index.GetNTotal();
    EXPECT_EQ(num, addn);
    index.Finalize();
}

}