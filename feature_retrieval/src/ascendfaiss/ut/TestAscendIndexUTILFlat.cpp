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
#include <gtest/gtest.h>
#include <numeric>
#include <vector>
#include "common/threadpool/AscendThreadPool.h"
#include "faiss/ascenddaemon/impl_device/IndexILFlat.h"
#include "faiss/impl/AuxIndexStructures.h"
#include "securec.h"
#include "simu/AscendSimu.h"


namespace {
constexpr int DIM = 512;
constexpr size_t BASE_SIZE = 1000;
constexpr uint32_t CUBE_ALIGN = 16;
constexpr int CAP = 120000;
constexpr int MAX_CAP = 12000000;              // Upper limit for capacity
constexpr size_t MAX_BASE_SPACE = 12288000000; // max bytes to store base vectors.
constexpr int QUERY_NUM = 2;
constexpr int TOPK = 1;
constexpr int64_t RESOURCE_SIZE = static_cast<int64_t>(2) * static_cast<int64_t>(1024 * 1024 * 1024);
constexpr ascend::AscendMetricType IP_TYPE = ascend::AscendMetricType::ASCEND_METRIC_INNER_PRODUCT;
constexpr ascend::AscendMetricType L2_TYPE = ascend::AscendMetricType::ASCEND_METRIC_L2;
const unsigned int TABLELEN_REDUNDANCY = 48;

constexpr int UNINITIALIZE_NTOTAL = -1; // uninitialize ntotal

enum class ErrorCode {
    APP_ERR_OK = 0,
    APP_ERR_INVALID_PARAM = 2001,
    APP_ERR_ILLEGAL_OPERATION = 2009,
    APP_ERR_INDEX_NOT_INIT = 2016
};

TEST(TestAscendIndexUTILFlat, test_GetNTotal)
{
    aclrtSetDevice(0);
    ascend::IndexILFlat index;

    // 未初始化index时获取ntotal
    int ntotal = index.GetNTotal();
    EXPECT_EQ(ntotal, UNINITIALIZE_NTOTAL);

    auto ret = index.Init(DIM, CAP, IP_TYPE, RESOURCE_SIZE);
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));

    // 初始化index但未add底库时获取ntotal
    ntotal = index.GetNTotal();
    EXPECT_EQ(ntotal, 0);

    std::vector<float16_t> addVec(BASE_SIZE * DIM, 0);
    std::vector<ascend::idx_t> ids(BASE_SIZE);
    std::iota(ids.begin(), ids.end(), 0);
    ret = index.AddFeatures(BASE_SIZE, addVec.data(), ids.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));

    // add底库后获取ntotal
    ntotal = index.GetNTotal();
    EXPECT_EQ(ntotal, static_cast<int>(BASE_SIZE));

    // SetNTotal后获取ntotal
    ret = index.SetNTotal(BASE_SIZE / 2);
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));
    ntotal = index.GetNTotal();
    EXPECT_EQ(ntotal, static_cast<int>(BASE_SIZE / 2));

    std::vector<float16_t> query(QUERY_NUM * DIM, 0);
    std::vector<float> dist(QUERY_NUM * QUERY_NUM, 0.0);
    std::vector<ascend::idx_t> indices(QUERY_NUM * QUERY_NUM, 0);

    ret = index.Search(QUERY_NUM, query.data(), TOPK, indices.data(), dist.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));

    index.Finalize();
}

TEST(TestAscendIndexUTILFlat, test_SetNTotal)
{
    aclrtSetDevice(0);
    ascend::IndexILFlat index;

    // 未初始化index时设置ntotal
    auto ret = index.SetNTotal(0);
    EXPECT_NE(ret, static_cast<int>(ErrorCode::APP_ERR_OK));

    ret = index.Init(DIM, CAP, IP_TYPE, RESOURCE_SIZE);
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));

    // ntotal设置为无效值
    ret = index.SetNTotal(0 - static_cast<int>(BASE_SIZE));
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_INVALID_PARAM));

    ret = index.SetNTotal(CAP + 1);
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_INVALID_PARAM));

    // ntotal设置为有效值
    ret = index.SetNTotal(BASE_SIZE);
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));
    int ntotal = index.GetNTotal();
    EXPECT_EQ(ntotal, static_cast<int>(BASE_SIZE));

    index.Finalize();
}

TEST(TestAscendIndexUTILFlat, test_AddFeatures)
{
    aclrtSetDevice(0);
    ascend::IndexILFlat index;

    std::vector<float16_t> addVec(BASE_SIZE * DIM, 0);
    std::vector<ascend::idx_t> ids(BASE_SIZE);
    std::iota(ids.begin(), ids.end(), 0);

    // 未初始化index时添加底库
    auto ret = index.AddFeatures(BASE_SIZE, addVec.data(), ids.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_INDEX_NOT_INIT));

    ret = index.Init(DIM, CAP, IP_TYPE, RESOURCE_SIZE);
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));

    // 无效参数调用
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));

    ret = index.AddFeatures(-1, addVec.data(), ids.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_INVALID_PARAM));

    ret = index.AddFeatures(CAP + 1, addVec.data(), ids.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_INVALID_PARAM));

    ret = index.AddFeatures(BASE_SIZE, addVec.data(), nullptr);
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_INVALID_PARAM));

    ret = index.AddFeatures(BASE_SIZE, nullptr, ids.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_INVALID_PARAM));

    // 正常添加底库
    ret = index.AddFeatures(BASE_SIZE, addVec.data(), ids.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));

    // add底库后获取ntotal
    int ntotal = index.GetNTotal();
    EXPECT_EQ(ntotal, static_cast<int>(BASE_SIZE));

    std::vector<float16_t> query(QUERY_NUM * DIM, 0);
    std::vector<float> dist(QUERY_NUM * QUERY_NUM, 0.0);
    std::vector<ascend::idx_t> indices(QUERY_NUM * QUERY_NUM, 0);

    ret = index.Search(QUERY_NUM, query.data(), TOPK, indices.data(), dist.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));

    index.Finalize();
}

TEST(TestAscendIndexUTILFlat, test_GetFeatures)
{
    aclrtSetDevice(0);
    ascend::IndexILFlat index;

    std::vector<float16_t> addVec(BASE_SIZE * DIM);
    std::iota(addVec.begin(), addVec.end(), 0);

    std::vector<ascend::idx_t> ids(BASE_SIZE);
    std::iota(ids.begin(), ids.end(), 0);

    std::vector<ascend::idx_t> idsOut(BASE_SIZE);
    std::iota(idsOut.begin(), idsOut.end(), 0);
    std::vector<float16_t> addVecOut(BASE_SIZE * DIM);

    // 未初始化index时get底库
    auto ret = index.GetFeatures(BASE_SIZE, addVecOut.data(), idsOut.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_INDEX_NOT_INIT));

    ret = index.Init(DIM, CAP, IP_TYPE, RESOURCE_SIZE);
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));

    // 未add底库时get底库
    ret = index.GetFeatures(BASE_SIZE, addVecOut.data(), idsOut.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_INVALID_PARAM));

    ret = index.AddFeatures(BASE_SIZE, addVec.data(), ids.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));

    // 无效参数调用
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));

    ret = index.GetFeatures(-1, addVecOut.data(), idsOut.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_INVALID_PARAM));

    ret = index.GetFeatures(CAP + 1, addVecOut.data(), idsOut.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_INVALID_PARAM));

    ret = index.GetFeatures(BASE_SIZE, addVecOut.data(), nullptr);
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_INVALID_PARAM));

    ret = index.GetFeatures(BASE_SIZE, nullptr, idsOut.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_INVALID_PARAM));

    // 正常get底库
    ret = index.GetFeatures(BASE_SIZE, addVecOut.data(), idsOut.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));
    for (size_t i = 0; i < BASE_SIZE; i++) {
        EXPECT_EQ(addVec[i], addVecOut[i]);
    }

    // get超过ntotal的下标的底库向量
    std::iota(idsOut.begin(), idsOut.end(), BASE_SIZE);
    ret = index.GetFeatures(BASE_SIZE, addVecOut.data(), idsOut.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_INVALID_PARAM));

    index.Finalize();
}

TEST(TestAscendIndexUTILFlat, test_RemoveFeatures)
{
    aclrtSetDevice(0);
    ascend::IndexILFlat index;

    std::vector<float16_t> addVec(BASE_SIZE * DIM);
    std::iota(addVec.begin(), addVec.end(), 0);

    std::vector<ascend::idx_t> ids(BASE_SIZE);
    std::iota(ids.begin(), ids.end(), 0);

    auto ret = index.Init(DIM, CAP, IP_TYPE, RESOURCE_SIZE);
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));

    ret = index.AddFeatures(BASE_SIZE, addVec.data(), ids.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));

    // 无效参数调用
    ret = index.RemoveFeatures(0, ids.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));

    ret = index.RemoveFeatures(-1, ids.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_INVALID_PARAM));

    ret = index.RemoveFeatures(CAP + 1, ids.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_INVALID_PARAM));

    ret = index.RemoveFeatures(BASE_SIZE, nullptr);
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_INVALID_PARAM));

    // 正常remove底库
    std::vector<float16_t> addVecOut(BASE_SIZE * DIM);
    ret = index.RemoveFeatures(BASE_SIZE, ids.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));

    ret = index.GetFeatures(10, addVecOut.data(), ids.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));
    for (size_t i = 0; i < 10; i++) {
        EXPECT_EQ(0, addVecOut[i]);
    }

    // get超过ntotal的下标的底库向量
    std::iota(ids.begin(), ids.end(), BASE_SIZE);
    ret = index.RemoveFeatures(BASE_SIZE, ids.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_INVALID_PARAM));

    index.Finalize();
}

TEST(TestAscendIndexUTILFlat, test_Init)
{
    aclrtSetDevice(0);
    ascend::IndexILFlat index;

    // 无效参数
    auto ret = index.Init(DIM, 0, IP_TYPE, RESOURCE_SIZE);
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_INVALID_PARAM));

    ret = index.Init(DIM + 1, MAX_BASE_SPACE / DIM / sizeof(float16_t), IP_TYPE, RESOURCE_SIZE);
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_INVALID_PARAM));

    int dim = 385;
    ret = index.Init(dim, CAP, IP_TYPE, RESOURCE_SIZE);
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_INVALID_PARAM));

    // 正常init
    ret = index.Init(DIM, CAP, IP_TYPE, RESOURCE_SIZE);
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));

    // 二次init
    ret = index.Init(DIM, CAP, IP_TYPE, RESOURCE_SIZE);
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_ILLEGAL_OPERATION));

    index.Finalize();

    // finalize之后，其他接口会调用失败
    int ntotal = index.GetNTotal();
    EXPECT_EQ(ntotal, UNINITIALIZE_NTOTAL);
}

TEST(TestAscendIndexUTILFlat, test_ComputeDistance)
{
    aclrtSetDevice(0);
    ascend::IndexILFlat index;
    index.Init(DIM, CAP, IP_TYPE, RESOURCE_SIZE);
    uint32_t ntotalPad = (BASE_SIZE + CUBE_ALIGN - 1) / CUBE_ALIGN * CUBE_ALIGN;

    std::vector<float16_t> addVec(BASE_SIZE * DIM, 0);
    std::vector<ascend::idx_t> ids(BASE_SIZE);
    std::iota(ids.begin(), ids.end(), 0);
    auto ret = index.AddFeatures(BASE_SIZE, addVec.data(), ids.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));

    float threshold = 0.75;
    std::vector<float16_t> query(QUERY_NUM * DIM, 0);
    std::vector<int> num(QUERY_NUM);
    std::vector<float> dist(QUERY_NUM * ntotalPad);
    std::vector<float> distByThreshold(QUERY_NUM * ntotalPad);
    std::vector<uint32_t> indices(QUERY_NUM * ntotalPad);

    ret = index.ComputeDistance(QUERY_NUM, query.data(), dist.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));

    ret = index.ComputeDistanceByThreshold(QUERY_NUM, query.data(), threshold, num.data(), indices.data(),
                                           distByThreshold.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));

    ret = index.ComputeDistanceByIdx(QUERY_NUM, query.data(), num.data(), indices.data(), distByThreshold.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));

    index.Finalize();
}

static void CreateMappingTable(float *table, unsigned int tableLen)
{
    for (unsigned int i = 0; i < tableLen; i++) {
        // 创建映射表，score分值加2， 保证在2.0~3.0之间
        *(table + i) = i * 1.0;
    }
}

TEST(TestAscendIndexUTILFlat, test_Search)
{
    aclrtSetDevice(0);
    ascend::IndexILFlat index;
    index.Init(DIM, CAP, IP_TYPE, RESOURCE_SIZE);

    std::vector<float16_t> addVec(BASE_SIZE * DIM, 0);
    std::vector<ascend::idx_t> ids(BASE_SIZE);
    std::iota(ids.begin(), ids.end(), 0);
    auto ret = index.AddFeatures(BASE_SIZE, addVec.data(), ids.data());
    EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));

    int ntotal = index.GetNTotal();
    EXPECT_EQ(ntotal, static_cast<int>(BASE_SIZE));

    std::vector<int> queryList = {128, 64, 48, 32, 30, 18, 16, 8, 6, 4, 2, 1};
    auto iter = queryList.begin();
    while (iter != queryList.end()) {
        int queryNum = *iter;

        std::vector<float16_t> query(queryNum * DIM, 0);
        std::vector<float> dist(queryNum * TOPK, 0.0);
        std::vector<ascend::idx_t> indices(queryNum * TOPK, 0);

        ret = index.Search(queryNum, query.data(), TOPK, indices.data(), dist.data());
        EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));

        float threshold = 0.75;
        std::vector<int> num(queryNum);
        ret = index.SearchByThreshold(
            queryNum, query.data(), threshold, TOPK, num.data(), indices.data(), dist.data());
        EXPECT_EQ(ret, static_cast<int>(ErrorCode::APP_ERR_OK));

        unsigned int tableLen = 10000;
        std::vector<float> table(tableLen + TABLELEN_REDUNDANCY);
        CreateMappingTable(table.data(), tableLen + TABLELEN_REDUNDANCY);
        ret = index.SearchByThreshold(
            queryNum, query.data(), threshold, TOPK, num.data(), indices.data(), dist.data(), tableLen, table.data());

        iter++;
    }

    index.Finalize();
}

int TestComputeDistanceByIdx(ascend::IndexILFlat &index, float16_t *queryData)
{
    std::vector<int> num(QUERY_NUM, 1);
    std::vector<uint32_t> indices(QUERY_NUM);
    for (int i = 0; i < indices.size(); i++) {
        indices[i] = i;
    }
    std::vector<float> dist(QUERY_NUM);
    auto ret = index.ComputeDistanceByIdx(QUERY_NUM, queryData, num.data(), indices.data(), dist.data());
    return ret;
}

TEST(TestAscendIndexUTILFlat, test_muti_index_search)
{
    const int indexNum = 1600;
    const int base = 100;
    const int capacity = 100;
    std::vector<ascend::IndexILFlat *> indexFlats;
    aclrtSetDevice(0);

    for (int i = 0; i < indexNum; i++) {
        indexFlats.emplace_back(new ascend::IndexILFlat());
        auto ret = indexFlats[i]->Init(DIM, capacity, IP_TYPE, RESOURCE_SIZE);
        EXPECT_EQ(ret, 0);
        std::vector<float16_t> addVec(base * DIM, 0);
        std::vector<ascend::idx_t> ids(base);
        std::iota(ids.begin(), ids.end(), 0);
        ret = indexFlats[i]->AddFeatures(base, addVec.data(), ids.data());
        EXPECT_EQ(ret, 0);

        int ntotal = indexFlats[i]->GetNTotal();
        EXPECT_EQ(ntotal, static_cast<int>(base));

        std::vector<float16_t> query(QUERY_NUM * DIM, 0);
        std::vector<float> dist(QUERY_NUM * TOPK, 0.0);
        std::vector<ascend::idx_t> indices(QUERY_NUM * TOPK, 0);

        float threshold = 0.75;
        std::vector<int> num(QUERY_NUM);
        ret = indexFlats[i]->SearchByThreshold(
            QUERY_NUM, query.data(), threshold, TOPK, num.data(), indices.data(), dist.data());
        EXPECT_EQ(ret, 0);

        ret = TestComputeDistanceByIdx(*indexFlats[i], query.data());
        EXPECT_EQ(ret, 0);
    }

    for (auto index : indexFlats) {
        index->Finalize();
        delete index;
    }
    aclrtResetDevice(0);
}

} // namespace