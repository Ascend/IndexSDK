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


#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>
#include <memory>
#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include "unistd.h"
#include "AscendIndexGreat.h"
#include "stub/great/GreatMock.h"
#include "ErrorCode.h"

using namespace testing;
using namespace std;
using namespace faiss::ascend;

namespace {
    string g_mode = "KMode";
    vector<int> g_deviceList {0};
    bool g_verbose = false;
    const string K_INDEX_PATH = "/home/KMode.index";
    const string A_INDEX_PATH = "/home/AMode.index";
    const string CODE_BOOKS_PATH = "/home/CODE_BOOKS_PATH";


class TestAscendIndexUTGreat : public Test {
public:
    void SetUp() override
    {
        // 这里不直接make_shared创建智能指针是因为EXPECT_CALL会到导致智能指针循环依赖；
        // 因此智能指针不释放内存，而是在TearDown中手动释放
        GreatMock::defaultGreat = shared_ptr<GreatMock>(new GreatMock(), [] (auto) {});
        EXPECT_CALL(*GreatMock::defaultGreat, CreateIndex(g_mode, g_deviceList, g_verbose)).WillRepeatedly(
            Return(nullptr));
    }

    void TearDown() override
    {
        delete GreatMock::defaultGreat.get();
        GreatMock::defaultGreat.reset();

        GlobalMockObject::verify();
    }
};

void MockDefaultIndex()
{
    // 这里析构SetUp里new出来的对象，方便后续重新new对象mock新行为
    delete GreatMock::defaultGreat.get();
    GreatMock::defaultGreat.reset();
 
    // 这里不直接make_shared创建智能指针是因为EXPECT_CALL会到导致智能指针循环依赖；
    // 因此智能指针不释放内存，而是在TearDown中手动释放
    GreatMock::defaultGreat =
        shared_ptr<GreatMock>(new GreatMock(), [] (auto) {});
    EXPECT_CALL(*GreatMock::defaultGreat, CreateIndex(g_mode, g_deviceList, g_verbose)).WillRepeatedly(
        Return(GreatMock::defaultGreat));
}

TEST_F(TestAscendIndexUTGreat, TestLoadIndex_KMode)
{
    // 构造异常三方指针
    auto invalidIndex = std::make_unique<AscendIndexGreat>(g_mode, g_deviceList, g_verbose);
    auto ret = invalidIndex->LoadIndex(K_INDEX_PATH);
    EXPECT_NE(ret, APP_ERR_OK);
 
    // 正常流程
    MockDefaultIndex();
    IndexGreat* indexGreat = nullptr;
    EXPECT_CALL(*GreatMock::defaultGreat, LoadIndex(K_INDEX_PATH)).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexGreat>(g_mode, g_deviceList, g_verbose);
    ret = validIndex->LoadIndex(K_INDEX_PATH);
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST_F(TestAscendIndexUTGreat, TestLoadIndex_AKMode)
{
    // 构造异常三方指针
    auto invalidIndex = std::make_unique<AscendIndexGreat>(g_mode, g_deviceList, g_verbose);
    auto ret = invalidIndex->LoadIndex(A_INDEX_PATH, K_INDEX_PATH);
    EXPECT_NE(ret, APP_ERR_OK);
    // 正常流程
    MockDefaultIndex();
    IndexGreat* indexGreat = nullptr;
    EXPECT_CALL(*GreatMock::defaultGreat, LoadIndex(A_INDEX_PATH, K_INDEX_PATH)).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexGreat>(g_mode, g_deviceList, g_verbose);
    ret = validIndex->LoadIndex(A_INDEX_PATH, K_INDEX_PATH);
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST_F(TestAscendIndexUTGreat, TestWriteIndex_KMode)
{
    // 构造异常三方指针
    auto invalidIndex = std::make_unique<AscendIndexGreat>(g_mode, g_deviceList, g_verbose);
    auto ret = invalidIndex->WriteIndex(K_INDEX_PATH);
    EXPECT_NE(ret, APP_ERR_OK);
    // 正常流程
    MockDefaultIndex();
    EXPECT_CALL(*GreatMock::defaultGreat, WriteIndex(K_INDEX_PATH)).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexGreat>(g_mode, g_deviceList, g_verbose);
    ret = validIndex->WriteIndex(K_INDEX_PATH);
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST_F(TestAscendIndexUTGreat, TestWriteIndex_AKMode)
{
    // 构造异常三方指针
    auto invalidIndex = std::make_unique<AscendIndexGreat>(g_mode, g_deviceList, g_verbose);
    auto ret = invalidIndex->WriteIndex(A_INDEX_PATH, K_INDEX_PATH);
    EXPECT_NE(ret, APP_ERR_OK);
    // 正常流程
    MockDefaultIndex();
    EXPECT_CALL(*GreatMock::defaultGreat, WriteIndex(A_INDEX_PATH, K_INDEX_PATH)).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexGreat>(g_mode, g_deviceList, g_verbose);
    ret = validIndex->WriteIndex(A_INDEX_PATH, K_INDEX_PATH);
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST_F(TestAscendIndexUTGreat, TestAddCodeBooks)
{
    // 构造异常三方指针
    auto invalidIndex = std::make_unique<AscendIndexGreat>(g_mode, g_deviceList, g_verbose);
    auto ret = invalidIndex->AddCodeBooks(CODE_BOOKS_PATH);
    EXPECT_NE(ret, APP_ERR_OK);
    // 正常流程
    MockDefaultIndex();
    EXPECT_CALL(*GreatMock::defaultGreat, AddCodeBooks).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexGreat>(g_mode, g_deviceList, g_verbose);
    ret = validIndex->AddCodeBooks(CODE_BOOKS_PATH);
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST_F(TestAscendIndexUTGreat, TestAdd)
{
    // 构造异常三方指针
    vector<float> baseData {0.08};
    auto invalidIndex = std::make_unique<AscendIndexGreat>(g_mode, g_deviceList, g_verbose);
    auto ret = invalidIndex->Add(baseData);
    EXPECT_NE(ret, APP_ERR_OK);
    // 正常流程
    MockDefaultIndex();
    EXPECT_CALL(*GreatMock::defaultGreat, Add).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexGreat>(g_mode, g_deviceList, g_verbose);
    ret = validIndex->Add(baseData);
    EXPECT_EQ(ret, APP_ERR_OK);
}
 
TEST_F(TestAscendIndexUTGreat, TestAddWithIds)
{
    vector<float> baseData {0.08};
    vector<int64_t> ids {2};
    // 构造异常三方指针
    auto invalidIndex = std::make_unique<AscendIndexGreat>(g_mode, g_deviceList, g_verbose);
    auto ret = invalidIndex->AddWithIds(baseData, ids);
    EXPECT_NE(ret, APP_ERR_OK);
    // 正常流程
    MockDefaultIndex();
    EXPECT_CALL(*GreatMock::defaultGreat, AddWithIds).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexGreat>(g_mode, g_deviceList, g_verbose);
    ret = validIndex->AddWithIds(baseData, ids);
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST_F(TestAscendIndexUTGreat, TestSearch)
{
    // prepare data
    size_t n = 10;                     // 查询batch=10
    vector<float> queryData(n, 0.05f); // 查询向量，设置为10个0.5
    int topK = 2;                      // 查询结果返回最相似的2个
    vector<float> dists(n * topK);       // 查询距离结果，长度为10 * 2
    vector<int64_t> labels(n * topK);    // 查询结果label，长度为10 * 2
    AscendIndexSearchParams searchParams(n, queryData, topK, dists, labels);
    // 构造异常三方指针
    auto invalidIndex = std::make_unique<AscendIndexGreat>(g_mode, g_deviceList, g_verbose);
    auto ret = invalidIndex->Search(searchParams);
    EXPECT_NE(ret, APP_ERR_OK);
    // 正常流程
    MockDefaultIndex();
    EXPECT_CALL(*GreatMock::defaultGreat, Search).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexGreat>(g_mode, g_deviceList, g_verbose);
    ret = validIndex->Search(searchParams);
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST_F(TestAscendIndexUTGreat, TestSearchWithMask)
{
    // prepare data
    size_t n = 10;
    uint8_t maskN = 169;
    vector<float> queryData(n, 0.1f);
    int topK = 2;
    vector<float> dists(n * topK);
    vector<int64_t> labels(n * topK);
    vector<uint8_t> mask(n, maskN);
    AscendIndexSearchParams searchParams(n, queryData, topK, dists, labels);
    // 构造异常三方指针
    auto invalidIndex = std::make_unique<AscendIndexGreat>(g_mode, g_deviceList, g_verbose);
    auto ret = invalidIndex->SearchWithMask(searchParams, mask);
    EXPECT_NE(ret, APP_ERR_OK);
    // 正常流程
    MockDefaultIndex();
    EXPECT_CALL(*GreatMock::defaultGreat, SearchWithMask).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexGreat>(g_mode, g_deviceList, g_verbose);
    ret = validIndex->SearchWithMask(searchParams, mask);
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST_F(TestAscendIndexUTGreat, TestSetHyperSearchParams)
{
    int nProbeL1 = 50;
    int nProbeL2 = 22;
    int l3SegmentNum = 70;
    int ef = 150;
    AscendIndexVstarHyperParams params(nProbeL1, nProbeL2, l3SegmentNum);
    AscendIndexHyperParams hyperParams(g_mode, params, ef);
    // 构造异常三方指针
    auto invalidIndex = std::make_unique<AscendIndexGreat>(g_mode, g_deviceList, g_verbose);
    auto ret = invalidIndex->SetHyperSearchParams(hyperParams);
    EXPECT_NE(ret, APP_ERR_OK);
    // 正常流程
    MockDefaultIndex();
    EXPECT_CALL(*GreatMock::defaultGreat, SetHyperSearchParams).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexGreat>(g_mode, g_deviceList, g_verbose);
    ret = validIndex->SetHyperSearchParams(hyperParams);
    EXPECT_EQ(ret, APP_ERR_OK);
}
 
TEST_F(TestAscendIndexUTGreat, TestGetHyperSearchParams)
{
    int nProbeL1 = 50;
    int nProbeL2 = 22;
    int l3SegmentNum = 70;
    int ef = 150;
    AscendIndexVstarHyperParams params(nProbeL1, nProbeL2, l3SegmentNum);
    AscendIndexHyperParams hyperParams(g_mode, params, ef);
    // 构造异常三方指针
    auto invalidIndex = std::make_unique<AscendIndexGreat>(g_mode, g_deviceList, g_verbose);
    auto ret = invalidIndex->GetHyperSearchParams(hyperParams);
    EXPECT_NE(ret, APP_ERR_OK);
    // 正常流程
    MockDefaultIndex();
    EXPECT_CALL(*GreatMock::defaultGreat, GetHyperSearchParams).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexGreat>(g_mode, g_deviceList, g_verbose);
    ret = validIndex->GetHyperSearchParams(hyperParams);
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST_F(TestAscendIndexUTGreat, TestGetDim)
{
    int dim = 0;
    // 构造异常三方指针
    auto invalidIndex = std::make_unique<AscendIndexGreat>(g_mode, g_deviceList, g_verbose);
    auto ret = invalidIndex->GetDim(dim);
    EXPECT_NE(ret, APP_ERR_OK);
    // 正常流程
    MockDefaultIndex();
    EXPECT_CALL(*GreatMock::defaultGreat, GetDim).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexGreat>(g_mode, g_deviceList, g_verbose);
    ret = validIndex->GetDim(dim);
    EXPECT_EQ(ret, APP_ERR_OK);
}
 
TEST_F(TestAscendIndexUTGreat, TestGetNTotal)
{
    uint64_t ntotal = 0;
    // 构造异常三方指针
    auto invalidIndex = std::make_unique<AscendIndexGreat>(g_mode, g_deviceList, g_verbose);
    auto ret = invalidIndex->GetNTotal(ntotal);
    EXPECT_NE(ret, APP_ERR_OK);
    // 正常流程
    MockDefaultIndex();
    EXPECT_CALL(*GreatMock::defaultGreat, GetNTotal).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexGreat>(g_mode, g_deviceList, g_verbose);
    ret = validIndex->GetNTotal(ntotal);
    EXPECT_EQ(ret, APP_ERR_OK);
}
 
TEST_F(TestAscendIndexUTGreat, TestReset)
{
    // 构造异常三方指针
    auto invalidIndex = std::make_unique<AscendIndexGreat>(g_mode, g_deviceList, g_verbose);
    auto ret = invalidIndex->Reset();
    EXPECT_NE(ret, APP_ERR_OK);
    // 正常流程
    MockDefaultIndex();
    EXPECT_CALL(*GreatMock::defaultGreat, Reset).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexGreat>(g_mode, g_deviceList, g_verbose);
    ret = validIndex->Reset();
    EXPECT_EQ(ret, APP_ERR_OK);
}

} // namespace
