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
#include "AscendIndexVStar.h"
#include "stub/vstar/VstarMock.h"

using namespace testing;
using namespace std;
using namespace faiss::ascend;

namespace  {

const vector<int> DEVICE_LIST {0};
const int APP_ERR_OK = 0;

class TestAscendIndexUTVStar : public Test {
public:
    void SetUp() override
    {
        // 这里不直接make_shared创建智能指针是因为EXPECT_CALL会到导致智能指针循环依赖；
        // 因此智能指针不释放内存，而是在TearDown中手动释放
        VstarMock::defaultIndex =
            shared_ptr<VstarMock>(new VstarMock(), [] (auto) {});
        EXPECT_CALL(*VstarMock::defaultIndex, CreateIndex(DEVICE_LIST, false)).WillRepeatedly(
            Return(nullptr));
    }

    void TearDown() override
    {
        delete VstarMock::defaultIndex.get();
        VstarMock::defaultIndex.reset();

        GlobalMockObject::verify();
    }
};

void MockDefaultIndex()
{
    // 这里析构SetUp里new出来的对象，方便后续重新new对象mock新行为
    delete VstarMock::defaultIndex.get();
    VstarMock::defaultIndex.reset();

    // 这里不直接make_shared创建智能指针是因为EXPECT_CALL会到导致智能指针循环依赖；
    // 因此智能指针不释放内存，而是在TearDown中手动释放
    VstarMock::defaultIndex =
        shared_ptr<VstarMock>(new VstarMock(), [] (auto) {});
    EXPECT_CALL(*VstarMock::defaultIndex, CreateIndex(DEVICE_LIST, false)).WillRepeatedly(
        Return(VstarMock::defaultIndex));
}

TEST_F(TestAscendIndexUTVStar, TestLoadIndex)
{
    const string indexPath = "/home/index.index";

    // 构造异常三方指针
    auto invalidIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    auto ret = invalidIndex->LoadIndex(indexPath);
    EXPECT_NE(ret, APP_ERR_OK);

    // 正常流程
    MockDefaultIndex();
    IndexVStar* indexVstar = nullptr;
    EXPECT_CALL(*VstarMock::defaultIndex, LoadIndex(indexPath, indexVstar)).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    ret = validIndex->LoadIndex(indexPath);
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST_F(TestAscendIndexUTVStar, TestWriteIndex)
{
    const string indexPath = "/home/index.index";

    // 构造异常三方指针
    auto invalidIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    auto ret = invalidIndex->WriteIndex(indexPath);
    EXPECT_NE(ret, APP_ERR_OK);

    // 正常流程
    MockDefaultIndex();
    EXPECT_CALL(*VstarMock::defaultIndex, WriteIndex(indexPath)).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    ret = validIndex->WriteIndex(indexPath);
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST_F(TestAscendIndexUTVStar, TestAddCodeBooksByIndex)
{
    // 异常三方指针
    auto invalidIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    auto ret = invalidIndex->AddCodeBooksByIndex(*invalidIndex);
    EXPECT_NE(ret, APP_ERR_OK);

    // 正常流程
    MockDefaultIndex();
    EXPECT_CALL(*VstarMock::defaultIndex, AddCodeBooksByIndex).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    ret = validIndex->AddCodeBooksByIndex(*validIndex);
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST_F(TestAscendIndexUTVStar, TestAddCodeBooksByPath)
{
    const string codeBooksPath = "/home/codeBooks";

    // 异常三方指针
    auto invalidIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    auto ret = invalidIndex->AddCodeBooksByPath(codeBooksPath);
    EXPECT_NE(ret, APP_ERR_OK);

    // 正常流程
    MockDefaultIndex();
    EXPECT_CALL(*VstarMock::defaultIndex, AddCodeBooksByPath).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    ret = validIndex->AddCodeBooksByPath(codeBooksPath);
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST_F(TestAscendIndexUTVStar, TestAdd)
{
    vector<float> baseData {0.3};  // 仅测试流程是否跑到，底库数值设置为一个0.3

    auto invalidIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    auto ret = invalidIndex->Add(baseData);
    EXPECT_NE(ret, APP_ERR_OK);

    // 正常流程
    MockDefaultIndex();
    EXPECT_CALL(*VstarMock::defaultIndex, Add).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    ret = validIndex->Add(baseData);
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST_F(TestAscendIndexUTVStar, TestAddWithIds)
{
    vector<float> baseData {0.3};  // 仅测试流程是否跑到，底库数值设置为一个0.3
    vector<int64_t> ids {5};       // 仅测试流程是否跑到，底库对应id设置为5

    auto invalidIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    auto ret = invalidIndex->AddWithIds(baseData, ids);
    EXPECT_NE(ret, APP_ERR_OK);

    // 正常流程
    MockDefaultIndex();
    EXPECT_CALL(*VstarMock::defaultIndex, AddWithIds).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    ret = validIndex->AddWithIds(baseData, ids);
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST_F(TestAscendIndexUTVStar, TestDeleteByIds)
{
    vector<int64_t> ids {5};       // 仅测试流程是否跑到，底库删除id设置为5

    auto invalidIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    auto ret = invalidIndex->DeleteByIds(ids);
    EXPECT_NE(ret, APP_ERR_OK);

    // 正常流程
    MockDefaultIndex();
    EXPECT_CALL(*VstarMock::defaultIndex, DeleteByIds).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    ret = validIndex->DeleteByIds(ids);
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST_F(TestAscendIndexUTVStar, TestDeleteById)
{
    int64_t id {5};  // 仅测试流程是否跑到，底库删除id设置为5

    auto invalidIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    auto ret = invalidIndex->DeleteById(id);
    EXPECT_NE(ret, APP_ERR_OK);

    // 正常流程
    MockDefaultIndex();
    EXPECT_CALL(*VstarMock::defaultIndex, DeleteById).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    ret = validIndex->DeleteById(id);
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST_F(TestAscendIndexUTVStar, TestDeleteByRange)
{
    int64_t startId {5};  // 仅测试流程是否跑到，底库删除起始id设置为5
    int64_t endId {16};   // // 仅测试流程是否跑到，底库删除结束id设置为16

    auto invalidIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    auto ret = invalidIndex->DeleteByRange(startId, endId);
    EXPECT_NE(ret, APP_ERR_OK);

    // 正常流程
    MockDefaultIndex();
    EXPECT_CALL(*VstarMock::defaultIndex, DeleteByRange).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    ret = validIndex->DeleteByRange(startId, endId);
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST_F(TestAscendIndexUTVStar, TestSearch)
{
    // prepare data
    size_t n = 10;                     // 查询batch=10
    vector<float> queryData(10, 0.1);  // 查询向量，设置为10个0.1
    int topK = 5;                      // 查询结果返回最相似的5个
    vector<float> dists(20 * 5);       // 查询距离结果，长度为20 * 5
    vector<int64_t> labels(20 * 5);    // 查询结果label，长度为20 * 5
    AscendIndexSearchParams searchParams(n, queryData, topK, dists, labels);

    auto invalidIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    auto ret = invalidIndex->Search(searchParams);
    EXPECT_NE(ret, APP_ERR_OK);

    // 正常流程
    MockDefaultIndex();
    EXPECT_CALL(*VstarMock::defaultIndex, Search).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    ret = validIndex->Search(searchParams);
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST_F(TestAscendIndexUTVStar, TestSearchWithMask)
{
    // prepare data
    size_t n = 10;                     // 查询batch=10
    vector<float> queryData(10, 0.1);  // 查询向量，设置为10个0.1
    int topK = 5;                      // 查询结果返回最相似的5个
    vector<float> dists(20 * 5);       // 查询距离结果，长度为20 * 5
    vector<int64_t> labels(20 * 5);    // 查询结果label，长度为20 * 5
    vector<uint8_t> mask(10, 177);     // mask掩码，设置为10个177
    AscendIndexSearchParams searchParams(n, queryData, topK, dists, labels);

    auto invalidIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    auto ret = invalidIndex->SearchWithMask(searchParams, mask);
    EXPECT_NE(ret, APP_ERR_OK);

    // 正常流程
    MockDefaultIndex();
    EXPECT_CALL(*VstarMock::defaultIndex, SearchWithMask).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    ret = validIndex->SearchWithMask(searchParams, mask);
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST_F(TestAscendIndexUTVStar, TestMultiSearch)
{
    // prepare data
    size_t n = 10;                     // 查询batch=10
    vector<float> queryData(10, 0.1);  // 查询向量，设置为10个0.1
    int topK = 5;                      // 查询结果返回最相似的5个
    vector<float> dists(20 * 5);       // 查询距离结果，长度为20 * 5
    vector<int64_t> labels(20 * 5);    // 查询结果label，长度为20 * 5
    vector<AscendIndexVStar*> indexes(1);  // 多index检索，设置为1个
    AscendIndexSearchParams searchParams(n, queryData, topK, dists, labels);

    auto invalidIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    auto ret = invalidIndex->MultiSearch(indexes, searchParams, true);
    EXPECT_NE(ret, APP_ERR_OK);

    MockDefaultIndex();
    EXPECT_CALL(*VstarMock::defaultIndex, MultiSearch).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    indexes[0] = validIndex.get();
    ret = validIndex->MultiSearch(indexes, searchParams, true);
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST_F(TestAscendIndexUTVStar, TestMultiSearchWithMask)
{
    // prepare data
    size_t n = 10;                     // 查询batch=10
    vector<float> queryData(10, 0.1);  // 查询向量，设置为10个0.1
    int topK = 5;                      // 查询结果返回最相似的5个
    vector<float> dists(20 * 5);       // 查询距离结果，长度为20 * 5
    vector<int64_t> labels(20 * 5);    // 查询结果label，长度为20 * 5
    vector<uint8_t> mask(10, 177);     // mask掩码，设置为10个177
    vector<AscendIndexVStar*> indexes(1);  // 多index检索，设置为1个
    AscendIndexSearchParams searchParams(n, queryData, topK, dists, labels);

    auto invalidIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    auto ret = invalidIndex->MultiSearchWithMask(indexes, searchParams, mask, true);
    EXPECT_NE(ret, APP_ERR_OK);

    // 正常流程
    MockDefaultIndex();
    EXPECT_CALL(*VstarMock::defaultIndex, MultiSearchWithMask).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    indexes[0] = validIndex.get();
    ret = validIndex->MultiSearchWithMask(indexes, searchParams, mask, true);
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST_F(TestAscendIndexUTVStar, TestSetHyperSearchParams)
{
    // 检索超参设置，nProbeL1=30, nProbeL2=25, l3SegmentNum=70
    AscendIndexVstarHyperParams params(30, 25, 70);

    auto invalidIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    auto ret = invalidIndex->SetHyperSearchParams(params);
    EXPECT_NE(ret, APP_ERR_OK);

    // 正常流程
    MockDefaultIndex();
    EXPECT_CALL(*VstarMock::defaultIndex, SetHyperSearchParams).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    ret = validIndex->SetHyperSearchParams(params);
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST_F(TestAscendIndexUTVStar, TestGetHyperSearchParams)
{
    // 检索超参设置，nProbeL1=30, nProbeL2=25, l3SegmentNum=70
    AscendIndexVstarHyperParams params(30, 25, 70);
    auto invalidIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    auto ret = invalidIndex->GetHyperSearchParams(params);
    EXPECT_NE(ret, APP_ERR_OK);

    // 正常流程
    MockDefaultIndex();
    EXPECT_CALL(*VstarMock::defaultIndex, GetHyperSearchParams).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    ret = validIndex->GetHyperSearchParams(params);
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST_F(TestAscendIndexUTVStar, TestGetDim)
{
    auto invalidIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    int dim = 0;
    auto ret = invalidIndex->GetDim(dim);
    EXPECT_NE(ret, APP_ERR_OK);

    // 正常流程
    MockDefaultIndex();
    EXPECT_CALL(*VstarMock::defaultIndex, GetDim).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    ret = validIndex->GetDim(dim);
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST_F(TestAscendIndexUTVStar, TestGetNTotal)
{
    auto invalidIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    uint64_t ntotal = 0;
    auto ret = invalidIndex->GetNTotal(ntotal);

    // 正常流程
    MockDefaultIndex();
    EXPECT_CALL(*VstarMock::defaultIndex, GetNTotal).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    ret = validIndex->GetNTotal(ntotal);
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST_F(TestAscendIndexUTVStar, TestReset)
{
    auto invalidIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    auto ret = invalidIndex->Reset();
    EXPECT_NE(ret, APP_ERR_OK);

    // 正常流程
    MockDefaultIndex();
    EXPECT_CALL(*VstarMock::defaultIndex, Reset).Times(1).WillOnce(Return(APP_ERR_OK));
    auto validIndex = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    ret = validIndex->Reset();
    EXPECT_EQ(ret, APP_ERR_OK);
}

TEST_F(TestAscendIndexUTVStar, TestMultiSearchErrorParam)
{
    MockDefaultIndex();

    auto index = std::make_unique<AscendIndexVStar>(DEVICE_LIST);
    // prepare data
    size_t n = 10;                     // 查询batch=10
    vector<float> queryData(10, 0.1);  // 查询向量，设置为10个0.1
    int topK = 5;                      // 查询结果返回最相似的5个
    vector<float> dists(20 * 5);       // 查询距离结果，长度为20 * 5
    vector<int64_t> labels(20 * 5);    // 查询结果label，长度为20 * 5
    vector<uint8_t> mask(10, 177);     // mask掩码，设置为10个177
    vector<AscendIndexVStar*> indexes(10, nullptr);  // 多index检索，设置为10个nullptr
    AscendIndexSearchParams searchParams(n, queryData, topK, dists, labels);

    // null param
    auto ret = index->MultiSearch(indexes, searchParams, true);
    EXPECT_NE(ret, APP_ERR_OK);

    ret = index->MultiSearchWithMask(indexes, searchParams, mask, true);
    EXPECT_NE(ret, APP_ERR_OK);

    // indexes 个数异常
    vector<AscendIndexVStar*> zeroIndexes;  // 0个
    ret = index->MultiSearch(zeroIndexes, searchParams, true);
    EXPECT_NE(ret, APP_ERR_OK);

    vector<AscendIndexVStar*> overIndexes(10001, nullptr);  // 10001个（大于10000）
    ret = index->MultiSearchWithMask(overIndexes, searchParams, mask, true);
    EXPECT_NE(ret, APP_ERR_OK);
}

} // namespace