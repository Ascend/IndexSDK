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


#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <map>
#include <thread>
#include "ascendfaiss/ascenddaemon/utils/AscendOperator.h"

using namespace testing;
using namespace std;

namespace ascend {

class TestAscendOperator : public Test {
public:
    void TearDown() override
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TestAscendOperator, init_aclopCreateHandle_failed)
{
    AscendOpDesc desc("TestOp");
    AscendOperator op(desc);
    MOCKER_CPP(&aclopCreateHandle).stubs().will(returnValue(1));
    EXPECT_FALSE(op.init());
}

TEST_F(TestAscendOperator, getInputNumDims)
{
    AscendOpDesc desc("TestOp");
    AscendOperator op(desc);

    // index为-1，无效
    string actualMsg;
    int index = -1;
    try {
        op.getInputNumDims(index);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    string expectMsg("index >= 0");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);

    // index为0，实际input tensor数量为0
    index = 0;
    try {
        op.getInputNumDims(index);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    expectMsg = std::string("index < static_cast<int>");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);

    // index为0，实际input tensor数量为1，dim为2
    std::vector<int64_t> shape { 1, 1 };
    desc.addInputTensorDesc(ACL_FLOAT16, shape.size(), shape.data(), ACL_FORMAT_ND);
    AscendOperator newOp(desc);
    EXPECT_EQ(shape.size(), newOp.getInputNumDims(index));
}

TEST_F(TestAscendOperator, getInputDim)
{
    AscendOpDesc desc("TestOp");
    AscendOperator op(desc);

    // index为-1，无效
    string actualMsg;
    int index = -1;
    int dimIndex = 0;
    try {
        op.getInputDim(index, dimIndex);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    string expectMsg("index >= 0");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);

    // index为0，实际input tensor数量为0
    index = 0;
    try {
        op.getInputDim(index, dimIndex);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    expectMsg = std::string("index < static_cast<int>");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);

    // index为0，dimIndex为2，实际input tensor数量为1，dim为2
    dimIndex = 2;
    try {
        op.getInputDim(index, dimIndex);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    expectMsg = std::string("index < static_cast<int>");

    // index为0，dimIndex为1，实际input tensor数量为1，dim为2
    dimIndex = 1;
    std::vector<int64_t> shape { 1, 1 };
    desc.addInputTensorDesc(ACL_FLOAT16, shape.size(), shape.data(), ACL_FORMAT_ND);
    AscendOperator newOp(desc);
    EXPECT_EQ(shape.at(dimIndex), newOp.getInputDim(index, dimIndex));
}

TEST_F(TestAscendOperator, getInputSize)
{
    AscendOpDesc desc("TestOp");
    AscendOperator op(desc);

    // index为-1，无效
    string actualMsg;
    int index = -1;
    try {
        op.getInputSize(index);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    string expectMsg("index >= 0");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);

    // index为0，实际input tensor数量为0
    index = 0;
    try {
        op.getInputSize(index);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    expectMsg = std::string("index < static_cast<int>");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);

    // index为0，实际input tensor数量为1
    std::vector<int64_t> shape { 1, 1 };
    desc.addInputTensorDesc(ACL_FLOAT16, shape.size(), shape.data(), ACL_FORMAT_ND);
    AscendOperator newOp(desc);
    EXPECT_EQ(shape.size(), newOp.getInputSize(index));
}

TEST_F(TestAscendOperator, getOutputNumDims)
{
    AscendOpDesc desc("TestOp");
    AscendOperator op(desc);

    // index为-1，无效
    string actualMsg;
    int index = -1;
    try {
        op.getOutputNumDims(index);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    string expectMsg("index >= 0");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);

    // index为0，实际output tensor数量为0
    index = 0;
    try {
        op.getOutputNumDims(index);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    expectMsg = std::string("index < static_cast<int>");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);

    // index为0，实际output tensor数量为1
    std::vector<int64_t> shape { 1, 1 };
    desc.addOutputTensorDesc(ACL_FLOAT16, shape.size(), shape.data(), ACL_FORMAT_ND);
    AscendOperator newOp(desc);
    EXPECT_EQ(shape.size(), newOp.getOutputNumDims(index));
}

TEST_F(TestAscendOperator, getOutputDim)
{
    AscendOpDesc desc("TestOp");
    AscendOperator op(desc);

    // index为-1，无效
    string actualMsg;
    int index = -1;
    int dimIndex = 0;
    try {
        op.getOutputDim(index, dimIndex);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    string expectMsg("index >= 0");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);

    // index为0，实际output tensor数量为0
    index = 0;
    try {
        op.getOutputDim(index, dimIndex);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    expectMsg = std::string("index < static_cast<int>");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);

    // index为0，dimIndex为2，实际output tensor数量为1，dim为2
    dimIndex = 2;
    try {
        op.getOutputDim(index, dimIndex);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    expectMsg = std::string("index < static_cast<int>");

    // index为0，dimIndex为1，实际output tensor数量为1，dim为2
    dimIndex = 1;
    std::vector<int64_t> shape { 1, 1 };
    desc.addOutputTensorDesc(ACL_FLOAT16, shape.size(), shape.data(), ACL_FORMAT_ND);
    AscendOperator newOp(desc);
    EXPECT_EQ(shape.at(dimIndex), newOp.getOutputDim(index, dimIndex));
}

TEST_F(TestAscendOperator, getOutputSize)
{
    AscendOpDesc desc("TestOp");
    AscendOperator op(desc);

    // index为-1，无效
    string actualMsg;
    int index = -1;
    try {
        op.getOutputSize(index);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    string expectMsg("index >= 0");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);

    // index为0，实际output tensor数量为0
    index = 0;
    try {
        op.getOutputSize(index);
    } catch (std::exception &e) {
        actualMsg = e.what();
    }
    expectMsg = std::string("index < static_cast<int>");
    EXPECT_TRUE(actualMsg.find(expectMsg) != string::npos);

    // index为0，实际output tensor数量为1
    std::vector<int64_t> shape { 1, 1 };
    desc.addOutputTensorDesc(ACL_FLOAT16, shape.size(), shape.data(), ACL_FORMAT_ND);
    AscendOperator newOp(desc);
    EXPECT_EQ(shape.size(), newOp.getOutputSize(index));
}
} // namespace ascend