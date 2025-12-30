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
#include "ThreadSafeDistComputeOpsManager.h"

using namespace testing;
using namespace std;
using namespace faiss::ascend;

namespace ascend {

class TestThreadSafeDistComputeOpsMng : public Test {
public:
    void SetUp() override
    {
        MOCKER_CPP(&aclopExecWithHandle).stubs().will(returnValue(ACL_ERROR_NONE));
        auto &opsMng = GetThreadSafeDistComputeOpsMng();
        opsMng.initialize();

        // 构造一个默认的算子
        string opTypeName = "TestOps";
        IndexTypeIdx indexType = IndexTypeIdx::ITI_FLAT_IP;
        vector<int> opsKeys({ 64, 8 });
        OpsMngKey opsKey(opsKeys);
        vector<int64_t> shape;
        vector<pair<aclDataType, vector<int64_t>>> input { { ACL_FLOAT16, shape } };
        vector<pair<aclDataType, vector<int64_t>>> output { { ACL_FLOAT16, shape } };
        auto ret = opsMng.resetOp(opTypeName, indexType, opsKey, input, output);
        EXPECT_EQ(ret, APP_ERR_OK);
        EXPECT_FALSE(opsMng.getDistComputeOps(indexType).empty());
    }

    void TearDown() override
    {
        auto &opsMng = GetThreadSafeDistComputeOpsMng();
        opsMng.uninitialize();
        GlobalMockObject::verify();
    }

    DistComputeOpsManager &GetThreadSafeDistComputeOpsMng()
    {
        static ThreadSafeDistComputeOpsManager instance;
        return instance;
    }
};

TEST_F(TestThreadSafeDistComputeOpsMng, run_op_with_invalid_index_type)
{
    auto &opsMng = GetThreadSafeDistComputeOpsMng();

    IndexTypeIdx invalidIndexType = IndexTypeIdx::ITI_MAX;
    vector<int> opsKeys({ 64, 8 });
    OpsMngKey opsKey(opsKeys);
    vector<const AscendTensorBase *> input;
    vector<const AscendTensorBase *> output;
    aclrtStream stream { 0 };
    auto ret = opsMng.runOp(invalidIndexType, opsKey, input, output, stream);
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);
}

TEST_F(TestThreadSafeDistComputeOpsMng, run_op_with_invalid_ops_key)
{
    auto &opsMng = GetThreadSafeDistComputeOpsMng();

    IndexTypeIdx indexType = IndexTypeIdx::ITI_FLAT_IP;
    vector<int> invalidOpsKeys({ 8, 64 });
    OpsMngKey invalidOpsKey(invalidOpsKeys);
    vector<const AscendTensorBase *> input;
    vector<const AscendTensorBase *> output;
    aclrtStream stream { 0 };
    auto ret = opsMng.runOp(indexType, invalidOpsKey, input, output, stream);
    EXPECT_EQ(ret, APP_ERR_ACL_OP_NOT_FOUND);
}

TEST_F(TestThreadSafeDistComputeOpsMng, run_op_success)
{
    auto &opsMng = GetThreadSafeDistComputeOpsMng();

    IndexTypeIdx indexType = IndexTypeIdx::ITI_FLAT_IP;
    vector<int> opsKeys({ 64, 8 });
    OpsMngKey opsKey(opsKeys);
    AscendTensor<uint16_t, DIMS_2> tensor1({ 1, 1 });
    AscendTensor<uint16_t, DIMS_2> tensor2({ 2, 2 });
    vector<const AscendTensorBase *> input { &tensor1 };
    vector<const AscendTensorBase *> output { &tensor2 };
    aclrtStream stream { 0 };
    auto ret = opsMng.runOp(indexType, opsKey, input, output, stream);
    EXPECT_EQ(ret, APP_ERR_OK);
    ret = opsMng.runOp(indexType, opsKey, input, output, stream);
    EXPECT_EQ(ret, APP_ERR_OK);
}

// 典型使用场景：每个index先调用resetOp再调用runOp
// 这里如果将IsMultiThreadMode的返回值设为false，使用非线程安全的DistComputeOpsManager，则多次运行用例会概率性报错
// 考虑到是概率性，且需要执行多次影响门禁，这里不再添加用例看护
TEST_F(TestThreadSafeDistComputeOpsMng, reset_op_and_run_op_in_multi_thread)
{
    auto &opsMng = GetThreadSafeDistComputeOpsMng();

    string opTypeName = "MultiThreadOps";
    IndexTypeIdx indexType = IndexTypeIdx::ITI_INT8_L2_FULL_MASK;
    vector<int> opsKeys({ 128, 64 });
    OpsMngKey opsKey(opsKeys);
    vector<pair<aclDataType, vector<int64_t>>> resetInput { make_pair<>(ACL_FLOAT16, vector<int64_t>()) };
    vector<pair<aclDataType, vector<int64_t>>> resetOutput { make_pair<>(ACL_FLOAT16, vector<int64_t>()) };
    AscendTensor<uint16_t, DIMS_2> tensor1({ 1, 1 });
    AscendTensor<uint16_t, DIMS_2> tensor2({ 2, 2 });
    vector<const AscendTensorBase *> runInput { &tensor1 };
    vector<const AscendTensorBase *> runOutput { &tensor2 };
    aclrtStream stream { 0 };

    // index1先reset算子
    auto ret = opsMng.resetOp(opTypeName, indexType, opsKey, resetInput, resetOutput);
    EXPECT_EQ(ret, APP_ERR_OK);
    // 其他index继续reset算子、运行算子
    auto indexRestOp = [&opsMng, &opTypeName, &indexType, &opsKey, &resetInput, &resetOutput] () {
        auto opRet = opsMng.resetOp(opTypeName, indexType, opsKey, resetInput, resetOutput);
        EXPECT_EQ(opRet, APP_ERR_OK);
    };
    // index1再reset算子
    auto indexRunOp = [&opsMng, &indexType, &opsKey, &runInput, &runOutput, &stream] () {
        auto opRet = opsMng.runOp(indexType, opsKey, runInput, runOutput, stream);
        EXPECT_EQ(opRet, APP_ERR_OK);
    };
    uint32_t threadNum = 50;
    vector<thread> threads;
    for (uint32_t i = 0; i < threadNum; i++) {
        if (i % 2 == 0) { // 2分之1的index在reset算子
            threads.emplace_back(thread(indexRestOp));
        } else {
            threads.emplace_back(thread(indexRunOp));
        }
    }

    for (auto &t : threads) {
        t.join();
    }
}

} // namespace ascend
