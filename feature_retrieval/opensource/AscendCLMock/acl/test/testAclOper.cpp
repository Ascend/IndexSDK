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

#include <cstring>
#include <numeric>
#include "gtest/gtest.h"
#include "acl.h"
#include "simu/AscendSimuEnv.h"
#include "simu/AscendSimuDevice.h"
#include "simu/AscendSimuLog.h"
#include "simu/AscendSimuExecFlow.h"

class testOperator : public ::testing::Test {
protected:
    static void SetUpTestCase()
    {
        // 设置模拟环境
        LOGGER().Reset();
        LOGGER().SetLogLevel(ACL_INFO); // 日志等级设为INFO
        LOGGER().SetLogFile("./logOperator.txt"); // 日志目录为log.txt

        auto *device0 = new AscendSimuDevice(0, 8); // device0 8核
        auto *device1 = new AscendSimuDevice(1, 8); // device1 8核

        ENV().construct("Ascend310P", ACL_HOST, {device0, device1}); // 模拟环境310P 两个设备 HOST
    }

    static void TearDownTestCase()
    {
        LOGGER().Reset();
        ENV().destruct();
    }
};

TEST_F(testOperator, hostMemBasic)
{
    uint64_t size = 1024;
    void *hostPtrA = nullptr;
    void *hostPtrB = nullptr;
    EXPECT_EQ(aclrtMallocHost(&hostPtrA, size), ACL_SUCCESS);
    EXPECT_EQ(aclrtMallocHost(&hostPtrB, size), ACL_SUCCESS);

    EXPECT_EQ(aclrtMemset(hostPtrA, size, 1, 1026), ACL_ERROR_STORAGE_OVER_LIMIT);
    EXPECT_EQ(aclrtMemset(hostPtrA, size, 10, 100), ACL_SUCCESS);

    for (auto i = 0; i < 100; i++) {
        EXPECT_EQ(((uint8_t *)hostPtrA)[i], 10);
    }

    EXPECT_EQ(aclrtMemcpy(hostPtrB, size, hostPtrA, size, ACL_MEMCPY_HOST_TO_HOST), ACL_SUCCESS);
    EXPECT_EQ(memcmp(hostPtrA, hostPtrB, size), 0);

    EXPECT_EQ(aclrtFreeHost(hostPtrA), ACL_SUCCESS);
    EXPECT_EQ(aclrtFreeHost(hostPtrB), ACL_SUCCESS);
}

TEST_F(testOperator, devMemBasic)
{
    uint64_t size = 1024;
    void *devPtrA = nullptr;
    void *devPtrB = nullptr;
    ENV().m_runMode = ACL_DEVICE;
    EXPECT_EQ(aclrtMalloc(&devPtrA, size, ACL_MEM_MALLOC_NORMAL_ONLY), ACL_SUCCESS);
    EXPECT_EQ(aclrtMalloc(&devPtrB, size, ACL_MEM_MALLOC_NORMAL_ONLY), ACL_SUCCESS);

    EXPECT_EQ(aclrtMemset(devPtrA, size, 1, 1026), ACL_ERROR_STORAGE_OVER_LIMIT);
    EXPECT_EQ(aclrtMemset(devPtrA, size, 10, 100), ACL_SUCCESS);

    for (auto i = 0; i < 100; i++) {
        EXPECT_EQ(((uint8_t *)devPtrA)[i], 10);
    }

    EXPECT_EQ(aclrtMemcpy(devPtrB, size, devPtrA, size, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    EXPECT_EQ(memcmp(devPtrA, devPtrB, size), 0);

    EXPECT_EQ(aclrtFreeHost(devPtrA), ACL_SUCCESS);
    EXPECT_EQ(aclrtFreeHost(devPtrB), ACL_SUCCESS);

    ENV().m_runMode = ACL_HOST;
}

TEST_F(testOperator, devAsnycCpy)
{
    EXPECT_EQ(aclInit(nullptr), ACL_SUCCESS);
    EXPECT_EQ(aclrtSetDevice(0), ACL_SUCCESS);

    void *hostPtrA = nullptr;
    void *hostPtrB = nullptr;

    uint64_t size = 1024;
    EXPECT_EQ(aclrtMallocHost(&hostPtrA, size), ACL_SUCCESS);
    EXPECT_EQ(aclrtMallocHost(&hostPtrB, size), ACL_SUCCESS);

    EXPECT_EQ(aclrtMemset(hostPtrA, size, 10, 100), ACL_SUCCESS);

    for (auto i = 0; i < 100; i++) {
        EXPECT_EQ(((uint8_t *)hostPtrA)[i], 10);
    }

    EXPECT_EQ(aclrtMemcpyAsync(hostPtrB, size, hostPtrA, size, ACL_MEMCPY_HOST_TO_DEVICE, nullptr), ACL_SUCCESS);
    EXPECT_EQ(aclrtSynchronizeStream(nullptr), ACL_SUCCESS);
    EXPECT_EQ(memcmp(hostPtrA, hostPtrB, size), 0);

    EXPECT_EQ(aclrtFreeHost(hostPtrA), ACL_SUCCESS);
    EXPECT_EQ(aclrtFreeHost(hostPtrB), ACL_SUCCESS);

    EXPECT_EQ(aclrtResetDevice(0), ACL_SUCCESS);
    EXPECT_EQ(aclFinalize(), ACL_SUCCESS);
}

// 简单模拟一维的向量内积算子
void simuInnerProductOpOneDim(aclopHandle &opHandle)
{
    int dim = opHandle.inputDesc[0].dims[0];
    auto *pvecA = reinterpret_cast<int32_t *>(opHandle.inputData[0].data);
    auto *pvecB = reinterpret_cast<int32_t *>(opHandle.inputData[1].data);
    auto output = reinterpret_cast<int32_t *>(opHandle.outputData[0].data);

    std::vector<int32_t> vecA(pvecA, pvecA + dim);
    std::vector<int32_t> vecB(pvecB, pvecB + dim);

    *output = std::inner_product(vecA.begin(), vecA.end(), vecB.begin(), 0);
}

TEST_F(testOperator, simuOpInnerProduct)
{
    EXPECT_EQ(aclInit(nullptr), ACL_SUCCESS);
    EXPECT_EQ(aclrtSetDevice(0), ACL_SUCCESS);

    // 注册算子异步回调 C = A dot B
    REG_OP("simuInnerProductOpOneDim", simuInnerProductOpOneDim);

    std::vector<int32_t> vecA {1, 2, 3, 4, 5, 6};
    std::vector<int32_t> vecB {2, 3, 4, 5, 6, 7};
    int dim = vecA.size();

    int32_t result = 0;
    aclTensorDesc inputDescA;
    inputDescA.dims = reinterpret_cast<const int64_t *>(&dim);
    inputDescA.numDims = 1;
    aclTensorDesc inputDescB;
    inputDescB.dims = reinterpret_cast<const int64_t *>(&dim);
    inputDescB.numDims = 1;
    std::vector<aclTensorDesc *> inputDesc{&inputDescA, &inputDescB};

    aclTensorDesc outputDescC;
    outputDescC.numDims = 1;
    std::vector<aclTensorDesc *> outputDesc{&outputDescC};

    aclopHandle *opHandle;
    EXPECT_EQ(aclopCreateHandle("simuInnerProductOpOneDim",
        inputDesc.size(),
        inputDesc.data(),
        outputDesc.size(),
        outputDesc.data(),
        nullptr, &opHandle), ACL_SUCCESS);

    aclDataBuffer inputBufferA;
    inputBufferA.data = vecA.data();
    inputBufferA.size = vecA.size();

    aclDataBuffer inputBufferB;
    inputBufferB.data = vecB.data();
    inputBufferB.size = vecB.size();

    aclDataBuffer outputBufferC;
    outputBufferC.data = &result;
    outputBufferC.size = 1;

    std::vector<aclDataBuffer *> inputBuffer{&inputBufferA, &inputBufferB};
    std::vector<aclDataBuffer *> outputBuffer{&outputBufferC};
    EXPECT_EQ(aclopExecWithHandle(opHandle,
        inputBuffer.size(),
        inputBuffer.data(),
        outputBuffer.size(),
        outputBuffer.data(), nullptr), ACL_SUCCESS); // 使用默认stream

    EXPECT_EQ(aclrtSynchronizeStream(nullptr), ACL_SUCCESS); // 等待计算完成

    EXPECT_EQ(result, 112); // 期望为112 (1 * 2) + (2 * 3) + (3 * 4) + (4 * 5) + (5 * 6) + (6 * 7) = 112

    UNREG_OP("simuInnerProductOpOneDim");
    aclopDestroyHandle(opHandle);

    EXPECT_EQ(aclrtResetDevice(0), ACL_SUCCESS);
    EXPECT_EQ(aclFinalize(), ACL_SUCCESS);
}