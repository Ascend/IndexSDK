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
#include "gtest/gtest.h"
#include "acl.h"
#include "simu/AscendSimuLog.h"
#include "simu/AscendSimuEnv.h"

class testEnv : public ::testing::Test {
protected:
    static void SetUpTestCase()
    {
        auto *device0 = new AscendSimuDevice(0, 8); // device0 8核
        auto *device1 = new AscendSimuDevice(1, 8); // device1 6核
        auto *device2 = new AscendSimuDevice(2, 8); // device1 6核
        auto *device3 = new AscendSimuDevice(3, 8); // device1 6核

        ENV().construct("Ascend310P", ACL_HOST, {device0, device1, device2, device3}); // 模拟环境310P 两个设备 HOST
    }

    static void TearDownTestCase() {}
};

extern "C" void AscendEnvInit()
{
}

extern "C" void AscendEnvFinalize()
{
}

TEST_F(testEnv, CreateSimuEnviorment1)
{
    EXPECT_EQ(aclInit(nullptr), ACL_SUCCESS);
    for (auto i = 0u; i < 4; i++) {
        aclrtSetDevice(i);
    }

    for (auto i = 0u; i < 4; i++) {
        aclrtResetDevice(i);
    }
    EXPECT_EQ(aclFinalize(), ACL_SUCCESS);
}


TEST_F(testEnv, CreateSimuEnviorment)
{
    aclrtRunMode runmode;
    EXPECT_EQ(aclInit(nullptr), ACL_SUCCESS);
    EXPECT_EQ(aclrtGetRunMode(&runmode), ACL_SUCCESS);
    EXPECT_EQ(runmode, ACL_HOST);

    EXPECT_EQ(ENV().getDeviceCount(), 4);
    EXPECT_EQ(aclrtGetSocName(), "Ascend310P");

    aclrtSetDevice(0);
    int32_t deviceId;
    EXPECT_EQ(aclrtGetDevice(&deviceId), ACL_SUCCESS);
    EXPECT_EQ(deviceId, 0);
    aclrtResetDevice(0);

    aclrtSetDevice(1);
    EXPECT_EQ(aclrtGetDevice(&deviceId), ACL_SUCCESS);
    EXPECT_EQ(deviceId, 1);
    aclrtResetDevice(1);
    EXPECT_EQ(aclFinalize(), ACL_SUCCESS);
}

TEST_F(testEnv, CreateContextAndStream)
{
    auto device0 = ENV().getDevice(0);
    auto device1 = ENV().getDevice(1);
    EXPECT_NE(device0, nullptr);
    EXPECT_NE(device1, nullptr);

    // 添加
    EXPECT_EQ(aclInit(nullptr), ACL_SUCCESS);
    EXPECT_EQ(DEVICE(0)->GetRefCnt(), 0);
    EXPECT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
    EXPECT_EQ(DEVICE(0)->GetRefCnt(), 1);

    aclrtContext ctx1;
    EXPECT_EQ(aclrtCreateContext(&ctx1, 0), ACL_SUCCESS);
    EXPECT_EQ(device0->GetActiveContext(), ctx1); // 此时device0的active context已经切换为ctx1
    EXPECT_NE(device1->GetActiveContext(), ctx1); // 此时device1的active context应该还是default ctx

    aclrtStream s1;
    EXPECT_EQ(aclrtCreateStream(&s1), ACL_SUCCESS); // 此时应该在device0 ctxt1上创建s1
    EXPECT_EQ(s1->deviceId, 0);
    EXPECT_EQ(s1->ctxt, ctx1);
    EXPECT_EQ(ctx1->streams.size(), 2);
    EXPECT_EQ(ctx1->streams[0], ctx1->m_defaultStream);
    EXPECT_EQ(ctx1->streams[1], s1);

    aclrtContext ctx2;
    EXPECT_EQ(DEVICE(1)->GetRefCnt(), 0);
    EXPECT_EQ(aclrtCreateContext(&ctx2, 1), ACL_SUCCESS); // 此时这个线程切换到device1的ctx2上
    EXPECT_EQ(DEVICE(1)->GetRefCnt(), 1);
    EXPECT_EQ(ENV().getActiveDeviceId(), 1); // 此时激活device应该切换为device1
    EXPECT_EQ(device0->GetActiveContext(), ctx1); // device0的激活ctx 依然是ctx1
    EXPECT_EQ(device1->GetActiveContext(), ctx2); // device1的激活ctx 改为ctx2

    aclrtStream s2;
    EXPECT_EQ(aclrtCreateStream(&s2), ACL_SUCCESS); // 在ctx2 device1上创建s2
    EXPECT_EQ(s2->deviceId, 1);
    EXPECT_EQ(s2->ctxt, ctx2);
    EXPECT_EQ(ctx2->streams.size(), 2);
    EXPECT_EQ(ctx2->streams[0], ctx2->m_defaultStream);
    EXPECT_EQ(ctx2->streams[1], s2);

    // 切换回ctx1
    EXPECT_EQ(aclrtSetCurrentContext(ctx1), ACL_SUCCESS);
    EXPECT_EQ(ENV().getActiveDeviceId(), 0); // ctx1绑定device0
    EXPECT_EQ(device0->GetActiveContext(), ctx1); // device0 激活ctx为ctx1

    // 删除 s2
    EXPECT_EQ(aclrtDestroyStream(s2), ACL_SUCCESS);
    EXPECT_EQ(ctx2->streams.size(), 1);
    EXPECT_EQ(std::find(ctx2->streams.cbegin(), ctx2->streams.cend(), s2), ctx2->streams.cend());
    EXPECT_EQ(device1->StreamIsRunning(s2), false);

    // 删除ctx2
    EXPECT_EQ(aclrtDestroyContext(ctx2), ACL_SUCCESS);
    EXPECT_EQ(device1->GetActiveContext(), nullptr);

    // 删除s1
    EXPECT_EQ(aclrtDestroyStream(s1), ACL_SUCCESS);
    EXPECT_EQ(ctx1->streams.size(), 1);
    EXPECT_EQ(std::find(ctx1->streams.cbegin(), ctx1->streams.cend(), s1), ctx1->streams.cend());
    EXPECT_EQ(device0->StreamIsRunning(s1), false);

    // 删除ctx1
    EXPECT_EQ(aclrtDestroyContext(ctx1), ACL_SUCCESS);
    EXPECT_EQ(device0->GetActiveContext(), nullptr);

    EXPECT_EQ(aclrtResetDevice(0), ACL_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(1), ACL_SUCCESS);
    EXPECT_EQ(DEVICE(0)->GetRefCnt(), 0);
    EXPECT_EQ(DEVICE(1)->GetRefCnt(), 0);
    EXPECT_EQ(ENV().getActiveDeviceId(), -1);

    EXPECT_EQ(aclFinalize(), ACL_SUCCESS);
}

TEST_F(testEnv, SynchronizeStream)
{
    // Initialize
    EXPECT_EQ(aclInit(nullptr), ACL_SUCCESS);
    EXPECT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
    EXPECT_EQ(ENV().getActiveDeviceId(), 0);

    std::vector<aclrtStream> streams;
    for (auto i = 0u ; i < 2; i++) {
        aclrtStream stream = nullptr;
        aclrtCreateStream(&stream); // 这里使用默认context创建stream
        streams.push_back(stream);
    }

    EXPECT_EQ(DEVICE(0)->GetRefCnt(), 1);
    EXPECT_EQ(DEVICE(0)->m_defaultContext->streams.size(), 3); // default stream + steam * 2
    EXPECT_EQ(DEVICE(0)->m_defaultContext->streams[1], streams[0]);
    EXPECT_EQ(DEVICE(0)->m_defaultContext->streams[2], streams[1]);

    for (auto i = 0u; i < 2; i++) {
        EXPECT_EQ(aclrtSynchronizeStream(streams[i]), ACL_SUCCESS);
        EXPECT_EQ(aclrtDestroyStream(streams[i]), ACL_SUCCESS);
    }
    streams.clear();

    EXPECT_EQ(DEVICE(0)->m_defaultContext->streams.size(), 1);

    EXPECT_EQ(aclrtResetDevice(0), ACL_SUCCESS);
    EXPECT_EQ(DEVICE(0)->GetRefCnt(), 0);

    EXPECT_EQ(aclFinalize(), ACL_SUCCESS);
}
