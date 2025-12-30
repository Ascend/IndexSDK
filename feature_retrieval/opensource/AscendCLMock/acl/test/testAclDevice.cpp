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

#include "gtest/gtest.h"
#include "acl.h"
#include "simu/AscendSimuLog.h"
#include "simu/AscendSimuDevice.h"

class testDevice : public ::testing::Test {
protected:
    static void SetUpTestCase()
    {
    }

    static void TearDownTestCase()
    {
    }
};

TEST_F(testDevice, CreateDefaultContextAndStream)
{
    // 创建一个device 并set 会创建一个
    auto simuDevice = AscendSimuDevice(0, 8); // 这里假设一个模拟的device存在，ID为0，并且有8个核
    simuDevice.Init(); // 初始化的时候 会默认创建一个context和stream

    EXPECT_EQ(simuDevice.m_deviceId, 0);
    EXPECT_EQ(simuDevice.m_aicore, 8);
    EXPECT_EQ(simuDevice.m_refCnt, 1);
    EXPECT_NE(simuDevice.m_defaultContext, nullptr);
    EXPECT_EQ(simuDevice.m_defaultContext, simuDevice.m_activeContexts[simuDevice.m_deviceId]);
    EXPECT_EQ(simuDevice.m_ctxts.size(), 1);
    EXPECT_EQ(simuDevice.m_ctxts[0], simuDevice.m_defaultContext);

    auto defaultStream = simuDevice.m_ctxts[0]->m_defaultStream;
    EXPECT_NE(defaultStream, nullptr);
    auto execflow = simuDevice.m_flowMap[defaultStream];
    EXPECT_EQ(execflow->isRuning(), true);

    simuDevice.DeInit();

    EXPECT_EQ(simuDevice.m_refCnt, 0);
    EXPECT_EQ(simuDevice.m_defaultContext, nullptr);
    EXPECT_EQ(simuDevice.m_activeContexts[simuDevice.m_deviceId], nullptr);
    EXPECT_EQ(simuDevice.m_ctxts.size(), 0);
    EXPECT_EQ(simuDevice.m_flowMap.size(), 0);
}