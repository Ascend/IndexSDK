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
#include <thread>
#include <limits>
#include "ock/tools/topo/TopoDetectParamVerify.h"

namespace ock {
namespace tools {
namespace topo {
namespace tests {

class TestTopoDetectParamVerify : public testing::Test {
public:
    void SetUp(void) override
    {
        param = std::make_shared<TopoDetectParam>();
        deviceIdA = {0};
        deviceIdB = {1};
        validCpuIdA = {1};
        validCpuIdB = {8};
        invalidCpuId = {1024};
        param->SetModel(DetectModel::PARALLEL);
        AddDevice(deviceIdA, validCpuIdA, validCpuIdB);
    }
    void TearDown(void) override
    {}

    void AddDevice(hmm::OckHmmDeviceId devId, uint32_t startId, uint32_t endId)
    {
        auto devInfo = DeviceCpuSet(devId);
        devInfo.cpuIds.push_back(CpuIdRange(startId, endId));
        param->AppendDeviceInfo(devInfo);
    }
    void CheckExpectSuccess(void)
    {
        auto ret = TopoDetectParamVerify::CheckAll(*param);
        EXPECT_EQ(ret->retCode, hmm::HMM_SUCCESS);
        EXPECT_TRUE(ret->msg.empty());
    }
    void CheckExpectFailed(void)
    {
        auto ret = TopoDetectParamVerify::CheckAll(*param);
        EXPECT_NE(ret->retCode, hmm::HMM_SUCCESS);
        EXPECT_FALSE(ret->msg.empty());
    }
    std::shared_ptr<TopoDetectParam> param;
    hmm::OckHmmDeviceId deviceIdA;
    hmm::OckHmmDeviceId deviceIdB;
    uint32_t invalidCpuId;
    uint32_t validCpuIdA;
    uint32_t validCpuIdB;
};

TEST_F(TestTopoDetectParamVerify, check_mode)
{
    param->SetModel(DetectModel::PARALLEL);
    CheckExpectSuccess();
    param->SetModel(DetectModel::SERIAL);
    CheckExpectSuccess();
    param->SetModel(DetectModel::UNKNOWN);
    CheckExpectFailed();
}
TEST_F(TestTopoDetectParamVerify, validCpuId)
{
    AddDevice(validCpuIdB, validCpuIdA, validCpuIdB);
    CheckExpectSuccess();
}
TEST_F(TestTopoDetectParamVerify, end_invalidCpuId)
{
    AddDevice(validCpuIdB, validCpuIdA, invalidCpuId);
    CheckExpectFailed();
}
TEST_F(TestTopoDetectParamVerify, start_invalidCpuId)
{
    AddDevice(validCpuIdB, invalidCpuId, invalidCpuId);
    CheckExpectFailed();
}
TEST_F(TestTopoDetectParamVerify, invalid_order_cpuId)
{
    AddDevice(validCpuIdB, validCpuIdB, validCpuIdA);
    CheckExpectFailed();
}
TEST_F(TestTopoDetectParamVerify, invalid_cpuId_equal_cpucout)
{
    AddDevice(validCpuIdB, std::thread::hardware_concurrency(), std::thread::hardware_concurrency());
    CheckExpectFailed();
}
TEST_F(TestTopoDetectParamVerify, invalid_cpuId_max_int_value)
{
    AddDevice(validCpuIdB, validCpuIdA, std::numeric_limits<int>::max() + validCpuIdB);
    CheckExpectFailed();
}
TEST_F(TestTopoDetectParamVerify, threadNumPerDevice)
{
    param->ThreadNumPerDevice(conf::OckSysConf::ToolConf().threadNumPerDevice.minValue - 1);
    CheckExpectFailed();
    param->ThreadNumPerDevice(conf::OckSysConf::ToolConf().threadNumPerDevice.minValue);
    CheckExpectSuccess();
    param->ThreadNumPerDevice(conf::OckSysConf::ToolConf().threadNumPerDevice.maxValue + 1);
    CheckExpectFailed();
    param->ThreadNumPerDevice(conf::OckSysConf::ToolConf().threadNumPerDevice.maxValue);
    CheckExpectSuccess();
}
TEST_F(TestTopoDetectParamVerify, packageNum)
{
    param->TestTime(conf::OckSysConf::ToolConf().testTime.minValue - 1);
    CheckExpectFailed();
    param->TestTime(conf::OckSysConf::ToolConf().testTime.minValue);
    CheckExpectSuccess();
    param->TestTime(conf::OckSysConf::ToolConf().testTime.maxValue + 1);
    CheckExpectFailed();
    param->TestTime(conf::OckSysConf::ToolConf().testTime.maxValue);
    CheckExpectSuccess();
}
TEST_F(TestTopoDetectParamVerify, packageBytesPerTransfer)
{
    param->PackageBytesPerTransfer(conf::OckSysConf::ToolConf().packagePerTransfer.minValue - 1);
    CheckExpectFailed();
    param->PackageBytesPerTransfer(conf::OckSysConf::ToolConf().packagePerTransfer.minValue);
    CheckExpectSuccess();
    param->PackageBytesPerTransfer(conf::OckSysConf::ToolConf().packagePerTransfer.maxValue + 1);
    CheckExpectFailed();
    param->PackageBytesPerTransfer(conf::OckSysConf::ToolConf().packagePerTransfer.maxValue);
    CheckExpectSuccess();
}
}  // namespace tests
}  // namespace topo
}  // namespace tools
}  // namespace ock