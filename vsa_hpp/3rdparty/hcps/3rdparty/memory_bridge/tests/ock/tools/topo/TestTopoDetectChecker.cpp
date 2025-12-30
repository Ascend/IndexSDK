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

#include <ostream>
#include <cstdint>
#include <memory>
#include <string>
#include <chrono>
#include <thread>
#include <gtest/gtest.h>
#include "ock/utils/StrUtils.h"
#include "ock/conf/OckSysConf.h"
#include "ock/tools/topo/TopoDetectChecker.h"
#include "ock/acladapter/WithEnvAclMock.h"

namespace ock {
namespace tools {
namespace topo {
namespace tests {

class TestTopoDetectChecker : public acladapter::WithEnvAclMock<testing::Test> {
public:
    void SetUp(void) override
    {
        acladapter::WithEnvAclMock<testing::Test>::SetUp();
        // 设置deviceId不连续时的测试场景
        deviceIdA = {0};
        deviceIdB = {3};
    }
    void TearDown(void) override
    {
        acladapter::WithEnvAclMock<testing::Test>::TearDown();
    }
    std::shared_ptr<TopoDetectParam> BuildParam(hmm::OckHmmDeviceId devId, DetectModel model)
    {
        auto ret = std::make_shared<TopoDetectParam>();
        ret->SetModel(model);
        ret->AppendDeviceInfo(DeviceCpuSet(devId));
        ret->PackageMBytesPerTransfer(conf::OckSysConf::ToolConf().packageMBPerTransfer.minValue);
        ret->TestTime(conf::OckSysConf::ToolConf().testTime.minValue);
        return ret;
    }
    std::shared_ptr<TopoDetectParam> BuildParam(
        hmm::OckHmmDeviceId devIdA, hmm::OckHmmDeviceId devIdB, DetectModel model)
    {
        auto ret = std::make_shared<TopoDetectParam>();
        ret->SetModel(model);
        ret->AppendDeviceInfo(DeviceCpuSet(devIdA));
        ret->AppendDeviceInfo(DeviceCpuSet(devIdB));
        ret->PackageMBytesPerTransfer(conf::OckSysConf::ToolConf().packageMBPerTransfer.minValue);
        ret->TestTime(conf::OckSysConf::ToolConf().testTime.minValue);
        return ret;
    }
    void CheckTestResult(const TopoDetectResult &result, hmm::OckHmmDeviceId devIdA, hmm::OckHmmDeviceId devIdB,
        const TopoDetectParam &param)
    {
        EXPECT_TRUE(result.deviceInfo.deviceId == devIdA || result.deviceInfo.deviceId == devIdB);
        EXPECT_EQ(result.errorCode, hmm::HMM_SUCCESS);
    }
    void CheckSingleResult(const std::vector<TopoDetectResult> &resultVec, const TopoDetectParam &param)
    {
        ASSERT_EQ(resultVec.size(), param.GetDeviceInfo().size());
        CheckTestResult(resultVec.front(), deviceIdA, deviceIdA, param);
        EXPECT_NE(utils::ToString(resultVec), "");
    }
    void CheckTwoResult(const std::vector<TopoDetectResult> &resultVec, const TopoDetectParam &param)
    {
        ASSERT_EQ(resultVec.size(), param.GetDeviceInfo().size());
        CheckTestResult(resultVec.front(), deviceIdA, deviceIdB, param);
        CheckTestResult(resultVec.back(), deviceIdA, deviceIdB, param);
        EXPECT_NE(utils::ToString(resultVec), "");
    }
    void CheckTwoResultBasic(
        const std::vector<TopoDetectResult> &resultVec, const TopoDetectParam &param, hmm::OckHmmErrorCode errorCode)
    {
        ASSERT_EQ(resultVec.size(), param.GetDeviceInfo().size());
        auto &deviceInfoA = resultVec.front().deviceInfo;
        EXPECT_TRUE(deviceInfoA.deviceId == deviceIdA || deviceInfoA.deviceId == deviceIdB);
        auto &deviceInfoB = resultVec.front().deviceInfo;
        EXPECT_TRUE(deviceInfoB.deviceId == deviceIdA || deviceInfoB.deviceId == deviceIdB);
        EXPECT_NE(utils::ToString(resultVec), "");
    }
    void CheckTwoResultFailed(
        const std::vector<TopoDetectResult> &resultVec, const TopoDetectParam &param, hmm::OckHmmErrorCode errorCode)
    {
        CheckTwoResultBasic(resultVec, param, errorCode);
        EXPECT_EQ(resultVec.front().errorCode, errorCode);
        EXPECT_EQ(resultVec.back().errorCode, errorCode);
    }
    void CheckTwoResultAtleastOneFailed(
        const std::vector<TopoDetectResult> &resultVec, const TopoDetectParam &param, hmm::OckHmmErrorCode errorCode)
    {
        CheckTwoResultBasic(resultVec, param, errorCode);
        EXPECT_TRUE(resultVec.front().errorCode == errorCode || resultVec.back().errorCode == errorCode);
    }
    void CheckTwoDeviceAllResult(const std::vector<TopoDetectResult> &resultVec, const TopoDetectParam &param)
    {
        const uint32_t copyKindCount = 4U;
        ASSERT_EQ(resultVec.size(), param.GetDeviceInfo().size() * copyKindCount);
        EXPECT_NE(utils::ToString(resultVec), "");
        std::unordered_set<acladapter::OckMemoryCopyKind> retKindSet;
        for (auto &result : resultVec) {
            CheckTestResult(result, deviceIdA, deviceIdB, param);
            retKindSet.insert(result.copyKind);
        }
        EXPECT_EQ(retKindSet.size(), copyKindCount);
    }

    std::vector<TopoDetectResult> DetectCheck(std::shared_ptr<TopoDetectParam> param)
    {
        auto detecter = TopoDetectChecker::Create(param);
        return detecter->Check();
    }

    void MockMallocAndFree(void)
    {
        MOCKER(aclInit).stubs().will(returnValue(ACL_SUCCESS));
        MOCKER(aclFinalize).stubs().will(returnValue(ACL_SUCCESS));
        MOCKER(aclrtSetDevice).stubs().will(returnValue(ACL_SUCCESS));
        MOCKER(aclrtResetDevice).stubs().will(returnValue(ACL_SUCCESS));
        MOCKER(aclrtMalloc).stubs().will(invoke(acladapter::aclmock::FakeAclrtMalloc));
        MOCKER(aclrtFree).stubs().will(invoke(acladapter::aclmock::FakeAclrtFree));
        MOCKER(aclrtMallocHost).stubs().will(invoke(acladapter::aclmock::FakeAclrtMallocHost));
        MOCKER(aclrtFreeHost).stubs().will(invoke(acladapter::aclmock::FakeAclrtFreeHost));
        MOCKER(aclrtGetDeviceCount).stubs().will(invoke(acladapter::aclmock::FakeAclGetDeviceCount));
    }

    hmm::OckHmmDeviceId deviceIdA;
    hmm::OckHmmDeviceId deviceIdB;
};

TEST_F(TestTopoDetectChecker, serial_one_device)
{
    auto param = BuildParam(deviceIdA, DetectModel::SERIAL);
    param->SetTransferType((uint32_t)acladapter::OckMemoryCopyKind::HOST_TO_DEVICE);
    auto ret = DetectCheck(param);
    CheckSingleResult(ret, *param);
}
TEST_F(TestTopoDetectChecker, serial_two_device)
{
    auto param = BuildParam(deviceIdA, deviceIdB, DetectModel::SERIAL);
    param->SetTransferType((uint32_t)acladapter::OckMemoryCopyKind::DEVICE_TO_HOST);
    auto ret = DetectCheck(param);
    CheckTwoResult(ret, *param);
}
TEST_F(TestTopoDetectChecker, parallel_one_device)
{
    auto param = BuildParam(deviceIdA, DetectModel::PARALLEL);
    param->SetTransferType((uint32_t)acladapter::OckMemoryCopyKind::HOST_TO_HOST);
    auto ret = DetectCheck(param);
    CheckSingleResult(ret, *param);
}
TEST_F(TestTopoDetectChecker, parallel_two_device)
{
    auto param = BuildParam(deviceIdA, deviceIdB, DetectModel::PARALLEL);
    param->SetTransferType((uint32_t)acladapter::OckMemoryCopyKind::DEVICE_TO_DEVICE);
    auto ret = DetectCheck(param);
    CheckTwoResult(ret, *param);
}
TEST_F(TestTopoDetectChecker, parallel_two_device_all_transfer_type)
{
    uint32_t transferTypeAll = (uint32_t)acladapter::OckMemoryCopyKind::HOST_TO_HOST +
                              (uint32_t)acladapter::OckMemoryCopyKind::DEVICE_TO_HOST +
                              (uint32_t)acladapter::OckMemoryCopyKind::DEVICE_TO_DEVICE +
                              (uint32_t)acladapter::OckMemoryCopyKind::HOST_TO_DEVICE;
    auto param = BuildParam(deviceIdA, deviceIdB, DetectModel::PARALLEL);
    param->SetTransferType(transferTypeAll);
    auto ret = DetectCheck(param);
    CheckTwoDeviceAllResult(ret, *param);
}
TEST_F(TestTopoDetectChecker, thread_started_failed)
{
    GlobalMockObject::verify();  // 需要清空WithEnvAclMock设置的桩，不能重复MOCKER aclrtSetDevice
    MOCKER(aclInit).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclFinalize).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtSetDevice).stubs().will(returnValue(ACL_ERROR_RT_NO_DEVICE));
    auto param = BuildParam(deviceIdA, deviceIdB, DetectModel::PARALLEL);
    param->SetTransferType((uint32_t)acladapter::OckMemoryCopyKind::HOST_TO_DEVICE);
    auto ret = DetectCheck(param);
    CheckTwoResultFailed(ret, *param, ACL_ERROR_RT_NO_DEVICE);
}
aclError FakeAclrtMemCpyWithSleep(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(200U));
    return ACL_SUCCESS;
};
TEST_F(TestTopoDetectChecker, parallel_thread_run_time_out)
{
    GlobalMockObject::verify();  // 需要清空WithEnvAclMock设置的桩，不能重复MOCKER aclrtMemcpy
    conf::OckTopoDetectConf conf;
    conf.toolParrallMaxQueryTimes = 1U;             // 只轮询一次
    conf.toolParrallQueryIntervalMicroSecond = 1U;  // 每次等待1微秒
    MockMallocAndFree();
    MOCKER(aclrtMemcpy).stubs().will(invoke(FakeAclrtMemCpyWithSleep));
    MOCKER(conf::OckSysConf::ToolConf).stubs().will(returnValue(conf));
    auto param = BuildParam(deviceIdA, deviceIdB, DetectModel::PARALLEL);
    param->SetTransferType((uint32_t)acladapter::OckMemoryCopyKind::HOST_TO_DEVICE);
    auto ret = DetectCheck(param);
    CheckTwoResultAtleastOneFailed(ret, *param, hmm::HMM_ERROR_WAIT_TIME_OUT);
}
TEST_F(TestTopoDetectChecker, serial_copy_failed)
{
    GlobalMockObject::verify();  // 需要清空WithEnvAclMock设置的桩，不能重复MOCKER aclrtMemcpy
    MockMallocAndFree();
    MOCKER(aclrtMemcpy).stubs().will(returnValue(hmm::HMM_ERROR_SPACE_NOT_ENOUGH));
    auto param = BuildParam(deviceIdA, deviceIdB, DetectModel::SERIAL);
    param->SetTransferType((uint32_t)acladapter::OckMemoryCopyKind::HOST_TO_DEVICE);
    auto ret = DetectCheck(param);
    CheckTwoResultFailed(ret, *param, hmm::HMM_ERROR_SPACE_NOT_ENOUGH);
}
}  // namespace tests
}  // namespace topo
}  // namespace tools
}  // namespace ock