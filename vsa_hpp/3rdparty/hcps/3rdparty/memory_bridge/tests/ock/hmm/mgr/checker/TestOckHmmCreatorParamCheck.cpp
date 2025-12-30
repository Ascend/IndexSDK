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


#include <sstream>
#include <unistd.h>
#include <limits>
#include "gtest/gtest.h"
#include "acl/acl.h"
#include "ock/utils/StrUtils.h"
#include "ock/hmm/mgr/checker/OckHmmCreatorParamCheck.h"
#include "ock/acladapter/WithEnvAclMock.h"

namespace ock {
namespace hmm {
namespace test {

class TestOckHmmCreatorParamCheck : public acladapter::WithEnvAclMock<testing::Test> {
public:
    void SetUp() override
    {
        acladapter::WithEnvAclMock<testing::Test>::SetUp();
        aclInit(nullptr);
        goodSample.deviceId = deviceInfoConf.deviceId.minValue;
        CPU_ZERO(&goodSample.cpuSet);
        CPU_SET(0, &goodSample.cpuSet);
        goodSample.transferThreadNum = deviceInfoConf.transferThreadNum.minValue;
        goodSample.memorySpec.devSpec.maxDataCapacity = deviceInfoConf.deviceBaseCapacity.minValue;
        goodSample.memorySpec.devSpec.maxSwapCapacity = deviceInfoConf.deviceBufferCapacity.minValue;
        goodSample.memorySpec.hostSpec.maxDataCapacity = deviceInfoConf.hostBaseCapacity.minValue;
        goodSample.memorySpec.hostSpec.maxSwapCapacity = deviceInfoConf.hostBufferCapacity.minValue;
    }

    void TearDown(void) override
    {
        aclFinalize();
        acladapter::WithEnvAclMock<testing::Test>::TearDown();
    }

    template<typename T, typename FunT>
    void BuildTestCase(OckHmmDeviceInfoVec &goodScene, OckHmmDeviceInfoVec &badScene,
                       const conf::ParamRange<T> &range, FunT fun)
    {
        OckHmmDeviceInfo goodMinInfo = goodSample;
        fun(goodMinInfo, range.minValue);
        goodScene.push_back(goodMinInfo);

        OckHmmDeviceInfo goodMaxInfo = goodSample;
        fun(goodMaxInfo, range.maxValue);
        goodScene.push_back(goodMaxInfo);

        const T times = 2;
        OckHmmDeviceInfo goodMidInfo = goodSample;
        fun(goodMinInfo, (range.minValue + range.maxValue) / times);
        goodScene.push_back(goodMidInfo);

        OckHmmDeviceInfo badMaxInfo = goodSample;
        fun(badMaxInfo, range.maxValue + 1);
        badScene.push_back(badMaxInfo);
        
        if (range.minValue > std::numeric_limits<T>::min()) {
            OckHmmDeviceInfo badMinInfo = goodSample;
            fun(badMinInfo, range.minValue - 1);
            badScene.push_back(badMinInfo);
        }
    }

    void ExpectSceneResult(const OckHmmDeviceInfoVec &scene, OckHmmErrorCode retCode)
    {
        for (uint32_t i = 0; i < scene.size(); ++i) {
            EXPECT_EQ(OckHmmCreatorParamCheck::CheckParam(scene[i]), retCode);
        }
        EXPECT_EQ(OckHmmCreatorParamCheck::CheckParam(scene), retCode);
    }

    OckHmmDeviceInfo goodSample;
    const conf::OckHmmDeviceInfoConf &deviceInfoConf = conf::OckSysConf::DeviceInfoConf();
};

TEST_F(TestOckHmmCreatorParamCheck, check_device_id_exists)
{
    OckHmmErrorCode retCode = HMM_SUCCESS;
    OckHmmCreatorParamCheck::CheckDeviceIdExists(retCode, goodSample.deviceId);
    EXPECT_EQ(retCode, HMM_SUCCESS);

    uint32_t deviceCount = 0;
    aclrtGetDeviceCount(&deviceCount);
    OckHmmDeviceInfo deviceInfo = goodSample;
    deviceInfo.deviceId = deviceCount;
    OckHmmCreatorParamCheck::CheckDeviceIdExists(retCode, deviceInfo.deviceId);
    EXPECT_EQ(retCode, HMM_ERROR_INPUT_PARAM_DEVICE_NOT_EXISTS);

    OckHmmDeviceInfoVec deviceInfoVec = {goodSample, deviceInfo};
    EXPECT_EQ(OckHmmCreatorParamCheck::CheckParam(goodSample), HMM_SUCCESS);
    EXPECT_EQ(OckHmmCreatorParamCheck::CheckParam(deviceInfo), HMM_ERROR_INPUT_PARAM_DEVICE_NOT_EXISTS);
    EXPECT_EQ(OckHmmCreatorParamCheck::CheckParam(deviceInfoVec), HMM_ERROR_INPUT_PARAM_DEVICE_NOT_EXISTS);
}

TEST_F(TestOckHmmCreatorParamCheck, check_cpu_id_exists)
{
    OckHmmErrorCode retCode = HMM_SUCCESS;
    OckHmmCreatorParamCheck::CheckCpuIdExists(retCode, goodSample.cpuSet);
    EXPECT_EQ(retCode, HMM_SUCCESS);

    OckHmmDeviceInfo deviceInfo = goodSample;
    uint32_t cpuCount = sysconf(_SC_NPROCESSORS_CONF);
    CPU_SET(cpuCount, &deviceInfo.cpuSet);
    OckHmmCreatorParamCheck::CheckCpuIdExists(retCode, deviceInfo.cpuSet);
    EXPECT_EQ(retCode, HMM_ERROR_INPUT_PARAM_CPUID_NOT_EXISTS);

    OckHmmDeviceInfoVec deviceInfoVec = {goodSample, deviceInfo};
    EXPECT_EQ(OckHmmCreatorParamCheck::CheckParam(goodSample), HMM_SUCCESS);
    EXPECT_EQ(OckHmmCreatorParamCheck::CheckParam(deviceInfo), HMM_ERROR_INPUT_PARAM_CPUID_NOT_EXISTS);
    EXPECT_EQ(OckHmmCreatorParamCheck::CheckParam(deviceInfoVec), HMM_ERROR_INPUT_PARAM_CPUID_NOT_EXISTS);
}

TEST_F(TestOckHmmCreatorParamCheck, check_param_transferThreadNum)
{
    OckHmmDeviceInfoVec goodScene;
    OckHmmDeviceInfoVec badScene;
    BuildTestCase(goodScene, badScene, deviceInfoConf.transferThreadNum,
        [](OckHmmDeviceInfo &info, uint32_t threadNum) { info.transferThreadNum = threadNum; });

    ExpectSceneResult(goodScene, HMM_SUCCESS);
    ExpectSceneResult(badScene, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
}

TEST_F(TestOckHmmCreatorParamCheck, check_param_memorySpec_devSpec_maxDataCapacity)
{
    OckHmmDeviceInfoVec goodScene;
    OckHmmDeviceInfoVec badScene;
    BuildTestCase(goodScene, badScene, deviceInfoConf.deviceBaseCapacity,
        [](OckHmmDeviceInfo &info, uint64_t newSize) { info.memorySpec.devSpec.maxDataCapacity = newSize; });

    ExpectSceneResult(goodScene, HMM_SUCCESS);
    ExpectSceneResult(badScene, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
}

TEST_F(TestOckHmmCreatorParamCheck, check_param_memorySpec_devSpec_maxSwapCapacity)
{
    OckHmmDeviceInfoVec goodScene;
    OckHmmDeviceInfoVec badScene;
    BuildTestCase(goodScene, badScene, deviceInfoConf.deviceBufferCapacity,
        [](OckHmmDeviceInfo &info, uint64_t newSize) { info.memorySpec.devSpec.maxSwapCapacity = newSize; });

    ExpectSceneResult(goodScene, HMM_SUCCESS);
    ExpectSceneResult(badScene, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
}

TEST_F(TestOckHmmCreatorParamCheck, check_param_memorySpec_hostSpec_maxDataCapacity)
{
    OckHmmDeviceInfoVec goodScene;
    OckHmmDeviceInfoVec badScene;
    BuildTestCase(goodScene, badScene, deviceInfoConf.hostBaseCapacity,
        [](OckHmmDeviceInfo &info, uint64_t newSize) { info.memorySpec.hostSpec.maxDataCapacity = newSize; });

    ExpectSceneResult(goodScene, HMM_SUCCESS);
    ExpectSceneResult(badScene, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
}

TEST_F(TestOckHmmCreatorParamCheck, check_param_memorySpec_hostSpec_maxSwapCapacity)
{
    OckHmmDeviceInfoVec goodScene;
    OckHmmDeviceInfoVec badScene;
    BuildTestCase(goodScene, badScene, deviceInfoConf.hostBufferCapacity,
        [](OckHmmDeviceInfo &info, uint64_t newSize) { info.memorySpec.hostSpec.maxSwapCapacity = newSize; });

    ExpectSceneResult(goodScene, HMM_SUCCESS);
    ExpectSceneResult(badScene, HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE);
}
}
}
}
