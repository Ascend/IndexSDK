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


#include <string>
#include "gtest/gtest.h"
#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"
#include "ock/utils/StrUtils.h"
#include "ock/conf/OckSysConf.h"

namespace ock {
namespace hmm {
namespace test {

class TestOckHmmHeteroMemoryMgr : public testing::Test {
public:
    void SetCpuIds(cpu_set_t &cpuSet, std::vector<uint32_t> cpuIds)
    {
        CPU_ZERO(&cpuSet);
        for (uint32_t i = 0; i < cpuIds.size(); ++i) {
            CPU_SET(cpuIds[i], &cpuSet);
        }
    }

    void SetMemorySpec(uint64_t deviceBase, uint64_t deviceBuffer, uint64_t hostBase, uint64_t hostBuffer,
                       OckHmmMemorySpecification &memorySpec)
    {
        memorySpec.devSpec.maxDataCapacity = deviceBase;
        memorySpec.devSpec.maxSwapCapacity = deviceBuffer;
        memorySpec.hostSpec.maxDataCapacity = hostBase;
        memorySpec.hostSpec.maxSwapCapacity = hostBuffer;
    }

    OckHmmDeviceInfoVec BuildDeviceInfoVec()
    {
        OckHmmDeviceInfoVec deviceInfoVec;
        OckHmmDeviceInfo minDeviceInfo;
        OckHmmDeviceInfo maxDeviceInfo;
        
        minDeviceInfo.deviceId = deviceInfoConf.deviceId.minValue;
        SetCpuIds(minDeviceInfo.cpuSet, {0U, 1U, 2U});
        minDeviceInfo.transferThreadNum = deviceInfoConf.transferThreadNum.minValue;
        SetMemorySpec(deviceInfoConf.deviceBaseCapacity.minValue, deviceInfoConf.deviceBufferCapacity.minValue,
                      deviceInfoConf.hostBaseCapacity.minValue, deviceInfoConf.hostBufferCapacity.minValue,
                      minDeviceInfo.memorySpec);
        deviceInfoVec.push_back(minDeviceInfo);

        maxDeviceInfo.deviceId = deviceInfoConf.deviceId.maxValue;
        SetCpuIds(maxDeviceInfo.cpuSet, {3U, 4U, 5U});
        maxDeviceInfo.transferThreadNum = deviceInfoConf.transferThreadNum.maxValue;
        SetMemorySpec(deviceInfoConf.deviceBaseCapacity.maxValue, deviceInfoConf.deviceBufferCapacity.maxValue,
                      deviceInfoConf.hostBaseCapacity.maxValue, deviceInfoConf.hostBufferCapacity.maxValue,
                      maxDeviceInfo.memorySpec);
        deviceInfoVec.push_back(maxDeviceInfo);

        return deviceInfoVec;
    }

    OckHmmPureDeviceInfoVec BuildPureDeviceInfoVec()
    {
        OckHmmPureDeviceInfoVec pureDeviceInfoVec;
        OckHmmPureDeviceInfo minPureDeviceInfo;
        OckHmmPureDeviceInfo maxPureDeviceInfo;

        minPureDeviceInfo.deviceId = deviceInfoConf.deviceId.minValue;
        SetCpuIds(minPureDeviceInfo.cpuSet, {0U, 1U, 2U});
        SetMemorySpec(deviceInfoConf.deviceBaseCapacity.minValue, deviceInfoConf.deviceBufferCapacity.minValue,
                      0ULL, 0ULL, minPureDeviceInfo.memorySpec);
        pureDeviceInfoVec.push_back(minPureDeviceInfo);

        maxPureDeviceInfo.deviceId = deviceInfoConf.deviceId.maxValue;
        SetCpuIds(maxPureDeviceInfo.cpuSet, {3U, 4U, 5U});
        SetMemorySpec(deviceInfoConf.deviceBaseCapacity.maxValue, deviceInfoConf.deviceBufferCapacity.maxValue,
                      0ULL, 0ULL, maxPureDeviceInfo.memorySpec);
        pureDeviceInfoVec.push_back(maxPureDeviceInfo);

        return pureDeviceInfoVec;
    }

    conf::OckHmmDeviceInfoConf deviceInfoConf = conf::OckSysConf::DeviceInfoConf();
};

TEST_F(TestOckHmmHeteroMemoryMgr, output_continous_cpuIds)
{
    std::string cpuIds = "[0-2]";
    cpu_set_t cpuSet;
    SetCpuIds(cpuSet, {0U, 1U, 2U});
    std::stringstream ss;
    ss << cpuSet;
    EXPECT_EQ(ss.str(), cpuIds);
}

TEST_F(TestOckHmmHeteroMemoryMgr, output_discontinuous_cpuIds)
{
    std::string cpuIds = "[0-2, 4, 6, 8-10, 12]";
    cpu_set_t cpuSet;
    SetCpuIds(cpuSet, {0U, 1U, 2U, 4U, 6U, 8U, 9U, 10U, 12U});
    std::stringstream ss;
    ss << cpuSet;
    EXPECT_EQ(ss.str(), cpuIds);
}

TEST_F(TestOckHmmHeteroMemoryMgr, output_deviceInfo)
{
    std::string minInfo = "{'deviceId':0,'cpuSet':[0-2],'transferThreadNum':1,'memorySpec':{"\
                          "'devSpec':{'maxDataCapacity':1073741824,'maxSwapCapacity':67108864},"\
                          "'hostSpec':{'maxDataCapacity':1073741824,'maxSwapCapacity':67108864}}}";
    std::string maxInfo = "{'deviceId':15,'cpuSet':[3-5],'transferThreadNum':128,'memorySpec':{"\
                          "'devSpec':{'maxDataCapacity':549755813888,'maxSwapCapacity':8589934592},"\
                          "'hostSpec':{'maxDataCapacity':549755813888,'maxSwapCapacity':8589934592}}}";

    auto ret = BuildDeviceInfoVec();
    EXPECT_EQ(utils::ToString(ret[0]), minInfo);
    EXPECT_EQ(utils::ToString(ret[1]), maxInfo);

    std::string deviceInfoVec = "[";
    deviceInfoVec.append(minInfo).append(",").append(maxInfo).append("]");
    EXPECT_EQ(utils::ToString(ret), deviceInfoVec);
}

TEST_F(TestOckHmmHeteroMemoryMgr, output_pureDeviceInfo)
{
    std::string minInfo = "{'deviceId':0,'cpuSet':[0-2],'memorySpec':{"\
                          "'devSpec':{'maxDataCapacity':1073741824,'maxSwapCapacity':67108864},"\
                          "'hostSpec':{'maxDataCapacity':0,'maxSwapCapacity':0}}}";
    std::string maxInfo = "{'deviceId':15,'cpuSet':[3-5],'memorySpec':{"\
                          "'devSpec':{'maxDataCapacity':549755813888,'maxSwapCapacity':8589934592},"\
                          "'hostSpec':{'maxDataCapacity':0,'maxSwapCapacity':0}}}";

    auto ret = BuildPureDeviceInfoVec();
    EXPECT_EQ(utils::ToString(ret[0]), minInfo);
    EXPECT_EQ(utils::ToString(ret[1]), maxInfo);

    std::string pureDeviceInfoVec = "[";
    pureDeviceInfoVec.append(minInfo).append(",").append(maxInfo).append("]");
    EXPECT_EQ(utils::ToString(ret), pureDeviceInfoVec);
}
}
}
}