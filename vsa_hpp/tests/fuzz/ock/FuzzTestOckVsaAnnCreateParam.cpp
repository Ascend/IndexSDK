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
#include <iostream>
#include <fstream>
#include <securec.h>
#include "secodeFuzz.h"
#include "ock/vsa/neighbor/base/OckVsaAnnCreateParam.h"
namespace ock {
static constexpr size_t DF_FUZZ_EXEC_COUNT = 1;
static constexpr size_t DF_FUZZ_EXEC_SECOND = 10800;
const uint64_t DIM = 256; // 256: dim
const uint32_t MAX_TIME_SCOPE = 256;
const uint32_t MAX_TOKEN_SCOPE = 2500U;
const uint64_t MAX_FEATURE_ROW_COUNT = 262144ULL * 64ULL * 10ULL;
const uint32_t EXT_KEY_ATTR_BYTE_SIZE = 22U;
const uint32_t DEVICE_ID = 0U;
class FuzzTestOckVsaAnnCreateParam : public testing::Test {
public:
    void SetUp(void) override
    {
        DT_Set_Running_Time_Second(DF_FUZZ_EXEC_SECOND);
        DT_Enable_Leak_Check(0, 0);
        DT_Set_Report_Path("/home/pjy/vsa/tests/fuzz/build/report");
    }

    void BuildCreateParam()
    {
        auto deviceInfo = std::make_shared<hmm::OckHmmDeviceInfo>();
        deviceInfo->deviceId = DEVICE_ID;
        CPU_SET(1U, &deviceInfo->cpuSet); // 设置1号CPU核
        CPU_SET(2U, &deviceInfo->cpuSet);
        CPU_SET(3U, &deviceInfo->cpuSet);
        CPU_SET(4U, &deviceInfo->cpuSet);
        param = vsa::neighbor::OckVsaAnnCreateParam::Create(deviceInfo->cpuSet, deviceInfo->deviceId,
        MAX_FEATURE_ROW_COUNT, 2500U, EXT_KEY_ATTR_BYTE_SIZE);
    }

    void TestCreateParam()
    {
        double firstThreshold = 0.9;
        double secondThreshold = 0.8;
        double distanceTheshold = 0.8;
        EXPECT_EQ(param->MaxFeatureRowCount(), MAX_FEATURE_ROW_COUNT);
        EXPECT_EQ(param->DeviceId(), DEVICE_ID);
        EXPECT_EQ(param->TokenNum(), 2500U);
        param->CpuSet();
        EXPECT_EQ(param->ExtKeyAttrByteSize(), EXT_KEY_ATTR_BYTE_SIZE);
        EXPECT_EQ(param->ExtKeyAttrBlockSize(), 262144U);
        EXPECT_EQ(param->BlockRowCount(), 262144U);
        EXPECT_EQ(param->GroupBlockCount(), 64U);
        EXPECT_EQ(param->SliceRowCount(), 64U);
        EXPECT_EQ(param->GroupRowCount(), 262144U * 64U);
        EXPECT_EQ(param->MaxGroupCount(), 10U);
        EXPECT_EQ(param->GroupSliceCount(), 262144U);
        EXPECT_EQ(param->DistanceThreshold(), distanceTheshold);
        param->SetFirstClassNeighborCellThreshold(firstThreshold);
        param->SetSecondClassNeighborCellThreshold(secondThreshold);
        EXPECT_EQ(param->FirstClassNeighborCellThreshold(), firstThreshold);
        EXPECT_EQ(param->SecondClassNeighborCellThreshold(), secondThreshold);
    }

    std::shared_ptr<vsa::neighbor::OckVsaAnnCreateParam> param;
};

TEST_F(FuzzTestOckVsaAnnCreateParam, create_param)
{
    uint32_t seed = 0;
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, "create_param", 0)
    {
        BuildCreateParam();
        TestCreateParam();
    }
    DT_FUZZ_END()
}
}