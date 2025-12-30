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
#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"
#include "ock/vsa/neighbor/base/OckVsaAnnCreateParam.h"
namespace ock {
namespace vsa {
namespace neighbor {
namespace test {
using namespace hmm;
class TestOckVsaAnnCreateParam : public testing::Test {
public:
    void BuildParam()
    {
        CPU_ZERO(&cpuSet);
        for (uint32_t i = 0; i < minCpuNum; ++i) {
            CPU_SET(i, &cpuSet);
        }
        param = OckVsaAnnCreateParam::Create(cpuSet, deviceId, maxFeatureRowCount, tokenNum, extKeyAttrByteSize,
            extKeyAttrBlockSize);
    }

    std::shared_ptr<OckVsaAnnCreateParam> param;
    uint32_t minCpuNum = 4;
    uint64_t maxFeatureRowCount{ 16777216 };
    hmm::OckHmmDeviceId deviceId{ 0 };
    uint32_t tokenNum{ 100UL };
    cpu_set_t cpuSet;
    uint32_t extKeyAttrByteSize{8UL};
    uint32_t extKeyAttrBlockSize{262144UL};
    double distanceThreshold{0.8};
    uint32_t blockRowCount{262144UL};
    uint32_t groupBlockCount{64UL};
    uint32_t sliceRowCount{64UL};
    double defaultFirstThreshold{0.9};
    double defaultSecondThreshold{0.8};
};

TEST_F(TestOckVsaAnnCreateParam, Create)
{
    BuildParam();
    uint32_t groupSliceCount = 262144UL;
    std::string cpuIds = "[0-3]";
    std::stringstream ss;
    ss << cpuSet;
    EXPECT_EQ(param->ExtKeyAttrByteSize(), extKeyAttrByteSize);
    EXPECT_EQ(param->ExtKeyAttrBlockSize(), extKeyAttrBlockSize);
    EXPECT_EQ(param->MaxFeatureRowCount(), maxFeatureRowCount);
    EXPECT_EQ(param->DistanceThreshold(), distanceThreshold);
    EXPECT_EQ(param->DeviceId(), deviceId);
    EXPECT_EQ(param->TokenNum(), tokenNum);
    EXPECT_EQ(ss.str(), cpuIds);
    EXPECT_EQ(param->BlockRowCount(), blockRowCount);
    EXPECT_EQ(param->GroupBlockCount(), groupBlockCount);
    EXPECT_EQ(param->SliceRowCount(), sliceRowCount);
    EXPECT_EQ(param->GroupRowCount(), groupBlockCount * blockRowCount);
    EXPECT_EQ(param->GroupSliceCount(), groupSliceCount);
    EXPECT_EQ(param->MaxGroupCount(), utils::SafeDivUp(maxFeatureRowCount, groupBlockCount * blockRowCount));
}

TEST_F(TestOckVsaAnnCreateParam, Copy)
{
    BuildParam();
    uint64_t newMaxFeatureRowCount = 262144 * 128;
    auto newParam = param->Copy(newMaxFeatureRowCount);
    std::string cpuIds = "[0-3]";
    std::stringstream ss;
    ss << cpuSet;
    EXPECT_EQ(newParam->ExtKeyAttrByteSize(), param->ExtKeyAttrByteSize());
    EXPECT_EQ(newParam->ExtKeyAttrBlockSize(), param->ExtKeyAttrBlockSize());
    EXPECT_EQ(newParam->MaxFeatureRowCount(), newMaxFeatureRowCount);
    EXPECT_EQ(param->DistanceThreshold(), distanceThreshold);
    EXPECT_EQ(newParam->DeviceId(), param->DeviceId());
    EXPECT_EQ(newParam->TokenNum(), param->TokenNum());
    EXPECT_EQ(ss.str(), cpuIds);
    EXPECT_EQ(newParam->BlockRowCount(), param->BlockRowCount());
    EXPECT_EQ(newParam->GroupBlockCount(), param->GroupBlockCount());
    EXPECT_EQ(newParam->SliceRowCount(), param->SliceRowCount());
    EXPECT_EQ(newParam->GroupRowCount(), param->GroupRowCount());
    EXPECT_EQ(newParam->GroupSliceCount(), param->GroupSliceCount());
    EXPECT_EQ(newParam->MaxGroupCount(), utils::SafeDivUp(newMaxFeatureRowCount, groupBlockCount * blockRowCount));
}

TEST_F(TestOckVsaAnnCreateParam, Print)
{
    BuildParam();
    std::string printInfo =
        "{'maxFeatureRowCount': 16777216, 'deviceId': 0, 'tokenNum': 100, 'cpuSet': [0-3], 'extKeyAttrByteSize': 8, "
        "'extKeyAttrBlockSize': 262144, 'blockRowCount': 262144, 'groupBlockCount': 64, 'sliceRowCount': 64, "
        "'groupRowCount': 16777216, 'maxGroupCount': 1, 'groupSliceCount': 262144, 'distanceThreshold': 0.8, "
        "'firstClassThreshold': 0.9, 'secondClassThreshold': 0.8}";
    std::stringstream ss;
    ss << *param;
    EXPECT_EQ(ss.str(), printInfo);
}

TEST_F(TestOckVsaAnnCreateParam, CheckValid_maxFeatureRowCount)
{
    maxFeatureRowCount = 8388608UL;
    BuildParam();
    EXPECT_EQ(OckVsaAnnCreateParam::CheckValid(*param), VSA_ERROR_MAX_FEATURE_ROW_COUNT_OUT);
    maxFeatureRowCount = 1258291200UL;
    BuildParam();
    EXPECT_EQ(OckVsaAnnCreateParam::CheckValid(*param), VSA_ERROR_MAX_FEATURE_ROW_COUNT_OUT);
    maxFeatureRowCount = 16777218ULL;
    BuildParam();
    EXPECT_EQ(OckVsaAnnCreateParam::CheckValid(*param), VSA_ERROR_MAX_FEATURE_ROW_COUNT_DIVISIBLE);
}
TEST_F(TestOckVsaAnnCreateParam, CheckValid_DeviceId)
{
    uint32_t errorDeviceId = 10;
    deviceId = errorDeviceId;
    BuildParam();
    EXPECT_EQ(OckVsaAnnCreateParam::CheckValid(*param), VSA_ERROR_DEVICE_NOT_EXISTS);
}
TEST_F(TestOckVsaAnnCreateParam, CheckValid_CpuSet)
{
    cpu_set_t errorCpuSet;
    CPU_ZERO(&errorCpuSet);
    for (uint32_t i = 0; i < minCpuNum - 1; ++i) {
        CPU_SET(i, &errorCpuSet);
    }
    param = OckVsaAnnCreateParam::Create(errorCpuSet, deviceId, maxFeatureRowCount, tokenNum, extKeyAttrByteSize,
        extKeyAttrBlockSize);
    EXPECT_EQ(OckVsaAnnCreateParam::CheckValid(*param), VSA_ERROR_CPUID_OUT);
    uint32_t errorCpuId = 100;
    CPU_SET(errorCpuId, &errorCpuSet);
    param = OckVsaAnnCreateParam::Create(errorCpuSet, deviceId, maxFeatureRowCount, tokenNum, extKeyAttrByteSize,
        extKeyAttrBlockSize);
    EXPECT_EQ(OckVsaAnnCreateParam::CheckValid(*param), VSA_ERROR_CPUID_NOT_EXISTS);
}
TEST_F(TestOckVsaAnnCreateParam, CheckValid_AssignedParam)
{
    tokenNum = 0;
    BuildParam();
    EXPECT_EQ(OckVsaAnnCreateParam::CheckValid(*param), VSA_ERROR_TOKEN_NUM_OUT);
    tokenNum = 300001UL;
    BuildParam();
    EXPECT_EQ(OckVsaAnnCreateParam::CheckValid(*param), VSA_ERROR_TOKEN_NUM_OUT);
    tokenNum = 100UL;
    extKeyAttrByteSize = 23UL;
    BuildParam();
    EXPECT_EQ(OckVsaAnnCreateParam::CheckValid(*param), VSA_ERROR_EXTKEYATTR_BYTE_SIZE_OUT);
    extKeyAttrByteSize = 8UL;
    extKeyAttrBlockSize = 131072UL;
    BuildParam();
    EXPECT_EQ(OckVsaAnnCreateParam::CheckValid(*param), VSA_ERROR_EXTKEYATTR_BLOCK_SIZE_OUT);
    extKeyAttrBlockSize = 262145UL;
    BuildParam();
    EXPECT_EQ(OckVsaAnnCreateParam::CheckValid(*param), VSA_ERROR_GROUP_ROW_COUNT_DIVISIBLE);
}
TEST_F(TestOckVsaAnnCreateParam, SetThreshold)
{
    tokenNum = 0;
    BuildParam();

    double firstThreshold = 1.5;
    double secondThreshold = 1.5;
    param->SetFirstClassNeighborCellThreshold(firstThreshold);
    param->SetSecondClassNeighborCellThreshold(secondThreshold);

    EXPECT_EQ(param->FirstClassNeighborCellThreshold(), defaultFirstThreshold);
    EXPECT_EQ(param->SecondClassNeighborCellThreshold(), defaultSecondThreshold);
}
} // namespace test
} // namespace neighbor
} // namespace vsa
} // namespace ock