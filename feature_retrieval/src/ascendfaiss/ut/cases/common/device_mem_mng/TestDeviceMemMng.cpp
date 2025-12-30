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
#include <map>
#include <thread>
#include "ascendfaiss/ascenddaemon/utils/MemorySpace.h"
#include "ascendfaiss/ascenddaemon/utils/DeviceMemMng.h"
#include "stub/hmm/HmmMock.h"

using namespace testing;
using namespace std;

namespace ascend {

class TestDeviceMemMng : public Test {
public:
    void TearDown() override
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TestDeviceMemMng, InitHmm)
{
    uint32_t deviceId = 0;
    size_t hostCapacity = 1;
    std::shared_ptr<ascend::HmmIntf> intf = nullptr;
    MOCKER_CPP(&HmmIntf::CreateHmm).stubs().will(returnValue(intf));
    DeviceMemMng mng;
    auto ret = mng.InitHmm(deviceId, hostCapacity);
    EXPECT_EQ(ret, APP_ERR_INNER_ERROR);
}

TEST_F(TestDeviceMemMng, SetHeteroParam)
{
    DeviceMemMng mng;
    uint32_t deviceId = 0;
    size_t deviceCapacity = 0;
    size_t deviceBuffer = 0;
    size_t hostCapacity = 0;
    size_t devVecSize = 0;
    auto ret = mng.SetHeteroParam(deviceId, deviceCapacity, deviceBuffer, hostCapacity, devVecSize);
    EXPECT_EQ(ret, APP_ERR_INVALID_PARAM);

    std::shared_ptr<ascend::HmmIntf> intf = nullptr;
    MOCKER_CPP(&HmmIntf::CreateHmm).stubs().will(returnValue(intf));
    deviceBuffer = 1;
    ret = mng.SetHeteroParam(deviceId, deviceCapacity, deviceBuffer, hostCapacity, devVecSize);
    EXPECT_EQ(ret, APP_ERR_INNER_ERROR);
}

TEST_F(TestDeviceMemMng, GetFunctions)
{
    DeviceMemMng mng;
    bool usingGroupSearch = mng.UsingGroupSearch();
    EXPECT_FALSE(usingGroupSearch);

    mng.SetHeteroStrategy();

    DevMemStrategy strategy = mng.GetStrategy();
    EXPECT_EQ(strategy, DevMemStrategy::HETERO_MEM);

    size_t capacity = mng.GetDeviceCapacity();
    EXPECT_EQ(0, capacity);

    size_t deviceBuffer = mng.GetDeviceBuffer();
    EXPECT_EQ(0, deviceBuffer);
}

TEST_F(TestDeviceMemMng, CreateDeviceVector)
{
    DeviceMemMng mng;
    mng.CreateDeviceVector<int8_t>();

    mng.SetHeteroStrategy();
    mng.CreateDeviceVector<int8_t>(MemorySpace::DEVICE_HUGEPAGE);
}
}
