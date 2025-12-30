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

#include <memory>
#include <vector>
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include "ock/hmm/mgr/MockOckHmmSingleDeviceMgr.h"
#include "ock/hmm/mgr/checker/OckHmmComposeDeviceMgrParamCheck.h"
#include "ock/hmm/mgr/OckHmmComposeDeviceMgrExt.h"
#include "ock/hmm/mgr/WithEnvOckHmmComposeDeviceMgrExt.h"

namespace ock {
namespace hmm {
namespace test {

class TestOckHmmComposeDeviceMgrExtBasicQuery : public WithEnvOckHmmComposeDeviceMgrExt<testing::Test> {
public:
};

TEST_F(TestOckHmmComposeDeviceMgrExtBasicQuery, GetUsedInfo_return_null_while_param_check_failed)
{
    MOCKER(OckHmmComposeDeviceMgrParamCheck::CheckGetUsedInfo).stubs().will(returnValue(HMM_ERROR_UNKOWN_INNER_ERROR));
    uint64_t anyFragThreshold = 1U;
    EXPECT_FALSE(composeMgr->GetUsedInfo(anyFragThreshold, deviceIdA).get());
}
TEST_F(TestOckHmmComposeDeviceMgrExtBasicQuery, GetUsedInfo_return_null_while_device_not_exists)
{
    MOCKER(OckHmmComposeDeviceMgrParamCheck::CheckGetUsedInfo).stubs().will(returnValue(HMM_SUCCESS));
    uint64_t anyFragThreshold = 1U;
    EXPECT_FALSE(composeMgr->GetUsedInfo(anyFragThreshold, unknownDeviceId).get());
}
TEST_F(TestOckHmmComposeDeviceMgrExtBasicQuery, GetUsedInfo_return_correct)
{
    auto usedInfo = std::make_shared<OckHmmResourceUsedInfo>();
    usedInfo->devUsedInfo.swapUsedBytes = 311U;  // 设置一个任意特别的数
    MOCKER(OckHmmComposeDeviceMgrParamCheck::CheckGetUsedInfo).stubs().will(returnValue(HMM_SUCCESS));
    uint64_t anyFragThreshold = 1U;
    EXPECT_CALL(devMgrB, GetUsedInfo(anyFragThreshold)).WillOnce(testing::Return(usedInfo));
    auto retUsedInfo = composeMgr->GetUsedInfo(anyFragThreshold, deviceIdB);
    ASSERT_TRUE(retUsedInfo.get() != nullptr);
    EXPECT_EQ(*usedInfo, *retUsedInfo);
}
}  // namespace test
}  // namespace hmm
}  // namespace ock