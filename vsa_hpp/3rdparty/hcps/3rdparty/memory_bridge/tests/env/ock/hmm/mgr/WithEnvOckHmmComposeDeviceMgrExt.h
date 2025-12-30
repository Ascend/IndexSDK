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
namespace ock {
namespace hmm {
template <typename BaseT>
class WithEnvOckHmmComposeDeviceMgrExt : public BaseT {
public:
    // 这里将deviceId设置为质数，主要是方便测试时发现问题
    WithEnvOckHmmComposeDeviceMgrExt(void)
        : deviceIdA(2U), deviceIdB(3U), deviceIdC(5U), deviceIdD(7U), unknownDeviceId(11U),
          devMgrA(AddDevMgr(deviceIdA)), devMgrB(AddDevMgr(deviceIdB)), devMgrC(AddDevMgr(deviceIdC)),
          devMgrD(AddDevMgr(deviceIdD)), composeMgr(ext::CreateComposeDeviceMgr(devMgrVec))
    {}
    void TearDown(void) override
    {
        GlobalMockObject::verify();
    }
    MockOckHmmSingleDeviceMgr &AddDevMgr(OckHmmDeviceId deviceId)
    {
        auto mgrMock = new MockOckHmmSingleDeviceMgr();
        devMgrVec.push_back(std::shared_ptr<OckHmmSingleDeviceMgr>(mgrMock));
        EXPECT_CALL(*mgrMock, GetDeviceId()).WillRepeatedly(testing::Return(deviceId));
        return *mgrMock;
    }

    OckHmmDeviceId deviceIdA;
    OckHmmDeviceId deviceIdB;
    OckHmmDeviceId deviceIdC;
    OckHmmDeviceId deviceIdD;
    OckHmmDeviceId unknownDeviceId;

    std::vector<std::shared_ptr<OckHmmSingleDeviceMgr>> devMgrVec;

    MockOckHmmSingleDeviceMgr &devMgrA;
    MockOckHmmSingleDeviceMgr &devMgrB;
    MockOckHmmSingleDeviceMgr &devMgrC;
    MockOckHmmSingleDeviceMgr &devMgrD;

    std::shared_ptr<OckHmmComposeDeviceMgr> composeMgr;
};
}  // namespace hmm
}  // namespace ock