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
#include "ock/hmm/mgr/OckHmmComposeDeviceMgrExt.h"
#include "ock/hmm/mgr/WithEnvOckHmmComposeDeviceMgrExt.h"
#include "ock/hmm/mgr/MockOckHmmHMObject.h"
#include "ock/hmm/mgr/data/OckHmmHMOObjectIDGenerator.h"
#include "ock/hmm/mgr/checker/OckHmmComposeDeviceMgrParamCheck.h"

namespace ock {
namespace hmm {
namespace test {

class TestOckHmmComposeDeviceMgrExtCopyHmo : public WithEnvOckHmmComposeDeviceMgrExt<testing::Test> {
public:
    TestOckHmmComposeDeviceMgrExtCopyHmo()
        : WithEnvOckHmmComposeDeviceMgrExt<testing::Test>(), hmoBytes(64U * 1024U * 1024U),
          correctOffset(32U * 1024U * 1024U), overOffset(65U * 1024U * 1024U), lessOffset(2U * 1024U * 1024U),
          correctLength(2U * 1024U * 1024U), overLength(33U * 1024U * 1024U), objIdInDevA(23U), objIdInDevB(29U)
    {}

    void SetUp() override
    {
        WithEnvOckHmmComposeDeviceMgrExt<testing::Test>::SetUp();
        srcHmoObj = std::make_shared<MockOckHmmHMObject>();
        dstHmoObj = std::make_shared<MockOckHmmHMObject>();
    }

    void MockHmo(MockOckHmmHMObject &hmo, uint64_t bytes, OckHmmHMOObjectID objId)
    {
        EXPECT_CALL(hmo, GetId()).WillRepeatedly(testing::Return(objId));
        EXPECT_CALL(hmo, GetByteSize()).WillRepeatedly(testing::Return(bytes));
        EXPECT_CALL(hmo, Addr()).WillRepeatedly(testing::Return((uintptr_t)(&hmo)));
        EXPECT_CALL(devMgrA, CopyHMO(testing::_, testing::_, testing::_, testing::_, testing::_))
                            .WillRepeatedly(testing::Return(HMM_SUCCESS));
    }

    void DoTestCopyHMODeviceIdNotEqual(
        uint64_t bytes, uint64_t dstOffset, uint64_t srcOffset, size_t length, OckHmmErrorCode expectRetCode)
    {
        MockHmo(*srcHmoObj, bytes, objIdInDevA);
        MockHmo(*dstHmoObj, bytes, objIdInDevB);
        auto retCode = composeMgr->CopyHMO(*dstHmoObj, dstOffset, *srcHmoObj, srcOffset, length);
        EXPECT_EQ(retCode, expectRetCode);
    }

    void DoTestCopyHMODeviceIdEqual(
        uint64_t bytes, uint64_t dstOffset, uint64_t srcOffset, size_t length, OckHmmErrorCode expectRetCode)
    {
        MockHmo(*srcHmoObj, bytes, objIdInDevA);
        MockHmo(*dstHmoObj, bytes, objIdInDevA);
        auto retCode = composeMgr->CopyHMO(*dstHmoObj, dstOffset, *srcHmoObj, srcOffset, length);
        EXPECT_EQ(retCode, expectRetCode);
    }

    uint64_t hmoBytes;
    uint64_t correctOffset;
    uint64_t overOffset;
    uint64_t lessOffset;
    size_t correctLength;
    size_t overLength;

    OckHmmHMOObjectID objIdInDevA;
    OckHmmHMOObjectID objIdInDevB;

    std::shared_ptr<MockOckHmmHMObject> srcHmoObj;
    std::shared_ptr<MockOckHmmHMObject> dstHmoObj;
};

TEST_F(TestOckHmmComposeDeviceMgrExtCopyHmo, CopyHmo_return_while_deviceid_not_equal)
{
    MOCKER(OckHmmHMOObjectIDGenerator::ParseDeviceId).stubs().with(eq(objIdInDevA)).will(returnValue(deviceIdA));
    MOCKER(OckHmmHMOObjectIDGenerator::ParseDeviceId).stubs().with(eq(objIdInDevB)).will(returnValue(deviceIdB));
    DoTestCopyHMODeviceIdNotEqual(
        hmoBytes, correctOffset, correctOffset, correctLength, HMM_ERROR_INPUT_PARAM_DEVICEID_NOT_EQUAL);
}
TEST_F(TestOckHmmComposeDeviceMgrExtCopyHmo, CopyHmo_return_while_deviceid_not_exists)
{
    MOCKER(OckHmmHeteroMemoryMgrParamCheck::CheckHmoIsValid).stubs().will(returnValue(HMM_SUCCESS));
    DoTestCopyHMODeviceIdEqual(
        hmoBytes, correctOffset, correctOffset, correctLength, HMM_ERROR_INPUT_PARAM_DEVICE_NOT_EXISTS);
}
TEST_F(TestOckHmmComposeDeviceMgrExtCopyHmo, CopyHmo_return_while_length_equals_zero)
{
    MOCKER(OckHmmHMOObjectIDGenerator::ParseDeviceId).stubs().will(returnValue(deviceIdA));
    MOCKER(OckHmmHeteroMemoryMgrParamCheck::CheckHmoIsValid).stubs().will(returnValue(HMM_SUCCESS));
    DoTestCopyHMODeviceIdEqual(hmoBytes, correctOffset, correctOffset, 0, HMM_SUCCESS);
}
}
}
}