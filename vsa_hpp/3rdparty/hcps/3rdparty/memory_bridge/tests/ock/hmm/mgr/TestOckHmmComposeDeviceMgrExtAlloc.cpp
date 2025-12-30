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
#include "ock/hmm/mgr/MockOckHmmHMObject.h"
#include "ock/hmm/mgr/MockOckHmmMemoryGuard.h"
#include "ock/hmm/mgr/checker/OckHmmComposeDeviceMgrParamCheck.h"
#include "ock/hmm/mgr/OckHmmComposeDeviceMgrExt.h"
#include "ock/hmm/mgr/WithEnvOckHmmComposeDeviceMgrExt.h"

namespace ock {
namespace hmm {
namespace test {

class TestOckHmmComposeDeviceMgrExtAlloc : public WithEnvOckHmmComposeDeviceMgrExt<testing::Test> {
public:
    using allocPolicy = OckHmmMemoryAllocatePolicy;
    /*
    @brief 这里的objIdInDev是任意数，这里用不同的素数方便定位问题
    */
    TestOckHmmComposeDeviceMgrExtAlloc()
        : WithEnvOckHmmComposeDeviceMgrExt<testing::Test>(), needHmoBytes(64U * 1024U * 1024U), objIdInDevA(23U),
          objIdInDevB(29U), objIdInDevC(37U), objIdInDevD(43U)
    {}
    void MockAllDeviceMgrAlloc(uint64_t hmoBytes, allocPolicy policy)
    {
        MockDeviceMgrAlloc(devMgrA, hmoBytes, policy, objIdInDevA);
        MockDeviceMgrAlloc(devMgrB, hmoBytes, policy, objIdInDevB);
        MockDeviceMgrAlloc(devMgrC, hmoBytes, policy, objIdInDevC);
        MockDeviceMgrAlloc(devMgrD, hmoBytes, policy, objIdInDevD);

        MockDeviceMgrMalloc(devMgrA, hmoBytes, policy);
        MockDeviceMgrMalloc(devMgrB, hmoBytes, policy);
        MockDeviceMgrMalloc(devMgrC, hmoBytes, policy);
        MockDeviceMgrMalloc(devMgrD, hmoBytes, policy);
    }
    void MockDeviceMgrAlloc(MockOckHmmSingleDeviceMgr &devMgr, uint64_t hmoBytes, allocPolicy policy,
        OckHmmHMOObjectID objId)
    {
        auto hmoObj = new MockOckHmmHMObject();
        EXPECT_CALL(*hmoObj, GetId()).WillRepeatedly(testing::Return(objId));
        EXPECT_CALL(*hmoObj, GetByteSize()).WillRepeatedly(testing::Return(hmoBytes));
        EXPECT_CALL(*hmoObj, Addr()).WillRepeatedly(testing::Return((uintptr_t)(hmoObj)));
        if (policy == allocPolicy::LOCAL_HOST_ONLY) {
            EXPECT_CALL(*hmoObj, Location())
                .WillRepeatedly(testing::Return(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY));
        } else {
            EXPECT_CALL(*hmoObj, Location()).WillRepeatedly(testing::Return(OckHmmHeteroMemoryLocation::DEVICE_DDR));
        }
        EXPECT_CALL(devMgr, AllocImpl(hmoBytes, policy))
            .WillRepeatedly(testing::Return(std::make_pair(HMM_SUCCESS, std::shared_ptr<OckHmmHMObject>(hmoObj))));
    }
    void MockDeviceMgrMalloc(MockOckHmmSingleDeviceMgr &devMgr, uint64_t hmoBytes, allocPolicy policy)
    {
        EXPECT_CALL(devMgr, Malloc(hmoBytes, policy)).WillRepeatedly(testing::Invoke([hmoBytes, policy]() {
            auto hmoObj = new MockOckHmmMemoryGuard();
            EXPECT_CALL(*hmoObj, ByteSize()).WillRepeatedly(testing::Return(hmoBytes));
            EXPECT_CALL(*hmoObj, Addr()).WillRepeatedly(testing::Return((uintptr_t)(hmoObj)));
            if (policy == allocPolicy::LOCAL_HOST_ONLY) {
                EXPECT_CALL(*hmoObj, Location())
                    .WillRepeatedly(testing::Return(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY));
            } else {
                EXPECT_CALL(*hmoObj, Location())
                    .WillRepeatedly(testing::Return(OckHmmHeteroMemoryLocation::DEVICE_DDR));
            }
            return std::unique_ptr<OckHmmMemoryGuard>(hmoObj);
        }));
    }
    void MockAllDeviceMgrAllocNoSpace(uint64_t hmoBytes, allocPolicy policy)
    {
        MockDeviceMgrAllocNoSpace(devMgrA, hmoBytes, policy);
        MockDeviceMgrAllocNoSpace(devMgrB, hmoBytes, policy);
        MockDeviceMgrAllocNoSpace(devMgrC, hmoBytes, policy);
        MockDeviceMgrAllocNoSpace(devMgrD, hmoBytes, policy);
    }
    void MockDeviceMgrAllocNoSpace(MockOckHmmSingleDeviceMgr &devMgr, uint64_t hmoBytes, allocPolicy policy)
    {
        EXPECT_CALL(devMgr, AllocImpl(hmoBytes, policy))
            .WillRepeatedly(
            testing::Return(std::make_pair(HMM_ERROR_SPACE_NOT_ENOUGH, std::shared_ptr<OckHmmHMObject>())));
        EXPECT_CALL(devMgr, Malloc(hmoBytes, policy)).WillRepeatedly(testing::Invoke([]() {
            return std::unique_ptr<OckHmmMemoryGuard>(nullptr);
        }));
    }

    uint64_t needHmoBytes;
    OckHmmHMOObjectID objIdInDevA;
    OckHmmHMOObjectID objIdInDevB;
    OckHmmHMOObjectID objIdInDevC;
    OckHmmHMOObjectID objIdInDevD;
};

TEST_F(TestOckHmmComposeDeviceMgrExtAlloc, alloc_while_device_not_exists)
{
    EXPECT_NE(composeMgr->Alloc(unknownDeviceId, needHmoBytes, allocPolicy::LOCAL_HOST_ONLY).first, HMM_SUCCESS);
}
TEST_F(TestOckHmmComposeDeviceMgrExtAlloc, alloc_while_all_no_space)
{
    MockAllDeviceMgrAllocNoSpace(needHmoBytes, allocPolicy::DEVICE_DDR_ONLY);
    MockAllDeviceMgrAllocNoSpace(needHmoBytes, allocPolicy::LOCAL_HOST_ONLY);
    MockAllDeviceMgrAllocNoSpace(needHmoBytes, allocPolicy::DEVICE_DDR_FIRST);
    EXPECT_NE(composeMgr->Alloc(needHmoBytes, allocPolicy::LOCAL_HOST_ONLY).first, HMM_SUCCESS);
    EXPECT_NE(composeMgr->Alloc(needHmoBytes, allocPolicy::DEVICE_DDR_FIRST).first, HMM_SUCCESS);
    EXPECT_NE(composeMgr->Alloc(deviceIdA, needHmoBytes, allocPolicy::LOCAL_HOST_ONLY).first, HMM_SUCCESS);
    EXPECT_NE(composeMgr->Alloc(deviceIdA, needHmoBytes, allocPolicy::DEVICE_DDR_FIRST).first, HMM_SUCCESS);
    EXPECT_TRUE(composeMgr->Malloc(needHmoBytes, allocPolicy::LOCAL_HOST_ONLY).get() == nullptr);
    EXPECT_TRUE(composeMgr->Malloc(needHmoBytes, allocPolicy::DEVICE_DDR_FIRST).get() == nullptr);
}
TEST_F(TestOckHmmComposeDeviceMgrExtAlloc, alloc_while_no_space_in_host_and_enough_space_in_ddr)
{
    MockAllDeviceMgrAlloc(needHmoBytes, allocPolicy::DEVICE_DDR_ONLY);
    MockAllDeviceMgrAlloc(needHmoBytes, allocPolicy::DEVICE_DDR_FIRST);
    MockAllDeviceMgrAllocNoSpace(needHmoBytes, allocPolicy::LOCAL_HOST_ONLY);
    EXPECT_EQ(composeMgr->Alloc(needHmoBytes, allocPolicy::DEVICE_DDR_ONLY).first, HMM_SUCCESS);
    EXPECT_NE(composeMgr->Alloc(needHmoBytes, allocPolicy::LOCAL_HOST_ONLY).first, HMM_SUCCESS);
    EXPECT_EQ(composeMgr->Alloc(needHmoBytes, allocPolicy::DEVICE_DDR_FIRST).first, HMM_SUCCESS);
    EXPECT_EQ(composeMgr->Alloc(deviceIdB, needHmoBytes, allocPolicy::DEVICE_DDR_ONLY).first, HMM_SUCCESS);
    EXPECT_NE(composeMgr->Alloc(deviceIdB, needHmoBytes, allocPolicy::LOCAL_HOST_ONLY).first, HMM_SUCCESS);
    EXPECT_EQ(composeMgr->Alloc(deviceIdB, needHmoBytes, allocPolicy::DEVICE_DDR_FIRST).first, HMM_SUCCESS);
    EXPECT_FALSE(composeMgr->Malloc(needHmoBytes, allocPolicy::DEVICE_DDR_ONLY).get() == nullptr);
    EXPECT_TRUE(composeMgr->Malloc(needHmoBytes, allocPolicy::LOCAL_HOST_ONLY).get() == nullptr);
    EXPECT_FALSE(composeMgr->Malloc(needHmoBytes, allocPolicy::DEVICE_DDR_FIRST).get() == nullptr);
}
TEST_F(TestOckHmmComposeDeviceMgrExtAlloc, alloc_while_no_space_in_ddr_and_enough_space_in_host)
{
    MockAllDeviceMgrAlloc(needHmoBytes, allocPolicy::LOCAL_HOST_ONLY);
    MockAllDeviceMgrAlloc(needHmoBytes, allocPolicy::DEVICE_DDR_FIRST);
    MockAllDeviceMgrAllocNoSpace(needHmoBytes, allocPolicy::DEVICE_DDR_ONLY);
    EXPECT_NE(composeMgr->Alloc(needHmoBytes, allocPolicy::DEVICE_DDR_ONLY).first, HMM_SUCCESS);
    EXPECT_EQ(composeMgr->Alloc(needHmoBytes, allocPolicy::LOCAL_HOST_ONLY).first, HMM_SUCCESS);
    EXPECT_EQ(composeMgr->Alloc(needHmoBytes, allocPolicy::DEVICE_DDR_FIRST).first, HMM_SUCCESS);
    EXPECT_NE(composeMgr->Alloc(deviceIdD, needHmoBytes, allocPolicy::DEVICE_DDR_ONLY).first, HMM_SUCCESS);
    EXPECT_EQ(composeMgr->Alloc(deviceIdD, needHmoBytes, allocPolicy::LOCAL_HOST_ONLY).first, HMM_SUCCESS);
    EXPECT_EQ(composeMgr->Alloc(deviceIdD, needHmoBytes, allocPolicy::DEVICE_DDR_FIRST).first, HMM_SUCCESS);
    EXPECT_TRUE(composeMgr->Malloc(needHmoBytes, allocPolicy::DEVICE_DDR_ONLY).get() == nullptr);
    EXPECT_FALSE(composeMgr->Malloc(needHmoBytes, allocPolicy::LOCAL_HOST_ONLY).get() == nullptr);
    EXPECT_FALSE(composeMgr->Malloc(needHmoBytes, allocPolicy::DEVICE_DDR_FIRST).get() == nullptr);
}
TEST_F(TestOckHmmComposeDeviceMgrExtAlloc, alloc_while_one_dev_has_ddr_space)
{
    MockDeviceMgrAllocNoSpace(devMgrA, needHmoBytes, allocPolicy::DEVICE_DDR_ONLY);
    MockDeviceMgrAllocNoSpace(devMgrB, needHmoBytes, allocPolicy::DEVICE_DDR_ONLY);
    MockDeviceMgrAlloc(devMgrC, needHmoBytes, allocPolicy::DEVICE_DDR_ONLY, objIdInDevC);
    MockDeviceMgrMalloc(devMgrC, needHmoBytes, allocPolicy::DEVICE_DDR_ONLY);
    MockDeviceMgrAllocNoSpace(devMgrD, needHmoBytes, allocPolicy::DEVICE_DDR_ONLY);

    MockAllDeviceMgrAllocNoSpace(needHmoBytes, allocPolicy::LOCAL_HOST_ONLY);

    EXPECT_EQ(composeMgr->Alloc(needHmoBytes, allocPolicy::DEVICE_DDR_ONLY).first, HMM_SUCCESS);
    EXPECT_NE(composeMgr->Alloc(needHmoBytes, allocPolicy::LOCAL_HOST_ONLY).first, HMM_SUCCESS);
    EXPECT_EQ(composeMgr->Alloc(needHmoBytes, allocPolicy::DEVICE_DDR_FIRST).first, HMM_SUCCESS);
    EXPECT_EQ(composeMgr->Alloc(needHmoBytes, allocPolicy::DEVICE_DDR_FIRST).second->GetId(), objIdInDevC);
    EXPECT_FALSE(composeMgr->Malloc(needHmoBytes, allocPolicy::DEVICE_DDR_ONLY).get() == nullptr);
    EXPECT_FALSE(composeMgr->Malloc(needHmoBytes, allocPolicy::DEVICE_DDR_FIRST).get() == nullptr);
    EXPECT_TRUE(composeMgr->Malloc(needHmoBytes, allocPolicy::LOCAL_HOST_ONLY).get() == nullptr);
}
TEST_F(TestOckHmmComposeDeviceMgrExtAlloc, alloc_while_one_dev_has_host_space)
{
    MockDeviceMgrAllocNoSpace(devMgrA, needHmoBytes, allocPolicy::LOCAL_HOST_ONLY);
    MockDeviceMgrAllocNoSpace(devMgrB, needHmoBytes, allocPolicy::LOCAL_HOST_ONLY);
    MockDeviceMgrAllocNoSpace(devMgrC, needHmoBytes, allocPolicy::LOCAL_HOST_ONLY);
    MockDeviceMgrAlloc(devMgrD, needHmoBytes, allocPolicy::LOCAL_HOST_ONLY, objIdInDevD);
    MockDeviceMgrMalloc(devMgrD, needHmoBytes, allocPolicy::LOCAL_HOST_ONLY);
    MockAllDeviceMgrAllocNoSpace(needHmoBytes, allocPolicy::DEVICE_DDR_ONLY);

    EXPECT_NE(composeMgr->Alloc(needHmoBytes, allocPolicy::DEVICE_DDR_ONLY).first, HMM_SUCCESS);
    EXPECT_EQ(composeMgr->Alloc(needHmoBytes, allocPolicy::LOCAL_HOST_ONLY).first, HMM_SUCCESS);
    EXPECT_EQ(composeMgr->Alloc(needHmoBytes, allocPolicy::DEVICE_DDR_FIRST).first, HMM_SUCCESS);
    EXPECT_EQ(composeMgr->Alloc(needHmoBytes, allocPolicy::DEVICE_DDR_FIRST).second->GetId(), objIdInDevD);
    EXPECT_TRUE(composeMgr->Malloc(needHmoBytes, allocPolicy::DEVICE_DDR_ONLY).get() == nullptr);
    EXPECT_FALSE(composeMgr->Malloc(needHmoBytes, allocPolicy::DEVICE_DDR_FIRST).get() == nullptr);
    EXPECT_FALSE(composeMgr->Malloc(needHmoBytes, allocPolicy::LOCAL_HOST_ONLY).get() == nullptr);
}
}  // namespace test
}  // namespace hmm
}  // namespace ock