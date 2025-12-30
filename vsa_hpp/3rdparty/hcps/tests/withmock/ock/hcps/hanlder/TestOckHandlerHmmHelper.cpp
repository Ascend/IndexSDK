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

#include <cstdlib>
#include <gtest/gtest.h>
#include "ock/hcps/error/OckHcpsErrorCode.h"
#include "ock/hcps/handler/OckHandlerHmmHelper.h"
#include "ock/hmm/mgr/MockOckHmmSingleDeviceMgr.h"
#include "ock/hcps/WithEnvOckHeteroHandler.h"

namespace ock {
namespace hcps {
namespace handler {
namespace helper {
class TestOckHandlerHmmHelper : public WithEnvOckHeteroHandler<testing::Test> {
public:
    using BaseT = WithEnvOckHeteroHandler<testing::Test>;
    hmm::MockOckHmmSingleDeviceMgr hmmMgr;

    void SetUp(void) override
    {
        BaseT::SetUp();
        handler = CreateSingleDeviceHandler(errorCode);
    }
    void TearDown(void) override
    {
        handler.reset();
        BaseT::TearDown();
    }

    void ExpectCallHostAlloc(uint32_t callTimes, hmm::OckHmmErrorCode retCode)
    {
        EXPECT_CALL(hmmMgr, AllocImpl(hmoBytes, hmm::OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY))
            .Times(callTimes)
            .WillRepeatedly(testing::Return(std::make_pair(retCode, std::shared_ptr<hmm::OckHmmHMObject>())));
    }
    void ExpectCallDeviceAlloc(uint32_t callTimes, hmm::OckHmmErrorCode retCode)
    {
        EXPECT_CALL(hmmMgr, AllocImpl(hmoBytes, hmm::OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY))
            .Times(callTimes)
            .WillRepeatedly(testing::Return(std::make_pair(retCode, std::shared_ptr<hmm::OckHmmHMObject>())));
    }

    uint64_t hmoBytes{1024ULL};
    hmm::OckHmmErrorCode errorCode{hmm::HMM_SUCCESS};
    std::shared_ptr<OckHeteroHandler> handler;
};

TEST_F(TestOckHandlerHmmHelper, MakeHostHmoDeque_success)
{
    const uint64_t hmoCount = 3UL;
    ExpectCallHostAlloc(hmoCount, hmm::HMM_SUCCESS);
    auto ret = MakeHostHmoDeque(hmmMgr, hmoBytes, hmoCount, errorCode);
    EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
    EXPECT_EQ(ret.size(), hmoCount);
}
TEST_F(TestOckHandlerHmmHelper, MakeDeviceHmoDeque_success)
{
    const uint64_t hmoCount = 3UL;
    ExpectCallDeviceAlloc(hmoCount, hmm::HMM_SUCCESS);
    auto ret = MakeDeviceHmoDeque(hmmMgr, hmoBytes, hmoCount, errorCode);
    EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
    EXPECT_EQ(ret.size(), hmoCount);
}
TEST_F(TestOckHandlerHmmHelper, MakeHostHmoDeque_failed)
{
    const uint64_t hmoCount = 3UL;
    ExpectCallHostAlloc(1UL, hmm::HMM_ERROR_HMO_BUFFER_NOT_ALLOCED);
    auto ret = MakeHostHmoDeque(hmmMgr, hmoBytes, hmoCount, errorCode);
    EXPECT_EQ(errorCode, hmm::HMM_ERROR_HMO_BUFFER_NOT_ALLOCED);
    EXPECT_EQ(ret.size(), hmoCount);
}
TEST_F(TestOckHandlerHmmHelper, MakeDeviceHmoDeque_failed)
{
    const uint64_t hmoCount = 3UL;
    ExpectCallDeviceAlloc(1UL, hmm::HMM_ERROR_HMO_BUFFER_NOT_ALLOCED);
    auto ret = MakeDeviceHmoDeque(hmmMgr, hmoBytes, hmoCount, errorCode);
    EXPECT_EQ(errorCode, hmm::HMM_ERROR_HMO_BUFFER_NOT_ALLOCED);
    EXPECT_EQ(ret.size(), hmoCount);
}
TEST_F(TestOckHandlerHmmHelper, MakeHostHmo)
{
    auto ret = MakeHostHmo(*handler, hmoBytes, errorCode);
    EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
    EXPECT_EQ(ret->GetByteSize(), hmoBytes);
}
TEST_F(TestOckHandlerHmmHelper, MakeHostHmoWithErrorCode)
{
    errorCode = hmm::HMM_ERROR_INPUT_PARAM_EMPTY;
    auto ret = MakeHostHmo(*handler, hmoBytes, errorCode);
    EXPECT_EQ(ret, nullptr);
}
TEST_F(TestOckHandlerHmmHelper, MakeDeviceHmo)
{
    auto ret = MakeDeviceHmo(*handler, hmoBytes, errorCode);
    EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
    EXPECT_EQ(ret->GetByteSize(), hmoBytes);
}
TEST_F(TestOckHandlerHmmHelper, MakeDeviceHmoWithErrorCode)
{
    errorCode = hmm::HMM_ERROR_INPUT_PARAM_EMPTY;
    auto ret = MakeDeviceHmo(*handler, hmoBytes, errorCode);
    EXPECT_EQ(ret, nullptr);
}
TEST_F(TestOckHandlerHmmHelper, AddTroupe)
{
    OckHeteroOperatorTroupe troupe;
    troupe.push_back(std::make_shared<OckHeteroOperatorGroup>());
    troupe.back()->push_back(OckSimpleHeteroOperator<acladapter::OckTaskResourceType::HOST_CPU>::Create(
        [](OckHeteroStreamContext &) { return hmm::HMM_SUCCESS; }));
    auto stream = MakeStream(*handler, errorCode);
    EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
    stream->AddOps(troupe);
    EXPECT_EQ(hmm::HMM_SUCCESS, stream->WaitExecComplete());
}
TEST_F(TestOckHandlerHmmHelper, MergeMultiHMObjectsToHost)
{
    std::vector<std::shared_ptr<hmm::OckHmmHMObject>> deviceHmoVector;
    uint32_t HmoNum = 4UL;
    for (uint32_t i = 0; i < HmoNum; ++i) {
        auto ret = MakeDeviceHmo(*handler, hmoBytes, errorCode);
        deviceHmoVector.push_back(ret);
    }
    auto mergeResult = MergeMultiHMObjectsToHost(*handler, deviceHmoVector, errorCode);
    EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
    EXPECT_EQ(mergeResult->GetByteSize(), hmoBytes * HmoNum);
    for (uint32_t i = 0; i < HmoNum; ++i) {
        EXPECT_EQ(memcmp(reinterpret_cast<uint8_t *>(mergeResult->Addr()) + hmoBytes * i,
            reinterpret_cast<uint8_t *>(deviceHmoVector.at(i)->Addr()), hmoBytes),
            0);
    }
}
}  // namespace helper
}  // namespace handler
}  // namespace hcps
}  // namespace ock
