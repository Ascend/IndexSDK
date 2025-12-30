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
#include <thread>
#include <chrono>
#include "gtest/gtest.h"
#include "acl/acl.h"
#include "mockcpp/mockcpp.hpp"
#include "ock/hmm/mgr/WithEnvOckHmmSingleDeviceMgrExt.h"
namespace ock {
namespace hmm {
namespace test {
class TestOckHmmHMOObject : public WithEnvOckHmmSingleDeviceMgrExt<testing::Test> {
public:
    using BaseT = WithEnvOckHmmSingleDeviceMgrExt<testing::Test>;
    void ExpectBuffer(std::shared_ptr<OckHmmHMOBuffer> buffer, std::shared_ptr<OckHmmHMObject> hmo, uint64_t offset,
        uint64_t length, OckHmmHeteroMemoryLocation location)
    {
        ExpectBufferExceptAddr(buffer, hmo, offset, length, location);
        EXPECT_EQ(buffer->Address(), hmo->Addr());
        EXPECT_EQ(buffer->FlushData(), HMM_SUCCESS);
        EXPECT_EQ(buffer->ErrorCode(), HMM_SUCCESS);
    }
    void ExpectBufferExceptAddr(std::shared_ptr<OckHmmHMOBuffer> buffer, std::shared_ptr<OckHmmHMObject> hmo,
        uint64_t offset, uint64_t length, OckHmmHeteroMemoryLocation location)
    {
        ASSERT_NE(buffer.get(), nullptr);
        EXPECT_EQ(buffer->Location(), location);
        EXPECT_EQ(buffer->Offset(), offset);
        EXPECT_EQ(buffer->Size(), length);
        EXPECT_EQ(buffer->GetId(), hmo->GetId());
        EXPECT_NE(buffer->Address(), 0UL);
    }
    void ExpectReleasedBuffer(std::shared_ptr<OckHmmHMOBuffer> buffer)
    {
        EXPECT_EQ(buffer->Location(), OckHmmHeteroMemoryLocation::DEVICE_DDR);  // 非法buffer的Location固定
        EXPECT_EQ(buffer->Address(), 0UL);
        EXPECT_EQ(buffer->GetId(), 0UL);
        EXPECT_EQ(buffer->Size(), 0UL);
        EXPECT_EQ(buffer->Offset(), 0UL);
        EXPECT_EQ(buffer->ErrorCode(), HMM_ERROR_WAIT_TIME_OUT);
        EXPECT_EQ(buffer->FlushData(), HMM_ERROR_HMO_BUFFER_NOT_ALLOCED);
    }
};
TEST_F(TestOckHmmHMOObject, getbuffer_succeed_while_same_location)
{
    const uint32_t hmoBytes = 1024U * 1024U * 64U;
    auto hmo = this->AllocHMO(*devDataAlloc, hmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
    ASSERT_NE(hmo.get(), nullptr);
    const uint64_t offset = 0;
    MockGetUsedInfo(*devSwapAlloc, hmoBytes, hmoBytes, 0ULL, devSwapLeftBytes);
    auto buffRet = hmo->GetBuffer(OckHmmHeteroMemoryLocation::DEVICE_DDR, offset, hmoBytes);
    ExpectBuffer(buffRet, hmo, offset, hmoBytes, OckHmmHeteroMemoryLocation::DEVICE_DDR);
}
TEST_F(TestOckHmmHMOObject, getbuffer_succeed_while_invalid_offset_length)
{
    const uint32_t hmoBytes = 1024U * 1024U * 64U;
    auto hmo = this->AllocHMO(*devDataAlloc, hmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
    ASSERT_NE(hmo.get(), nullptr);
    const uint64_t offset = hmoBytes;
    MockGetUsedInfo(*devSwapAlloc, hmoBytes, hmoBytes, 0ULL, devSwapLeftBytes);
    auto buffRet = hmo->GetBuffer(OckHmmHeteroMemoryLocation::DEVICE_DDR, offset, hmoBytes);
    EXPECT_EQ(buffRet.get(), nullptr);
    buffRet = hmo->GetBuffer(OckHmmHeteroMemoryLocation::DEVICE_DDR, 0, hmoBytes + 1);
    EXPECT_EQ(buffRet.get(), nullptr);
}
TEST_F(TestOckHmmHMOObject, getbuffer_async_succeed_while_same_location)
{
    const uint32_t hmoBytes = 1024U * 1024U * 64U;
    auto hmo = this->AllocHMO(*devDataAlloc, hmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
    ASSERT_NE(hmo.get(), nullptr);
    const uint64_t offset = 0;
    MockGetUsedInfo(*devSwapAlloc, hmoBytes, hmoBytes, 0ULL, devSwapLeftBytes);
    auto asyncResult = hmo->GetBufferAsync(OckHmmHeteroMemoryLocation::DEVICE_DDR, offset, hmoBytes);
    ASSERT_NE(asyncResult.get(), nullptr);
    auto buffRet = asyncResult->WaitResult();
    ExpectBuffer(buffRet, hmo, offset, hmoBytes, OckHmmHeteroMemoryLocation::DEVICE_DDR);
}
TEST_F(TestOckHmmHMOObject, getbuffer_succeed_while_host_to_device)
{
    const uint32_t hmoBytes = 1024U * 1024U * 64U;
    auto hmo = this->AllocHMO(*hostDataAlloc, hmoBytes, OckHmmMemoryAllocatePolicy::LOCAL_HOST_ONLY);
    const uint64_t offset = 0;
    this->MockAllocFreeWithNewDelete(*devSwapAlloc);
    MOCKER(aclrtMemcpy).stubs().will(returnValue(ACL_SUCCESS));
    MockGetUsedInfo(*devSwapAlloc, hmoBytes, hmoBytes, 0ULL, devSwapLeftBytes);
    auto buffRet = hmo->GetBuffer(OckHmmHeteroMemoryLocation::DEVICE_DDR, offset, hmoBytes);
    ExpectBufferExceptAddr(buffRet, hmo, offset, hmoBytes, OckHmmHeteroMemoryLocation::DEVICE_DDR);
    EXPECT_EQ(buffRet->ErrorCode(), HMM_SUCCESS);
    EXPECT_EQ(buffRet->FlushData(), HMM_SUCCESS);
}
TEST_F(TestOckHmmHMOObject, getbuffer_succeed_while_device_to_host)
{
    const uint32_t hmoBytes = 1024U * 1024U * 64U;
    auto hmo = this->AllocHMO(*devDataAlloc, hmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
    const uint64_t offset = 0;
    this->MockAllocFreeWithNewDelete(*hostSwapAlloc);
    MOCKER(aclrtMemcpy).stubs().will(returnValue(ACL_SUCCESS));
    MockGetUsedInfo(*hostSwapAlloc, hmoBytes, hmoBytes, 0ULL, hostSwapLeftBytes);
    auto buffRet = hmo->GetBuffer(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, offset, hmoBytes);
    ExpectBufferExceptAddr(buffRet, hmo, offset, hmoBytes, OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY);
    EXPECT_EQ(buffRet->ErrorCode(), HMM_SUCCESS);
    EXPECT_EQ(buffRet->FlushData(), HMM_SUCCESS);
}
TEST_F(TestOckHmmHMOObject, getbuffer_succeed_while_acl_move_failed)
{
    const uint32_t hmoBytes = 1024U * 1024U * 64U;
    auto hmo = this->AllocHMO(*devDataAlloc, hmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
    const uint64_t offset = 0;
    this->MockAllocFreeWithNewDelete(*hostSwapAlloc);
    MOCKER(aclrtMemcpy).stubs().will(returnValue(ACL_ERROR_RT_MEMORY_ALLOCATION));
    MockGetUsedInfo(*hostSwapAlloc, hmoBytes, hmoBytes, 0ULL, hostSwapLeftBytes);
    auto buffRet = hmo->GetBuffer(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, offset, hmoBytes);
    ExpectBufferExceptAddr(buffRet, hmo, offset, hmoBytes, OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY);
    EXPECT_EQ(buffRet->ErrorCode(), ACL_ERROR_RT_MEMORY_ALLOCATION);
}
TEST_F(TestOckHmmHMOObject, release_buffer_with_same_location)
{
    const uint32_t hmoBytes = 1024U * 1024U * 64U;
    auto hmo = this->AllocHMO(*devDataAlloc, hmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
    const uint64_t offset = 0;
    MockGetUsedInfo(*devSwapAlloc, hmoBytes, hmoBytes, 0ULL, devSwapLeftBytes);
    auto buffRet = hmo->GetBuffer(OckHmmHeteroMemoryLocation::DEVICE_DDR, offset, hmoBytes);
    hmo->ReleaseBuffer(buffRet);
    ExpectReleasedBuffer(buffRet);
    EXPECT_EQ(buffRet->FlushData(), HMM_ERROR_HMO_BUFFER_NOT_ALLOCED);
}
TEST_F(TestOckHmmHMOObject, release_buffer_with_diff_location)
{
    const uint32_t hmoBytes = 1024U * 1024U * 64U;
    auto hmo = this->AllocHMO(*devDataAlloc, hmoBytes, OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
    const uint64_t offset = 0;
    this->MockAllocFreeWithNewDelete(*hostSwapAlloc);
    MockGetUsedInfo(*hostSwapAlloc, hmoBytes, hmoBytes, 0ULL, hostSwapLeftBytes);
    auto buffRet = hmo->GetBuffer(OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, offset, hmoBytes);
    hmo->ReleaseBuffer(buffRet);
    ExpectReleasedBuffer(buffRet);
    EXPECT_EQ(buffRet->FlushData(), HMM_ERROR_HMO_BUFFER_NOT_ALLOCED);
}
}  // namespace test
}  // namespace hmm
}  // namespace ock
