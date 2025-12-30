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


#ifndef OCK_MEMORY_BRIDGE_OCK_ADAPTER_TASK_WITH_ENV_ACL_MOCK_H
#define OCK_MEMORY_BRIDGE_OCK_ADAPTER_TASK_WITH_ENV_ACL_MOCK_H
#include "securec.h"
#include "mockcpp/mockcpp.hpp"
#include "acl/acl.h"
namespace ock {
namespace acladapter {
namespace aclmock {
const size_t MOCK_FREE_MEMORY = 515ULL * 1024ULL * 1024ULL * 1024ULL;
const size_t MOCK_TOTAL_MEMORY = 520ULL * 1024ULL * 1024ULL * 1024ULL;

aclError FakeAclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy);
aclError FakeAclrtFree(void *devPtr);
aclError FakeAclrtMallocHost(void **devPtr, size_t size);
aclError FakeAclrtFreeHost(void *devPtr);
aclError FakeAclrtMemCpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind);
aclError FakeAclrtGetMemInfo(aclrtMemAttr attr, size_t *free, size_t *total);
// 开始AclMock的新的地址消毒任务，会备份之前的内存泄露数据
void StartNewAclMockAsan(void);
// 做AclMock模块的地址消毒(会做消毒，标识当前的错误，不会主动释放内存)
void DoAclMockAsan(void);
// 不做AclMock模块的地址消毒(不做消毒，并主动释放通过本模块申请的内存)
void UnDoAclMockAsan(void);

const uint32_t MAX_DEVICE_COUNT = 8U;
aclError FakeAclGetDeviceCount(uint32_t *count);
}  // namespace aclmock
template <typename BaseT>
class WithEnvAclMock : public BaseT {
public:
    void SetUp(void) override
    {
        aclmock::StartNewAclMockAsan();
        BaseT::SetUp();
        MOCKER(aclInit).stubs().will(returnValue(ACL_SUCCESS));
        MOCKER(aclFinalize).stubs().will(returnValue(ACL_SUCCESS));
        MOCKER(aclrtSetDevice).stubs().will(returnValue(ACL_SUCCESS));
        MOCKER(aclrtResetDevice).stubs().will(returnValue(ACL_SUCCESS));
        MOCKER(aclrtCreateContext).stubs().will(returnValue(ACL_SUCCESS));
        MOCKER(aclrtDestroyContext).stubs().will(returnValue(ACL_SUCCESS));
        MOCKER(aclrtMalloc).stubs().will(invoke(aclmock::FakeAclrtMalloc));
        MOCKER(aclrtFree).stubs().will(invoke(aclmock::FakeAclrtFree));
        MOCKER(aclrtMallocHost).stubs().will(invoke(aclmock::FakeAclrtMallocHost));
        MOCKER(aclrtFreeHost).stubs().will(invoke(aclmock::FakeAclrtFreeHost));
        MOCKER(aclrtMemcpy).stubs().will(invoke(aclmock::FakeAclrtMemCpy));
        MOCKER(aclrtGetMemInfo).stubs().will(invoke(aclmock::FakeAclrtGetMemInfo));
        MOCKER(aclrtGetDeviceCount).stubs().will(invoke(aclmock::FakeAclGetDeviceCount));
        MOCKER(aclrtCreateStream).stubs().will(returnValue(ACL_SUCCESS));
        MOCKER(aclrtSynchronizeStream).stubs().will(returnValue(ACL_SUCCESS));
        MOCKER(aclrtSynchronizeStreamWithTimeout).stubs().will(returnValue(ACL_SUCCESS));
        MOCKER(aclrtDestroyStream).stubs().will(returnValue(ACL_SUCCESS));
    }
    void TearDown(void) override
    {
        BaseT::TearDown();
        GlobalMockObject::verify();
        aclmock::DoAclMockAsan();
    }
};
}  // namespace acladapter
}  // namespace ock
#endif