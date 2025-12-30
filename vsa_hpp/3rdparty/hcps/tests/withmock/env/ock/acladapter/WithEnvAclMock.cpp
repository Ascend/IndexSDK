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

#include <unordered_set>
#include <thread>
#include <mutex>
#include "gtest/gtest.h"
#include "ock/acladapter/WithEnvAclMock.h"
namespace ock {
namespace acladapter {
namespace aclmock {

class MemoryClearGuard {
public:
    explicit MemoryClearGuard(void) : doAsan(true)
    {}
    void Add(uint8_t *addr)
    {
        std::lock_guard<std::mutex> guard(mutex);
        addrSet.insert(addr);
    }
    void Remove(uint8_t *addr)
    {
        std::lock_guard<std::mutex> guard(mutex);
        addrSet.erase(addr);
    }
    void StartNewAclMockAsan(void)
    {
        std::lock_guard<std::mutex> guard(mutex);
        addrSet.clear();
    }
    void DoAclMockAsan(void)
    {
        std::lock_guard<std::mutex> guard(mutex);
        for (auto addr : addrSet) {
            if (doAsan) {
                EXPECT_EQ(addr, nullptr);
            } else {
                delete[] addr;
            }
        }
        addrSet.clear();
    }
    void UnDoAclMockAsan(void)
    {
        std::lock_guard<std::mutex> guard(mutex);
        doAsan = false;
    }
    static MemoryClearGuard &Instance() noexcept
    {
        static MemoryClearGuard ins;
        return ins;
    }

private:
    bool doAsan;
    std::mutex mutex;
    std::unordered_set<uint8_t *> addrSet;
};
MemoryClearGuard &g_insOfMemoryClearGuard = MemoryClearGuard::Instance();
void StartNewAclMockAsan(void)
{
    MemoryClearGuard::Instance().StartNewAclMockAsan();
}
void DoAclMockAsan(void)
{
    MemoryClearGuard::Instance().DoAclMockAsan();
}
void UnDoAclMockAsan(void)
{
    MemoryClearGuard::Instance().UnDoAclMockAsan();
}
aclError FakeAclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    return FakeAclrtMallocHost(devPtr, size);
}
aclError FakeAclrtFree(void *devPtr)
{
    return FakeAclrtFreeHost(devPtr);
}
aclError FakeAclrtMemCpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind)
{
    return memcpy_s(dst, destMax, src, count);
}
aclError FakeAclrtMallocHost(void **devPtr, size_t size)
{
    uint8_t *addr = new uint8_t[size];
    *devPtr = addr;
    MemoryClearGuard::Instance().Add(addr);
    return ACL_SUCCESS;
}
aclError FakeAclrtFreeHost(void *devPtr)
{
    uint8_t *addr = (uint8_t *)devPtr;
    MemoryClearGuard::Instance().Remove(addr);
    if (devPtr != nullptr) {
        delete[] addr;
    }
    return ACL_SUCCESS;
}
aclError FakeAclrtGetMemInfo(aclrtMemAttr attr, size_t *free, size_t *total)
{
    *free = MOCK_FREE_MEMORY;
    *total = MOCK_TOTAL_MEMORY;
    return ACL_SUCCESS;
}
aclError FakeAclGetDeviceCount(uint32_t *count)
{
    *count = MAX_DEVICE_COUNT;
    return ACL_SUCCESS;
}
}  // namespace aclmock
}  // namespace acladapter
}  // namespace ock