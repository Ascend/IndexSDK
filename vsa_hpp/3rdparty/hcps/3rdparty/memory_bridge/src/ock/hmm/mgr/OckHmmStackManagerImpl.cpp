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

#include <list>
#include "ock/hmm/mgr/OckHmmStackManager.h"
#include "ock/log/OckLogger.h"

namespace ock {
namespace hmm {
OckHmmStackBuffer::OckHmmStackBuffer(uintptr_t address, uint64_t size) : addr(address), length(size)
{}
OckHmmStackBuffer::OckHmmStackBuffer(const OckHmmStackBuffer &other) : addr(other.addr), length(other.length)
{}
OckHmmStackBuffer::OckHmmStackBuffer(void) : addr(0), length(0)
{}
uintptr_t OckHmmStackBuffer::Address(void) const
{
    return addr;
}
uint64_t OckHmmStackBuffer::Size(void) const
{
    return length;
}
bool operator==(const OckHmmStackBuffer &lhs, const OckHmmStackBuffer &rhs)
{
    return lhs.Address() == rhs.Address() && lhs.Size() == rhs.Size();
}
class OckHmmStackManagerImpl : public OckHmmStackManager {
public:
    virtual ~OckHmmStackManagerImpl() noexcept = default;
    explicit OckHmmStackManagerImpl(std::shared_ptr<OckHmmHMObject> hmoObj)
        : curAddr(hmoObj->Addr()), leftLength(hmoObj->GetByteSize()), hmo(hmoObj)
    {}
    OckHmmStackBuffer GetBuffer(uint64_t length, OckHmmErrorCode &errorCode) override
    {
        if (length > leftLength) {
            OCK_HMM_LOG_ERROR(
                "Left memory(length=" << leftLength << ") not enough, need length=" << length << ". hmo:" << *hmo);
            errorCode = HMM_ERROR_STACK_MANAGE_SPACE_NOT_ENOUGH;
            return OckHmmStackBuffer();
        }
        if (length == 0ULL) {
            return OckHmmStackBuffer();
        }
        allocatedBufferList.emplace_back(curAddr, length);
        curAddr += length;
        leftLength -= length;
        return allocatedBufferList.back();
    }
    /*
    1. 当输入的buffer对应的addr为0时，返回
    2. 当输入的buffer对应的addr之后还有内存没有释放时，以warn记录错误, 返回
    3. 当输入的buffer不存在时，记录warn，返回
    */
    void ReleaseBuffer(const OckHmmStackBuffer &buffer) override
    {
        if (allocatedBufferList.empty() || buffer.Address() == 0ULL || buffer.Size() == 0ULL) {
            OCK_HMM_LOG_WARN("ReleaseBuffer(" << buffer << ") not exists or null");
            return;
        } else if (allocatedBufferList.back() == buffer) {
            allocatedBufferList.pop_back();
            if (allocatedBufferList.empty()) {
                curAddr = hmo->Addr();
                leftLength = hmo->GetByteSize();
            } else {
                curAddr = allocatedBufferList.back().Address() + allocatedBufferList.back().Size();
                leftLength = hmo->GetByteSize() - (curAddr - hmo->Addr());
            }
        } else if (allocatedBufferList.size() <= 1ULL) {
            OCK_HMM_LOG_WARN("ReleaseBuffer(" << buffer << ") not exists. stack list is empty.");
            return;
        } else if (allocatedBufferList.front() == buffer) {
            OCK_HMM_LOG_WARN(
                "ReleaseBuffer(" << buffer << ") is first data, stack list count=" << allocatedBufferList.size());
            allocatedBufferList.pop_front();
            return;
        } else if (ReleaseCaseMiddleData(buffer)) {
            OCK_HMM_LOG_WARN(
                "ReleaseBuffer(" << buffer << ") is first data, stack list count=" << allocatedBufferList.size());
        } else {
            OCK_HMM_LOG_WARN(
                "ReleaseBuffer(" << buffer << ") not exists stack list count=" << allocatedBufferList.size());
        }
    }

private:
    /*
    @brief 此处调用时，说明至少有2个数据, 调用者保证。
    */
    bool ReleaseCaseMiddleData(const OckHmmStackBuffer &buffer)
    {
        auto iter = allocatedBufferList.end();
        for (iter--; iter != allocatedBufferList.begin(); --iter) {
            if (*iter == buffer) {
                allocatedBufferList.erase(iter);
                OCK_HMM_LOG_WARN("ReleaseBuffer(" << buffer
                                                  << ") is middle data. current count:" << allocatedBufferList.size()
                                                  << " last buffer is " << allocatedBufferList.back());
                return true;
            }
        }
        return false;
    }
    uintptr_t curAddr;
    uintptr_t leftLength;
    std::shared_ptr<OckHmmHMObject> hmo;
    std::list<OckHmmStackBuffer> allocatedBufferList{};
};
std::shared_ptr<OckHmmStackManager> OckHmmStackManager::Create(std::shared_ptr<OckHmmHMObject> hmo)
{
    return std::make_shared<OckHmmStackManagerImpl>(hmo);
}
OckHmmStackBufferGuard::~OckHmmStackBufferGuard() noexcept
{
    stackMgr->ReleaseBuffer(buffer);
}
OckHmmStackBufferGuard::OckHmmStackBufferGuard(
    std::shared_ptr<OckHmmStackManager> stackManager, const OckHmmStackBuffer &stackBuffer)
    : buffer(stackBuffer), stackMgr(stackManager)
{}
uintptr_t OckHmmStackBufferGuard::Address(void) const
{
    return buffer.Address();
}
uint64_t OckHmmStackBufferGuard::Size(void) const
{
    return buffer.Size();
}
}  // namespace hmm
}  // namespace ock