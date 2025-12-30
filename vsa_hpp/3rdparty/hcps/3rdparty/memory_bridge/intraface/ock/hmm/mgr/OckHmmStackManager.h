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


#ifndef OCK_MEMORY_BRIDGE_HMM_STACK_MANAGER_H
#define OCK_MEMORY_BRIDGE_HMM_STACK_MANAGER_H
#include <cstdint>
#include <utility>
#include <ostream>
#include "ock/hmm/mgr/OckHmmHMObject.h"

namespace ock {
namespace hmm {
/*
@brief 本类对HMO对象做类Stack的管理，可以基于HMO对象做内存的二次分配
本管理对象适合于事务型内存分配管理。
本管理对象的内存方式是先进后出
分配速度比HMM快。
*/
class OckHmmStackBuffer : public OckHmmHMOBufferBase {
public:
    virtual ~OckHmmStackBuffer() noexcept = default;
    OckHmmStackBuffer(uintptr_t address, uint64_t size);
    OckHmmStackBuffer(const OckHmmStackBuffer &stackBuffer);
    OckHmmStackBuffer(void);
    uintptr_t Address(void) const;
    uint64_t Size(void) const;

private:
    uintptr_t addr;
    uint64_t length;
};
bool operator==(const OckHmmStackBuffer &lhs, const OckHmmStackBuffer &rhs);

/*
@brief: 为了性能，本对象并不支持多线程并发调用
*/
class OckHmmStackManager {
public:
    virtual ~OckHmmStackManager() noexcept = default;
    /*
    @brief 此处如果异常，错误是内存不够
    @param errorCode 错误码，忽略输入值，当空间不足时设置错误， 并返回空的OckHmmStackBuffer
    */
    virtual OckHmmStackBuffer GetBuffer(uint64_t length, OckHmmErrorCode &errorCode) = 0;
    /*
    @brief GetBuffer与Release应该一一对应，正常应该满足后GetBuffer数据先ReleaseBuffer
    内存申请和归还必须是一一对应的， 不能申请空间与归还空间不一致
    1. 当输入的buffer对应的addr为0时，返回
    2. 当输入的buffer对应的addr之后还有内存没有释放时，以warn记录错误, 返回
    3. 当输入的buffer不存在时，记录warn，返回
    */
    virtual void ReleaseBuffer(const OckHmmStackBuffer &buffer) = 0;

    static std::shared_ptr<OckHmmStackManager> Create(std::shared_ptr<OckHmmHMObject> hmo);
};
class OckHmmStackBufferGuard : public OckHmmHMOBufferBase {
public:
    virtual ~OckHmmStackBufferGuard() noexcept;
    explicit OckHmmStackBufferGuard(std::shared_ptr<OckHmmStackManager> stackManager,
        const OckHmmStackBuffer &stackBuffer);
    uintptr_t Address(void) const;
    uint64_t Size(void) const;

private:
    OckHmmStackBuffer buffer;
    std::shared_ptr<OckHmmStackManager> stackMgr;
};
}  // namespace hmm
}  // namespace ock
#endif