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


#ifndef OCK_MEMORY_BRIDGE_ASYNC_RESULT_H
#define OCK_MEMORY_BRIDGE_ASYNC_RESULT_H
#include <cstdint>
#include <memory>
#include "ock/hmm/mgr/OckHmmErrorCode.h"
namespace ock {
namespace hmm {
class OckHmmAsyncResultBase {
public:
    virtual ~OckHmmAsyncResultBase() noexcept = default;
};

template <typename _ResultT>
class OckHmmAsyncResult : public OckHmmAsyncResultBase {
public:
    virtual ~OckHmmAsyncResult() noexcept = default;
    
    /*
    @timeout 为0时代表无限等待(3600秒)。 timeout的单位是ms
    @return 当没有结果(比如：等待超时时)，可能返回null
    */
    virtual std::shared_ptr<_ResultT> WaitResult(uint32_t timeout = 0) const = 0;

    virtual void Cancel(void) = 0;
};

class OckHmmResultBase {
public:
    virtual ~OckHmmResultBase() noexcept = default;
    virtual OckHmmErrorCode ErrorCode(void) const = 0;
};
}  // namespace hmm
}  // namespace ock
#endif