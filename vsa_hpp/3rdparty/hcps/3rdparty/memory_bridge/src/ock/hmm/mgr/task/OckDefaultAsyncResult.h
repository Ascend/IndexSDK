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


#ifndef OCK_MEMORY_BRIDGE_DEFAULT_ASYNC_RESULT_H
#define OCK_MEMORY_BRIDGE_DEFAULT_ASYNC_RESULT_H
#include "ock/hmm/mgr/OckHmmAsyncResult.h"
namespace ock {
namespace hmm {

template <typename _ResultT, typename _WaiterT, typename _OutCreateT>
class OckDefaultAsyncResult : public OckHmmAsyncResult<_ResultT> {
public:
    virtual ~OckDefaultAsyncResult() noexcept = default;
    explicit OckDefaultAsyncResult(std::shared_ptr<_WaiterT> waiterT) : waiter(waiterT)
    {}
    std::shared_ptr<_ResultT> WaitResult(uint32_t timeout = 0) const override
    {
        return _OutCreateT::Create(waiter->WaitResult(timeout));
    }
    void Cancel(void) override
    {
        waiter->Cancel();
    }

private:
    std::shared_ptr<_WaiterT> waiter;
};
template <typename _ResultT>
class OckDefaultAsyncResult<_ResultT, _ResultT, _ResultT> : public OckHmmAsyncResult<_ResultT> {
public:
    virtual ~OckDefaultAsyncResult() noexcept = default;
    explicit OckDefaultAsyncResult(std::shared_ptr<_ResultT> res) : result(res)
    {}
    std::shared_ptr<_ResultT> WaitResult(uint32_t timeout = 0) const override
    {
        return result;
    }
    void Cancel(void) override
    {
        return;
    }

private:
    std::shared_ptr<_ResultT> result;
};
}  // namespace hmm
}  // namespace ock
#endif