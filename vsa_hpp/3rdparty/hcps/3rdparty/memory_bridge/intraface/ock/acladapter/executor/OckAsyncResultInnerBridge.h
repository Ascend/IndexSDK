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


#ifndef OCK_MEMORY_BRIDGE_OCK_ADAPTER_TASK_ASYNC_RESULT_INNER_BRIDGE_H
#define OCK_MEMORY_BRIDGE_OCK_ADAPTER_TASK_ASYNC_RESULT_INNER_BRIDGE_H
#include <mutex>
#include <memory>
#include <chrono>
#include <atomic>
#include <condition_variable>
#include "ock/log/OckLogger.h"
#include "ock/hmm/mgr/OckHmmAsyncResult.h"
#include "ock/conf/OckSysConf.h"
#include "ock/acladapter/executor/OckAsyncTaskBase.h"
namespace ock {
namespace acladapter {

template <typename _ResultT>
class OckAsyncResultInnerBridge : public OckUserWaitInfoBase, public hmm::OckHmmAsyncResult<_ResultT> {
public:
    virtual ~OckAsyncResultInnerBridge() noexcept = default;
    explicit OckAsyncResultInnerBridge(void);
    void SetResult(std::shared_ptr<_ResultT> result);
    bool IsRunComplete(void) const;
    /*
    @param timeout 当timeout=0时代表无限等待(3600秒)
    */
    std::shared_ptr<_ResultT> WaitResult(uint32_t timeout = 0) const override;
    bool WaitTimeOut(void) const override;
    void Cancel(void) override;

private:
    mutable std::atomic<bool> waitTimeOut;
    std::shared_ptr<_ResultT> result{ nullptr };
    std::atomic<bool> runComplete;
    mutable std::mutex mutex{};
    mutable std::condition_variable condVar{};
};

class OckDefaultResult : public hmm::OckHmmResultBase {
public:
    virtual ~OckDefaultResult() noexcept = default;
    explicit OckDefaultResult(hmm::OckHmmErrorCode errorCode);
    hmm::OckHmmErrorCode ErrorCode(void) const override;

private:
    hmm::OckHmmErrorCode errorCode;
};
template <typename _ResultT>
OckAsyncResultInnerBridge<_ResultT>::OckAsyncResultInnerBridge(void) : waitTimeOut(false), runComplete(false)
{}
template <typename _ResultT>
bool OckAsyncResultInnerBridge<_ResultT>::WaitTimeOut(void) const
{
    return waitTimeOut.load();
}
template <typename _ResultT>
bool OckAsyncResultInnerBridge<_ResultT>::IsRunComplete(void) const
{
    return runComplete.load();
}
template <typename _ResultT>
void OckAsyncResultInnerBridge<_ResultT>::SetResult(std::shared_ptr<_ResultT> data)
{
    std::unique_lock<std::mutex> lock(mutex);
    result = data;
    condVar.notify_all();
    runComplete.store(true);
}
template <typename _ResultT>
void OckAsyncResultInnerBridge<_ResultT>::Cancel(void)
{
    waitTimeOut.store(true);
    std::unique_lock<std::mutex> lock(mutex);
    condVar.notify_all();
}
template <typename _ResultT>
std::shared_ptr<_ResultT> OckAsyncResultInnerBridge<_ResultT>::WaitResult(uint32_t timeout) const
{
    if (timeout == 0) {
        timeout = static_cast<uint32_t>(conf::OckSysConf::HmmConf().maxWaitTimeMilliSecond);
    } else if (timeout > conf::OckSysConf::HmmConf().maxWaitTimeMilliSecond) {
        OCK_HMM_LOG_WARN("timeout(" << timeout << ") is out of range [1, " <<
                         conf::OckSysConf::HmmConf().maxWaitTimeMilliSecond <<
                         "]. The wait time will be set as the upper bound of this range.");
        timeout = static_cast<uint32_t>(conf::OckSysConf::HmmConf().maxWaitTimeMilliSecond);
    }
    std::unique_lock<std::mutex> lock(mutex);
    // 先判断是否已经产生结果，避免Wait时已经notify_all从而产生无限等待
    if (result.get() != nullptr) {
        return result;
    }
    auto status = condVar.wait_for(lock, std::chrono::milliseconds(timeout));
    if (status == std::cv_status::timeout || waitTimeOut.load()) {
        OCK_HMM_LOG_ERROR("WaitResult time out");
        waitTimeOut.store(true);
        return std::shared_ptr<_ResultT>();
    }
    return result;
}

}  // namespace acladapter
}  // namespace ock
#endif