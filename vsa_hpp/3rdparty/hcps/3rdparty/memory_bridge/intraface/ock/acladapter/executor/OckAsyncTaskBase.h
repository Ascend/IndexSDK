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


#ifndef OCK_MEMORY_BRIDGE_OCK_ADAPTER_OCK_ASYNC_TASK_BASE_H
#define OCK_MEMORY_BRIDGE_OCK_ADAPTER_OCK_ASYNC_TASK_BASE_H
#include <cstdint>
#include <ostream>
#include "ock/hmm/mgr/OckHmmHMObject.h"
#include "ock/acladapter/data/OckTaskResourceType.h"
#include "ock/acladapter/executor/OckTaskStatisticsMgr.h"
namespace ock {
namespace acladapter {

class OckAsyncTaskParamBase {
public:
    virtual ~OckAsyncTaskParamBase() noexcept = default;
};
class OckDftAsyncTaskParam : public OckAsyncTaskParamBase {
public:
    virtual ~OckDftAsyncTaskParam() noexcept = default;
    explicit OckDftAsyncTaskParam(void) = default;
};
std::ostream &operator<<(std::ostream &os, const OckAsyncTaskParamBase &data);
std::ostream &operator<<(std::ostream &os, const OckDftAsyncTaskParam &data);
class OckAsyncTaskContext {
public:
    virtual ~OckAsyncTaskContext() noexcept = default;
    virtual hmm::OckHmmDeviceId GetDeviceId(void) const = 0;
    virtual std::shared_ptr<OckTaskStatisticsMgr> StatisticsMgr(void) = 0;
};
std::ostream &operator<<(std::ostream &os, const OckAsyncTaskContext &context);
class OckAsyncTaskBase {
public:
    virtual ~OckAsyncTaskBase() noexcept = default;
    virtual std::string Name(void) const = 0;
    virtual std::string ParamInfo(void) const = 0;
    /*
    @brief 任务的主要资源类型，方便任务编排
    */
    virtual OckTaskResourceType ResourceType(void) const = 0;
    /*
    @brief 前置条件是否满足， 有些任务有自己的执行要求。（例如利用Buffer完成HMO传输，Buffer有空间限制，
    HMO传输任务需要排队执行)
    */
    virtual bool PreConditionMet(void) = 0;
    virtual void Run(OckAsyncTaskContext &context) = 0;
    /*
    @brief 取消任务执行，取消的同时需要向等待方发送通知
    */
    virtual void Cancel(void) = 0;
};

class OckTaskConditionWaitBase {
public:
    virtual ~OckTaskConditionWaitBase() noexcept = default;
    virtual bool Wait(void) = 0;
};
class OckUserWaitInfoBase {
public:
    virtual ~OckUserWaitInfoBase() noexcept = default;
    /*
    @return 用户是否等待超时，如果用户等待超时，相应的任务也可以取消执行了
    */
    virtual bool WaitTimeOut(void) const = 0;
};
std::ostream &operator<<(std::ostream &os, const OckAsyncTaskBase &task);
}  // namespace acladapter
}  // namespace ock
#endif