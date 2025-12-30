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


#ifndef OCK_MEMORY_BRIDGE_OCK_ADAPTER_TASK_OCK_ASYNC_TASK_EXT_H
#define OCK_MEMORY_BRIDGE_OCK_ADAPTER_TASK_OCK_ASYNC_TASK_EXT_H
#include <memory>
#include "ock/acladapter/executor/OckAsyncTaskBase.h"
#include "ock/acladapter/executor/OckAsyncResultInnerBridge.h"
namespace ock {
namespace acladapter {

template <typename _ParamT, typename _ResultT>
class OckTaskFunOp {
public:
    virtual ~OckTaskFunOp() noexcept = default;
    virtual OckTaskResourceType ResourceType(void) const = 0;
    virtual bool PreConditionMet(void) = 0;
    virtual std::shared_ptr<_ResultT> Run(
        OckAsyncTaskContext &context, _ParamT &param, OckUserWaitInfoBase &waitInfo) = 0;
};
template <typename _ParamT, typename _ResultT, typename _FunOpT>
class OckAsyncTaskExt : public OckAsyncTaskBase {
public:
    typedef _ParamT ParamT;
    typedef _ResultT ResultT;
    typedef _FunOpT FunOpT;
    typedef OckAsyncResultInnerBridge<_ResultT> BridgeT;
    virtual ~OckAsyncTaskExt() noexcept = default;
    explicit OckAsyncTaskExt(std::shared_ptr<_ParamT> &parameter,
        std::shared_ptr<OckAsyncResultInnerBridge<_ResultT>> &innerBridge,
        std::shared_ptr<OckTaskConditionWaitBase> &waitBase, std::shared_ptr<_FunOpT> func);

    std::string Name(void) const override;
    std::string ParamInfo(void) const override;
    OckTaskResourceType ResourceType(void) const override;
    bool PreConditionMet(void) override;
    void Run(OckAsyncTaskContext &context) override;
    void Cancel(void) override;
    /*
    @param param 拷贝参数，不能为null
    @param bridge 异步结果桥接器，不能为null
    @param waiter 为Task前置条件等待器，为null时意味者无前置条件
    */
    static std::shared_ptr<OckAsyncTaskBase> Create(std::shared_ptr<_ParamT> param,
        std::shared_ptr<OckAsyncResultInnerBridge<_ResultT>> bridge,
        std::shared_ptr<OckTaskConditionWaitBase> waiter = std::shared_ptr<OckTaskConditionWaitBase>(nullptr))
    {
        return std::make_shared<OckAsyncTaskExt<_ParamT, _ResultT, _FunOpT>>(param, bridge, waiter,
            std::make_shared<_FunOpT>());
    }
    static std::shared_ptr<OckAsyncTaskBase> Create(std::shared_ptr<_ParamT> param, std::shared_ptr<_FunOpT> funOp,
        std::shared_ptr<OckAsyncResultInnerBridge<_ResultT>> bridge,
        std::shared_ptr<OckTaskConditionWaitBase> waiter = std::shared_ptr<OckTaskConditionWaitBase>(nullptr))
    {
        return std::make_shared<OckAsyncTaskExt<_ParamT, _ResultT, _FunOpT>>(param, bridge, waiter, funOp);
    }

private:
    std::shared_ptr<_ParamT> param;
    std::shared_ptr<OckAsyncResultInnerBridge<_ResultT>> bridge;
    std::shared_ptr<OckTaskConditionWaitBase> waiter;
    std::shared_ptr<_FunOpT> funOp;
};
using OckDftTaskFunOp = OckTaskFunOp<OckDftAsyncTaskParam, OckDefaultResult>;
template <typename _ParamT, typename _ResultT, typename _FunOpT>
OckAsyncTaskExt<_ParamT, _ResultT, _FunOpT>::OckAsyncTaskExt(std::shared_ptr<_ParamT> &parameter,
    std::shared_ptr<OckAsyncResultInnerBridge<_ResultT>> &innerBridge,
    std::shared_ptr<OckTaskConditionWaitBase> &waitBase, std::shared_ptr<_FunOpT> func)
    : param(parameter), bridge(innerBridge), waiter(waitBase), funOp(func)
{}
template <typename _ParamT, typename _ResultT, typename _FunOpT>
std::string OckAsyncTaskExt<_ParamT, _ResultT, _FunOpT>::Name(void) const
{
    return typeid(_FunOpT).name();
}
template <typename _ParamT, typename _ResultT, typename _FunOpT>
std::string OckAsyncTaskExt<_ParamT, _ResultT, _FunOpT>::ParamInfo(void) const
{
    std::ostringstream osStr;
    osStr << *param;
    return osStr.str();
}
template <typename _ParamT, typename _ResultT, typename _FunOpT>
bool OckAsyncTaskExt<_ParamT, _ResultT, _FunOpT>::PreConditionMet(void)
{
    return funOp->PreConditionMet();
}
template <typename _ParamT, typename _ResultT, typename _FunOpT>
OckTaskResourceType OckAsyncTaskExt<_ParamT, _ResultT, _FunOpT>::ResourceType(void) const
{
    return funOp->ResourceType();
}
template <typename _ParamT, typename _ResultT, typename _FunOpT>
void OckAsyncTaskExt<_ParamT, _ResultT, _FunOpT>::Run(OckAsyncTaskContext &context)
{
    if (waiter.get() != nullptr) {
        waiter->Wait();
    }
    if (bridge->WaitTimeOut()) {
        return;
    }
    bridge->SetResult(funOp->Run(context, *param, *bridge));
}

template <typename _ParamT, typename _ResultT, typename _FunOpT>
void OckAsyncTaskExt<_ParamT, _ResultT, _FunOpT>::Cancel(void)
{
    bridge->Cancel();
}
}  // namespace acladapter
}  // namespace ock
#endif