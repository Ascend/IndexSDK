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


#ifndef OCK_MEMORY_BRIDGE_OCK_ADAPTER_DFT_FUNCTION_OPERATOR_TASK_H
#define OCK_MEMORY_BRIDGE_OCK_ADAPTER_DFT_FUNCTION_OPERATOR_TASK_H
#include <memory>
#include <functional>
#include "ock/acladapter/executor/OckAsyncResultInnerBridge.h"
#include "ock/acladapter/task/OckAsyncTaskExt.h"
namespace ock {
namespace acladapter {

class OckDftFunOp : public OckDftTaskFunOp {
public:
    virtual ~OckDftFunOp() noexcept = default;
    explicit OckDftFunOp(std::function<hmm::OckHmmErrorCode()> funcOp,
        OckTaskResourceType taskResourceType = OckTaskResourceType::DEVICE_STREAM);

    OckTaskResourceType ResourceType(void) const override;
    bool PreConditionMet(void) override;
    std::shared_ptr<OckDefaultResult> Run(
        OckAsyncTaskContext &context, OckDftAsyncTaskParam &param, OckUserWaitInfoBase &waitInfo) override;

private:
    std::function<hmm::OckHmmErrorCode()> funOp;
    OckTaskResourceType resourceType;
};
using OckDftFunOpTask = OckAsyncTaskExt<OckDftAsyncTaskParam, OckDefaultResult, OckDftFunOp>;
}  // namespace acladapter
}  // namespace ock
#endif