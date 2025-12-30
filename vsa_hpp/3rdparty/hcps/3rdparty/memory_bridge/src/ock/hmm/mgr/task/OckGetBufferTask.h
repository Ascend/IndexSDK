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


#ifndef OCK_MEMORY_BRIDGE_OCK_HMM_OCK_GET_BUFFER_TASK_H
#define OCK_MEMORY_BRIDGE_OCK_HMM_OCK_GET_BUFFER_TASK_H
#include <memory>
#include "ock/hmm/mgr/task/OckGetBufferParam.h"
#include "ock/acladapter/executor/OckAsyncResultInnerBridge.h"
#include "ock/acladapter/task/OckAsyncTaskExt.h"
#include "ock/hmm/mgr/OckHmmHMOBufferExt.h"
namespace ock {
namespace hmm {

class OckGetBufferFunOp : public acladapter::OckTaskFunOp<OckGetBufferParam, OckHmmHMOBufferExt> {
public:
    virtual ~OckGetBufferFunOp() noexcept = default;
    explicit OckGetBufferFunOp(void) = default;

    acladapter::OckTaskResourceType ResourceType(void) const override;
    bool PreConditionMet(void) override;
    std::shared_ptr<OckHmmHMOBufferExt> Run(acladapter::OckAsyncTaskContext &context, OckGetBufferParam &param,
        acladapter::OckUserWaitInfoBase &waitInfo) override;
};
using OckGetBufferTask = acladapter::OckAsyncTaskExt<OckGetBufferParam, OckHmmHMOBufferExt, OckGetBufferFunOp>;

}  // namespace hmm
}  // namespace ock
#endif