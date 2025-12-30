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


#ifndef OCK_MEMORY_BRIDGE_OCK_ADAPTER_TASK_OCK_MEMORY_COPY_TASK_H
#define OCK_MEMORY_BRIDGE_OCK_ADAPTER_TASK_OCK_MEMORY_COPY_TASK_H
#include <memory>
#include "ock/acladapter/param/OckMemoryCopyParam.h"
#include "ock/acladapter/executor/OckAsyncResultInnerBridge.h"
#include "ock/acladapter/task/OckAsyncTaskExt.h"
namespace ock {
namespace acladapter {

class OckMemoryCopyFunOp : public OckTaskFunOp<OckMemoryCopyParam, OckDefaultResult> {
public:
    virtual ~OckMemoryCopyFunOp() noexcept = default;
    explicit OckMemoryCopyFunOp(void);

    OckTaskResourceType ResourceType(void) const override;
    bool PreConditionMet(void) override;
    std::shared_ptr<OckDefaultResult> Run(
        OckAsyncTaskContext &context, OckMemoryCopyParam &param, OckUserWaitInfoBase &waitInfo) override;
};
using OckMemoryCopyTask = OckAsyncTaskExt<OckMemoryCopyParam, OckDefaultResult, OckMemoryCopyFunOp>;
}  // namespace acladapter
}  // namespace ock
#endif