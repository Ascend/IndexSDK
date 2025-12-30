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


#ifndef OCK_MEMORY_BRIDGE_TOOL_TOPO_DECTECT_OCK_MEMORY_COPY_TEST_TASK_H
#define OCK_MEMORY_BRIDGE_TOOL_TOPO_DECTECT_OCK_MEMORY_COPY_TEST_TASK_H
#include <memory>
#include "ock/acladapter/param/OckMemoryCopyParam.h"
#include "ock/acladapter/executor/OckAsyncResultInnerBridge.h"
#include "ock/acladapter/task/OckAsyncTaskExt.h"
#include "ock/acladapter/task/OckMemoryCopyTask.h"

namespace ock {
namespace tools {
namespace topo {

class OckMemoryCopyTestResultReceiver {
public:
    virtual ~OckMemoryCopyTestResultReceiver() noexcept = default;

    virtual void Notify(hmm::OckHmmDeviceId deviceId, uint64_t movedBytes) = 0;
};
class OckMemoryCopyTestParam : public acladapter::OckMemoryCopyParam {
public:
    virtual ~OckMemoryCopyTestParam() noexcept = default;
    explicit OckMemoryCopyTestParam(void *destination, size_t destinationMax, const void *source, size_t copyCount,
        acladapter::OckMemoryCopyKind copyKind);

    OckMemoryCopyTestParam(const OckMemoryCopyTestParam &) = delete;
    OckMemoryCopyTestParam &operator=(const OckMemoryCopyTestParam &) = delete;

    void TestTime(uint32_t value);
    uint32_t TestTime(void);
    OckMemoryCopyTestResultReceiver *Receiver(void);
    void Receiver(OckMemoryCopyTestResultReceiver *value);
private:
    uint32_t testTimeSeconds;
    OckMemoryCopyTestResultReceiver *receiver;
};
class OckMemoryCopyTestFunOp : public acladapter::OckTaskFunOp<OckMemoryCopyTestParam, acladapter::OckDefaultResult> {
public:
    virtual ~OckMemoryCopyTestFunOp() noexcept = default;
    explicit OckMemoryCopyTestFunOp(void);

    acladapter::OckTaskResourceType ResourceType(void) const override;
    bool PreConditionMet(void) override;
    std::shared_ptr<acladapter::OckDefaultResult> Run(acladapter::OckAsyncTaskContext &context,
        OckMemoryCopyTestParam &param, acladapter::OckUserWaitInfoBase &waitInfo) override;

private:
    acladapter::OckMemoryCopyFunOp copyOp{};
};
using OckMemoryCopyTestTask =
    acladapter::OckAsyncTaskExt<OckMemoryCopyTestParam, acladapter::OckDefaultResult, OckMemoryCopyTestFunOp>;
}  // namespace topo
}  // namespace tools
}  // namespace ock
#endif