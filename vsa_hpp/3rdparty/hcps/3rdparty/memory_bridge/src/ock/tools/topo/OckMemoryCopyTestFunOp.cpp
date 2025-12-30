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

#include "ock/tools/topo/OckMemoryCopyTestFunOp.h"

namespace ock {
namespace tools {
namespace topo {
OckMemoryCopyTestParam::OckMemoryCopyTestParam(void *destination, size_t destinationMax, const void *source,
    size_t copyCount, acladapter::OckMemoryCopyKind copyKind)
    : acladapter::OckMemoryCopyParam(destination, destinationMax, source, copyCount, copyKind),
      testTimeSeconds(1),
      receiver(nullptr)
{}
void OckMemoryCopyTestParam::TestTime(uint32_t value)
{
    testTimeSeconds = value;
}
uint32_t OckMemoryCopyTestParam::TestTime(void)
{
    return testTimeSeconds;
}
OckMemoryCopyTestResultReceiver *OckMemoryCopyTestParam::Receiver(void)
{
    return receiver;
}
void OckMemoryCopyTestParam::Receiver(OckMemoryCopyTestResultReceiver *value)
{
    receiver = value;
}
OckMemoryCopyTestFunOp::OckMemoryCopyTestFunOp(void)
{}
acladapter::OckTaskResourceType OckMemoryCopyTestFunOp::ResourceType(void) const
{
    return copyOp.ResourceType();
}
bool OckMemoryCopyTestFunOp::PreConditionMet(void)
{
    return copyOp.PreConditionMet();
}
std::shared_ptr<acladapter::OckDefaultResult> OckMemoryCopyTestFunOp::Run(
    acladapter::OckAsyncTaskContext &context, OckMemoryCopyTestParam &param, acladapter::OckUserWaitInfoBase &waitInfo)
{
    auto untilTime = std::chrono::steady_clock::now() + std::chrono::seconds(param.TestTime());
    while (std::chrono::steady_clock::now() < untilTime && !waitInfo.WaitTimeOut()) {
        auto ret = copyOp.Run(context, param, waitInfo);
        if (ret->ErrorCode() != hmm::HMM_SUCCESS) {
            return ret;
        } else {
            auto receiver = param.Receiver();
            if (receiver != nullptr) {
                receiver->Notify(context.GetDeviceId(), param.SrcCount());
            }
        }
    }
    return std::make_shared<acladapter::OckDefaultResult>(hmm::HMM_SUCCESS);
}

}  // namespace topo
}  // namespace tools
}  // namespace ock