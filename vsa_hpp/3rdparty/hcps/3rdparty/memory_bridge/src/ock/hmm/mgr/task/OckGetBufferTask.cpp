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

#include "ock/hmm/mgr/task/OckGetBufferTask.h"
#include "ock/acladapter/param/OckMemoryCopyParam.h"
#include "ock/acladapter/task/OckMemoryCopyTask.h"
#include "ock/hmm/mgr/OckHmmMemoryGuardExt.h"
namespace ock {
namespace hmm {

acladapter::OckTaskResourceType OckGetBufferFunOp::ResourceType(void) const
{
    return acladapter::OckTaskResourceType::MEMORY_TRANSFER;
}
bool OckGetBufferFunOp::PreConditionMet(void)
{
    return true;
}
std::shared_ptr<OckHmmHMOBufferExt> OckGetBufferFunOp::Run(
    acladapter::OckAsyncTaskContext &context, OckGetBufferParam &param, acladapter::OckUserWaitInfoBase &waitInfo)
{
    auto memoryGuard = std::make_unique<OckHmmMemoryGuardExt>(param.MemAlloc(),
        param.MemAlloc()->Alloc(param.Length(), waitInfo), param.Length());
    if (memoryGuard->Addr() == 0UL || waitInfo.WaitTimeOut()) {
        return std::shared_ptr<OckHmmHMOBufferExt>();
    }
    auto kind = acladapter::CalcMemoryCopyKind(param.SrcHmo().Location(), param.MemAlloc()->Location());
    uint8_t *dstAddr = (uint8_t *)memoryGuard->Addr();
    uint8_t *srcAddr = (uint8_t *)param.SrcHmo().Addr() + param.Offset();
    auto copyParam = std::make_shared<acladapter::OckMemoryCopyParam>((void *)dstAddr,
        param.SrcHmo().GetByteSize() - param.Offset(),
        (const void *)srcAddr,
        memoryGuard->ByteSize(),
        kind);

    acladapter::OckMemoryCopyFunOp copyOp;
    auto dftResult = copyOp.Run(context, *copyParam, waitInfo);
    return OckHmmHMOBufferExt::Create(dftResult->ErrorCode(), std::move(memoryGuard), param.Offset(), param.SrcHmo());
}
}  // namespace hmm
}  // namespace ock