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

#include "ock/log/OckHcpsLogger.h"
#include "ock/hcps/hcop/topk_dist_comp_op/OckTopkDistCompOpDataBuffer.h"
#include "ock/hcps/hcop/topk_dist_comp_op/OckTopkDistCompOpFactory.h"
#include "ock/hcps/hcop/topk_dist_comp_op/OckTopkDistCompOpRun.h"
#include "ock/vsa/OckVsaErrorCode.h"

namespace ock {
namespace hcps {
namespace hcop {
struct OckTopkDistCompMeta {
    OckTopkDistCompOpMeta opSpec;
    OckTopkDistCompBufferMeta bufferSpec;
};
OckHcpsErrorCode CreateAndRunOpSync(const OckTopkDistCompMeta &meta,
    std::shared_ptr<OckTopkDistCompOpFactory> compOpFactory, std::shared_ptr<OckTopkDistCompOpHmoGroup> hmoGroup,
    std::shared_ptr<OckHeteroStreamBase> streamBase, std::shared_ptr<handler::OckHeteroHandler> handler)
{
    if (compOpFactory == nullptr || hmoGroup == nullptr || streamBase == nullptr || handler == nullptr) {
        OCK_HCPS_LOG_ERROR("Input param cannot be nullptr");
        return vsa::VSA_ERROR_INVALID_INPUT_PARAM;
    }
    // alloc buffer
    auto dataBuffer = hcop::OckTopkDistCompOpDataBuffer::Create(meta.opSpec, meta.bufferSpec);
    auto errorCode =
        compOpFactory->AllocTmpSpace(meta.opSpec, dataBuffer->GetTopkBuffer(), dataBuffer->GetDistBuffers(), handler);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    errorCode = dataBuffer->AllocBuffersFromHmoGroup(hmoGroup, handler->HmmMgrPtr());
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    dataBuffer->SetHyperParameters(hmoGroup);

    // 创建算子
    auto topkDistCompOp = compOpFactory->Create(meta.opSpec, meta.bufferSpec, dataBuffer->GetTopkBuffer(),
        dataBuffer->GetDistBuffers(), handler);

    // 运行算子
    streamBase->AddOp(topkDistCompOp);

    // 同步
    return streamBase->WaitExecComplete();
}

OckHcpsErrorCode OckTopkDistCompOpRun::RunMultiGroupsSync(
    const std::vector<std::shared_ptr<OckTopkDistCompOpHmoGroup>> &hmoGroups,
    std::shared_ptr<handler::OckHeteroHandler> handler)
{
    if (handler == nullptr) {
        OCK_HCPS_LOG_ERROR("Input param cannot be nullptr");
        return vsa::VSA_ERROR_INVALID_INPUT_PARAM;
    }
    OckHcpsErrorCode errCode = hmm::HMM_SUCCESS;
    auto streamBase = hcps::handler::helper::MakeStream(*handler, errCode, hcps::OckDevStreamType::AI_CPU);
    OCK_CHECK_RETURN_ERRORCODE(errCode);
    OckTopkDistCompOpMeta opSpec;
    OckTopkDistCompBufferMeta bufferSpec;
    UpdateMetaFromHmoGroup(opSpec, bufferSpec, hmoGroups[0]);
    auto compOpFactory = OckTopkDistCompOpFactory::Create(opSpec);
    for (size_t i = 0; i < hmoGroups.size(); ++i) {
        bufferSpec.pageId = hmoGroups[i]->groupId;
        OckTopkDistCompMeta meta = { opSpec, bufferSpec };
        errCode = CreateAndRunOpSync(meta, compOpFactory, hmoGroups[i], streamBase, handler);
        OCK_HCPS_LOG_DEBUG("finish running comp op for group = " << i);
        OCK_CHECK_RETURN_ERRORCODE(errCode);
    }
    return errCode;
}

OckHcpsErrorCode OckTopkDistCompOpRun::RunOneGroupSync(std::shared_ptr<OckTopkDistCompOpHmoGroup> hmoGroup,
    std::shared_ptr<OckHeteroStreamBase> streamBase, std::shared_ptr<handler::OckHeteroHandler> handler)
{
    if (hmoGroup == nullptr || streamBase == nullptr || handler == nullptr) {
        OCK_HCPS_LOG_ERROR("Input param cannot be nullptr");
        return vsa::VSA_ERROR_INVALID_INPUT_PARAM;
    }
    OckTopkDistCompOpMeta opSpec;
    OckTopkDistCompBufferMeta bufferSpec;
    UpdateMetaFromHmoGroup(opSpec, bufferSpec, hmoGroup);
    auto compOpFactory = OckTopkDistCompOpFactory::Create(opSpec);
    OckTopkDistCompMeta meta = { opSpec, bufferSpec };
    return CreateAndRunOpSync(meta, compOpFactory, hmoGroup, streamBase, handler);
}
} // namespace hcop
} // namespace hcps
} // namespace ock