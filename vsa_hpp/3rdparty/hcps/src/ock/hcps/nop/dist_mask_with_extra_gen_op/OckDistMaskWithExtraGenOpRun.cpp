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


#include "ock/hcps/nop/dist_mask_with_extra_gen_op/OckDistMaskWithExtraGenOpDataBuffer.h"
#include "ock/hcps/nop/dist_mask_with_extra_gen_op/OckDistMaskWithExtraGenOpFactory.h"
#include "ock/hcps/nop/dist_mask_with_extra_gen_op/OckDistMaskWithExtraGenOpRun.h"
namespace ock {
namespace hcps {
namespace nop {
static void AddMaskWithExtraOp(std::shared_ptr<OckDistMaskWithExtraGenOpHmoGroup> maskGroup,
    uint64_t offsetInSingleMask, uint64_t groupMaskLen, std::shared_ptr<OckDistMaskWithExtraGenOpFactory> factory,
    OckDistMaskWithExtraGenOpMeta opSpec, std::shared_ptr<OckHeteroStreamBase> streamBase)
{
    auto dataBuffer = OckDistMaskWithExtraGenOpDataBuffer::Create(opSpec);
    auto ret = dataBuffer->AllocBuffersFromHmoGroup(maskGroup, offsetInSingleMask, groupMaskLen);
    if (ret != hmm::HMM_SUCCESS) {
        OCK_HCPS_LOG_ERROR("alloc buffers from hmoGroup failed, the errorCode is " << ret);
        return;
    }
    auto extraMaskOp = factory->Create(dataBuffer);
    streamBase->AddOp(extraMaskOp);
}

static void AddExtraMaskOpsInOneBatch(std::shared_ptr<OckDistMaskWithExtraGenOpHmoGroups> hmoGroups,
    std::shared_ptr<OckDistMaskWithExtraGenOpFactory> factory, std::shared_ptr<OckHeteroStreamBase> streamBase,
    OckDistMaskWithExtraGenOpMeta opSpec, uint64_t offsetInBatchedMask, uint32_t batchId)
{
    uint64_t offsetInSingleMask = offsetInBatchedMask;
    uint64_t groupMaskLen = utils::SafeDivUp(opSpec.featureAttrBlockSize * opSpec.blockCount, OPS_DATA_TYPE_ALIGN);
    for (uint32_t i = 0; i < hmoGroups->attrTimes.size(); ++i) {
        for (uint32_t j = 0; j < hmoGroups->attrTimes[i]->size(); ++j) {
            auto maskGroup = std::make_shared<OckDistMaskWithExtraGenOpHmoGroup>();
            maskGroup->queryTimes = hmoGroups->queryTimes[batchId];
            maskGroup->queryTokenIds = hmoGroups->queryTokenIds[batchId];
            maskGroup->attrTimes = hmoGroups->attrTimes[i]->at(j);
            maskGroup->attrTokenQuotients = hmoGroups->attrTokenQuotients[i]->at(j);
            maskGroup->attrTokenRemainders = hmoGroups->attrTokenRemainders[i]->at(j);
            maskGroup->extraMask = hmoGroups->extraMask;
            maskGroup->mask = hmoGroups->mask;
            AddMaskWithExtraOp(maskGroup, offsetInSingleMask, groupMaskLen, factory, opSpec, streamBase);
            offsetInSingleMask += groupMaskLen;
        }
    }
}

OckHcpsErrorCode OckDistMaskWithExtraGenOpRun::AddMaskWithExtraOpsMultiBatches(
    std::shared_ptr<OckDistMaskWithExtraGenOpHmoGroups> hmoGroups, std::shared_ptr<OckHeteroStreamBase> streamBase)
{
    OckDistMaskWithExtraGenOpMeta opSpec;
    opSpec.tokenNum = hmoGroups->tokenNum;
    opSpec.featureAttrBlockSize = hmoGroups->featureAttrBlockSize;
    opSpec.blockCount = hmoGroups->blockCount;
    auto factory = OckDistMaskWithExtraGenOpFactory::Create(opSpec);
    uint64_t offsetInBatchedMask = 0;
    uint32_t queryCount = static_cast<uint32_t>(hmoGroups->queryTimes.size());
    for (uint32_t i = 0; i < queryCount; ++i) {
        AddExtraMaskOpsInOneBatch(hmoGroups, factory, streamBase, opSpec, offsetInBatchedMask, i);
        OCK_CHECK_RETURN_ERRORCODE(streamBase->WaitExecComplete());
        offsetInBatchedMask += hmoGroups->maskLen;
    }
    return hmm::HMM_SUCCESS;
}

void OckDistMaskWithExtraGenOpRun::AddMaskWithExtraOpsSingleBatch(
    std::shared_ptr<OckDistMaskWithExtraGenOpHmoGroups> hmoGroups, std::shared_ptr<OckHeteroStreamBase> streamBase)
{
    OckDistMaskWithExtraGenOpMeta opSpec;
    opSpec.tokenNum = hmoGroups->tokenNum;
    opSpec.featureAttrBlockSize = hmoGroups->featureAttrBlockSize;
    opSpec.blockCount = hmoGroups->blockCount;
    static auto factory = OckDistMaskWithExtraGenOpFactory::Create(opSpec);
    AddExtraMaskOpsInOneBatch(hmoGroups, factory, streamBase, opSpec, 0, 0);
}
}
}
}
