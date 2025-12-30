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


#include "ock/hcps/nop/trans_data_custom_attr_op/OckTransDataCustomAttrOpDataBuffer.h"
#include "ock/hcps/nop/trans_data_custom_attr_op/OckTransDataCustomAttrOpFactory.h"
#include "ock/hcps/nop/trans_data_custom_attr_op/OckTransDataCustomAttrOpRun.h"
namespace ock {
namespace hcps {
namespace nop {
void OckTransDataCustomAttrOpRun::AddTransCustomAttrOp(std::shared_ptr<OckTransDataCustomAttrOpHmoBlock> hmoBlock,
    handler::OckHeteroHandler &handler, std::shared_ptr<OckHeteroStreamBase> streamBase)
{
    OckTransDataCustomAttrOpMeta opSpec;
    opSpec.customAttrLen = static_cast<int64_t>(hmoBlock->customAttrLen);
    opSpec.customAttrBlockSize = static_cast<int64_t>(hmoBlock->customAttrBlockSize);
    opSpec.copyCount = static_cast<int64_t>(hmoBlock->copyCount);
    OckTransDataCustomAttrBufferMeta bufferSpec;

    auto factory = OckTransDataCustomAttrOpFactory::Create(opSpec);
    auto dataBuffer = OckTransDataCustomAttrOpDataBuffer::Create(opSpec, bufferSpec);
    auto ret = dataBuffer->AllocBuffersFromHmoBlock(hmoBlock, handler.HmmMgrPtr());
    if (ret != hmm::HMM_SUCCESS) {
        OCK_HCPS_LOG_ERROR("alloc buffers from hmoBlock failed, the errorCode is " << ret);
        return;
    }
    std::vector<int64_t> attr = { static_cast<int64_t>(hmoBlock->offsetInBlock) };
    FillBuffer<int64_t>(dataBuffer->InputAttr(), attr);
    auto transOp = factory->Create(dataBuffer);
    streamBase->AddOp(transOp);
}

// copyCount 其实就三种情况：上一步不满block的填充、满blocks、最后不足一个block的尾巴
} // namespace nop
} // namespace hcps
} // namespace ock
