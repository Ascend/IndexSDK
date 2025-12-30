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


#include "ock/hcps/nop/trans_data_shaped_op/OckTransDataShapedOpDataBuffer.h"
#include "ock/hcps/nop/trans_data_shaped_op/OckTransDataShapedOpFactory.h"
#include "ock/hcps/nop/trans_data_shaped_op/OckTransDataShapedOpRun.h"
namespace ock {
namespace hcps {
namespace nop {
    void OckTransDataShapedOpRun::AddTransShapedOp(std::shared_ptr<OckTransDataShapedOpHmoBlock> hmoBlock,
                                                   handler::OckHeteroHandler &handler,
                                                   std::shared_ptr<OckHeteroStreamBase> streamBase)
    {
        OckTransDataShapedOpMeta opSpec;
        opSpec.dims = hmoBlock->dims;
        opSpec.codeBlockSize = hmoBlock->codeBlockSize;
        opSpec.addNum = hmoBlock->addNum;
        OckTransDataShapedBufferMeta bufferSpec;
        bufferSpec.ntotal = hmoBlock->addNum;
        auto factory = OckTransDataShapedOpFactory::Create(opSpec);
        auto dataBuffer = OckTransDataShapedOpDataBuffer::Create(opSpec, bufferSpec);
        auto ret = dataBuffer->AllocBuffersFromHmoBlock(hmoBlock, handler.HmmMgrPtr());
        if (ret != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("alloc buffers from hmoBlock failed, the errorCode is " << ret);
            return;
        }
        std::vector<int64_t> attr = { hmoBlock->offsetInDstHmo };
        FillBuffer<int64_t>(dataBuffer->InputAttr(), attr);
        auto transOp = factory->Create(dataBuffer);
        streamBase->AddOp(transOp);
    }
}
}
}
