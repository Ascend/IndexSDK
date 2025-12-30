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

#include "ock/hcps/nop/remove_data_shaped_op/OckRemoveDataShapedOpDataBuffer.h"
#include "ock/hcps/nop/remove_data_shaped_op/OckRemoveDataShapedOpFactory.h"
#include "ock/hcps/nop/remove_data_shaped_op/OckRemoveDataShapedOpRun.h"
namespace ock {
namespace hcps {
namespace nop {
std::shared_ptr<OckHeteroOperatorBase> OckRemoveDataShapedOpRun::GenRemoveShapedOp(
    std::shared_ptr<OckRemoveDataShapedOpHmoBlock> hmoBlock, handler::OckHeteroHandler &handler)
{
    OckRemoveDataShapedOpMeta opSpec;
    opSpec.removeCount = hmoBlock->removeCount;
    auto factory = OckRemoveDataShapedOpFactory::Create(opSpec);
    auto dataBuffer = OckRemoveDataShapedOpDataBuffer::Create(opSpec);
    auto ret = dataBuffer->AllocBuffersFromHmoBlock(hmoBlock, handler.HmmMgrPtr());
    if (ret != hmm::HMM_SUCCESS) {
        OCK_HCPS_LOG_ERROR("alloc buffers from hmoBlock failed, the errorCode is " << ret);
        return std::shared_ptr<OckHeteroOperatorBase>();
    }
    std::vector<int64_t> attrs = {
        Type::INT8, CUBE_ALIGN, CUBE_ALIGN_INT8, utils::SafeDivUp(hmoBlock->dims, CUBE_ALIGN_INT8) };
    FillBuffer<int64_t>(dataBuffer->InputAttrs(), attrs);
    auto removeOp = factory->Create(dataBuffer);
    return removeOp;
}
}
}
}