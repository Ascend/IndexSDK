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


#include "ock/hcps/nop/remove_data_custom_attr_op/OckRemoveDataCustomAttrOpDataBuffer.h"
#include "ock/hcps/nop/remove_data_custom_attr_op/OckRemoveDataCustomAttrOpFactory.h"
#include "ock/hcps/nop/remove_data_custom_attr_op/OckRemoveDataCustomAttrOpRun.h"
namespace ock {
namespace hcps {
namespace nop {
std::shared_ptr<OckHeteroOperatorBase> OckRemoveDataCustomAttrOpRun::CreateOp(
    std::shared_ptr<OckRemoveDataCustomAttrOpHmoBlock> hmoBlock, handler::OckHeteroHandler &handler)
{
    OckRemoveDataCustomAttrOpMeta opSpec;
    opSpec.removeSize = static_cast<int64_t>(hmoBlock->removeSize);
    OckRemoveDataCustomAttrBufferMeta bufferSpec;
    bufferSpec.customAttrLen = static_cast<int64_t>(hmoBlock->customAttrLen);
    bufferSpec.customAttrBlockSize = static_cast<int64_t>(hmoBlock->customAttrBlockSize);
    auto factory = OckRemoveDataCustomAttrOpFactory::Create(opSpec); // 内部做了算子reset
    auto dataBuffer = OckRemoveDataCustomAttrOpDataBuffer::Create(opSpec, bufferSpec);
    // 为算子分配buffer空间(根据hmoBlock自动填充数据)
    auto ret = dataBuffer->AllocBuffersFromHmoBlock(hmoBlock, handler.HmmMgrPtr());
    if (ret != hmm::HMM_SUCCESS) {
        return std::shared_ptr<OckHeteroOperatorBase>();
    }
    return factory->Create(dataBuffer);
}
} // namespace nop
} // namespace hcps
} // namespace ock
