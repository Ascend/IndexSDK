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


#ifndef HCPS_OCK_TRANS_DATA_CUSTOM_ATTR_OP_DATA_BUFFER_H
#define HCPS_OCK_TRANS_DATA_CUSTOM_ATTR_OP_DATA_BUFFER_H
#include <memory>
#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"
#include "ock/hcps/error/OckHcpsErrorCode.h"
#include "ock/hcps/nop/OckOpDataBuffer.h"
#include "ock/hcps/nop/trans_data_custom_attr_op/OckTransDataCustomAttrMeta.h"

namespace ock {
namespace hcps {
namespace nop {
class OckTransDataCustomAttrOpDataBuffer : public OckOpDataBuffer {
public:
    virtual ~OckTransDataCustomAttrOpDataBuffer() noexcept = default;
    virtual OckHcpsErrorCode AllocBuffersFromHmoBlock(std::shared_ptr<OckTransDataCustomAttrOpHmoBlock> hmoBlock,
        std::shared_ptr<hmm::OckHmmHeteroMemoryMgrBase> devMgr) = 0;
    virtual std::shared_ptr<OckDataBuffer> &InputSrc() = 0;
    virtual std::shared_ptr<OckDataBuffer> &InputAttr() = 0;
    virtual std::shared_ptr<OckDataBuffer> &OutputDst() = 0;
    static std::shared_ptr<OckTransDataCustomAttrOpDataBuffer> Create(const OckTransDataCustomAttrOpMeta &opSpec,
        const OckTransDataCustomAttrBufferMeta &bufferSpec);
};
} // namespace nop
} // namespace hcps
} // namespace ock
#endif // HCPS_OCK_TRANS_DATA_CUSTOM_ATTR_OP_DATA_BUFFER_H
