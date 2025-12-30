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

#ifndef HCPS_PIER_OCKREMOVEDATASHAPEDOPDATABUFFER_H
#define HCPS_PIER_OCKREMOVEDATASHAPEDOPDATABUFFER_H
#include <memory>
#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"
#include "ock/hcps/error/OckHcpsErrorCode.h"
#include "ock/hcps/nop/remove_data_shaped_op/OckRemoveDataShapedMeta.h"
#include "ock/hcps/nop/OckOpDataBuffer.h"

namespace ock {
namespace hcps {
namespace nop {
class OckRemoveDataShapedOpDataBuffer : public OckOpDataBuffer {
public:
    virtual ~OckRemoveDataShapedOpDataBuffer() noexcept = default;
    virtual OckHcpsErrorCode AllocBuffersFromHmoBlock(std::shared_ptr<OckRemoveDataShapedOpHmoBlock> hmoBlock,
                                                      std::shared_ptr<hmm::OckHmmHeteroMemoryMgrBase> devMgr) = 0;
    virtual std::shared_ptr<OckDataBuffer> &InputSrcPos() = 0;
    virtual std::shared_ptr<OckDataBuffer> &InputAttrs() = 0;
    virtual std::shared_ptr<OckDataBuffer> &OutputDstPos() = 0;
    static std::shared_ptr<OckRemoveDataShapedOpDataBuffer> Create(const OckRemoveDataShapedOpMeta &opSpec);
};
} // namespace nop
} // namespace hcps
} // namespace ock
#endif // HCPS_PIER_OCKREMOVEDATASHAPEDOPDATABUFFER_H
