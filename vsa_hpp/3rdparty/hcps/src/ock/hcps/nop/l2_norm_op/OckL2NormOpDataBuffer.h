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


#ifndef HCPS_OCKL2NORMOPDATABUFFER_H
#define HCPS_OCKL2NORMOPDATABUFFER_H
#include <memory>
#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"
#include "ock/acladapter/utils/OckAscendFp16.h"
#include "ock/hcps/error/OckHcpsErrorCode.h"
#include "ock/hcps/nop/l2_norm_op/OckL2NormMeta.h"
#include "ock/hcps/nop/OckOpDataBuffer.h"
namespace ock {
namespace hcps {
namespace nop {
class OckL2NormOpDataBuffer : public OckOpDataBuffer {
public:
    virtual ~OckL2NormOpDataBuffer() noexcept = default;
    virtual OckHcpsErrorCode AllocBuffersFromHmoBlock(std::shared_ptr<OckL2NormOpHmoBlock> hmoBlock,
        std::shared_ptr<hmm::OckHmmHeteroMemoryMgrBase> devMgr, int64_t offset) = 0;
    virtual std::shared_ptr<OckDataBuffer> &InputVectors() = 0;
    virtual std::shared_ptr<OckDataBuffer> &InputTransfer() = 0;
    virtual std::shared_ptr<OckDataBuffer> &InputActualNum() = 0;
    virtual std::shared_ptr<OckDataBuffer> &OutputResult() = 0;
    static std::shared_ptr<OckL2NormOpDataBuffer> Create(const OckL2NormOpMeta &opSpec,
        const OckL2NormBufferMeta &bufferSpec);
};
} // namespace nop
} // namespace hcps
} // namespace ock
#endif // HCPS_OCKL2NORMOPDATABUFFER_H
