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


#ifndef HCPS_PIER_OCKDISTMASKWITHEXTRAGENOPDATABUFFER_H
#define HCPS_PIER_OCKDISTMASKWITHEXTRAGENOPDATABUFFER_H
#include <memory>
#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"
#include "ock/hcps/error/OckHcpsErrorCode.h"
#include "ock/hcps/nop/dist_mask_with_extra_gen_op/OckDistMaskWithExtraGenMeta.h"
#include "ock/hcps/nop/OckOpDataBuffer.h"
namespace ock {
namespace hcps {
namespace nop {
class OckDistMaskWithExtraGenOpDataBuffer : public OckOpDataBuffer {
public:
    virtual ~OckDistMaskWithExtraGenOpDataBuffer() noexcept = default;
    virtual std::shared_ptr<OckDataBuffer> &InputQueryTime() = 0;
    virtual std::shared_ptr<OckDataBuffer> &InputTokenBitSet() = 0;
    virtual std::shared_ptr<OckDataBuffer> &InputAttrTimes() = 0;
    virtual std::shared_ptr<OckDataBuffer> &InputAttrTokenQs() = 0;
    virtual std::shared_ptr<OckDataBuffer> &InputAttrTokenRs() = 0;
    virtual std::shared_ptr<OckDataBuffer> &InputExtraMask() = 0;
    virtual std::shared_ptr<OckDataBuffer> &OutputMask() = 0;
    virtual OckHcpsErrorCode AllocBuffersFromHmoGroup(std::shared_ptr<OckDistMaskWithExtraGenOpHmoGroup> hmoGroup,
                                                      uint64_t offsetInSingleMask, uint64_t groupMaskLen) = 0;
    static std::shared_ptr<OckDistMaskWithExtraGenOpDataBuffer> Create(const OckDistMaskWithExtraGenOpMeta &opSpec);
};
} // namespace nop
} // namespace hcps
} // namespace ock
#endif // HCPS_PIER_OCKDISTMASKWITHEXTRAGENOPDATABUFFER_H
