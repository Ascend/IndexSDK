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


#ifndef OCK_HCPS_HCOP_TOPK_FLAT_OP_DATA_BUFFER
#define OCK_HCPS_HCOP_TOPK_FLAT_OP_DATA_BUFFER
#include <memory>
#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"
#include "ock/acladapter/utils/OckAscendFp16.h"
#include "ock/hcps/nop/topk_flat_op/OckTopkFlatMeta.h"
#include "ock/hcps/nop/OckOpDataBuffer.h"
namespace ock {
namespace hcps {
namespace nop {
class OckTopkFlatOpDataBuffer : public OckOpDataBuffer {
public:
    virtual ~OckTopkFlatOpDataBuffer() noexcept = default;
    virtual std::shared_ptr<OckDataBuffer> &InputDists() = 0;
    virtual std::shared_ptr<OckDataBuffer> &InputMinDists() = 0;
    virtual std::shared_ptr<OckDataBuffer> &InputSizes() = 0;
    virtual std::shared_ptr<OckDataBuffer> &InputFlags() = 0;
    virtual std::shared_ptr<OckDataBuffer> &InputAttrs() = 0;
    virtual std::shared_ptr<OckDataBuffer> &OutputDists() = 0;
    virtual std::shared_ptr<OckDataBuffer> &OutputLabels() = 0;
    static std::shared_ptr<OckTopkFlatOpDataBuffer> Create(const OckTopkFlatOpMeta &opSpec,
        const OckTopkFlatBufferMeta &bufferSpec);
};
} // namespace nop
} // namespace hcps
} // namespace ock
#endif // OCK_HCPS_HCOP_TOPK_FLAT_OP_DATA_BUFFER