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


#ifndef OCK_HCPS_HCOP_DISTINT8COSMAX_OP_DATA_BUFFER
#define OCK_HCPS_HCOP_DISTINT8COSMAX_OP_DATA_BUFFER
#include <memory>
#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"
#include "ock/acladapter/utils/OckAscendFp16.h"
#include "ock/hcps/nop/dist_int8_cos_max_op/OckDistInt8CosMaxMeta.h"
#include "ock/hcps/nop/OckOpDataBuffer.h"
namespace ock {
namespace hcps {
namespace nop {
class OckDistInt8CosMaxOpDataBuffer : public OckOpDataBuffer {
public:
    virtual ~OckDistInt8CosMaxOpDataBuffer() noexcept = default;
    virtual std::shared_ptr<OckDataBuffer> &InputQueries() = 0;
    virtual std::shared_ptr<OckDataBuffer> &InputMask() = 0;
    virtual std::shared_ptr<OckDataBuffer> &InputShaped() = 0;
    virtual std::shared_ptr<OckDataBuffer> &InputQueriesNorm() = 0;
    virtual std::shared_ptr<OckDataBuffer> &InputCodesNorm() = 0;
    virtual std::shared_ptr<OckDataBuffer> &InputActualSize() = 0;
    virtual std::shared_ptr<OckDataBuffer> &OutputDists() = 0;
    virtual std::shared_ptr<OckDataBuffer> &OutputMaxDists() = 0;
    virtual std::shared_ptr<OckDataBuffer> &OutputFlag() = 0;
    static std::shared_ptr<OckDistInt8CosMaxOpDataBuffer> Create(const OckDistInt8CosMaxOpMeta &opSpec,
        const OckDistInt8CosMaxBufferMeta &bufferSpec);
};
} // namespace nop
} // namespace hcps
} // namespace ock
#endif // OCK_HCPS_HCOP_DISTINT8COSMAX_OP_DATA_BUFFER