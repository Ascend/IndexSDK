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


#ifndef OCK_HCPS_HCOP_TOPK_DIST_COMP_OP_FACTORY
#define OCK_HCPS_HCOP_TOPK_DIST_COMP_OP_FACTORY
#include <memory>
#include <vector>
#include "OckTopkDistCompOp.h"
namespace ock {
namespace hcps {
namespace hcop {
class OckTopkDistCompOpFactory {
public:
    virtual ~OckTopkDistCompOpFactory() noexcept = default;
    static bool Support()
    {
        return true;
    }
    virtual hmm::OckHmmErrorCode AllocTmpSpace(const OckTopkDistCompOpMeta &opSpec,
        const std::shared_ptr<nop::OckTopkFlatOpDataBuffer> &topkBuffer,
        const std::vector<std::shared_ptr<nop::OckDistInt8CosMaxOpDataBuffer>> &distBuffers,
        std::shared_ptr<handler::OckHeteroHandler> handler) = 0;
    virtual std::shared_ptr<OckHeteroOperatorBase> Create(const OckTopkDistCompOpMeta &opSpec,
        const OckTopkDistCompBufferMeta &bufferSpec, const std::shared_ptr<nop::OckTopkFlatOpDataBuffer> &topkBuffer,
        const std::vector<std::shared_ptr<nop::OckDistInt8CosMaxOpDataBuffer>> &distBuffers,
        std::shared_ptr<handler::OckHeteroHandler> handler) = 0;
    static std::shared_ptr<OckTopkDistCompOpFactory> Create(const OckTopkDistCompOpMeta &opSpec);
};
} // namespace hcop
} // namespace hcps
} // namespace ock
#endif // OCK_HCPS_HCOP_TOPK_DIST_COMP_OP_FACTORY