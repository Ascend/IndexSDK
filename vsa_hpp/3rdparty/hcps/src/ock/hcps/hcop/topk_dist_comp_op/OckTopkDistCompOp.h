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


#ifndef OCK_HCPS_HCOP_ASCEND_COMPOSED_OPERATOR_H
#define OCK_HCPS_HCOP_ASCEND_COMPOSED_OPERATOR_H
#include <vector>
#include <memory>
#include "acl/acl.h"
#include "ock/hcps/stream/OckHeteroOperatorBase.h"
#include "ock/hcps/nop/topk_flat_op/OckTopkFlatOpFactory.h"
#include "ock/hcps/nop/dist_int8_cos_max_op/OckDistInt8CosMaxOpFactory.h"
#include "ock/hcps/hcop/topk_dist_comp_op/OckTopkDistCompMeta.h"
#include "OckTopkDistCompOpDataBuffer.h"
#include "ock/hcps/handler/OckHandlerHmmHelper.h"

namespace ock {
namespace hcps {
namespace hcop {
/*
@brief 调度方会确保 OckAscendOperator 的调用在指定的设备的acl环境中
*/
class OckTopkDistCompOp : public OckHeteroOperatorBase {
public:
    virtual ~OckTopkDistCompOp() noexcept = default;
    static std::shared_ptr<OckHeteroOperatorBase> Create(
        int64_t numDistOps,
        const std::shared_ptr<nop::OckTopkFlatOpFactory> &topkOpFactory,
        const std::shared_ptr<nop::OckDistInt8CosMaxOpFactory> &distOpFactory,
        const std::shared_ptr<nop::OckTopkFlatOpDataBuffer> &topkBuffer,
        const std::vector<std::shared_ptr<nop::OckDistInt8CosMaxOpDataBuffer>> &distBuffers,
        std::shared_ptr<handler::OckHeteroHandler> handler);
};
} // namespace hcop
} // namespace hcps
} // namespace ock
#endif // OCK_HCPS_HCOP_ASCEND_COMPOSED_OPERATOR_H