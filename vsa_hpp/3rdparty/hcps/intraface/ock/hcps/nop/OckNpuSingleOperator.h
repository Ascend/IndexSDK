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


#ifndef OCK_HCPS_NOP_NPU_SINGLE_OPERATOR_H
#define OCK_HCPS_NOP_NPU_SINGLE_OPERATOR_H
#include <vector>
#include <memory>
#include "acl/acl.h"
#include "ock/hcps/nop/OckNpuOp.h"
#include "ock/hcps/nop/OckOpDataBuffer.h"
#include "ock/hcps/stream/OckHeteroOperatorBase.h"
namespace ock {
namespace hcps {
namespace nop {
/*
@brief 调度方会确保 OckAscendOperator 的调用在指定的设备的acl环境中
*/
class OckNpuSingleOperator : public OckHeteroOperatorBase {
public:
    virtual ~OckNpuSingleOperator() noexcept = default;
    static std::shared_ptr<OckHeteroOperatorBase> Create(std::shared_ptr<OckNpuOp> &ascendOp,
        std::shared_ptr<OckOpDataBuffer> opBuffer, acladapter::OckTaskResourceType resourceType);
};
} // namespace nop
} // namespace hcps
} // namespace ock
#endif // OCK_HCPS_NOP_NPU_SINGLE_OPERATOR_H