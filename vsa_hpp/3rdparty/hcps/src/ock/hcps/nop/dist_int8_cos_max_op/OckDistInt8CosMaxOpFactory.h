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


#ifndef OCK_HCPS_HCOP_DISTINT8COSMAX_OP_FACTORY
#define OCK_HCPS_HCOP_DISTINT8COSMAX_OP_FACTORY
#include <memory>
#include <vector>
#include "ock/hcps/nop/dist_int8_cos_max_op/OckDistInt8CosMaxMeta.h"
#include "ock/hcps/nop/OckNpuSingleOperator.h"
namespace ock {
namespace hcps {
namespace nop {
class OckDistInt8CosMaxOpFactory {
public:
    virtual ~OckDistInt8CosMaxOpFactory() noexcept = default;
    static bool Support(const OckDistInt8CosMaxOpMeta &spec);
    virtual std::shared_ptr<OckHeteroOperatorBase> Create(
        std::shared_ptr<OckOpDataBuffer> opBuffer) = 0;
    static std::shared_ptr<OckDistInt8CosMaxOpFactory> Create(const OckDistInt8CosMaxOpMeta &spec);
};
} // namespace nop
} // namespace hcps
} // namespace ock
#endif // OCK_HCPS_HCOP_DISTINT8COSMAX_OP_FACTORY