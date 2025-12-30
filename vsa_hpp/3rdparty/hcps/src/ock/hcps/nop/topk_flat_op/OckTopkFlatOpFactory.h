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


#ifndef OCK_HCPS_HCOP_TOPK_FLAT_OP_FACTORY
#define OCK_HCPS_HCOP_TOPK_FLAT_OP_FACTORY
#include <memory>
#include <vector>
#include <set>
#include "ock/hcps/nop/topk_flat_op/OckTopkFlatMeta.h"
#include "ock/hcps/nop/OckNpuSingleOperator.h"
namespace ock {
namespace hcps {
namespace nop {
class OckTopkFlatOpFactory {
public:
    virtual ~OckTopkFlatOpFactory() noexcept = default;
    static bool Support(const OckTopkFlatOpMeta &spec)
    {
        std::set<std::string> supportedLabelType = { "int64", "uint16" };
        std::set<int64_t> supportedBatch{ 64, 48, 36, 32, 24, 18, 16, 12, 8, 6, 4, 2, 1 };
        return (supportedLabelType.find(spec.outLabelsDataType) != supportedLabelType.end()) &&
            (supportedBatch.find(spec.batch) != supportedBatch.end()) && (spec.codeBlockSize % UNIT_BLOCK_SIZE == 0) &&
            (spec.codeBlockSize >= UNIT_BLOCK_SIZE);
    };
    virtual std::shared_ptr<OckHeteroOperatorBase> Create(std::shared_ptr<OckOpDataBuffer> opBuffer) = 0;
    static std::shared_ptr<OckTopkFlatOpFactory> Create(const OckTopkFlatOpMeta &opSpec);
};
} // namespace nop
} // namespace hcps
} // namespace ock
#endif // OCK_HCPS_HCOP_TOPK_FLAT_OP_FACTORY