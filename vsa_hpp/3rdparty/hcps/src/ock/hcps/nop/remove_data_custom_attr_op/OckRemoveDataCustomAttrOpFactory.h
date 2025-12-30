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


#ifndef HCPS_OCK_REMOVE_DATA_CUSTOM_ATTR_OP_FACTORY_H
#define HCPS_OCK_REMOVE_DATA_CUSTOM_ATTR_OP_FACTORY_H
#include <memory>
#include <vector>
#include "ock/hcps/nop/OckNpuSingleOperator.h"
#include "ock/hcps/nop/remove_data_custom_attr_op/OckRemoveDataCustomAttrMeta.h"

namespace ock {
namespace hcps {
namespace nop {
class OckRemoveDataCustomAttrOpFactory {
public:
    virtual ~OckRemoveDataCustomAttrOpFactory() noexcept = default;
    virtual std::shared_ptr<OckHeteroOperatorBase> Create(
        std::shared_ptr<OckOpDataBuffer> opBuffer) = 0;
    static std::shared_ptr<OckRemoveDataCustomAttrOpFactory> Create(const OckRemoveDataCustomAttrOpMeta &spec);
};
}
}
}
#endif // HCPS_OCK_REMOVE_DATA_CUSTOM_ATTR_OP_FACTORY_H
