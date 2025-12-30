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


#ifndef HCPS_OCKTRANSDATASHAPEDOPFACTORY_H
#define HCPS_OCKTRANSDATASHAPEDOPFACTORY_H
#include <memory>
#include <vector>
#include "ock/hcps/nop/trans_data_shaped_op/OckTransDataShapedMeta.h"
#include "ock/hcps/nop/OckNpuSingleOperator.h"

namespace ock {
namespace hcps {
namespace nop {
class OckTransDataShapedOpFactory {
public:
    virtual ~OckTransDataShapedOpFactory() noexcept = default;
    virtual std::shared_ptr<OckHeteroOperatorBase> Create(std::shared_ptr<OckOpDataBuffer> opBuffer) = 0;
    static std::shared_ptr<OckTransDataShapedOpFactory> Create(const OckTransDataShapedOpMeta &spec);
};
}
}
}
#endif // HCPS_OCKTRANSDATASHAPEDOPFACTORY_H
