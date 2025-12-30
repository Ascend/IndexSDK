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

#include "ock/hcps/nop/trans_data_shaped_op/OckTransDataShapedOpFactory.h"

namespace ock {
namespace hcps {
namespace nop {
class OckTransDataShapedOpFactoryImpl : public OckTransDataShapedOpFactory {
public:
    virtual ~OckTransDataShapedOpFactoryImpl() noexcept = default;
    explicit OckTransDataShapedOpFactoryImpl(const OckTransDataShapedOpMeta &spec)
    {
        auto paramsInfo = std::make_shared<NpuParamsInfo>();
        paramsInfo->inputParamTypes = { ACL_INT8, ACL_INT64 };
        paramsInfo->outputParamTypes = { ACL_INT8 };
        paramsInfo->inputParamFormats = { ACL_FORMAT_ND, ACL_FORMAT_ND };
        paramsInfo->outputParamFormats = { ACL_FORMAT_ND };
        paramsInfo->inputParamShapes =
            std::vector<std::vector<int64_t>>{ { static_cast<int64_t>(spec.addNum), static_cast<int64_t>(spec.dims) },
            { TRANSDATA_SHAPED_ATTR_IDX_COUNT } };
        paramsInfo->outputParamShapes =
            std::vector<std::vector<int64_t>>{ { utils::SafeDivUp(static_cast<int64_t>(spec.codeBlockSize), CUBE_ALIGN),
            utils::SafeDivUp(static_cast<int64_t>(spec.dims), CUBE_ALIGN_INT8), CUBE_ALIGN, CUBE_ALIGN_INT8 } };
        ascendOp.reset();
        ascendOp = std::make_shared<OckNpuOp>("TransdataShaped", paramsInfo);
        ascendOp->Enable();
    }

    std::shared_ptr<OckHeteroOperatorBase> Create(std::shared_ptr<OckOpDataBuffer> opBuffer) override
    {
        return OckNpuSingleOperator::Create(ascendOp, opBuffer, acladapter::OckTaskResourceType::DEVICE_AI_CPU);
    };

private:
    std::shared_ptr<OckNpuOp> ascendOp{ nullptr };
};

std::shared_ptr<OckTransDataShapedOpFactory> OckTransDataShapedOpFactory::Create(const OckTransDataShapedOpMeta &spec)
{
    return std::make_shared<OckTransDataShapedOpFactoryImpl>(spec);
}
} // namespace nop
} // namespace hcps
} // namespace ock