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

#include <set>
#include "ock/hcps/nop/l2_norm_op/OckL2NormOpFactory.h"
namespace ock {
namespace hcps {
namespace nop {
class OckL2NormOpFactoryImpl : public OckL2NormOpFactory {
public:
    virtual ~OckL2NormOpFactoryImpl() noexcept = default;
    explicit OckL2NormOpFactoryImpl(const OckL2NormOpMeta &spec)
    {
        if (!Support(spec)) {
            OCK_HCPS_LOG_ERROR("meta is not supported");
        }
        auto paramsInfo = std::make_shared<NpuParamsInfo>();
        paramsInfo->inputParamTypes = { ACL_INT8, ACL_FLOAT16, ACL_UINT32 };
        paramsInfo->outputParamTypes = { ACL_FLOAT16 };
        paramsInfo->inputParamFormats = { ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND };
        paramsInfo->outputParamFormats = { ACL_FORMAT_ND };
        paramsInfo->inputParamShapes =
            std::vector<std::vector<int64_t>>{ { L2NORM_COMPUTE_BATCH, static_cast<int64_t>(spec.dims) },
            { TRANSFER_SIZE, CUBE_ALIGN },
            { SIZE_ALIGN } };
        paramsInfo->outputParamShapes = std::vector<std::vector<int64_t>>{ { L2NORM_COMPUTE_BATCH } };
        ascendOp.reset();
        ascendOp = std::make_shared<OckNpuOp>("Int8L2Norm", paramsInfo);
        ascendOp->Enable();
    }
    bool Support(const OckL2NormOpMeta &spec) const
    {
        std::set<int64_t> supportedDims = { 512, 256, 128 };
        return (supportedDims.find(spec.dims) != supportedDims.end());
    };

    std::shared_ptr<OckHeteroOperatorBase> Create(std::shared_ptr<OckOpDataBuffer> opBuffer) override
    {
        return OckNpuSingleOperator::Create(ascendOp, opBuffer, acladapter::OckTaskResourceType::DEVICE_AI_CUBE);
    };

private:
    std::shared_ptr<OckNpuOp> ascendOp{ nullptr };
};

std::shared_ptr<OckL2NormOpFactory> OckL2NormOpFactory::Create(const OckL2NormOpMeta &spec)
{
    return std::make_shared<OckL2NormOpFactoryImpl>(spec);
}
} // namespace nop
} // namespace hcps
} // namespace ock
