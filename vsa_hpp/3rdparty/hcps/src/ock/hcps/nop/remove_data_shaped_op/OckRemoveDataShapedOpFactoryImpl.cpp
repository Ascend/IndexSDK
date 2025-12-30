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


#include "ock/hcps/nop/remove_data_shaped_op/OckRemoveDataShapedOpFactory.h"
namespace ock {
namespace hcps {
namespace nop {
class OckRemoveDataShapedOpFactoryImpl : public OckRemoveDataShapedOpFactory {
public:
    virtual ~OckRemoveDataShapedOpFactoryImpl() noexcept = default;
    explicit OckRemoveDataShapedOpFactoryImpl(const OckRemoveDataShapedOpMeta &spec)
    {
        auto paramsInfo = std::make_shared<NpuParamsInfo>();
        paramsInfo->inputParamTypes = { ACL_UINT64, ACL_INT64 };
        paramsInfo->outputParamTypes = { ACL_UINT64 };
        paramsInfo->inputParamFormats = { ACL_FORMAT_ND, ACL_FORMAT_ND };
        paramsInfo->outputParamFormats = { ACL_FORMAT_ND };
        paramsInfo->inputParamShapes = std::vector<std::vector<int64_t>>{ { static_cast<int64_t>(spec.removeCount) },
                                                                          { REMOVEDATA_SHAPED_IDX_COUNT } };
        paramsInfo->outputParamShapes = std::vector<std::vector<int64_t>>{ { static_cast<int64_t>(spec.removeCount) } };
        ascendOp.reset();
        ascendOp = std::make_shared<OckNpuOp>("RemovedataShaped", paramsInfo);
        ascendOp->Enable();
    }

    std::shared_ptr<OckHeteroOperatorBase> Create(std::shared_ptr<OckOpDataBuffer> opBuffer) override
    {
        return OckNpuSingleOperator::Create(ascendOp, opBuffer, acladapter::OckTaskResourceType::DEVICE_AI_CPU);
    };

private:
    std::shared_ptr<OckNpuOp> ascendOp{ nullptr };
};

std::shared_ptr<OckRemoveDataShapedOpFactory> OckRemoveDataShapedOpFactory::Create(
    const OckRemoveDataShapedOpMeta &spec)
{
    return std::make_shared<OckRemoveDataShapedOpFactoryImpl>(spec);
}
} // namespace nop
} // namespace hcps
} // namespace ock
