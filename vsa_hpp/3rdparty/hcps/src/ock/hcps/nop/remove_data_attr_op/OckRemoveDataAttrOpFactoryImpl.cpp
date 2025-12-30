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

#include "ock/hcps/nop/remove_data_attr_op/OckRemoveDataAttrOpFactory.h"

namespace ock {
namespace hcps {
namespace nop {
class OckRemoveDataAttrOpFactoryImpl : public OckRemoveDataAttrOpFactory {
public:
    virtual ~OckRemoveDataAttrOpFactoryImpl() noexcept = default;
    explicit OckRemoveDataAttrOpFactoryImpl(const OckRemoveDataAttrOpMeta &spec)
    {
        auto paramsInfo = std::make_shared<NpuParamsInfo>();
        paramsInfo->inputParamTypes = { ACL_UINT64, ACL_INT64};
        paramsInfo->outputParamTypes = { ACL_UINT64 };
        paramsInfo->inputParamFormats = { ACL_FORMAT_ND, ACL_FORMAT_ND };
        paramsInfo->outputParamFormats = { ACL_FORMAT_ND };
        paramsInfo->inputParamShapes =
            std::vector<std::vector<int64_t>>{ { spec.removeSize }, { REMOVEDATA_ATTR_IDX_COUNT } };
        paramsInfo->outputParamShapes = std::vector<std::vector<int64_t>>{ { spec.removeSize } };
        ascendOp.reset();
        ascendOp = std::make_shared<OckNpuOp>("RemovedataAttr", paramsInfo); // 算子名字
        ascendOp->Enable();
    }

    std::shared_ptr<OckHeteroOperatorBase> Create(std::shared_ptr<OckOpDataBuffer> opBuffer) override
    {
        return OckNpuSingleOperator::Create(ascendOp, opBuffer,
            acladapter::OckTaskResourceType::DEVICE_AI_CPU); // ai cpu 算子
    };

private:
    std::shared_ptr<OckNpuOp> ascendOp{ nullptr };
};

std::shared_ptr<OckRemoveDataAttrOpFactory> OckRemoveDataAttrOpFactory::Create(
    const OckRemoveDataAttrOpMeta &spec)
{
    return std::make_shared<OckRemoveDataAttrOpFactoryImpl>(spec);
}
} // namespace nop
} // namespace hcps
} // namespace ock