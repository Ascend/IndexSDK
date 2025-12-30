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

#include "ock/hcps/nop/topk_flat_op/OckTopkFlatOpFactory.h"
namespace ock {
namespace hcps {
namespace nop {
class OckTopkFlatOpFactoryImpl : public OckTopkFlatOpFactory {
public:
    virtual ~OckTopkFlatOpFactoryImpl() noexcept = default;
    explicit OckTopkFlatOpFactoryImpl(const OckTopkFlatOpMeta &opSpec)
    {
        if (!Support(opSpec)) {
            OCK_HCPS_LOG_ERROR("meta is not supported");
        }
        auto paramsInfo = std::make_shared<NpuParamsInfo>();
        paramsInfo->inputParamTypes = { ACL_FLOAT16, ACL_FLOAT16, ACL_UINT32, ACL_UINT16, ACL_INT64 };
        paramsInfo->outputParamTypes = { ACL_FLOAT16, opSpec.outLabelsDataType == "uint16" ? ACL_UINT16 : ACL_INT64 };
        paramsInfo->inputParamFormats = { ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND };
        paramsInfo->outputParamFormats = { ACL_FORMAT_ND, ACL_FORMAT_ND };
        paramsInfo->inputParamShapes = std::vector<std::vector<int64_t>>{ { 0, opSpec.batch, opSpec.codeBlockSize },
            { 0, opSpec.batch, utils::SafeDivUp(opSpec.codeBlockSize, BURST_LEN) * 2LL },
            { 0, CORE_NUM, SIZE_ALIGN },
            { 0, FLAG_NUM, FLAG_SIZE },
            { TOPK_FLAT_ATTR_IDX_COUNT } };
        paramsInfo->outputParamShapes = std::vector<std::vector<int64_t>>{ { opSpec.batch, 0 }, { opSpec.batch, 0 } };
        ascendOp.reset();
        ascendOp = std::make_shared<OckNpuOp>("TopkFlat", paramsInfo);
        ascendOp->Enable();
    }

    std::shared_ptr<OckHeteroOperatorBase> Create(std::shared_ptr<OckOpDataBuffer> opBuffer) override
    {
        return OckNpuSingleOperator::Create(ascendOp, opBuffer, acladapter::OckTaskResourceType::DEVICE_AI_CPU);
    };

private:
    std::shared_ptr<OckNpuOp> ascendOp{ nullptr };
};

std::shared_ptr<OckTopkFlatOpFactory> OckTopkFlatOpFactory::Create(const OckTopkFlatOpMeta &opSpec)
{
    return std::make_shared<OckTopkFlatOpFactoryImpl>(opSpec);
}
} // namespace nop
} // namespace hcps
} // namespace ock