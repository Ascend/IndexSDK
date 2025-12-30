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
#include "ock/log/OckHcpsLogger.h"
#include "ock/hcps/nop/dist_mask_gen_op/OckDistMaskGenOpFactory.h"
namespace ock {
namespace hcps {
namespace nop {
class OckDistMaskGenOpFactoryImpl : public OckDistMaskGenOpFactory {
public:
    virtual ~OckDistMaskGenOpFactoryImpl() noexcept = default;
    explicit OckDistMaskGenOpFactoryImpl(const OckDistMaskGenOpMeta &spec)
    {
        auto paramsInfo = std::make_shared<NpuParamsInfo>();
        paramsInfo->inputParamTypes = { ACL_INT32, ACL_UINT8, ACL_INT32, ACL_INT32, ACL_UINT8 };
        paramsInfo->outputParamTypes = { ACL_UINT8 };
        paramsInfo->inputParamFormats = { ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND };
        paramsInfo->outputParamFormats = { ACL_FORMAT_ND };
        paramsInfo->inputParamShapes = std::vector<std::vector<int64_t>>{ { OPS_DATA_TYPE_ALIGN },
            { utils::SafeDivUp(spec.tokenNum, OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES },
            { spec.featureAttrBlockSize * spec.blockCount },
            { spec.featureAttrBlockSize * spec.blockCount },
            { spec.featureAttrBlockSize * spec.blockCount * OPS_DATA_TYPE_TIMES } };
        paramsInfo->outputParamShapes =
            std::vector<std::vector<int64_t>>{
                { utils::SafeDivUp(spec.featureAttrBlockSize * spec.blockCount, OPS_DATA_TYPE_ALIGN) } };
        ascendOp.reset();
        ascendOp = std::make_shared<OckNpuOp>("DistanceMaskGenerator", paramsInfo);
        ascendOp->Enable();
    }

    std::shared_ptr<OckHeteroOperatorBase> Create(std::shared_ptr<OckOpDataBuffer> opBuffer) override
    {
        return OckNpuSingleOperator::Create(ascendOp, opBuffer, acladapter::OckTaskResourceType::DEVICE_AI_CUBE);
    };

private:
    std::shared_ptr<OckNpuOp> ascendOp{ nullptr };
};

std::shared_ptr<OckDistMaskGenOpFactory> OckDistMaskGenOpFactory::Create(const OckDistMaskGenOpMeta &spec)
{
    return std::make_shared<OckDistMaskGenOpFactoryImpl>(spec);
}
} // namespace nop
} // namespace hcps
} // namespace ock
