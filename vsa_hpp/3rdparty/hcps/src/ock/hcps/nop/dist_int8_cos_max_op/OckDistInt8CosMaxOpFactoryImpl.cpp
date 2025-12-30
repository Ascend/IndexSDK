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
#include "ock/hcps/nop/dist_int8_cos_max_op/OckDistInt8CosMaxOpFactory.h"
namespace ock {
namespace hcps {
namespace nop {
class OckDistInt8CosMaxOpFactoryImpl : public OckDistInt8CosMaxOpFactory {
public:
    virtual ~OckDistInt8CosMaxOpFactoryImpl() noexcept = default;
    explicit OckDistInt8CosMaxOpFactoryImpl(const OckDistInt8CosMaxOpMeta &spec)
    {
        if (!Support(spec)) {
            OCK_HCPS_LOG_ERROR("meta is not supported");
        }
        auto paramsInfo = std::make_shared<NpuParamsInfo>();
        paramsInfo->inputParamTypes = { ACL_INT8, ACL_UINT8, ACL_INT8, ACL_FLOAT16, ACL_FLOAT16, ACL_UINT32 };
        paramsInfo->outputParamTypes = { ACL_FLOAT16, ACL_FLOAT16, ACL_UINT16 };
        paramsInfo->inputParamFormats = { ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
            ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND };
        paramsInfo->outputParamFormats = { ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND };
        paramsInfo->inputParamShapes = std::vector<std::vector<int64_t>>{ { spec.batch, spec.dims },
            { spec.batch, utils::SafeDiv(spec.codeBlockSize, 8LL) }, // divUp to 8
            { utils::SafeDiv(spec.codeBlockSize, CUBE_ALIGN), utils::SafeDiv(spec.dims, CUBE_ALIGN_INT8), CUBE_ALIGN,
            CUBE_ALIGN_INT8 },
            { utils::SafeRoundUp(spec.batch, FP16_ALIGN) },
            { spec.codeBlockSize },
            { CORE_NUM, SIZE_ALIGN } };
        paramsInfo->outputParamShapes = std::vector<std::vector<int64_t>>{ { spec.batch, int64_t(spec.codeBlockSize) },
            { spec.batch, utils::SafeDivUp(spec.codeBlockSize, BURST_LEN) * 2LL },
            { FLAG_NUM, FLAG_SIZE } };
        ascendOp.reset();
        ascendOp = std::make_shared<OckNpuOp>("DistanceInt8CosMaxs", paramsInfo);
        ascendOp->Enable();
    }
    bool Support(const OckDistInt8CosMaxOpMeta &spec) const
    {
        std::set<int64_t> supportedBatch = { 64, 48, 36, 32, 24, 18, 16, 12, 8, 6, 4, 2, 1 };
        std::set<int64_t> supportedDims = { 512, 256, 128 };
        return (supportedBatch.find(spec.batch) != supportedBatch.end()) &&
            (supportedDims.find(spec.dims) != supportedDims.end()) && (spec.codeBlockSize % UNIT_BLOCK_SIZE == 0) &&
            (spec.codeBlockSize >= UNIT_BLOCK_SIZE);
    };

    std::shared_ptr<OckHeteroOperatorBase> Create(std::shared_ptr<OckOpDataBuffer> opBuffer) override
    {
        return OckNpuSingleOperator::Create(ascendOp, opBuffer, acladapter::OckTaskResourceType::DEVICE_AI_CUBE);
    };

private:
    std::shared_ptr<OckNpuOp> ascendOp{ nullptr };
};

std::shared_ptr<OckDistInt8CosMaxOpFactory> OckDistInt8CosMaxOpFactory::Create(const OckDistInt8CosMaxOpMeta &spec)
{
    return std::make_shared<OckDistInt8CosMaxOpFactoryImpl>(spec);
}
} // namespace nop
} // namespace hcps
} // namespace ock