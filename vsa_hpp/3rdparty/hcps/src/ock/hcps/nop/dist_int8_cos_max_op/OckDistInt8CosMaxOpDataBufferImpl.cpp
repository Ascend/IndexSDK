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

#include <utility>
#include "ock/hcps/nop/OckOpDataBufferGen.h"
#include "ock/hcps/nop/dist_int8_cos_max_op/OckDistInt8CosMaxOpDataBuffer.h"

namespace ock {
namespace hcps {
namespace nop {
class OckDistInt8CosMaxOpDataBufferImpl : public OckOpDataBufferGen<OckDistInt8CosMaxOpDataBuffer> {
public:
    virtual ~OckDistInt8CosMaxOpDataBufferImpl() noexcept = default;
    OckDistInt8CosMaxOpDataBufferImpl(const OckDistInt8CosMaxOpMeta &opSpec,
        const OckDistInt8CosMaxBufferMeta &bufferSpec)
    {
        numInputs = 6U;
        numOutputs = 3U;
        int64_t burstsOfBlock = utils::SafeDivUp(opSpec.codeBlockSize, BURST_LEN) * 2;
        int64_t idxMaskLen = utils::SafeDivUp(std::max(bufferSpec.ntotal, opSpec.codeBlockSize), BINARY_BYTE_SIZE);
        AddParamInfo<int8_t>({ opSpec.batch, opSpec.dims });
        AddParamInfo<uint8_t>({ opSpec.batch, idxMaskLen });
        AddParamInfo<int8_t>({
            utils::SafeDiv(opSpec.codeBlockSize, CUBE_ALIGN),
            utils::SafeDiv(opSpec.dims, CUBE_ALIGN_INT8), CUBE_ALIGN, CUBE_ALIGN_INT8
        });
        AddParamInfo<OckFloat16>({ utils::SafeRoundUp(opSpec.batch, FP16_ALIGN) });
        AddParamInfo<OckFloat16>({ opSpec.codeBlockSize });
        AddParamInfo<uint32_t>({ CORE_NUM, SIZE_ALIGN });
        AddParamInfo<OckFloat16>({ opSpec.batch, opSpec.codeBlockSize });
        AddParamInfo<OckFloat16>({ opSpec.batch, burstsOfBlock });
        AddParamInfo<uint16_t>({ FLAG_NUM, FLAG_SIZE });
        inputParams.resize(numInputs);
        outputParams.resize(numOutputs);
    }
    std::shared_ptr<OckDataBuffer> &InputQueries() override
    {
        return inputParams[0U];
    }
    std::shared_ptr<OckDataBuffer> &InputMask() override
    {
        return inputParams[1U];
    }
    std::shared_ptr<OckDataBuffer> &InputShaped() override
    {
        return inputParams[2U];
    }
    std::shared_ptr<OckDataBuffer> &InputQueriesNorm() override
    {
        return inputParams[3U];
    }
    std::shared_ptr<OckDataBuffer> &InputCodesNorm() override
    {
        return inputParams[4U];
    }
    std::shared_ptr<OckDataBuffer> &InputActualSize() override
    {
        return inputParams[5U];
    }
    std::shared_ptr<OckDataBuffer> &OutputDists() override
    {
        return outputParams[0U];
    }
    std::shared_ptr<OckDataBuffer> &OutputMaxDists() override
    {
        return outputParams[1U];
    }
    std::shared_ptr<OckDataBuffer> &OutputFlag() override
    {
        return outputParams[2U];
    }
};

std::shared_ptr<OckDistInt8CosMaxOpDataBuffer> OckDistInt8CosMaxOpDataBuffer::Create(
    const OckDistInt8CosMaxOpMeta &opSpec, const OckDistInt8CosMaxBufferMeta &bufferSpec)
{
    return std::make_shared<OckDistInt8CosMaxOpDataBufferImpl>(opSpec, bufferSpec);
}
} // namespace nop
} // namespace hcps
} // namespace ock