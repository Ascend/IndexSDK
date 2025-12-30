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
#include "ock/hcps/nop/topk_flat_op/OckTopkFlatOpDataBuffer.h"

namespace ock {
namespace hcps {
namespace nop {
class OckTopkFlatOpDataBufferImpl : public OckOpDataBufferGen<OckTopkFlatOpDataBuffer> {
public:
    virtual ~OckTopkFlatOpDataBufferImpl() noexcept = default;
    OckTopkFlatOpDataBufferImpl(const OckTopkFlatOpMeta &opSpec, const OckTopkFlatBufferMeta &bufferSpec)
    {
        numInputs = 5U;
        numOutputs = 2U;

        AddParamInfo<OckFloat16>({ bufferSpec.blockNum, opSpec.batch, opSpec.codeBlockSize });
        AddParamInfo<OckFloat16>({
            bufferSpec.blockNum, opSpec.batch, utils::SafeDivUp(opSpec.codeBlockSize, BURST_LEN) * 2U
        });
        AddParamInfo<uint32_t>({ bufferSpec.blockNum, CORE_NUM, SIZE_ALIGN });
        AddParamInfo<uint16_t>({ bufferSpec.blockNum, FLAG_NUM, FLAG_SIZE });
        AddParamInfo<int64_t>({ TOPK_FLAT_ATTR_IDX_COUNT });
        AddParamInfo<OckFloat16>({ opSpec.batch, bufferSpec.k });
        if (opSpec.outLabelsDataType == "uint16") {
            AddParamInfo<uint16_t>({ opSpec.batch, bufferSpec.k });
        } else {
            AddParamInfo<int64_t>(std::vector<int64_t>{ opSpec.batch, bufferSpec.k });
        }

        inputParams.resize(numInputs);
        outputParams.resize(numOutputs);
    }
    std::shared_ptr<OckDataBuffer> &InputDists()
    {
        return inputParams[0U];
    }
    std::shared_ptr<OckDataBuffer> &InputMinDists()
    {
        return inputParams[1U];
    }
    std::shared_ptr<OckDataBuffer> &InputSizes()
    {
        return inputParams[2U];
    }
    std::shared_ptr<OckDataBuffer> &InputFlags()
    {
        return inputParams[3U];
    }
    std::shared_ptr<OckDataBuffer> &InputAttrs()
    {
        return inputParams[4U];
    }
    std::shared_ptr<OckDataBuffer> &OutputDists()
    {
        return outputParams[0U];
    }
    std::shared_ptr<OckDataBuffer> &OutputLabels()
    {
        return outputParams[1U];
    }
};

std::shared_ptr<OckTopkFlatOpDataBuffer> OckTopkFlatOpDataBuffer::Create(const OckTopkFlatOpMeta &opSpec,
    const OckTopkFlatBufferMeta &bufferSpec)
{
    return std::make_shared<OckTopkFlatOpDataBufferImpl>(opSpec, bufferSpec);
}
} // namespace nop
} // namespace hcps
} // namespace ock