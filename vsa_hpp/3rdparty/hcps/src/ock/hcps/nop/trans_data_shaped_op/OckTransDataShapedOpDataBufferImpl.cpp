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

#include "ock/log/OckHcpsLogger.h"
#include "ock/hcps/nop/OckOpDataBufferGen.h"
#include "ock/hcps/nop/trans_data_shaped_op/OckTransDataShapedOpDataBuffer.h"

namespace ock {
namespace hcps {
namespace nop {
class OckTransDataShapedOpDataBufferImpl : public OckOpDataBufferGen<OckTransDataShapedOpDataBuffer> {
public:
    virtual ~OckTransDataShapedOpDataBufferImpl() noexcept = default;
    OckTransDataShapedOpDataBufferImpl(const OckTransDataShapedOpMeta &opSpec,
        const OckTransDataShapedBufferMeta &bufferSpec)
    {
        numInputs = 2U;
        numOutputs = 1U;
        AddParamInfo<int8_t>({ static_cast<int64_t>(bufferSpec.ntotal), static_cast<int64_t>(opSpec.dims) });
        AddParamInfo<int64_t>({ TRANSDATA_SHAPED_ATTR_IDX_COUNT });
        AddParamInfo<int8_t>({
            utils::SafeDivUp(static_cast<int64_t>(opSpec.codeBlockSize), CUBE_ALIGN),
            utils::SafeDivUp(static_cast<int64_t>(opSpec.dims), CUBE_ALIGN_INT8),
            CUBE_ALIGN,
            CUBE_ALIGN_INT8
        });
        inputParams.resize(numInputs);
        outputParams.resize(numOutputs);
    }
    OckHcpsErrorCode AllocBuffersFromHmoBlock(std::shared_ptr<OckTransDataShapedOpHmoBlock> hmoBlock,
        std::shared_ptr<hmm::OckHmmHeteroMemoryMgrBase> devMgr) override
    {
        auto dataBuffer = hmoBlock->srcHmo->GetBuffer(hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR, 0U,
            hmoBlock->srcHmo->GetByteSize());
        if (dataBuffer.get() == nullptr) {
            OCK_HCPS_LOG_ERROR("hmo block src hmo get buffer failed!");
            return HCPS_ERROR_GET_BUFFER_FAILED;
        }
        inputParams[0] = std::make_shared<OckDataBuffer>(dataBuffer, paramsShapes[0U]);

        auto attrRet = devMgr->Alloc(paramsByteSizes[1U], hmm::OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
        if (attrRet.first != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("attr hmo alloc failed, the errorCode is " << attrRet.first);
            return attrRet.first;
        }
        inputParams[1] = std::make_shared<OckDataBuffer>(attrRet.second, paramsShapes[1U]);

        outputParams[0] = std::make_shared<OckDataBuffer>(hmoBlock->dstHmo, paramsShapes[2U]);

        return hmm::HMM_SUCCESS;
    }
    std::shared_ptr<OckDataBuffer> &InputSrc() override
    {
        return inputParams[0U];
    }
    std::shared_ptr<OckDataBuffer> &InputAttr() override
    {
        return inputParams[1U];
    }
    
    std::shared_ptr<OckDataBuffer> &OutputDst() override
    {
        return outputParams[0U];
    }
};

std::shared_ptr<OckTransDataShapedOpDataBuffer> OckTransDataShapedOpDataBuffer::Create(
    const OckTransDataShapedOpMeta &opSpec, const OckTransDataShapedBufferMeta &bufferSpec)
{
    return std::make_shared<OckTransDataShapedOpDataBufferImpl>(opSpec, bufferSpec);
}
} // namespace nop
} // namespace hcps
} // namespace ock