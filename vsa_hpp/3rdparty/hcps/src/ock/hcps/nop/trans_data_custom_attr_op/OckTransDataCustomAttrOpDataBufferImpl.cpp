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
#include "ock/hcps/nop/trans_data_custom_attr_op/OckTransDataCustomAttrOpDataBuffer.h"

namespace ock {
namespace hcps {
namespace nop {
class OckTransDataCustomAttrOpDataBufferImpl : public OckOpDataBufferGen<OckTransDataCustomAttrOpDataBuffer> {
public:
    virtual ~OckTransDataCustomAttrOpDataBufferImpl() noexcept = default;
    OckTransDataCustomAttrOpDataBufferImpl(const OckTransDataCustomAttrOpMeta &opSpec,
        const OckTransDataCustomAttrBufferMeta &bufferSpec)
    {
        numInputs = 2U;
        numOutputs = 1U;
        AddParamInfo<uint8_t>({ opSpec.copyCount, opSpec.customAttrLen });
        AddParamInfo<int64_t>({ TRANSDATA_CUSTOM_ATTR_IDX_COUNT });
        AddParamInfo<uint8_t>({ opSpec.customAttrLen, opSpec.customAttrBlockSize });
        inputParams.resize(numInputs);
        outputParams.resize(numOutputs);
    }
    OckHcpsErrorCode AllocBuffersFromHmoBlock(std::shared_ptr<OckTransDataCustomAttrOpHmoBlock> hmoBlock,
        std::shared_ptr<hmm::OckHmmHeteroMemoryMgrBase> devMgr) override
    {
        auto dataBuffer = hmoBlock->srcHmo->GetBuffer(hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR, 0U,
            hmoBlock->srcHmo->GetByteSize());
        if (dataBuffer.get() == nullptr) {
            OCK_HCPS_LOG_ERROR("hmo block src hmo get buffer failed!");
            return HCPS_ERROR_GET_BUFFER_FAILED;
        }
        inputParams[0] = std::make_shared<OckDataBuffer>(dataBuffer, paramsShapes[0U]);

        auto ret = devMgr->Alloc(paramsByteSizes[1U], hmm::OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
        if (ret.first != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("attr hmo alloc failed, the errorCode is " << ret.first);
            return ret.first;
        }
        inputParams[1] = std::make_shared<OckDataBuffer>(ret.second, paramsShapes[1U]);

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

std::shared_ptr<OckTransDataCustomAttrOpDataBuffer> OckTransDataCustomAttrOpDataBuffer::Create(
    const OckTransDataCustomAttrOpMeta &opSpec, const OckTransDataCustomAttrBufferMeta &bufferSpec)
{
    return std::make_shared<OckTransDataCustomAttrOpDataBufferImpl>(opSpec, bufferSpec);
}
} // namespace nop
} // namespace hcps
} // namespace ock