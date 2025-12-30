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
#include "ock/hcps/nop/remove_data_attr_op/OckRemoveDataAttrOpDataBuffer.h"

namespace ock {
namespace hcps {
namespace nop {
class OckRemoveDataAttrOpDataBufferImpl : public OckOpDataBufferGen<OckRemoveDataAttrOpDataBuffer> {
public:
    virtual ~OckRemoveDataAttrOpDataBufferImpl() noexcept = default;
    OckRemoveDataAttrOpDataBufferImpl(const OckRemoveDataAttrOpMeta &opSpec,
        const OckRemoveDataAttrBufferMeta &bufferSpec)
    {
        numInputs = 2U;
        numOutputs = 1U;
        AddParamInfo<uint8_t>({ opSpec.removeSize });
        AddParamInfo<int64_t>({ REMOVEDATA_ATTR_IDX_COUNT });
        AddParamInfo<uint8_t>({ opSpec.removeSize });
        inputParams.resize(numInputs);
        outputParams.resize(numOutputs);
    }
    OckHcpsErrorCode AllocBuffersFromHmoBlock(std::shared_ptr<OckRemoveDataAttrOpHmoBlock> hmoBlock,
        std::shared_ptr<hmm::OckHmmHeteroMemoryMgrBase> devMgr) override
    {
        inputParams[0] = std::make_shared<OckDataBuffer>(hmoBlock->srcHmo, paramsShapes[0U]);

        auto ret = devMgr->Alloc(paramsByteSizes[1U], hmm::OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
        if (ret.first != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("RemoveDataAttrOp attr hmo alloc failed, the errorCode is " << ret.first);
            return ret.first;
        }
        inputParams[1] = std::make_shared<OckDataBuffer>(ret.second, paramsShapes[1U]);
        std::vector<int64_t> attr = { static_cast<int64_t>(hmoBlock->dataType),
            static_cast<int64_t>(hmoBlock->copyNum) };
        FillBuffer<int64_t>(inputParams[1], attr);

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

std::shared_ptr<OckRemoveDataAttrOpDataBuffer> OckRemoveDataAttrOpDataBuffer::Create(
    const OckRemoveDataAttrOpMeta &opSpec, const OckRemoveDataAttrBufferMeta &bufferSpec)
{
    return std::make_shared<OckRemoveDataAttrOpDataBufferImpl>(opSpec, bufferSpec);
}
} // namespace nop
} // namespace hcps
} // namespace ock