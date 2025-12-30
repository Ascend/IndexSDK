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
#include "ock/hcps/nop/remove_data_shaped_op/OckRemoveDataShapedOpDataBuffer.h"
namespace ock {
namespace hcps {
namespace nop {
class OckRemoveDataShapedOpDataBufferImpl : public OckOpDataBufferGen<OckRemoveDataShapedOpDataBuffer> {
public:
    virtual ~OckRemoveDataShapedOpDataBufferImpl() noexcept = default;
    explicit OckRemoveDataShapedOpDataBufferImpl(const OckRemoveDataShapedOpMeta &opSpec)
    {
        numInputs = 2U;
        numOutputs = 1U;
        AddParamInfo<uint64_t>({ opSpec.removeCount });
        AddParamInfo<int64_t>({ REMOVEDATA_SHAPED_IDX_COUNT });
        AddParamInfo<uint64_t>({ opSpec.removeCount });
        inputParams.resize(numInputs);
        outputParams.resize(numOutputs);
    }
    OckHcpsErrorCode AllocBuffersFromHmoBlock(std::shared_ptr<OckRemoveDataShapedOpHmoBlock> hmoBlock,
        std::shared_ptr<hmm::OckHmmHeteroMemoryMgrBase> devMgr) override
    {
        auto srcBuffer = hmoBlock->srcPosHmo->GetBuffer(hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR, 0U,
            hmoBlock->srcPosHmo->GetByteSize());
        if (srcBuffer.get() == nullptr) {
            OCK_HCPS_LOG_ERROR("hmo block src pos hmo get buffer failed!");
            return HCPS_ERROR_GET_BUFFER_FAILED;
        }
        inputParams[0] = std::make_shared<OckDataBuffer>(srcBuffer, paramsShapes[0U]);

        auto attrRet = devMgr->Alloc(paramsByteSizes[1U], hmm::OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
        if (attrRet.first != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("attr hmo alloc failed, the errorCode is " << attrRet.first);
            return attrRet.first;
        }
        inputParams[1] = std::make_shared<OckDataBuffer>(attrRet.second, paramsShapes[1U]);

        auto dstBuffer = hmoBlock->dstPosHmo->GetBuffer(hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR, 0U,
            hmoBlock->dstPosHmo->GetByteSize());
        if (dstBuffer.get() == nullptr) {
            OCK_HCPS_LOG_ERROR("hmo block dst pos hmo get buffer failed!");
            return HCPS_ERROR_GET_BUFFER_FAILED;
        }
        outputParams[0] = std::make_shared<OckDataBuffer>(dstBuffer, paramsShapes[2U]);

        return hmm::HMM_SUCCESS;
    }
    std::shared_ptr<OckDataBuffer> &InputSrcPos() override
    {
        return inputParams[0U];
    }
    std::shared_ptr<OckDataBuffer> &InputAttrs() override
    {
        return inputParams[1U];
    }

    std::shared_ptr<OckDataBuffer> &OutputDstPos() override
    {
        return outputParams[0U];
    }
};

std::shared_ptr<OckRemoveDataShapedOpDataBuffer> OckRemoveDataShapedOpDataBuffer::Create(
    const OckRemoveDataShapedOpMeta &opSpec)
{
    return std::make_shared<OckRemoveDataShapedOpDataBufferImpl>(opSpec);
}
} // namespace nop
} // namespace hcps
} // namespace ock