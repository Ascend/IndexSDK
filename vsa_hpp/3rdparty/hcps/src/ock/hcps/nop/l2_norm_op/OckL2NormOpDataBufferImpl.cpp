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
#include "ock/hcps/nop/l2_norm_op/OckL2NormOpDataBuffer.h"

namespace ock {
namespace hcps {
namespace nop {
class OckL2NormOpDataBufferImpl : public OckOpDataBufferGen<OckL2NormOpDataBuffer> {
public:
    virtual ~OckL2NormOpDataBufferImpl() noexcept = default;
    OckL2NormOpDataBufferImpl(const OckL2NormOpMeta &opSpec, const OckL2NormBufferMeta &bufferSpec)
    {
        numInputs = 3U;
        numOutputs = 1U;
        AddParamInfo<int8_t>({ static_cast<int64_t>(bufferSpec.ntotal), static_cast<int64_t>(opSpec.dims) });
        AddParamInfo<OckFloat16>({ TRANSFER_SIZE, CUBE_ALIGN });
        AddParamInfo<uint32_t>({ SIZE_ALIGN });
        AddParamInfo<OckFloat16>({ static_cast<int64_t>(bufferSpec.ntotal) });
        inputParams.resize(numInputs);
        outputParams.resize(numOutputs);
    }
    OckHcpsErrorCode AllocBuffersFromHmoBlock(std::shared_ptr<OckL2NormOpHmoBlock> hmoBlock,
        std::shared_ptr<hmm::OckHmmHeteroMemoryMgrBase> devMgr, int64_t offset) override
    {
        auto dataBuffer = hmoBlock->dataBase->GetBuffer(hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR,
            offset * hmoBlock->dims * sizeof(int8_t), L2NORM_COMPUTE_BATCH * hmoBlock->dims * sizeof(int8_t));
        if (dataBuffer.get() == nullptr) {
            OCK_HCPS_LOG_ERROR("hmo block data base get buffer failed!");
            return HCPS_ERROR_GET_BUFFER_FAILED;
        }
        inputParams[0U] = std::make_shared<OckDataBuffer>(dataBuffer, paramsShapes[0U]);

        auto transferRet = devMgr->Alloc(paramsByteSizes[1U], hmm::OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
        if (transferRet.first != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("transfer hmo alloc failed, the errorCode is " << transferRet.first);
            return transferRet.first;
        }
        inputParams[1U] = std::make_shared<OckDataBuffer>(transferRet.second, paramsShapes[1U]);

        auto actualNumRet = devMgr->Alloc(paramsByteSizes[2U], hmm::OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
        if (actualNumRet.first != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("actualNum hmo alloc failed, the errorCode is " << actualNumRet.first);
            return actualNumRet.first;
        }
        inputParams[2U] = std::make_shared<OckDataBuffer>(actualNumRet.second, paramsShapes[2U]);

        auto resultBuffer = hmoBlock->normResult->GetBuffer(hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR,
            offset * sizeof(OckFloat16), L2NORM_COMPUTE_BATCH * sizeof(OckFloat16));
        if (resultBuffer.get() == nullptr) {
            OCK_HCPS_LOG_ERROR("hmo block norm result get buffer failed!");
            return HCPS_ERROR_GET_BUFFER_FAILED;
        }
        outputParams[0U] = std::make_shared<OckDataBuffer>(resultBuffer, paramsShapes[3U]);

        return hmm::HMM_SUCCESS;
    }
    std::shared_ptr<OckDataBuffer> &InputVectors() override
    {
        return inputParams[0U];
    }
    std::shared_ptr<OckDataBuffer> &InputTransfer() override
    {
        return inputParams[1U];
    }
    std::shared_ptr<OckDataBuffer> &InputActualNum() override
    {
        return inputParams[2U];
    }
    
    std::shared_ptr<OckDataBuffer> &OutputResult() override
    {
        return outputParams[0U];
    }
};

std::shared_ptr<OckL2NormOpDataBuffer> OckL2NormOpDataBuffer::Create(
    const OckL2NormOpMeta &opSpec, const OckL2NormBufferMeta &bufferSpec)
{
    return std::make_shared<OckL2NormOpDataBufferImpl>(opSpec, bufferSpec);
}
} // namespace nop
} // namespace hcps
} // namespace ock