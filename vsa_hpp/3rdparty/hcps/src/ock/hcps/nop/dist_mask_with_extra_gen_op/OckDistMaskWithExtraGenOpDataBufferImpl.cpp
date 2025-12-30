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

#include "ock/hcps/nop/OckOpDataBufferGen.h"
#include "ock/hcps/nop/dist_mask_with_extra_gen_op/OckDistMaskWithExtraGenOpDataBuffer.h"

namespace ock {
namespace hcps {
namespace nop {
class OckDistMaskWithExtraGenOpDataBufferImpl : public OckOpDataBufferGen<OckDistMaskWithExtraGenOpDataBuffer> {
public:
    virtual ~OckDistMaskWithExtraGenOpDataBufferImpl() noexcept = default;
    explicit OckDistMaskWithExtraGenOpDataBufferImpl(const OckDistMaskWithExtraGenOpMeta &opSpec)
    {
        numInputs = 6U;
        numOutputs = 1U;
        AddParamInfo<int32_t>({ OPS_DATA_TYPE_ALIGN });
        AddParamInfo<uint8_t>({
            utils::SafeDivUp(static_cast<int64_t>(opSpec.tokenNum), OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES
        });
        AddParamInfo<int32_t>({ static_cast<int64_t>(opSpec.featureAttrBlockSize * opSpec.blockCount) });
        AddParamInfo<int32_t>({ static_cast<int64_t>(opSpec.featureAttrBlockSize * opSpec.blockCount) });
        AddParamInfo<uint8_t>({
            static_cast<int64_t>(opSpec.featureAttrBlockSize * opSpec.blockCount) * OPS_DATA_TYPE_TIMES
        });
        AddParamInfo<uint8_t>({
            utils::SafeDivUp(static_cast<int64_t>(opSpec.featureAttrBlockSize * opSpec.blockCount), OPS_DATA_TYPE_ALIGN)
        });
        AddParamInfo<uint8_t>({
            utils::SafeDivUp(static_cast<int64_t>(opSpec.featureAttrBlockSize * opSpec.blockCount), OPS_DATA_TYPE_ALIGN)
        });
        inputParams.resize(numInputs);
        outputParams.resize(numOutputs);
    }
    std::shared_ptr<OckDataBuffer> &InputQueryTime() override
    {
        return inputParams[0U];
    }
    std::shared_ptr<OckDataBuffer> &InputTokenBitSet() override
    {
        return inputParams[1U];
    }
    std::shared_ptr<OckDataBuffer> &InputAttrTimes() override
    {
        return inputParams[2U];
    }
    std::shared_ptr<OckDataBuffer> &InputAttrTokenQs() override
    {
        return inputParams[3U];
    }
    std::shared_ptr<OckDataBuffer> &InputAttrTokenRs() override
    {
        return inputParams[4U];
    }
    std::shared_ptr<OckDataBuffer> &InputExtraMask() override
    {
        return inputParams[5U];
    }

    std::shared_ptr<OckDataBuffer> &OutputMask() override
    {
        return outputParams[0U];
    }

    OckHcpsErrorCode AllocBuffersFromHmoGroup(std::shared_ptr<OckDistMaskWithExtraGenOpHmoGroup> hmoGroup,
        uint64_t offsetInSingleMask, uint64_t groupMaskLen) override
    {
        auto queryTimesBuffer = hmoGroup->queryTimes->GetBuffer(hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR, 0U,
            hmoGroup->queryTimes->GetByteSize());
        if (queryTimesBuffer.get() == nullptr) {
            OCK_HCPS_LOG_ERROR("hmo block queryTimes get buffer failed!");
            return HCPS_ERROR_GET_BUFFER_FAILED;
        }
        inputParams[0U] = std::make_shared<OckDataBuffer>(queryTimesBuffer, paramsShapes[0U]);
        auto queryTokenIdsBuffer = hmoGroup->queryTokenIds->GetBuffer(hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR, 0U,
            hmoGroup->queryTokenIds->GetByteSize());
        if (queryTokenIdsBuffer.get() == nullptr) {
            OCK_HCPS_LOG_ERROR("hmo block queryTokenIds get buffer failed!");
            return HCPS_ERROR_GET_BUFFER_FAILED;
        }
        inputParams[1U] = std::make_shared<OckDataBuffer>(queryTokenIdsBuffer, paramsShapes[1U]);
        auto attrTimesBuffer = hmoGroup->attrTimes->GetBuffer(hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR, 0U,
            hmoGroup->attrTimes->GetByteSize());
        if (attrTimesBuffer.get() == nullptr) {
            OCK_HCPS_LOG_ERROR("hmo block attrTimes get buffer failed!");
            return HCPS_ERROR_GET_BUFFER_FAILED;
        }
        inputParams[2U] = std::make_shared<OckDataBuffer>(attrTimesBuffer, paramsShapes[2U]);
        auto attrTokenQSsBuffer = hmoGroup->attrTokenQuotients->GetBuffer(hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR,
            0U, hmoGroup->attrTokenQuotients->GetByteSize());
        if (attrTokenQSsBuffer.get() == nullptr) {
            OCK_HCPS_LOG_ERROR("hmo block attrTokenQuotients get buffer failed!");
            return HCPS_ERROR_GET_BUFFER_FAILED;
        }
        inputParams[3U] = std::make_shared<OckDataBuffer>(attrTokenQSsBuffer, paramsShapes[3U]);
        auto attrTokenRSBuffer = hmoGroup->attrTokenRemainders->GetBuffer(hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR,
            0U, hmoGroup->attrTokenRemainders->GetByteSize());
        if (attrTokenRSBuffer.get() == nullptr) {
            OCK_HCPS_LOG_ERROR("hmo block attrTokenRemainders get buffer failed!");
            return HCPS_ERROR_GET_BUFFER_FAILED;
        }
        inputParams[4U] = std::make_shared<OckDataBuffer>(attrTokenRSBuffer, paramsShapes[4U]);
        auto extraMaskBuffer = hmoGroup->extraMask->GetBuffer(hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR,
            offsetInSingleMask * sizeof(uint8_t), groupMaskLen * sizeof(uint8_t));
        if (extraMaskBuffer.get() == nullptr) {
            OCK_HCPS_LOG_ERROR("hmo block extra mask get buffer failed!");
            return HCPS_ERROR_GET_BUFFER_FAILED;
        }
        inputParams[5U] = std::make_shared<OckDataBuffer>(extraMaskBuffer, paramsShapes[5U]);
        auto maskBuffer = hmoGroup->mask->GetBuffer(hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR,
            offsetInSingleMask * sizeof(uint8_t), groupMaskLen * sizeof(uint8_t));
        if (maskBuffer.get() == nullptr) {
            OCK_HCPS_LOG_ERROR("hmo block mask get buffer failed!");
            return HCPS_ERROR_GET_BUFFER_FAILED;
        }
        outputParams[0U] = std::make_shared<OckDataBuffer>(maskBuffer, paramsShapes[6U]);
        return hmm::HMM_SUCCESS;
    }
};

std::shared_ptr<OckDistMaskWithExtraGenOpDataBuffer> OckDistMaskWithExtraGenOpDataBuffer::Create(
    const OckDistMaskWithExtraGenOpMeta &opSpec)
{
    return std::make_shared<OckDistMaskWithExtraGenOpDataBufferImpl>(opSpec);
}
} // namespace nop
} // namespace hcps
} // namespace ock
