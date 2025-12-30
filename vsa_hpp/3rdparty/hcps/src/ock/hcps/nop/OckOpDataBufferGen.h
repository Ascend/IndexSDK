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
#include "ock/hcps/nop/OckOpDataBuffer.h"

namespace ock {
namespace hcps {
namespace nop {
template <class base_T> class OckOpDataBufferGen : public base_T {
public:
    virtual ~OckOpDataBufferGen() noexcept = default;
    OckHcpsErrorCode AllocBuffers(const std::shared_ptr<hmm::OckHmmHeteroMemoryMgrBase> &devMgr) override
    {
        OCK_CHECK_RETURN_ERRORCODE(AllocInputBuffers(devMgr));
        OCK_CHECK_RETURN_ERRORCODE(AllocOutputBuffers(devMgr));
        return hmm::HMM_SUCCESS;
    }
    OckHcpsErrorCode AllocInputBuffers(const std::shared_ptr<hmm::OckHmmHeteroMemoryMgrBase> &devMgr) override
    {
        for (uint16_t i = 0; i < numInputs; ++i) {
            OCK_CHECK_RETURN_ERRORCODE(AllocBuffer(devMgr, inputParams[i], static_cast<uint32_t>(paramsByteSizes[i]),
                paramsShapes[i]));
        }
        return hmm::HMM_SUCCESS;
    }
    OckHcpsErrorCode AllocOutputBuffers(const std::shared_ptr<hmm::OckHmmHeteroMemoryMgrBase> &devMgr) override
    {
        for (uint16_t i = 0; i < numOutputs; ++i) {
            OCK_CHECK_RETURN_ERRORCODE(
                AllocBuffer(devMgr, outputParams[i], static_cast<uint32_t>(paramsByteSizes[numInputs + i]),
                    paramsShapes[numInputs + i]));
        }
        return hmm::HMM_SUCCESS;
    }
    std::vector<std::shared_ptr<OckDataBuffer>> &GetInputParams() override
    {
        return inputParams;
    }
    std::vector<std::shared_ptr<OckDataBuffer>> &GetOutputParams() override
    {
        return outputParams;
    }
    std::vector<int64_t> &GetParamsByteSizes() override
    {
        return paramsByteSizes;
    }
    std::vector<std::vector<int64_t>> &GetParamsShapes() override
    {
        return paramsShapes;
    }

protected:
    template <typename T> void AddParamInfo(const std::vector<int64_t> &shape)
    {
        paramsByteSizes.push_back(static_cast<int64_t>(sizeof(T) *
            std::accumulate(shape.cbegin(), shape.cend(), 1LL, std::multiplies<int64_t>())));
        paramsShapes.push_back(shape);
    }
    static OckHcpsErrorCode AllocBuffer(const std::shared_ptr<hmm::OckHmmHeteroMemoryMgrBase> &devMgr,
        std::shared_ptr<OckDataBuffer> &paramBuffer, uint32_t bytes, const std::vector<int64_t> &shape = {})
    {
        auto hmo = devMgr->Alloc(bytes, hmm::OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
        OCK_CHECK_RETURN_ERRORCODE(hmo.first);
        paramBuffer = std::make_shared<OckDataBuffer>(hmo.second, shape);
        return hmm::HMM_SUCCESS;
    }
    uint16_t numInputs{ 0 };
    uint16_t numOutputs{ 0 };
    std::vector<int64_t> paramsByteSizes{};
    std::vector<std::vector<int64_t>> paramsShapes{};
    std::vector<std::shared_ptr<OckDataBuffer>> inputParams{};
    std::vector<std::shared_ptr<OckDataBuffer>> outputParams{};
};
} // namespace nop
} // namespace hcps
} // namespace ock