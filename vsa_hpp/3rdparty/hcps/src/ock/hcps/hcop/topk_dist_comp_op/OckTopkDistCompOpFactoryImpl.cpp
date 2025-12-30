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

#include "ock/hcps/hcop/topk_dist_comp_op/OckTopkDistCompOpFactory.h"
#include "ock/hcps/handler/OckHandlerHmmHelper.h"
namespace ock {
namespace hcps {
namespace hcop {
class OckTopkDistCompOpFactoryImpl : public OckTopkDistCompOpFactory {
public:
    virtual ~OckTopkDistCompOpFactoryImpl() noexcept = default;
    explicit OckTopkDistCompOpFactoryImpl(const OckTopkDistCompOpMeta &opSpec)
    {
        topkOpFactory = nop::OckTopkFlatOpFactory::Create(opSpec.ToTopkOpMeta());
        distOpFactory = nop::OckDistInt8CosMaxOpFactory::Create(opSpec.ToDistOpMeta());
    }
    hmm::OckHmmErrorCode AllocTmpSpace(const OckTopkDistCompOpMeta &opSpec,
        const std::shared_ptr<nop::OckTopkFlatOpDataBuffer> &topkBuffer,
        const std::vector<std::shared_ptr<nop::OckDistInt8CosMaxOpDataBuffer>> &distBuffers,
        std::shared_ptr<handler::OckHeteroHandler> handler) override
    {
        hmm::OckHmmErrorCode errorCode = hmm::HMM_SUCCESS;
        int64_t maskMinBytes = sizeof(uint8_t) * opSpec.batch * utils::SafeDiv(opSpec.codeBlockSize, 8LL);
        int64_t tmpSpaceBytes = CalcTmpSpace(maskMinBytes, topkBuffer);
        if (tmpSpaces.empty() ||
            static_cast<uint64_t>(tmpSpaceBytes) > tmpSpaces[tmpSpaces.size() - 1]->GetByteSize()) {
            auto hmo = handler::helper::MakeDeviceHmo(*handler, tmpSpaceBytes, errorCode);
            OCK_CHECK_RETURN_ERRORCODE(errorCode);
            tmpSpaces.push_back(hmo);
        }

        auto hmoHolder = tmpSpaces[tmpSpaces.size() - 1];
        int64_t tmpOffset = 0;
        topkBuffer->InputDists() = std::make_shared<nop::OckDataBuffer>(hmoHolder->Addr() + tmpOffset,
            topkBuffer->GetParamsByteSizes()[0U], hmoHolder, topkBuffer->GetParamsShapes()[0U]);
        tmpOffset += topkBuffer->GetParamsByteSizes()[0U];
        topkBuffer->InputMinDists() = std::make_shared<nop::OckDataBuffer>(hmoHolder->Addr() + tmpOffset,
            topkBuffer->GetParamsByteSizes()[1U], hmoHolder, topkBuffer->GetParamsShapes()[1U]);
        tmpOffset += topkBuffer->GetParamsByteSizes()[1U];
        topkBuffer->InputSizes() = std::make_shared<nop::OckDataBuffer>(hmoHolder->Addr() + tmpOffset,
            topkBuffer->GetParamsByteSizes()[2U], hmoHolder, topkBuffer->GetParamsShapes()[2U]);
        tmpOffset += topkBuffer->GetParamsByteSizes()[2U];
        topkBuffer->InputFlags() = std::make_shared<nop::OckDataBuffer>(hmoHolder->Addr() + tmpOffset,
            topkBuffer->GetParamsByteSizes()[3U], hmoHolder, topkBuffer->GetParamsShapes()[3U]);
        tmpOffset += topkBuffer->GetParamsByteSizes()[3U];
        topkBuffer->InputAttrs() = std::make_shared<nop::OckDataBuffer>(hmoHolder->Addr() + tmpOffset,
            topkBuffer->GetParamsByteSizes()[4U], hmoHolder, topkBuffer->GetParamsShapes()[4U]);
        tmpOffset += topkBuffer->GetParamsByteSizes()[4U];

        // 虽然mask长度由buffer meta决定，但是mask临时空间仅对无mask场景有用，所以可以用op meta中的单block的mask
        auto tmpMask = std::make_shared<nop::OckDataBuffer>(hmoHolder->Addr() + tmpOffset, maskMinBytes, hmoHolder);
        for (auto distBuffer : distBuffers) {
            if (distBuffer->InputMask() == nullptr) {
                distBuffer->InputMask() = tmpMask;
            }
        }
        return errorCode;
    }
    std::shared_ptr<OckHeteroOperatorBase> Create(const OckTopkDistCompOpMeta &opSpec,
        const OckTopkDistCompBufferMeta &bufferSpec, const std::shared_ptr<nop::OckTopkFlatOpDataBuffer> &topkBuffer,
        const std::vector<std::shared_ptr<nop::OckDistInt8CosMaxOpDataBuffer>> &distBuffers,
        std::shared_ptr<handler::OckHeteroHandler> handler) override
    {
        return OckTopkDistCompOp::Create(GetNumDistOps(opSpec, bufferSpec), topkOpFactory, distOpFactory, topkBuffer,
            distBuffers, handler);
    };

private:
    int64_t CalcTmpSpace(int64_t maskMinBytes, const std::shared_ptr<nop::OckTopkFlatOpDataBuffer> &topkBuffer)
    {
        if (topkBuffer == nullptr) {
            return 0;
        }
        return maskMinBytes + topkBuffer->GetParamsByteSizes()[0U] + /* InputDists */
            topkBuffer->GetParamsByteSizes()[1U] +                   /* InputMinDists */
            topkBuffer->GetParamsByteSizes()[2U] +                   /* InputSizes */
            topkBuffer->GetParamsByteSizes()[3U] +                   /* InputFlags */
            topkBuffer->GetParamsByteSizes()[4U];                    /* InputAttrs */
    }
    std::shared_ptr<nop::OckTopkFlatOpFactory> topkOpFactory{ nullptr };
    std::shared_ptr<nop::OckDistInt8CosMaxOpFactory> distOpFactory{ nullptr };
    std::vector<std::shared_ptr<hmm::OckHmmHMObject>> tmpSpaces{};
};

std::shared_ptr<OckTopkDistCompOpFactory> OckTopkDistCompOpFactory::Create(const OckTopkDistCompOpMeta &opSpec)
{
    return std::make_shared<OckTopkDistCompOpFactoryImpl>(opSpec);
}
} // namespace hcop
} // namespace hcps
} // namespace ock