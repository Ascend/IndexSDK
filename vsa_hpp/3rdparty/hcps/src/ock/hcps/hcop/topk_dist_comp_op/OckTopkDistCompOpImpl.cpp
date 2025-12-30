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
#include "ock/hcps/hcop/topk_dist_comp_op/OckTopkDistCompOp.h"
namespace ock {
namespace hcps {
namespace hcop {
class OckTopkDistCompOpImpl : public OckTopkDistCompOp {
public:
    virtual ~OckTopkDistCompOpImpl() noexcept = default;
    explicit OckTopkDistCompOpImpl(int64_t numDistanceOps,
        const std::shared_ptr<nop::OckTopkFlatOpFactory> &topkOpFactory,
        const std::shared_ptr<nop::OckDistInt8CosMaxOpFactory> &distOpFactory,
        const std::shared_ptr<nop::OckTopkFlatOpDataBuffer> &topkBuffer,
        const std::vector<std::shared_ptr<nop::OckDistInt8CosMaxOpDataBuffer>> &distBuffers,
        std::shared_ptr<handler::OckHeteroHandler> heteroHandler)
        : numDistOps(numDistanceOps), handler(std::move(heteroHandler))
    {
        topkOp = topkOpFactory->Create(topkBuffer);
        for (int64_t i = 0; i < numDistOps; ++i) {
            distOpList.push_back(distOpFactory->Create(distBuffers[i]));
        }
    }

    acladapter::OckTaskResourceType ResourceType() const override
    {
        return acladapter::OckTaskResourceType::OP_TASK;
    };

    hmm::OckHmmErrorCode Run(OckHeteroStreamContext &context) override
    {
        auto ret = topkOp->Run(context);
        if (ret != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("start running topk op fail!");
        }
        OCK_HCPS_LOG_DEBUG("start running topk op");

        auto distStream = handler::helper::MakeStream(*handler, ret, OckDevStreamType::AI_CORE);
        if (ret != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("Fail to Create Stream for Dist Op!");
        }
        for (int64_t i = 0; i < numDistOps; ++i) {
            distStream->AddOp(distOpList[i]);
            OCK_HCPS_LOG_DEBUG("start running dist op [" << i << "]");
        }
        ret = distStream->WaitExecComplete();
        if (ret != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("dist WaitExecComplete fail! ret :" << ret);
        }
        return ret;
    }

private:
    int64_t numDistOps{ 0 };
    std::shared_ptr<OckTopkDistCompOpDataBuffer> dataBuffer{ nullptr };
    std::shared_ptr<OckHeteroOperatorBase> topkOp{ nullptr };
    std::vector<std::shared_ptr<OckHeteroOperatorBase>> distOpList{};
    std::shared_ptr<handler::OckHeteroHandler> handler{ nullptr };
};

std::shared_ptr<OckHeteroOperatorBase> OckTopkDistCompOp::Create(int64_t numDistOps,
    const std::shared_ptr<nop::OckTopkFlatOpFactory> &topkOpFactory,
    const std::shared_ptr<nop::OckDistInt8CosMaxOpFactory> &distOpFactory,
    const std::shared_ptr<nop::OckTopkFlatOpDataBuffer> &topkBuffer,
    const std::vector<std::shared_ptr<nop::OckDistInt8CosMaxOpDataBuffer>> &distBuffers,
    std::shared_ptr<handler::OckHeteroHandler> handler)
{
    return std::make_shared<OckTopkDistCompOpImpl>(numDistOps, topkOpFactory, distOpFactory, topkBuffer, distBuffers,
        std::move(handler));
}
} // namespace hcop
} // namespace hcps
} // namespace ock