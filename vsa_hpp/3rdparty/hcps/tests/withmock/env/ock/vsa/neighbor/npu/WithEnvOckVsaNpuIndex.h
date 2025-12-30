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

#include <memory>
#include <gtest/gtest.h>
#include "ock/vsa/neighbor/npu/OckVsaAnnNpuIndex.h"
#include "mockcpp/mockcpp.hpp"
#include "ock/hcps/WithEnvOckHeteroHandler.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace npu {
static hcps::OckHcpsErrorCode FakeNormOpRun(std::shared_ptr<hcps::nop::OckL2NormOpHmoBlock> hmoBlock,
    hcps::handler::OckHeteroHandler &handler, std::shared_ptr<hcps::OckHeteroStreamBase> streamBase)
{
    return hmm::HMM_SUCCESS;
}
static void FakeShapeOpRun(std::shared_ptr<hcps::nop::OckTransDataShapedOpHmoBlock> hmoBlock,
    hcps::handler::OckHeteroHandler &handler, std::shared_ptr<hcps::OckHeteroStreamBase> streamBase)
{
    return;
}
static void FakeCustomOpRun(std::shared_ptr<hcps::nop::OckTransDataCustomAttrOpHmoBlock> hmoBlock,
    hcps::handler::OckHeteroHandler &handler, std::shared_ptr<hcps::OckHeteroStreamBase> streamBase)
{
    return;
}
template <typename _BaseT> class WithEnvOckVsaNpuIndex : public hcps::handler::WithEnvOckHeteroHandler<_BaseT> {
public:
    using BaseT = hcps::handler::WithEnvOckHeteroHandler<_BaseT>;
    using OckVsaAnnNpuIndexT = OckVsaAnnNpuIndex<int8_t, 256ULL, 2ULL, attr::OckTimeSpaceAttrTrait>;
    void SetUp(void) override
    {
        BaseT::SetUp();
        this->hmmSpec.devSpec.maxDataCapacity = 10UL * 1024UL * 1024UL * 1024UL;
        this->hmmSpec.devSpec.maxSwapCapacity = 8UL * 64UL * 1024UL * 1024UL;
        this->hmmSpec.hostSpec.maxDataCapacity = 2UL * 1024UL * 1024UL * 1024UL;
        this->hmmSpec.hostSpec.maxSwapCapacity = 8UL * 64UL * 1024UL * 1024UL;
        MOCKER(hcps::nop::OckL2NormOpRun::ComputeNormSync).stubs().will(invoke(FakeNormOpRun));
        MOCKER(hcps::nop::OckTransDataShapedOpRun::AddTransShapedOp).stubs().will(invoke(FakeShapeOpRun));
        MOCKER(hcps::nop::OckTransDataCustomAttrOpRun::AddTransCustomAttrOp).stubs().will(invoke(FakeCustomOpRun));
    }
    void TearDown(void) override
    {
        npuIndex.reset();
        handler.reset();
        BaseT::TearDown();
    }
    void InitOckVsaAnnCreateParam(void)
    {
        param = OckVsaAnnCreateParam::Create(this->cpuSet, this->deviceId, maxFeatureRowCount, tokenNum,
            extKeyAttrsByteSize, extKeyAttrBlockSize);
    }
    void InitOckVsaHPPKernelExt(void)
    {
        InitOckVsaAnnCreateParam();
        handler = this->CreateSingleDeviceHandler(errorCode);
        npuIndex = std::make_shared<OckVsaAnnNpuIndexT>(handler, dftTrait, param);
    }
    OckVsaErrorCode errorCode{ hmm::HMM_SUCCESS };
    uint64_t maxFeatureRowCount{ 16777216ULL * 3ULL };
    uint32_t extKeyAttrsByteSize{ 10 };
    uint32_t extKeyAttrBlockSize{ 262144 };
    uint32_t tokenNum{ 2500 };
    attr::OckTimeSpaceAttrTrait dftTrait{ tokenNum };
    std::shared_ptr<hcps::handler::OckHeteroHandler> handler;
    std::shared_ptr<OckVsaAnnCreateParam> param;
    std::shared_ptr<OckVsaAnnNpuIndexT> npuIndex;
};
} // namespace npu
} // namespace neighbor
} // namespace vsa
} // namespace ock