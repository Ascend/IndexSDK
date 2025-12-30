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
#include "ock/vsa/neighbor/hpp/kernel/OckVsaHPPKernelExt.h"
#include "ock/hcps/WithEnvOckHeteroHandler.h"
#include "ock/vsa/neighbor/base/OckVsaAnnFactory.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
template <typename _BaseT>
class WithEnvOckVsaHPPIndex : public hcps::handler::WithEnvOckHeteroHandler<_BaseT> {
public:
    using BaseT = hcps::handler::WithEnvOckHeteroHandler<_BaseT>;
    using RegisterT = OckVsaAnnIndexFactoryRegister<int8_t, 256ULL, 2ULL, attr::OckTimeSpaceAttrTrait>;
    using OckVsaAnnIndexBaseT = OckVsaAnnIndexBase<int8_t, 256ULL, 2ULL, attr::OckTimeSpaceAttrTrait>;
    using OckVsaAnnHPPIndexT = OckVsaAnnHPPIndex<int8_t, 256ULL, 2ULL, attr::OckTimeSpaceAttrTrait>;
    using OckVsaHPPKernelExtT = OckVsaHPPKernelExt<int8_t, 256ULL, 2ULL, attr::OckTimeSpaceAttrTrait, 16ULL>;
    void SetUp(void) override
    {
        BaseT::SetUp();
        this->hmmSpec.devSpec.maxDataCapacity = 10UL * 1024UL * 1024UL * 1024UL;
        this->hmmSpec.devSpec.maxSwapCapacity = 8UL * 64UL * 1024UL * 1024UL;
        this->hmmSpec.hostSpec.maxDataCapacity = 40UL * 1024UL * 1024UL * 1024UL;
        this->hmmSpec.hostSpec.maxSwapCapacity = 8UL * 64UL * 1024UL * 1024UL;
    }
    void TearDown(void) override
    {
        handler.reset();
        hppKernel.reset();
        index.reset();
        BaseT::TearDown();
    }
    void InitOckVsaAnnCreateParam(void)
    {
        param = OckVsaAnnCreateParam::Create(this->cpuSet,
            this->deviceId,
            maxFeatureRowCount,
            tokenNum,
            extKeyAttrsByteSize,
            extKeyAttrsBlockSize);
    }
    void InitIndex(void)
    {
        InitOckVsaAnnCreateParam();
        index = RegisterT::Instance().GetFactory("HPPTS")->Create(this->param, this->dftTrait, this->errorCode);
        std::cout << "test 1" << std::endl;
        hppIndex = dynamic_cast<OckVsaAnnHPPIndexT *>(index.get());
        hppKernel = hppIndex->Kernel();
        handler = hppIndex->Handler();
    }
    OckVsaErrorCode errorCode{hmm::HMM_SUCCESS};
    uint64_t maxFeatureRowCount{16777216ULL * 14ULL};
    uint32_t extKeyAttrsByteSize{10};
    uint32_t extKeyAttrsBlockSize{262144};
    uint32_t tokenNum{2500};
    uint32_t blockRowCount{262144UL};
    uint32_t groupBlockCount{64UL};
    uint32_t sliceRowCount{256UL};
    attr::OckTimeSpaceAttrTrait dftTrait{tokenNum};
    std::shared_ptr<OckVsaAnnCreateParam> param;
    std::shared_ptr<OckVsaAnnIndexBaseT> index;
    OckVsaAnnHPPIndexT *hppIndex;
    std::shared_ptr<OckVsaHPPKernelExtT> hppKernel;
    std::shared_ptr<hcps::handler::OckHeteroHandler> handler;
};
}  // namespace hpp
}  // namespace neighbor
}  // namespace vsa
}  // namespace ock