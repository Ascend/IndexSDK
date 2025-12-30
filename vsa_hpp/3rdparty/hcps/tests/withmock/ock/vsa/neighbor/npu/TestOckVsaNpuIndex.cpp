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

#include <gtest/gtest.h>
#include "ock/vsa/neighbor/npu/OckVsaAnnNpuIndex.h"
#include "ock/vsa/neighbor/npu/WithEnvOckVsaNpuIndex.h"
#include "ock/vsa/neighbor/base/WithEnvOckFeatureBuild.h"
#include "ock/vsa/neighbor/base/OckVsaAnnFactory.h"
#include "ock/vsa/neighbor/npu/impl/OckVsaAnnIndexCapacity.h"
#include "ock/hcps/handler/OckHeteroHandler.h"
#include "ock/hmm/mgr/OckHmmMemorySpecification.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace npu {
namespace test {
class TestOckVsaNpuIndex : public WithEnvOckVsaNpuIndex<WithEnvOckFeatureBuild<testing::Test>> {
public:
    using BaseT = WithEnvOckVsaNpuIndex<testing::Test>;
    using RegisterT = OckVsaAnnIndexFactoryRegister<int8_t, 256ULL, 2ULL, attr::OckTimeSpaceAttrTrait>;
};
TEST_F(TestOckVsaNpuIndex, createNpuIndex)
{
    this->InitOckVsaAnnCreateParam();
    OckVsaErrorCode errorCode = VSA_SUCCESS;
    const uint64_t blockByteSize = this->param->BlockRowCount() * sizeof(int8_t) * 256ULL;
    hmm::OckHmmMemorySpecification spec;
    spec.devSpec.maxDataCapacity = OckVsaAnnIndexCapacity::DeviceSpace(*(this->param), sizeof(int8_t) * 256ULL, 2ULL);
    spec.devSpec.maxSwapCapacity = blockByteSize * 4ULL; // 4 * 64MB
    spec.hostSpec.maxDataCapacity = OckVsaAnnIndexCapacity::HostSpace(*(this->param), sizeof(int8_t) * 256ULL, 2ULL);
    spec.hostSpec.maxSwapCapacity = spec.devSpec.maxSwapCapacity;

    auto handler =
        hcps::handler::OckHeteroHandler::CreateSingleDeviceHandler(param->DeviceId(), param->CpuSet(), spec, errorCode);
    EXPECT_EQ(errorCode, VSA_SUCCESS);
    auto index =
            std::make_shared<ock::vsa::neighbor::npu::OckVsaAnnNpuIndex<int8_t, 256ULL, 2ULL,
            attr::OckTimeSpaceAttrTrait>>(handler, this->dftTrait, this->param);
    ASSERT_TRUE(index.get() != nullptr);
}
TEST_F(TestOckVsaNpuIndex, addFeature)
{
    this->InitOckVsaAnnCreateParam();
    OckVsaErrorCode errorCode = VSA_SUCCESS;
    const uint64_t blockByteSize = this->param->BlockRowCount() * sizeof(int8_t) * 256ULL;
    hmm::OckHmmMemorySpecification spec;
    spec.devSpec.maxDataCapacity = OckVsaAnnIndexCapacity::DeviceSpace(*(this->param), sizeof(int8_t) * 256ULL, 2ULL);
    spec.devSpec.maxSwapCapacity = blockByteSize * 4ULL; // 4 * 64MB
    spec.hostSpec.maxDataCapacity = OckVsaAnnIndexCapacity::HostSpace(*(this->param), sizeof(int8_t) * 256ULL, 2ULL);
    spec.hostSpec.maxSwapCapacity = spec.devSpec.maxSwapCapacity;

    auto handler =
        hcps::handler::OckHeteroHandler::CreateSingleDeviceHandler(param->DeviceId(), param->CpuSet(), spec, errorCode);
    EXPECT_EQ(errorCode, VSA_SUCCESS);
    auto index =
            std::make_shared<ock::vsa::neighbor::npu::OckVsaAnnNpuIndex<int8_t, 256ULL, 2ULL,
            attr::OckTimeSpaceAttrTrait>> (handler, this->dftTrait, this->param);
    ASSERT_TRUE(index.get() != nullptr);
    auto featureParam = this->BuildFeature(this->param->BlockRowCount(), this->param->ExtKeyAttrByteSize());
    EXPECT_EQ(index->AddFeature(featureParam), hmm::HMM_SUCCESS);
}
} // namespace test
} // namespace npu
} // namespace neighbor
} // namespace vsa
} // namespace ock