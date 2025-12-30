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
#include <vector>
#include <gtest/gtest.h>
#include "mockcpp/mockcpp.hpp"
#include "ock/hmm/OckHmmFactory.h"
#include "ock/acladapter/WithEnvAclMock.h"
#include "ock/hcps/hcop/topk_dist_comp_op/OckTopkDistCompOpDataBuffer.h"
namespace ock {
namespace hcps {
namespace hcop {
namespace test {
class TestOckTopkDistCompOpDataBuffer : public acladapter::WithEnvAclMock<testing::Test> {
public:
    void SetUp(void) override
    {
        acladapter::WithEnvAclMock<testing::Test>::SetUp();
        deviceInfo = std::make_shared<hmm::OckHmmDeviceInfo>();
        deviceInfo->deviceId = 0U;
        CPU_SET(1U, &deviceInfo->cpuSet);                                                   // 设置1号CPU核
        CPU_SET(2U, &deviceInfo->cpuSet);                                                   // 设置2号CPU核
        deviceInfo->memorySpec.devSpec.maxDataCapacity = 1024ULL * 1024ULL * 1024ULL;       // 1G
        deviceInfo->memorySpec.devSpec.maxSwapCapacity = 3ULL * 64ULL * 1024ULL * 1024ULL;  // 3 * 64M
        deviceInfo->memorySpec.hostSpec.maxDataCapacity = 1024ULL * 1024ULL * 1024ULL;      // 1G
        deviceInfo->memorySpec.hostSpec.maxSwapCapacity = 3ULL * 64ULL * 1024ULL * 1024ULL; // 3 * 64M
        deviceInfo->transferThreadNum = 2ULL;                                               // 2个线程
    }
    void TearDown(void) override
    {
        DestroyHmmDeviceMgr(); // 需要提前reset，否则打桩不生效。
        aclrtResetDevice(0);
        acladapter::WithEnvAclMock<testing::Test>::TearDown();
    }
    void BuildHmmDeviceMgr(void)
    {
        auto factory = hmm::OckHmmFactory::Create();
        auto ret = factory->CreateSingleDeviceMemoryMgr(deviceInfo);
        ASSERT_EQ(ret.first, hmm::HMM_SUCCESS);
        devMgr = ret.second;
    }
    void DestroyHmmDeviceMgr(void)
    {
        if (devMgr.get() != nullptr) {
            devMgr.reset();
        }
    }
    std::shared_ptr<hmm::OckHmmHMObject> AllocHmo(int64_t byteSize)
    {
        auto hmoRet = devMgr->Alloc(byteSize, hmm::OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
        if (hmoRet.first != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("Fail to Alloc Hmo! The error code is " << hmoRet.first);
            return nullptr;
        }
        return hmoRet.second;
    }
    std::shared_ptr<hmm::OckHmmDeviceInfo> deviceInfo;
    std::shared_ptr<hmm::OckHmmSingleDeviceMgr> devMgr;

    OckTopkDistCompOpMeta opSpec;
    OckTopkDistCompBufferMeta bufferSpec;
};

TEST_F(TestOckTopkDistCompOpDataBuffer, create_buffer_from_hmo)
{
    BuildHmmDeviceMgr();
    auto hmoGroup = std::make_shared<OckTopkDistCompOpHmoGroup>();
    hmoGroup->usingMask = false;
    hmoGroup->batch = opSpec.batch;
    hmoGroup->dims = opSpec.dims;
    hmoGroup->k = bufferSpec.k;
    hmoGroup->blockSize = opSpec.codeBlockSize;
    hmoGroup->defaultNumBlocks = opSpec.defaultNumDistOps;
    hmoGroup->ntotal = opSpec.codeBlockSize * opSpec.defaultNumDistOps;
    hmoGroup->groupId = 0;
    hmoGroup->queriesHmo = AllocHmo(opSpec.batch * opSpec.dims * sizeof(int8_t));
    hmoGroup->queriesNormHmo = AllocHmo(utils::SafeRoundUp(opSpec.batch, nop::FP16_ALIGN) * sizeof(OckFloat16));
    hmoGroup->maskHMO = nullptr;
    hmoGroup->topkDistsHmo = AllocHmo(opSpec.batch * bufferSpec.k * sizeof(OckFloat16));
    hmoGroup->topkLabelsHmo = AllocHmo(opSpec.batch * bufferSpec.k * sizeof(int64_t));
    hmoGroup->featuresHmo.resize(opSpec.defaultNumDistOps);
    hmoGroup->normsHmo.resize(opSpec.defaultNumDistOps);
    for (int64_t i = 0; i < opSpec.defaultNumDistOps; ++i) {
        hmoGroup->featuresHmo[i] = AllocHmo(opSpec.codeBlockSize * opSpec.dims * sizeof(int8_t));
        hmoGroup->normsHmo[i] = AllocHmo(opSpec.codeBlockSize * sizeof(OckFloat16));
    }
    UpdateMetaFromHmoGroup(opSpec, bufferSpec, hmoGroup);
    auto buffer = ock::hcps::hcop::OckTopkDistCompOpDataBuffer::Create(opSpec, bufferSpec);
    auto ret = buffer->PrepareHyperParameters(hmoGroup);
    EXPECT_EQ(ret, hmm::HMM_SUCCESS);
}
} // namespace test
} // namespace hcop
} // namespace hcps
} // namespace ock
