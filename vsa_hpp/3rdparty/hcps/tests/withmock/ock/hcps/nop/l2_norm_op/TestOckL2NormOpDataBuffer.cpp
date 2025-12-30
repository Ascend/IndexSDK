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
#include "ock/hcps/nop/l2_norm_op/OckL2NormOpDataBuffer.h"
namespace ock {
namespace hcps {
namespace nop {
namespace test {
class TestOckL2NormOpDataBuffer : public acladapter::WithEnvAclMock<testing::Test> {
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
        BuildHmmDeviceMgr();
        hostDataBytes.resize(4U, 0LL);
    }
    void TearDown(void) override
    {
        hmoBlock.reset();
        devMgr.reset();
        deviceInfo.reset();
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
    hmm::OckHmmErrorCode BuildHmoBlock(void)
    {
        hmoBlock = std::make_shared<OckL2NormOpHmoBlock>();
        auto hmoRet = devMgr->Alloc(L2NORM_COMPUTE_BATCH * 256U * sizeof(int8_t),
            hmm::OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
        OCK_CHECK_RETURN_ERRORCODE(hmoRet.first);
        hmoBlock->dataBase = hmoRet.second;
        hmoRet =
            devMgr->Alloc(L2NORM_COMPUTE_BATCH * sizeof(OckFloat16), hmm::OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
        OCK_CHECK_RETURN_ERRORCODE(hmoRet.first);
        hmoBlock->normResult = hmoRet.second;
        hmoBlock->dims = 256U;
        hmoBlock->addNum = L2NORM_COMPUTE_BATCH;
        return hmm::HMM_SUCCESS;
    }
    std::shared_ptr<hmm::OckHmmDeviceInfo> deviceInfo;
    std::shared_ptr<hmm::OckHmmSingleDeviceMgr> devMgr;
    std::shared_ptr<OckL2NormOpHmoBlock> hmoBlock;

    OckL2NormOpMeta opSpec;
    OckL2NormBufferMeta bufferSpec;
    std::vector<uint64_t> hostDataBytes;
};

TEST_F(TestOckL2NormOpDataBuffer, alloc_buffers_from_hmo_group)
{
    BuildHmoBlock();
    auto buffer = ock::hcps::nop::OckL2NormOpDataBuffer::Create(opSpec, bufferSpec);
    EXPECT_EQ(buffer->AllocBuffersFromHmoBlock(hmoBlock, devMgr, 0U), hmm::HMM_SUCCESS);
    EXPECT_EQ(*(buffer->InputVectors()), *(buffer->GetInputParams()[0U]));
    EXPECT_EQ(*(buffer->InputTransfer()), *(buffer->GetInputParams()[1U]));
    EXPECT_EQ(*(buffer->InputActualNum()), *(buffer->GetInputParams()[2U]));
    EXPECT_EQ(*(buffer->OutputResult()), *(buffer->GetOutputParams()[0U]));
}
} // namespace test
} // namespace nop
} // namespace hcps
} // namespace ock