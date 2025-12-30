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
#include "ock/hcps/nop/trans_data_custom_attr_op/OckTransDataCustomAttrOpDataBuffer.h"
namespace ock {
namespace hcps {
namespace nop {
namespace test {
class TestOckTransDataCustomAttrOpDataBuffer : public acladapter::WithEnvAclMock<testing::Test> {
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
        DestroyHmmDeviceMgr(); // 需要提前reset，否则打桩不生效。
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
    void DestroyHmmDeviceMgr(void)
    {
        if (devMgr.get() != nullptr) {
            devMgr.reset();
        }
    }
    void BuildHmoBlock(void)
    {
        hmoBlock = std::make_shared<OckTransDataCustomAttrOpHmoBlock>();
        hmoBlock->srcHmo = devMgr
                               ->Alloc(opSpec.copyCount * opSpec.customAttrLen * sizeof(uint8_t),
            hmm::OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY)
                               .second;
        hmoBlock->dstHmo = devMgr
                               ->Alloc(opSpec.customAttrLen * opSpec.customAttrBlockSize * sizeof(uint8_t),
            hmm::OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY)
                               .second;
    }
    std::shared_ptr<hmm::OckHmmDeviceInfo> deviceInfo;
    std::shared_ptr<hmm::OckHmmSingleDeviceMgr> devMgr;
    std::shared_ptr<OckTransDataCustomAttrOpHmoBlock> hmoBlock;

    OckTransDataCustomAttrOpMeta opSpec;
    OckTransDataCustomAttrBufferMeta bufferSpec;
    std::vector<uint64_t> hostDataBytes;
};

TEST_F(TestOckTransDataCustomAttrOpDataBuffer, alloc_buffers_from_hmo_block)
{
    BuildHmoBlock();
    auto buffer = ock::hcps::nop::OckTransDataCustomAttrOpDataBuffer::Create(opSpec, bufferSpec);
    buffer->AllocBuffersFromHmoBlock(hmoBlock, devMgr);
    EXPECT_EQ(*(buffer->InputSrc()), *(buffer->GetInputParams()[0U]));
    EXPECT_EQ(*(buffer->InputAttr()), *(buffer->GetInputParams()[1U]));
    EXPECT_EQ(*(buffer->OutputDst()), *(buffer->GetOutputParams()[0U]));
}
} // namespace test
} // namespace nop
} // namespace hcps
} // namespace ock
