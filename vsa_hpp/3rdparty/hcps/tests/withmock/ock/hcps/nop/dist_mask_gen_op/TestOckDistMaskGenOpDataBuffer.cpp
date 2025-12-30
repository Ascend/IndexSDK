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
#include "ock/log/OckHcpsLogger.h"
#include "ock/acladapter/WithEnvAclMock.h"
#include "ock/hcps/nop/dist_mask_gen_op/OckDistMaskGenOpDataBuffer.h"
namespace ock {
namespace hcps {
namespace nop {
namespace test {
class TestOckDistMaskGenOpDataBuffer : public acladapter::WithEnvAclMock<testing::Test> {
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
        hostDataBytes.resize(6U, 0LL);
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
    void PrepareHmos(void)
    {
        hmoBlock = std::make_shared<OckDistMaskGenOpHmoGroup>();
        hmoBlock->queryTimes = AllocWithCheck(OPS_DATA_TYPE_ALIGN * sizeof(int32_t));
        hmoBlock->queryTokenIds = AllocWithCheck(utils::SafeDivUp(2500U, OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES);
        hmoBlock->attrTimes = AllocWithCheck(DEFAULT_CODE_BLOCK_SIZE * sizeof(int32_t));
        hmoBlock->attrTokenQuotients = AllocWithCheck(DEFAULT_CODE_BLOCK_SIZE * sizeof(int32_t));
        hmoBlock->attrTokenRemainders = AllocWithCheck(DEFAULT_CODE_BLOCK_SIZE * OPS_DATA_TYPE_TIMES);
        hmoBlock->mask = AllocWithCheck(DEFAULT_CODE_BLOCK_SIZE / OPS_DATA_TYPE_ALIGN);
    }

    std::shared_ptr<hmm::OckHmmDeviceInfo> deviceInfo;
    std::shared_ptr<hmm::OckHmmSingleDeviceMgr> devMgr;

    OckDistMaskGenOpMeta opSpec;
    std::vector<uint64_t> hostDataBytes;

    std::shared_ptr<OckDistMaskGenOpHmoGroup> hmoBlock;

private:
    std::shared_ptr<hmm::OckHmmHMObject> AllocWithCheck(int64_t byteSize)
    {
        auto hmoRet = devMgr->Alloc(byteSize, hmm::OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY);
        if (hmoRet.first != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("Fail to alloc hmo! ErrCode = " << hmoRet.second);
            return nullptr;
        }
        return hmoRet.second;
    }
};

TEST_F(TestOckDistMaskGenOpDataBuffer, alloc_buffers_from_hmos)
{
    auto buffer = ock::hcps::nop::OckDistMaskGenOpDataBuffer::Create(opSpec);
    PrepareHmos();
    buffer->AllocBuffersFromHmoGroup(hmoBlock, 0U, DEFAULT_CODE_BLOCK_SIZE / OPS_DATA_TYPE_ALIGN);
    EXPECT_EQ(*(buffer->InputQueryTime()), *(buffer->GetInputParams()[0U]));
    EXPECT_EQ(*(buffer->InputTokenBitSet()), *(buffer->GetInputParams()[1U]));
    EXPECT_EQ(*(buffer->InputAttrTimes()), *(buffer->GetInputParams()[2U]));
    EXPECT_EQ(*(buffer->InputAttrTokenQs()), *(buffer->GetInputParams()[3U]));
    EXPECT_EQ(*(buffer->InputAttrTokenRs()), *(buffer->GetInputParams()[4U]));
    EXPECT_EQ(*(buffer->OutputMask()), *(buffer->GetOutputParams()[0U]));
}
} // namespace test
} // namespace nop
} // namespace hcps
} // namespace ock
