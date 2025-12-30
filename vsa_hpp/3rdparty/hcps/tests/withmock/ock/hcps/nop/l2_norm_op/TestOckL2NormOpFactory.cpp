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
#include "acl/acl.h"
#include "mockcpp/mockcpp.hpp"
#include "ock/acladapter/WithEnvAclMock.h"
#include "ock/hmm/OckHmmFactory.h"
#include "ock/hcps/modelpath/OckModelPath.h"
#include "ock/hcps/modelpath/OckSetModelPath.h"
#include "ock/hcps/nop/l2_norm_op/OckL2NormOpDataBuffer.h"
#include "ock/hcps/nop/l2_norm_op/OckL2NormOpFactory.h"
namespace ock {
namespace hcps {
namespace nop {
namespace test {
class TestOckL2NormOpFactory : public acladapter::WithEnvAclMock<testing::Test> {
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
        SetModelPath();
    }
    void TearDown(void) override
    {
        DestroyHmmDeviceMgr(); // 需要提前reset，否则打桩不生效。
        aclrtResetDevice(0);
        aclFinalize();
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
    static void SetModelPath()
    {
        OckModelPath::Instance().SetPath("./modelPath");
        OckSetModelPath::Instance().NotifyDevice();
    }

    std::shared_ptr<hmm::OckHmmDeviceInfo> deviceInfo;
    std::shared_ptr<hmm::OckHmmSingleDeviceMgr> devMgr;

    OckL2NormOpMeta opSpec;
    OckL2NormBufferMeta bufferSpec;
};
TEST_F(TestOckL2NormOpFactory, create_op)
{
    auto buffer = ock::hcps::nop::OckL2NormOpDataBuffer::Create(opSpec, bufferSpec);
    auto errorCode = buffer->AllocBuffers(devMgr);
    EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
    auto distOp =
        OckL2NormOpFactory::Create(opSpec)->Create(buffer);
    EXPECT_EQ(distOp->ResourceType(), acladapter::OckTaskResourceType::DEVICE_AI_CUBE);
}
} // namespace test
} // namespace nop
} // namespace hcps
} // namespace ock
