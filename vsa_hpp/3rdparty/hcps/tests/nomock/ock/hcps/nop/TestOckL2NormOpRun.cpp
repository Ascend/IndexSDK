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


#include <iostream>
#include "gtest/gtest.h"
#include "ock/hmm/OckHmmFactory.h"
#include "ock/hmm/mgr/OckHmmHMObject.h"
#include "ock/utils/OckSafeUtils.h"
#include "ock/log/OckTestLoggerEx.h"
#include "ock/hcps/handler/OckHandlerHmmHelper.h"
#include "ock/hcps/nop/l2_norm_op/OckL2NormOpDataBuffer.h"
#include "ock/hcps/nop/l2_norm_op/OckL2NormOpFactory.h"
#include "ock/hcps/nop/l2_norm_op/OckL2NormOpRun.h"

namespace ock {
namespace hcps {
namespace nop {
namespace test {
class TestOckL2NormOpRun : public testing::Test {
public:
    void SetUp(void) override
    {
        BuildDeviceInfo();
        aclrtSetDevice(deviceInfo->deviceId);
        hmm::OckHmmErrorCode errorCode = hmm::HMM_SUCCESS;
        handler = handler::OckHeteroHandler::CreateSingleDeviceHandler(
            deviceInfo->deviceId, deviceInfo->cpuSet, deviceInfo->memorySpec, errorCode);
        streamBase = handler::helper::MakeStream(*handler, errorCode, OckDevStreamType::AI_CORE);
    }

    void TearDown(void) override
    {
        aclrtResetDevice(deviceInfo->deviceId);
    }

    void PrepareHostData(int64_t addNum)
    {
        std::vector<int8_t> hostData(addNum * 256U, 1U);
        dataHmo = AllocHmo(addNum * 256U * sizeof(int8_t));
        WriteHmo(dataHmo, hostData);
    }

    bool CheckResults(const std::shared_ptr<OckL2NormOpHmoBlock> &hmoBlock, float correctNorm)
    {
        std::vector<OckFloat16> outNorm(hmoBlock->addNum, sampleFp16Elements);
        ReadHmo(hmoBlock->normResult, outNorm);
        for (size_t i = 0; i < outNorm.size(); ++i) {
            if (!utils::SafeFloatEqual(acladapter::OckAscendFp16::Fp16ToFloat(outNorm[i]), correctNorm, 0.0001F)) {
                std::cout << acladapter::OckAscendFp16::Fp16ToFloat(outNorm[i]) << " " << i << std::endl;
                return false;
            }
        }
        return true;
    }

    std::shared_ptr<handler::OckHeteroHandler> handler;
    std::shared_ptr<OckHeteroStreamBase> streamBase;

    std::shared_ptr<hmm::OckHmmHMObject> dataHmo;
    float correctSampleNorm{ 0.625 };
    OckFloat16 sampleFp16Elements{ acladapter::OckAscendFp16::FloatToFp16(0) };

private:
    void BuildDeviceInfo()
    {
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
    std::shared_ptr<hmm::OckHmmHMObject> AllocHmo(int64_t byteSize)
    {
        return handler->HmmMgrPtr()->Alloc(byteSize, hmm::OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY).second;
    }
    template <typename T> void WriteHmo(const std::shared_ptr<hmm::OckHmmHMObject> &hmo, const std::vector<T> &vec)
    {
        aclrtMemcpy(reinterpret_cast<void *>(hmo->Addr()), hmo->GetByteSize(), vec.data(), vec.size() * sizeof(T),
            ACL_MEMCPY_HOST_TO_DEVICE);
    }
    template <typename T> void ReadHmo(const std::shared_ptr<hmm::OckHmmHMObject> &hmo, std::vector<T> &vec)
    {
        aclrtMemcpy(vec.data(), vec.size() * sizeof(T), reinterpret_cast<void *>(hmo->Addr()), vec.size() * sizeof(T),
            ACL_MEMCPY_DEVICE_TO_HOST);
    }
    std::shared_ptr<hmm::OckHmmDeviceInfo> deviceInfo;
};
TEST_F(TestOckL2NormOpRun, run)
{
    ock::hmm::OckHmmErrorCode errorCode = hmm::HMM_SUCCESS;
    PrepareHostData(DEFAULT_CODE_BLOCK_SIZE - 1U);
    auto hmoBlock = OckL2NormOpRun::BuildNormHmoBlock(dataHmo, *handler, 256U, DEFAULT_CODE_BLOCK_SIZE - 1U, errorCode);
    EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
    OckL2NormOpRun::ComputeNormSync(hmoBlock, *handler, streamBase);
    streamBase->WaitExecComplete();

    // 检查算子运行结果
    EXPECT_EQ(CheckResults(hmoBlock, correctSampleNorm), true);
}
} // namespace test
} // namespace nop
} // namespace hcps
} // namespace ock