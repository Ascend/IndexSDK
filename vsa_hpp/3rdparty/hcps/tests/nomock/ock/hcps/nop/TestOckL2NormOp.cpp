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
#include "acl/acl.h"
#include "ock/hmm/OckHmmFactory.h"
#include "ock/hmm/mgr/OckHmmHMObject.h"
#include "ock/utils/OckSafeUtils.h"
#include "ock/log/OckTestLoggerEx.h"
#include "ock/hcps/handler/OckHandlerHmmHelper.h"
#include "ock/hcps/nop/l2_norm_op/OckL2NormOpDataBuffer.h"
#include "ock/hcps/nop/l2_norm_op/OckL2NormOpFactory.h"

namespace ock {
namespace hcps {
namespace nop {
namespace test {
class TestOckL2NormOp : public testing::Test {
public:
    void SetUp(void) override
    {
        BuildDeviceInfo();
        aclrtSetDevice(deviceInfo->deviceId);
        hmm::OckHmmErrorCode errorCode = hmm::HMM_SUCCESS;
        handler = handler::OckHeteroHandler::CreateSingleDeviceHandler(
            deviceInfo->deviceId, deviceInfo->cpuSet, deviceInfo->memorySpec, errorCode);
        streamBase = handler::helper::MakeStream(*handler, errorCode, OckDevStreamType::AI_CORE);
        PrepareHostData();
    }

    void TearDown(void) override
    {
        aclrtResetDevice(deviceInfo->deviceId);
    }

    void PrepareHostData()
    {
        vectors.resize(bufferSpec.ntotal * opSpec.dims, sampleBaseElements);
        transfer.resize(TRANSFER_SIZE * CUBE_ALIGN, acladapter::OckAscendFp16::FloatToFp16(0));
        for (int i = 0; i < TRANSFER_SIZE / CUBE_ALIGN; ++i) {
            for (int j = 0; j < CUBE_ALIGN; ++j) {
                transfer[i * CUBE_ALIGN * CUBE_ALIGN + j * CUBE_ALIGN + j] = sampleTransferElements;
            }
        }
        actualNum.resize(SIZE_ALIGN);
        actualNum[0] = static_cast<uint32_t>(bufferSpec.ntotal);
        result.resize(bufferSpec.ntotal, sampleFp16Elements);
    }

    void CheckResults(const OckL2NormOpMeta &opSpec, const OckL2NormBufferMeta &bufferSpec,
        const std::shared_ptr<OckL2NormOpDataBuffer> &dataBuffer, float correctNorm)
    {
        std::vector<OckFloat16> outNorm(bufferSpec.ntotal, sampleFp16Elements);
        aclrtMemcpy(outNorm.data(), outNorm.size() * sizeof(OckFloat16),
            reinterpret_cast<void *>(dataBuffer->OutputResult()->Addr()), dataBuffer->OutputResult()->GetByteSize(),
            ACL_MEMCPY_DEVICE_TO_HOST);
        EXPECT_TRUE(utils::SafeFloatEqual(acladapter::OckAscendFp16::Fp16ToFloat(outNorm[0]), correctNorm, 0.0001F));
    }

    std::shared_ptr<handler::OckHeteroHandler> handler;
    std::shared_ptr<OckHeteroStreamBase> streamBase;

    OckL2NormOpMeta opSpec;
    OckL2NormBufferMeta bufferSpec;

    std::vector<int8_t> vectors;
    std::vector<OckFloat16> transfer;
    std::vector<uint32_t> actualNum;
    std::vector<OckFloat16> result;
    int8_t sampleBaseElements{ 1 };
    float correctSampleNorm{ 0.625 };
    OckFloat16 sampleFp16Elements{ acladapter::OckAscendFp16::FloatToFp16(0) };
    OckFloat16 sampleTransferElements{ acladapter::OckAscendFp16::FloatToFp16(1) };

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

    std::shared_ptr<hmm::OckHmmDeviceInfo> deviceInfo;
};
TEST_F(TestOckL2NormOp, run)
{
    // alloc buffer
    std::shared_ptr<OckL2NormOpDataBuffer> dataBuffer =
        OckL2NormOpDataBuffer::Create(opSpec, bufferSpec);
    auto errorCode = dataBuffer->AllocBuffers(handler->HmmMgrPtr());
    EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
    
    // 创建算子
    auto distOp = OckL2NormOpFactory::Create(opSpec)->Create(dataBuffer);

    // 填充数据
    FillBuffer<int8_t>(dataBuffer->InputVectors(), vectors);
    FillBuffer<OckFloat16>(dataBuffer->InputTransfer(), transfer);
    FillBuffer<uint32_t>(dataBuffer->InputActualNum(), actualNum);
    FillBuffer<OckFloat16>(dataBuffer->OutputResult(), result);

    // 运行算子
    streamBase->AddOp(distOp);
    streamBase->WaitExecComplete();

    // 检查算子运行结果
    CheckResults(opSpec, bufferSpec, dataBuffer, correctSampleNorm);
}
} // namespace test
} // namespace nop
} // namespace hcps
} // namespace ock