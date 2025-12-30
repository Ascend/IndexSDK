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

#include "gtest/gtest.h"
#include "acl/acl.h"
#include "ock/hmm/mgr/OckHmmHMObject.h"
#include "ock/utils/OckSafeUtils.h"
#include "ock/log/OckTestLoggerEx.h"
#include "ock/hcps/stream/OckHeteroStreamBase.h"
#include "ock/hcps/handler/OckHeteroHandler.h"
#include "ock/hcps/handler/OckHandlerHmmHelper.h"
#include "ock/hcps/nop/dist_int8_cos_max_op/OckDistInt8CosMaxOpDataBuffer.h"
#include "ock/hcps/nop/dist_int8_cos_max_op/OckDistInt8CosMaxOpFactory.h"
namespace ock {
namespace hcps {
namespace nop {
namespace test {
class TestOckDistInt8CosMaxOp : public testing::Test {
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
        queries.resize(opSpec.batch * opSpec.dims, sampleQueryElements);
        queriesNorm.resize((opSpec.batch + FP16_ALIGN - 1U) / FP16_ALIGN * FP16_ALIGN, sampleQueryNormElements);
        shaped.resize(opSpec.codeBlockSize * opSpec.dims, sampleBaseElements);
        codesNorm.resize(opSpec.codeBlockSize, sampleBaseNormElements);

        actualSize.resize(CORE_NUM * SIZE_ALIGN);
        actualSize[0U] = opSpec.codeBlockSize;
        actualSize[1U] = 0U;
        actualSize[2U] = utils::SafeDivUp(bufferSpec.ntotal, BINARY_BYTE_SIZE);
        actualSize[3U] = 0U;
        actualSize[4U] = opSpec.codeBlockSize;

        outFlag.resize(FLAG_NUM * FLAG_SIZE, 0U);
    }

    static void CheckResults(const OckDistInt8CosMaxOpMeta &opSpec, const OckDistInt8CosMaxBufferMeta &bufferSpec,
        const std::shared_ptr<OckDistInt8CosMaxOpDataBuffer> &dataBuffer, float correctDist)
    {
        std::vector<OckFloat16> outDist(opSpec.batch * opSpec.codeBlockSize);
        aclrtMemcpy(outDist.data(), outDist.size() * sizeof(OckFloat16),
            reinterpret_cast<void *>(dataBuffer->OutputDists()->Addr()), dataBuffer->OutputDists()->GetByteSize(),
            ACL_MEMCPY_DEVICE_TO_HOST);
        EXPECT_EQ(utils::SafeFloatEqual(acladapter::OckAscendFp16::Fp16ToFloat(outDist[0]), correctDist,
            0.0001), // 0.0001: fp16精度比double低
            true);

        std::vector<uint16_t> newOutFlag(FLAG_NUM * FLAG_SIZE, 7U);
        aclrtMemcpy(newOutFlag.data(), newOutFlag.size() * sizeof(uint16_t),
            reinterpret_cast<void *>(dataBuffer->OutputFlag()->Addr()), dataBuffer->OutputFlag()->GetByteSize(),
            ACL_MEMCPY_DEVICE_TO_HOST);
        for (uint16_t i = 0; i < CORE_NUM; ++i) {
            EXPECT_EQ(newOutFlag[i * FLAG_SIZE], 1U);
        }
        for (uint16_t i = 0; i < CORE_NUM; ++i) {
            EXPECT_EQ(newOutFlag[(i + CORE_NUM) * FLAG_SIZE], 0U);
        }
    }

    std::shared_ptr<handler::OckHeteroHandler> handler;
    std::shared_ptr<OckHeteroStreamBase> streamBase;

    OckDistInt8CosMaxOpMeta opSpec;
    OckDistInt8CosMaxBufferMeta bufferSpec;

    std::vector<int8_t> queries;
    std::vector<uint8_t> mask;
    std::vector<int8_t> shaped;
    std::vector<OckFloat16> queriesNorm;
    std::vector<OckFloat16> codesNorm;
    std::vector<uint32_t> actualSize;
    std::vector<uint16_t> outFlag;
    float correctSampleDist{ 0.01 };
    int8_t sampleQueryElements{ 1 };
    int8_t sampleBaseElements{ 2 };
    OckFloat16 sampleQueryNormElements{ acladapter::OckAscendFp16::FloatToFp16(0.0625) };
    OckFloat16 sampleBaseNormElements{ acladapter::OckAscendFp16::FloatToFp16(0.03125) };

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

TEST_F(TestOckDistInt8CosMaxOp, run)
{
    // alloc buffer
    std::shared_ptr<OckDistInt8CosMaxOpDataBuffer> dataBuffer =
        OckDistInt8CosMaxOpDataBuffer::Create(opSpec, bufferSpec);
    auto errorCode = dataBuffer->AllocBuffers(handler->HmmMgrPtr());
    EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
    // 创建算子
    auto distOp =
        OckDistInt8CosMaxOpFactory::Create(opSpec)->Create(dataBuffer);

    // 填充数据
    FillBuffer<int8_t>(dataBuffer->InputQueries(), queries);
    FillBuffer<int8_t>(dataBuffer->InputShaped(), shaped);
    FillBuffer<OckFloat16>(dataBuffer->InputQueriesNorm(), queriesNorm);
    FillBuffer<OckFloat16>(dataBuffer->InputCodesNorm(), codesNorm);
    FillBuffer<uint32_t>(dataBuffer->InputActualSize(), actualSize);
    FillBuffer<uint16_t>(dataBuffer->OutputFlag(), outFlag);

    // 运行算子
    streamBase->AddOp(distOp);
    streamBase->WaitExecComplete();

    // 检查算子运行结果
    CheckResults(opSpec, bufferSpec, dataBuffer, correctSampleDist);
}
} // namespace test
} // namespace nop
} // namespace hcps
} // namespace ock
