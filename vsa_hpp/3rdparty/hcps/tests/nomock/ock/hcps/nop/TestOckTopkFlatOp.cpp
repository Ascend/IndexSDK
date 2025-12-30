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
#include "ock/hcps/handler/OckHeteroHandler.h"
#include "ock/hcps/handler/OckHandlerHmmHelper.h"
#include "ock/hcps/stream/OckHeteroStreamBase.h"
#include "ock/hcps/nop/topk_flat_op/OckTopkFlatOpDataBuffer.h"
#include "ock/hcps/nop/topk_flat_op/OckTopkFlatOpFactory.h"
namespace ock {
namespace hcps {
namespace nop {
namespace test {
class TestOckTopkFlatOp : public testing::Test {
public:
    void SetUp(void) override
    {
        BuildDeviceInfo();
        aclrtSetDevice(deviceInfo->deviceId);
        hmm::OckHmmErrorCode errorCode = hmm::HMM_SUCCESS;
        handler = handler::OckHeteroHandler::CreateSingleDeviceHandler(
            deviceInfo->deviceId, deviceInfo->cpuSet, deviceInfo->memorySpec, errorCode);
        streamBase = handler::helper::MakeStream(*handler, errorCode, OckDevStreamType::AI_CPU);
        PrepareHostData();
    }

    void TearDown(void) override
    {
        aclrtResetDevice(deviceInfo->deviceId);
    }

    void PrepareHostData()
    {
        bufferSpec.k = 4;
        dists.resize(bufferSpec.blockNum * opSpec.batch * opSpec.codeBlockSize, 0);
        minDists.resize(bufferSpec.blockNum * opSpec.batch * utils::SafeDivUp(opSpec.codeBlockSize, BURST_LEN) * 2LL,
            0);
        for (uint16_t i = 0; i < 4U; ++i) {
            dists[i] = acladapter::OckAscendFp16::FloatToFp16(sampleDists[i]);
            minDists[i * 2U] = acladapter::OckAscendFp16::FloatToFp16(sampleDists[i]);
            minDists[i * 2U + 1U] = acladapter::OckAscendFp16::FloatToFp16(sampleDists[i]);
        }

        sizes.resize(bufferSpec.blockNum * CORE_NUM * SIZE_ALIGN, 0);
        sizes[0U] = opSpec.codeBlockSize;
        sizes[1U] = 0;                                // 当前block的首行在整个底库里面的行offset
        sizes[2U] = (opSpec.codeBlockSize + 7U) / 8U; // mask用的
        sizes[3U] = 0;                                // 使用mask
        sizes[4U] = opSpec.codeBlockSize;

        flags.resize(bufferSpec.blockNum * FLAG_NUM * FLAG_SIZE, 1);

        attrs = {
            0U, bufferSpec.k, BURST_LEN, bufferSpec.blockNum, 0U, 1U, opSpec.codeBlockSize * bufferSpec.blockNum,
            0U, opSpec.codeBlockSize
        };
    }

    static void CheckResults(const OckTopkFlatOpMeta &opSpec, const OckTopkFlatBufferMeta &bufferSpec,
        const std::shared_ptr<OckTopkFlatOpDataBuffer> &dataBuffer, const std::vector<float> &sampleDists,
        const std::vector<int64_t> &sampleLabels)
    {
        // 比较结果
        std::vector<OckFloat16> outputDists(opSpec.batch * bufferSpec.k);
        std::vector<int64_t> outputLabels(opSpec.batch * bufferSpec.k);
        aclrtMemcpy(outputDists.data(), outputDists.size() * sizeof(OckFloat16),
            reinterpret_cast<void *>(dataBuffer->OutputDists()->Addr()), dataBuffer->OutputDists()->GetByteSize(),
            ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(outputLabels.data(), outputLabels.size() * sizeof(int64_t),
            reinterpret_cast<void *>(dataBuffer->OutputLabels()->Addr()), dataBuffer->OutputLabels()->GetByteSize(),
            ACL_MEMCPY_DEVICE_TO_HOST);
        for (uint16_t i = 0; i < 4U; ++i) {
            EXPECT_EQ(utils::SafeFloatEqual(acladapter::OckAscendFp16::Fp16ToFloat(outputDists[i]),
                sampleDists[sampleLabels[i]], 0.0001),  // 0.0001: fp16精度比double低
                true);
            EXPECT_EQ(outputLabels[i], sampleLabels[i]);
        }
    }

    std::shared_ptr<handler::OckHeteroHandler> handler;
    std::shared_ptr<OckHeteroStreamBase> streamBase;

    OckTopkFlatOpMeta opSpec;
    OckTopkFlatBufferMeta bufferSpec;
    std::vector<float> sampleDists{ 0.12345, 0.234, 0.4621, 0.11 }; // 构造前4个底库向量的距离(其他向量的距离值都为0)
    std::vector<int64_t> sampleLabels{ 2, 1, 0, 3 }; // 构造前4个底库向量的topk排序

    std::vector<OckFloat16> dists;
    std::vector<OckFloat16> minDists;
    std::vector<uint32_t> sizes;
    std::vector<uint16_t> flags;
    std::vector<int64_t> attrs;

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
TEST_F(TestOckTopkFlatOp, run)
{
    // alloc buffer
    std::shared_ptr<OckTopkFlatOpDataBuffer> dataBuffer = OckTopkFlatOpDataBuffer::Create(opSpec, bufferSpec);
    auto errorCode = dataBuffer->AllocBuffers(handler->HmmMgrPtr());
    EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);

    // 创建算子
    auto topkFlatOp =
        OckTopkFlatOpFactory::Create(opSpec)->Create(dataBuffer);

    // 填充数据
    FillBuffer<OckFloat16>(dataBuffer->InputDists(), dists);
    FillBuffer<OckFloat16>(dataBuffer->InputMinDists(), minDists);
    FillBuffer<uint32_t>(dataBuffer->InputSizes(), sizes);
    FillBuffer<uint16_t>(dataBuffer->InputFlags(), flags);
    FillBuffer<int64_t>(dataBuffer->InputAttrs(), attrs);

    // 运行算子
    streamBase->AddOp(topkFlatOp);
    streamBase->WaitExecComplete();

    // 检查算子运行结果
    CheckResults(opSpec, bufferSpec, dataBuffer, sampleDists, sampleLabels);
}
}  // namespace test
}  // namespace nop
}  // namespace hcps
}  // namespace ock
