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
#include "ock/hmm/mgr/OckHmmHMObject.h"
#include "ock/utils/OckSafeUtils.h"
#include "ock/log/OckTestLoggerEx.h"
#include "ock/hcps/handler/OckHeteroHandler.h"
#include "ock/hcps/handler/OckHandlerHmmHelper.h"
#include "ock/hcps/stream/OckHeteroStreamBase.h"
#include "ock/hcps/algo/OckShape.h"
#include "ock/hcps/algo/impl/OckShapeImpl.h"
#include "ock/hcps/nop/trans_data_shaped_op/OckTransDataShapedOpDataBuffer.h"
#include "ock/hcps/nop/trans_data_shaped_op/OckTransDataShapedOpFactory.h"

namespace ock {
namespace hcps {
namespace nop {
namespace test {
class TestOckTransDataShapedOp : public testing::Test {
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
        src.resize(bufferSpec.ntotal * opSpec.dims, 0U);
        for (int64_t i = 0; i < bufferSpec.ntotal; ++i) {
            for (int64_t j = 0; j < opSpec.dims; ++j) {
                src[i * opSpec.dims + j] = i + j;
            }
        }
        attrs.resize(TRANSDATA_SHAPED_ATTR_IDX_COUNT, 0U);
        dst.resize(bufferSpec.ntotal * opSpec.dims, 0U);
        shapedData.resize(bufferSpec.ntotal * opSpec.dims, 0U);
        uintptr_t addr = reinterpret_cast<uintptr_t>(shapedData.data());
        ock::hcps::algo::OckShape<> shape(addr, shapedData.size());
        shape.AddData(src.data(), bufferSpec.ntotal);
    }

    bool CheckResults(const OckTransDataShapedOpMeta &opSpec, const OckTransDataShapedBufferMeta &bufferSpec,
        const std::shared_ptr<OckTransDataShapedOpDataBuffer> &dataBuffer)
    {
        std::vector<int8_t> res(bufferSpec.ntotal * opSpec.dims, 0U);
        aclrtMemcpy(res.data(), res.size(),
            reinterpret_cast<void *>(dataBuffer->OutputDst()->Addr()), dataBuffer->OutputDst()->GetByteSize(),
            ACL_MEMCPY_DEVICE_TO_HOST);
        for (size_t i = 0; i < res.size(); ++i) {
            if (res[i] != shapedData[i]) {
                return false;
                break;
            }
        }
        return true;
    }

    std::shared_ptr<handler::OckHeteroHandler> handler;
    std::shared_ptr<OckHeteroStreamBase> streamBase;

    OckTransDataShapedOpMeta opSpec;
    OckTransDataShapedBufferMeta bufferSpec;

    std::vector<int8_t> src;
    std::vector<int64_t> attrs;
    std::vector<int8_t> dst;
    std::vector<int8_t> shapedData;

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
TEST_F(TestOckTransDataShapedOp, run)
{
    // alloc buffer
    std::shared_ptr<OckTransDataShapedOpDataBuffer> dataBuffer =
        OckTransDataShapedOpDataBuffer::Create(opSpec, bufferSpec);
    auto errorCode = dataBuffer->AllocBuffers(handler->HmmMgrPtr());
    EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);

    // 创建算子
    auto distOp = OckTransDataShapedOpFactory::Create(opSpec)->Create(dataBuffer);

    // 填充数据
    FillBuffer<int8_t>(dataBuffer->InputSrc(), src);
    FillBuffer<int64_t>(dataBuffer->InputAttr(), attrs);
    FillBuffer<int8_t>(dataBuffer->OutputDst(), dst);

    // 运行算子
    streamBase->AddOp(distOp);
    streamBase->WaitExecComplete();

    // 检查算子运行结果
    EXPECT_EQ(CheckResults(opSpec, bufferSpec, dataBuffer), true);
}
} // namespace test
} // namespace nop
} // namespace hcps
} // namespace ock
