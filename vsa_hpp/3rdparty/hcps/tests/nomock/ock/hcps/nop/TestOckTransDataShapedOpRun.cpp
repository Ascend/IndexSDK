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
#include "ock/utils/OckSafeUtils.h"
#include "ock/log/OckTestLoggerEx.h"
#include "ock/hcps/handler/OckHandlerHmmHelper.h"
#include "ock/hcps/stream/OckHeteroStreamBase.h"
#include "ock/hcps/algo/OckShape.h"
#include "ock/hcps/algo/impl/OckShapeImpl.h"
#include "ock/hcps/nop/trans_data_shaped_op/OckTransDataShapedOpRun.h"

namespace ock {
namespace hcps {
namespace nop {
namespace test {
class TestOckTransDataShapedOpRun : public testing::Test {
public:
    void SetUp(void) override
    {
        BuildDeviceInfo();
        aclrtSetDevice(deviceInfo->deviceId);
        hmm::OckHmmErrorCode errorCode = hmm::HMM_SUCCESS;
        handler = handler::OckHeteroHandler::CreateSingleDeviceHandler(
            deviceInfo->deviceId, deviceInfo->cpuSet, deviceInfo->memorySpec, errorCode);
        streamBase = handler::helper::MakeStream(*handler, errorCode, OckDevStreamType::AI_CPU);
    }

    void TearDown(void) override
    {
        aclrtResetDevice(deviceInfo->deviceId);
    }

    void PrepareHostData(int64_t addNum)
    {
        hmoBlock = std::make_shared<OckTransDataShapedOpHmoBlock>();
        hmoBlock->srcHmo = AllocHmo(addNum * 256U);
        hmoBlock->dstHmo = AllocHmo(DEFAULT_CODE_BLOCK_SIZE * 256U);
        hmoBlock->dims = 256U;
        hmoBlock->codeBlockSize = DEFAULT_CODE_BLOCK_SIZE;
        hmoBlock->addNum = addNum;
        hmoBlock->offsetInDstHmo = DEFAULT_CODE_BLOCK_SIZE - addNum;
        std::vector<int8_t> src(DEFAULT_CODE_BLOCK_SIZE * 256U, 0U);
        for (int64_t i = 0; i < DEFAULT_CODE_BLOCK_SIZE; ++i) {
            for (int64_t j = 0; j < 256U; ++j) {
                src[i * 256U + j] = i + j;
            }
        }
        WriteHmo(hmoBlock->srcHmo, src);
        shapedData.resize(DEFAULT_CODE_BLOCK_SIZE * 256U, 0U);
        uintptr_t addr = reinterpret_cast<uintptr_t>(shapedData.data());
        ock::hcps::algo::OckShape<> shape(addr, shapedData.size());
        shape.AddData(src.data(), DEFAULT_CODE_BLOCK_SIZE);
    }

    bool CheckResults()
    {
        std::vector<int8_t> res(hmoBlock->addNum * hmoBlock->dims, 0U);
        ReadHmo(hmoBlock->dstHmo, hmoBlock->offsetInDstHmo * hmoBlock->dims, res);
        for (size_t i = 0; i < res.size(); ++i) {
            if (res[i] != shapedData[i]) {
                std::cout << i << std::endl;
                std::cout << int(res[i]) << std::endl;
                std::cout << int(shapedData[i]) << std::endl;
                return false;
                break;
            }
        }
        return true;
    }

    std::shared_ptr<hcps::handler::OckHeteroHandler> handler;
    std::shared_ptr<OckHeteroStreamBase> streamBase;

    std::shared_ptr<OckTransDataShapedOpHmoBlock> hmoBlock;
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
    std::shared_ptr<hmm::OckHmmHMObject> AllocHmo(int64_t byteSize)
    {
        return handler->HmmMgrPtr()->Alloc(byteSize, hmm::OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY).second;
    }
    template <typename T> void WriteHmo(std::shared_ptr<hmm::OckHmmHMObject> hmo, const std::vector<T> &vec)
    {
        aclrtMemcpy(reinterpret_cast<void *>(hmo->Addr()), hmo->GetByteSize(), vec.data(), hmo->GetByteSize(),
            ACL_MEMCPY_HOST_TO_DEVICE);
    }
    template <typename T> void ReadHmo(std::shared_ptr<hmm::OckHmmHMObject> hmo, int64_t offset, std::vector<T> &vec)
    {
        aclrtMemcpy(vec.data(), vec.size() * sizeof(T), reinterpret_cast<void *>(hmo->Addr() + offset),
            vec.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    }
    std::shared_ptr<hmm::OckHmmDeviceInfo> deviceInfo;
};
TEST_F(TestOckTransDataShapedOpRun, run)
{
    PrepareHostData(DEFAULT_CODE_BLOCK_SIZE - 16U);
    OckTransDataShapedOpRun::AddTransShapedOp(hmoBlock, *handler, streamBase);
    streamBase->WaitExecComplete();

    // 检查算子运行结果
    EXPECT_EQ(CheckResults(), true);
}
} // namespace test
} // namespace nop
} // namespace hcps
} // namespace ock