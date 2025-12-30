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
#include "ock/hcps/handler/OckHandlerHmmHelper.h"
#include "ock/hcps/stream/OckHeteroStreamBase.h"
#include "ock/hcps/nop/trans_data_custom_attr_op/OckTransDataCustomAttrOpRun.h"

namespace ock {
namespace hcps {
namespace nop {
namespace test {
class TestOckTransDataCustomAttrOpRun : public testing::Test {
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

    void PrepareData(uint32_t customAttrBlockSize, uint32_t customAttrLen)
    {
        customAttrs.resize(customAttrBlockSize * customAttrLen);
        for (uint32_t i = 0; i < customAttrs.size(); ++i) {
            customAttrs[i] = i % customAttrLen + 1;
        }

        hmoBlock = std::make_shared<OckTransDataCustomAttrOpHmoBlock>();
        hmoBlock->customAttrLen = customAttrLen;
        hmoBlock->customAttrBlockSize = customAttrBlockSize;
        hmoBlock->copyCount = customAttrBlockSize;
        hmoBlock->offsetInBlock = 0;
        hmoBlock->srcHmo = AllocHmo(customAttrBlockSize * customAttrLen);
        hmoBlock->dstHmo = AllocHmo(customAttrBlockSize * customAttrLen);
        WriteHmo(hmoBlock->srcHmo, customAttrs);
    }

    bool CheckResults()
    {
        std::vector<uint8_t> attrs(hmoBlock->customAttrBlockSize * hmoBlock->customAttrLen);
        ReadHmo(hmoBlock->dstHmo, attrs);
        for (uint32_t i = 0; i < hmoBlock->customAttrLen; ++i) {
            for (uint32_t j = 0; j < hmoBlock->customAttrBlockSize; ++j) {
                if (attrs[i * hmoBlock->customAttrBlockSize + j] != (i + 1)) {
                    return false;
                }
            }
        }
        return true;
    }

    std::shared_ptr<hcps::handler::OckHeteroHandler> handler;
    std::shared_ptr<OckHeteroStreamBase> streamBase;

    std::shared_ptr<OckTransDataCustomAttrOpHmoBlock> hmoBlock;
    std::vector<uint8_t> customAttrs;

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
    template <typename T> void ReadHmo(std::shared_ptr<hmm::OckHmmHMObject> hmo, std::vector<T> &vec)
    {
        aclrtMemcpy(vec.data(), vec.size() * sizeof(T), reinterpret_cast<void *>(hmo->Addr()), vec.size() * sizeof(T),
                    ACL_MEMCPY_DEVICE_TO_HOST);
    }
    std::shared_ptr<hmm::OckHmmDeviceInfo> deviceInfo;
};
TEST_F(TestOckTransDataCustomAttrOpRun, run)
{
    PrepareData(20U, 5U);
    OckTransDataCustomAttrOpRun::AddTransCustomAttrOp(hmoBlock, *handler, streamBase);
    streamBase->WaitExecComplete();

    // 检查算子运行结果
    EXPECT_EQ(CheckResults(), true);
}
} // namespace test
} // namespace nop
} // namespace hcps
} // namespace ock