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
#include "ock/hcps/nop/remove_data_custom_attr_op/OckRemoveDataCustomAttrOpRun.h"
#include "ock/hcps/nop/trans_data_custom_attr_op/OckTransDataCustomAttrOpRun.h"

namespace ock {
namespace hcps {
namespace nop {
namespace test {
class TestOckRemoveDataCustomAttrOpRun : public testing::Test {
public:
    void SetUp(void) override
    {
        BuildDeviceInfo();
        aclrtSetDevice(deviceInfo->deviceId);
        hmm::OckHmmErrorCode errorCode = hmm::HMM_SUCCESS;
        handler = handler::OckHeteroHandler::CreateSingleDeviceHandler(deviceInfo->deviceId, deviceInfo->cpuSet,
            deviceInfo->memorySpec, errorCode);
        streamBase = handler::helper::MakeStream(*handler, errorCode, OckDevStreamType::AI_CPU);
    }

    void TearDown(void) override
    {
        aclrtResetDevice(deviceInfo->deviceId);
    }

    // 这个算子是个ai cpu算子
    void PrepareTestData(uint32_t customAttrLen, uint32_t customAttrBlockSize)
    {
        transHmoBlock = std::make_shared<OckTransDataCustomAttrOpHmoBlock>();
        transHmoBlock->customAttrLen = customAttrLen;
        transHmoBlock->customAttrBlockSize = customAttrBlockSize;
        transHmoBlock->copyCount = customAttrBlockSize;
        transHmoBlock->offsetInBlock = 0;
        transHmoBlock->srcHmo = AllocHmo(customAttrLen * customAttrBlockSize);
        transHmoBlock->dstHmo = AllocHmo(customAttrLen * customAttrBlockSize);
        std::vector<uint8_t> customAttrs(customAttrBlockSize * customAttrLen);
        for (uint32_t i = 0; i < customAttrBlockSize; ++i) {
            for (uint32_t j = 0; j < customAttrLen; ++j) {
                customAttrs[i * customAttrLen + j] = i;
            }
        }
        WriteHmo(transHmoBlock->srcHmo, customAttrs);

        removeHmoBlock = std::make_shared<OckRemoveDataCustomAttrOpHmoBlock>();
        removeHmoBlock->srcHmo = AllocHmo(sizeof(uint64_t) * ids.size());
        removeHmoBlock->dstHmo = AllocHmo(sizeof(uint64_t) * ids.size());
        removeHmoBlock->removeSize = ids.size();
        removeHmoBlock->customAttrLen = customAttrLen;
        removeHmoBlock->customAttrBlockSize = customAttrBlockSize;

        std::vector<uint64_t> customAttrSrcAddr(ids.size());
        std::vector<uint64_t> customAttrDstAddr(ids.size());
        for (uint64_t i = 0; i < ids.size(); i++) {
            uint64_t srcIdx = customAttrBlockSize - i - 1;
            customAttrSrcAddr[i] = transHmoBlock->dstHmo->Addr() + srcIdx;
            customAttrDstAddr[i] = transHmoBlock->dstHmo->Addr() + ids[i];
        }
        WriteHmo(removeHmoBlock->srcHmo, customAttrSrcAddr);
        WriteHmo(removeHmoBlock->dstHmo, customAttrDstAddr);
    }

    bool CheckResults()
    {
        uint32_t customAttrBlockSize = transHmoBlock->customAttrBlockSize;
        uint32_t customAttrLen = transHmoBlock->customAttrLen;
        std::vector<uint8_t> res(customAttrBlockSize * customAttrLen);
        ReadHmo(transHmoBlock->dstHmo, res);
        for (uint32_t i = 0; i < customAttrLen; ++i) {
            for (uint32_t j = 0; j < ids.size(); ++j) {
                if (res[i * customAttrBlockSize + ids[j]] != customAttrBlockSize - 1U - j) {
                    return false;
                }
            }
        }
        return true;
    }

    std::shared_ptr<hcps::handler::OckHeteroHandler> handler;
    std::shared_ptr<OckHeteroStreamBase> streamBase;

    std::shared_ptr<OckRemoveDataCustomAttrOpHmoBlock> removeHmoBlock;
    std::shared_ptr<OckTransDataCustomAttrOpHmoBlock> transHmoBlock;

private:
    void BuildDeviceInfo()
    {
        deviceInfo = std::make_shared<hmm::OckHmmDeviceInfo>();
        deviceInfo->deviceId = 3U;
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
    std::vector<int64_t> ids = { 0, 10, 20 };
};
TEST_F(TestOckRemoveDataCustomAttrOpRun, run)
{
    PrepareTestData(5U, 36U);
    OckTransDataCustomAttrOpRun::AddTransCustomAttrOp(transHmoBlock, *handler, streamBase);
    streamBase->WaitExecComplete();
    auto op = OckRemoveDataCustomAttrOpRun::CreateOp(removeHmoBlock, *handler);
    streamBase->AddOp(op);
    streamBase->WaitExecComplete();

    // 检查算子运行结果
    EXPECT_EQ(CheckResults(), true);
}
} // namespace test
} // namespace nop
} // namespace hcps
} // namespace ock