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
#include "ock/hcps/handler/OckHeteroHandler.h"
#include "ock/hcps/handler/OckHandlerHmmHelper.h"
#include "ock/hcps/stream/OckHeteroStreamBase.h"
#include "ock/hcps/nop/dist_mask_with_extra_gen_op/OckDistMaskWithExtraGenOpRun.h"

namespace ock {
namespace hcps {
namespace nop {
namespace test {
namespace {
const uint32_t TOKEN_NUM = 2500U;
}
class TestOckDistMaskWithExtraGenOpRun : public testing::Test {
public:
    void SetUp(void) override
    {
        BuildDeviceInfo();
        aclrtSetDevice(deviceInfo->deviceId);
        hmm::OckHmmErrorCode errorCode = hmm::HMM_SUCCESS;
        handler = handler::OckHeteroHandler::CreateSingleDeviceHandler(deviceInfo->deviceId, deviceInfo->cpuSet,
            deviceInfo->memorySpec, errorCode);
        streamBase = handler::helper::MakeStream(*handler, errorCode, OckDevStreamType::AI_CORE);
    }

    void TearDown(void) override
    {
        aclrtResetDevice(deviceInfo->deviceId);
    }

    void PrepareHostData(void)
    {
        times.resize(DEFAULT_CODE_BLOCK_SIZE * DEFAULT_GROUP_BLOCK_NUM, 0);
        for (uint32_t i = 0; i < times.size(); ++i) {
            times[i] = i % 4U;
        }
        tokenIdQs.resize(DEFAULT_CODE_BLOCK_SIZE * DEFAULT_GROUP_BLOCK_NUM, 0);
        tokenIdRs.resize(DEFAULT_CODE_BLOCK_SIZE * DEFAULT_GROUP_BLOCK_NUM * OPS_DATA_TYPE_TIMES, 64U);
        for (int32_t i = 0; i < DEFAULT_CODE_BLOCK_SIZE * DEFAULT_GROUP_BLOCK_NUM; ++i) {
            tokenIdRs[OPS_DATA_TYPE_TIMES * i] = 1 << (i % 8U);
        }
        queryTimesHost.resize(OPS_DATA_TYPE_ALIGN, 0);
        queryTimesHost[0] = 0;
        queryTimesHost[1] = 0 - 10U;
        queryTokenIdsHost.resize(utils::SafeDivUp(TOKEN_NUM, OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES, 0U);
        for (uint32_t i = 0; i < utils::SafeDivUp(TOKEN_NUM, OPS_DATA_TYPE_ALIGN); ++i) {
            queryTokenIdsHost[i * OPS_DATA_TYPE_TIMES + 1] = 64U;
        }
        queryTokenIdsHost[0] = 1U;
    }

    void PrepareData()
    {
        maskHmoGroups = std::make_shared<OckDistMaskWithExtraGenOpHmoGroups>();
        for (uint32_t i = 0; i < 2U; ++i) {
            auto timeHmo = AllocHmo(DEFAULT_CODE_BLOCK_SIZE * DEFAULT_GROUP_BLOCK_NUM * sizeof(int32_t));
            auto tokenQsHmo = AllocHmo(DEFAULT_CODE_BLOCK_SIZE * DEFAULT_GROUP_BLOCK_NUM * sizeof(int32_t));
            auto tokenRsHmo =
                AllocHmo(DEFAULT_CODE_BLOCK_SIZE * DEFAULT_GROUP_BLOCK_NUM * OPS_DATA_TYPE_TIMES * sizeof(uint8_t));
            WriteHmo(timeHmo, times);
            WriteHmo(tokenQsHmo, tokenIdQs);
            WriteHmo(tokenRsHmo, tokenIdRs);
            maskHmoGroups->attrTimes.emplace_back(
                hmm::OckHmmHMObject::CreateSubHmoList(timeHmo, timeHmo->GetByteSize()));
            maskHmoGroups->attrTokenQuotients.emplace_back(
                hmm::OckHmmHMObject::CreateSubHmoList(tokenQsHmo, tokenQsHmo->GetByteSize()));
            maskHmoGroups->attrTokenRemainders.emplace_back(
                hmm::OckHmmHMObject::CreateSubHmoList(tokenRsHmo, tokenRsHmo->GetByteSize()));
        }
        uint32_t batch = 2U;
        for (uint32_t i = 0; i < batch; ++i) {
            auto queryTimeHmo = AllocHmo(OPS_DATA_TYPE_ALIGN * sizeof(int32_t));
            auto queryTokenIdHmo = AllocHmo(utils::SafeDivUp(2500U, OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES);
            WriteHmo(queryTimeHmo, queryTimesHost);
            WriteHmo(queryTokenIdHmo, queryTokenIdsHost);
            maskHmoGroups->queryTimes.emplace_back(queryTimeHmo);
            maskHmoGroups->queryTokenIds.emplace_back(queryTokenIdHmo);
        }

        maskHmoGroups->mask =
            AllocHmo(((DEFAULT_CODE_BLOCK_SIZE * DEFAULT_GROUP_BLOCK_NUM * 2U) / OPS_DATA_TYPE_ALIGN) * batch);
        maskHmoGroups->extraMask =
            AllocHmo(((DEFAULT_CODE_BLOCK_SIZE * DEFAULT_GROUP_BLOCK_NUM * 2U) / OPS_DATA_TYPE_ALIGN) * batch);
        aclrtMemset(reinterpret_cast<void *>(maskHmoGroups->extraMask->Addr()), maskHmoGroups->extraMask->GetByteSize(),
            0U, maskHmoGroups->extraMask->GetByteSize());
        extraMask.resize((DEFAULT_CODE_BLOCK_SIZE * DEFAULT_GROUP_BLOCK_NUM * 2U) / OPS_DATA_TYPE_ALIGN * batch, 1U);
        WriteHmo(maskHmoGroups->extraMask, extraMask);
        maskHmoGroups->tokenNum = TOKEN_NUM;
        maskHmoGroups->featureAttrBlockSize = DEFAULT_CODE_BLOCK_SIZE;
        maskHmoGroups->blockCount = DEFAULT_GROUP_BLOCK_NUM;
        maskHmoGroups->maskLen = (DEFAULT_CODE_BLOCK_SIZE * DEFAULT_GROUP_BLOCK_NUM * 2U) / OPS_DATA_TYPE_ALIGN;
    }

    bool CheckResults(std::shared_ptr<OckDistMaskWithExtraGenOpHmoGroups> hmoGroups)
    {
        std::vector<uint8_t> resMask(hmoGroups->maskLen * 2U, 0U);
        ReadHmo(hmoGroups->mask, resMask);
        for (uint32_t i = 0; i < resMask.size(); ++i) {
            if (resMask[i] != 1U) {
                std::cout << i << ", " << resMask[i] << std::endl;
                return false;
            }
        }
        return true;
    }

    std::shared_ptr<handler::OckHeteroHandler> handler;
    std::shared_ptr<OckHeteroStreamBase> streamBase;

    std::shared_ptr<OckDistMaskWithExtraGenOpHmoGroups> maskHmoGroups;
    std::vector<int32_t> times;
    std::vector<int32_t> tokenIdQs;
    std::vector<uint8_t> tokenIdRs;
    std::vector<int32_t> queryTimesHost;
    std::vector<uint8_t> queryTokenIdsHost;
    std::vector<uint8_t> extraMask;

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
    template <typename T> void WriteHmo(const std::shared_ptr<hmm::OckHmmSubHMObject> &hmo, const std::vector<T> &vec)
    {
        aclrtMemcpy(reinterpret_cast<void *>(hmo->Addr()), hmo->GetByteSize(), vec.data(), vec.size() * sizeof(T),
            ACL_MEMCPY_HOST_TO_DEVICE);
    }
    template <typename T> void ReadHmo(const std::shared_ptr<hmm::OckHmmSubHMObject> &hmo, std::vector<T> &vec)
    {
        aclrtMemcpy(vec.data(), vec.size() * sizeof(T), reinterpret_cast<void *>(hmo->Addr()), vec.size() * sizeof(T),
            ACL_MEMCPY_DEVICE_TO_HOST);
    }

    std::shared_ptr<hmm::OckHmmDeviceInfo> deviceInfo;
};
TEST_F(TestOckDistMaskWithExtraGenOpRun, run)
{
    PrepareHostData();
    PrepareData();
    OckDistMaskWithExtraGenOpRun::AddMaskWithExtraOpsMultiBatches(maskHmoGroups, streamBase);
    streamBase->WaitExecComplete();
    EXPECT_EQ(CheckResults(maskHmoGroups), true);
}
} // namespace test
} // namespace nop
} // namespace hcps
} // namespace ock
