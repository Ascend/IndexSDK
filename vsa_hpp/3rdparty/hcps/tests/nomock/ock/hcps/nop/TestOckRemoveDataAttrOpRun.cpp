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
#include <ctime>
#include "gtest/gtest.h"
#include "acl/acl.h"
#include "ock/utils/OckSafeUtils.h"
#include "ock/hcps/handler/OckHandlerHmmHelper.h"
#include "ock/hcps/stream/OckHeteroStreamBase.h"
#include "ock/hcps/nop/remove_data_attr_op/OckRemoveDataAttrOpRun.h"

namespace ock {
namespace hcps {
namespace nop {
namespace test {
class TestOckRemoveDataAttrOpRun : public testing::Test {
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

    void PrepareAttrData()
    {
        attrTime = AllocHmo(DEFAULT_CODE_BLOCK_SIZE * sizeof(int32_t));
        std::vector<int32_t> times(DEFAULT_CODE_BLOCK_SIZE);
        for (uint32_t i = 0; i < times.size(); ++i) {
            times[i] = i % 8U;
        }
        WriteHmo(attrTime, times);

        attrTokenQs = AllocHmo(DEFAULT_CODE_BLOCK_SIZE * sizeof(int32_t));
        std::vector<int32_t> attrQs(DEFAULT_CODE_BLOCK_SIZE);
        for (uint32_t i = 0; i < attrQs.size(); ++i) {
            attrQs[i] = i % 8U;
        }
        WriteHmo(attrTokenQs, attrQs);

        attrTokenRs = AllocHmo(DEFAULT_CODE_BLOCK_SIZE * sizeof(uint8_t) * OPS_DATA_TYPE_TIMES);
        std::vector<uint8_t> attrRs(DEFAULT_CODE_BLOCK_SIZE * OPS_DATA_TYPE_TIMES, 64U);
        for (uint32_t i = 0; i < DEFAULT_CODE_BLOCK_SIZE; ++i) {
            attrRs[i * OPS_DATA_TYPE_TIMES] = i % 8U;
        }
        WriteHmo(attrTokenRs, attrRs);
    }

    void PrepareData()
    {
        timeHmoBlock = std::make_shared<OckRemoveDataAttrOpHmoBlock>();
        timeHmoBlock->removeSize = 4U;
        timeHmoBlock->dataType = Type::INT32;
        timeHmoBlock->copyNum = 1U;
        timeHmoBlock->srcHmo = AllocHmo(timeHmoBlock->removeSize * sizeof(uint64_t));
        timeHmoBlock->dstHmo = AllocHmo(timeHmoBlock->removeSize * sizeof(uint64_t));
        std::vector<uint64_t> dstAddress(4U);
        for (uint32_t i = 0; i < dstAddress.size(); ++i) {
            dstAddress[i] = attrTime->Addr() + i * sizeof(int32_t);
        }
        WriteHmo(timeHmoBlock->dstHmo, dstAddress);
        std::vector<uint64_t> srcAddress(4U);
        for (uint32_t i = 0; i < srcAddress.size(); ++i) {
            srcAddress[i] = attrTime->Addr() + (DEFAULT_CODE_BLOCK_SIZE - 1 - i) * sizeof(int32_t);
        }
        WriteHmo(timeHmoBlock->srcHmo, srcAddress);

        qsHmoBlock = std::make_shared<OckRemoveDataAttrOpHmoBlock>();
        qsHmoBlock->removeSize = 4U;
        qsHmoBlock->dataType = Type::INT32;
        qsHmoBlock->copyNum = 1U;
        qsHmoBlock->srcHmo = AllocHmo(qsHmoBlock->removeSize * sizeof(uint64_t));
        qsHmoBlock->dstHmo = AllocHmo(qsHmoBlock->removeSize * sizeof(uint64_t));
        for (uint32_t i = 0; i < dstAddress.size(); ++i) {
            dstAddress[i] = attrTokenQs->Addr() + i * sizeof(int32_t);
        }
        WriteHmo(qsHmoBlock->dstHmo, dstAddress);
        for (uint32_t i = 0; i < srcAddress.size(); ++i) {
            srcAddress[i] = attrTokenQs->Addr() + (DEFAULT_CODE_BLOCK_SIZE - 1 - i) * sizeof(int32_t);
        }
        WriteHmo(qsHmoBlock->srcHmo, srcAddress);

        rsHmoBlock = std::make_shared<OckRemoveDataAttrOpHmoBlock>();
        rsHmoBlock->removeSize = 4U;
        rsHmoBlock->dataType = Type::UINT8;
        rsHmoBlock->copyNum = 2U;
        rsHmoBlock->srcHmo = AllocHmo(rsHmoBlock->removeSize * sizeof(uint64_t));
        rsHmoBlock->dstHmo = AllocHmo(rsHmoBlock->removeSize * sizeof(uint64_t));
        for (uint32_t i = 0; i < dstAddress.size(); ++i) {
            dstAddress[i] = attrTokenRs->Addr() + i * sizeof(uint8_t) * OPS_DATA_TYPE_TIMES;
        }
        WriteHmo(rsHmoBlock->dstHmo, dstAddress);
        for (uint32_t i = 0; i < srcAddress.size(); ++i) {
            srcAddress[i] =
                attrTokenRs->Addr() + (DEFAULT_CODE_BLOCK_SIZE - 1 - i) * sizeof(uint8_t) * OPS_DATA_TYPE_TIMES;
        }
        WriteHmo(rsHmoBlock->srcHmo, srcAddress);
    }

    bool CheckResults()
    {
        std::vector<int32_t> times(timeHmoBlock->removeSize);
        ReadHmo(attrTime, times);
        for (uint32_t i = 0; i < times.size(); ++i) {
            if (times[i] != static_cast<int32_t>(7U - i)) {
                return false;
            }
        }

        std::vector<int32_t> qs(qsHmoBlock->removeSize);
        ReadHmo(attrTokenQs, qs);
        for (uint32_t i = 0; i < qs.size(); ++i) {
            if (qs[i] != static_cast<int32_t>(7U - i)) {
                return false;
            }
        }

        std::vector<uint8_t> rs(rsHmoBlock->removeSize * OPS_DATA_TYPE_TIMES);
        ReadHmo(attrTokenRs, rs);
        for (uint32_t i = 0; i < rsHmoBlock->removeSize; ++i) {
            if (rs[i * OPS_DATA_TYPE_TIMES] != static_cast<uint8_t>(7U - i)) {
                return false;
            }
            if (rs[(i * OPS_DATA_TYPE_TIMES) + 1] != static_cast<uint8_t>(64U)) {
                return false;
            }
        }
        return true;
    }

    std::shared_ptr<hcps::handler::OckHeteroHandler> handler;
    std::shared_ptr<OckHeteroStreamBase> streamBase;

    std::shared_ptr<hmm::OckHmmHMObject> attrTime;
    std::shared_ptr<hmm::OckHmmHMObject> attrTokenQs;
    std::shared_ptr<hmm::OckHmmHMObject> attrTokenRs;

    std::shared_ptr<OckRemoveDataAttrOpHmoBlock> timeHmoBlock;
    std::shared_ptr<OckRemoveDataAttrOpHmoBlock> qsHmoBlock;
    std::shared_ptr<OckRemoveDataAttrOpHmoBlock> rsHmoBlock;

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
TEST_F(TestOckRemoveDataAttrOpRun, run)
{
    PrepareAttrData();
    PrepareData();
    auto op = OckRemoveDataAttrOpRun::CreateOp(timeHmoBlock, *handler);
    streamBase->AddOp(op);
    streamBase->WaitExecComplete();
    op = OckRemoveDataAttrOpRun::CreateOp(qsHmoBlock, *handler);
    streamBase->AddOp(op);
    streamBase->WaitExecComplete();
    op = OckRemoveDataAttrOpRun::CreateOp(rsHmoBlock, *handler);
    streamBase->AddOp(op);
    streamBase->WaitExecComplete();

    // 检查算子运行结果
    EXPECT_EQ(CheckResults(), true);
}
} // namespace test
} // namespace nop
} // namespace hcps
} // namespace ock