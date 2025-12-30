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
#include "ock/hcps/algo/OckShape.h"
#include "ock/hcps/algo/impl/OckShapeImpl.h"
#include "ock/hcps/nop/remove_data_shaped_op/OckRemoveDataShapedOpRun.h"

namespace ock {
namespace hcps {
namespace nop {
namespace test {
class TestOckRemoveDataShapedOpRun : public testing::Test {
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

    void PrepareData()
    {
        srand(time(nullptr));
        hmoBlock = std::make_shared<OckRemoveDataShapedOpHmoBlock>();
        hmoBlock->removeCount = 4U;
        hmoBlock->dims = dims;
        hmoBlock->srcPosHmo = AllocHmo(hmoBlock->removeCount * sizeof(uint64_t));
        hmoBlock->dstPosHmo = AllocHmo(hmoBlock->removeCount * sizeof(uint64_t));

        unShapedData.resize(DEFAULT_CODE_BLOCK_SIZE * dims);
        for (uint32_t i = 0; i < unShapedData.size(); ++i) {
            unShapedData[i] = rand();
        }

        std::vector<int8_t> temp(unShapedData.size());
        algo::OckShape<> shape(reinterpret_cast<uintptr_t>(temp.data()), temp.size());
        shape.AddData(unShapedData.data(), DEFAULT_CODE_BLOCK_SIZE);
        shapedData = AllocHmo(temp.size());
        WriteHmo(shapedData, temp);

        std::vector<uint64_t> dstPos(hmoBlock->removeCount);
        for (uint32_t i = 0; i < dstPos.size(); ++i) {
            dstPos[i] = CalcAddressInHmo(shapedData, i);
        }
        WriteHmo(hmoBlock->dstPosHmo, dstPos);

        std::vector<uint64_t> srcPos(hmoBlock->removeCount);
        for (uint32_t i = 0; i < srcPos.size(); ++i) {
            srcPos[i] = CalcAddressInHmo(shapedData, DEFAULT_CODE_BLOCK_SIZE - 1U - i);
        }
        WriteHmo(hmoBlock->srcPosHmo, srcPos);
    }

    bool CheckResults()
    {
        auto buffer = shapedData->GetBuffer(hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0ULL,
            shapedData->GetByteSize());
        algo::OckShape<> shape(buffer->Address(), DEFAULT_CODE_BLOCK_SIZE, DEFAULT_CODE_BLOCK_SIZE);
        std::vector<int8_t> res(dims);
        for (uint32_t i = 0; i < hmoBlock->removeCount; ++i) {
            shape.GetData(i, res.data());
            for (uint32_t j = 0; j < dims; ++j) {
                if (res[j] != unShapedData[(DEFAULT_CODE_BLOCK_SIZE - 1 - i) * dims + j]) {
                    std::cout << "wrong result!" << std::endl;
                    return false;
                }
            }
        }
        return true;
    }

    std::shared_ptr<hcps::handler::OckHeteroHandler> handler;
    std::shared_ptr<OckHeteroStreamBase> streamBase;

    std::shared_ptr<OckRemoveDataShapedOpHmoBlock> hmoBlock;
    std::shared_ptr<hmm::OckHmmHMObject> shapedData;
    std::vector<int8_t> unShapedData;

    uint32_t dims = 256U;

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
    uint64_t CalcAddressInHmo(std::shared_ptr<hmm::OckHmmHMObject> hmo, uint32_t pos)
    {
        uintptr_t addr = hmo->Addr() + (pos / CUBE_ALIGN) * dims * CUBE_ALIGN * sizeof(int8_t) +
            (pos % CUBE_ALIGN) * CUBE_ALIGN_INT8;
        return static_cast<uint64_t>(addr);
    }
    std::shared_ptr<hmm::OckHmmDeviceInfo> deviceInfo;
};
TEST_F(TestOckRemoveDataShapedOpRun, run)
{
    PrepareData();
    auto op = OckRemoveDataShapedOpRun::GenRemoveShapedOp(hmoBlock, *handler);
    streamBase->AddOp(op);
    streamBase->WaitExecComplete();

    // 检查算子运行结果
    EXPECT_EQ(CheckResults(), true);
}
} // namespace test
} // namespace nop
} // namespace hcps
} // namespace ock