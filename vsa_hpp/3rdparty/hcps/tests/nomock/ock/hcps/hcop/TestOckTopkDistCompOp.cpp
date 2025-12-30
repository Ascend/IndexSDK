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
#include <gtest/gtest.h>
#include "acl/acl.h"
#include "ock/hmm/OckHmmFactory.h"
#include "ock/hmm/mgr/OckHmmHMObject.h"
#include "ock/acladapter/utils/OckAscendFp16.h"
#include "ock/utils/OckSafeUtils.h"
#include "ock/log/OckTestLoggerEx.h"
#include "ock/hcps/handler/OckHandlerHmmHelper.h"
#include "ock/hcps/hcop/topk_dist_comp_op/OckTopkDistCompOpRun.h"
namespace ock {
namespace hcps {
namespace hcop {
namespace test {
class TestOckTopkDistCompOp : public testing::Test {
public:
    TestOckTopkDistCompOp(void) {}
    void SetUp(void) override
    {
        BuildDeviceInfo();
        OCK_CHECK_ERRORCODE_MSG(aclrtSetDevice(deviceInfo->deviceId), "acl set device");
        hmm::OckHmmErrorCode errorCode = hmm::HMM_SUCCESS;
        handler = handler::OckHeteroHandler::CreateSingleDeviceHandler(
            deviceInfo->deviceId, deviceInfo->cpuSet, deviceInfo->memorySpec, errorCode);
        streamBase = handler::helper::MakeStream(*handler, errorCode, OckDevStreamType::AI_CPU);
    }
    void TearDown(void) override
    {
        OCK_CHECK_ERRORCODE_MSG(aclrtResetDevice(deviceInfo->deviceId), "acl reset device");
    }

    void PrepareData(uint64_t ntotal, uint32_t totalNumBlocks,
        uint64_t blockSize = nop::FLAT_DEFAULT_DIST_COMPUTE_BATCH, uint64_t batch = 1)
    {
        hmoGroup = std::make_shared<OckTopkDistCompOpHmoGroup>(false, batch, dims, k, blockSize, totalNumBlocks,
            static_cast<int64_t>(ntotal), 0U);
        hmoGroup->SetQueryHmos(AllocHmo(batch * dims * sizeof(int8_t)),
                               AllocHmo(utils::SafeRoundUp(batch, nop::FP16_ALIGN) * sizeof(OckFloat16)),
                               nullptr);
        hmoGroup->SetOutputHmos(AllocHmo(batch * k * sizeof(OckFloat16)),
                                AllocHmo(batch * k * sizeof(int64_t)));
        for (uint64_t i = 0; i < totalNumBlocks; ++i) {
            hmoGroup->PushDataBase(AllocHmo(blockSize * dims * sizeof(int8_t)),
                                   AllocHmo(blockSize * sizeof(OckFloat16)));
        }

        std::vector<std::vector<int8_t>> base(totalNumBlocks);
        std::vector<std::vector<OckFloat16>> baseNorm(totalNumBlocks);
        for (uint64_t i = 0; i < totalNumBlocks; ++i) {
            base[i].resize(blockSize * dims, 2);                                            // 2: 测试特例
            baseNorm[i].resize(blockSize, acladapter::OckAscendFp16::FloatToFp16(0.03125)); // 0.03125：测试特例
            WriteHmo<int8_t>(hmoGroup->featuresHmo[i], base[i]);
            WriteHmo<OckFloat16>(hmoGroup->normsHmo[i], baseNorm[i]);
        }
        std::vector<int8_t> queries(batch * dims, 1);
        std::vector<OckFloat16> queriesNorm((batch + nop::FP16_ALIGN - 1U) / nop::FP16_ALIGN * nop::FP16_ALIGN,
            acladapter::OckAscendFp16::FloatToFp16(0.0625)); // 0.0625：测试查询向量特例
        WriteHmo<int8_t>(hmoGroup->queriesHmo, queries);
        WriteHmo<OckFloat16>(hmoGroup->queriesNormHmo, queriesNorm);
    }

    void CheckResults(uint64_t batch = 1)
    {
        std::vector<OckFloat16> outDists(batch * k);
        ReadHmo(hmoGroup->topkDistsHmo, outDists);
        std::cout << "acladapter::OckAscendFp16::Fp16ToFloat(outDists[0]) = "
        << acladapter::OckAscendFp16::Fp16ToFloat(outDists[0]) << std::endl;
        EXPECT_EQ(utils::SafeFloatEqual(acladapter::OckAscendFp16::Fp16ToFloat(outDists[0]), sampleDist, fp16Acc),
            true);
    }

    std::shared_ptr<handler::OckHeteroHandler> handler;
    std::shared_ptr<OckHeteroStreamBase> streamBase;

    // 调用算子时所需超参
    uint64_t dims{ 256 };
    uint64_t k{ 10 }; // 10: topk的k值

    float sampleDist = 0.01; // 构造用例的距离值全是0.01
    double fp16Acc = 0.0001; //  fp16精度比double低
    std::shared_ptr<OckTopkDistCompOpHmoGroup> hmoGroup;

private:
    void BuildDeviceInfo()
    {
        deviceInfo = std::make_shared<hmm::OckHmmDeviceInfo>();
        deviceInfo->deviceId = 3U;
        CPU_SET(1U, &deviceInfo->cpuSet);                                                   // 设置1号CPU核
        CPU_SET(2U, &deviceInfo->cpuSet);                                                   // 设置2号CPU核
        deviceInfo->memorySpec.devSpec.maxDataCapacity = 4U * 1024ULL * 1024ULL * 1024ULL;  // 4G
        deviceInfo->memorySpec.devSpec.maxSwapCapacity = 8ULL * 64ULL * 1024ULL * 1024ULL;  // 8 * 64M
        deviceInfo->memorySpec.hostSpec.maxDataCapacity = 1024ULL * 1024ULL * 1024ULL;      // 1G
        deviceInfo->memorySpec.hostSpec.maxSwapCapacity = 3ULL * 64ULL * 1024ULL * 1024ULL; // 8 * 64M
        deviceInfo->transferThreadNum = 2ULL;                                               // 2个线程
    }
    std::shared_ptr<hmm::OckHmmHMObject> AllocHmo(int64_t byteSize)
    {
        return handler->HmmMgrPtr()->Alloc(byteSize, hmm::OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY).second;
    }
    template <typename T> void WriteHmo(const std::shared_ptr<hmm::OckHmmSubHMObject> &hmo, const std::vector<T> &vec)
    {
        EXPECT_EQ(hmo->GetByteSize(), vec.size() * sizeof(T));
        aclrtMemcpy(reinterpret_cast<void *>(hmo->Addr()), hmo->GetByteSize(), vec.data(), vec.size() * sizeof(T),
            ACL_MEMCPY_HOST_TO_DEVICE);
    }
    template <typename T> void ReadHmo(const std::shared_ptr<hmm::OckHmmSubHMObject> &hmo, std::vector<T> &vec)
    {
        EXPECT_EQ(hmo->GetByteSize(), vec.size() * sizeof(T));
        aclrtMemcpy(vec.data(), vec.size() * sizeof(T), reinterpret_cast<void *>(hmo->Addr()), hmo->GetByteSize(),
            ACL_MEMCPY_DEVICE_TO_HOST);
    }

    std::shared_ptr<hmm::OckHmmDeviceInfo> deviceInfo;
};

TEST_F(TestOckTopkDistCompOp, run)
{
    uint64_t ntotal = nop::FLAT_DEFAULT_DIST_COMPUTE_BATCH * nop::DEFAULT_PAGE_BLOCK_NUM;
    uint32_t totalNumBlocks = nop::DEFAULT_PAGE_BLOCK_NUM;
    uint64_t blockSize = nop::FLAT_DEFAULT_DIST_COMPUTE_BATCH;
    uint64_t batch = 1;
    PrepareData(ntotal, totalNumBlocks, blockSize, batch);
    OckTopkDistCompOpRun::RunOneGroupSync(hmoGroup, streamBase, handler);
    // 检查算子运行结果
    CheckResults(batch);
}

TEST_F(TestOckTopkDistCompOp, run_batch_32_last_page_not_full)
{
    uint32_t totalNumBlocks = 33U;
    uint64_t blockSize = nop::FLAT_DEFAULT_DIST_COMPUTE_BATCH;
    uint64_t ntotal = totalNumBlocks * blockSize;
    uint64_t batch = 32U;
    PrepareData(ntotal, totalNumBlocks, blockSize, batch);
    OckTopkDistCompOpRun::RunOneGroupSync(hmoGroup, streamBase, handler);
    // 检查算子运行结果
    CheckResults(batch);
}

TEST_F(TestOckTopkDistCompOp, run_batch_32_last_block_not_full)
{
    uint32_t totalNumBlocks = 34U;
    uint64_t blockSize = nop::FLAT_DEFAULT_DIST_COMPUTE_BATCH;
    uint64_t ntotal = totalNumBlocks * blockSize - 1U;
    uint64_t batch = 32U;
    PrepareData(ntotal, totalNumBlocks, blockSize, batch);
    OckTopkDistCompOpRun::RunOneGroupSync(hmoGroup, streamBase, handler);
    // 检查算子运行结果
    CheckResults(batch);
}

TEST_F(TestOckTopkDistCompOp, run_65536_batch_32_last_block_not_full)
{
    uint32_t totalNumBlocks = 34U;
    uint64_t blockSize = 65536U;
    uint64_t ntotal = totalNumBlocks * blockSize - 1U;
    uint64_t batch = 32U;
    PrepareData(ntotal, totalNumBlocks, blockSize, batch);
    OckTopkDistCompOpRun::RunOneGroupSync(hmoGroup, streamBase, handler);
    // 检查算子运行结果
    CheckResults(batch);
}
} // namespace test
} // namespace hcop
} // namespace hcps
} // namespace ock
