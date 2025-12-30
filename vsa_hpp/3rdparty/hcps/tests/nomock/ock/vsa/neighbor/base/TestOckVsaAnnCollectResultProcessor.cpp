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

#include <vector>
#include <iostream>
#include "gtest/gtest.h"
#include "acl/acl.h"
#include "ock/hcps/algo/OckShape.h"
#include "ock/hcps/algo/impl/OckShapeImpl.h"
#include "ock/hcps/nop/l2_norm_op/OckL2NormOpRun.h"
#include "ock/vsa/neighbor/base/OckVsaAnnCollectResultProcessor.h"
#include "ock/vsa/neighbor/npu/OckVsaAnnCollectResultNpuProcessor.h"
namespace ock {
namespace vsa {
namespace neighbor {
namespace adapter {
namespace test {
class TestOckVsaAnnCollectResultProcessor : public testing::Test {
public:
    void SetUp(void) override
    {
        BuildDeviceInfo();
        aclrtSetDevice(deviceInfo->deviceId);
        hmm::OckHmmErrorCode errorCode = hmm::HMM_SUCCESS;
        handler = hcps::handler::OckHeteroHandler::CreateSingleDeviceHandler(deviceInfo->deviceId, deviceInfo->cpuSet,
            deviceInfo->memorySpec, errorCode);
        streamBase = hcps::handler::helper::MakeStream(*handler, errorCode, hcps::OckDevStreamType::AI_CORE);
        queryCond.resize(256U, 1U);
        processor = OckVsaAnnCollectResultProcessor<int8_t, 256U>::CreateNPUProcessor(handler, queryCond, 10U);
        featureSet = OckVsaAnnFeatureSet::Create(false, 65536U);
        idxMap = hcps::hfo::OckOneSideIdxMap::Create(65536U, *handler->HmmMgrPtr());
    }

    void TearDown(void) override
    {
        idxMap.reset();
        featureSet.reset();
        processor.reset();
        streamBase.reset();
        handler.reset();
        aclrtResetDevice(deviceInfo->deviceId);
    }

    void PrepareHostData(uint32_t count)
    {
        hostData.resize(count * dims, 1U);
        for (uint32_t i = 0; i < 10U; ++i) {
            hostData[i * dims] = i + 1;
        }
        for (uint32_t i = 10U; i < dims; ++i) {
            for (uint32_t j = 0; j < dims; ++j) {
                hostData[i * dims + j] = i % (j + 1U);
            }
        }
        shapedData.resize(count * dims, 0U);
        uintptr_t addr = reinterpret_cast<uintptr_t>(shapedData.data());
        ock::hcps::algo::OckShape<> shape(addr, shapedData.size());
        shape.AddData(hostData.data(), count);
    }

    void PrepareData(uint32_t count)
    {
        auto hmoBlock = std::make_shared<hcps::nop::OckL2NormOpHmoBlock>();
        hmm::OckHmmErrorCode errorCode = hmm::HMM_SUCCESS;
        hmoBlock->dataBase = hcps::handler::helper::MakeDeviceHmo(*handler, count * dims, errorCode);
        hmoBlock->normResult = hcps::handler::helper::MakeDeviceHmo(*handler, count * sizeof(OckFloat16), errorCode);
        hmoBlock->dims = dims;
        hmoBlock->addNum = count;
        WriteHmo(hmoBlock->dataBase, hostData);
        hcps::nop::OckL2NormOpRun::ComputeNormSync(hmoBlock, *handler, streamBase);
        streamBase->WaitExecComplete();
        WriteHmo(hmoBlock->dataBase, shapedData);
        auto vsaFeature = OckVsaAnnFeature(hmoBlock->dataBase, hmoBlock->normResult, nullptr, count, count);
        featureSet->AddFeature(std::make_shared<OckVsaAnnFeature>(vsaFeature));
    }

    void PrepareIdxMap(uint32_t count)
    {
        idx.resize(count);
        for (uint64_t i = 0; i < idx.size(); ++i) {
            idx[i] = i;
        }
        idxMap->BatchAdd(idx.data(), idx.size());
    }

    std::shared_ptr<hcps::handler::OckHeteroHandler> handler;
    std::shared_ptr<hcps::OckHeteroStreamBase> streamBase;
    std::shared_ptr<OckVsaAnnCollectResultProcessor<int8_t, 256U>> processor;
    std::shared_ptr<OckVsaAnnFeatureSet> featureSet;
    std::shared_ptr<hcps::hfo::OckOneSideIdxMap> idxMap;

    std::vector<int8_t> queryCond;
    const uint32_t dims{ 256U };
    std::vector<int8_t> hostData;
    std::vector<int8_t> shapedData;
    std::vector<uint64_t> idx;

public:
    void BuildDeviceInfo()
    {
        deviceInfo = std::make_shared<hmm::OckHmmDeviceInfo>();
        deviceInfo->deviceId = 1U;
        CPU_SET(1U, &deviceInfo->cpuSet);                                                     // 设置1号CPU核
        CPU_SET(2U, &deviceInfo->cpuSet);                                                     // 设置2号CPU核
        deviceInfo->memorySpec.devSpec.maxDataCapacity = 8ULL * 1024ULL * 1024ULL * 1024ULL;  // 4G
        deviceInfo->memorySpec.devSpec.maxSwapCapacity = 3ULL * 64ULL * 1024ULL * 1024ULL;    // 3 * 64M
        deviceInfo->memorySpec.hostSpec.maxDataCapacity = 8ULL * 1024ULL * 1024ULL * 1024ULL; // 1G
        deviceInfo->memorySpec.hostSpec.maxSwapCapacity = 3ULL * 64ULL * 1024ULL * 1024ULL;   // 3 * 64M
        deviceInfo->transferThreadNum = 2ULL;                                                 // 2个线程
    }

    template <typename T> void WriteHmo(std::shared_ptr<hmm::OckHmmHMObject> hmo, const std::vector<T> &data)
    {
        aclrtMemcpy(reinterpret_cast<void *>(hmo->Addr()), hmo->GetByteSize(), data.data(), data.size() * sizeof(T),
            ACL_MEMCPY_HOST_TO_DEVICE);
    }

    std::shared_ptr<hmm::OckHmmDeviceInfo> deviceInfo;
};

TEST_F(TestOckVsaAnnCollectResultProcessor, get_top_n_results)
{
    PrepareHostData(65536U);
    PrepareData(65536U);
    PrepareIdxMap(65536U);
    OckVsaErrorCode errorCode = 0;
    EXPECT_EQ(processor->Init(), hmm::HMM_SUCCESS);
    auto res = processor->GetTopNResults(featureSet, 10U, errorCode);
    std::vector<std::vector<hcps::algo::FloatNode>> nodes = *res;
    for (uint32_t i = 0; i < 10U; ++i) {
        EXPECT_EQ(utils::SafeFloatEqual(nodes[0][i].distance, 1.0F, 0.0001F), true);
    }
}

TEST_F(TestOckVsaAnnCollectResultProcessor, notify_and_get_top_n_results)
{
    EXPECT_EQ(processor->Init(), hmm::HMM_SUCCESS);
    for (uint32_t i = 0; i < 3U; ++i) {
        PrepareHostData(65536U);
        PrepareData(65536U);
    }
    PrepareIdxMap(65536U);
    hmm::OckHmmErrorCode errorCode = hmm::HMM_SUCCESS;
    processor->NotifyResult(std::make_shared<OckVsaAnnFeature>(featureSet->GetFeature(0U)), idxMap, false, errorCode);
    processor->NotifyResult(std::make_shared<OckVsaAnnFeature>(featureSet->GetFeature(1U)), idxMap, false, errorCode);
    processor->NotifyResult(std::make_shared<OckVsaAnnFeature>(featureSet->GetFeature(2U)), idxMap, false, errorCode);
    EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
    auto res = processor->NotifyResultEnd(errorCode);
    EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
    for (uint32_t i = 0; i < res.size(); ++i) {
        EXPECT_EQ(utils::SafeFloatEqual(res[i].distance, 1.0F, 0.0001F), true);
    }
}

TEST_F(TestOckVsaAnnCollectResultProcessor, get_sample_cell_topN_result)
{
    EXPECT_EQ(processor->Init(), hmm::HMM_SUCCESS);
    uint32_t blockRowCount = 262144U;
    uint32_t blockNum = 64U;
    auto sampleMgr = relation::OckVsaSampleFeatureMgr<int8_t, 256U>(handler, blockRowCount * blockNum, blockRowCount);
    hmm::OckHmmErrorCode errorCode = hmm::HMM_SUCCESS;
    PrepareHostData(blockRowCount);
    for (uint32_t i = 0; i < blockNum; ++i) {
        auto hostUnShapedFeature =
            hcps::handler::helper::MakeHostHmo(*handler, hostData.size() * sizeof(int8_t), errorCode);
        WriteHmo(hostUnShapedFeature, hostData);
        auto hostShapedFeature =
            hcps::handler::helper::MakeHostHmo(*handler, shapedData.size() * sizeof(int8_t), errorCode);
        WriteHmo(hostShapedFeature, shapedData);
        auto hmoNormBlock =
            hcps::nop::OckL2NormOpRun::BuildNormHmoBlock(hostUnShapedFeature, *handler, 256U, blockRowCount, errorCode);
        EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
        hcps::nop::OckL2NormOpRun::ComputeNormSync(hmoNormBlock, *handler, streamBase);
        auto hostNorm = hcps::handler::helper::CopyToHostHmo(*handler, hmoNormBlock->normResult, errorCode);
        auto ret = sampleMgr.AddSampleFeature(hostShapedFeature, hostNorm, blockRowCount);
        EXPECT_EQ(ret, hmm::HMM_SUCCESS);
        EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
    }
    auto ret = processor->GetSampleCellTopNResult(sampleMgr, 10U, errorCode);
    auto distances = ret->TopDistance(0U);
    for (uint32_t i = 0; i < distances.size(); ++i) {
        EXPECT_EQ(utils::SafeFloatEqual(distances[i], 1.0F, 0.0001F), true);
    }
}
}
}
}
}
}