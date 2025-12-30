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

#include <gtest/gtest.h>
#include "ptest/ptest.h"
#include "ock/vsa/neighbor/hpp/OckVsaAnnHPPIndex.h"
#include "ock/vsa/neighbor/hpp/WithEnvOckVsaHPPIndex.h"
#include "ock/vsa/neighbor/base/OckVsaAnnCollectResultProcessor.h"
#include "ock/vsa/neighbor/MockOckVsaAnnCollectResultProcessor.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
class TestOckVsaHPPKernel : public WithEnvOckVsaHPPIndex<testing::Test> {
public:
    using BaseT = WithEnvOckVsaHPPIndex<testing::Test>;
    using OckVsaAnnSingleBatchQueryConditionT =
        OckVsaAnnSingleBatchQueryCondition<int8_t, 256UL, attr::OckTimeSpaceAttrTrait>;
    using OckVsaAnnCollectResultProcessorT = adapter::OckVsaAnnCollectResultProcessor<int8_t, 256UL>;

    void SetUp(void) override
    {
        nullRawSearchOp = hcps::OckSimpleHeteroOperator<acladapter::OckTaskResourceType::HOST_CPU>::Create(
            [](hcps::OckHeteroStreamContext &) { return hmm::HMM_SUCCESS; });
        BaseT::SetUp();
    }
    void TearDown(void) override
    {
        BaseT::TearDown();
    }

    std::shared_ptr<npu::OckVsaAnnRawBlockInfo> BuildBlockInfo(void)
    {
        auto ret = std::make_shared<npu::OckVsaAnnRawBlockInfo>();
        ret->feature = hcps::handler::helper::MakeHostHmo(*handler,
            this->param->GroupRowCount() * BaseT::OckVsaHPPKernelExtT::DimSize() *
                sizeof(BaseT::OckVsaHPPKernelExtT::DataT),
            this->errorCode);
        ret->norm = hcps::handler::helper::MakeHostHmo(
            *handler, this->param->GroupRowCount() * BaseT::OckVsaHPPKernelExtT::NormTypeByteSize(), this->errorCode);
        ret->rowCount = this->param->GroupRowCount();
        ret->keyAttrTime = hcps::handler::helper::MakeHostHmo(
            *handler, this->param->GroupRowCount() * sizeof(uint32_t), this->errorCode);
        ret->keyAttrQuotient = hcps::handler::helper::MakeHostHmo(
            *handler, this->param->GroupRowCount() * sizeof(uint16_t), this->errorCode);
        ret->keyAttrRemainder = hcps::handler::helper::MakeHostHmo(
            *handler, this->param->GroupRowCount() * sizeof(uint16_t), this->errorCode);
        ret->extKeyAttr = hcps::handler::helper::MakeHostHmo(
            *handler, this->param->GroupRowCount() * this->param->ExtKeyAttrByteSize(), this->errorCode);
        return ret;
    }
    void BuildLables(void)
    {
        outterLabels.reserve(this->param->GroupRowCount());
        for (uint32_t i = 0; i < this->param->GroupRowCount(); ++i) {
            outterLabels.push_back(i);
        }
    }
    std::deque<std::shared_ptr<hcps::hfo::OckTokenIdxMap>>& BuildTokenIdxMap(void)
    {
        std::deque<std::shared_ptr<hcps::hfo::OckTokenIdxMap>> tokenIdxMapDeque;

        for (uint32_t i = 0; i < this->param->MaxGroupCount(); ++i) {
            auto tokenIdxMap = hcps::hfo::OckTokenIdxMap::Create(this->param->TokenNum(), this->handler->HmmMgr(), 0UL);
            for (uint32_t j = 0; j < this->param->GroupRowCount(); ++j) {
                tokenIdxMap->AddData(j % this->param->TokenNum(), j);
            }
        }

        return tokenIdxMapDeque;
    }
    void AddMaskData(std::vector<std::shared_ptr<hmm::OckHmmSubHMObject>> &maskData, bool setAll = false)
    {
        maskData.push_back(hcps::handler::helper::MakeDeviceHmo(
            *handler, utils::SafeDivUp(this->param->GroupRowCount(), __CHAR_BIT__), errorCode));
        if (setAll) {
            memset_s(reinterpret_cast<uint8_t *>(maskData.back()->Addr()),
                maskData.back()->GetByteSize(),
                0xFF,
                maskData.back()->GetByteSize());
        } else {
            memset_s(reinterpret_cast<uint8_t *>(maskData.back()->Addr()),
                maskData.back()->GetByteSize(),
                0,
                maskData.back()->GetByteSize());
        }
    }
    void AddPartitialMaskData(std::vector<std::shared_ptr<hmm::OckHmmSubHMObject>> &maskData)
    {
        maskData.push_back(hcps::handler::helper::MakeDeviceHmo(
            *handler, utils::SafeDivUp(this->param->GroupRowCount(), __CHAR_BIT__), errorCode));
        uint64_t *pBeginAddr = reinterpret_cast<uint64_t *>(maskData.back()->Addr());
        uint64_t *pEndAddr =
            pBeginAddr + utils::SafeDivUp(this->param->GroupRowCount(), sizeof(uint64_t) * __CHAR_BIT__);
        for (uint64_t *pAddr = pBeginAddr; pAddr < pEndAddr; ++pAddr) {
            *pAddr = 0x0FULL;
        }
    }
    OckVsaAnnSingleBatchQueryConditionT BuildQueryCondition(void)
    {
        queryFeature = std::vector<int8_t>(256UL);
        filterTrait.bitSet.SetAll();
        extraMask = std::vector<uint8_t>(utils::SafeDivUp(this->param->GroupRowCount(), __CHAR_BIT__));
        return OckVsaAnnSingleBatchQueryConditionT(0UL,
            queryFeature.data(),
            &filterTrait,
            true,
            topk,
            extraMask.data(),
            utils::SafeDivUp(this->param->GroupRowCount(), __CHAR_BIT__),
            false,
            true);
    }
    void BuildAndTestIndex(void)
    {
        this->InitIndex();
        ASSERT_EQ(this->errorCode, hmm::HMM_SUCCESS);
        ASSERT_TRUE(index.get() != nullptr);
        ASSERT_TRUE(hppIndex != nullptr);
        ASSERT_TRUE(hppKernel != nullptr);
    }
    void TestAddFeature(void)
    {
        auto blockInfo = this->BuildBlockInfo();
        auto tokenIdxMap = BuildTokenIdxMap();

        auto addFeatureTime = fast::hdt::TestTimeGuard();
        auto relHmoGroup = std::make_shared<relation::OckVsaNeighborRelationHmoGroup>();
        relHmoGroup->validateRowCount = 1UL;
        EXPECT_EQ(hppKernel->AddFeature(*blockInfo, outterLabels, tokenIdxMap, relHmoGroup), hmm::HMM_SUCCESS);
        OCK_VSA_HPP_LOG_INFO("AddFeature Used Time=" << addFeatureTime.ElapsedMicroSeconds() << "us");
    }
    void QueryAllExcludeByTSCondition(void)
    {
        std::vector<std::shared_ptr<hmm::OckHmmSubHMObject>> maskData;
        AddMaskData(maskData);

        OckFloatTopNQueue topNResult(topk);
        auto timeGuard = fast::hdt::TestTimeGuard();
        OckVsaHPPSearchStaticInfo stInfo;
        EXPECT_EQ(
            hppKernel->Search(BuildQueryCondition(), maskData, nullRawSearchOp, topNResult, stInfo), hmm::HMM_SUCCESS);

        OCK_VSA_HPP_LOG_INFO("AllExclude Search Used Time=" << timeGuard.ElapsedMicroSeconds() << "us");
    }
    void QueryAllIncludeByTsCondition(void)
    {
        std::vector<std::shared_ptr<hmm::OckHmmSubHMObject>> maskData;
        AddMaskData(maskData, true);
        OckFloatTopNQueue topNResult(topk);

        auto timeGuard = fast::hdt::TestTimeGuard();
        filterTrait.minTime = std::numeric_limits<int32_t>::min();
        filterTrait.maxTime = std::numeric_limits<int32_t>::max();
        filterTrait.bitSet.SetAll();

        OckVsaHPPSearchStaticInfo stInfo;
        EXPECT_EQ(
            hppKernel->Search(BuildQueryCondition(), maskData, nullRawSearchOp, topNResult, stInfo), hmm::HMM_SUCCESS);

        OCK_VSA_HPP_LOG_INFO("AllInclude Search Used Time=" << timeGuard.ElapsedMicroSeconds() << "us");
    }
    void QueryPartitialIncludeByTsCondition(void)
    {
        std::vector<std::shared_ptr<hmm::OckHmmSubHMObject>> maskData;
        AddPartitialMaskData(maskData);
        OckFloatTopNQueue topNResult(topk);

        auto timeGuard = fast::hdt::TestTimeGuard();
        filterTrait.minTime = std::numeric_limits<int32_t>::min();
        filterTrait.maxTime = std::numeric_limits<int32_t>::max();
        filterTrait.bitSet.Set(0);

        OckVsaHPPSearchStaticInfo stInfo;
        EXPECT_EQ(
            hppKernel->Search(BuildQueryCondition(), maskData, nullRawSearchOp, topNResult, stInfo), hmm::HMM_SUCCESS);

        OCK_VSA_HPP_LOG_INFO("Partitial Include Search Used Time=" << timeGuard.ElapsedMicroSeconds() << "us");
    }
    void GetFeatureByLabelTest(void)
    {
        std::vector<int8_t> feature(256UL);
        EXPECT_EQ(this->hppKernel->GetFeatureByLabel(outterLabels[0], feature.data()), hmm::HMM_SUCCESS);

        attr::OckTimeSpaceAttr tsAttr;
        EXPECT_EQ(this->hppKernel->GetFeatureAttrByLabel(
            outterLabels[0], reinterpret_cast<attr::OckTimeSpaceAttrTrait::KeyTypeTuple *>(&tsAttr)),
            hmm::HMM_SUCCESS);

        EXPECT_EQ(this->hppKernel->ValidRowCount(), param->GroupRowCount());

        this->hppKernel->SetDroppedByLabel(1ULL, outterLabels.data());
        EXPECT_EQ(this->hppKernel->ValidRowCount(), param->GroupRowCount() - 1ULL);
    }
    std::vector<int8_t> queryFeature;
    attr::OckTimeSpaceAttrTrait filterTrait;
    uint32_t topk{10UL};
    std::vector<uint8_t> extraMask;
    std::vector<uint64_t> outterLabels;
    std::shared_ptr<hcps::OckHeteroOperatorBase> nullRawSearchOp;
};
/*
@brief 注意： 为了减少用例执行时间， 这里将多个场景放在一起执行了
*/
TEST_F(TestOckVsaHPPKernel, hppKernel_add_and_query)
{
    BuildAndTestIndex();
    this->BuildLables();
    MOCKER(&OckVsaAnnCollectResultProcessorT::CreateNPUProcessor)
        .stubs()
        .will(returnValue(std::shared_ptr<OckVsaAnnCollectResultProcessorT>(
            new MockOckVsaAnnCollectResultProcessor<int8_t, 256ULL>(topk, this->param->GroupRowCount(), 1UL))));

    OckHmmSetLogLevel(OCK_LOG_LEVEL_INFO);
    TestAddFeature();

    QueryAllExcludeByTSCondition();

    QueryAllIncludeByTsCondition();

    MOCKER(impl::UsingPureSliceCopy).stubs().will(returnValue(false));
    QueryPartitialIncludeByTsCondition();

    GetFeatureByLabelTest();
    OckHmmSetLogLevel(OCK_LOG_LEVEL_ERROR);
}
}  // namespace hpp
}  // namespace neighbor
}  // namespace vsa
}  // namespace ock