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

#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include "acl/acl.h"
#include "mockcpp/mockcpp.hpp"
#include "ock/hmm/OckHmmFactory.h"
#include "ock/utils/OckSafeUtils.h"
#include "ock/acladapter/utils/OckAscendFp16.h"
#include "ock/hcps/stream/MockOckHeteroStreamBase.h"
#include "ock/hcps/WithEnvOckHeteroHandler.h"
#include "ock/hcps/nop/dist_mask_gen_op/OckDistMaskGenOpRun.h"
namespace ock {
namespace hcps {
namespace nop {
namespace test {
class TestOckDistMaskGenOpRun : public handler::WithEnvOckHeteroHandler<testing::Test> {
public:
    using BaseT = hcps::handler::WithEnvOckHeteroHandler<testing::Test>;
    void SetUp(void) override
    {
        BaseT::SetUp();
        hmm::OckHmmErrorCode errorCode = hmm::HMM_SUCCESS;
        handler = CreateSingleDeviceHandler(errorCode);
    }

    void TearDown(void) override
    {
        hmoGroups.reset();
        handler.reset();
        BaseT::TearDown();
    }

    void PrepareData()
    {
        hmoGroups = std::make_shared<OckDistMaskGenOpHmoGroups>();
        for (uint32_t i = 0; i < 2U; ++i) {
                auto timeHmo = AllocHmo(DEFAULT_CODE_BLOCK_SIZE * sizeof(int32_t));
                auto tokenQsHmo = AllocHmo(DEFAULT_CODE_BLOCK_SIZE * sizeof(int32_t));
                auto tokenRsHmo = AllocHmo(DEFAULT_CODE_BLOCK_SIZE * OPS_DATA_TYPE_TIMES * sizeof(uint8_t));
                hmoGroups->attrTimes.emplace_back(
                    hmm::OckHmmHMObject::CreateSubHmoList(timeHmo, 262144ULL * 256ULL * sizeof(uint8_t)));
                hmoGroups->attrTokenQuotients.emplace_back(
                    hmm::OckHmmHMObject::CreateSubHmoList(tokenQsHmo, 262144ULL * 256ULL * sizeof(uint8_t)));
                hmoGroups->attrTokenRemainders.emplace_back(
                    hmm::OckHmmHMObject::CreateSubHmoList(tokenRsHmo, 262144ULL * 256ULL * sizeof(uint8_t)));
        }

        uint32_t batch = 2U;
        for (uint32_t i = 0; i < batch; ++i) {
            auto queryTimeHmo = AllocHmo(OPS_DATA_TYPE_ALIGN * sizeof(int32_t));
            auto queryTokenIdHmo = AllocHmo(utils::SafeDivUp(2500U, OPS_DATA_TYPE_ALIGN) * OPS_DATA_TYPE_TIMES);
            hmoGroups->queryTimes.emplace_back(queryTimeHmo);
            hmoGroups->queryTokenIds.emplace_back(queryTokenIdHmo);
        }

        hmoGroups->mask = AllocHmo(((DEFAULT_CODE_BLOCK_SIZE * 2U) / OPS_DATA_TYPE_ALIGN) * batch);
        hmoGroups->featureAttrBlockSize = DEFAULT_CODE_BLOCK_SIZE;
        hmoGroups->blockCount = 1U;
        hmoGroups->maskLen = (DEFAULT_CODE_BLOCK_SIZE * 2U) / OPS_DATA_TYPE_ALIGN;
    }

    std::shared_ptr<hcps::handler::OckHeteroHandler> handler;
    std::shared_ptr<OckDistMaskGenOpHmoGroups> hmoGroups;

private:
    std::shared_ptr<hmm::OckHmmHMObject> AllocHmo(int64_t byteSize)
    {
        return handler->HmmMgrPtr()->Alloc(byteSize, hmm::OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY).second;
    }
};

TEST_F(TestOckDistMaskGenOpRun, add_dist_mask_gen_ops)
{
    PrepareData();
    auto mockStream = std::make_shared<MockOckHeteroStreamBase>();
    EXPECT_CALL(*mockStream, AddOp(testing::_)).WillRepeatedly(testing::InvokeWithoutArgs([]() {
        return;
    }));
    EXPECT_CALL(*mockStream, WaitExecComplete(testing::_)).WillRepeatedly(testing::InvokeWithoutArgs([]() {
        return hmm::HMM_SUCCESS;
    }));
    EXPECT_EQ(OckDistMaskGenOpRun::AddMaskOpsMultiBatches(hmoGroups, mockStream), hmm::HMM_SUCCESS);
    EXPECT_CALL(*mockStream, AddOp(testing::_)).Times(0);
}
} // namespace test
} // namespace nop
} // namespace hcps
} // namespace ock