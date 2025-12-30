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
#include "ock/hcps/hcop/topk_dist_comp_op/OckTopkDistCompOpRun.h"
namespace ock {
namespace hcps {
namespace hcop {
namespace test {
class TestOckTopKDistCompOpRun : public handler::WithEnvOckHeteroHandler<testing::Test> {
public:
    using BaseT = hcps::handler::WithEnvOckHeteroHandler<testing::Test>;
    void SetUp(void) override
    {
        BaseT::SetUp();
        hmm::OckHmmErrorCode errorCode = hmm::HMM_SUCCESS;
        handler = CreateSingleDeviceHandler(errorCode);
        PrePareStream();
    }

    void TearDown(void) override
    {
        mockStream.reset();
        hmoGroup.reset();
        handler.reset();
        BaseT::TearDown();
    }

    void PrePareStream(void)
    {
        mockStream = std::make_shared<MockOckHeteroStreamBase>();
        EXPECT_CALL(*mockStream, AddOp(testing::_)).WillRepeatedly(testing::InvokeWithoutArgs([]() {
            return;
        }));
        EXPECT_CALL(*mockStream, WaitExecComplete(testing::_)).WillRepeatedly(testing::InvokeWithoutArgs([]() {
            return hmm::HMM_SUCCESS;
        }));
    }

    void PrepareData(uint64_t ntotal, uint32_t totalNumBlocks, uint64_t batch = 1)
    {
        hmoGroup = std::make_shared<OckTopkDistCompOpHmoGroup>(false, batch, 256U, 10U, nop::DEFAULT_CODE_BLOCK_SIZE,
            totalNumBlocks, static_cast<int64_t>(ntotal), 0U);
        hmoGroup->SetQueryHmos(AllocHmo(batch * 256U * sizeof(int8_t)),
                               AllocHmo(utils::SafeRoundUp(batch, nop::FP16_ALIGN) * sizeof(OckFloat16)),
                               nullptr);
        hmoGroup->SetOutputHmos(AllocHmo(batch * 10U * sizeof(OckFloat16)),
                                AllocHmo(batch * 10U * sizeof(int64_t)));
        for (uint64_t i = 0; i < totalNumBlocks; ++i) {
            hmoGroup->PushDataBase(AllocHmo(nop::DEFAULT_CODE_BLOCK_SIZE * 256U * sizeof(int8_t)),
                                   AllocHmo(nop::DEFAULT_CODE_BLOCK_SIZE * sizeof(OckFloat16)));
        }
    }

    std::shared_ptr<handler::OckHeteroHandler> handler;
    std::shared_ptr<OckTopkDistCompOpHmoGroup> hmoGroup;
    std::shared_ptr<MockOckHeteroStreamBase> mockStream;

private:
    std::shared_ptr<hmm::OckHmmHMObject> AllocHmo(int64_t byteSize)
    {
        return handler->HmmMgrPtr()->Alloc(byteSize, hmm::OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY).second;
    }
};

TEST_F(TestOckTopKDistCompOpRun, add_op_one_group)
{
    uint64_t ntotal = nop::DEFAULT_CODE_BLOCK_SIZE * nop::DEFAULT_PAGE_BLOCK_NUM;
    uint32_t totalNumBlocks = nop::DEFAULT_PAGE_BLOCK_NUM;
    PrepareData(ntotal, totalNumBlocks);

    OckTopkDistCompOpRun::RunOneGroupSync(hmoGroup, mockStream, handler);
    EXPECT_EQ(mockStream->WaitExecComplete(10U), hmm::HMM_SUCCESS);
    EXPECT_CALL(*mockStream, AddOp(testing::_)).Times(0);
}
} // namespace test
} // namespace hcop
} // namespace hcps
} // namespace ock