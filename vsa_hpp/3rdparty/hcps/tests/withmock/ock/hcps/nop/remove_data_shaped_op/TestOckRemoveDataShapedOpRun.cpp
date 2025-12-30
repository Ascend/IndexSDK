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
#include "ock/hcps/stream/MockOckHeteroStreamBase.h"
#include "ock/hcps/WithEnvOckHeteroHandler.h"
#include "ock/hcps/nop/remove_data_shaped_op/OckRemoveDataShapedOpRun.h"
namespace ock {
namespace hcps {
namespace nop {
namespace test {
class TestOckRemoveDataShapedOpRun : public handler::WithEnvOckHeteroHandler<testing::Test> {
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
        hmoBlock.reset();
        handler.reset();
        BaseT::TearDown();
    }

    void PrepareHostData(uint32_t removeCount)
    {
        hmoBlock = std::make_shared<OckRemoveDataShapedOpHmoBlock>();
        hmoBlock->dims = 256U;
        hmoBlock->removeCount = removeCount;
        hmoBlock->srcPosHmo = AllocHmo(removeCount * sizeof(uint64_t));
        hmoBlock->dstPosHmo = AllocHmo(removeCount * sizeof(uint64_t));
    }

    std::shared_ptr<handler::OckHeteroHandler> handler;
    std::shared_ptr<OckRemoveDataShapedOpHmoBlock> hmoBlock;

private:
    std::shared_ptr<hmm::OckHmmHMObject> AllocHmo(int64_t byteSize)
    {
        return handler->HmmMgrPtr()->Alloc(byteSize, hmm::OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY).second;
    }
};

TEST_F(TestOckRemoveDataShapedOpRun, generate_remove_shaped_op)
{
    PrepareHostData(32U);
    auto mockStream = std::make_shared<MockOckHeteroStreamBase>();
    EXPECT_CALL(*mockStream, AddOp(testing::_)).WillRepeatedly(testing::InvokeWithoutArgs([]() {
        return;
    }));
    EXPECT_CALL(*mockStream, WaitExecComplete(testing::_)).WillRepeatedly(testing::InvokeWithoutArgs([]() {
        return hmm::HMM_SUCCESS;
    }));
    auto op = OckRemoveDataShapedOpRun::GenRemoveShapedOp(hmoBlock, *handler);
    mockStream->AddOp(op);
    EXPECT_EQ(mockStream->WaitExecComplete(10U), hmm::HMM_SUCCESS);
    EXPECT_CALL(*mockStream, AddOp(testing::_)).Times(0);
}
} // namespace test
} // namespace nop
} // namespace hcps
} // namespace ock
