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
#include "ock/hcps/nop/trans_data_shaped_op/OckTransDataShapedOpRun.h"
namespace ock {
namespace hcps {
namespace nop {
namespace test {
class TestOckTransDataShapedOpRun : public handler::WithEnvOckHeteroHandler<testing::Test> {
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

    void PrepareHostData(int64_t addNum)
    {
        hmoBlock = std::make_shared<OckTransDataShapedOpHmoBlock>();
        hmoBlock->srcHmo = AllocHmo(addNum * 256U);
        hmoBlock->dstHmo = AllocHmo(DEFAULT_CODE_BLOCK_SIZE * 256U);
        hmoBlock->dims = 256U;
        hmoBlock->codeBlockSize = DEFAULT_CODE_BLOCK_SIZE;
        hmoBlock->addNum = addNum;
        hmoBlock->offsetInDstHmo = DEFAULT_CODE_BLOCK_SIZE - addNum;
        std::vector<int8_t> src(DEFAULT_CODE_BLOCK_SIZE * 256U, 0U);
        for (int64_t i = 0; i < DEFAULT_CODE_BLOCK_SIZE; ++i) {
            for (int64_t j = 0; j < 256U; ++j) {
                src[i * 256U + j] = i + j;
            }
        }
        WriteHmo(hmoBlock->srcHmo, src);
    }

    std::shared_ptr<handler::OckHeteroHandler> handler;
    std::shared_ptr<OckTransDataShapedOpHmoBlock> hmoBlock;

private:
    std::shared_ptr<hmm::OckHmmHMObject> AllocHmo(int64_t byteSize)
    {
        return handler->HmmMgrPtr()->Alloc(byteSize, hmm::OckHmmMemoryAllocatePolicy::DEVICE_DDR_ONLY).second;
    }

    template <typename T> void WriteHmo(const std::shared_ptr<hmm::OckHmmHMObject> &hmo, const std::vector<T> &vec)
    {
        aclrtMemcpy(reinterpret_cast<void *>(hmo->Addr()), hmo->GetByteSize(), vec.data(), vec.size() * sizeof(T),
                    ACL_MEMCPY_HOST_TO_DEVICE);
    }
};

TEST_F(TestOckTransDataShapedOpRun, add_trans_data_shaped_ops)
{
    PrepareHostData(DEFAULT_CODE_BLOCK_SIZE);
    auto mockStream = std::make_shared<MockOckHeteroStreamBase>();
    EXPECT_CALL(*mockStream, AddOp(testing::_)).WillRepeatedly(testing::InvokeWithoutArgs([]() {
        return;
    }));
    EXPECT_CALL(*mockStream, WaitExecComplete(testing::_)).WillRepeatedly(testing::InvokeWithoutArgs([]() {
        return hmm::HMM_SUCCESS;
    }));
    OckTransDataShapedOpRun::AddTransShapedOp(hmoBlock, *handler, mockStream);
    EXPECT_EQ(mockStream->WaitExecComplete(10U), hmm::HMM_SUCCESS);
    EXPECT_CALL(*mockStream, AddOp(testing::_)).Times(0);
}
} // namespace test
} // namespace nop
} // namespace hcps
} // namespace ock