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
#include "ock/hcps/nop/l2_norm_op/OckL2NormOpRun.h"
namespace ock {
namespace hcps {
namespace nop {
namespace test {
class TestOckL2NormOpRun : public handler::WithEnvOckHeteroHandler<testing::Test> {
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
        dataBase.reset();
        handler.reset();
        BaseT::TearDown();
    }

    void PrepareData(int64_t addNum)
    {
        std::vector<int8_t> hostData(addNum * 256U, 1U);
        dataBase = AllocHmo(addNum * 256U * sizeof(int8_t));
        WriteHmo(dataBase, hostData);
    }

    std::shared_ptr<hcps::handler::OckHeteroHandler> handler;
    std::shared_ptr<hmm::OckHmmHMObject> dataBase;

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

TEST_F(TestOckL2NormOpRun, add_l2_norm_ops)
{
    PrepareData(L2NORM_COMPUTE_BATCH);
    auto mockStream = std::make_shared<MockOckHeteroStreamBase>();
    EXPECT_CALL(*mockStream, AddOp(testing::_)).WillRepeatedly(testing::InvokeWithoutArgs([]() {
        return;
    }));
    EXPECT_CALL(*mockStream, WaitExecComplete(testing::_)).WillRepeatedly(testing::InvokeWithoutArgs([]() {
        return hmm::HMM_SUCCESS;
    }));
    ock::hmm::OckHmmErrorCode errorCode = hmm::HMM_SUCCESS;
    auto hmoBlock = OckL2NormOpRun::BuildNormHmoBlock(dataBase, *handler, 256U, L2NORM_COMPUTE_BATCH, errorCode);
    EXPECT_EQ(errorCode, hmm::HMM_SUCCESS);
    EXPECT_EQ(OckL2NormOpRun::ComputeNormSync(hmoBlock, *handler, mockStream), hmm::HMM_SUCCESS);
    EXPECT_CALL(*mockStream, AddOp(testing::_)).Times(0);
}
} // namespace test
} // namespace nop
} // namespace hcps
} // namespace ock