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

#include <cstdlib>
#include <gtest/gtest.h>
#include "ock/hcps/handler/OckHeteroHandler.h"
#include "ock/hcps/WithEnvOckHeteroHandler.h"

namespace ock {
namespace hcps {
namespace handler {
namespace test {
class TestOckHeteroHandler : public WithEnvOckHeteroHandler<testing::Test> {
public:
    using BaseT = WithEnvOckHeteroHandler<testing::Test>;
};

TEST_F(TestOckHeteroHandler, CreateSingleDeviceHandler_Success)
{
    hmm::OckHmmErrorCode errorCode = hmm::HMM_SUCCESS;
    auto handler = CreateSingleDeviceHandler(errorCode);

    ASSERT_EQ(errorCode, hmm::HMM_SUCCESS);
    EXPECT_TRUE(handler->Service().get() != nullptr);

    uint32_t hmoBytes = 1024UL;
    auto hmoRet = handler->HmmMgr().Alloc(hmoBytes);
    EXPECT_EQ(hmoRet.first, hmm::HMM_SUCCESS);
}
}  // namespace test
}  // namespace handler
}  // namespace hcps
}  // namespace ock
