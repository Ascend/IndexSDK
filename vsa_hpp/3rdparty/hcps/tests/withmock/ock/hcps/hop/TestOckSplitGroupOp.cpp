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
#include <vector>
#include <gtest/gtest.h>
#include "ock/hcps/hop/OckSplitGroupOp.h"
#include "ock/hcps/WithEnvOckHeteroStream.h"
#include "ock/acladapter/WithEnvAclMock.h"
#include "ock/utils/OckSafeUtils.h"

namespace ock {
namespace hcps {
namespace hop {
class TestOckSplitGroupOp : public WithEnvOckHeteroStream<acladapter::WithEnvAclMock<testing::Test>> {
public:
    TestOckSplitGroupOp(void) : datas(dataCount)
    {}
    uint64_t dataCount{100};
    uint64_t stepInterval{15};
    std::vector<uint32_t> datas;
};

TEST_F(TestOckSplitGroupOp, MakeOckSplitGroupOps_Success)
{
    auto ops = MakeOckSplitGroupOps<std::vector<uint32_t>::iterator>(
        datas.begin(), datas.end(), stepInterval, [](std::vector<uint32_t>::iterator, std::vector<uint32_t>::iterator) {
            return hmm::HMM_SUCCESS;
        });
    EXPECT_EQ(ops->size(), utils::SafeDivUp(dataCount, stepInterval));

    InitStream();
    this->stream->AddOps(*ops);
    EXPECT_EQ(this->stream->WaitExecComplete(), hmm::HMM_SUCCESS);
}
TEST_F(TestOckSplitGroupOp, MakeOckSplitGroupAtmoicOps_Failed)
{
    auto ops = MakeOckSplitGroupAtmoicOps<std::vector<uint32_t>::iterator>(
        datas.begin(), datas.end(), stepInterval, [](std::vector<uint32_t>::iterator) {
            return hmm::HMM_ERROR_EXEC_FAILED;
        });
    EXPECT_EQ(ops->size(), utils::SafeDivUp(dataCount, stepInterval));
    InitStream();
    this->stream->AddOps(*ops);
    EXPECT_EQ(this->stream->WaitExecComplete(), hmm::HMM_ERROR_EXEC_FAILED);
}
TEST_F(TestOckSplitGroupOp, MakeOckSplitGroupAtmoicOpsNoReturn)
{
    auto ops = MakeOckSplitGroupAtmoicOpsNoReturn<std::vector<uint32_t>::iterator>(
        datas.begin(), datas.end(), stepInterval, [](std::vector<uint32_t>::iterator) {});
    EXPECT_EQ(ops->size(), utils::SafeDivUp(dataCount, stepInterval));
    InitStream();
    this->stream->AddOps(*ops);
    EXPECT_EQ(this->stream->WaitExecComplete(), hmm::HMM_SUCCESS);
}
}  // namespace hop
}  // namespace hcps
}  // namespace ock
