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
#include "ock/hcps/stream/OckHeteroOperatorBase.h"

namespace ock {
namespace hcps {
namespace test {
class TestOckHeteroOperatorBase : public testing::Test {
public:
    using HostSimpleOp = OckSimpleHeteroOperator<acladapter::OckTaskResourceType::HOST_CPU>;

    std::shared_ptr<OckHeteroOperatorBase> CreateOp(void)
    {
        return HostSimpleOp::Create([](OckHeteroStreamContext &) { return hmm::HMM_SUCCESS; });
    }
};

TEST_F(TestOckHeteroOperatorBase, CreateGroup_one_op)
{
    auto grp = OckHeteroOperatorBase::CreateGroup(CreateOp());
    EXPECT_EQ(grp->size(), 1UL);
}
TEST_F(TestOckHeteroOperatorBase, CreateGroup_two_op)
{
    auto grp = OckHeteroOperatorBase::CreateGroup(CreateOp(), CreateOp());
    EXPECT_EQ(grp->size(), 2UL);
}
TEST_F(TestOckHeteroOperatorBase, CreateGroup_three_op)
{
    auto grp = OckHeteroOperatorBase::CreateGroup(CreateOp(), CreateOp(), CreateOp());
    EXPECT_EQ(grp->size(), 3UL);
}
}  // namespace test
}  // namespace hcps
}  // namespace ock
