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
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPSliceIdMgr.h"
#include "ock/utils/StrUtils.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
namespace impl {
class TestOckVsaHPPSliceIdMgr : public testing::Test {
public:
    void SetUp(void) override
    {
        idMgr = OckVsaHPPSliceIdMgr::Create(groupCount);
    }
    std::shared_ptr<OckVsaHPPSliceIdMgr> idMgr;
    uint32_t groupCount{10UL};
};
TEST_F(TestOckVsaHPPSliceIdMgr, empty)
{
    EXPECT_EQ(idMgr->SliceCount(), 0UL);
    EXPECT_EQ(idMgr->SliceSet(0UL).size(), 0UL);
    EXPECT_EQ(idMgr->SliceSet(groupCount - 1UL).size(), 0UL);
}
TEST_F(TestOckVsaHPPSliceIdMgr, addOneSlice)
{
    EXPECT_TRUE(idMgr->AddSlice(0UL, 0UL));   // 第一次增加数据场景
    EXPECT_FALSE(idMgr->AddSlice(0UL, 0UL));  // 重复增加数据场景
    EXPECT_EQ(idMgr->SliceCount(), 1UL);
    const OckVsaHPPSliceIdMgr &cstIdMgr = *idMgr;
    EXPECT_EQ(cstIdMgr.SliceSet(0UL).size(), 1UL);
    EXPECT_EQ(idMgr->SliceSet(groupCount - 1UL).size(), 0UL);
}
TEST_F(TestOckVsaHPPSliceIdMgr, writeSliceSet)
{
    idMgr->SliceSet(0UL).insert(0UL);         // 第一次增加数据场景
    EXPECT_FALSE(idMgr->AddSlice(0UL, 0UL));  // 重复增加数据场景
    EXPECT_EQ(idMgr->SliceCount(), 1UL);
    EXPECT_EQ(idMgr->SliceSet(0UL).size(), 1UL);
    EXPECT_EQ(idMgr->SliceSet(groupCount - 1UL).size(), 0UL);
}
TEST_F(TestOckVsaHPPSliceIdMgr, osStream)
{
    idMgr->SliceSet(0UL).insert(0UL);         // 第一次增加数据场景
    EXPECT_FALSE(idMgr->AddSlice(0UL, 0UL));  // 重复增加数据场景
    EXPECT_EQ(idMgr->SliceCount(), 1UL);
    EXPECT_EQ("{0:[0]1:[]2:[]3:[]4:[]5:[]6:[]7:[]8:[]9:[]}", utils::ToString(*idMgr));
}
}  // namespace impl
}  // namespace hpp
}  // namespace neighbor
}  // namespace vsa
}  // namespace ock