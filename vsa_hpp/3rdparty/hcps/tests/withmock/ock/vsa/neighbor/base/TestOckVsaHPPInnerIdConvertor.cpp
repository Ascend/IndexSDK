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
#include "ock/vsa/neighbor/base/OckVsaHPPInnerIdConvertor.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace adapter {
TEST(TestOckVsaHPPInnerIdConvertor, parse_correct)
{
    EXPECT_EQ(OckVsaHPPInnerIdConvertor(32U).ToIdx(0xFF00FF11UL, 0x110011FFUL), 0xFF00FF11110011FFULL);
    EXPECT_EQ(OckVsaHPPInnerIdConvertor(32U).ToIdx(0x110011FFUL, 0xFF00FF11UL), 0x110011FFFF00FF11ULL);
    EXPECT_EQ(OckVsaHPPInnerIdConvertor(22U).ToIdx(2UL, 4194303UL), 0xBFFFFFULL);
    EXPECT_EQ(OckVsaHPPInnerIdConvertor(22U).ToIdx(25UL, 4194303UL), 0x67FFFFFULL);
}

TEST(TestOckVsaHPPInnerIdConvertor, ToGroupOffset)
{
    OckVsaHPPInnerIdConvertor cvt(22U);
    uint32_t grpId = 25UL;
    uint32_t offset = 4194303UL;
    auto innerIdx = OckVsaHPPIdx(grpId, offset);
    auto retIdx = cvt.ToGroupOffset(cvt.ToIdx(grpId, offset));
    EXPECT_EQ(grpId, retIdx.grpId);
    EXPECT_EQ(offset, retIdx.offset);
    EXPECT_EQ(innerIdx.grpId, grpId);
    EXPECT_EQ(innerIdx.offset, offset);
}
TEST(TestOckVsaHPPInnerIdConvertor, CalcBitCount)
{
    EXPECT_EQ(29UL, OckVsaHPPInnerIdConvertor::CalcBitCount(400000000ULL));      // 4亿条数据 29bit
    EXPECT_EQ(24UL, OckVsaHPPInnerIdConvertor::CalcBitCount(262144ULL * 64ULL)); // 一个缺省Group24bit
}
} // vsa
} // namespace neighbor
} // namespace vsa
} // namespace ock