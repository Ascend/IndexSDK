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


#include "gtest/gtest.h"
#include "acl/acl.h"
#include "ock/acladapter/utils/OckAscendFp16.h"

namespace ock {
namespace acladapter {
namespace test {
TEST(TestOckAscendFp16, float_to_fp16_to_float)
{
    float a = 1.1;
    OckFloat16 b = OckAscendFp16::FloatToFp16(a);
    float c = OckAscendFp16::Fp16ToFloat(b);
    EXPECT_LT(abs(a - c), 0.001);
}

TEST(TestOckAscendFp16, fp16_to_float_to_fp16)
{
    OckFloat16 a = 12345;
    float b = OckAscendFp16::Fp16ToFloat(a);
    OckFloat16 c = OckAscendFp16::FloatToFp16(b);
    EXPECT_EQ(a, c);
}
} // namespace test
} // namespace acladapter
} // namespace ock
