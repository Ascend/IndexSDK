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
#include "ock/utils/OckTypeTraits.h"
namespace ock {
namespace utils {
namespace traits {
TEST(TestOckTypeTrait, distance)
{
    std::vector<uint32_t> datas({1UL, 3UL, 4UL, 2UL, 5UL, 6UL});
    EXPECT_EQ(Distance(datas.begin(), datas.end()), Distance(datas.data(), datas.data() + datas.size()));
    EXPECT_EQ(Distance(1UL, 4UL), 3L);  // 数值距离
}
}  // namespace traits
}  // namespace utils
}  // namespace ock