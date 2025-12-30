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
#include "secodeFuzz.h"
#include "ock/utils/OckTypeTraits.h"
namespace ock {
namespace utils {
namespace traits {
static constexpr size_t DF_FUZZ_EXEC_COUNT = 300000000;
static constexpr size_t DF_FUZZ_EXEC_SECOND = 10800;
TEST(TestOckTypeTrait, distance)
{
    std::string name = "distance";
    DT_Set_Running_Time_Second(DF_FUZZ_EXEC_SECOND);
    DT_Enable_Leak_Check(0, 0);
    uint32_t seed = 0;
    DT_FUZZ_START(seed, DF_FUZZ_EXEC_COUNT, name, 0)
    {
        s32 dataSize =
                *(s32 *)DT_SetGetNumberRange(&g_Element[0], 5U, 6U, 100U);
        std::vector<uint32_t> datas(dataSize);
        for (int i = 0; i < dataSize; ++i) {
            datas[i] = i + 1;
        }
        EXPECT_EQ(Distance(datas.begin(), datas.end()), Distance(datas.data(), datas.data() + datas.size()));
        EXPECT_EQ(Distance(1UL, 4UL), 3L); // 数值距离
    }
    DT_FUZZ_END()
}
} // namespace traits
} // namespace utils
}