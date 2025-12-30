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


#include <ascenddaemon/utils/DeviceVector.h>
#include <gtest/gtest.h>
#include <vector>

namespace {
TEST(DEVICE_VECTOR, ALL)
{
    int size = 512;
    ascend::DeviceVector<int32_t> dVec1;
    std::vector<int32_t> stdVec(size);
    for (int i = 0; i < size; i++) {
        stdVec[i] = (int32_t)(rand() % 100);
    }

    // append, size, capacity
    dVec1.append(stdVec.data(), stdVec.size());
    EXPECT_EQ(dVec1.size(), size);
    EXPECT_EQ(dVec1.capacity(), size * 3 / 2);

    // operator[]
    for (int i = 0; i < size; i++) {
        EXPECT_EQ(stdVec[i], dVec1[i]);
    }

    // reclaim
    auto freeSize = dVec1.reclaim(true);
    EXPECT_EQ(freeSize, (size * 3 / 2 - size) * sizeof(int32_t));

    // resize
    dVec1.resize(200);
    EXPECT_EQ(dVec1.size(), 200);

    // cpy to stl vector
    std::vector<int32_t> toVec = dVec1.copyToStlVector();
    EXPECT_EQ(memcmp(toVec.data(), dVec1.data(), dVec1.size()), 0);

    // clear
    dVec1.clear();
    EXPECT_EQ(dVec1.size(), 0);
    EXPECT_EQ(dVec1.capacity(), 0);
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    ascend::DeviceScope deviceScope;
    return RUN_ALL_TESTS();
}
