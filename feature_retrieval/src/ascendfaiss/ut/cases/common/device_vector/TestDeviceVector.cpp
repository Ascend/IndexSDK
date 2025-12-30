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
#include <map>
#include "DeviceVector.h"

using namespace testing;
using namespace std;

namespace ascend {

class TestDeviceVector : public Test {
};

TEST_F(TestDeviceVector, create_empty_device_vector)
{
    // given
    DeviceVector<int8_t> dv;

    // then
    size_t expectVal = 0;
    EXPECT_EQ(dv.size(), expectVal);
    EXPECT_EQ(dv.capacity(), expectVal);
    EXPECT_EQ(dv.data(), nullptr);
}

TEST_F(TestDeviceVector, resize_exact)
{
    // given
    DeviceVector<char> dv;
    vector<size_t> nums { 10, 100, 1000, 10000 };
    
    // then
    for (auto num : nums) {
        dv.resize(num, true);
        EXPECT_EQ(dv.size(), num);
        EXPECT_EQ(dv.capacity(), num);
        EXPECT_NE(dv.data(), nullptr);
    }

    // when
    size_t num = 15;
    dv.resize(num, true);
    EXPECT_EQ(dv.size(), num);
    EXPECT_EQ(dv.capacity(), nums.back());
    EXPECT_NE(dv.data(), nullptr);
}

TEST_F(TestDeviceVector, resize_with_ExpandPolicy)
{
    // given
    DeviceVector<uint8_t> dv;
    map<size_t, size_t> size2CapacityMap {
        { 10, 16 },
        { 100, 128 },
        { 1000, 1024 },
        { 10000, 11250 },
    };

    // then
    for (auto valPair : size2CapacityMap) {
        dv.resize(valPair.first);
        EXPECT_EQ(dv.size(), valPair.first);
        EXPECT_EQ(dv.capacity(), valPair.second);
        EXPECT_NE(dv.data(), nullptr);
    }

    // then
    size_t num = 100;
    dv.resize(num);
    EXPECT_EQ(dv.size(), num);
    EXPECT_EQ(dv.capacity(), size2CapacityMap.rbegin()->second);
    EXPECT_NE(dv.data(), nullptr);
}

TEST_F(TestDeviceVector, resize_with_ExpandPolicySlim)
{
    // given
    DeviceVector<uint8_t, ExpandPolicySlim> dv;
    map<size_t, size_t> size2CapacityMap {
        { 10, 20 },
        { 100, 200 },
        { 1000, 2000 },
        { 10000, 20000 },
    };

    // then
    for (auto valPair : size2CapacityMap) {
        dv.resize(valPair.first);
        EXPECT_EQ(dv.size(), valPair.first);
        EXPECT_EQ(dv.capacity(), valPair.second);
        EXPECT_NE(dv.data(), nullptr);
    }
}

TEST_F(TestDeviceVector, modify_value)
{
    // given
    DeviceVector<size_t> dv;
    size_t num = 100;
    size_t power = 2;

    // when
    dv.resize(num);
    for (size_t i = 0; i < num; i++) {
        dv[i] = i * power;
    }

    // then
    auto data = dv.data();
    for (size_t i = 0; i < num; i++) {
        EXPECT_EQ(data[i], i * power);
    }
}

TEST_F(TestDeviceVector, clear)
{
    // given
    DeviceVector<int> dv;
    size_t num = 100;

    // when
    dv.resize(num);
    dv[10] = 10; // 设置第10个元素值为10
    dv.clear();

    // then
    size_t expectVal = 0;
    EXPECT_EQ(dv.size(), expectVal);
    EXPECT_EQ(dv.capacity(), expectVal);
    EXPECT_EQ(dv.data(), nullptr);
}

TEST_F(TestDeviceVector, copyToStlVector)
{
    // given
    DeviceVector<float> dv;

    // when
    auto data = dv.copyToStlVector();

    // then
    EXPECT_TRUE(data.empty());

    // when
    size_t num = 10;
    size_t idx = 9;
    size_t expectVal = 100;
    dv.resize(num);
    dv[idx] = expectVal;
    data = dv.copyToStlVector();

    // then
    EXPECT_EQ(data.size(), num);
    EXPECT_EQ(data[idx], expectVal);
}

TEST_F(TestDeviceVector, reclaim_exact)
{
    // given
    DeviceVector<double> dv;
    size_t num = 100;
    dv.resize(num);

    // when
    auto freeSize = dv.reclaim(true);

    // then
    EXPECT_EQ(dv.size(), num);
    EXPECT_EQ(dv.capacity(), num);
    EXPECT_EQ(freeSize, static_cast<size_t>(224)); // 内部算法释放了224字节
}


TEST_F(TestDeviceVector, reclaim_unexact)
{
    // given
    DeviceVector<double> dv;
    size_t num = 100;
    dv.resize(num);

    // when
    auto freeSize = dv.reclaim(false);

    // then
    EXPECT_EQ(dv.size(), num);
    EXPECT_EQ(dv.capacity(), static_cast<size_t>(128)); // 内部算法实际占用了128字节的空间
    EXPECT_EQ(freeSize, static_cast<size_t>(0));

    // when
    num = 10000; // 先扩容成10000，取一个较大的值
    dv.resize(num);
    dv.resize(100); // resize为100，让其空间空余出来
    freeSize = dv.reclaim(false);

    // then
    EXPECT_EQ(freeSize, static_cast<size_t>(77952)); // 内部算法释放了77952字节
}

} // namespace ascend