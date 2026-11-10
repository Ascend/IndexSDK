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

#include <algorithm>
#include <map>
#include <memory>
#include <mockcpp/mockcpp.hpp>
#include <numeric>
#include <vector>

#include "../TestEnvGuard.h"
#include "DeviceMemArena.h"
#include "DeviceVector.h"

using namespace testing;
using namespace std;

namespace ascend
{

class TestDeviceVector : public Test
{
   protected:
    void SetUp() override
    {
        // Pools are keyed by device id and the ACL mock refuses aclrtGetDevice() until a
        // device is active, so state the precondition here instead of inheriting it from
        // whichever test happened to run first.
        ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);

        // Keep per-test host memory small for pools created below.
        chunkMbEnv_.reset(new ScopedEnv("ASCENDFAISS_MEM_POOL_CHUNK_MB", "4"));

        // Start from an empty registry. Tests here assert that a freed block is handed
        // straight back out, which only holds on a pool that no earlier test has
        // fragmented, so without this they break under --gtest_filter, --gtest_shuffle
        // or sharding.
        ResetMemoryPoolsForTest();

        poolWasEnabled_ = MemoryPoolEnabled();
        SetMemoryPoolEnabled(true);
    }

    void TearDown() override
    {
        // Drop the chunks, including any still pinned by a pointer a failing test
        // leaked, so they cannot outlive this test and reach the real aclrtFree().
        ResetMemoryPoolsForTest();
        SetMemoryPoolEnabled(poolWasEnabled_);
        GlobalMockObject::verify();
        chunkMbEnv_.reset();
    }

   private:
    bool poolWasEnabled_{false};
    std::unique_ptr<ScopedEnv> chunkMbEnv_;
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
    vector<size_t> nums{10, 100, 1000, 10000};

    // then
    for (auto num : nums)
    {
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
    map<size_t, size_t> size2CapacityMap{
        {10, 16},
        {100, 128},
        {1000, 1024},
        {10000, 11250},
    };

    // then
    for (auto valPair : size2CapacityMap)
    {
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
    map<size_t, size_t> size2CapacityMap{
        {10, 20},
        {100, 200},
        {1000, 2000},
        {10000, 20000},
    };

    // then
    for (auto valPair : size2CapacityMap)
    {
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
    for (size_t i = 0; i < num; i++)
    {
        dv[i] = i * power;
    }

    // then
    auto data = dv.data();
    for (size_t i = 0; i < num; i++)
    {
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
    dv[10] = 10;  // 设置第10个元素值为10
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
    EXPECT_EQ(freeSize, static_cast<size_t>(224));  // 内部算法释放了224字节
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
    EXPECT_EQ(dv.capacity(), static_cast<size_t>(128));  // 内部算法实际占用了128字节的空间
    EXPECT_EQ(freeSize, static_cast<size_t>(0));

    // when
    num = 10000;  // 先扩容成10000，取一个较大的值
    dv.resize(num);
    dv.resize(100);  // resize为100，让其空间空余出来
    freeSize = dv.reclaim(false);

    // then
    EXPECT_EQ(freeSize, static_cast<size_t>(77952));  // 内部算法释放了77952字节
}

TEST_F(TestDeviceVector, memcpySChunked_zero_count)
{
    uint8_t dst[4] = {1, 2, 3, 4};
    uint8_t src[4] = {5, 6, 7, 8};
    EXPECT_EQ(memcpySChunked(dst, sizeof(dst), src, 0), EOK);
    EXPECT_EQ(dst[0], 1);
}

TEST_F(TestDeviceVector, memcpySChunked_normal_copy)
{
    constexpr size_t kLen = 1024;
    std::vector<uint8_t> src(kLen);
    std::vector<uint8_t> dst(kLen, 0);
    for (size_t i = 0; i < kLen; ++i)
    {
        src[i] = static_cast<uint8_t>(i & 0xff);
    }

    EXPECT_EQ(memcpySChunked(dst.data(), dst.size(), src.data(), src.size()), EOK);
    EXPECT_EQ(dst, src);
}

TEST_F(TestDeviceVector, memcpySChunked_count_exceeds_capacity)
{
    uint8_t dst[8] = {};
    uint8_t src[16] = {};
    EXPECT_EQ(memcpySChunked(dst, sizeof(dst), src, sizeof(src)), ERANGE);
}

TEST_F(TestDeviceVector, arena_reuses_reallocated_blocks)
{
    constexpr size_t kSteps = 4096;
    auto arena = std::make_shared<DeviceMemArena>();
    DeviceVector<uint8_t> codes(arena);

    for (size_t n = 1; n <= kSteps; ++n)
    {
        codes.resize(n, false);
    }

    EXPECT_EQ(codes.size(), kSteps);
    EXPECT_EQ(arena->AllocationCount(), 1u);
    EXPECT_EQ(arena->AllocatedBytes(), codes.capacity());
    codes.clear();
    EXPECT_EQ(arena->AllocationCount(), 0u);
    EXPECT_EQ(arena->AllocatedBytes(), 0u);
}

TEST_F(TestDeviceVector, arena_append_uses_existing_capacity)
{
    auto arena = std::make_shared<DeviceMemArena>();
    DeviceVector<uint8_t> values(arena);
    std::vector<uint8_t> initial(100, 1);
    const uint8_t extra = 2;

    values.append(initial.data(), initial.size(), false);
    ASSERT_GT(values.capacity(), values.size());
    auto *dataBefore = values.data();
    const size_t capacityBefore = values.capacity();
    const size_t allocatedBefore = arena->AllocatedBytes();

    while (values.size() < capacityBefore)
    {
        values.append(&extra, 1, false);
        EXPECT_EQ(values.data(), dataBefore);
    }
    EXPECT_EQ(arena->AllocatedBytes(), allocatedBefore);
    EXPECT_EQ(arena->AllocationCount(), 1u);
}

TEST_F(TestDeviceVector, arena_realloc_preserves_data)
{
    auto arena = std::make_shared<DeviceMemArena>();
    DeviceVector<uint8_t> values(arena);
    std::vector<uint8_t> input(128);
    std::iota(input.begin(), input.end(), static_cast<uint8_t>(0));

    values.append(input.data(), input.size(), true);
    values.resize(4096, true);
    auto output = values.copyToStlVector();

    ASSERT_EQ(output.size(), 4096u);
    EXPECT_TRUE(std::equal(input.begin(), input.end(), output.begin()));
}

TEST_F(TestDeviceVector, arena_empty_reallocation_releases_previous_capacity)
{
    auto arena = std::make_shared<DeviceMemArena>();
    DeviceVector<uint8_t> values(arena);
    values.reserve(64);
    values.reserve(128);
    EXPECT_EQ(values.size(), 0u);
    EXPECT_EQ(arena->AllocationCount(), 1u);
    EXPECT_EQ(arena->AllocatedBytes(), values.capacity());

    values.resize(8);
    values.resize(0);
    values.reserve(256);
    EXPECT_EQ(arena->AllocationCount(), 1u);
    EXPECT_EQ(arena->AllocatedBytes(), values.capacity());
    values.clear();
    EXPECT_EQ(arena->AllocationCount(), 0u);
}

TEST_F(TestDeviceVector, arena_vector_reuses_memoryspace_vector_storage)
{
    auto arena = std::make_shared<DeviceMemArena>();
    DeviceVector<uint8_t> plain(MemorySpace::DEVICE_HUGEPAGE);
    DeviceVector<uint8_t> grouped(arena);
    plain.resize(128, true);
    auto *released = plain.data();
    ASSERT_NE(released, nullptr);
    plain.clear();
    // Both vectors reach the same (device, space) pool, and SetUp() left that pool
    // empty, so the block plain just returned is the exact best fit for the next
    // request of the same size.
    grouped.resize(128, true);
    EXPECT_EQ(grouped.data(), released);
    grouped.clear();
    plain.resize(128, true);
    EXPECT_EQ(plain.data(), released);
}

TEST_F(TestDeviceVector, arena_copy_failure_keeps_old_storage_and_returns_new_allocation)
{
    auto arena = std::make_shared<DeviceMemArena>();
    DeviceVector<uint8_t> values(arena);
    const std::vector<uint8_t> input(128, 0x5a);
    values.append(input.data(), input.size(), true);
    auto *original = values.data();
    const size_t capacity = values.capacity();
#ifdef HOSTCPU
    MOCKER_CPP(&aclrtMemcpy).stubs().will(returnValue(ACL_ERROR_FAILURE));
#else
    MOCKER_CPP(&memcpy_s).stubs().will(returnValue(ERANGE));
#endif
    EXPECT_THROW(values.resize(4096, true), std::exception);
    EXPECT_EQ(values.data(), original);
    EXPECT_EQ(values.capacity(), capacity);
    EXPECT_EQ(values.size(), input.size());
    EXPECT_EQ(arena->AllocationCount(), 1u);
    EXPECT_EQ(arena->AllocatedBytes(), capacity);
    GlobalMockObject::verify();
    EXPECT_EQ(values.copyToStlVector(), input);
}

TEST_F(TestDeviceVector, aclrtMemcpyChunked_normal_and_invalid)
{
    constexpr size_t kLen = 256;
    std::vector<uint8_t> src(kLen, 0xab);
    std::vector<uint8_t> dst(kLen, 0);

    EXPECT_EQ(aclrtMemcpyChunked(dst.data(), dst.size(), src.data(), 0, ACL_MEMCPY_HOST_TO_HOST), ACL_SUCCESS);
    EXPECT_EQ(aclrtMemcpyChunked(dst.data(), dst.size(), src.data(), src.size(), ACL_MEMCPY_HOST_TO_HOST), ACL_SUCCESS);
    EXPECT_EQ(dst, src);
    EXPECT_EQ(aclrtMemcpyChunked(dst.data(), dst.size() / 2, src.data(), src.size(), ACL_MEMCPY_HOST_TO_HOST),
              ACL_ERROR_FAILURE);
}

}  // namespace ascend
