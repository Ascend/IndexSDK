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


#include <ascenddaemon/AscendResourcesProxy.h>
#include <ascenddaemon/utils/AscendUtils.h>
#include <ascenddaemon/utils/StaticUtils.h>
#include <gtest/gtest.h>
#include <vector>

namespace {
TEST(TestAscendResources, Init)
{
    ascend::AscendResourcesProxy res;

    aclrtStream stream = res.getDefaultStream();
    EXPECT_NE(stream, nullptr);
}

TEST(TestAscendResources, Stream)
{
    ascend::AscendResourcesProxy res;

    aclrtStream stream = res.getDefaultStream();
    EXPECT_NE(stream, nullptr);

    std::vector<aclrtStream> vecStr = res.getAlternateStreams();
    EXPECT_EQ(vecStr.size(), 2);

    for (auto &str : vecStr) {
        EXPECT_NE(str, nullptr);
    }
}

TEST(TestAscendResources, Mem)
{
    ascend::AscendResourcesProxy res;
    res.initialize();
    aclrtStream stream = res.getDefaultStream();
    EXPECT_NE(stream, nullptr);

    ascend::AscendMemory &memMan = res.getMemoryManager();

    int defaultMemSize = 0x8000000; // default memory size allocated
    int byteAligned = 32;  // byte Aligned in Ascend
    EXPECT_EQ(memMan.getSizeAvailable(), (size_t)defaultMemSize);
    EXPECT_EQ(memMan.getHighWater(), (size_t)0);

    {
        const int testSize = 512;
        auto mem = memMan.getMemory(stream, testSize);
        EXPECT_EQ(mem.size(), (size_t)testSize);
        EXPECT_EQ(mem.stream(), stream);

        EXPECT_EQ(memMan.getSizeAvailable(), (size_t)(defaultMemSize - testSize));
        EXPECT_EQ(memMan.getHighWater(), (size_t)0);
    }

    EXPECT_EQ(memMan.getSizeAvailable(), (size_t)defaultMemSize);
    EXPECT_EQ(memMan.getHighWater(), (size_t)0);

    {
        auto mem = memMan.getMemory(stream, defaultMemSize + 8);
        EXPECT_EQ(mem.size(), defaultMemSize + ascend::utils::roundUp(8, byteAligned));
        EXPECT_EQ(mem.stream(), stream);

        EXPECT_EQ(memMan.getSizeAvailable(), defaultMemSize);
        EXPECT_EQ(memMan.getHighWater(), defaultMemSize + ascend::utils::roundUp(8, byteAligned));
    }
    std::cout << memMan.toString() << std::endl;
}
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    ascend::DeviceScope deviceScope;
    return RUN_ALL_TESTS();
}
