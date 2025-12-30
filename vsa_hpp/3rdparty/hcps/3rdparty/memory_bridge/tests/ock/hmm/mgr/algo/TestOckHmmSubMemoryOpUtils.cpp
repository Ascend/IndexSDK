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
#include "ock/hmm/mgr/algo/OckHmmSubMemoryOpUtils.h"

namespace ock {
namespace hmm {
namespace test {

class TestOckHmmSubMemoryOpUtils : public testing::Test {
public:
    void SetUp()
    {
        auto front = std::make_shared<MemBlock>(MemBlock{12ULL << 20U, 14ULL << 20U, 2ULL << 20U, 1027U});
        auto middle = std::make_shared<MemBlock>(MemBlock{14ULL << 20U, 18ULL << 20U, 4ULL << 20U, 1029U});
        auto back = std::make_shared<MemBlock>(MemBlock{18ULL << 20U, 38ULL << 20U, 20ULL << 20U, 1045U});
        auto mergedMiddleBackBlock = std::make_shared<MemBlock>(
            MemBlock{14ULL << 20U, 38ULL << 20U, 24ULL << 20U, 1049U});
        mergedAllBlock = std::make_shared<MemBlock>(MemBlock{12ULL << 20U, 38ULL << 20U, 26ULL << 20U, 1051U});

        neighborBlocksInOrder.first = front;
        neighborBlocksInOrder.second = mergedMiddleBackBlock;
        nonNeighborBlocksInOrder.first = front;
        nonNeighborBlocksInOrder.second = back;
        middleBlock = middle;
    }
    std::pair<std::shared_ptr<MemBlock>, std::shared_ptr<MemBlock>> neighborBlocksInOrder;
    std::pair<std::shared_ptr<MemBlock>, std::shared_ptr<MemBlock>> nonNeighborBlocksInOrder;
    std::shared_ptr<MemBlock> mergedAllBlock;
    std::shared_ptr<MemBlock> middleBlock;
    size_t levelFront = 1027U;    // 2M 对应 level 1027
    size_t levelMiddle = 1029;    // 4M 对应 level 1029
    size_t levelBack = 1045U;     // 20M 对应 level 1027+18=1045
    size_t levelMergedMiddleBack = 1049U; // 24M 对应 level 1025+4=1049
    size_t levelMergedAll = 1051U;  // 26M 对应 level 1045+6=1051
};

TEST(OckHmmSubMemoryOpUtils, CheckAvailable)
{
    auto memPool = MemPool();
    memPool.availableSize = 8ULL * 64 * 1024 * 1024;
    EXPECT_EQ(OckHmmSubMemoryOpUtils::CheckAvailable(64ULL << 20U, memPool, "HOST_SWAP"), HMM_SUCCESS);
    EXPECT_EQ(OckHmmSubMemoryOpUtils::CheckAvailable(640ULL << 20U, memPool, "DEVICE_DATA"),
              HMM_ERROR_DEVICE_BUFFER_SPACE_NOT_ENOUGH);
}

TEST(OckHmmSubMemoryOpUtils, GetLevel)
{
    EXPECT_EQ(OckHmmSubMemoryOpUtils::GetLevel(2ULL * 1024 * 1024), 1027U);
    EXPECT_EQ(OckHmmSubMemoryOpUtils::GetLevel(5ULL * 1024 * 1024), 1030U);
    EXPECT_EQ(OckHmmSubMemoryOpUtils::GetLevel(4ULL * 1024 * 1024 * 1024), 5121U);
    EXPECT_EQ(OckHmmSubMemoryOpUtils::GetLevel(34ULL << 20U), 1059U);
}

TEST_F(TestOckHmmSubMemoryOpUtils, GetOriginMemBlocks)
{
    // 19056893953字节 = 18,174M + 72K + 1B
    auto blocks = OckHmmSubMemoryOpUtils::GetOriginMemBlocks(10U, 19056893952U);
    EXPECT_EQ(blocks.size(), 2U);
    EXPECT_EQ(blocks[0U].first, 10ULL);
    EXPECT_EQ(blocks[0U].second, 18174U * 1024ULL * 1024ULL);
    EXPECT_EQ(blocks[1U].first, 10ULL + 18174U * 1024ULL * 1024ULL);
    EXPECT_EQ(blocks[1U].second, 72U * 1024ULL);

    blocks = OckHmmSubMemoryOpUtils::GetOriginMemBlocks(10U, 1024ULL * 1024ULL + 257ULL);
    EXPECT_EQ(blocks.size(), 2U);
    EXPECT_EQ(blocks[0U].first, 10ULL);
    EXPECT_EQ(blocks[0U].second, 1024ULL * 1024ULL);
    EXPECT_EQ(blocks[1U].first, 10ULL + 1024ULL * 1024ULL);
    EXPECT_EQ(blocks[1U].second, 256ULL);
}

TEST_F(TestOckHmmSubMemoryOpUtils, Compare)
{
    auto comp = Compare();
    EXPECT_EQ(comp(neighborBlocksInOrder.first, neighborBlocksInOrder.second), true);
    EXPECT_EQ(comp(nonNeighborBlocksInOrder.first, nonNeighborBlocksInOrder.second), true);
    EXPECT_EQ(comp(neighborBlocksInOrder.second, neighborBlocksInOrder.first), false);
    EXPECT_EQ(comp(nonNeighborBlocksInOrder.second, nonNeighborBlocksInOrder.first), false);
    EXPECT_EQ(comp(neighborBlocksInOrder.first, nonNeighborBlocksInOrder.first), false);
}

TEST_F(TestOckHmmSubMemoryOpUtils, EqualAndNotEqual)
{
    EXPECT_EQ(*(neighborBlocksInOrder.first), *(nonNeighborBlocksInOrder.first));
    EXPECT_NE(*(neighborBlocksInOrder.second), *(nonNeighborBlocksInOrder.second));
}

TEST_F(TestOckHmmSubMemoryOpUtils, AddMemBlock)
{
    auto memPool = MemPool();
    auto &mb1 = nonNeighborBlocksInOrder.first;  // 12~14
    auto &mb2 = nonNeighborBlocksInOrder.second;  // 18~38

    OckHmmSubMemoryOpUtils::AddMemBlock(mb1, memPool);
    OckHmmSubMemoryOpUtils::AddMemBlock(mb2, memPool);

    EXPECT_EQ(mb1->sizeLevel, levelFront);
    EXPECT_EQ(mb2->sizeLevel, levelBack);
    EXPECT_EQ(memPool.availableBlocks[mb1->sizeLevel].size(), 1U);
    EXPECT_EQ(memPool.availableBlocks[mb2->sizeLevel].size(), 1U);
    EXPECT_EQ(memPool.memBlockSet.size(), 2U);  // 增加了两个不相邻的内存块后，set里应该有2个内存块
    EXPECT_EQ(**(memPool.availableBlocks[mb1->sizeLevel].begin()), *mb1);
    EXPECT_EQ(**(memPool.availableBlocks[mb2->sizeLevel].begin()), *mb2);
    auto iter = memPool.memBlockSet.begin();
    EXPECT_EQ(**iter, *mb1);
    iter++;
    EXPECT_EQ(**iter, *mb2);
}

TEST_F(TestOckHmmSubMemoryOpUtils, RemoveMemBlock)
{
    auto memPool = MemPool();
    auto &mb1 = nonNeighborBlocksInOrder.first;
    auto &mb2 = nonNeighborBlocksInOrder.second;

    OckHmmSubMemoryOpUtils::AddMemBlock(mb1, memPool);
    OckHmmSubMemoryOpUtils::AddMemBlock(mb2, memPool);
    auto level1 = mb1->sizeLevel;
    auto level2 = mb2->sizeLevel;
    OckHmmSubMemoryOpUtils::RemoveMemBlock(mb1, memPool);

    EXPECT_EQ(memPool.memBlockSet.size(), 1U);
    EXPECT_EQ(memPool.availableBlocks.find(level1), memPool.availableBlocks.end());
    EXPECT_EQ(memPool.availableBlocks[level2].size(), 1U);
    EXPECT_EQ(**(memPool.availableBlocks[level2].begin()), *mb2);
    EXPECT_EQ(**(memPool.memBlockSet.begin()), *mb2);
}

TEST_F(TestOckHmmSubMemoryOpUtils, MergeWhilePreBlockContinuous)
{
    auto memPool = MemPool();
    auto &mbPre = neighborBlocksInOrder.first;
    auto &mbNew = neighborBlocksInOrder.second;

    OckHmmSubMemoryOpUtils::AddMemBlock(mbPre, memPool);
    OckHmmSubMemoryOpUtils::TryMergeWithAvailableBlocks(mbNew, memPool);
    OckHmmSubMemoryOpUtils::AddMemBlock(mbNew, memPool);

    EXPECT_EQ(memPool.memBlockSet.size(), 1U);
    EXPECT_EQ(memPool.availableBlocks.find(mbPre->sizeLevel), memPool.availableBlocks.end());
    EXPECT_EQ(memPool.availableBlocks[mbNew->sizeLevel].size(), 1U);
    EXPECT_EQ(**(memPool.availableBlocks[mbNew->sizeLevel].begin()), *mergedAllBlock);
    EXPECT_EQ(**(memPool.memBlockSet.begin()), *mergedAllBlock);
}

TEST_F(TestOckHmmSubMemoryOpUtils, MergeWhileNextBlockContinuous)
{
    auto memPool = MemPool();
    auto &mbNext = neighborBlocksInOrder.second;
    auto &mbNew = neighborBlocksInOrder.first;

    OckHmmSubMemoryOpUtils::AddMemBlock(mbNext, memPool);
    OckHmmSubMemoryOpUtils::TryMergeWithAvailableBlocks(mbNew, memPool);
    OckHmmSubMemoryOpUtils::AddMemBlock(mbNew, memPool);

    EXPECT_EQ(memPool.memBlockSet.size(), 1U);
    EXPECT_EQ(memPool.availableBlocks.find(mbNext->sizeLevel), memPool.availableBlocks.end());
    EXPECT_EQ(memPool.availableBlocks[mbNew->sizeLevel].size(), 1U);
    EXPECT_EQ(**(memPool.availableBlocks[mbNew->sizeLevel].begin()), *mergedAllBlock);
    EXPECT_EQ(**(memPool.memBlockSet.begin()), *mergedAllBlock);
}

TEST_F(TestOckHmmSubMemoryOpUtils, MergeWhilePreBlockNotContinuous)
{
    auto memPool = MemPool();
    auto &mbPre = nonNeighborBlocksInOrder.first;
    auto &mbNew = nonNeighborBlocksInOrder.second;

    OckHmmSubMemoryOpUtils::AddMemBlock(mbPre, memPool);
    OckHmmSubMemoryOpUtils::TryMergeWithAvailableBlocks(mbNew, memPool);
    OckHmmSubMemoryOpUtils::AddMemBlock(mbNew, memPool);

    EXPECT_EQ(memPool.memBlockSet.size(), 2U);  // 2：没有真正发生合并，所以set里有2个blocks
    EXPECT_EQ(memPool.availableBlocks[mbPre->sizeLevel].size(), 1U);
    EXPECT_EQ(memPool.availableBlocks[mbNew->sizeLevel].size(), 1U);
    EXPECT_EQ(**(memPool.availableBlocks[mbPre->sizeLevel].begin()), *mbPre);
    EXPECT_EQ(**(memPool.availableBlocks[mbNew->sizeLevel].begin()), *mbNew);
    EXPECT_EQ(**(memPool.memBlockSet.begin()), *mbPre);
    EXPECT_EQ(**(++memPool.memBlockSet.begin()), *mbNew);
}

TEST_F(TestOckHmmSubMemoryOpUtils, MergeWhileNextBlockNotContinuous)
{
    auto memPool = MemPool();
    auto &mbNext = nonNeighborBlocksInOrder.second;
    auto &mbNew = nonNeighborBlocksInOrder.first;

    OckHmmSubMemoryOpUtils::AddMemBlock(mbNext, memPool);
    OckHmmSubMemoryOpUtils::TryMergeWithAvailableBlocks(mbNew, memPool);
    OckHmmSubMemoryOpUtils::AddMemBlock(mbNew, memPool);

    EXPECT_EQ(memPool.memBlockSet.size(), 2U);  // 2：没有真正发生合并，所以set里有2个blocks
    EXPECT_EQ(memPool.availableBlocks[mbNext->sizeLevel].size(), 1U);
    EXPECT_EQ(memPool.availableBlocks[mbNew->sizeLevel].size(), 1U);
    EXPECT_EQ(**(memPool.availableBlocks[mbNew->sizeLevel].begin()), *mbNew);
    EXPECT_EQ(**(memPool.availableBlocks[mbNext->sizeLevel].begin()), *mbNext);
    EXPECT_EQ(**(memPool.memBlockSet.begin()), *mbNew);
    EXPECT_EQ(**(++memPool.memBlockSet.begin()), *mbNext);
}

TEST_F(TestOckHmmSubMemoryOpUtils, MergeWhilePreAndNextBlockContinuous)
{
    auto memPool = MemPool();
    auto &mbPre = nonNeighborBlocksInOrder.first;
    auto &mbNext = nonNeighborBlocksInOrder.second;
    auto &mbNew = middleBlock;

    OckHmmSubMemoryOpUtils::AddMemBlock(mbNext, memPool);
    OckHmmSubMemoryOpUtils::AddMemBlock(mbPre, memPool);
    OckHmmSubMemoryOpUtils::TryMergeWithAvailableBlocks(mbNew, memPool);
    OckHmmSubMemoryOpUtils::AddMemBlock(mbNew, memPool);

    EXPECT_EQ(memPool.memBlockSet.size(), 1U);
    EXPECT_EQ(memPool.availableBlocks.find(mbPre->sizeLevel), memPool.availableBlocks.end());
    EXPECT_EQ(memPool.availableBlocks.find(mbNext->sizeLevel), memPool.availableBlocks.end());
    EXPECT_EQ(memPool.availableBlocks[mbNew->sizeLevel].size(), 1U);
    EXPECT_EQ(**(memPool.availableBlocks[mbNew->sizeLevel].begin()), *mergedAllBlock);
    EXPECT_EQ(**(memPool.memBlockSet.begin()), *mergedAllBlock);
}

TEST_F(TestOckHmmSubMemoryOpUtils, TryToAllocFromCurMemBlockNotEqual)
{
    auto &mb1 = neighborBlocksInOrder.first;  // 12~14
    auto &mb2 = neighborBlocksInOrder.second;  // 14~38
    uintptr_t memAddress1;
    auto memPool = MemPool();

    OckHmmSubMemoryOpUtils::AddMemBlock(mergedAllBlock, memPool);  // 12~38
    auto ret = OckHmmSubMemoryOpUtils::TryToAllocFromCurMemBlock(
        mergedAllBlock, mb1->length, memAddress1, memPool);

    EXPECT_EQ(ret, HMM_SUCCESS);
    EXPECT_EQ(memAddress1, mb1->startAddress);
    EXPECT_EQ(memPool.memBlockSet.size(), 1U);
    EXPECT_EQ(memPool.availableBlocks.find(levelMergedAll), memPool.availableBlocks.end());
    EXPECT_EQ(memPool.availableBlocks[mb2->sizeLevel].size(), 1U);
    EXPECT_EQ(**(memPool.availableBlocks[mb2->sizeLevel].begin()), *mb2);
}

TEST_F(TestOckHmmSubMemoryOpUtils, TryToAllocFromCurMemBlockEqual)
{
    auto &mb1 = neighborBlocksInOrder.first;
    auto &mb2 = neighborBlocksInOrder.second;
    uintptr_t memAddress1;
    uintptr_t memAddress2;

    auto memPool = MemPool();
    OckHmmSubMemoryOpUtils::AddMemBlock(mergedAllBlock, memPool);
    OckHmmSubMemoryOpUtils::TryToAllocFromCurMemBlock(
        mergedAllBlock, mb1->length, memAddress1, memPool);
    auto m2StartAddress = mb2->startAddress;
    auto ret = OckHmmSubMemoryOpUtils::TryToAllocFromCurMemBlock(
        *(memPool.memBlockSet.begin()), mb2->length, memAddress2, memPool);

    EXPECT_EQ(ret, HMM_SUCCESS);
    EXPECT_EQ(memAddress2, m2StartAddress);
    EXPECT_EQ(memPool.memBlockSet.size(), 0U);
    EXPECT_EQ(memPool.availableBlocks.find(levelMergedMiddleBack), memPool.availableBlocks.end());
}

}  // namespace test
}  // namespace hmm
}  // namespace ock
