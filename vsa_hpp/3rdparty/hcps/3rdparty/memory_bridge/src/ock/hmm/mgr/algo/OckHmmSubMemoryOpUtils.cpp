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


#include "OckHmmSubMemoryOpUtils.h"

namespace ock {
namespace hmm {

bool operator==(const MemBlock &lhs, const MemBlock &rhs)
{
    return lhs.startAddress == rhs.startAddress &&
           lhs.endAddress == rhs.endAddress &&
           lhs.length == rhs.length &&
           lhs.sizeLevel == rhs.sizeLevel;
}

bool operator!=(const MemBlock &lhs, const MemBlock &rhs)
{
    return !(lhs == rhs);
}

std::ostream &operator<<(std::ostream &os, const MemBlock& memBlock)
{
    return os << "{'length':" << memBlock.length
              << ",'sizeLevel':" << memBlock.sizeLevel << "}";
}

std::ostream &operator<<(std::ostream &os, const MemPool &pool)
{
    os << "'totalSize':" << pool.totalSize << ",'availableSize':" << pool.availableSize
       << ",'availableBlocks.size': " << pool.availableBlocks.size();
    os << ", 'detail':{";
    for (auto const &pair : pool.availableBlocks) {
        os << pair.first << ":" << pair.second.size() << ",";
    }
    return os << "}";
}

std::vector<std::pair<uintptr_t, uint64_t>> OckHmmSubMemoryOpUtils::GetOriginMemBlocks(
    uintptr_t initAddr, uint64_t initSize)
{
    std::vector<std::pair<uintptr_t, uint64_t>> originMemBlocks;
    uintptr_t startAddr = initAddr;
    uint64_t leftBytes = initSize;

    HelpSplit(startAddr, leftBytes, originMemBlocks, conf::OckSysConf::HmmConf().subMemoryLargestBaseSize);
    HelpSplit(startAddr, leftBytes, originMemBlocks, conf::OckSysConf::HmmConf().subMemoryLargerBaseSize);
    HelpSplit(startAddr, leftBytes, originMemBlocks, conf::OckSysConf::HmmConf().subMemoryBaseSize);

    return originMemBlocks;
}

// 计算level，添加到Set、dList并计数加一
void OckHmmSubMemoryOpUtils::AddMemBlock(const std::shared_ptr<MemBlock>& memBlock, MemPool &memPool)
{
    auto level = GetLevel(memBlock->length);
    memBlock->sizeLevel = level;
    InsertIntoAvailableMaps(memPool.availableBlocks, memPool.availableSizes, level, memBlock);
    memPool.memBlockSet.insert(memBlock);
    memPool.availableSize += memBlock->length;
}

// 从Set和dList中删除并计数减一
std::set<std::shared_ptr<MemBlock>, Compare>::iterator OckHmmSubMemoryOpUtils::RemoveMemBlock(
    const std::shared_ptr<MemBlock> &memBlock, MemPool &memPool)
{
    auto level = memBlock->sizeLevel;
    auto iter = memPool.memBlockSet.find(memBlock);
    if (iter == memPool.memBlockSet.end()) {
        return iter;
    }
    memPool.availableSize -= memBlock->length;
    RemoveFromAvailableMaps(memPool.availableBlocks, memPool.availableSizes, level, memBlock);
    return memPool.memBlockSet.erase(iter);
}

// 检查空间是否够用
OckHmmErrorCode OckHmmSubMemoryOpUtils::CheckAvailable(size_t size, MemPool &memPool, const std::string &memName)
{
    if (size > memPool.availableSize) {
        OCK_HMM_LOG_WARN("Fail to Allocate using AscendSubMemAllocator! With alloc type:" << memName <<
                          ", size = " << size << ", availableSize = " << memPool.availableSize);
        return HMM_ERROR_DEVICE_BUFFER_SPACE_NOT_ENOUGH;
    }
    return HMM_SUCCESS;
}

OckHmmErrorCode OckHmmSubMemoryOpUtils::TryToAllocFromCurMemBlock(
    const std::shared_ptr<MemBlock>& curBlock, uint64_t sizeAligned, uintptr_t& memAddress, MemPool &memPool)
{
    if (curBlock->length == sizeAligned) {
        OCK_HMM_LOG_DEBUG("curBlock:" << *curBlock << ", sizeAligned = " << sizeAligned << ", alloc the whole block");
        memAddress = curBlock->startAddress;
        RemoveMemBlock(curBlock, memPool);
        return HMM_SUCCESS;
    } else if (curBlock->length > sizeAligned) {
        OCK_HMM_LOG_DEBUG("curBlock:" << *curBlock << ", sizeAligned = " << sizeAligned << ", alloc the left part");
        memAddress = curBlock->startAddress;
        auto newMemBlockPtr = std::make_shared<MemBlock>();
        newMemBlockPtr->startAddress = curBlock->startAddress + sizeAligned;
        newMemBlockPtr->endAddress = curBlock->endAddress;
        newMemBlockPtr->length = curBlock->length - sizeAligned;
        RemoveMemBlock(curBlock, memPool);
        AddMemBlock(newMemBlockPtr, memPool);
        return HMM_SUCCESS;
    } else {
        OCK_HMM_LOG_ERROR("Cannot alloc from current block " << *curBlock << ", sizeAligned = " << sizeAligned
                          << ". Please check the outer level logic to see whether having started from the level"
                             " not lower than the acquired block.");
        return HMM_ERROR_SPACE_NOT_ENOUGH;
    }
}

void OckHmmSubMemoryOpUtils::TryMergeWithAvailableBlocks(const std::shared_ptr<MemBlock>& newBlock, MemPool &memPool)
{
    // 第一个首地址比new大的块，如果没有，则返回end(注意，end并非最后一个块)
    auto iterL = memPool.memBlockSet.lower_bound(newBlock);
    // 如果set为空或者set中所有现有块的首地址都比新块的小，就没有块可能从后面合并到新块中，否则可以尝试合并后面的块
    if (!memPool.memBlockSet.empty() && iterL != memPool.memBlockSet.end()
                                     && newBlock->endAddress == (*iterL)->startAddress) {
        newBlock->endAddress = (*iterL)->endAddress;
        newBlock->length += (*iterL)->length;
        iterL = RemoveMemBlock(*iterL, memPool);
    }
    // 如果set为空或者set中所有现有块的首地址都比新块的大，就没有块可能从前面合并到新块中，否则可以尝试合并前面的块
    if (!memPool.memBlockSet.empty() && iterL != memPool.memBlockSet.begin()
                                     && (*(--iterL))->endAddress == newBlock->startAddress) {
        newBlock->startAddress = (*iterL)->startAddress;
        newBlock->length += (*iterL)->length;
        RemoveMemBlock(*iterL, memPool);
    }
}

void OckHmmSubMemoryOpUtils::InsertIntoAvailableMaps(
    std::map<size_t, std::unordered_set<std::shared_ptr<MemBlock>>> &blocksMap, std::map<size_t, uint64_t> &sizesMap,
    size_t level, const std::shared_ptr<MemBlock> &memBlock)
{
    auto iter = blocksMap.find(level);
    if (iter == blocksMap.end()) {
        blocksMap[level] = std::unordered_set<std::shared_ptr<MemBlock>>({ memBlock });
        sizesMap[level] = memBlock->length;
    } else {
        blocksMap[level].insert(memBlock);
        sizesMap[level] += memBlock->length;
    }
    OCK_HMM_LOG_DEBUG("level = " << level << ", sizesMap[level] = " << sizesMap[level]);
}

void OckHmmSubMemoryOpUtils::RemoveFromAvailableMaps(
    std::map<size_t, std::unordered_set<std::shared_ptr<MemBlock>>> &blocksMap, std::map<size_t, uint64_t> &sizesMap,
    size_t level, const std::shared_ptr<MemBlock> &memBlock)
{
    auto iter = blocksMap.find(level);
    if (iter == blocksMap.end()) {
        return;
    }
    if (sizesMap[level] >= memBlock->length) {
        sizesMap[level] -= memBlock->length;
    } else {
        OCK_HMM_LOG_ERROR("Cannot do subtraction! sizesMap[level] (" << sizesMap[level] << ") < memBlock->length (" <<
            memBlock->length << ")");
    }
    iter->second.erase(memBlock);
    if (iter->second.empty()) {
        blocksMap.erase(level);
        sizesMap.erase(level);
    }
}

void OckHmmSubMemoryOpUtils::HelpSplit(uintptr_t &startAddr, uint64_t &leftBytes,
                                       std::vector<std::pair<uintptr_t, uint64_t>> &blocks,
                                       uint64_t stageBytes)
{
    if (leftBytes == 0) {
        return;
    }
    uint64_t blockBytes = utils::SafeRoundDown(leftBytes, stageBytes);
    if (blockBytes > 0) {
        blocks.emplace_back(startAddr, blockBytes);
        startAddr += blockBytes;
        leftBytes -= blockBytes;
    }
}

}  // namespace hmm
}  // namespace ock