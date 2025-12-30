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


#ifndef OCK_HMM_OCKHMMSUBMEMORYOPUTILS_H
#define OCK_HMM_OCKHMMSUBMEMORYOPUTILS_H

#include <mutex>
#include <set>
#include <list>
#include <vector>
#include <iterator>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include "ock/log/OckLogger.h"
#include "ock/conf/OckSysConf.h"
#include "ock/utils/OckSafeUtils.h"
#include "ock/hmm/mgr/OckHmmErrorCode.h"

namespace ock {
namespace hmm {

struct MemBlock {
    MemBlock() = default;
    MemBlock(uintptr_t startAddr, uintptr_t endAddr, size_t size, size_t level)
        : startAddress(startAddr), endAddress(endAddr), length(size), sizeLevel(level) {}
    uintptr_t startAddress{0};
    uintptr_t endAddress{0};
    size_t length{0};
    size_t sizeLevel{0};
};
bool operator==(const MemBlock &lhs, const MemBlock &rhs);
bool operator!=(const MemBlock &lhs, const MemBlock &rhs);
std::ostream &operator<<(std::ostream &os, const MemBlock& memBlock);

// 定义Set里面的序
struct Compare {
    bool operator()(const std::shared_ptr<MemBlock>& lhs, const std::shared_ptr<MemBlock>& rhs)
    {
        return lhs->startAddress < rhs->startAddress;  // 从小到大，反之从大到小
    }
};

struct MemPool {
    std::set<std::shared_ptr<MemBlock>, Compare> memBlockSet{};
    std::map<size_t, std::unordered_set<std::shared_ptr<MemBlock>>> availableBlocks{};
    std::map<size_t, uint64_t> availableSizes{};
    uint64_t totalSize{0LL};
    uint64_t availableSize{0LL};
};
std::ostream &operator <<(std::ostream &os, const MemPool &pool);
class OckHmmSubMemoryOpUtils {
public:
    static inline size_t GetLevel(size_t length)
    {
        if (length < conf::OckSysConf::HmmConf().subMemoryBaseSize) {
            length = conf::OckSysConf::HmmConf().subMemoryBaseSize;
        }
        // 256->0, 512->1, 768->2
        if (length < conf::OckSysConf::HmmConf().subMemoryLargerBaseSize) {
            return utils::SafeDivDown(length, conf::OckSysConf::HmmConf().subMemoryBaseSize) - 1;
        // [1k,2k)->3, [2k,2k)->4, ..., [1023k, 1M)->1025
        } else if (length < conf::OckSysConf::HmmConf().subMemoryLargestBaseSize) {
            return 2U + utils::SafeDivDown(length, conf::OckSysConf::HmmConf().subMemoryLargerBaseSize);
        // [1M, 2M)->1026, [2M,3M)->1027, ...
        } else {
            return 1025U + utils::SafeDivDown(length, conf::OckSysConf::HmmConf().subMemoryLargestBaseSize);
        }
    }

    static inline size_t Align(size_t length)
    {
        if (length < conf::OckSysConf::HmmConf().subMemoryLargerBaseSize) {
            return utils::SafeRoundUp(length, conf::OckSysConf::HmmConf().subMemoryBaseSize);
        } else if (length < conf::OckSysConf::HmmConf().subMemoryLargestBaseSize) {
            return utils::SafeRoundUp(length, conf::OckSysConf::HmmConf().subMemoryLargerBaseSize);
        } else {
            return utils::SafeRoundUp(length, conf::OckSysConf::HmmConf().subMemoryLargestBaseSize);
        }
    }

    // 拆分原始空间
    static std::vector<std::pair<uintptr_t, uint64_t>> GetOriginMemBlocks(uintptr_t initAddr, uint64_t initSize);

    // 计算level，添加到Set、dList并计数加一
    static void AddMemBlock(const std::shared_ptr<MemBlock>& memBlock, MemPool &memPool);

    // 从Set和dList中删除并计数减一
    static std::set<std::shared_ptr<MemBlock>, Compare>::iterator RemoveMemBlock(
        const std::shared_ptr<MemBlock> &memBlock, MemPool &memPool);

    // 检查空间是否够用
    static OckHmmErrorCode CheckAvailable(size_t size, MemPool &memPool, const std::string &memName);

    static OckHmmErrorCode TryToAllocFromCurMemBlock(
            const std::shared_ptr<MemBlock>& curBlock, uint64_t sizeAligned, uintptr_t &memAddress, MemPool &memPool);

    static void TryMergeWithAvailableBlocks(const std::shared_ptr<MemBlock>& newBlock, MemPool &memPool);

private:
    static void InsertIntoAvailableMaps(std::map<size_t, std::unordered_set<std::shared_ptr<MemBlock>>> &blocksMap,
                                        std::map<size_t, uint64_t> &sizesMap,
                                        size_t level, const std::shared_ptr<MemBlock>& memBlock);

    static void RemoveFromAvailableMaps(std::map<size_t, std::unordered_set<std::shared_ptr<MemBlock>>> &blocksMap,
                                        std::map<size_t, uint64_t> &sizesMap,
                                        size_t level, const std::shared_ptr<MemBlock>& memBlock);

    static void HelpSplit(uintptr_t &startAddr, uint64_t &leftBytes,
                          std::vector<std::pair<uintptr_t, uint64_t>> &blocks, uint64_t stageBytes);
};

}  // namespace hmm
}  // namespace ock

#endif  // OCK_HMM_OCKHMMSUBMEMORYOPUTILS_H
