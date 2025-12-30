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

#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include "ock/hmm/mgr/OckHmmErrorCode.h"
#include "ock/conf/OckSysConf.h"
#include "ock/acladapter/utils/OckSyncUtils.h"
#include "ock/hmm/mgr/OckHmmSubMemoryAlloc.h"
namespace ock {
namespace hmm {
OckHmmMemoryUsedInfoLocal::OckHmmMemoryUsedInfoLocal(uint64_t usedByteCount, uint64_t unusedFragByteCount,
    uint64_t leftByteCount)
    : usedBytes(usedByteCount), unusedFragBytes(unusedFragByteCount), leftBytes(leftByteCount)
{}
class OckHmmSubMemoryAllocImpl : public OckHmmSubMemoryAlloc {
public:
    virtual ~OckHmmSubMemoryAllocImpl() noexcept = default;
    OckHmmSubMemoryAllocImpl(std::unique_ptr<acladapter::OckAdapterMemoryGuard> &&memGuard, uint64_t memSize,
        const std::string& memName)
        : memorySize(memSize), memoryGuard(std::move(memGuard)), name(memName)
    {
        auto originMemBlocks = OckHmmSubMemoryOpUtils::GetOriginMemBlocks(this->memoryGuard->Addr(), memorySize);
        for (auto block : originMemBlocks) {
            MountToMemPool(block.first, block.second);
            memPool_.totalSize += block.second;
        }
        wastedFragBytes_ = memorySize - memPool_.totalSize;
        OCK_HMM_LOG_INFO("Name: " << name << " Submemory: " << *(this->memoryGuard) << " pool:" << memPool_);
    }

    void IncBindMemoryToMemPool(std::unique_ptr<acladapter::OckAdapterMemoryGuard> &&incMemoryGuard, uint64_t byteSize,
        const std::string &incName) override
    {
        this->incMemoryGuardVector.push_back(std::move(incMemoryGuard));
        uint64_t incMemSize = 0ULL;
        // 内存分级
        auto originMemBlocks =
            OckHmmSubMemoryOpUtils::GetOriginMemBlocks(this->incMemoryGuardVector.back()->Addr(), byteSize);
        // 内存装载
        for (auto block : originMemBlocks) {
            MountToMemPool(block.first, block.second);
            incMemSize += block.second;
        }
        // 内存池参数更新
        memPool_.totalSize += incMemSize;
        wastedFragBytes_ = byteSize - incMemSize;
        // 二次分配成员变量更新
        memorySize += byteSize;
        OCK_HMM_LOG_INFO("Name: " << incName << " Submemory: " << *(this->incMemoryGuardVector.back()) << " pool:" <<
            memPool_);
    }

    OckHmmHeteroMemoryLocation Location(void) const override
    {
        return memoryGuard->Location();
    }

    std::string Name(void) const override
    {
        return name;
    };
    // 由调用者保证 byteSize 为合理大小
    uintptr_t Alloc(uint64_t byteSize) override
    {
        std::unique_lock<std::mutex> lock(memMutex);
        uintptr_t memAddress = 0UL;
        AllocImpl(byteSize, memAddress);
        return memAddress;
    }
    // 由调用者保证 byteSize 为合理大小
    uintptr_t Alloc(uint64_t byteSize, const acladapter::OckUserWaitInfoBase &waitInfo) override
    {
        std::unique_lock<std::mutex> lock(memMutex);
        uintptr_t memAddress = 0UL;
        while (!waitInfo.WaitTimeOut()) {
            auto ret = AllocImpl(byteSize, memAddress);
            if (ret == HMM_SUCCESS) {
                return memAddress;
            }
            condVar.wait(lock);
        }
        return 0UL;
    }
    // 由调用者保证 addr 为通过此二次分配对象分配的地址 且 byteSize 为合理大小
    void Free(uintptr_t addr, uint64_t byteSize) override
    {
        std::unique_lock<std::mutex> lock(memMutex);
        MountToMemPool(addr, byteSize);
        condVar.notify_all();
    }

    // 虽然alloc/free对齐小内存块只有256和1K的倍数，但是free时可能会发生合并，因此每个level所含的内存块的大小
    // 不再一致；alloc/free时会通过availableSizes管理每个level的总大小，但是阈值所处的level，如果非空，需要
    // 遍历统计其中尺寸小于阈值的内存块
    std::shared_ptr<OckHmmMemoryUsedInfoLocal> GetUsedInfo(uint64_t fragThreshold) const override
    {
        std::unique_lock<std::mutex> lock(memMutex);
        // alloc/free的对齐带来的碎片
        uint64_t unusedFragBytes = wastedFragBytes_;
        // 连续空间小于fragThreshold的内存块也认为是碎片
        size_t thresholdLevel = OckHmmSubMemoryOpUtils::GetLevel(fragThreshold);
        auto thresholdIter = memPool_.availableBlocks.find(thresholdLevel);
        if (thresholdIter != memPool_.availableBlocks.end()) {
            for (const auto& mb : thresholdIter->second) {
                unusedFragBytes += (mb->length < fragThreshold ? mb->length : 0);
            }
        }
        for (auto pair : memPool_.availableSizes) {
            if (pair.first >= thresholdLevel) {
                break;
            }
            unusedFragBytes += pair.second;
        }
        // usedBytes, unusedFragBytes, leftBytes
        return std::make_shared<OckHmmMemoryUsedInfoLocal>(
                memorySize - memPool_.availableSize - wastedFragBytes_, unusedFragBytes,
                memPool_.availableSize + wastedFragBytes_);
    }

private:
    OckHmmErrorCode AllocImpl(uint64_t byteSize, uintptr_t &memAddress)
    {
        size_t sizeAligned = OckHmmSubMemoryOpUtils::Align(byteSize);  // 定制化对齐
        auto ret = OckHmmSubMemoryOpUtils::CheckAvailable(sizeAligned, memPool_, name);
        if (ret != HMM_SUCCESS) {
            return HMM_ERROR_DEVICE_BUFFER_SPACE_NOT_ENOUGH;
        }

        size_t startLevel = OckHmmSubMemoryOpUtils::GetLevel(sizeAligned);
        auto iter = memPool_.availableBlocks.lower_bound(startLevel);
        if (iter != memPool_.availableBlocks.end()) {
            ret = OckHmmSubMemoryOpUtils::TryToAllocFromCurMemBlock(
                *((iter->second).begin()), sizeAligned, memAddress, memPool_);
            if (ret == HMM_SUCCESS) {
                wastedFragBytes_ += (sizeAligned - byteSize);
                return HMM_SUCCESS;
            }
        }

        OCK_HMM_LOG_WARN("Allocate Fail! There is no available space(needbytes="
                          << byteSize << ") in the buffer agree with your request in " << name);
        return HMM_ERROR_DEVICE_BUFFER_SPACE_NOT_ENOUGH;
    }

    void MountToMemPool(uintptr_t address, size_t byteSize)
    {
        size_t length = OckHmmSubMemoryOpUtils::Align(byteSize);  // 定制化对齐

        auto newMemBlockPtr = std::make_shared<MemBlock>(address, address + length, length, 0);

        OckHmmSubMemoryOpUtils::TryMergeWithAvailableBlocks(newMemBlockPtr, memPool_);
        OckHmmSubMemoryOpUtils::AddMemBlock(newMemBlockPtr, memPool_);
        wastedFragBytes_ -= (length - byteSize);

        OCK_HMM_LOG_DEBUG("Memory free success" << ", length =" << length);
    }
    uint64_t memorySize;
    mutable std::mutex memMutex{};
    std::condition_variable condVar{};
    std::unique_ptr<acladapter::OckAdapterMemoryGuard> memoryGuard;
    std::vector<std::unique_ptr<acladapter::OckAdapterMemoryGuard>> incMemoryGuardVector{};
    MemPool memPool_{};
    uint64_t wastedFragBytes_{ 0ULL };
    std::string name;
};
std::shared_ptr<OckHmmSubMemoryAlloc> OckHmmSubMemoryAlloc::Create(
    std::unique_ptr<acladapter::OckAdapterMemoryGuard> &&memoryGuard, uint64_t memorySize, const std::string& name)
{
    return std::make_shared<OckHmmSubMemoryAllocImpl>(std::move(memoryGuard), memorySize, name);
}
std::ostream &operator<<(std::ostream &os, const OckHmmMemoryUsedInfoLocal &usedInfo)
{
    return os << "{'usedBytes':" << usedInfo.usedBytes << ", 'unusedFragBytes':" << usedInfo.unusedFragBytes
              << ",'leftBytes':" << usedInfo.leftBytes << "}";
}
std::ostream &operator<<(std::ostream &os, const OckHmmSubMemoryAlloc &memAlloc)
{
    auto usedInfo = memAlloc.GetUsedInfo((conf::OckSysConf::HmmConf().defaultFragThreshold));
    return os << "{'Name' is " << memAlloc.Name() << ", 'Location':" << memAlloc.Location()
              << ", 'usedInfo':" << *usedInfo << "}";
}

}  // namespace hmm
}  // namespace ock