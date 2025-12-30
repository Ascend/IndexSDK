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


#ifndef OCK_MEMORY_BRIDGE_HMM_SUB_MEMORY_ALLOC_H
#define OCK_MEMORY_BRIDGE_HMM_SUB_MEMORY_ALLOC_H
#include <cstdint>
#include <memory>
#include <ostream>
#include "ock/hmm/mgr/OckHmmHeteroMemoryLocation.h"
#include "ock/hmm/mgr/OckHmmHMObject.h"
#include "ock/acladapter/utils/OckAdapterMemoryGuard.h"
#include "ock/hmm/mgr/algo/OckHmmSubMemoryOpUtils.h"
namespace ock {
namespace hmm {

struct OckHmmMemoryUsedInfoLocal {
    OckHmmMemoryUsedInfoLocal(uint64_t usedByteCount, uint64_t unusedFragByteCount, uint64_t leftByteCount);
    uint64_t usedBytes;        // 已经使用的内存大小
    uint64_t unusedFragBytes;  // 未使用的碎片内存大小
    uint64_t leftBytes;        // 剩余内存大小
};

class OckHmmSubMemoryAlloc {
public:
    virtual ~OckHmmSubMemoryAlloc() noexcept = default;

    virtual OckHmmHeteroMemoryLocation Location(void) const = 0;
    virtual std::string Name(void) const = 0;
    virtual uintptr_t Alloc(uint64_t byteSize) = 0;
    virtual uintptr_t Alloc(uint64_t byteSize, const acladapter::OckUserWaitInfoBase &waitInfo) = 0;
    virtual void Free(uintptr_t addr, uint64_t byteSize) = 0;
    virtual std::shared_ptr<OckHmmMemoryUsedInfoLocal> GetUsedInfo(uint64_t fragThreshold) const = 0;
    virtual void IncBindMemoryToMemPool(std::unique_ptr<acladapter::OckAdapterMemoryGuard> &&incMemoryGuard,
        uint64_t byteSize, const std::string& name) = 0;

    static std::shared_ptr<OckHmmSubMemoryAlloc> Create(
        std::unique_ptr<acladapter::OckAdapterMemoryGuard> &&memoryGuard, uint64_t memorySize,
        const std::string& name = "");
};
struct OckHmmSubMemoryAllocSet {
    explicit OckHmmSubMemoryAllocSet(std::shared_ptr<OckHmmSubMemoryAlloc> devDataAlloc,
        std::shared_ptr<OckHmmSubMemoryAlloc> devSwapAlloc, std::shared_ptr<OckHmmSubMemoryAlloc> hostDataAlloc,
        std::shared_ptr<OckHmmSubMemoryAlloc> hostSwapAlloc);
    std::shared_ptr<OckHmmSubMemoryAlloc> devData;
    std::shared_ptr<OckHmmSubMemoryAlloc> devSwap;
    std::shared_ptr<OckHmmSubMemoryAlloc> hostData;
    std::shared_ptr<OckHmmSubMemoryAlloc> hostSwap;
};
std::ostream &operator<<(std::ostream &os, const OckHmmMemoryUsedInfoLocal &usedInfo);
std::ostream &operator<<(std::ostream &os, const OckHmmSubMemoryAlloc &memAlloc);
}  // namespace hmm
}  // namespace ock
#endif