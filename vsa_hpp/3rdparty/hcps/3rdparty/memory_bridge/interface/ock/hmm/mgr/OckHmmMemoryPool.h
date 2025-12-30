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


#ifndef OCK_MEMORY_BRIDGE_HMM_MEMORY_POOL_H
#define OCK_MEMORY_BRIDGE_HMM_MEMORY_POOL_H
#include <memory>
#include "ock/hmm/mgr/OckHmmHeteroMemoryLocation.h"

namespace ock {
namespace hmm {
class OckHmmMemoryGuard {
public:
    virtual ~OckHmmMemoryGuard() noexcept = default;
    virtual uintptr_t Addr(void) const = 0;
    virtual hmm::OckHmmHeteroMemoryLocation Location(void) const = 0;
    virtual uint64_t ByteSize(void) const = 0;
};

std::ostream &operator<<(std::ostream &os, const OckHmmMemoryGuard &data);
class OckHmmMemoryPool {
public:
    virtual ~OckHmmMemoryPool() noexcept = default;
    virtual std::unique_ptr<OckHmmMemoryGuard> Malloc(uint64_t size, OckHmmMemoryAllocatePolicy policy) = 0;
    virtual uint8_t *AllocateHost(size_t byteSize) = 0;
    virtual void DeallocateHost(uint8_t *addr, size_t byteSize) = 0;
};
}  // namespace hmm
}  // namespace ock
#endif