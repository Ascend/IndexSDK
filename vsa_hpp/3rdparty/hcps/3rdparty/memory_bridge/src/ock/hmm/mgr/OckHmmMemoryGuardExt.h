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


#ifndef OCK_MEMORY_BRIDGE_HMM_MEMORY_GUARD_EXT_H
#define OCK_MEMORY_BRIDGE_HMM_MEMORY_GUARD_EXT_H
#include "ock/hmm/mgr/OckHmmMemoryPool.h"
#include "ock/hmm/mgr/OckHmmSubMemoryAlloc.h"

namespace ock {
namespace hmm {
class OckHmmMemoryGuardExt : public OckHmmMemoryGuard {
public:
    virtual ~OckHmmMemoryGuardExt() noexcept;
    explicit OckHmmMemoryGuardExt(std::shared_ptr<OckHmmSubMemoryAlloc> subMemAlloc, uintptr_t address,
        uint64_t byteCount);

    uintptr_t Addr(void) const override;
    hmm::OckHmmHeteroMemoryLocation Location(void) const override;
    uint64_t ByteSize(void) const override;

private:
    std::shared_ptr<OckHmmSubMemoryAlloc> memAlloc;
    uintptr_t addr;
    uint64_t byteSize;
};
}  // namespace hmm
}  // namespace ock
#endif