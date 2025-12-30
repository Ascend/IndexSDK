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

#include "ock/hmm/mgr/OckHmmMemoryGuardExt.h"
namespace ock {
namespace hmm {
OckHmmMemoryGuardExt::~OckHmmMemoryGuardExt() noexcept
{
    memAlloc->Free(addr, byteSize);
}
OckHmmMemoryGuardExt::OckHmmMemoryGuardExt(
    std::shared_ptr<OckHmmSubMemoryAlloc> subMemAlloc, uintptr_t address, uint64_t byteCount)
    : memAlloc(subMemAlloc), addr(address), byteSize(byteCount)
{}
uintptr_t OckHmmMemoryGuardExt::OckHmmMemoryGuardExt::Addr(void) const
{
    return addr;
}
hmm::OckHmmHeteroMemoryLocation OckHmmMemoryGuardExt::OckHmmMemoryGuardExt::Location(void) const
{
    return memAlloc->Location();
}
uint64_t OckHmmMemoryGuardExt::OckHmmMemoryGuardExt::ByteSize(void) const
{
    return byteSize;
}
}  // namespace hmm
}  // namespace ock