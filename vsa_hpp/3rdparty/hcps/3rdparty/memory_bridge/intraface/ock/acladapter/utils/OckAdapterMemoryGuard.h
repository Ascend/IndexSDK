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


#ifndef OCK_MEMORY_BRIDGE_OCK_ADAPTER_OCK_MEMORY_GUARD_H
#define OCK_MEMORY_BRIDGE_OCK_ADAPTER_OCK_MEMORY_GUARD_H
#include <memory>
#include "ock/hmm/mgr/OckHmmMemoryPool.h"
#include "ock/acladapter/executor/OckAsyncTaskExecuteService.h"
namespace ock {
namespace acladapter {

class OckAdapterMemoryGuard : public hmm::OckHmmMemoryGuard {
public:
    virtual ~OckAdapterMemoryGuard() noexcept = default;
    virtual uint8_t *ReleaseGuard(void) = 0;
    virtual uint8_t *GetAddr(void) = 0;

    static std::unique_ptr<OckAdapterMemoryGuard> Create(OckAsyncTaskExecuteService &service, uint8_t *guardAddr,
        uint64_t byteSize, hmm::OckHmmHeteroMemoryLocation location);
};
std::ostream &operator<<(std::ostream &os, const OckAdapterMemoryGuard &data);
}  // namespace acladapter
}  // namespace ock
#endif