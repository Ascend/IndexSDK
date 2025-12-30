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

#ifndef OCK_MEMORY_BRIDGE_HMM_MGR_MOCK_OCK_HMM_MEMORY_GUARD_H
#define OCK_MEMORY_BRIDGE_HMM_MGR_MOCK_OCK_HMM_MEMORY_GUARD_H
#include <gmock/gmock.h>
#include "ock/hmm/mgr/OckHmmMemoryPool.h"
namespace ock {
namespace hmm {

class MockOckHmmMemoryGuard : public OckHmmMemoryGuard {
public:
    MOCK_CONST_METHOD0(ByteSize, uint64_t());
    MOCK_CONST_METHOD0(Addr, uintptr_t());
    MOCK_CONST_METHOD0(Location, OckHmmHeteroMemoryLocation());
};

}  // namespace hmm
}  // namespace ock
#endif