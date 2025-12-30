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


#ifndef OCK_MEMORY_BRIDGE_HMM_MGR_MOCK_OCK_HMM_SUB_MEMORY_ALLOC_H
#define OCK_MEMORY_BRIDGE_HMM_MGR_MOCK_OCK_HMM_SUB_MEMORY_ALLOC_H
#include <gmock/gmock.h>
#include <string>
#include "ock/hmm/mgr/OckHmmSubMemoryAlloc.h"

namespace ock {
namespace hmm {

class MockOckHmmSubMemoryAlloc : public OckHmmSubMemoryAlloc {
public:
    MOCK_CONST_METHOD0(Location, OckHmmHeteroMemoryLocation());
    MOCK_CONST_METHOD0(Name, std::string());
    MOCK_METHOD1(Alloc, uintptr_t(uint64_t byteSize));
    MOCK_METHOD2(Alloc, uintptr_t(uint64_t byteSize, const acladapter::OckUserWaitInfoBase &waitInfo));
    MOCK_METHOD2(Free, void(uintptr_t addr, uint64_t byteSize));
    MOCK_CONST_METHOD1(GetUsedInfo, std::shared_ptr<OckHmmMemoryUsedInfoLocal>(uint64_t));
    MOCK_METHOD3(IncBindMemoryToMemPool, void(std::unique_ptr<acladapter::OckAdapterMemoryGuard> &&incMemoryGuard,
        uint64_t byteSize, const std::string &name));
};

}  // namespace hmm
}  // namespace ock
#endif