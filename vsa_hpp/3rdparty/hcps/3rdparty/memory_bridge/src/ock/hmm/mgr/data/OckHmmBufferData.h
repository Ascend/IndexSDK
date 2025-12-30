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


#ifndef OCK_MEMORY_BRIDGE_OCK_HMM_OCK_HMO_BUFFER_DATA_H
#define OCK_MEMORY_BRIDGE_OCK_HMM_OCK_HMO_BUFFER_DATA_H
#include <cstdint>
#include <ostream>
#include "ock/hmm/mgr/OckHmmHeteroMemoryLocation.h"
#include "ock/hmm/mgr/OckHmmHMObject.h"
namespace ock {
namespace hmm {

struct OckHmmBufferData {
    OckHmmBufferData(uintptr_t address, uint64_t byteCount, uint64_t offsetInBuffer,
        OckHmmHeteroMemoryLocation memLocation, OckHmmHMOObjectID objId);
    uintptr_t addr;
    uint64_t byteSize;
    uint64_t offset;
    OckHmmHeteroMemoryLocation location;
    OckHmmHMOObjectID hmoObjId;
};

bool operator==(const OckHmmBufferData &lhs, const OckHmmBufferData &rhs);
bool operator!=(const OckHmmBufferData &lhs, const OckHmmBufferData &rhs);
std::ostream &operator<<(std::ostream &os, const OckHmmBufferData &data);

}  // namespace hmm
}  // namespace ock
#endif