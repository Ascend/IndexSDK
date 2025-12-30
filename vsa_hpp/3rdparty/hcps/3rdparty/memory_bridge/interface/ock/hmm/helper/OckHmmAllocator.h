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

#ifndef OCK_MEMORY_BRIDGE_OCK_HMM_ALLOCATOR_H
#define OCK_MEMORY_BRIDGE_OCK_HMM_ALLOCATOR_H
#include <memory>
#include <vector>
#include "ock/hmm/mgr/OckHmmMemoryPool.h"

namespace ock {
namespace hmm {
namespace helper {
template <typename _Tp>
class OckHmmAllocator {
public:
    typedef _Tp value_type;
public:
    OckHmmAllocator(OckHmmMemoryPool &hmmMemPool) : hmmPool(hmmMemPool)
    {}
    _Tp *allocate(size_t nCount)
    {
        return reinterpret_cast<_Tp *>(hmmPool.AllocateHost(nCount * sizeof(_Tp)));
    }
    void deallocate(_Tp *addr, size_t nCount)
    {
        hmmPool.DeallocateHost(reinterpret_cast<uint8_t *>(addr), nCount * sizeof(_Tp));
    }

private:
    OckHmmMemoryPool &hmmPool;
};

using OckHmmUint8Allocator = hmm::helper::OckHmmAllocator<uint8_t>;
using OckHmmUint16Allocator = hmm::helper::OckHmmAllocator<uint16_t>;
using OckHmmUint32Allocator = hmm::helper::OckHmmAllocator<uint32_t>;
using OckHmmUint64Allocator = hmm::helper::OckHmmAllocator<uint64_t>;
using OckHmmUint8Vector = std::vector<uint8_t, OckHmmUint8Allocator>;
using OckHmmUint16Vector = std::vector<uint16_t, OckHmmUint16Allocator>;
using OckHmmUint32Vector = std::vector<uint32_t, OckHmmUint32Allocator>;
using OckHmmUint64Vector = std::vector<uint64_t, OckHmmUint64Allocator>;
using OckHmmInt8Allocator = hmm::helper::OckHmmAllocator<int8_t>;
using OckHmmInt16Allocator = hmm::helper::OckHmmAllocator<int16_t>;
using OckHmmInt32Allocator = hmm::helper::OckHmmAllocator<int32_t>;
using OckHmmInt64Allocator = hmm::helper::OckHmmAllocator<int64_t>;
using OckHmmInt8Vector = std::vector<int8_t, OckHmmInt8Allocator>;
using OckHmmInt16Vector = std::vector<int16_t, OckHmmInt16Allocator>;
using OckHmmInt32Vector = std::vector<int32_t, OckHmmInt32Allocator>;
using OckHmmInt64Vector = std::vector<int64_t, OckHmmInt64Allocator>;

}  // namespace helper
}  // namespace hmm
}  // namespace ock
#endif