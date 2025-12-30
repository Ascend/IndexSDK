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


#ifndef OCK_VSA_NEIGHBOR_ADAPTER_ID_CONVERTER_H
#define OCK_VSA_NEIGHBOR_ADAPTER_ID_CONVERTER_H
#include <cstdint>
#include <utility>
#include <deque>

namespace ock {
namespace vsa {
namespace neighbor {
namespace adapter {
struct OckVsaHPPIdx {
    OckVsaHPPIdx(uint32_t groupId, uint32_t offsetInGroup);
    bool operator<=(const OckVsaHPPIdx &other)
    {
        if (this->grpId < other.grpId || (this->grpId == other.grpId && this->offset <= other.offset)) {
            return true;
        } else {
            return false;
        }
    }
    uint32_t grpId;
    uint32_t offset;
};
/*
@brief 该类主要是为了将grpId与Offset组合后连续存储，从而可以减少空间利用
*/
class OckVsaHPPInnerIdConvertor {
public:
    // 这里使用22U，代表4,194,304条数据，也就是说，对于256场景，每个Group 4G数据
    explicit OckVsaHPPInnerIdConvertor(uint32_t offsetBitCount = 22U);
    // 频繁调用，使用内联，提升速度
    inline uint64_t ToIdx(uint32_t groupId, uint32_t offset) const
    {
        return ((uint64_t)groupId << offsetBitCount) + offset;
    }
    inline OckVsaHPPIdx ToGroupOffset(uint64_t idx) const
    {
        return OckVsaHPPIdx(static_cast<uint32_t>(idx >> offsetBitCount),
            static_cast<uint32_t>(idx & (offsetMask - 1ULL)));
    }

    static uint32_t CalcBitCount(uint64_t rowCount);

private:
    const uint64_t offsetMask;
    const uint64_t groupMask;
    const uint64_t offsetBitCount;
};
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif