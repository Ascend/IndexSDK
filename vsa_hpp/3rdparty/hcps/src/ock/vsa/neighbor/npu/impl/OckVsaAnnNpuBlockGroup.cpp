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

#include "ock/vsa/OckVsaErrorCode.h"
#include "ock/log/OckHcpsLogger.h"
#include "ock/vsa/neighbor/npu/OckVsaAnnNpuBlockGroup.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace npu {

uint64_t OckVsaAnnNpuBlockGroup::GetByteSize(void) const
{
    if (BlockCount() > 0) {
        uint64_t featureByteSize = features.empty() ? 0 : features.size() * features[0]->GetByteSize();
        uint64_t normsByteSize = norms.empty() ? 0 : norms.size() * norms[0]->GetByteSize();
        uint64_t keyAttrsTimeByteSize = keyAttrsTime.empty() ? 0 : keyAttrsTime.size() * keyAttrsTime[0]->GetByteSize();
        uint64_t keyAttrsQuotientByteSize =
                keyAttrsQuotient.empty() ? 0 : keyAttrsQuotient.size() * keyAttrsQuotient[0]->GetByteSize();
        uint64_t keyAttrsRemainderByteSize =
                keyAttrsRemainder.empty() ? 0 : keyAttrsRemainder.size() * keyAttrsRemainder[0]->GetByteSize();
        uint64_t extKeyAttrsByteSize = extKeyAttrs.empty() ? 0 : extKeyAttrs.size() * extKeyAttrs[0]->GetByteSize();
        return featureByteSize + normsByteSize + keyAttrsTimeByteSize + keyAttrsQuotientByteSize
               + keyAttrsRemainderByteSize + extKeyAttrsByteSize;
    }
    return 0ULL;
}

uint32_t OckVsaAnnNpuBlockGroup::BlockCount(void) const
{
    return static_cast<uint32_t>(features.size());
}
uint64_t OckVsaAnnRawBlockInfo::GetByteSize(void) const
{
    return feature->GetByteSize() + norm->GetByteSize();
}
uint64_t OckVsaAnnKeyAttrInfo::KeyAttrTimeBytes(void)
{
    return sizeof(int32_t);
}
uint64_t OckVsaAnnKeyAttrInfo::KeyAttrQuotientBytes(void)
{
    return sizeof(uint32_t);
}
uint64_t OckVsaAnnKeyAttrInfo::KeyAttrRemainderBytes(void)
{
    return sizeof(int16_t);
}
uint64_t OckVsaAnnKeyAttrInfo::KeyAttrAllBytes(void)
{
    return KeyAttrTimeBytes() + KeyAttrQuotientBytes() + KeyAttrRemainderBytes();
}

int CountBit(uint8_t tokenRemainder)
{
    if (tokenRemainder == 0) {
        return 0;
    }
    int bitNum = 8;
    return static_cast<int>(sizeof(unsigned int) * bitNum - 1 - __builtin_clz(tokenRemainder));
}
void OckVsaAnnKeyAttrInfo::AttrCvt(
    attr::OckTimeSpaceAttr &toData, uintptr_t attrTime, uintptr_t attrQuotient, uintptr_t attrRemainder)
{
    toData.time = *(reinterpret_cast<int32_t*>(attrTime));
    uint32_t tokenQuotient = (*(reinterpret_cast<uint32_t*>(attrQuotient))) / OPS_DATA_TYPE_TIMES * __CHAR_BIT__;
    uint8_t tokenRemainder = (*(reinterpret_cast<uint8_t*>(attrRemainder)));
    int remainder = CountBit(tokenRemainder);
    toData.tokenId =  tokenQuotient + remainder;
}
}  // namespace npu
}  // namespace neighbor
}  // namespace vsa
}  // namespace ock