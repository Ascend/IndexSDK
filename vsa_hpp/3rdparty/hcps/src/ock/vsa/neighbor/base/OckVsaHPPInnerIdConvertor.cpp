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

#include "ock/vsa/neighbor/base/OckVsaHPPInnerIdConvertor.h"
namespace ock {
namespace vsa {
namespace neighbor {
namespace adapter {
OckVsaHPPIdx::OckVsaHPPIdx(uint32_t groupId, uint32_t offsetInGroup) : grpId(groupId), offset(offsetInGroup) {}

OckVsaHPPInnerIdConvertor::OckVsaHPPInnerIdConvertor(uint32_t offsetBitNum)
    : offsetMask(1 << offsetBitNum), groupMask(0xFFFFFFFFFFFFFFFF ^ offsetMask), offsetBitCount(offsetBitNum)
{}

uint32_t OckVsaHPPInnerIdConvertor::CalcBitCount(uint64_t rowCount)
{
    uint32_t ret = 0;
    uint64_t value = 1ULL;
    while (value < rowCount) {
        value <<= 1U;
        ret += 1U;
    }
    return ret;
}
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock