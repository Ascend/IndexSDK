; /*
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

#include "ock/hmm/mgr/task/OckGetBufferParam.h"
namespace ock {
namespace hmm {
OckGetBufferParam::OckGetBufferParam(std::shared_ptr<OckHmmSubMemoryAlloc> memoryAlloc, uint64_t bufferOffset,
    uint64_t size, OckHmmHMObjectExt &srcHmoObj)
    : memAlloc(memoryAlloc), offset(bufferOffset), length(size), srcHmo(srcHmoObj)
{}
std::shared_ptr<OckHmmSubMemoryAlloc> OckGetBufferParam::MemAlloc(void)
{
    return memAlloc;
}
uint64_t OckGetBufferParam::Offset(void) const
{
    return offset;
}
uint64_t OckGetBufferParam::Length(void) const
{
    return length;
}
OckHmmHMObjectExt &OckGetBufferParam::SrcHmo(void)
{
    return srcHmo;
}
const OckHmmHMObjectExt &OckGetBufferParam::SrcHmo(void) const
{
    return srcHmo;
}
std::ostream &operator<<(std::ostream &os, const OckGetBufferParam &param)
{
    return os << "{'offset':" << param.Offset() << ",'length':" << param.Length() << ",'hmo':" << param.SrcHmo() << "}";
}
}  // namespace hmm
}  // namespace ock