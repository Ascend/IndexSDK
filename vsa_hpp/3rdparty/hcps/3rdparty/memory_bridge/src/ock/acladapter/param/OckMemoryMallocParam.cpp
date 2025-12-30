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

#include "ock/acladapter/param/OckMemoryMallocParam.h"
namespace ock {
namespace acladapter {
OckMemoryMallocParam::OckMemoryMallocParam(void **ptrAddress, size_t mallocSize,
    hmm::OckHmmHeteroMemoryLocation memLocation)
    : ptrAddr(ptrAddress), byteSize(mallocSize), location(memLocation)
{}

void **OckMemoryMallocParam::PtrAddr(void)
{
    return ptrAddr;
}
const void *OckMemoryMallocParam::Addr(void) const
{
    return *ptrAddr;
}
size_t OckMemoryMallocParam::Size(void) const
{
    return byteSize;
}
hmm::OckHmmHeteroMemoryLocation OckMemoryMallocParam::Location(void) const
{
    return location;
}
std::ostream &operator<<(std::ostream &os, const OckMemoryMallocParam &param)
{
    return os << "{'size':" << param.Size() << "',location'" << param.Location() << "}";
}
}  // namespace acladapter
}  // namespace ock