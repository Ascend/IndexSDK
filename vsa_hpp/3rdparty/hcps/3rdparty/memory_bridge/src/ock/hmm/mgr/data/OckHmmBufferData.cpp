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

#include "ock/hmm/mgr/data/OckHmmBufferData.h"
namespace ock {
namespace hmm {
OckHmmBufferData::OckHmmBufferData(uintptr_t address, uint64_t byteCount, uint64_t offsetInBuffer,
    OckHmmHeteroMemoryLocation memLocation, OckHmmHMOObjectID objId)
    : addr(address), byteSize(byteCount), offset(offsetInBuffer), location(memLocation), hmoObjId(objId)
{}
bool operator==(const OckHmmBufferData &lhs, const OckHmmBufferData &rhs)
{
    return lhs.addr == rhs.addr && lhs.byteSize == rhs.byteSize && lhs.offset == rhs.offset &&
           lhs.location == rhs.location;
}
bool operator!=(const OckHmmBufferData &lhs, const OckHmmBufferData &rhs)
{
    return !(lhs == rhs);
}
std::ostream &operator<<(std::ostream &os, const OckHmmBufferData &data)
{
    return os << "{'byteSize':" << data.byteSize << ",'offset':" << data.offset
              << ",'location':" << data.location << "}";
}

}  // namespace hmm
}  // namespace ock