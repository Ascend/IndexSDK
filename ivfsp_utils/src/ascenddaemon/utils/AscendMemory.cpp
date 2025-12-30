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


#include <ascenddaemon/utils/AscendMemory.h>

#include <ascenddaemon/utils/AscendUtils.h>

namespace ascendSearch {
AscendMemoryReservation::AscendMemoryReservation() : parent(nullptr), dataPtr(nullptr),
    dataSize(0), aclStream(nullptr) {}

AscendMemoryReservation::AscendMemoryReservation(AscendMemory *mem, void *data, size_t dSize, aclrtStream stream)
    : parent(mem), dataPtr(data), dataSize(dSize), aclStream(stream)
{}

AscendMemoryReservation::AscendMemoryReservation(AscendMemoryReservation &&m) noexcept
{
    parent = m.parent;
    dataPtr = m.dataPtr;
    dataSize = m.dataSize;
    aclStream = m.aclStream;

    m.parent = nullptr;
    m.dataPtr = nullptr;
    m.aclStream = nullptr;
}

AscendMemoryReservation::~AscendMemoryReservation()
{
    if (dataPtr) {
        parent->returnAllocation(*this);
    }

    dataPtr = nullptr;
}

AscendMemoryReservation &AscendMemoryReservation::operator = (ascendSearch::AscendMemoryReservation &&m)
{
    if (dataPtr) {
        ASCEND_THROW_IF_NOT(parent);
        parent->returnAllocation(*this);
    }
    parent = m.parent;
    dataPtr = m.dataPtr;
    dataSize = m.dataSize;
    aclStream = m.aclStream;

    m.parent = nullptr;
    m.dataPtr = nullptr;
    m.aclStream = nullptr;

    return *this;
}

void *AscendMemoryReservation::get()
{
    return dataPtr;
}

size_t AscendMemoryReservation::size() const
{
    return dataSize;
}

aclrtStream AscendMemoryReservation::stream()
{
    return aclStream;
}

AscendMemory::~AscendMemory() {}
} // namespace ascendSearch
