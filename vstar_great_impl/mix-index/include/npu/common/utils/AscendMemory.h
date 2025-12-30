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


#ifndef ASCEND_ASCENDMEMORY_INCLUDED
#define ASCEND_ASCENDMEMORY_INCLUDED

#include <string>

#include "acl/acl.h"

namespace ascendSearchacc {

class AscendMemory;

class AscendMemoryReservation {
public:
    AscendMemoryReservation();

    AscendMemoryReservation(AscendMemory *mem, void *data, size_t dataSize, aclrtStream stream);

    AscendMemoryReservation(AscendMemoryReservation &&m) noexcept;

    ~AscendMemoryReservation();

    AscendMemoryReservation &operator=(AscendMemoryReservation &&m);

    void *get();

    size_t size();

    aclrtStream stream();

public:
    void *dataPtr;
    size_t dataSize;
    aclrtStream aclStream;

private:
    AscendMemory *parent;
};

class AscendMemory {
public:
    virtual ~AscendMemory();

    // / Obtains a temporary memory allocation for our device,
    // / whose usage is ordered with respect to the given stream.
    virtual AscendMemoryReservation getMemory(aclrtStream stream, size_t size) = 0;

    // / Returns the current size available without calling aclrtMalloc
    virtual size_t getSizeAvailable() const = 0;

    // / Returns a string containing our current memory manager state
    virtual std::string toString() const = 0;

    // / Returns the high-water mark of aclrtMalloc allocations for our device
    virtual size_t getHighWater() const = 0;

protected:
    friend class AscendMemoryReservation;

    virtual void returnAllocation(AscendMemoryReservation &m) = 0;
};
}  // namespace ascendSearchacc

#endif  // ASCEND_ASCENDMEMORY_H
