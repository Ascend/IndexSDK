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


#include <npu/common/utils/AscendStackMemory.h>
#include <npu/common/utils/StaticUtils.h>
#include <npu/common/utils/MemorySpace.h>
#include <npu/common/utils/LogUtils.h>
#include <npu/common/utils/AscendAssert.h>

namespace ascendSearchacc {
AscendStackMemory::Stack::Stack(size_t sz)
    : isOwner(true),
      start(nullptr),
      end(nullptr),
      size(sz),
      head(nullptr),
      mallocCurrent(0),
      highWaterMemoryUsed(0),
      highWaterMalloc(0),
      mallocWarning(true)
{
    if (size > 0) {
        allocMemorySpace(MemorySpace::DEVICE_HUGEPAGE, &start, size);
    }

    if (start != nullptr) {
        head = start;
        end = start + size;
    }
}

AscendStackMemory::Stack::Stack(void *p, size_t sz, bool isOwner)
    : isOwner(isOwner),
      start((char *)p),
      end(((char *)p) + sz),
      size(sz),
      head((char *)p),
      mallocCurrent(0),
      highWaterMemoryUsed(0),
      highWaterMalloc(0),
      mallocWarning(true)
{
}

AscendStackMemory::Stack::~Stack()
{
    reset();
}

bool AscendStackMemory::Stack::alloc(size_t sz)
{
    if (sz != size && sz > 0) {
        reset();

        size = sz;
        allocMemorySpace(MemorySpace::DEVICE_HUGEPAGE, &start, size);

        head = start;
        end = start + size;
    }

    if (sz == 0) {
        reset();
    }

    return true;
}

void AscendStackMemory::Stack::reset()
{
    if (isOwner && start != nullptr) {
        FreeMemorySpace(MemorySpace::DEVICE_HUGEPAGE, start);
    }

    isOwner = true;
    start = nullptr;
    end = nullptr;
    size = 0;
    head = nullptr;
    mallocCurrent = 0;
    highWaterMemoryUsed = 0;
    highWaterMalloc = 0;
    mallocWarning = true;
    lastUsers.clear();
}

size_t AscendStackMemory::Stack::getSizeAvailable() const
{
    return (end - head);
}

char *AscendStackMemory::Stack::getAlloc(size_t sz)
{
    if (sz > static_cast<size_t>(end - head)) {
        if (mallocWarning) {
            // Print our requested size before we attempt the allocation
            APP_LOG_WARNING("[ascendfaiss] increase temp memory to avoid aclrtMalloc, "
                            "or decrease query/add size (alloc %zu B, highwater %zu B)\n",
                            sz, highWaterMalloc);
        }

        char *p = nullptr;
        allocMemorySpace(MemorySpace::DEVICE_HUGEPAGE, &p, sz);
        mallocCurrent += sz;
        highWaterMalloc = std::max(highWaterMalloc, mallocCurrent);
        return p;
    } else {
        // We can make the allocation out of our stack
        // Find all the ranges that we overlap that may have been
        // previously allocated; our allocation will be [head, endAlloc)
        char *startAlloc = head;
        char *endAlloc = head + sz;

        while (lastUsers.size() > 0) {
            auto &prevUser = lastUsers.back();

            // Because there is a previous user, we must overlap it
            ASCEND_THROW_IF_NOT(prevUser.start <= endAlloc && prevUser.end >= startAlloc);

            // stream != prevUser.stream never happened  [before 2021-03]
            // Synchronization required, never come here [before 2021-03]
            // After aclrtMalloc by some stream, memory can used for all device [2021-03]
            // No need to care stream of prev-user

            if (endAlloc < prevUser.end) {
                // Update the previous user info
                prevUser.start = endAlloc;
                break;
            }

            // If we're the exact size of the previous request, then we
            // don't need to continue
            bool done = (prevUser.end == endAlloc);
            lastUsers.pop_back();
            if (done) {
                break;
            }
        }

        head = endAlloc;
        ASCEND_THROW_IF_NOT(head <= end);

        highWaterMemoryUsed = std::max(highWaterMemoryUsed, static_cast<size_t>(head - start));
        return startAlloc;
    }
}

void AscendStackMemory::Stack::returnAlloc(char *p, size_t sz, aclrtStream stream)
{
    if (p < start || p >= end) {
        // This is not on our stack; it was a one-off allocation
        FreeMemorySpace(MemorySpace::DEVICE_HUGEPAGE, p);
        ASCEND_THROW_IF_NOT(mallocCurrent >= sz);
        mallocCurrent -= sz;
    } else {
        // This is on our stack
        // Allocations should be freed in the reverse order they are made
        ASCEND_THROW_IF_NOT(p + sz == head);

        head = p;
        lastUsers.emplace_back(Range(p, p + sz, stream));
    }
}

std::string AscendStackMemory::Stack::toString() const
{
    std::stringstream s;

    s << "Total memory " << size << " [" << static_cast<void *>(start) << ", " << static_cast<void *>(end) << ")\n";
    s << "     Available memory " << static_cast<size_t>(end - head) << " [" << static_cast<void *>(head) << ", "
      << static_cast<void *>(end) << ")\n";
    s << "     High water temp alloc " << highWaterMemoryUsed << "\n";
    s << "     High water aclrtMalloc " << highWaterMalloc << "\n";

    size_t i = lastUsers.size();
    for (auto it = lastUsers.rbegin(); it != lastUsers.rend(); ++it) {
        s << i-- << ": size " << static_cast<size_t>(it->end - it->start) << " stream " << it->stream << " ["
          << static_cast<void *>(it->start) << ", " << static_cast<void *>(it->end) << ")\n";
    }

    return s.str();
}

size_t AscendStackMemory::Stack::getHighWater() const
{
    return highWaterMalloc;
}

AscendStackMemory::AscendStackMemory() : stack(0), refCount(0)
{
}

bool AscendStackMemory::allocMemory(size_t sz)
{
    return stack.alloc(sz);
}

void AscendStackMemory::ref()
{
    ++refCount;
}

void AscendStackMemory::unRef()
{
    --refCount;
    if (refCount == 0) {
        stack.reset();
    }
}

void AscendStackMemory::resetStack()
{
    stack.reset();
}

AscendStackMemory::AscendStackMemory(size_t alloc) : stack(alloc), refCount(0)
{
}

AscendStackMemory::AscendStackMemory(void *p, size_t size, bool isOwner) : stack(p, size, isOwner), refCount(0)
{
}

AscendStackMemory::~AscendStackMemory()
{
}

void AscendStackMemory::setMallocWarning(bool flag)
{
    stack.mallocWarning = flag;
}

AscendMemoryReservation AscendStackMemory::getMemory(aclrtStream stream, size_t size)
{
    // We guarantee 32 byte alignment for allocations, so bump up `size`
    // to the next highest multiple of 32
    size = utils::roundUp(size, static_cast<size_t>(32));
    return AscendMemoryReservation(this, stack.getAlloc(size), size, stream);
}

size_t AscendStackMemory::getSizeAvailable() const
{
    return stack.getSizeAvailable();
}

std::string AscendStackMemory::toString() const
{
    return stack.toString();
}

size_t AscendStackMemory::getHighWater() const
{
    return stack.getHighWater();
}

void AscendStackMemory::returnAllocation(AscendMemoryReservation &m)
{
    ASCEND_THROW_IF_NOT(m.get());

    stack.returnAlloc((char *)m.get(), m.size(), m.stream());
}
}  // namespace ascendSearchacc