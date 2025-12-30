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


#include "ascenddaemon/utils/AscendStackMemory.h"
#include "ascenddaemon/utils/StaticUtils.h"
#include "ascenddaemon/utils/MemorySpace.h"
#include "common/utils/LogUtils.h"
#include "common/utils/AscendAssert.h"
#include "AscendUtils.h"

namespace ascend {
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
        AllocMemorySpace(MemorySpace::DEVICE_HUGEPAGE, &start, size);
    }

    if (start != nullptr) {
        head = start;
        end = start + size;
    }
}

AscendStackMemory::Stack::Stack(void* p, size_t sz, bool isOwner)
    : isOwner(isOwner),
      start((char*)p),
      end(((char*)p) + sz),
      size(sz),
      head((char*)p),
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
        AllocMemorySpace(MemorySpace::DEVICE_HUGEPAGE, &start, size);

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
    highWaterMemoryUsed = 0;
    mallocWarning = true;
}

size_t AscendStackMemory::Stack::getSizeAvailable() const
{
    return (end - head);
}

char* AscendStackMemory::Stack::getAlloc(size_t sz)
{
    if (sz > (size_t)(end - head)) {
        if (mallocWarning) {
            // Print our requested size before we attempt the allocation
            APP_LOG_WARNING("[ascendfaiss] increase temp memory to avoid aclrtMalloc, "
                "or decrease query/add size (alloc %zu B, highwater %zu B)\n", sz, highWaterMalloc);
        }

        char* p = nullptr;
        AllocMemorySpace(MemorySpace::DEVICE_HUGEPAGE, &p, sz);
        mallocCurrent += sz;
        highWaterMalloc = std::max(highWaterMalloc, mallocCurrent);
        return p;
    } else {
        // We can make the allocation out of our stack
        // Find all the ranges that we overlap that may have been
        // previously allocated; our allocation will be [head, endAlloc)
        char* startAlloc = head;
        char* endAlloc = head + sz;

        head = endAlloc;
        ASCEND_THROW_IF_NOT(head <= end);

        highWaterMemoryUsed = std::max(highWaterMemoryUsed, (size_t)(head - start));
        return startAlloc;
    }
}

void AscendStackMemory::Stack::returnAlloc(char* p, size_t sz, aclrtStream)
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
    }
}

std::string AscendStackMemory::Stack::toString() const
{
    std::stringstream s;

    s << "Total memory " << size << " ["
      << (void*)start << ", " << (void*)end << ")\n";
    s << "     Available memory " << (size_t)(end - head)
      << " [" << (void*)head << ", " << (void*)end << ")\n";
    s << "     High water temp alloc " << highWaterMemoryUsed << "\n";
    s << "     High water aclrtMalloc " << highWaterMalloc << "\n";

    return s.str();
}

size_t AscendStackMemory::Stack::getHighWater() const
{
    return highWaterMalloc;
}

AscendStackMemory::ThreadSafeStack::ThreadSafeStack(size_t size) : Stack(size)
{
}

AscendStackMemory::ThreadSafeStack::ThreadSafeStack(void* p, size_t size, bool isOwner) : Stack(p, size, isOwner)
{
}

AscendStackMemory::ThreadSafeStack::~ThreadSafeStack()
{
}

bool AscendStackMemory::ThreadSafeStack::alloc(size_t size)
{
    std::unique_lock<std::recursive_mutex> lock(mtx);
    cv.wait(lock, [this] {
        return (mallocNum == 0);
    });
    return Stack::alloc(size);
}

void AscendStackMemory::ThreadSafeStack::reset()
{
    std::unique_lock<std::recursive_mutex> lock(mtx);
    cv.wait(lock, [this] {
        return (mallocNum == 0);
    });
    return Stack::reset();
}

size_t AscendStackMemory::ThreadSafeStack::getSizeAvailable() const
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    return Stack::getSizeAvailable();
}

char* AscendStackMemory::ThreadSafeStack::getAlloc(size_t size)
{
    std::unique_lock<std::recursive_mutex> lock(mtx);
    auto curId = std::this_thread::get_id();
    cv.wait(lock, [curId, this] {
        return ((mallocNum == 0) || (threadId == curId));
    });
    threadId = curId;
    mallocNum++;
    return Stack::getAlloc(size);
}

void AscendStackMemory::ThreadSafeStack::returnAlloc(char *p, size_t size, aclrtStream stream)
{
    std::unique_lock<std::recursive_mutex> lock(mtx);
    mallocNum--;
    Stack::returnAlloc(p, size, stream);
    if (mallocNum == 0) {
        lock.unlock();
        cv.notify_all();
    }
}

std::string AscendStackMemory::ThreadSafeStack::toString() const
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    return Stack::toString();
}

size_t AscendStackMemory::ThreadSafeStack::getHighWater() const
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    return Stack::getHighWater();
}

bool AscendStackMemory::allocMemory(size_t sz)
{
    return stack->alloc(sz);
}

void AscendStackMemory::ref()
{
    ++refCount;
}

void AscendStackMemory::unRef()
{
    --refCount;
    if (refCount == 0) {
        stack->reset();
    }
}

void AscendStackMemory::resetStack()
{
    stack->reset();
}

AscendStackMemory::AscendStackMemory()
    : refCount(0)
{
    if (AscendMultiThreadManager::IsMultiThreadMode()) {
        stack = std::make_unique<ThreadSafeStack>(0);
    } else {
        stack = std::make_unique<Stack>(0);
    }
}

AscendStackMemory::AscendStackMemory(size_t alloc)
    : refCount(0)
{
    if (AscendMultiThreadManager::IsMultiThreadMode()) {
        stack = std::make_unique<ThreadSafeStack>(alloc);
    } else {
        stack = std::make_unique<Stack>(alloc);
    }
}

AscendStackMemory::AscendStackMemory(void* p, size_t size, bool isOwner)
    : refCount(0)
{
    if (AscendMultiThreadManager::IsMultiThreadMode()) {
        stack = std::make_unique<ThreadSafeStack>(p, size, isOwner);
    } else {
        stack = std::make_unique<Stack>(p, size, isOwner);
    }
}

AscendStackMemory::~AscendStackMemory()
{
}

void AscendStackMemory::setMallocWarning(bool flag)
{
    stack->mallocWarning = flag;
}

AscendMemoryReservation AscendStackMemory::getMemory(aclrtStream stream, size_t size)
{
#ifdef HOSTCPU
    // use 512 rather than 32 byte to achieve optimal performance
    size = utils::roundUp(size, static_cast<size_t>(512));
#else
    // We guarantee 32 byte alignment for allocations, so bump up `size`
    // to the next highest multiple of 32
    size = utils::roundUp(size, static_cast<size_t>(32));
#endif
    return AscendMemoryReservation(this,
                                   stack->getAlloc(size),
                                   size,
                                   stream);
}

size_t AscendStackMemory::getSizeAvailable() const
{
    return stack->getSizeAvailable();
}

std::string AscendStackMemory::toString() const
{
    return stack->toString();
}

size_t AscendStackMemory::getHighWater() const
{
    return stack->getHighWater();
}

void AscendStackMemory::returnAllocation(AscendMemoryReservation& m)
{
    ASCEND_THROW_IF_NOT(m.get());

    stack->returnAlloc((char*) m.get(), m.size(), m.stream());
}
}  // namespace ascend