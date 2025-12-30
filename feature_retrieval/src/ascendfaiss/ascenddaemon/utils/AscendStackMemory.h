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


#ifndef ASCEND_STACK_DEVICE_MEMORY_INCLUDED
#define ASCEND_STACK_DEVICE_MEMORY_INCLUDED

#include <unordered_map>
#include <mutex>
#include <thread>
#include <memory>
#include <condition_variable>
#include "ascenddaemon/utils/AscendMemory.h"

namespace ascend {
/// Not MT Safe
class AscendStackMemory : public AscendMemory {
public:
    /// Allocate a new region of memory that we manage
    explicit AscendStackMemory(size_t alloc);

    /// For instantiating a singleton
    AscendStackMemory();

    /// Manage a region of memory for a particular device, with or
    /// without ownership
    AscendStackMemory(void* p, size_t size, bool isOwner);

    ~AscendStackMemory() override;

    // alloc memory after get singleton
    bool allocMemory(size_t sz);

    // holder bind to singleton
    void ref();

    // holder release from singleton
    void unRef();

    void resetStack();

    /// Enable or disable the warning about not having enough temporary memory
    /// when aclrtMalloc gets called
    void setMallocWarning(bool flag);

    AscendMemoryReservation getMemory(aclrtStream stream, size_t size) override;

    size_t getSizeAvailable() const override;
    std::string toString() const override;
    size_t getHighWater() const override;

protected:
    void returnAllocation(AscendMemoryReservation& m) override;

    struct Stack {
        /// Constructor that allocates memory via aclrtMalloc
        Stack(size_t sz);

        /// Constructor that references a pre-allocated region of memory
        Stack(void* p, size_t size, bool isOwner);

        virtual ~Stack();

        Stack(const Stack&) = delete;
        Stack& operator=(const Stack&) = delete;

        virtual bool alloc(size_t sz);

        virtual void reset();

        /// Returns how much size is available for an allocation without
        /// calling aclrtMalloc
        virtual size_t getSizeAvailable() const;

        /// Obtains an allocation; all allocations are guaranteed to be 8
        /// byte aligned
        virtual char* getAlloc(size_t size);

        /// Returns an allocation
        virtual void returnAlloc(char* p, size_t size, aclrtStream stream);

        /// Returns the stack state
        virtual std::string toString() const;

        /// Returns the high-water mark of aclrtMalloc activity
        virtual size_t getHighWater() const;

        /// Do we own our region of memory?
        bool isOwner;

        /// Where our allocation begins and ends
        /// [start, end) is valid
        char* start;
        char* end;

        /// Total size end - start
        size_t size;

        /// Stack head within [start, end)
        char* head;

        /// How much aclrtMalloc memory is currently outstanding?
        size_t mallocCurrent;

        /// What's the high water mark in terms of memory used from the
        /// temporary buffer?
        size_t highWaterMemoryUsed;

        /// What's the high water mark in terms of memory allocated via
        /// aclrtMalloc?
        size_t highWaterMalloc;

        /// Whether or not a warning upon aclrtMalloc is generated
        bool mallocWarning;
    };

    class ThreadSafeStack : public Stack {
    public:
        ThreadSafeStack(size_t size);

        ThreadSafeStack(void* p, size_t size, bool isOwner);

        virtual ~ThreadSafeStack();

        ThreadSafeStack(const ThreadSafeStack&) = delete;

        ThreadSafeStack& operator=(const ThreadSafeStack&) = delete;

        bool alloc(size_t size) override;

        void reset() override;

        size_t getSizeAvailable() const override;

        char* getAlloc(size_t size) override;

        void returnAlloc(char* p, size_t size, aclrtStream stream) override;

        std::string toString() const override;

        size_t getHighWater() const override;

    private:
        mutable std::recursive_mutex mtx;
        std::condition_variable_any cv;
        std::thread::id threadId;
        size_t mallocNum {0};
    };

    /// Memory stack
    std::unique_ptr<Stack> stack;

    // refrence count for holder
    int refCount;
};
}  // namespace ascend

#endif

