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


#ifndef PURE_DEV_MEM_STRATEGY_H
#define PURE_DEV_MEM_STRATEGY_H

#include "DevVecMemStrategyIntf.h"
#include "MemorySpace.h"

namespace ascend {

template<typename T, typename P>
class PureDevMemStrategy : public DevVecMemStrategyIntf<T> {
public:
    explicit PureDevMemStrategy(MemorySpace space) : space(space) {}

    virtual ~PureDevMemStrategy()
    {
        Clear();
    }

    PureDevMemStrategy(const PureDevMemStrategy&) = delete;
    PureDevMemStrategy& operator=(const PureDevMemStrategy&) = delete;

    void Clear() override
    {
        if (this->dataPtr != nullptr) {
            FreeMemorySpace(space, static_cast<void *>(this->dataPtr));
            this->dataPtr = nullptr;
        }

        this->num = 0;
        this->vecCapacity = 0;
    }

    size_t Size() const override
    {
        return num;
    }

    size_t Capacity() const override
    {
        return vecCapacity;
    }

    T* Data() const override
    {
        return dataPtr;
    }

    T& operator[](size_t pos) override
    {
        ASCEND_THROW_IF_NOT(pos >= 0 && pos < num);
        return *(this->dataPtr + pos);
    }

    const T& operator[](size_t pos) const override
    {
        ASCEND_THROW_IF_NOT(pos >= 0 && pos < num);
        return *(this->dataPtr + pos);
    }

    std::vector<T> CopyToStlVector() const override
    {
        if ((this->num == 0) || (this->dataPtr == nullptr)) {
            return std::vector<T>();
        }

        std::vector<T> out(this->num);
        ASCEND_THROW_IF_NOT((this->num * sizeof(T)) < MEMCPY_S_THRESHOLD);
        auto ret = memcpy_s(out.data(), this->num * sizeof(T), this->dataPtr, this->num * sizeof(T));
        ASCEND_THROW_IF_NOT_FMT(ret == EOK, "Mem operator error %d", static_cast<int>(ret));

        return out;
    }

    void Append(const T* d, size_t n, bool reserveExact = false) override
    {
        if ((d == nullptr) || (n <= 0)) {
            return;
        }

        size_t reserveSize = this->num + n;
        if (!reserveExact) {
            reserveSize = expendPolicy(reserveSize);
        }

        Reserve(reserveSize);
        ASCEND_THROW_IF_NOT((this->num * sizeof(T)) < MEMCPY_S_THRESHOLD);
        auto ret = aclrtMemcpy(this->dataPtr + this->num, n * sizeof(T), d, n * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
        ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Mem operator error %d", static_cast<int>(ret));

        this->num += n;
    }

    void Resize(size_t newSize, bool reserveExact = false) override
    {
        if (this->num < newSize) {
            if (reserveExact) {
                Reserve(newSize);
            } else {
                Reserve(expendPolicy(newSize));
            }
        }
        this->num = newSize;
    }

    size_t Reclaim(bool exact) override
    {
        size_t freeSize = this->vecCapacity - this->num;

        if (exact) {
            this->Realloc(this->num);
            return freeSize * sizeof(T);
        }

        // If more than 1/4th of the space is free, then we want to
        // truncate to only having 1/8th of the space free; this still
        // preserves some space for new elements, but won't force us to
        // double our size right away
        constexpr size_t reclaimProportion = 4;
        constexpr size_t truncateProportion = 8;
        if (freeSize > (this->vecCapacity / reclaimProportion)) {
            size_t newFreeSize = this->vecCapacity / truncateProportion;
            size_t newCapacity = this->num + newFreeSize;

            size_t oldCapacity = this->vecCapacity;
            ASCEND_THROW_IF_NOT(newCapacity < oldCapacity);

            this->Realloc(newCapacity);

            return (oldCapacity - newCapacity) * sizeof(T);
        }

        return 0;
    }

    void Reserve(size_t newCapacity) override
    {
        if (newCapacity > this->vecCapacity) {
            this->Realloc(newCapacity);
        }
    }

private:
    void Realloc(size_t newCapacity)
    {
        ASCEND_THROW_IF_NOT(this->num <= newCapacity);
        ASCEND_THROW_IF_NOT(this->num * sizeof(T) < MEMCPY_S_THRESHOLD);
        T *newData = nullptr;
        if (newCapacity == 0) {
            FreeMemorySpace(space, static_cast<void *>(this->dataPtr));
            this->dataPtr = newData;
            this->vecCapacity = newCapacity;
            return;
        }
        AllocMemorySpace(space, &newData, newCapacity * sizeof(T));

        if (this->dataPtr != nullptr) {
#ifdef HOSTCPU
            if (this->num > 0) {
                auto ret = aclrtMemcpy(newData, newCapacity * sizeof(T),
                    this->dataPtr, this->num * sizeof(T), ACL_MEMCPY_DEVICE_TO_DEVICE);
                if (ret != ACL_SUCCESS) {
                    FreeMemorySpace(space, static_cast<void *>(newData));
                    ASCEND_THROW_FMT("aclrtMemcpy operator error %d", ret);
                }
            }
#else
            auto ret = memcpy_s(newData, newCapacity * sizeof(T), this->dataPtr, this->num * sizeof(T));
            if (ret != ACL_SUCCESS) {
                FreeMemorySpace(space, static_cast<void *>(newData));
                ASCEND_THROW_FMT("memcpy_s operator error %d", ret);
            }
#endif
            FreeMemorySpace(space, static_cast<void *>(this->dataPtr));
        }

        this->dataPtr = newData;
        this->vecCapacity = newCapacity;
    }

private:
    T* dataPtr { nullptr };
    size_t num { 0 };
    size_t vecCapacity { 0 };
    MemorySpace space { MemorySpace::DEVICE };
    P expendPolicy;
};

}  // namespace ascend

#endif  // PURE_DEV_MEM_STRATEGY_H