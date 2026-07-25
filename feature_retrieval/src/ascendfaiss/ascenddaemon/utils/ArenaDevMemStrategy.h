/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

#ifndef ARENA_DEV_MEM_STRATEGY_H
#define ARENA_DEV_MEM_STRATEGY_H

#include <memory>

#include "DevVecMemStrategyIntf.h"
#include "DeviceMemArena.h"

namespace ascend
{

// DeviceVector strategy that bump-allocates from a shared DeviceMemArena.
// Old blocks are abandoned on grow (reclaimed on arena Reset), matching add-heavy IVF.
template <typename T, typename P>
class ArenaDevMemStrategy : public DevVecMemStrategyIntf<T>
{
   public:
    explicit ArenaDevMemStrategy(std::shared_ptr<DeviceMemArena> arena) : arena_(std::move(arena))
    {
        ASCEND_THROW_IF_NOT_MSG(arena_ != nullptr, "DeviceMemArena is null");
    }

    ~ArenaDevMemStrategy() override { Clear(); }

    ArenaDevMemStrategy(const ArenaDevMemStrategy &) = delete;
    ArenaDevMemStrategy &operator=(const ArenaDevMemStrategy &) = delete;

    void Clear() override
    {
        // Do not return memory to the driver; arena Reset reclaims slabs.
        dataPtr_ = nullptr;
        num_ = 0;
        vecCapacity_ = 0;
    }

    size_t Size() const override { return num_; }

    size_t Capacity() const override { return vecCapacity_; }

    T *Data() const override { return dataPtr_; }

    T &operator[](size_t pos) override
    {
        ASCEND_THROW_IF_NOT(pos < num_);
        return *(dataPtr_ + pos);
    }

    const T &operator[](size_t pos) const override
    {
        ASCEND_THROW_IF_NOT(pos < num_);
        return *(dataPtr_ + pos);
    }

    std::vector<T> CopyToStlVector() const override
    {
        if ((num_ == 0) || (dataPtr_ == nullptr))
        {
            return std::vector<T>();
        }

        std::vector<T> out(num_);
        ASCEND_THROW_IF_NOT((num_ * sizeof(T)) < MEMCPY_S_THRESHOLD);
#ifdef HOSTCPU
        auto ret = aclrtMemcpy(out.data(), num_ * sizeof(T), dataPtr_, num_ * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
        ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy operator error %d", ret);
#else
        auto ret = memcpy_s(out.data(), num_ * sizeof(T), dataPtr_, num_ * sizeof(T));
        ASCEND_THROW_IF_NOT_FMT(ret == EOK, "memcpy_s operator error %d", static_cast<int>(ret));
#endif
        return out;
    }

    void Append(const T *d, size_t n, bool reserveExact = false) override
    {
        if ((d == nullptr) || (n == 0))
        {
            return;
        }

        size_t reserveSize = num_ + n;
        if (!reserveExact)
        {
            reserveSize = expandPolicy_(reserveSize);
        }

        Reserve(reserveSize);
        ASCEND_THROW_IF_NOT((num_ * sizeof(T)) < MEMCPY_S_THRESHOLD);
        auto ret = aclrtMemcpy(dataPtr_ + num_, n * sizeof(T), d, n * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
        ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Mem operator error %d", static_cast<int>(ret));
        num_ += n;
    }

    void Resize(size_t newSize, bool reserveExact = false) override
    {
        if (num_ < newSize)
        {
            if (reserveExact)
            {
                Reserve(newSize);
            }
            else
            {
                Reserve(expandPolicy_(newSize));
            }
        }
        num_ = newSize;
    }

    size_t Reclaim(bool exact) override
    {
        size_t freeSize = vecCapacity_ - num_;

        if (exact)
        {
            Realloc(num_);
            return freeSize * sizeof(T);
        }

        constexpr size_t reclaimProportion = 4;
        constexpr size_t truncateProportion = 8;
        if (freeSize > (vecCapacity_ / reclaimProportion))
        {
            size_t newFreeSize = vecCapacity_ / truncateProportion;
            size_t newCapacity = num_ + newFreeSize;
            size_t oldCapacity = vecCapacity_;
            ASCEND_THROW_IF_NOT(newCapacity < oldCapacity);
            Realloc(newCapacity);
            return (oldCapacity - newCapacity) * sizeof(T);
        }

        return 0;
    }

    void Reserve(size_t newCapacity) override
    {
        if (newCapacity > vecCapacity_)
        {
            Realloc(newCapacity);
        }
    }

   private:
    void Realloc(size_t newCapacity)
    {
        ASCEND_THROW_IF_NOT(num_ <= newCapacity);
        ASCEND_THROW_IF_NOT(num_ * sizeof(T) < MEMCPY_S_THRESHOLD);

        if (newCapacity == 0)
        {
            dataPtr_ = nullptr;
            vecCapacity_ = 0;
            return;
        }

        T *newData = static_cast<T *>(arena_->Allocate(newCapacity * sizeof(T)));
        ASCEND_THROW_IF_NOT_MSG(newData != nullptr, "DeviceMemArena Allocate returned null");

        if ((dataPtr_ != nullptr) && (num_ > 0))
        {
#ifdef HOSTCPU
            auto ret =
                aclrtMemcpy(newData, newCapacity * sizeof(T), dataPtr_, num_ * sizeof(T), ACL_MEMCPY_DEVICE_TO_DEVICE);
            ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy operator error %d", ret);
#else
            auto ret = memcpy_s(newData, newCapacity * sizeof(T), dataPtr_, num_ * sizeof(T));
            ASCEND_THROW_IF_NOT_FMT(ret == EOK, "memcpy_s operator error %d", static_cast<int>(ret));
#endif
        }
        // Intentionally do not free the old bump block; reclaimed on arena Reset.

        dataPtr_ = newData;
        vecCapacity_ = newCapacity;
    }

    std::shared_ptr<DeviceMemArena> arena_;
    T *dataPtr_{nullptr};
    size_t num_{0};
    size_t vecCapacity_{0};
    P expandPolicy_;
};

}  // namespace ascend

#endif  // ARENA_DEV_MEM_STRATEGY_H
