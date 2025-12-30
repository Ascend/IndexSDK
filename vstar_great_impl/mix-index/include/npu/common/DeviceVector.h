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


#ifndef ASCEND_DEVICEVECTOR_INCLUDED
#define ASCEND_DEVICEVECTOR_INCLUDED

#include <vector>

#include "npu/common/utils/AscendUtils.h"
#include "npu/common/utils/MemorySpace.h"
#include "npu/common/utils/StaticUtils.h"

namespace ascendSearchacc {
struct ExpandPolicy {
    size_t operator()(const size_t preferredSize) const
    {
        const int halfScaleThreshold = 512;
        const int quarterScaleThreshold = 1024;
        const int eighthScaleThreshold = 2048;
        size_t tmpPrefer = utils::nextHighestPowerOf2(preferredSize);
        if (preferredSize >= halfScaleThreshold && preferredSize < quarterScaleThreshold) {
            // scale 1/2 * preferredSize size more, to (3/2) 1.5 * preferredSize
            size_t tmp = preferredSize * 3 / 2;
            tmpPrefer = (tmpPrefer > tmp) ? tmp : tmpPrefer;
        } else if (preferredSize >= quarterScaleThreshold && preferredSize < eighthScaleThreshold) {
            // scale 1/4 * preferredSize size more, to (5/4) 1.25 * preferredSize
            size_t tmp = preferredSize * 5 / 4;
            tmpPrefer = (tmpPrefer > tmp) ? tmp : tmpPrefer;
        } else if (preferredSize >= eighthScaleThreshold) {
            // scale 1/8 * preferredSize size more, to (9/8) 1.125 * preferredSize
            size_t tmp = preferredSize * 9 / 8;
            tmpPrefer = (tmpPrefer > tmp) ? tmp : tmpPrefer;
        }

        return tmpPrefer;
    }
};

struct ExpandPolicySlim {
    size_t operator()(const size_t preferredSize) const
    {
        const size_t baseLine = 16 * 1024;
        const size_t growFactor = 2;
        const size_t growSize = 16 * 1024;
        if (preferredSize <= baseLine) {
            return growFactor * preferredSize;
        }
        return preferredSize + growSize;
    }
};

template <typename T, typename P = ExpandPolicy>
class DeviceVector {
public:
    DeviceVector(MemorySpace space = MemorySpace::DEVICE);

    ~DeviceVector();

    void clear();

    inline size_t size() const
    {
        return num;
    }

    inline size_t capacity() const
    {
        return vecCapacity;
    }

    inline T *data() const
    {
        return dataPtr;
    }

    inline T &operator[](size_t pos);

    inline const T &operator[](size_t pos) const;

    std::vector<T> copyToStlVector() const;

    void append(const T *d, size_t n, bool reserveExact = false);

    void resize(size_t newSize, bool reserveExact = false);

    size_t reclaim(bool exact);

    void reserve(size_t newCapacity);

private:
    void realloc(size_t newCapacity);

private:
    T *dataPtr;
    size_t num;
    size_t vecCapacity;
    MemorySpace space;
    P expendPolicy;
};

namespace {
// the memcpy_s function requires size < 2GB
const size_t MEMCPY_S_THRESHOLD = 0x80000000;
}  // namespace
 
template <typename T, typename P>
DeviceVector<T, P>::DeviceVector(MemorySpace space) : dataPtr(nullptr), num(0), vecCapacity(0), space(space)
{
}
 
template <typename T, typename P>
DeviceVector<T, P>::~DeviceVector()
{
    clear();
}
 
template <typename T, typename P>
void DeviceVector<T, P>::clear()
{
    if (this->dataPtr != nullptr) {
        FreeMemorySpace(space, static_cast<void *>(this->dataPtr));
        this->dataPtr = nullptr;
    }
 
    this->num = 0;
    this->vecCapacity = 0;
}
 
template <typename T, typename P>
inline const T &DeviceVector<T, P>::operator[](size_t pos) const
{
    ASCEND_THROW_IF_NOT(pos >= 0 && pos < num);
    return *(dataPtr + pos);
}
 
template <typename T, typename P>
inline T &DeviceVector<T, P>::operator[](size_t pos)
{
    ASCEND_THROW_IF_NOT(pos >= 0 && pos < num);
    return *(dataPtr + pos);
}
 
template <typename T, typename P>
std::vector<T> DeviceVector<T, P>::copyToStlVector() const
{
    if (this->num == 0 || this->dataPtr == nullptr) {
        return std::vector<T>();
    }
 
    std::vector<T> out(this->num);
    ASCEND_THROW_IF_NOT(this->num * sizeof(T) < MEMCPY_S_THRESHOLD);
    auto ret = memcpy_s(out.data(), this->num * sizeof(T), this->dataPtr, this->num * sizeof(T));
    ASCEND_THROW_IF_NOT_FMT(ret == EOK, "Mem operator error %d", ret);
 
    return out;
}
 
template <typename T, typename P>
void DeviceVector<T, P>::append(const T *d, size_t n, bool reserveExact)
{
    if (d == nullptr || n == 0) {
        return;
    }
 
    size_t reserveSize = this->num + n;
    if (!reserveExact) {
        reserveSize = expendPolicy(reserveSize);
    }
 
    reserve(reserveSize);
    ASCEND_THROW_IF_NOT(this->num * sizeof(T) < MEMCPY_S_THRESHOLD);
    auto ret = aclrtMemcpy(this->dataPtr + this->num, n * sizeof(T), d, n * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Mem operator error %d", ret);
 
    this->num += n;
}
 
template <typename T, typename P>
void DeviceVector<T, P>::resize(size_t newSize, bool reserveExact)
{
    if (this->num < newSize) {
        if (reserveExact) {
            reserve(newSize);
        } else {
            reserve(expendPolicy(newSize));
        }
    }
 
    this->num = newSize;
}
 
template <typename T, typename P>
size_t DeviceVector<T, P>::reclaim(bool exact)
{
    size_t freeSize = this->vecCapacity - this->num;
 
    if (exact) {
        this->realloc(this->num);
        return freeSize * sizeof(T);
    }
 
    // If more than 1/4th of the space is free, then we want to
    // truncate to only having 1/8th of the space free; this still
    // preserves some space for new elements, but won't force us to
    // double our size right away
    const int RECLAIM_PROPORTION = 4;
    const int TRUNCATE_PROPORTION = 8;
    if (freeSize > (this->vecCapacity / RECLAIM_PROPORTION)) {
        size_t newFreeSize = this->vecCapacity / TRUNCATE_PROPORTION;
        size_t newCapacity = this->num + newFreeSize;
 
        size_t oldCapacity = this->vecCapacity;
        ASCEND_THROW_IF_NOT(newCapacity < oldCapacity);
 
        this->realloc(newCapacity);
 
        return (oldCapacity - newCapacity) * sizeof(T);
    }
 
    return 0;
}
 
template <typename T, typename P>
void DeviceVector<T, P>::reserve(size_t newCapacity)
{
    if (newCapacity > this->vecCapacity) {
        this->realloc(newCapacity);
    }
}
 
template <typename T, typename P>
void DeviceVector<T, P>::realloc(size_t newCapacity)
{
    ASCEND_THROW_IF_NOT(this->num <= newCapacity);
 
    T *newData = nullptr;
    if (newCapacity) {
        allocMemorySpace(space, &newData, newCapacity * sizeof(T));
 
        if (this->dataPtr != nullptr) {
            ASCEND_THROW_IF_NOT(this->num * sizeof(T) < MEMCPY_S_THRESHOLD);
#ifdef HOSTCPU
            if (this->num > 0) {
                auto ret = aclrtMemcpy(newData, newCapacity * sizeof(T), this->dataPtr, this->num * sizeof(T),
                                       ACL_MEMCPY_DEVICE_TO_DEVICE);
                ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Mem operator error %d", ret);
            }
#else
            auto ret = memcpy_s(newData, newCapacity * sizeof(T), this->dataPtr, this->num * sizeof(T));
            ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Mem operator error %d", ret);
#endif
            FreeMemorySpace(space, static_cast<void *>(this->dataPtr));
        }
    } else {
        FreeMemorySpace(space, static_cast<void *>(this->dataPtr));
    }
 
    this->dataPtr = newData;
    this->vecCapacity = newCapacity;
}
}  // namespace ascendSearchacc

#endif  // ASCEND_DEVICEVECTOR_H
