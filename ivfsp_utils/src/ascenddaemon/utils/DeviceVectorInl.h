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


#ifndef ASCEND_DEVICEVECTORINL_INCLUDED
#define ASCEND_DEVICEVECTORINL_INCLUDED

#include <cstring>

#include <ascenddaemon/utils/DeviceVector.h>
#include <ascenddaemon/utils/StaticUtils.h>

namespace ascendSearch {
namespace {
// the memcpy_s function requires size < 2GB
const size_t MEMCPY_S_THRESHOLD = 0x80000000;
}

template<typename T, typename P>
DeviceVector<T, P>::DeviceVector(MemorySpace space) : dataPtr(nullptr), num(0), vecCapacity(0), space(space)
{}

template<typename T, typename P> DeviceVector<T, P>::~DeviceVector()
{
    clear();
}

template<typename T, typename P> void DeviceVector<T, P>::clear()
{
    if (this->dataPtr != nullptr) {
        FreeMemorySpace(space, static_cast<void *>(this->dataPtr));
        this->dataPtr = nullptr;
    }

    this->num = 0;
    this->vecCapacity = 0;
}

template<typename T, typename P> inline const T &DeviceVector<T, P>::operator[](size_t pos) const
{
    ASCEND_THROW_IF_NOT(pos >= 0 && pos < num);
    return *(dataPtr + pos);
}

template<typename T, typename P> inline T &DeviceVector<T, P>::operator[](size_t pos)
{
    ASCEND_THROW_IF_NOT(pos >= 0 && pos < num);
    return *(dataPtr + pos);
}

template<typename T, typename P> std::vector<T> DeviceVector<T, P>::copyToStlVector() const
{
    if (this->num == 0 || this->dataPtr == nullptr) {
        return std::vector<T>();
    }

    std::vector<T> out(this->num);
    ASCEND_THROW_IF_NOT(this->num * sizeof(T) < MEMCPY_S_THRESHOLD);
    auto err = memcpy_s(out.data(), this->num * sizeof(T), this->dataPtr, this->num * sizeof(T));
    ASCEND_THROW_IF_NOT_FMT(err == EOK, "memcpy err, err=%d", err);
    return out;
}

template<typename T, typename P> void DeviceVector<T, P>::append(const T *d, size_t n, bool reserveExact)
{
    if (d == nullptr || n <= 0) {
        return;
    }

    size_t reserveSize = this->num + n;
    if (!reserveExact) {
        reserveSize = expendPolicy(reserveSize);
    }

    reserve(reserveSize);
    ASCEND_THROW_IF_NOT(this->num * sizeof(T) < MEMCPY_S_THRESHOLD);
    auto ret = aclrtMemcpy(this->dataPtr + this->num, n * sizeof(T), d, n * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "failed to aclrtMemcpy (error %d)", static_cast<int>(ret));

    this->num += n;
}

template<typename T, typename P> void DeviceVector<T, P>::resize(size_t newSize, bool reserveExact)
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

template<typename T, typename P> size_t DeviceVector<T, P>::reclaim(bool exact)
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

template<typename T, typename P>
void DeviceVector<T, P>::reserve(size_t newCapacity)
{
    if (newCapacity > this->vecCapacity) {
        this->realloc(newCapacity);
    }
}

template<typename T, typename P>
void DeviceVector<T, P>::realloc(size_t newCapacity)
{
    ASCEND_THROW_IF_NOT(this->num <= newCapacity);

    T *newData = nullptr;
    if (newCapacity) {
        AllocMemorySpace(space, &newData, newCapacity * sizeof(T));

        if (this->dataPtr != nullptr) {
            ASCEND_THROW_IF_NOT(this->num * sizeof(T) < MEMCPY_S_THRESHOLD);
            auto ret = aclrtMemcpy(newData, newCapacity * sizeof(T),
                this->dataPtr, this->num * sizeof(T), ACL_MEMCPY_DEVICE_TO_DEVICE);
            if (ret != ACL_SUCCESS) {
                // 如果拷贝失败，将分配给newData的内存清除，否则内存泄露
                FreeMemorySpace(space, static_cast<void *>(newData));
                ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "failed to aclrtMemcpy (error %d)", static_cast<int>(ret));
            }
            // 若成功，按照原本逻辑清除dataPtr原有的内存，并在之后将newData赋给dataPtr
            FreeMemorySpace(space, static_cast<void *>(this->dataPtr));
        }
    } else {
        FreeMemorySpace(space, static_cast<void *>(this->dataPtr));
    }

    this->dataPtr = newData;
    this->vecCapacity = newCapacity;
}
} // namespace ascendSearch

#endif // ASCEND_DEVICEVECTORINL_H
