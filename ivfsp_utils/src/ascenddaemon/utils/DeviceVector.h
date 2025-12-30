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

#include <ascenddaemon/utils/AscendUtils.h>
#include <ascenddaemon/utils/MemorySpace.h>
#include <vector>

namespace ascendSearch {
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

template<typename T, typename P = ExpandPolicy>
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

    inline T* data() const
    {
        return dataPtr;
    }

    inline T& operator[](size_t pos);

    inline const T& operator[](size_t pos) const;

    std::vector<T> copyToStlVector() const;

    void append(const T* d, size_t n, bool reserveExact = false);

    void resize(size_t newSize, bool reserveExact = false);

    size_t reclaim(bool exact);

    void reserve(size_t newCapacity);

private:
    void realloc(size_t newCapacity);

private:
    T* dataPtr;
    size_t num;
    size_t vecCapacity;
    MemorySpace space;
    P expendPolicy;
};
}  // namespace ascendSearch
#include <ascenddaemon/utils/DeviceVectorInl.h>

#endif  // ASCEND_DEVICEVECTOR_H
