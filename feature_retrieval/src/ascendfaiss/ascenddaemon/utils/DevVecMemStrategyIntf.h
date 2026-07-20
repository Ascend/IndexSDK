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

#ifndef DEV_VEC_MEM_STRATEGY_INTF_H
#define DEV_VEC_MEM_STRATEGY_INTF_H

#include <algorithm>
#include <cstdint>
#include <vector>

#include "AscendUtils.h"
#include "hmm/AscendHMO.h"

namespace ascend
{

// the memcpy_s function requires size < 2GB
constexpr size_t MEMCPY_S_THRESHOLD = 0x80000000;

inline errno_t memcpySChunked(void *dst, size_t dstCapacity, const void *src, size_t count)
{
    if (count == 0)
    {
        return EOK;
    }
    if (count > dstCapacity)
    {
        return ERANGE;
    }
    const auto *srcBytes = static_cast<const uint8_t *>(src);
    auto *dstBytes = static_cast<uint8_t *>(dst);
    size_t offset = 0;
    while (offset < count)
    {
        const size_t remaining = count - offset;
        const size_t dstRemaining = dstCapacity - offset;
        const size_t chunk = std::min({remaining, dstRemaining, MEMCPY_S_THRESHOLD - 1});
        const errno_t ret = memcpy_s(dstBytes + offset, chunk, srcBytes + offset, chunk);
        if (ret != EOK)
        {
            return ret;
        }
        offset += chunk;
    }
    return EOK;
}

inline aclError aclrtMemcpyChunked(void *dst, size_t dstCapacity, const void *src, size_t count, aclrtMemcpyKind kind)
{
    if (count == 0)
    {
        return ACL_SUCCESS;
    }
    if (count > dstCapacity)
    {
        return ACL_ERROR_FAILURE;
    }
    const auto *srcBytes = static_cast<const uint8_t *>(src);
    auto *dstBytes = static_cast<uint8_t *>(dst);
    size_t offset = 0;
    while (offset < count)
    {
        const size_t remaining = count - offset;
        const size_t chunk = remaining < (MEMCPY_S_THRESHOLD - 1) ? remaining : (MEMCPY_S_THRESHOLD - 1);
        const aclError ret = aclrtMemcpy(dstBytes + offset, dstCapacity - offset, srcBytes + offset, chunk, kind);
        if (ret != ACL_SUCCESS)
        {
            return ret;
        }
        offset += chunk;
    }
    return ACL_SUCCESS;
}

struct ExpandPolicy
{
    size_t operator()(const size_t preferredSize) const
    {
        constexpr size_t halfScaleThreshold = 512;
        constexpr size_t quarterScaleThreshold = 1024;
        constexpr size_t eighthScaleThreshold = 2048;
        size_t outPrefer = utils::nextHighestPowerOf2(preferredSize);
        if (preferredSize >= halfScaleThreshold && preferredSize < quarterScaleThreshold)
        {
            // scale 1/2 * preferredSize size more, to (3/2) 1.5 * preferredSize
            size_t scaledPrefer = preferredSize * 3 / 2;
            outPrefer = (outPrefer > scaledPrefer) ? scaledPrefer : outPrefer;
        }
        else if (preferredSize >= quarterScaleThreshold && preferredSize < eighthScaleThreshold)
        {
            // scale 1/4 * preferredSize size more, to (5/4) 1.25 * preferredSize
            size_t scaledPrefer = preferredSize * 5 / 4;
            outPrefer = (outPrefer > scaledPrefer) ? scaledPrefer : outPrefer;
        }
        else if (preferredSize >= eighthScaleThreshold)
        {
            // scale 1/8 * preferredSize size more, to (9/8) 1.125 * preferredSize
            size_t scaledPrefer = preferredSize * 9 / 8;
            outPrefer = (outPrefer > scaledPrefer) ? scaledPrefer : outPrefer;
        }

        return outPrefer;
    }
};

struct ExpandPolicySlim
{
    size_t operator()(const size_t preferredSize) const
    {
        constexpr size_t baseLine = 16 * 1024;
        constexpr size_t growFactor = 2;
        constexpr size_t growSize = 16 * 1024;
        if (preferredSize <= baseLine)
        {
            return growFactor * preferredSize;
        }
        return preferredSize + growSize;
    }
};

template <typename T>
class DevVecMemStrategyIntf
{
   public:
    virtual ~DevVecMemStrategyIntf() {}

    virtual void Clear() = 0;

    virtual size_t Size() const = 0;

    virtual size_t Capacity() const = 0;

    virtual T *Data() const = 0;

    virtual T &operator[](size_t pos) = 0;

    virtual const T &operator[](size_t pos) const = 0;

    virtual std::vector<T> CopyToStlVector() const = 0;

    virtual void Append(const T *d, size_t n, bool reserveExact = false) = 0;

    virtual void Resize(size_t newSize, bool reserveExact = false) = 0;

    virtual size_t Reclaim(bool exact) = 0;

    virtual void Reserve(size_t newCapacity) = 0;

    virtual void PushData(bool) {}

    virtual std::shared_ptr<AscendHMO> GetHmo() { return nullptr; }
};

}  // namespace ascend

#endif  // DEV_VEC_MEM_STRATEGY_INTF_H
