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


#pragma once

#include "DiskAssert.h"
#include <securec.h>

namespace diskann_pro {

constexpr size_t LOCAL_SECUREC_MEM_MAX_LEN = 2147483640; // max secure_c buffer size (2GB - 1) rounded down by 8

template<typename D, typename S>
inline void SecMemcpyWithMemLimit(D *dest, size_t destBufferSize, const S *src, size_t srcBufferSize)
{
    if (srcBufferSize == 0) {
        return;
    }
    DISK_THROW_IF_NOT_FMT(LOCAL_SECUREC_MEM_MAX_LEN % sizeof(D) == 0,
        "SecMemcpyWithMemLimit currently only supports situation"
        " when LOCAL_SECUREC_MEM_MAX_LEN[%zu] divides size of dst type[%zu].\n",
        LOCAL_SECUREC_MEM_MAX_LEN, sizeof(D));
    DISK_THROW_IF_NOT_FMT(LOCAL_SECUREC_MEM_MAX_LEN % sizeof(S) == 0,
        "SecMemcpyWithMemLimit currently only supports situation"
        " when LOCAL_SECUREC_MEM_MAX_LEN[%zu] divides size of src type[%zu].\n",
        LOCAL_SECUREC_MEM_MAX_LEN, sizeof(S));

    int ret = 0;
    size_t dstMoveCount = LOCAL_SECUREC_MEM_MAX_LEN / sizeof(D);
    size_t srcMoveCount = LOCAL_SECUREC_MEM_MAX_LEN / sizeof(S);

    size_t operationCounts = srcBufferSize / LOCAL_SECUREC_MEM_MAX_LEN;

    // 每次迭代我们对destBuffer部分取其与操作上限的最小值，防止destBuffer每次拷贝的剩余部分仍大于操作上限导致操作失败
    for (size_t i = 0; i < operationCounts; ++i) {
        ret = memcpy_s(dest + i * dstMoveCount,
                       std::min(LOCAL_SECUREC_MEM_MAX_LEN, destBufferSize - i * LOCAL_SECUREC_MEM_MAX_LEN),
                       src + i * srcMoveCount,
                       LOCAL_SECUREC_MEM_MAX_LEN);
        DISK_THROW_IF_NOT_FMT(ret == 0, "memcpy_s failed: expect return value to be 0 but returned %d.\n", ret);
    }
    size_t destRemainBytes = destBufferSize - operationCounts * LOCAL_SECUREC_MEM_MAX_LEN;
    size_t srcRemainBytes = srcBufferSize - operationCounts * LOCAL_SECUREC_MEM_MAX_LEN;
    ret = memcpy_s(dest + operationCounts * dstMoveCount,
                   std::min(LOCAL_SECUREC_MEM_MAX_LEN, destRemainBytes),
                   src + operationCounts * srcMoveCount,
                   srcRemainBytes);
    DISK_THROW_IF_NOT_FMT(ret == 0, "memcpy_s failed: expect return value to be 0 but returned %d.\n", ret);
}

template<typename D>
inline void SecMemsetWithMemLimit(D *dest, size_t destBufferSize, int memsetValue, size_t memsetByteSize)
{
    if (memsetByteSize == 0) {
        return;
    }
    DISK_THROW_IF_NOT_FMT(LOCAL_SECUREC_MEM_MAX_LEN % sizeof(D) == 0,
        "SecMemsetWithMemLimit currently only supports situation"
        " when LOCAL_SECUREC_MEM_MAX_LEN[%zu] divides size of dst type[%zu].\n",
        LOCAL_SECUREC_MEM_MAX_LEN, sizeof(D));

    int ret = 0;
    size_t dstMoveCount = LOCAL_SECUREC_MEM_MAX_LEN / sizeof(D);
    size_t operationCounts = memsetByteSize / LOCAL_SECUREC_MEM_MAX_LEN;

    // 每次迭代我们对destBuffer部分取其与操作上限的最小值，防止destBuffer每次拷贝的剩余部分仍大于操作上限导致操作失败
    for (size_t i = 0; i < operationCounts; ++i) {
        ret = memset_s(dest + i * dstMoveCount,
                       std::min(LOCAL_SECUREC_MEM_MAX_LEN, destBufferSize - i * LOCAL_SECUREC_MEM_MAX_LEN),
                       memsetValue,
                       LOCAL_SECUREC_MEM_MAX_LEN);
        DISK_THROW_IF_NOT_FMT(ret == 0, "memset_s failed: expect return value to be 0 but returned %d.\n", ret);
    }
    size_t destRemainBytes = destBufferSize - operationCounts * LOCAL_SECUREC_MEM_MAX_LEN;
    size_t setRemainBytes = memsetByteSize - operationCounts * LOCAL_SECUREC_MEM_MAX_LEN;
    ret = memset_s(dest + operationCounts * dstMoveCount,
                   std::min(LOCAL_SECUREC_MEM_MAX_LEN, destRemainBytes),
                   memsetValue,
                   setRemainBytes);
    DISK_THROW_IF_NOT_FMT(ret == 0, "memset_s failed: expect return value to be 0 but returned %d.\n", ret);
}

}