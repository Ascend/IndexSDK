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


#ifndef ASCEND_TASK_QUEUE_ITEM_H
#define ASCEND_TASK_QUEUE_ITEM_H

#include <cstdint>

namespace ascendSearchacc {
using idx_t = uint64_t;

struct QueueItem {
    QueueItem() : distPtr(nullptr), idPtr(nullptr), flagPtr(nullptr), size(0), executing(false)
    {
    }

    QueueItem(const QueueItem &item)
    {
        distPtr = item.distPtr;
        extremePtr = item.extremePtr;
        idPtr = item.idPtr;
        flagPtr = item.flagPtr;
        size = item.size;
        executing = item.executing ? true : false;
    }

    QueueItem &operator=(const QueueItem &item)
    {
        distPtr = item.distPtr;
        extremePtr = item.extremePtr;
        idPtr = item.idPtr;
        flagPtr = item.flagPtr;
        size = item.size;
        executing = item.executing ? true : false;
        return *this;
    }

    void SetExecuting(float16_t *dist, idx_t *id, uint16_t *flag, uint32_t s)
    {
        ASCEND_THROW_IF_NOT(dist != nullptr);
        ASCEND_THROW_IF_NOT(id != nullptr);
        ASCEND_THROW_IF_NOT(flag != nullptr);
        ASCEND_THROW_IF_NOT(s != 0);

        distPtr = dist;
        idPtr = id;
        flagPtr = flag;
        flagPtrSec = flag + FLAG_ALIGN_OFFSET;
        size = s;
        executing = true;
    }

    void SetExecuting(float16_t *dist, float16_t *extreme, idx_t *id, uint16_t *flag, uint32_t s)
    {
        ASCEND_THROW_IF_NOT(dist != nullptr);
        ASCEND_THROW_IF_NOT(extreme != nullptr);
        ASCEND_THROW_IF_NOT(id != nullptr);
        ASCEND_THROW_IF_NOT(flag != nullptr);
        ASCEND_THROW_IF_NOT(s != 0);

        distPtr = dist;
        extremePtr = extreme;
        idPtr = id;
        flagPtr = flag;
        flagPtrSec = flag + FLAG_ALIGN_OFFSET;
        size = s;
        executing = true;
    }

    inline bool IsExecuting()
    {
        return executing;
    }

    float16_t *distPtr;  // distance result mem pointer

    float16_t *extremePtr;  // extreme distance result mem pointer

    idx_t *idPtr;  // ids mem pointer

    uint16_t *volatile flagPtr;  // flag mem pointer for aicore 0,
    // the first uint16_t will be setted to 1 when aicore finished calc

    uint16_t *volatile flagPtrSec;  // flag mem pointer for aicore 1,
    // the first uint16_t will be setted to 1 when aicore finished calc

    int size;  // size to idicate how many code to calc, and how many results to topk functor

    std::atomic<bool> executing;  // whether the item has beed added to stream for executing
};
}  // namespace ascendSearchacc
#endif  // ASCEND_TASK_QUEUE_ITEM_H
