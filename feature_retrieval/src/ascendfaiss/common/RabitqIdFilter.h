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

#ifndef ASCEND_COMMON_RABITQ_ID_FILTER_H
#define ASCEND_COMMON_RABITQ_ID_FILTER_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ops/cpukernel/impl/utils/kernel_shared_def.h"

namespace ascend
{

/** Host-side materialization of a FAISS IDSelector for IVF-RaBitQ search. */
struct RabitqIdFilterHost
{
    int64_t mode = aicpu::RABITQ_ID_FILTER_NONE;
    int64_t negate = 0;
    int64_t aux0 = 0;  // imin / sorted count / bitmap bit-count
    int64_t aux1 = 0;  // imax
    std::vector<int64_t> sortedIds;
    std::vector<uint8_t> bitmap;
    const int64_t *sortedView = nullptr;
    const uint8_t *bitmapView = nullptr;  // user buffer; must stay valid until search returns
    size_t viewBytes = 0;
    // Bumped by the host cache on rematerialize; not cleared by resetKeepCapacity().
    uint64_t generation = 0;

    const void *payloadSrc() const
    {
        if (mode == aicpu::RABITQ_ID_FILTER_SORTED)
        {
            return sortedView != nullptr ? static_cast<const void *>(sortedView) : sortedIds.data();
        }
        if (mode == aicpu::RABITQ_ID_FILTER_BITMAP)
        {
            return bitmapView != nullptr ? static_cast<const void *>(bitmapView) : bitmap.data();
        }
        return nullptr;
    }

    size_t payloadBytes() const
    {
        if (mode == aicpu::RABITQ_ID_FILTER_SORTED)
        {
            return sortedView != nullptr ? viewBytes : sortedIds.size() * sizeof(int64_t);
        }
        if (mode == aicpu::RABITQ_ID_FILTER_BITMAP)
        {
            return bitmapView != nullptr ? viewBytes : bitmap.size() * sizeof(uint8_t);
        }
        return 0;
    }

    void resetKeepCapacity()
    {
        mode = aicpu::RABITQ_ID_FILTER_NONE;
        negate = 0;
        aux0 = 0;
        aux1 = 0;
        sortedIds.clear();
        bitmap.clear();
        sortedView = nullptr;
        bitmapView = nullptr;
        viewBytes = 0;
    }
};

}  // namespace ascend

#endif
