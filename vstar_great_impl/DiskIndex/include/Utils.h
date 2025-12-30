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

#include "CommonIncludes.h"
#include "DiskAssert.h"

namespace diskann_pro {

inline bool FloatEqual(float a, float b)
{
    return fabs(a - b) < 1e-10f; // 1e-10f is the float comparison threshold
}

/* 计算阶段进行预取 */
template<typename T>
void inline Prefetch(const T *ptr, size_t prefetchSize = 128)
{
// 如不是arm系统，暂时不支持预取，之后需适配
#if defined(__aarch64__) || defined(__arm__)
    __asm__ volatile("prfm PLDL1STRM, [%0, #(%1)]"::"r"(ptr), "i"(prefetchSize));
#endif
}

}

struct PivotContainer {
    PivotContainer() = default;

    PivotContainer(size_t pivoID, float pivoDist) : pivID{pivoID}, pivDist{pivoDist}
    {
    }

    bool operator<(const PivotContainer &p) const
    {
        return p.pivDist < pivDist;
    }

    bool operator>(const PivotContainer &p) const
    {
        return p.pivDist > pivDist;
    }

    size_t pivID;
    float pivDist;
};
