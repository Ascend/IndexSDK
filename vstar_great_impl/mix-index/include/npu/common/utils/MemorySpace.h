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


#ifndef ASCEND_MEMORYSPACE_INCLUDED
#define ASCEND_MEMORYSPACE_INCLUDED

#include <cstddef>

namespace ascendSearchacc {
enum MemorySpace {
    DEVICE = 1,
    DEVICE_HUGEPAGE = 2,
};

void AllocMemorySpaceV(MemorySpace space, void **p, size_t size);

template <typename T>
inline void allocMemorySpace(MemorySpace space, T **p, size_t size)
{
    AllocMemorySpaceV(space, (void **)p, size);
}

void FreeMemorySpace(MemorySpace space, void *p);
}  // namespace ascendSearchacc

#endif  // ASCEND_MEMORYSPACE_H
