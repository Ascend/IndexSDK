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
#ifndef CAGRA_RABITQ_LIMITS_H
#define CAGRA_RABITQ_LIMITS_H

#include <cstdint>

constexpr uint32_t CAGRA_LOCAL_MEMORY_SIZE = 64 * 1024;
constexpr uint32_t CAGRA_MAX_TOPK = 1024;
#define CAGRA_RADIX_SCRATCH_WORDS(capacity) (2ULL * (capacity) + 2048 + 8)

// Mirrors the pointer layout in cagra_rabitq.cpp. get_min_k writes one result per warp,
// rather than a full candidate list; search_width is fixed to one.
constexpr uint64_t CagraRabitqLocalMemoryBytes(uint64_t dim, uint64_t degree, uint64_t capacity)
{
    return ((dim + 127) / 128 * 128) * 4 + (dim + 7) / 8 * 6 +
           (((capacity + degree + 31) / 32 * 32) + degree + (degree + 31) / 32) * 8 + 4 +
           (6 + CAGRA_RADIX_SCRATCH_WORDS(capacity)) * 4;
}

constexpr uint64_t CAGRA_MAX_LOCAL_MEMORY_BYTES = CagraRabitqLocalMemoryBytes(3072, 128, 1024);
static_assert(CAGRA_MAX_LOCAL_MEMORY_BYTES <= CAGRA_LOCAL_MEMORY_SIZE, "CAGRA scratch exceeds local memory");

#endif  // CAGRA_RABITQ_LIMITS_H
