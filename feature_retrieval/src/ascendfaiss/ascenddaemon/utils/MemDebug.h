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

#ifndef ASCEND_MEM_DEBUG_INCLUDED
#define ASCEND_MEM_DEBUG_INCLUDED

#include <cstddef>
#include <cstdint>

#include "ascenddaemon/utils/MemorySpace.h"

namespace ascend
{

// Opt-in via ASCENDFAISS_MEM_DEBUG=1
// Optional sample cadence: ASCENDFAISS_MEM_DEBUG_EVERY=N (default 64)
bool MemDebugEnabled();
size_t MemDebugEvery();

// printf-style to stderr + APP_LOG (only meaningful when MemDebugEnabled).
void MemDebugPrintf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

const char *MemorySpaceName(MemorySpace space);

bool QueryHbm(size_t *freeBytes, size_t *totalBytes);

void LogHbm(const char *tag, int deviceId = -1);

uint64_t NoteAlloc(MemorySpace space, size_t size, int deviceId, size_t freeBefore, size_t totalBefore);

void DumpAllocRingOnFailure(MemorySpace space, size_t size, int err, int deviceId);

}  // namespace ascend

#endif  // ASCEND_MEM_DEBUG_INCLUDED
