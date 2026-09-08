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

#ifndef ASCEND_ROARING_FILTER_INCLUDED
#define ASCEND_ROARING_FILTER_INCLUDED

#include <faiss/impl/IDSelector.h>

#include <cstddef>
#include <cstdint>

#include "common/RabitqIdFilter.h"

namespace faiss
{
namespace ascend
{

bool IdsFormArithmeticProgression(const idx_t *ids, size_t n, int64_t step);
bool FreezeRoaringLive(const void *liveBitmap, ::ascend::RabitqIdFilterHost &out);
bool FreezeRoaringIdxs(const idx_t *ids, size_t n, ::ascend::RabitqIdFilterHost &out);
bool FreezeRoaringUint32Sorted(const uint32_t *vals, size_t n, ::ascend::RabitqIdFilterHost &out);
bool FreezeRoaringFromRange(uint64_t minInclusive, uint64_t maxExclusive, uint32_t step,
                            ::ascend::RabitqIdFilterHost &out);
void AssertValidFrozenRoaring(const uint8_t *frozen, size_t n);

}  // namespace ascend
}  // namespace faiss

#endif
