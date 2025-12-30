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


#include <ascenddaemon/impl/AuxIndexStructures.h>

namespace ascendSearch {
namespace {
    const uint32_t OFFSET_BIT = 5;
    const uint32_t SHIFT_BIT = 3;
    const uint32_t ID_MASK = 7;
}

IDSelectorRange::IDSelectorRange(idx_t imin, idx_t imax)
    : imin(imin),
      imax(imax)
{
}

bool IDSelectorRange::is_member(idx_t id) const
{
    return id >= imin && id < imax;
}

IDSelectorBatch::IDSelectorBatch(size_t n, const idx_t *indices)
{
    ASCEND_THROW_IF_NOT(indices);
    nbits = 0;
    while (n > (1UL << nbits)) {
        nbits++;
    }
    nbits += OFFSET_BIT;

    mask = (1UL << nbits) - 1;
    bloom.resize(1UL << (nbits - SHIFT_BIT), 0);
    for (size_t i = 0; i < n; i++) {
        idx_t id = indices[i];
        set.insert(id);
        id &= mask;
        bloom[id >> SHIFT_BIT] |= 1UL << (id & ID_MASK);
    }
}

bool IDSelectorBatch::is_member(idx_t i) const
{
    unsigned long im = i & mask;
    if ((bloom[im >> SHIFT_BIT] & (1UL << (im & ID_MASK))) == 0UL) {
        return 0;
    }
    return set.count(i);
}
}  // namespace ascendSearch