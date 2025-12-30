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


#ifndef ASCEND_AUXINDEXSTRUCTURES_INCLUDED
#define ASCEND_AUXINDEXSTRUCTURES_INCLUDED

#include <unordered_set>
#include <vector>

#include <ascenddaemon/impl/Index.h>

namespace ascendSearch {
/// Encapsulates a set of ids to remove from technicolor-research/faiss-quickeradc.
struct IDSelector {
    using idx_t = Index::idx_t;
    virtual bool is_member(idx_t id) const = 0;
    virtual ~IDSelector()
    {
    }
};

/// remove ids between [imin, imax).
struct IDSelectorRange : IDSelector {
    idx_t imin;
    idx_t imax;

    IDSelectorRange(idx_t imin, idx_t imax);
    bool is_member(idx_t id) const override;
    ~IDSelectorRange() override
    {
    }
};

/** Remove ids from a set. Repetitions of ids in the indices set  passed to the
 * constructor does not hurt performanc. The hash function used for the bloom filter
 * and GCC's implementation of unordered_set are just the least significant bits of
 * the id. This works fine for random ids or ids in sequences but will produce many
 * hash collisions if lsb's are always the same */
struct IDSelectorBatch : IDSelector {
    std::unordered_set<idx_t> set;

    using uint8_t = unsigned char;
    std::vector<uint8_t> bloom;
    uint32_t nbits = 0;
    idx_t mask;

    IDSelectorBatch(size_t n, const idx_t *indices);
    bool is_member(idx_t i) const override;
    ~IDSelectorBatch() override
    {
    }
};
}  // namespace ascendSearch
#endif  // ASCEND_AUXINDEXSTRUCTURES_INCLUDED
