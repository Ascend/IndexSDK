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


#ifndef ASCEND_INDEX_FLAT_AT_INCLUDED
#define ASCEND_INDEX_FLAT_AT_INCLUDED

#include <faiss/MetaIndexes.h>
#include "ascend/AscendIndex.h"

namespace faiss {
namespace ascend {
const int64_t FLAT_AT_DEFAULT_MEM = 0x8000000; // 0x8000000 mean 128M(resource mem pool's size)

struct AscendIndexFlatATConfig : public AscendIndexConfig {
    inline AscendIndexFlatATConfig() {}

    inline AscendIndexFlatATConfig(std::initializer_list<int> devices, int64_t resourceSize = FLAT_AT_DEFAULT_MEM)
        : AscendIndexConfig(devices, resourceSize)
    {}

    inline AscendIndexFlatATConfig(std::vector<int> devices, int64_t resourceSize = FLAT_AT_DEFAULT_MEM)
        : AscendIndexConfig(devices, resourceSize)
    {}
};

class AscendIndexFlatATImpl;
class AscendIndexFlatAT : public AscendIndex {
public:
    // Construct an empty instance that can be added to
    AscendIndexFlatAT(int dims, int baseSize, AscendIndexFlatATConfig config = AscendIndexFlatATConfig());

    virtual ~AscendIndexFlatAT();

    // Clears all vectors from this index
    void reset() override;

    void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels,
        const SearchParameters *params = nullptr) const override;

    // AscendIndex object is NON-copyable
    AscendIndexFlatAT(const AscendIndexFlatAT&) = delete;
    AscendIndexFlatAT& operator=(const AscendIndexFlatAT&) = delete;

    void clearAscendTensor();

protected:
    std::shared_ptr<AscendIndexFlatATImpl> impl_;
};
} // ascend
} // faiss
#endif // ASCEND_INDEX_FLAT_INCLUDED