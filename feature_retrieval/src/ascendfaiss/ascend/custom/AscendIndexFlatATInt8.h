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


#ifndef ASCEND_INDEX_FLAT_AT_INT8_INCLUDED
#define ASCEND_INDEX_FLAT_AT_INT8_INCLUDED

#include <faiss/MetaIndexes.h>
#include "ascend/AscendIndex.h"

namespace faiss {
namespace ascend {
const int64_t FLAT_AT_INT8_DEFAULT_MEM = 0x8000000; // 0x8000000 mean 128M(resource mem pool's size)

struct AscendIndexFlatATInt8Config : public AscendIndexConfig {
    inline AscendIndexFlatATInt8Config() {}

    inline AscendIndexFlatATInt8Config(std::initializer_list<int> devices,
        int64_t resourceSize = FLAT_AT_INT8_DEFAULT_MEM)
        : AscendIndexConfig(devices, resourceSize)
    {}

    inline AscendIndexFlatATInt8Config(std::vector<int> devices, int64_t resourceSize = FLAT_AT_INT8_DEFAULT_MEM)
        : AscendIndexConfig(devices, resourceSize)
    {}
};

class AscendIndexFlatATInt8Impl;
class AscendIndexFlatATInt8 : public AscendIndex {
public:
    // Construct an empty instance that can be added to
    AscendIndexFlatATInt8(int dims, int baseSize, AscendIndexFlatATInt8Config config = AscendIndexFlatATInt8Config());

    virtual ~AscendIndexFlatATInt8();

    // Clears all vectors from this index
    void reset() override;

    void searchInt8(idx_t n, const int8_t *x, idx_t k, float *distances, idx_t *labels) const;

    // AscendIndex object is NON-copyable
    AscendIndexFlatATInt8(const AscendIndexFlatATInt8&) = delete;
    AscendIndexFlatATInt8& operator=(const AscendIndexFlatATInt8&) = delete;

    void clearAscendTensor();

    void sendMinMax(float qMin, float qMax);

protected:
    std::shared_ptr<AscendIndexFlatATInt8Impl> impl_;
};
} // ascend
} // faiss
#endif // ASCEND_INDEX_FLAT_AT_INT8_INCLUDED