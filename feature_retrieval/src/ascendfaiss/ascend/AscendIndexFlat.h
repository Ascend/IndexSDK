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


#ifndef ASCEND_INDEX_FLAT_INCLUDED
#define ASCEND_INDEX_FLAT_INCLUDED

#include <faiss/MetaIndexes.h>
#include "ascend/AscendIndex.h"

namespace faiss {
struct IndexFlat;
struct IndexFlatL2;
} // faiss

namespace faiss {
namespace ascend {
const int64_t FLAT_DEFAULT_MEM = 0x8000000; // 0x8000000 mean 128M(resource mem pool's size)

struct AscendIndexFlatConfig : public AscendIndexConfig {
    inline AscendIndexFlatConfig() {}

    inline AscendIndexFlatConfig(std::initializer_list<int> devices, int64_t resourceSize = FLAT_DEFAULT_MEM)
        : AscendIndexConfig(devices, resourceSize)
    {}

    inline AscendIndexFlatConfig(std::vector<int> devices, int64_t resourceSize = FLAT_DEFAULT_MEM)
        : AscendIndexConfig(devices, resourceSize)
    {}
};

class AscendIndexFlatImpl;
class AscendIndexFlat : public AscendIndex {
public:
    // Construct from a pre-existing faiss::IndexFlat instance
    AscendIndexFlat(const faiss::IndexFlat *index, AscendIndexFlatConfig config = AscendIndexFlatConfig());
    AscendIndexFlat(const faiss::IndexIDMap *index, AscendIndexFlatConfig config = AscendIndexFlatConfig());

    // Construct an empty instance that can be added to
    AscendIndexFlat(int dims, faiss::MetricType metric, AscendIndexFlatConfig config = AscendIndexFlatConfig());

    virtual ~AscendIndexFlat();

    // Initialize ourselves from the given CPU index; will overwrite
    // all data in ourselves
    void copyFrom(const faiss::IndexFlat *index);
    void copyFrom(const faiss::IndexIDMap *index);

    // Copy ourselves to the given CPU index; will overwrite all data
    // in the index instance
    void copyTo(faiss::IndexFlat *index) const;
    void copyTo(faiss::IndexIDMap *index) const;

    // Returns the number of vectors we contain
    size_t getBaseSize(int deviceId) const;

    // Returns the vectors of we contain
    void getBase(int deviceId, char* xb) const;

    // Returns the index of vector we contain
    void getIdxMap(int deviceId, std::vector<idx_t> &idxMap) const;

    void search_with_masks(idx_t n, const float *x, idx_t k,
        float *distances, idx_t *labels, const void *mask) const;

    void search_with_masks(idx_t n, const uint16_t *x, idx_t k,
        float *distances, idx_t *labels, const void *mask) const;

    // AscendIndex object is NON-copyable
    AscendIndexFlat(const AscendIndexFlat&) = delete;
    AscendIndexFlat& operator=(const AscendIndexFlat&) = delete;

protected:
    std::shared_ptr<AscendIndexFlatImpl> impl_;
};

// Wrapper around the Ascend implementation that looks like
// faiss::IndexFlatL2; copies over centroid data from a given
// faiss::IndexFlat
class AscendIndexFlatL2 : public AscendIndexFlat {
public:
    // Construct from a pre-existing faiss::IndexFlatL2 instance, copying
    // data over to the given Ascend
    AscendIndexFlatL2(faiss::IndexFlatL2 *index, AscendIndexFlatConfig config = AscendIndexFlatConfig());

    // Construct an empty instance that can be added to
    AscendIndexFlatL2(int dims, AscendIndexFlatConfig config = AscendIndexFlatConfig());

    // Destructor
    virtual ~AscendIndexFlatL2() {}

    // Initialize ourselves from the given CPU index; will overwrite
    // all data in ourselves
    void copyFrom(faiss::IndexFlat *index);

    // Copy ourselves to the given CPU index; will overwrite all data
    // in the index instance
    void copyTo(faiss::IndexFlat *index);

    // AscendIndex object is NON-copyable
    AscendIndexFlatL2(const AscendIndexFlatL2&) = delete;
    AscendIndexFlatL2& operator=(const AscendIndexFlatL2&) = delete;
};
} // ascend
} // faiss
#endif // ASCEND_INDEX_FLAT_INCLUDED