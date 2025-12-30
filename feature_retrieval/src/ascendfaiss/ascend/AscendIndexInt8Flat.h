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


#ifndef ASCEND_INDEX_INT8_FLAT_INCLUDED
#define ASCEND_INDEX_INT8_FLAT_INCLUDED

#include <faiss/IndexScalarQuantizer.h>
#include <faiss/MetaIndexes.h>
#include "ascend/AscendIndexInt8.h"

namespace faiss {
namespace ascend {
const int64_t INT8_FLAT_DEFAULT_MEM = 0x8000000; // 0x8000000 mean 128M(resource mem pool's size)
const uint32_t BLOCK_SIZE = 16384 * 16;

// index mode
enum class Int8IndexMode {
    DEFAULT_MODE = 0,
    PIPE_SEARCH_MODE,
    WITHOUT_NORM_MODE
};

struct AscendIndexInt8FlatConfig : public AscendIndexInt8Config {
    inline AscendIndexInt8FlatConfig(uint32_t blockSize = BLOCK_SIZE,
        Int8IndexMode indexMode = Int8IndexMode::DEFAULT_MODE)
        : dBlockSize(blockSize), dIndexMode(indexMode)
    {}

    inline AscendIndexInt8FlatConfig(std::initializer_list<int> devices, int64_t resourceSize = INT8_FLAT_DEFAULT_MEM,
        uint32_t blockSize = BLOCK_SIZE, Int8IndexMode indexMode = Int8IndexMode::DEFAULT_MODE)
        : AscendIndexInt8Config(devices, resourceSize), dBlockSize(blockSize), dIndexMode(indexMode)
    {}

    inline AscendIndexInt8FlatConfig(std::vector<int> devices, int64_t resourceSize = INT8_FLAT_DEFAULT_MEM,
        uint32_t blockSize = BLOCK_SIZE, Int8IndexMode indexMode = Int8IndexMode::DEFAULT_MODE)
        : AscendIndexInt8Config(devices, resourceSize), dBlockSize(blockSize), dIndexMode(indexMode)
    {}

    uint32_t dBlockSize;
    Int8IndexMode dIndexMode;
};

class AscendIndexInt8FlatImpl;
class AscendIndexInt8Flat : public AscendIndexInt8 {
public:
    // Construct an empty instance that can be added to
    AscendIndexInt8Flat(int dims, faiss::MetricType metric = faiss::METRIC_L2,
        AscendIndexInt8FlatConfig config = AscendIndexInt8FlatConfig());

    // Construct an index from CPU IndexSQ
    AscendIndexInt8Flat(const faiss::IndexScalarQuantizer *index,
        AscendIndexInt8FlatConfig config = AscendIndexInt8FlatConfig());

    // Construct an index from CPU IndexSQ
    AscendIndexInt8Flat(const faiss::IndexIDMap *index,
        AscendIndexInt8FlatConfig config = AscendIndexInt8FlatConfig());

    virtual ~AscendIndexInt8Flat();

    // Returns the number of vectors we contain
    size_t getBaseSize(int deviceId) const;

    // Returns the vectors of we contain
    void getBase(int deviceId, std::vector<int8_t> &xb) const;

    // Returns the index of vector we contain
    void getIdxMap(int deviceId, std::vector<idx_t> &idxMap) const;

    // Clears all vectors from this index
    void reset();

    // Initialize ourselves from the given CPU index; will overwrite
    // all data in ourselves
    void copyFrom(const faiss::IndexScalarQuantizer* index);
    void copyFrom(const faiss::IndexIDMap* index);

    // Copy ourselves to the given CPU index; will overwrite all data
    // in the index instance
    void copyTo(faiss::IndexScalarQuantizer* index) const;
    void copyTo(faiss::IndexIDMap* index) const;

    void search_with_masks(idx_t n, const int8_t *x, idx_t k,
        float *distances, idx_t *labels, const void *mask) const;

    void setPageSize(uint16_t pageBlockNum);

    // AscendIndex object is NON-copyable
    AscendIndexInt8Flat(const AscendIndexInt8Flat&) = delete;
    AscendIndexInt8Flat& operator=(const AscendIndexInt8Flat&) = delete;

protected:
    std::shared_ptr<AscendIndexInt8FlatImpl> impl_;
};
} // ascend
} // faiss
#endif // ASCEND_INDEX_INT8_FLAT_INCLUDED