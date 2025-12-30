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


#ifndef ASCEND_INDEX_SQ_INCLUDED
#define ASCEND_INDEX_SQ_INCLUDED

#include <faiss/IndexScalarQuantizer.h>
#include <faiss/MetaIndexes.h>
#include "ascend/AscendIndex.h"

namespace faiss {
namespace ascend {
const int64_t SQ_DEFAULT_MEM = 0x8000000; // 0x8000000 mean 128M(resource mem pool's size)

struct AscendIndexSQConfig : public AscendIndexConfig {
    inline AscendIndexSQConfig() : AscendIndexConfig({ 0 }, SQ_DEFAULT_MEM, DEFAULT_BLOCK_SIZE) {}

    inline AscendIndexSQConfig(std::initializer_list<int> devices, int64_t resourceSize = SQ_DEFAULT_MEM,
        uint32_t blockSize = DEFAULT_BLOCK_SIZE)
        : AscendIndexConfig(devices, resourceSize, blockSize)
    {}

    inline AscendIndexSQConfig(std::vector<int> devices, int64_t resourceSize = SQ_DEFAULT_MEM,
        uint32_t blockSize = DEFAULT_BLOCK_SIZE)
        : AscendIndexConfig(devices, resourceSize, blockSize)
    {}
};

class AscendIndexSQImpl;
class AscendIndexSQ : public AscendIndex {
public:
    // Construct an index from CPU IndexSQ
    AscendIndexSQ(const faiss::IndexScalarQuantizer *index, AscendIndexSQConfig config = AscendIndexSQConfig());

    AscendIndexSQ(const faiss::IndexIDMap *index, AscendIndexSQConfig config = AscendIndexSQConfig());

    AscendIndexSQ(int dims, faiss::ScalarQuantizer::QuantizerType qType = ScalarQuantizer::QuantizerType::QT_8bit,
        faiss::MetricType metric = MetricType::METRIC_L2, AscendIndexSQConfig config = AscendIndexSQConfig());

    virtual ~AscendIndexSQ();

    // Initialize ourselves from the given CPU index; will overwrite
    // all data in ourselves
    void copyFrom(const faiss::IndexScalarQuantizer *index);
    void copyFrom(const faiss::IndexIDMap *index);

    // Copy ourselves to the given CPU index; will overwrite all data
    // in the index instance
    void copyTo(faiss::IndexScalarQuantizer *index) const;
    void copyTo(faiss::IndexIDMap *index) const;

    // Returns the codes of we contain, should pre-alloc memory for xb(size from getBaseSize interface)
    void getBase(int deviceId, char* xb) const;

    // Returns the number of codes we contain
    size_t getBaseSize(int deviceId) const;

    // Returns the index of vector we contain
    void getIdxMap(int deviceId, std::vector<idx_t> &idxMap) const;

    void train(idx_t n, const float *x) override;

    void search_with_masks(idx_t n, const float *x, idx_t k,
        float *distances, idx_t *labels, const void *mask) const;

    void search_with_filter(idx_t n, const float *x, idx_t k,
        float *distances, idx_t *labels, const void *filters) const;

    // AscendIndex object is NON-copyable
    AscendIndexSQ(const AscendIndexSQ&) = delete;
    AscendIndexSQ& operator=(const AscendIndexSQ&) = delete;

private:
    std::shared_ptr<AscendIndexSQImpl> impl_;
};
} // ascend
} // faiss
#endif