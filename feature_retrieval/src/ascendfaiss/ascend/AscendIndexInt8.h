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


#ifndef ASCEND_ASCEND_INDEX_INT8_INCLUDED
#define ASCEND_ASCEND_INDEX_INT8_INCLUDED

#include <vector>
#include <unordered_map>
#include <memory>

#include <faiss/Index.h>
#include <faiss/impl/FaissAssert.h>
#include "ascend/AscendIndex.h"

namespace faiss {
namespace ascend {
using rpcContext = void *;
using ascend_idx_t = uint64_t;

const int64_t INDEX_INT8_DEFAULT_MEM = 0x2000000; // 0x2000000 mean 32M(resource mem pool's size)
class IndexImplBase;
struct AscendIndexInt8Config {
    inline AscendIndexInt8Config() : deviceList({ 0 }), resourceSize(INDEX_INT8_DEFAULT_MEM) {}

    inline AscendIndexInt8Config(std::initializer_list<int> devices, int64_t resources = INDEX_INT8_DEFAULT_MEM)
        : deviceList(devices), resourceSize(resources) {}

    inline AscendIndexInt8Config(std::vector<int> devices, int64_t resources = INDEX_INT8_DEFAULT_MEM)
        : deviceList(devices), resourceSize(resources) {}

    // Ascend devices mask on which the index is resident
    std::vector<int> deviceList;
    int64_t resourceSize;
};

class AscendIndexInt8Impl;
class AscendIndexInt8 {
public:
    AscendIndexInt8(int dims, faiss::MetricType metric, AscendIndexInt8Config config);

    virtual ~AscendIndexInt8();

    // Perform training on a representative set of vectors
    virtual void train(idx_t n, const int8_t *x);

    // `x` need to be resident on CPU
    // update the centroids clustering center of the training result
    virtual void updateCentroids(idx_t n, const int8_t *x);

    // `x` need to be resident on CPU
    // update the centroids clustering center of the training result
    virtual void updateCentroids(idx_t n, const char *x);

    // `x` need to be resident on CPU
    // Handles paged adds if the add set is too large;
    void add(idx_t n, const int8_t *x);

    // `x` and `ids` need to be resident on the CPU;
    // Handles paged adds if the add set is too large;
    void add_with_ids(idx_t n, const int8_t *x, const idx_t *ids);

    // `x` need to be resident on CPU
    // Handles paged adds if the add set is too large;
    void add(idx_t n, const char *x);

    // `x` and `ids` need to be resident on the CPU;
    // Handles paged adds if the add set is too large;
    void add_with_ids(idx_t n, const char *x, const idx_t *ids);

    // removes IDs from the index. Not supported by all
    // indexes. Returns the number of elements removed.
    size_t remove_ids(const faiss::IDSelector &sel);

    void assign(idx_t n, const int8_t *x, idx_t *labels, idx_t k = 1);

    // `x`, `distances` and `labels` need to be resident on the CPU
    void search(idx_t n, const int8_t *x, idx_t k, float *distances, idx_t *labels) const;

    // `x`, `distances` and `labels` need to be resident on the CPU
    void search(idx_t n, const char *x, idx_t k, float *distances, idx_t *labels) const;

    // reserve memory for the database.
    virtual void reserveMemory(size_t numVecs);

    // After adding vectors, one can call this to reclaim device memory
    // to exactly the amount needed. Returns space reclaimed in bytes
    virtual size_t reclaimMemory();
    // Return ImplBase for inner use
    IndexImplBase& GetIndexImplBase() const;

    // Get devices id of index
    std::vector<int> getDeviceList() const;

    // AscendIndex object is NON-copyable
    AscendIndexInt8(const AscendIndexInt8&) = delete;
    AscendIndexInt8& operator=(const AscendIndexInt8&) = delete;

    int getDim() const;

    faiss::idx_t getNTotal() const;

    bool isTrained() const;

    faiss::MetricType getMetricType() const;

public:
    // verbose level
    bool verbose = false;

protected:
    std::shared_ptr<AscendIndexInt8Impl> impl_;
};

#define VALUE_UNUSED(x) (void)(x)
} // namespace ascend
} // namespace faiss

#endif