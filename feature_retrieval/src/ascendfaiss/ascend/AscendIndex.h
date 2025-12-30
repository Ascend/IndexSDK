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


#ifndef ASCEND_ASCENDINDEX_INCLUDED
#define ASCEND_ASCENDINDEX_INCLUDED

#include <vector>
#include <unordered_map>
#include <memory>

#include <faiss/Index.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss {
namespace ascend {
using rpcContext = void *;
using ascend_idx_t = uint64_t;

const idx_t MAX_N = 1e9;
const idx_t MAX_K = 4096;
const int64_t INDEX_DEFAULT_MEM = 0x2000000; // 0x2000000 mean 32M(resource mem pool's size)
const int64_t INDEX_MAX_MEM = 0x100000000;    // 0x100000000 mean 4096MB
const uint32_t DEFAULT_BLOCK_SIZE = 16384 * 16;

struct AscendIndexConfig {
    inline AscendIndexConfig() : deviceList({ 0 }), resourceSize(INDEX_DEFAULT_MEM), dBlockSize(DEFAULT_BLOCK_SIZE) {}

    inline AscendIndexConfig(std::initializer_list<int> devices, int64_t resources = INDEX_DEFAULT_MEM,
        uint32_t blockSize = DEFAULT_BLOCK_SIZE)
        : deviceList(devices), resourceSize(resources), dBlockSize(blockSize) {}

    inline AscendIndexConfig(std::vector<int> devices, int64_t resources = INDEX_DEFAULT_MEM,
        uint32_t blockSize = DEFAULT_BLOCK_SIZE)
        : deviceList(devices), resourceSize(resources), dBlockSize(blockSize) {}

    // Ascend devices mask on which the index is resident
    std::vector<int> deviceList;
    int64_t resourceSize;
    bool slim = false;
    bool filterable = false;
    uint32_t dBlockSize = DEFAULT_BLOCK_SIZE;
};

class AscendIndexImpl;
class IndexImplBase;
class AscendIndex : public faiss::Index {
public:
    AscendIndex(int dims, faiss::MetricType metric, AscendIndexConfig config);

    virtual ~AscendIndex();

    // `x` need to be resident on CPU
    // Handles paged adds if the add set is too large;
    void add(idx_t n, const float *x) override;

    // `x` and `ids` need to be resident on the CPU;
    // Handles paged adds if the add set is too large;
    void add_with_ids(idx_t n, const float *x, const idx_t *ids) override;

    // removes IDs from the index. Not supported by all
    // indexes. Returns the number of elements removed.
    size_t remove_ids(const faiss::IDSelector &sel) override;

    // `x`, `distances` and `labels` need to be resident on the CPU
    void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels,
        const SearchParameters *params = nullptr) const override;

    // reserve memory for the database.
    virtual void reserveMemory(size_t numVecs);

    // After adding vectors, one can call this to reclaim device memory
    // to exactly the amount needed. Returns space reclaimed in bytes
    virtual size_t reclaimMemory();

    void reset() override;

    // Get devices id of index
    std::vector<int> getDeviceList();

    // Return ImplBase for inner use
    IndexImplBase& GetIndexImplBase() const;

    // AscendIndex object is NON-copyable
    AscendIndex(const AscendIndex&) = delete;
    AscendIndex& operator=(const AscendIndex&) = delete;

    void add(idx_t n, const uint16_t *x);

    void add_with_ids(idx_t n, const uint16_t *x, const idx_t *ids);

    void search(idx_t n, const uint16_t *x, idx_t k, float *distances, idx_t *labels) const;

protected:
    std::shared_ptr<AscendIndexImpl> impl_;
};

#define VALUE_UNUSED(x) (void)(x)
} // namespace ascend
} // namespace faiss

#endif