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


#ifndef ASCEND_INDEX_INT8_INCLUDED
#define ASCEND_INDEX_INT8_INCLUDED

#include <vector>

#include "ascenddaemon/impl/Index.h"
#include "ascenddaemon/AscendResourcesProxy.h"

namespace ascend {
struct IDSelector;

class IndexSchemaBase {
public:
    virtual ~IndexSchemaBase() {}
    virtual idx_t getSize() const = 0;
    virtual int getBlockSize() const = 0;
    virtual int getBurstsOfBlock() const = 0;
};

class IndexInt8 {
public:
    // index constructor, resourceSize = -1 means using default config of resource
    explicit IndexInt8(idx_t d = 0, MetricType metric = MetricType::METRIC_L2, int64_t resourceSize = -1);

    virtual ~IndexInt8();

    virtual APP_ERROR init();

    virtual void getBaseEnd() = 0;

    // Return the number of dimension we are indexing
    virtual int getDim() const final { return dims; };

    // Perform training on a representative set of vectors
    virtual void train(idx_t n, const int8_t *x);

    virtual APP_ERROR addVectors(AscendTensor<int8_t, DIMS_2> &rawData);

    // removes all elements from the database.
    virtual void reset() = 0;

    // reserve memory for the database.
    virtual void reserveMemory(size_t numVecs);

    // After adding vectors, one can call this to reclaim device memory
    // to exactly the amount needed. Returns space reclaimed in bytes
    virtual size_t reclaimMemory();

    // remove IDs from the index. Not supported by all indexes.
    //  Returns the number of elements removed.
    virtual size_t removeIds(const IDSelector &sel);

    // query n vectors of dimension d to the index
    // return at most k vectors. If there are not enough results for a query,
    // the result array is padded with -1s.
    APP_ERROR search(idx_t n, const int8_t *x, idx_t k, float16_t *distances, idx_t *labels, uint8_t *mask = nullptr);

    APP_ERROR search(std::vector<IndexInt8 *> indexes, idx_t n, const int8_t *x, idx_t k, float16_t *distances,
        idx_t *labels);

    virtual void setPageSize(uint16_t pageBlockNum) = 0;

public:
    // vector dimension
    int dims;

    // total nb of indexed vectors
    idx_t ntotal;

protected:
    virtual APP_ERROR searchImpl(int n, const int8_t *x, int k, float16_t *distances, idx_t *labels) = 0;

    virtual size_t removeIdsImpl(const IDSelector &sel) = 0;

    APP_ERROR addPaged(int n, const int8_t *x, const idx_t *ids);

    APP_ERROR searchBatched(int64_t n, const int8_t *x, int64_t k, float16_t *distance, idx_t *labels);

    APP_ERROR searchBatched(std::vector<IndexInt8 *> indexes, int64_t n, const int8_t *x, int64_t k,
        float16_t *distances, idx_t *labels);

    virtual APP_ERROR searchImpl(std::vector<IndexInt8 *> indexes, int n, int batchSize, const int8_t *x, int k,
        float16_t *distances, idx_t *labels) = 0;

    virtual void reorder(int indexNum, int n, int k, float16_t *distances, idx_t *labels);

    virtual void initSearchResult(int indexesSize, int n, int k, float16_t *distances, idx_t *labels);

protected:
    // Manage resources on ascend
    AscendResourcesProxy resources;

    // metric type
    MetricType metricType;

    // set if the Index does not require training, or if training is done already
    bool isTrained;

    // mask data
    uint8_t *maskData;
    size_t maskSearchedOffset;
    bool maskOnDevice = false;

    // support search batch sizes, default is no paging
    std::vector<int> searchBatchSizes;

    bool isSupportMultiSearch { false };
};
} // namespace ascend

#endif // ASCEND_INDEX_INT8_INCLUDED
