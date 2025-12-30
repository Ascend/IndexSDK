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


#ifndef ASCEND_INDEX_INCLUDED
#define ASCEND_INDEX_INCLUDED

#include <map>
#include <memory>
#include <cstdint>
#include <cstddef>

#include <ascenddaemon/AscendResourcesProxy.h>
#include <ascenddaemon/utils/TopkOp.h>
#include <ascenddaemon/utils/AscendOperator.h>
#include <common/AscendFp16.h>
#include <common/ErrorCode.h>
#include "common/utils/SocUtils.h"

namespace ascendSearch {
using idx_t = uint64_t;
const int CUBE_ALIGN = 16;
const int CUBE_ALIGN_INT8 = 32;
const int FILTER_ALIGN = 512;
const int FLAG_ALIGN = 32;
const int FLAG_ALIGN_OFFSET = 16; // core 0 use first 16 flag, and core 1 use the second 16 flag.
const int FLAG_NUM = 16; // each core have a flag, the max core is 16
const int FLAG_SIZE = 16;

const int SIZE_ALIGN = 8;
const double TIMEOUT_MS = 50000;
const int TIMEOUT_CHECK_TICK = 5120;

const int CORE_NUM = faiss::ascendSearch::SocUtils::GetInstance().GetCoreNum();
const int MAX_CORE_NUM = 16;
const int BIT_OF_UINT8 = 8;

struct IDSelector;
struct IDSelectorBatch;
struct IDSelectorRange;

enum class MetricType {
    METRIC_INNER_PRODUCT = 0,  // maximum inner product search
    METRIC_L2 = 1,             // squared L2 search
    METRIC_COSINE = 2,         // consine search
};

class Index {
public:
    using idx_t = uint64_t;

    // index constructor, resourceSize = -1 means using default config of resource
    explicit Index(ascendSearch::idx_t d = 0, int64_t resourceSize = -1);

    size_t getResourceSize()
    {
        return resources.getResourceSize();
    }

    virtual ~Index();

    virtual APP_ERROR init();

    // Perform training on a representative set of vectors
    virtual void train(idx_t n, const float16_t* x);

    // removes all elements from the database.
    virtual APP_ERROR reset() = 0;

    // reserve memory for the database.
    virtual APP_ERROR reserveMemory(size_t numVecs);

    // After adding vectors, one can call this to reclaim device memory
    // to exactly the amount needed. Returns space reclaimed in bytes
    virtual size_t reclaimMemory();

    // remove IDs from the index. Not supported by all indexes.
    //  Returns the number of elements removed.
    virtual size_t removeIds(const IDSelector& sel);

    // query n vectors of dimension d to the index
    // return at most k vectors. If there are not enough results for a query,
    // the result array is padded with -1s.
    APP_ERROR search(idx_t n, const float16_t* x, idx_t k, float16_t* distances, idx_t* labels);

    APP_ERROR search(std::vector<Index*> indexes, idx_t n, const float16_t *x, idx_t k,
        float16_t *distances, idx_t *labels);

    APP_ERROR searchFilter(idx_t n, const float16_t* x, idx_t k, float16_t* distances, idx_t* labels,
        uint8_t* mask);

    APP_ERROR searchFilter(idx_t n, const float16_t* x, idx_t k, float16_t* distances, idx_t* labels,
        uint64_t filterSize, uint32_t* filters);

    APP_ERROR searchFilter(std::vector<Index *> indexes, idx_t n, const float16_t *x, idx_t k, float16_t *distances,
        idx_t *labels, uint32_t filterSize, uint32_t *filters);
    
    APP_ERROR searchFilter(std::vector<Index *> indexes, idx_t n, const float16_t *x, idx_t k, float16_t *distances,
        idx_t *labels, uint32_t filterSize, uint32_t **filters);

    virtual int getDim() const
    {
        return dims;
    }

    virtual int getDimIn() const
    {
        return dims;
    }

public:
    // vector dimension
    int dims;

    // total nb of indexed vectors
    idx_t ntotal;

protected:
    virtual APP_ERROR searchImpl(int n, const float16_t* x, int k, float16_t* distances, idx_t* labels) = 0;

    virtual APP_ERROR searchImpl(std::vector<Index*> indexes, int n, int batchSize, const float16_t *x, int k,
        float16_t *distances, idx_t *labels);
    
    virtual APP_ERROR searchBatched(std::vector<Index *> indexes, int n, const float16_t *x, int k,
        float16_t *distances, idx_t *labels, uint32_t filterSize, uint32_t *filters);

    virtual APP_ERROR searchBatched(std::vector<Index *> indexes, int n, const float16_t *x, int k,
        float16_t *distances, idx_t *labels, uint32_t filterSize, uint32_t **filters);
      
    virtual APP_ERROR searchFilterImpl(std::vector<Index *> indexes, int n, int batchSize, const float16_t *x, int k,
        float16_t *distances, idx_t *labels, AscendTensor<uint8_t, DIMS_2> &maskData,
        AscendTensor<int, DIMS_1> &maskOffset);

    virtual APP_ERROR searchFilterImpl(int n, const float16_t *x, int k,
        float16_t *distances, idx_t *labels, uint8_t* masks, uint32_t maskLen);

    virtual size_t removeIdsBatch(const IDSelectorBatch &sel);
    virtual size_t removeIdsRange(const IDSelectorRange &sel);

    virtual void moveVectorForward(idx_t srcIdx, idx_t destIdx);
    virtual void releaseUnusageSpace(int oldTotal, int remove);

    APP_ERROR addPaged(int n, const float16_t* x, const idx_t* ids);

    virtual APP_ERROR searchBatched(int n, const float16_t* x, int k, float16_t* distance, idx_t* labels);

    APP_ERROR searchBatched(int n, const float16_t* x, int k, float16_t* distance, idx_t* labels, uint8_t* masks);

    virtual APP_ERROR searchBatched(int n, const float16_t* x, int k, float16_t* distance, idx_t* labels,
        uint64_t filterSize, uint32_t* filters);

    APP_ERROR searchBatched(std::vector<Index*> indexes, int n, const float16_t *x, int k,
        float16_t *distances, idx_t *labels);

    virtual APP_ERROR searchBatched(int n, const float16_t* x, int k, float16_t* distance, idx_t* labels,
        float16_t* l1distances, uint32_t filterSize, uint32_t* filters);

    virtual void reorder(int indexNum, int n, int k, float16_t *distances, idx_t *labels);

    virtual void initSearchResult(int indexesSize, int n, int k, float16_t *distances,
        idx_t *labels);

    APP_ERROR resetDistCompOperator(int numLists);

    void runDistanceCompute(AscendTensor<float16_t, DIMS_2>& queryVecs,
                            AscendTensor<float16_t, DIMS_4>& shapedData,
                            AscendTensor<float16_t, DIMS_1>& norms,
                            AscendTensor<float16_t, DIMS_2>& outDistances,
                            AscendTensor<uint16_t, DIMS_1>& flag,
                            aclrtStream stream);

    float16_t fvecNormL2sqr(const float16_t* x, size_t d) const;
    void fvecNormsL2sqr(float16_t* nr, const float16_t* x, size_t d, size_t nx) const;
    virtual void fvecNormsL2sqrAicpu(AscendTensor<float16_t, DIMS_1> &nr, AscendTensor<float16_t, DIMS_2> &x);

protected:
    // Manage resources on ascend
    AscendResourcesProxy resources;

    // set if the Index does not require training, or if training is done already
    bool isTrained;

    // support search batch sizes, default is no paging
    std::vector<int> searchBatchSizes;

    // shared ops
    std::map<int, std::unique_ptr<AscendOperator>> distComputeOps;
    TopkOp<std::greater<float16_t>, std::greater_equal<float16_t>, float16_t> topkOp;
};
}  // namespace ascendSearch

#endif  // ASCEND_INDEX_INCLUDED
