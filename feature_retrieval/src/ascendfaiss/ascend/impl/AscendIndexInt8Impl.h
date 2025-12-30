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


#ifndef ASCEND_INDEX_INT8_IMPL_INCLUDED
#define ASCEND_INDEX_INT8_IMPL_INCLUDED

#include <vector>
#include <unordered_map>
#include <memory>
#include <shared_mutex>

#include <faiss/Index.h>

#include "IndexInt8.h"
#include "IndexParam.h"
#include "ascend/AscendIndexInt8.h"
#include "ascend/IndexImplBase.h"

class AscendThreadPool;

namespace faiss {
namespace ascend {
class AscendIndexInt8Impl : public IndexImplBase {
public:
    AscendIndexInt8Impl(int dims, faiss::MetricType metric, AscendIndexInt8Config config, AscendIndexInt8 *intf);

    virtual ~AscendIndexInt8Impl();

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

    void assign(idx_t n, const int8_t *x, idx_t *labels, idx_t k = 1) const;

    // `x`, `distances` and `labels` need to be resident on the CPU
    void search(idx_t n, const int8_t *x, idx_t k, float *distances, idx_t *labels) const;

    // `x`, `distances` and `labels` need to be resident on the CPU
    void search(idx_t n, const char *x, idx_t k, float *distances, idx_t *labels) const;

    // reserve memory for the database.
    virtual void reserveMemory(size_t numVecs);

    // After adding vectors, one can call this to reclaim device memory
    // to exactly the amount needed. Returns space reclaimed in bytes
    virtual size_t reclaimMemory();

    // Get devices id of index
    std::vector<int> getDeviceList() const;

    std::shared_ptr<std::shared_lock<std::shared_mutex>> getReadLock() const;

    // AscendIndex object is NON-copyable
    AscendIndexInt8Impl(const AscendIndexInt8Impl &) = delete;
    AscendIndexInt8Impl &operator = (const AscendIndexInt8Impl &) = delete;

    const std::shared_ptr<AscendThreadPool> GetPool() const override;
    faiss::idx_t GetIdxFromDeviceMap(int deviceId, int idxId) const override;
    void CheckIndexParams(IndexImplBase &index, bool checkFilterable = false) const override;
    faiss::ascendSearch::AscendIndexIVFSPSQ *GetIVFSPSQPtr() const override;
    void* GetActualIndex(int deviceId, bool isNeedSetDevice) const override;

    int getDim() const;

    faiss::idx_t getNTotal() const;

    bool isTrained() const;

    faiss::MetricType getMetricType() const;

protected:
    // Does addImpl_ require IDs? If so, and no IDs are provided, we will
    // generate them sequentially based on the order in which the IDs are added
    virtual bool addImplRequiresIDs() const;

    // Overridden to actually perform the add
    virtual void addImpl(int n, const int8_t *x, const idx_t *ids) = 0;

    // Overridden to actually perform the search
    virtual void searchImpl(int n, const int8_t *x, int k, float *distances, idx_t *labels) const;

    // Overridden to actually perform the remove_ids
    virtual size_t removeImpl(const IDSelector &sel) = 0;

    // Handles paged adds if the add set is too large, passes to
    // addImpl to actually perform the add for the current page
    virtual void addPaged(int n, const int8_t *x, const idx_t *ids);

    // Handles paged search if the search set is too large, passes to
    // searchImpl to actually perform the search for the current page
    virtual void searchPaged(int n, const int8_t *x, int k, float *distances, idx_t *labels) const;

    // get the size of memory every database vector needed to store.
    virtual size_t getElementSize() const = 0;

    // merge topk results from all devices used in search process
    virtual void mergeSearchResult(size_t devices, std::vector<std::vector<float>> &dist,
        std::vector<std::vector<ascend_idx_t>> &label, idx_t n, idx_t k, float *distances,
        idx_t *labels) const;

    std::function<bool(float, float)> GetCompFunc() const;

    // post process after search results got from all devices
    virtual void searchPostProcess(size_t devices, std::vector<std::vector<float>> &dist,
        std::vector<std::vector<ascend_idx_t>> &label, idx_t n, idx_t k, float *distances,
        idx_t *labels) const;

    void initIndexes();
    void clearIndexes();

    virtual std::shared_ptr<::ascend::IndexInt8> createIndex(int deviceId) = 0;

    inline ::ascend::IndexInt8* getActualIndex(int deviceId, bool isNeedSetDevice = true) const
    {
        FAISS_THROW_IF_NOT_FMT(indexes.find(deviceId) != indexes.end(),
                               "deviceId is out of range, deviceId=%d.", deviceId);
        if (isNeedSetDevice) {
            FAISS_THROW_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE);
        }
        std::shared_ptr<::ascend::IndexInt8> index = indexes.at(deviceId);
        FAISS_THROW_IF_NOT_FMT(index.get() != nullptr, "Invalid index device id: %d\n", deviceId);
        return index.get();
    }
    void indexInt8Search(IndexParam<int8_t, uint16_t, ascend_idx_t> param, const int8_t *query, uint16_t *distance,
                         ascend_idx_t *label) const;

protected:
    AscendIndexInt8 *intf_;

    AscendIndexInt8Config indexConfig;

    // thread pool for multithread processing
    std::shared_ptr<AscendThreadPool> pool;

    // recorder assign index of each vector
    std::vector<std::vector<idx_t>> idxDeviceMap;

    std::vector<std::unordered_map<ascend_idx_t, ascend_idx_t>> label2Idx;

    // deviceId --> index object
    std::unordered_map<int, std::shared_ptr<::ascend::IndexInt8>> indexes;

    friend void Search(std::vector<AscendIndexInt8 *> indexes, idx_t n, const int8_t *x, idx_t k,
        float *distances, idx_t *labels, bool merged);

    friend void SearchPostProcess(std::vector<AscendIndexInt8 *> indexes, std::vector<std::vector<float>> &dist,
        std::vector<std::vector<ascend_idx_t>> &label, int n, int k, float *distances, idx_t *labels,
        bool merged);

    // vector dimension
    int dim { 512 };

    // total nb of indexed vectors
    faiss::idx_t ntotal { 0 };

    // set if the Index does not require training, or if training is done already
    bool trained { true };

    // type of metric this index uses for search
    faiss::MetricType metricType { faiss::METRIC_L2 };

    mutable std::shared_mutex mtx;
};
} // namespace ascend
} // namespace faiss

#endif