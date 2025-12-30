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


#ifndef ASCEND_ASCENDINDEX_IMPL_INCLUDED
#define ASCEND_ASCENDINDEX_IMPL_INCLUDED

#include <vector>
#include <unordered_map>
#include <memory>
#include <shared_mutex>

#include <faiss/Index.h>

#include "ascend/AscendIndex.h"
#include "ascend/IndexImplBase.h"
#include "ascenddaemon/impl/Index.h"
#include "common/IndexParam.h"
#include "common/threadpool/AscendThreadPool.h"

namespace faiss {
namespace ascend {

class AscendIndexImpl : public IndexImplBase {
public:
    AscendIndexImpl(int dims, faiss::MetricType metric, AscendIndexConfig config, AscendIndex *intf);

    AscendIndexImpl(int dims, faiss::MetricType metric, AscendIndexConfig config, AscendIndex *intf, bool enablePool);

    virtual ~AscendIndexImpl();

    // `x` need to be resident on CPU
    // Handles paged adds if the add set is too large;
    template <typename T>
    void add(idx_t n, const T *x)
    {
        APP_LOG_INFO("AscendIndex add operation started.\n");
        add_with_ids(n, x, nullptr);
        APP_LOG_INFO("AscendIndex add operation finished.\n");
    }

    // `x` and `ids` need to be resident on the CPU;
    // Handles paged adds if the add set is too large;
    template <typename T>
    void add_with_ids(idx_t n, const T *x, const idx_t *ids)
    {
        auto lock = ::ascend::AscendMultiThreadManager::GetWriteLock(mtx);
        APP_LOG_INFO("AscendIndex add_with_ids operation started.\n");
        FAISS_THROW_IF_NOT_MSG(x, "x can not be nullptr.");
        FAISS_THROW_IF_NOT_MSG(this->intf_->is_trained, "Index not trained");
        FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);
        FAISS_THROW_IF_NOT_FMT(this->intf_->ntotal + n < MAX_N, "ntotal must be < %ld", MAX_N);

        CheckFilterTime(n, ids);

        std::vector<idx_t> tmpIds;
        if (ids == nullptr && addImplRequiresIDs()) {
            tmpIds = std::vector<idx_t>(n);

            for (idx_t i = 0; i < n; ++i) {
                tmpIds[i] = this->intf_->ntotal + i;
            }

            ids = tmpIds.data();
        }
        addPaged(n, x, ids);
        APP_LOG_INFO("AscendIndex add_with_ids operation finished.\n");
    }

    // removes IDs from the index. Not supported by all
    // indexes. Returns the number of elements removed.
    size_t remove_ids(const faiss::IDSelector &sel);

    // `x`, `distances` and `labels` need to be resident on the CPU
    void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels) const;

    void search(idx_t n, const uint16_t *x, idx_t k, float *distances, idx_t *labels) const;

    // reserve memory for the database.
    virtual void reserveMemory(size_t numVecs);

    // After adding vectors, one can call this to reclaim device memory
    // to exactly the amount needed. Returns space reclaimed in bytes
    virtual size_t reclaimMemory();

    void reset();

    // Get devices id of index
    std::vector<int> getDeviceList() const;

    // AscendIndex object is NON-copyable
    AscendIndexImpl(const AscendIndexImpl &) = delete;
    AscendIndexImpl &operator = (const AscendIndexImpl &) = delete;

    std::shared_ptr<std::shared_lock<std::shared_mutex>> getReadLock() const;

    const std::shared_ptr<AscendThreadPool> GetPool() const override;
    faiss::idx_t GetIdxFromDeviceMap(int deviceId, int idxId) const override;
    void CheckIndexParams(IndexImplBase &index, bool checkFilterable = false) const override;
    faiss::ascendSearch::AscendIndexIVFSPSQ *GetIVFSPSQPtr() const override;
    void* GetActualIndex(int deviceId, bool isNeedSetDevice) const override;

protected:
    void initIndexes();
    void clearIndexes();

    virtual std::shared_ptr<::ascend::Index> createIndex(int deviceId) = 0;

    // Check whether the filtering timestamp in the ids is positive
    void CheckFilterTime(idx_t n, const idx_t *ids) const;

    // Does addImpl_ require IDs? If so, and no IDs are provided, we will
    // generate them sequentially based on the order in which the IDs are added
    virtual bool addImplRequiresIDs() const;

    // Handles paged adds if the add set is too large, passes to
    // addImpl to actually perform the add for the current page
    virtual void addPaged(int n, const float *x, const idx_t *ids);

    virtual void addPaged(int n, const uint16_t* x, const idx_t* ids);

    // Overridden to actually perform the add
    virtual void addImpl(int n, const float *x, const idx_t *ids) = 0;

    // get the page size for add
    virtual size_t getAddPagedSize(int n) const;

    // get the size of memory every database vector needed to store.
    virtual size_t getAddElementSize() const;

    void calcAddMap(int n, std::vector<int> &addMap);

    // Handles paged search if the search set is too large, passes to
    // searchImpl to actually perform the search for the current page
    virtual void searchPaged(int n, const float *x, int k, float *distances, idx_t *labels) const;

    // Overridden to actually perform the search
    virtual void searchImpl(int n, const float *x, int k, float *distances, idx_t *labels) const;

    virtual void searchPaged(int n, const uint16_t *x, int k, float *distances, idx_t *labels) const;

    virtual void searchImpl(int n, const uint16_t *x, int k, float *distances, idx_t *labels) const;

    // get the page size for search
    virtual size_t getSearchPagedSize(int n, int k) const;

    // Overridden to actually perform the remove_ids
    virtual size_t removeImpl(const IDSelector &sel);

    void removeSingle(std::vector<std::vector<ascend_idx_t>> &removes, int deviceNum, ascend_idx_t idx);

    void removeIdx(std::vector<std::vector<ascend_idx_t>> &removeMaps);

    // get the size of base.
    virtual size_t getBaseSize(int deviceId) const;

    // get the vector of base, but it not suitable for IVF, should pre-alloc memory for xb(size from getBaseSize)
    void getBase(int deviceId, char *xb) const;

    void getBasePaged(int deviceId, int n, char *codes) const;

    // Overridden to actually perform the get base
    virtual void getBaseImpl(int deviceId, int offset, int n, char *x) const;

    // get the size of every vector of base.
    virtual size_t getBaseElementSize() const;

    virtual size_t getBasePagedSize(int n) const;

    void getIdxMap(int deviceId, std::vector<idx_t> &idxMap) const;

    // merge topk results from all devices used in search process
    virtual void mergeSearchResult(size_t devices, std::vector<std::vector<float>> &dist,
                                   std::vector<std::vector<ascend_idx_t>> &label, idx_t n, idx_t k,
                                   float *distances, idx_t *labels) const;

    std::function<bool(float, float)> GetCompFunc() const;

    // post process after search results got from all devices
    virtual void searchPostProcess(size_t devices, std::vector<std::vector<float>> &dist,
                                   std::vector<std::vector<ascend_idx_t>> &label, idx_t n,
                                   idx_t k, float *distances, idx_t *labels) const;

    virtual void checkParameters(int dims, faiss::MetricType metric, AscendIndexConfig config, AscendIndex *intf) const;

    void indexSearch(IndexParam<uint16_t, uint16_t, ascend_idx_t> &param) const;

    inline ::ascend::Index* getActualIndex(int deviceId, bool isNeedSetDevice = true) const
    {
        FAISS_THROW_IF_NOT_FMT(indexes.find(deviceId) != indexes.end(),
                               "deviceId is out of range, deviceId=%d.", deviceId);
        if (isNeedSetDevice) {
            FAISS_THROW_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE);
        }
        std::shared_ptr<::ascend::Index> index = indexes.at(deviceId);
        FAISS_THROW_IF_NOT_FMT(index.get() != nullptr, "Invalid index device id: %d\n", deviceId);
        return index.get();
    }

    template <typename T>
    inline void check(int64_t n, const T *x, int64_t k, const float *distances,
                      const idx_t *labels) const
    {
        FAISS_THROW_IF_NOT_FMT((n > 0) && (n < MAX_N), "n must be > 0 and < %ld", MAX_N);
        FAISS_THROW_IF_NOT_FMT((k > 0) && (k <= MAX_K), "k must be > 0 and <= %ld", MAX_K);
        FAISS_THROW_IF_NOT_MSG(x != nullptr, "x cannot be nullptr.");
        FAISS_THROW_IF_NOT_MSG(distances != nullptr, "distance cannot be nullptr.");
        FAISS_THROW_IF_NOT_MSG(labels != nullptr, "labels cannot be nullptr.");
        FAISS_THROW_IF_NOT_MSG(intf_->is_trained, "Index not trained");
    }

    template <typename T>
    void searchProcess(idx_t n, const T *x, idx_t k, float *distances, idx_t *labels) const
    {
        auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
        APP_LOG_INFO("AscendIndex start to searchProcess: searchNum=%ld, topK=%ld.\n", n, k);
        check(n, x, k, distances, labels);
        FAISS_THROW_IF_NOT_MSG(indexes.size() > 0, "indexes.size must be >0");
    
        searchPaged(n, x, k, distances, labels);
        APP_LOG_INFO("AscendIndex searchProcess operation finished.\n");
    }

    virtual bool isSupportFp16Search() const { return false; }

protected:
    AscendIndex *intf_;

    AscendIndexConfig indexConfig;

    // thread pool for multithread processing
    std::shared_ptr<AscendThreadPool> pool;

    // recorder assign index of each vector
    std::vector<std::vector<idx_t>> idxDeviceMap;

    // deviceId --> index object
    std::unordered_map<int, std::shared_ptr<::ascend::Index>> indexes;

    mutable std::shared_mutex mtx;
};
} // namespace ascend
} // namespace faiss

#endif