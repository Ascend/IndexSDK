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

#include <faiss/Index.h>

#include "ascend/AscendIndex.h"
#include "common/threadpool/AscendThreadPool.h"

namespace faiss {
namespace ascendSearch {
class AscendIndexImpl {
public:
    AscendIndexImpl(int dims, faiss::MetricType metric, AscendIndexConfig config, AscendIndex *intf);

    virtual ~AscendIndexImpl();

    // `x` need to be resident on CPU
    // Handles paged adds if the add set is too large;
    void add(idx_t n, const float *x);

    // `x` and `ids` need to be resident on the CPU;
    // Handles paged adds if the add set is too large;
    void add_with_ids(idx_t n, const float *x, const idx_t *ids);

    // removes IDs from the index. Not supported by all
    // indexes. Returns the number of elements removed.
    size_t remove_ids(const faiss::IDSelector &sel);

    // `x`, `distances` and `labels` need to be resident on the CPU
    void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels) const;

    // reserve memory for the database.
    virtual void reserveMemory(size_t numVecs);

    // After adding vectors, one can call this to reclaim device memory
    // to exactly the amount needed. Returns space reclaimed in bytes
    virtual size_t reclaimMemory();

    void reset();

    // Get devices id of index
    std::vector<int> getDeviceList();

    // AscendIndex object is NON-copyable
    AscendIndexImpl(const AscendIndexImpl &) = delete;
    AscendIndexImpl &operator = (const AscendIndexImpl &) = delete;

protected:
    void initRpcCtx();

    void clearRpcCtx();

    virtual int CreateIndex(rpcContext ctx) = 0;

    void DestroyIndex(rpcContext ctx, int indexId) const;

    // check the config parameters(except deviceList) of index with another index is same
    virtual void checkParamsSame(AscendIndexImpl &index);

    // Does addImpl_ require IDs? If so, and no IDs are provided, we will
    // generate them sequentially based on the order in which the IDs are added
    virtual bool addImplRequiresIDs() const;

    // Handles paged adds if the add set is too large, passes to
    // addImpl to actually perform the add for the current page
    void addPaged(int n, const float *x, const idx_t *ids);

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

    // get the page size for search
    virtual size_t getSearchPagedSize(int n, int k) const;

    // Overridden to actually perform the remove_ids
    virtual size_t removeImpl(const IDSelector &sel);

    void removeSingle(std::vector<std::vector<ascend_idx_t>> &removes, int deviceNum, ascend_idx_t idx);

    void removeIdx(const std::vector<std::vector<ascend_idx_t>> &removeMaps) const;

    // get the size of base.
    virtual size_t getBaseSize(int deviceId) const;

    // get the vector of base, but it not suitable for IVF, should pre-alloc memory for xb(size from getBaseSize)
    void getBase(int deviceId, char* xb) const;

    void getBasePaged(int deviceId, int n, char* codes) const;

    // Overridden to actually perform the get base
    virtual void getBaseImpl(int deviceId, int offset, int n, char *x) const;

    // get the size of every vector of base.
    virtual size_t getBaseElementSize() const;

    virtual size_t getBasePagedSize(int n) const;

    void getIdxMap(int deviceId, std::vector<idx_t> &idxMap) const;

    // merge topk results from all devices used in search process
    virtual void mergeSearchResult(size_t devices, std::vector<std::vector<float>> &dist,
        std::vector<std::vector<ascend_idx_t>> &label, int n, int k, float *distances, idx_t *labels) const;

    // post process after search results got from all devices
    virtual void searchPostProcess(size_t devices, std::vector<std::vector<float>> &dist,
        std::vector<std::vector<ascend_idx_t>> &label, int n, int k, float *distances, idx_t *labels) const;

protected:
    AscendIndex *intf_ = nullptr;

    AscendIndexConfig indexConfig;

    // thread pool for multithread processing
    std::shared_ptr<AscendThreadPool> pool;

    // device --> context
    std::unordered_map<int, rpcContext> contextMap;

    // context --> index * n
    std::unordered_map<rpcContext, int> indexMap;

    // recorder assign index of each vector
    std::vector<std::vector<idx_t>> idxDeviceMap;

    friend void Search(std::vector<AscendIndex *> indexes, idx_t n, const float *x, idx_t k,
        float *distances, idx_t *labels, bool merged);

    friend void SearchWithFilter(std::vector<AscendIndex *> indexes, idx_t n, const float *x, idx_t k,
        float *distances, idx_t *labels, const void *filters, bool merged);
    
    friend void SearchWithFilter(std::vector<AscendIndex *> indexes, idx_t n, const float *x, idx_t k,
        float *distances, idx_t *labels, void *filters[], bool merged);
    friend void SearchPostProcess(std::vector<AscendIndex *> indexes, std::vector<std::vector<float>> &dist,
        std::vector<std::vector<ascend_idx_t>> &label, int n, int k, float *distances, idx_t *labels,
        bool merged);
};
} // namespace ascendSearch
} // namespace faiss

#endif
