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


#ifndef ASCEND_INDEX_INT8_FLAT_IMPL_INCLUDED
#define ASCEND_INDEX_INT8_FLAT_IMPL_INCLUDED

#include <faiss/IndexScalarQuantizer.h>
#include <faiss/Index.h>

#include "ascend/AscendIndexInt8Flat.h"
#include "ascend/impl/AscendIndexInt8Impl.h"
#include "ascend/impl/AscendInt8PipeSearchImpl.h"

namespace faiss {
namespace ascend {
class AscendIndexInt8FlatImpl : public AscendIndexInt8Impl {
public:
    // Construct an empty instance that can be added to
    AscendIndexInt8FlatImpl(int dims, faiss::MetricType metric,
        AscendIndexInt8FlatConfig config, AscendIndexInt8 *intf);

    // Construct an index from CPU IndexSQ
    AscendIndexInt8FlatImpl(const faiss::IndexScalarQuantizer *index,
        AscendIndexInt8FlatConfig config, AscendIndexInt8 *intf);

    // Construct an index from CPU IndexSQ
    AscendIndexInt8FlatImpl(const faiss::IndexIDMap *index,
        AscendIndexInt8FlatConfig config, AscendIndexInt8 *intf);

    virtual ~AscendIndexInt8FlatImpl();

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
    AscendIndexInt8FlatImpl(const AscendIndexInt8FlatImpl &) = delete;
    AscendIndexInt8FlatImpl &operator = (const AscendIndexInt8FlatImpl &) = delete;

protected:
    // Called from AscendIndex for add
    void addImpl(int n, const int8_t *x, const idx_t *ids);

    size_t removeImpl(const IDSelector &sel);

    void copyCode(const faiss::IndexScalarQuantizer* index, const idx_t *ids = nullptr);

    void copyImpl(int n, const int8_t *codes, const idx_t *ids);

    void calcAddMap(int n, std::vector<int> &addMap);

    void add2Device(int n, const int8_t *codes, const idx_t *ids, const std::vector<int> &addMap);
    
    void add2DeviceFast(int n, const int8_t *codes, const idx_t *ids, const std::vector<int> &addMap);

protected:
    void getPaged(int deviceId, int n, std::vector<int8_t> &xb) const;

    void getImpl(int deviceId, int offset, int n, int8_t *x) const;

    void searchPostProcess(size_t devices, std::vector<std::vector<float>> &dist,
        std::vector<std::vector<ascend_idx_t>> &label, idx_t n, idx_t k,
        float *distances, idx_t *labels) const;

    void searchPaged(int n, const int8_t *x, int k, float *distances, idx_t *labels) const override;
    void searchPagedImpl(int n, const int8_t *x, int k, float *distances, idx_t *labels) const;

    void removeSingle(std::vector<std::vector<ascend_idx_t>> &removes, int deviceNum, ascend_idx_t idx);

    void removeIdx(std::vector<std::vector<ascend_idx_t>> &removeMaps);

    size_t getElementSize() const;

    void CheckIndexParams(IndexImplBase &index, bool checkFilterable = false) const override;
    std::shared_ptr<::ascend::IndexInt8> createIndex(int deviceId) override;
    size_t indexInt8FlatGetBaseSize(int deviceId, faiss::MetricType metric) const;
    void int8FlatGetBase(int deviceId, uint32_t offset, uint32_t num,
                         std::vector<int8_t> &vectors, faiss::MetricType metric) const;
    void indexInt8FlatAdd(int deviceId, int n, int dim, int8_t *data) const;

protected:
    AscendIndexInt8FlatConfig int8FlatConfig;

    std::unique_ptr<AscendInt8PipeSearchImpl> pInt8PipeSearchImpl;

private:
    void getBase(int deviceId, int8_t *xb) const;

    void getPaged(int deviceId, int n, int8_t *xb) const;
    mutable std::unordered_map<int, std::mutex> getBaseMtx; // 仅供getBase接口使用，其他场景请勿使用，避免死锁
};
} // ascend
} // faiss
#endif // ASCEND_INDEX_INT8_FLAT_INCLUDED