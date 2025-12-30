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


#ifndef ASCEND_INDEX_FLAT_IMPL_INCLUDED
#define ASCEND_INDEX_FLAT_IMPL_INCLUDED

#include <faiss/MetaIndexes.h>

#include "ascend/AscendIndexFlat.h"
#include "ascend/impl/AscendIndexImpl.h"
#include "ascenddaemon/impl/IndexFlat.h"

namespace faiss {
struct IndexFlat;
struct IndexFlatL2;
} // faiss

namespace faiss {
namespace ascend {
class AscendIndexFlatImpl : public AscendIndexImpl {
public:
    // Construct from a pre-existing faiss::IndexFlat instance
    AscendIndexFlatImpl(const faiss::IndexFlat *index, AscendIndexFlatConfig config, AscendIndex *intf);
    AscendIndexFlatImpl(const faiss::IndexIDMap *index, AscendIndexFlatConfig config, AscendIndex *intf);

    // Construct an empty instance that can be added to
    AscendIndexFlatImpl(int dims, faiss::MetricType metric, AscendIndexFlatConfig config, AscendIndex *intf);

    virtual ~AscendIndexFlatImpl();

    // Initialize ourselves from the given CPU index; will overwrite
    // all data in ourselves
    void copyFrom(const faiss::IndexFlat *index);
    void copyFrom(const faiss::IndexIDMap *index);

    // Copy ourselves to the given CPU index; will overwrite all data
    // in the index instance
    void copyTo(faiss::IndexFlat *index) const;
    void copyTo(faiss::IndexIDMap *index) const;

    // Returns the number of vectors we contain
    size_t getBaseSize(int deviceId) const;

    // Returns the vectors of we contain
    void getBase(int deviceId, char* xb) const;

    // Returns the index of vector we contain
    void getIdxMap(int deviceId, std::vector<idx_t> &idxMap) const;

    // AscendIndex object is NON-copyable
    AscendIndexFlatImpl(const AscendIndexFlatImpl &) = delete;
    AscendIndexFlatImpl &operator = (const AscendIndexFlatImpl &) = delete;

    void CheckIndexParams(IndexImplBase &index, bool checkFilterable = false) const override;

    void addPaged(int n, const float* x, const idx_t* ids) override;

    void addPaged(int n, const uint16_t* x, const idx_t* ids) override;

    void search_with_masks(idx_t n, const float *x, idx_t k,
        float *distances, idx_t *labels, const void *mask) const;

    void search_with_masks_fp16(idx_t n, const uint16_t *x, idx_t k,
        float *distances, idx_t *labels, const void *mask) const;

protected:
    std::shared_ptr<::ascend::Index> createIndex(int deviceId) override;

    // Called from AscendIndex for add
    void addImpl(int n, const float *x, const idx_t *ids) override;

    void getBaseImpl(int deviceId, int offset, int n, char *x) const override;

    size_t getAddElementSize() const override;

    size_t getBaseElementSize() const override;

    inline ::ascend::IndexFlat* getActualIndex(int deviceId) const
    {
        FAISS_THROW_IF_NOT_FMT(indexes.find(deviceId) != indexes.end(),
                               "deviceId is out of range, deviceId=%d.", deviceId);
        FAISS_THROW_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE);
        std::shared_ptr<::ascend::Index> index = indexes.at(deviceId);
        auto *pIndex = dynamic_cast<::ascend::IndexFlat *>(index.get());
        FAISS_THROW_IF_NOT_FMT(pIndex != nullptr, "Invalid index device id: %d\n", deviceId);
        return pIndex;
    }

    bool isSupportFp16Search() const override { return true; }

private:
    void copyCode(const faiss::IndexFlat *index, const idx_t *ids = nullptr);

    void add2DeviceFast(int n, const float *codes, const idx_t *ids);

    void add2DeviceFastFp16(int n, const uint16_t *codes, const idx_t *ids);

    void add2DeviceFastImpl(int n, const uint16_t *transCodes, const idx_t *ids);

    void searchWithMaskProcess(idx_t n, const uint16_t *x, idx_t k,
        float *distances, idx_t *labels, const void *mask) const;

private:
    AscendIndexFlatConfig flatConfig;
    mutable std::unordered_map<int, std::mutex> getBaseMtx; // 仅供getBase接口使用，其他场景请勿使用，避免死锁
};
} // ascend
} // faiss
#endif // ASCEND_INDEX_FLAT_INCLUDED