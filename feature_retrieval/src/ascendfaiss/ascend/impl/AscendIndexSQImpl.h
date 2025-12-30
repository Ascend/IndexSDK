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


#ifndef ASCEND_INDEX_SQ_IMPL_INCLUDED
#define ASCEND_INDEX_SQ_IMPL_INCLUDED

#include <faiss/IndexScalarQuantizer.h>
#include <faiss/MetaIndexes.h>

#include "ascend/AscendIndex.h"
#include "ascend/AscendIndexSQ.h"
#include "ascend/impl/AscendIndexImpl.h"
#include "ascenddaemon/impl/IndexSQ.h"

namespace faiss {
namespace ascend {

class AscendIndexSQImpl : public AscendIndexImpl {
public:
    // Construct an index from CPU IndexSQ
    AscendIndexSQImpl(AscendIndexSQ *intf, const faiss::IndexScalarQuantizer *index, AscendIndexSQConfig config);

    AscendIndexSQImpl(AscendIndexSQ *intf, const faiss::IndexIDMap *index, AscendIndexSQConfig config);

    AscendIndexSQImpl(AscendIndexSQ *intf, int dims, faiss::ScalarQuantizer::QuantizerType qType,
        faiss::MetricType metric, AscendIndexSQConfig config);

    virtual ~AscendIndexSQImpl();

    // Initialize ourselves from the given CPU index; will overwrite
    // all data in ourselves
    void copyFrom(const faiss::IndexScalarQuantizer *index);
    void copyFrom(const faiss::IndexIDMap *index);

    // Copy ourselves to the given CPU index; will overwrite all data
    // in the index instance
    void copyTo(faiss::IndexScalarQuantizer *index) const;
    void copyTo(faiss::IndexIDMap *index) const;

    // Returns the codes of we contain, should pre-alloc memory for xb(size from getBaseSize interface)
    void getBase(int deviceId, char* codes) const;

    // Returns the number of codes we contain
    size_t getBaseSize(int deviceId) const;

    // Returns the index of vector we contain
    void getIdxMap(int deviceId, std::vector<idx_t> &idxMap) const;

    void train(idx_t n, const float *x);

    void search_with_masks(idx_t n, const float *x, idx_t k,
        float *distances, idx_t *labels, const void *mask) const;

    void search_with_filter(idx_t n, const float *x, idx_t k,
        float *distances, idx_t *labels, const void *filters) const;

    void CheckIndexParams(IndexImplBase &index, bool checkFilterable = false) const override;
    // AscendIndex object is NON-copyable
    AscendIndexSQImpl(const AscendIndexSQImpl&) = delete;
    AscendIndexSQImpl& operator=(const AscendIndexSQImpl&) = delete;

protected:
    std::shared_ptr<::ascend::Index> createIndex(int deviceId) override;
    // / Called from AscendIndex for add
    void addImpl(int n, const float *x, const idx_t *ids) override;

    size_t getAddElementSize() const override;

    void getBaseImpl(int deviceId, int offset, int n, char *x) const override;

    size_t getBaseElementSize() const override;

    inline ::ascend::IndexSQ* getActualIndex(int deviceId) const
    {
        FAISS_THROW_IF_NOT_FMT(indexes.find(deviceId) != indexes.end(),
                               "deviceId is out of range, deviceId=%d.", deviceId);
        FAISS_THROW_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE);
        std::shared_ptr<::ascend::Index> index = indexes.at(deviceId);
        auto *pIndex = dynamic_cast<::ascend::IndexSQ *>(index.get());
        FAISS_THROW_IF_NOT_FMT(pIndex != nullptr, "Invalid index device id: %d\n", deviceId);
        return pIndex;
    }

    AscendIndexSQ *intf_;
    faiss::ScalarQuantizer sq;

private:
    void checkParams() const;

    void updateDeviceSQTrainedValue() const;

    void copyCode(const faiss::IndexScalarQuantizer *index, const idx_t *ids = nullptr);

    void copyPaged(const faiss::IndexScalarQuantizer *index, const idx_t *ids);

    void copyImpl(int n, const uint8_t *codes, const idx_t *ids);

    void add2Device(int n, const uint8_t *codes, const idx_t *ids, float *preCompute, std::vector<int> &addMap);

    void add2DeviceFast(int n, const uint8_t *codes, const idx_t *ids, const float *preCompute,
        std::vector<int> &addMap);

    void calcPreCompute(const uint8_t *codes, float *compute, size_t n, float *xMem = nullptr);

    void indexAddVector(int deviceId, const int ntotal, const uint8_t *codes, const float *precomputedVal,
                        const idx_t *ids = nullptr) const;

private:
    AscendIndexSQConfig sqConfig;
    mutable std::unordered_map<int, std::mutex> getBaseMtx; // 仅供getBase接口使用，其他场景请勿使用，避免死锁
};
} // ascend
} // faiss
#endif