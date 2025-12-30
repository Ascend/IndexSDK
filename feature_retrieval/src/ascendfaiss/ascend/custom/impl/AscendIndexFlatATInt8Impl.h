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


#ifndef ASCEND_INDEX_FLAT_AT_INT8_IMPL_INCLUDED
#define ASCEND_INDEX_FLAT_AT_INT8_IMPL_INCLUDED
#include <faiss/MetaIndexes.h>

#include "ascend/custom/AscendIndexFlatATInt8.h"
#include "ascend/impl/AscendIndexImpl.h"
#include "index_custom/IndexFlatATInt8Aicpu.h"

namespace faiss {
namespace ascend {
class AscendIndexFlatATInt8Impl : public AscendIndexImpl {
public:
    // Construct an empty instance that can be added to
    AscendIndexFlatATInt8Impl(int dims, int baseSize, AscendIndexFlatATInt8Config config, AscendIndex *intf);

    virtual ~AscendIndexFlatATInt8Impl();

    // Clears all vectors from this index
    void reset();

    void searchInt8(idx_t n, const int8_t *x, idx_t k, float *distances, idx_t *labels) const;

    // AscendIndex object is NON-copyable
    AscendIndexFlatATInt8Impl(const AscendIndexFlatATInt8Impl &) = delete;
    AscendIndexFlatATInt8Impl &operator = (const AscendIndexFlatATInt8Impl &) = delete;

    void clearAscendTensor();

    void sendMinMax(float qMin, float qMax);

protected:
    std::shared_ptr<::ascend::Index> createIndex(int deviceId) override;

    // Flat index does not require IDs as there is no storage available for them
    bool addImplRequiresIDs() const;

    // Called from AscendIndex for add
    void addImpl(int n, const float *x, const idx_t *ids);

    size_t removeImpl(const IDSelector &sel);

    void searchPagedInt8(int n, const int8_t *x, int k, float *distances, idx_t *labels) const;

    void searchImplInt8(int n, const int8_t *x, int k, float *distances, idx_t *labels) const;

    void searchPaged(int n, const float *x, int k, float *distances, idx_t *labels) const;

    void searchImpl(int n, const float *x, int k, float *distances, idx_t *labels) const;

    size_t getSearchPagedSize(int n, int k) const;

    size_t getAddElementSize() const;

    size_t getBaseElementSize() const;

    inline ::ascend::IndexFlatATInt8Aicpu* getActualIndex (int deviceId) const
    {
        FAISS_THROW_IF_NOT_FMT(indexes.find(deviceId) != indexes.end(),
                               "deviceId is out of range, deviceId=%d.", deviceId);
        FAISS_THROW_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE);
        std::shared_ptr<::ascend::Index> index = indexes.at(deviceId);
        auto *pIndex = dynamic_cast<::ascend::IndexFlatATInt8Aicpu  *>(index.get());
        FAISS_THROW_IF_NOT_FMT(pIndex != nullptr, "Invalid index device id: %d\n", deviceId);
        return pIndex;
    }

private:
    void add2DeviceFast(int n, float *codes, const idx_t *ids);

private:
    AscendIndexFlatATInt8Config flatConfig;

    int baseSize;
};
} // ascend
} // faiss
#endif // ASCEND_INDEX_FLAT_AT_INT8_INCLUDED