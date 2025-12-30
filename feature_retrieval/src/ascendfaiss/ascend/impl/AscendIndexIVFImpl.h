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


#ifndef ASCEND_INDEX_IVF_IMPL_INCLUDED
#define ASCEND_INDEX_IVF_IMPL_INCLUDED

#include <faiss/Clustering.h>

#include <faiss/invlists/InvertedLists.h>

#include "ascend/AscendIndex.h"
#include "ascend/AscendIndexIVF.h"
#include "ascend/impl/AscendIndexImpl.h"
#include "ascenddaemon/impl/IndexIVF.h"

namespace faiss {
namespace ascend {
class AscendIndexQuantizerImpl;

class AscendIndexIVFImpl : public AscendIndexImpl {
public:
    AscendIndexIVFImpl(AscendIndex *intf, int dims, faiss::MetricType metric, int nlist,
        AscendIndexIVFConfig config = AscendIndexIVFConfig());

    virtual ~AscendIndexIVFImpl();

    // Returns the number of inverted lists we're managing
    inline int getNumLists() const
    {
        auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
        return nlist;
    }

    // Copy what we need from the CPU equivalent
    void copyFrom(const faiss::IndexIVF *index);

    // Copy what we have to the CPU equivalent
    void copyTo(faiss::IndexIVF *index) const;

    // Clears out all inverted lists, but retains the trained information
    void reset();

    // Sets the number of list probes per query
    virtual void setNumProbes(int nprobes);

    // Returns our current number of list probes per query
    inline int getNumProbes() const
    {
        auto lock = ::ascend::AscendMultiThreadManager::GetReadLock(mtx);
        return nprobe;
    }

    // reserve memory for the database.
    void reserveMemory(size_t numVecs) override;

    // After adding vectors, one can call this to reclaim device memory
    // to exactly the amount needed. Returns space reclaimed in bytes
    size_t reclaimMemory() override;

    // return the list length of a particular list
    virtual uint32_t getListLength(int listId) const;

    // return the list codes of a particular list
    virtual void getListCodesAndIds(int listId, std::vector<uint8_t> &codes, std::vector<ascend_idx_t> &ids) const;

    // AscendIndex object is NON-copyable
    AscendIndexIVFImpl(const AscendIndexIVFImpl &) = delete;
    AscendIndexIVFImpl &operator = (const AscendIndexIVFImpl &) = delete;

protected:
    virtual void threadUnsafeReset();

    void checkIVFParams();

    void initDeviceAddNumMap();

    // train L1 IVF quantizer
    virtual void trainQuantizer(idx_t n, const float *x, bool clearNpuData = true);

    // update coarse centroid to device
    void updateDeviceCoarseCenter();

    // Called from AscendIndex for remove
    size_t removeImpl(const IDSelector &sel) override;

    void searchPostProcess(size_t devices, std::vector<std::vector<float>> &dist,
        std::vector<std::vector<ascend_idx_t>> &label, idx_t n, idx_t k, float *distances,
        idx_t *labels) const override;

    void indexIVFFastGetListCodes(int deviceId, int nlist, InvertedLists *ivf) const;

    void indexIVFGetListCodes(int deviceId, int listId, std::vector<uint8_t> &codes,
                             std::vector<ascend_idx_t> &ids) const;

    inline ::ascend::IndexIVF* getActualIndex (int deviceId) const
    {
        FAISS_THROW_IF_NOT_FMT(indexes.find(deviceId) != indexes.end(),
                               "deviceId is out of range, deviceId=%d.", deviceId);
        FAISS_THROW_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE);
        std::shared_ptr<::ascend::Index> index = indexes.at(deviceId);
        auto *pIndex = dynamic_cast<::ascend::IndexIVF *>(index.get());
        FAISS_THROW_IF_NOT_FMT(pIndex != nullptr, "Invalid index device id: %d\n", deviceId);
        return pIndex;
    }

    // Number of vectors the quantizer contains
    int nlist;

    // top nprobe for quantizer searching
    int nprobe;

    // npu quantizer
    std::shared_ptr<AscendIndexQuantizerImpl> pQuantizerImpl;

    // config
    AscendIndexIVFConfig ivfConfig;

    // centroidId -> feature index number @ device
    std::vector<std::vector<int>> deviceAddNumMap;
    std::vector<float> centroids;
};
} // namespace ascend
} // namespace faiss
#endif
