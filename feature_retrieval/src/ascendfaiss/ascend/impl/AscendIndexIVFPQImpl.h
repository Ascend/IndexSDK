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

#ifndef ASCEND_INDEX_IVFPQ_IMPL_INCLUDED
#define ASCEND_INDEX_IVFPQ_IMPL_INCLUDED

#include <faiss/Clustering.h>
#include "ascend/AscendIndexIVFPQ.h"
#include "ascend/impl/AscendIndexIVFImpl.h"
#include "ascenddaemon/impl/IndexIVFPQ.h"
#include "ascend/utils/AscendIVFAddInfo.h"

namespace faiss {
namespace ascend {
class AscendIndexIVFPQImpl : public AscendIndexIVFImpl {
public:

    // Construct an empty index
    AscendIndexIVFPQImpl(AscendIndexIVFPQ* intf, int dims, int nlist, int msubs, int nbits,
                         faiss::MetricType metric = MetricType::METRIC_INNER_PRODUCT,
                         AscendIndexIVFPQConfig config = AscendIndexIVFPQConfig());

    virtual ~AscendIndexIVFPQImpl();

    // Initialize ourselves from the given CPU index; will overwrite
    void copyFrom(const faiss::IndexIVFPQ* index);

    // Copy ourselves to the given CPU index; will overwrite all data
    void copyTo(faiss::IndexIVFPQ* index) const;

    void initProductQuantizer();

    void train(idx_t n, const float* x);

    size_t getAddElementSize() const override;

    AscendIndexIVFPQImpl(const AscendIndexIVFPQImpl&) = delete;

    AscendIndexIVFPQImpl& operator=(const AscendIndexIVFPQImpl&) = delete;

    void searchImpl(int n, const float* x, int k, float* distances, idx_t* labels) const override;

    void deleteImpl(int n, const idx_t* ids);

    void deleteFromIVFPQ(IndexParam<void, void, ascend_idx_t>& param);

    idx_t findListId(idx_t id);

    void updateIdMapping(const std::vector<idx_t>& ids, const std::vector<idx_t>& listIds);

    void removeIdMapping(const std::vector<idx_t>& ids);

    std::vector<idx_t> update(idx_t n, const float* x, const idx_t* ids);

    void addPaged(int n, const float* x, const idx_t* ids);
    
    size_t getAddPagedSize(int n) const;

protected:
    void copyFromCentroids(const faiss::IndexIVFPQ* index);

    void copyFromCodebook(const faiss::IndexIVFPQ* index);

    void copyFromPQCodes(const faiss::IndexIVFPQ* index);

    std::vector<std::vector<std::pair<size_t, size_t>>> assignListsToDevices(const faiss::InvertedLists* invlists,
                                                                             size_t deviceCount);

    void uploadToDevicesParallel(const std::vector<std::vector<std::pair<size_t, size_t>>>& deviceAssignments,
                                 const faiss::InvertedLists* invlists);

    void copyToPQCodes(faiss::IndexIVFPQ* index) const;

    void indexSearch(IndexParam<float, float, ascend_idx_t>& param) const;

    void checkParams() const;

    std::shared_ptr<::ascend::Index> createIndex(int deviceId) override;

    void indexTrainImpl(int n, const float* x, int dim, int nlist);

    // Called from AscendIndex for add/add_with_ids
    void addL1(int n, const float* x, std::vector<int64_t>& assign);

    void addL2(int n, const float* x, std::vector<uint8_t>& pqCodes);

    void addImpl(int n, const float* x, const idx_t* ids) override;

    void copyVectorToDevice(int n);

    void indexIVFPQAdd(IndexParam<uint8_t, float, ascend_idx_t>& param);

    std::vector<uint8_t> encodeSingleVectorPQ(const float* vector);

    uint8_t findCentroidInSubQuantizer(size_t subq_idx, const float* sub_vector);

    float calDistance(const float* a, const float* b, size_t dim);

    inline ::ascend::IndexIVFPQ* getActualIndex(int deviceId) const
    {
        FAISS_THROW_IF_NOT_FMT(indexes.find(deviceId) != indexes.end(), "deviceId is out of range, deviceId=%d.",
                               deviceId);
        FAISS_THROW_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE);
        std::shared_ptr<::ascend::Index> index = indexes.at(deviceId);
        auto* pIndex = dynamic_cast<::ascend::IndexIVFPQ*>(index.get());
        FAISS_THROW_IF_NOT_FMT(pIndex != nullptr, "Invalid index device id: %d\n", deviceId);
        return pIndex;
    }

    void updateCoarseCenter(std::vector<float>& centerData);

    void trainPQCodeBook(idx_t n, const float* x);

    void trainSubQuantizer(size_t m, idx_t n, const float* x);

    void updatePQCodeBook();

    void savePQCodeBook(size_t m, const std::vector<float>& centroids);

    AscendIndexIVFPQ* intf_;

    std::vector<float> centroidsData;

    std::vector<float> centroidsOnHost;

    struct PQCodebook {
        size_t nlist;
        size_t dim;
        size_t nbits;
        size_t M;
        size_t ksub;
        size_t dsub;
        std::vector<std::vector<std::vector<float>>> codeBook;
    };

    PQCodebook pq;

    int msubs;
    int nbits;

private:
    AscendIndexIVFPQConfig ivfPQConfig;

    std::unordered_map<idx_t, idx_t> idToListMap;

    std::mutex mapMutex;

    struct ListInfo {
        std::unordered_set<idx_t> idSet;
    };
    std::vector<ListInfo> listInfos;

    std::unordered_map<int, AscendIVFAddInfo> assignCounts;
};
}  // namespace ascend
}  // namespace faiss
#endif