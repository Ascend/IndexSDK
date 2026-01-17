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


#ifndef ASCEND_INDEX_IVFFLAT_IMPL_INCLUDED
#define ASCEND_INDEX_IVFFLAT_IMPL_INCLUDED


#include "ascend/AscendIndexIVFFlat.h"
#include "ascend/impl/AscendIndexIVFImpl.h"
#include "ascenddaemon/impl/IndexIVFFlat.h"
#include "ascend/utils/AscendIVFAddInfo.h"

namespace faiss {
namespace ascend {
class AscendIndexIVFFlatImpl : public AscendIndexIVFImpl {
public:
    // Construct an empty index
    AscendIndexIVFFlatImpl(AscendIndexIVFFlat *intf, int dims, int nlist,
        faiss::MetricType metric = MetricType::METRIC_INNER_PRODUCT,
        AscendIndexIVFFlatConfig config = AscendIndexIVFFlatConfig());

    virtual ~AscendIndexIVFFlatImpl();

    void train(idx_t n, const float *x, bool clearNpuData = true);
    size_t getAddElementSize() const override;

    AscendIndexIVFFlatImpl(const AscendIndexIVFFlatImpl&) = delete;
    AscendIndexIVFFlatImpl& operator=(const AscendIndexIVFFlatImpl&) = delete;
    void addPaged(int n, const float* x, const idx_t* ids);
    size_t getAddPagedSize(int n) const;
    void searchImpl(int n, const float *x, int k, float *distances, idx_t *labels) const override;

    void copyFrom(const faiss::IndexIVFFlat* index);

    void copyFromCentroids(const faiss::IndexIVFFlat* index);

    void copyFromIVF(const faiss::IndexIVFFlat* index);

    void copyTo(faiss::IndexIVFFlat* index) const;

    void indexIVFFlatGetListCodes(int deviceId, int nlist, InvertedLists *ivf) const;

protected:
    void indexSearch(IndexParam<float, float, ascend_idx_t> &param) const;
    void checkParams() const;
    std::shared_ptr<::ascend::Index> createIndex(int deviceId) override;

    // Called from AscendIndex for add/add_with_ids
    void addL1(int n, const float *x, std::unique_ptr<idx_t[]> &assign);
    void addImpl(int n, const float *x, const idx_t *ids) override;
    void indexIVFFlatAdd(IndexParam<float, float, ascend_idx_t> &param);
    inline ::ascend::IndexIVFFlat* getActualIndex(int deviceId) const
    {
        FAISS_THROW_IF_NOT_FMT(indexes.find(deviceId) != indexes.end(),
                               "deviceId is out of range, deviceId=%d.", deviceId);
        FAISS_THROW_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE);
        std::shared_ptr<::ascend::Index> index = indexes.at(deviceId);
        auto *pIndex = dynamic_cast<::ascend::IndexIVFFlat *>(index.get());
        FAISS_THROW_IF_NOT_FMT(pIndex != nullptr, "Invalid index device id: %d\n", deviceId);
        return pIndex;
    }
    void updateCoarseCenter(std::vector<float> &centerData);
    void copyVectorToDevice(int n);
    void initFlatAtFp32();
    AscendIndexIVFFlat *intf_;
    std::vector<float> centroidsData;
    std::unique_ptr<::ascend::IndexIVFFlat> assignIndex; // 复用ivfflat一阶段检索能力加速add过程和npu聚类

private:
    AscendIndexIVFFlatConfig ivfflatConfig;
    std::unordered_map<int, AscendIVFAddInfo> assignCounts;
};
} // ascend
} // faiss
#endif