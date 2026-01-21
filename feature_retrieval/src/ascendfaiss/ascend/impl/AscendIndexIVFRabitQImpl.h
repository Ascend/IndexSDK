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


#ifndef ASCEND_INDEX_IVFRABITQ_IMPL_INCLUDED
#define ASCEND_INDEX_IVFRABITQ_IMPL_INCLUDED

#include <random>
#include "ascend/AscendIndexIVFRabitQ.h"
#include "ascend/impl/AscendIndexIVFImpl.h"
#include "ascenddaemon/impl/IndexIVFRabitQ.h"
#include "ascenddaemon/impl/IndexIVFFlat.h"
#include "ascend/utils/AscendIVFAddInfo.h"

namespace faiss {
namespace ascend {
struct RandomGenerator {
    std::mt19937 mt;

    /// random positive integer
    int rand_int();

    /// random int64_t
    int64_t rand_int64();

    /// generate random integer between 0 and max-1
    int rand_int(int max);

    /// between 0 and 1
    float rand_float();

    double rand_double();

    explicit RandomGenerator(int64_t seed = 1234);
};

class AscendIndexIVFRabitQImpl : public AscendIndexIVFImpl {
public:
    // Construct an empty index
    AscendIndexIVFRabitQImpl(AscendIndexIVFRabitQ *intf, int dims, int nlist,
        faiss::MetricType metric = MetricType::METRIC_INNER_PRODUCT,
        AscendIndexIVFRabitQConfig config = AscendIndexIVFRabitQConfig());

    virtual ~AscendIndexIVFRabitQImpl();

    void train(idx_t n, const float *x, bool clearNpuData = true);
    size_t getAddElementSize() const override;

    AscendIndexIVFRabitQImpl(const AscendIndexIVFRabitQImpl&) = delete;
    AscendIndexIVFRabitQImpl& operator=(const AscendIndexIVFRabitQImpl&) = delete;
    void addPaged(int n, const float* x, const idx_t* ids);
    size_t getAddPagedSize(int n) const;
    void searchImpl(int n, const float *x, int k, float *distances, idx_t *labels) const override;

protected:
    void mergeSearchResultSingleQuery(idx_t qIdx, size_t devices,
                                      std::vector<std::vector<float>>& dist,
                                      std::vector<std::vector<ascend_idx_t>>& label,
                                      idx_t n, idx_t k, size_t eachdeviceK,
                                      float* distances, idx_t* labels,
                                      std::function<bool(float, float)> &compFunc) const;
    // merge topk results from all devices used in search process
    virtual void mergeSearchResult(size_t devices, std::vector<std::vector<float>> &dist,
                                   std::vector<std::vector<ascend_idx_t>> &label, idx_t n, idx_t k,
                                   float *distances, idx_t *labels) const;
    void indexSearch(IndexParam<float, float, ascend_idx_t> &param) const;
    void checkParams() const;
    std::shared_ptr<::ascend::Index> createIndex(int deviceId) override;

    // Called from AscendIndex for add/add_with_ids
    void addL1(int n, const float *x, std::unique_ptr<idx_t[]> &assign);
    void addImpl(int n, const float *x, const idx_t *ids) override;
    void indexIVFRabitQAdd(IndexParam<float, float, ascend_idx_t> &param);
    inline ::ascend::IndexIVFRabitQ* getActualIndex(int deviceId) const
    {
        FAISS_THROW_IF_NOT_FMT(indexes.find(deviceId) != indexes.end(),
                               "deviceId is out of range, deviceId=%d.", deviceId);
        FAISS_THROW_IF_NOT(aclrtSetDevice(deviceId) == ACL_ERROR_NONE);
        std::shared_ptr<::ascend::Index> index = indexes.at(deviceId);
        auto *pIndex = dynamic_cast<::ascend::IndexIVFRabitQ *>(index.get());
        FAISS_THROW_IF_NOT_FMT(pIndex != nullptr, "Invalid index device id: %d\n", deviceId);
        return pIndex;
    }
    void updateCoarseCenter(std::vector<float> &centerData);
    void copyVectorToDevice(int n);
    void initFlatAtFp32();
    void randomOrthogonalGivens(int n, std::vector<float> &orthogonalMatrix);
    void uploadorthogonalMatrix(std::vector<float> &orthogonalMatrix);
    AscendIndexIVFRabitQ *intf_;
    std::vector<float> srcIndexes;
    std::vector<float> centroidsData;
    std::vector<float> orthogonalMatrix;
    std::unique_ptr<::ascend::IndexIVFFlat> assignIndex; // 复用ivfflat一阶段检索能力加速add过程和npu聚类

private:
    AscendIndexIVFRabitQConfig ivfrabitqConfig;
    std::unordered_map<int, AscendIVFAddInfo> assignCounts;
};
} // ascend
} // faiss
#endif