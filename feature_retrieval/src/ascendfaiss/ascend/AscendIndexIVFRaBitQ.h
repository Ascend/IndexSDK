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

#ifndef ASCEND_INDEX_IVFRABITQ_INCLUDED
#define ASCEND_INDEX_IVFRABITQ_INCLUDED

#include <faiss/Clustering.h>
#include <faiss/IndexIVFRaBitQ.h>

#include "ascend/AscendIndexIVF.h"

namespace faiss
{
namespace ascend
{

struct AscendIndexIVFRaBitQConfig : public AscendIndexIVFConfig
{
    inline AscendIndexIVFRaBitQConfig()
        : AscendIndexIVFConfig({0}, IVF_DEFAULT_MEM),
          useRandomOrthogonalMatrix(true),
          needRefine(false),
          matrixSeed(12345),
          refineAlpha(2)
    {
    }

    inline AscendIndexIVFRaBitQConfig(std::initializer_list<int> devices, int64_t resourceSize = IVF_DEFAULT_MEM)
        : AscendIndexIVFConfig(devices, resourceSize),
          useRandomOrthogonalMatrix(true),
          needRefine(false),
          matrixSeed(12345),
          refineAlpha(2)
    {
    }

    explicit inline AscendIndexIVFRaBitQConfig(std::vector<int> devices, int64_t resourceSize = IVF_DEFAULT_MEM)
        : AscendIndexIVFConfig(devices, resourceSize),
          useRandomOrthogonalMatrix(true),
          needRefine(false),
          matrixSeed(12345),
          refineAlpha(2)
    {
    }

    explicit inline AscendIndexIVFRaBitQConfig(std::vector<int> devices, bool useRandomOrthogonalMatrix_,
                                               bool needRefine_, int matrixSeed_, float alpha_,
                                               int64_t resourceSize = IVF_DEFAULT_MEM)
        : AscendIndexIVFConfig(devices, resourceSize),
          useRandomOrthogonalMatrix(useRandomOrthogonalMatrix_),
          needRefine(needRefine_),
          matrixSeed(matrixSeed_),
          refineAlpha(alpha_)
    {
    }
    bool useRandomOrthogonalMatrix;
    bool needRefine;
    int matrixSeed;
    float refineAlpha;

    // RaBitQ bits per dimension. 1 keeps the original format; 2 and 4 enable
    // the A3/L2 multi-bit proof-of-concept bit-plane format.
    int quantBits = 1;

    // Coarse k-means iterations. <=0 keeps AscendIndexIVFConfig::cp.niter (default 10).
    int niter = 0;

    // NPU k-means layout when useKmeansPP is true and more than one device is set:
    // Replica = TrainFp32 on device 0 (full sample);
    // Slice = DistributedTrainFp32 (shard the sample across devices).
    enum class TrainMode
    {
        Replica = 0,
        Slice = 1
    };
    TrainMode trainMode = TrainMode::Replica;
};

class AscendIndexIVFRaBitQImpl;
class AscendIndexIVFRaBitQ : public AscendIndexIVF
{
   public:
    AscendIndexIVFRaBitQ(int dims, faiss::MetricType metric, int nlist,
                         AscendIndexIVFRaBitQConfig config = AscendIndexIVFRaBitQConfig());

    ~AscendIndexIVFRaBitQ() override;

    AscendIndexIVFRaBitQ(const AscendIndexIVFRaBitQ&) = delete;
    AscendIndexIVFRaBitQ& operator=(const AscendIndexIVFRaBitQ&) = delete;
    AscendIndexIVFRaBitQ(AscendIndexIVFRaBitQ&&) = delete;
    AscendIndexIVFRaBitQ& operator=(AscendIndexIVFRaBitQ&&) = delete;

    void train(idx_t n, const float* x) override;

    // Copy all IVFRaBitQ state from a CPU carrier. For quantBits=2/4 the CPU
    // object stores the extended codes for persistence; CPU search is not supported.
    void copyFrom(const faiss::IndexIVFRaBitQ* index);

    // Copy all IVFRaBitQ state to a CPU carrier. The code_size field records
    // whether each inverted-list entry contains a 1-bit, 2-bit, or 4-bit code.
    void copyTo(faiss::IndexIVFRaBitQ* index) const;

    // Delete/update are rejected while copied or attached refine sources exist.
    void remove_ids(size_t n, const idx_t* ids);

    std::vector<idx_t> update(idx_t n, const float* x, const idx_t* ids);

    // NOLINTNEXTLINE(google-default-arguments) -- match FAISS Index::search
    void search(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels,
                const SearchParameters* params = nullptr) const override;

    void invalidateFilterCache() const;

    // Refine uses ID as the source row: existing IDs must cover [0, ntotal).
    // Copies n == ntotal rows. Further adds must use IDs ntotal, ntotal+1, ...
    void loadRefineSources(const float* x, idx_t n);

    // Zero-copy attachment with the same ID/row contract. Add/delete/update are rejected until reset/copyFrom.
    // Caller owns the buffer and must keep all n rows valid until reset/copyFrom or index destruction.
    void attachRefineSources(const float* x, idx_t n);

   protected:
    std::shared_ptr<AscendIndexIVFRaBitQImpl> impl_;
};
}  // namespace ascend
}  // namespace faiss
#endif
