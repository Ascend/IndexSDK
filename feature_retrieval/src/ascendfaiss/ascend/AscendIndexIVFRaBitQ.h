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

namespace faiss {
namespace ascend {

struct AscendIndexIVFRaBitQConfig : public AscendIndexIVFConfig {
    inline AscendIndexIVFRaBitQConfig() : AscendIndexIVFConfig({ 0 }, IVF_DEFAULT_MEM), useRandomOrthogonalMatrix(true),
                                          needRefine(false), matrixSeed(12345), refineAlpha(2) {}
    
    explicit inline AscendIndexIVFRaBitQConfig(std::initializer_list<int> devices,
                                               int64_t resourceSize = IVF_DEFAULT_MEM)
        : AscendIndexIVFConfig(devices, resourceSize), useRandomOrthogonalMatrix(true),
          needRefine(false), matrixSeed(12345), refineAlpha(2) {}
    
    explicit inline AscendIndexIVFRaBitQConfig(std::vector<int> devices, int64_t resourceSize = IVF_DEFAULT_MEM)
        : AscendIndexIVFConfig(devices, resourceSize), useRandomOrthogonalMatrix(true),
          needRefine(false), matrixSeed(12345), refineAlpha(2) {}

    explicit inline AscendIndexIVFRaBitQConfig(std::vector<int> devices, bool useRandomOrthogonalMatrix_,
                                               bool needRefine_, int matrixSeed_, float alpha_,
                                               int64_t resourceSize = IVF_DEFAULT_MEM)
        : AscendIndexIVFConfig(devices, resourceSize), useRandomOrthogonalMatrix(useRandomOrthogonalMatrix_),
          needRefine(needRefine_), matrixSeed(matrixSeed_), refineAlpha(alpha_) {}
    bool useRandomOrthogonalMatrix;
    bool needRefine;
    int matrixSeed;
    float refineAlpha;
};

class AscendIndexIVFRaBitQImpl;
class AscendIndexIVFRaBitQ : public AscendIndexIVF {
public:

    AscendIndexIVFRaBitQ(int dims, faiss::MetricType metric, int nlist,
                       AscendIndexIVFRaBitQConfig config = AscendIndexIVFRaBitQConfig());

    virtual ~AscendIndexIVFRaBitQ();

    AscendIndexIVFRaBitQ(const AscendIndexIVFRaBitQ&) = delete;
    AscendIndexIVFRaBitQ& operator=(const AscendIndexIVFRaBitQ&) = delete;

    void train(idx_t n, const float *x) override;
    
    // Copy what we need from a CPU IndexIVFRaBitQ
    // This copies all IVFRaBitQ-specific data for complete state transfer
    void copyFrom(const faiss::IndexIVFRaBitQ *index);

    // Copy what we have to a CPU IndexIVFRaBitQ
    // This copies all IVFRaBitQ-specific data for complete state transfer
    void copyTo(faiss::IndexIVFRaBitQ *index) const;

    void remove_ids(size_t n, const idx_t* ids);

    std::vector<idx_t> update(idx_t n, const float* x, const idx_t* ids);

protected:
    std::shared_ptr<AscendIndexIVFRaBitQImpl> impl_;
};
}  // namespace ascend
}  // namespace faiss
#endif
