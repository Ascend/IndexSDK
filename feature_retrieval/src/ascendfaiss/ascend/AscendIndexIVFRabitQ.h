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
#include "ascend/AscendIndexIVF.h"

namespace faiss {
namespace ascend {

struct AscendIndexIVFRabitQConfig : public AscendIndexIVFConfig {
    inline AscendIndexIVFRabitQConfig() : AscendIndexIVFConfig({ 0 }, IVF_DEFAULT_MEM), useRandomOrthogonalMatrix(true),
                                          needRefine(false), matrixSeed(12345), refineAlpha(2) {}
    
    explicit inline AscendIndexIVFRabitQConfig(std::initializer_list<int> devices,
                                               int64_t resourceSize = IVF_DEFAULT_MEM)
        : AscendIndexIVFConfig(devices, resourceSize), useRandomOrthogonalMatrix(true),
          needRefine(false), matrixSeed(12345), refineAlpha(2) {}
    
    explicit inline AscendIndexIVFRabitQConfig(std::vector<int> devices, int64_t resourceSize = IVF_DEFAULT_MEM)
        : AscendIndexIVFConfig(devices, resourceSize), useRandomOrthogonalMatrix(true),
          needRefine(false), matrixSeed(12345), refineAlpha(2) {}

    explicit inline AscendIndexIVFRabitQConfig(std::vector<int> devices, bool useRandomOrthogonalMatrix_,
                                               bool needRefine_, int matrixSeed_, float alpha_,
                                               int64_t resourceSize = IVF_DEFAULT_MEM)
        : AscendIndexIVFConfig(devices, resourceSize), useRandomOrthogonalMatrix(useRandomOrthogonalMatrix_),
          needRefine(needRefine_), matrixSeed(matrixSeed_), refineAlpha(alpha_) {}
    bool useRandomOrthogonalMatrix;
    bool needRefine;
    int matrixSeed;
    float refineAlpha;
};

class AscendIndexIVFRabitQImpl;
class AscendIndexIVFRabitQ : public AscendIndexIVF {
public:

    AscendIndexIVFRabitQ(int dims, faiss::MetricType metric, int nlist,
                       AscendIndexIVFRabitQConfig config = AscendIndexIVFRabitQConfig());

    virtual ~AscendIndexIVFRabitQ();

    AscendIndexIVFRabitQ(const AscendIndexIVFRabitQ&) = delete;
    AscendIndexIVFRabitQ& operator=(const AscendIndexIVFRabitQ&) = delete;

    void train(idx_t n, const float *x) override;

protected:
    std::shared_ptr<AscendIndexIVFRabitQImpl> impl_;
};
}  // namespace ascend
}  // namespace faiss
#endif
