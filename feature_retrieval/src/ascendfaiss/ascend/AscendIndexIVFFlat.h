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


#ifndef ASCEND_INDEX_IVFFLAT_INCLUDED
#define ASCEND_INDEX_IVFFLAT_INCLUDED

#include <faiss/IndexIVFFlat.h>
#include <faiss/Clustering.h>
#include "ascend/AscendIndexIVF.h"

namespace faiss {
namespace ascend {

struct AscendIndexIVFFlatConfig : public AscendIndexIVFConfig {
    inline AscendIndexIVFFlatConfig() : AscendIndexIVFConfig({ 0 }, IVF_DEFAULT_MEM) {}
    
    explicit inline AscendIndexIVFFlatConfig(std::initializer_list<int> devices, int64_t resourceSize = IVF_DEFAULT_MEM)
        : AscendIndexIVFConfig(devices, resourceSize) {}
    
    explicit inline AscendIndexIVFFlatConfig(std::vector<int> devices, int64_t resourceSize = IVF_DEFAULT_MEM)
        : AscendIndexIVFConfig(devices, resourceSize) {}
};

class AscendIndexIVFFlatImpl;
class AscendIndexIVFFlat : public AscendIndexIVF {
public:

    AscendIndexIVFFlat(int dims, faiss::MetricType metric, int nlist,
                       AscendIndexIVFFlatConfig config = AscendIndexIVFFlatConfig());

    virtual ~AscendIndexIVFFlat();

    // Initialize ourselves from the given CPU index; will overwrite
    void copyFrom(const faiss::IndexIVFFlat *index);

    // Copy ourselves to the given CPU index; will overwrite all data
    void copyTo(faiss::IndexIVFFlat *index) const;

    AscendIndexIVFFlat(const AscendIndexIVFFlat&) = delete;
    AscendIndexIVFFlat& operator=(const AscendIndexIVFFlat&) = delete;

    void train(idx_t n, const float *x) override;

protected:
    std::shared_ptr<AscendIndexIVFFlatImpl> impl_;
};
}  // namespace ascend
}  // namespace faiss
#endif
