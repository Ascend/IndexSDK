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

#ifndef ASCEND_INDEX_IVFPQ_INCLUDED
#define ASCEND_INDEX_IVFPQ_INCLUDED

#include <faiss/Clustering.h>
#include <faiss/IndexIVFPQ.h>
#include "ascend/AscendIndexIVF.h"

namespace faiss {
namespace ascend {

struct AscendIndexIVFPQConfig : public AscendIndexIVFConfig {
    inline AscendIndexIVFPQConfig() : AscendIndexIVFConfig({0}, IVF_DEFAULT_MEM) {}

    explicit inline AscendIndexIVFPQConfig(std::initializer_list<int> devices, int64_t resourceSize = IVF_DEFAULT_MEM)
        : AscendIndexIVFConfig(devices, resourceSize) {}

    explicit inline AscendIndexIVFPQConfig(std::vector<int> devices, int64_t resourceSize = IVF_DEFAULT_MEM)
        : AscendIndexIVFConfig(devices, resourceSize) {}
};

class AscendIndexIVFPQImpl;
class AscendIndexIVFPQ : public AscendIndexIVF {
public:

    AscendIndexIVFPQ(int dims, faiss::MetricType metric, int nlist, int msubs, int nbits,
                     AscendIndexIVFPQConfig config = AscendIndexIVFPQConfig());

    // Initialize ourselves from the given CPU index; will overwrite
    void copyFrom(const faiss::IndexIVFPQ* index);

    // Copy ourselves to the given CPU index; will overwrite all data
    void copyTo(faiss::IndexIVFPQ* index) const;

    virtual ~AscendIndexIVFPQ();

    AscendIndexIVFPQ(const AscendIndexIVFPQ&) = delete;
    AscendIndexIVFPQ& operator=(const AscendIndexIVFPQ&) = delete;

    void train(idx_t n, const float* x) override;

    void remove_ids(size_t n, const idx_t* ids);

    std::vector<idx_t> update(idx_t n, const float* x, const idx_t* ids);

protected:
    std::shared_ptr<AscendIndexIVFPQImpl> impl_;
};
}  // namespace ascend
}  // namespace faiss
#endif
