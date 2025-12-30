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


#ifndef ASCEND_INDEX_IVFSQ_INCLUDED
#define ASCEND_INDEX_IVFSQ_INCLUDED

#include <faiss/IndexScalarQuantizer.h>
#include "ascend/AscendIndexIVF.h"

struct fp16;

namespace faiss {
namespace ascend {
const int64_t IVFSQ_DEFAULT_TEMP_MEM = 0x18000000;

struct AscendIndexIVFSQConfig : public AscendIndexIVFConfig {
    inline AscendIndexIVFSQConfig() : AscendIndexIVFConfig({ 0 }, IVFSQ_DEFAULT_TEMP_MEM)
    {
        SetDefaultIVFSQConfig();
    }

    inline AscendIndexIVFSQConfig(std::initializer_list<int> devices, int64_t resourceSize = IVFSQ_DEFAULT_TEMP_MEM)
        : AscendIndexIVFConfig(devices, resourceSize)
    {
        SetDefaultIVFSQConfig();
    }

    inline AscendIndexIVFSQConfig(std::vector<int> devices, int64_t resourceSize = IVFSQ_DEFAULT_TEMP_MEM)
        : AscendIndexIVFConfig(devices, resourceSize)
    {
        SetDefaultIVFSQConfig();
    }

    inline void SetDefaultIVFSQConfig()
    {
        // increase iteration to 16 for better convergence
        // increase max_points_per_centroid to 512 for getting more data to train
        cp.niter = 16; // 16 iterator
        cp.max_points_per_centroid = 512; // 512 points per centroid
    }
};

class AscendIndexIVFSQImpl;
class AscendIndexIVFSQ : public AscendIndexIVF {
public:

    // Construct an index from CPU IndexIVFSQ
    AscendIndexIVFSQ(const faiss::IndexIVFScalarQuantizer *index,
        AscendIndexIVFSQConfig config = AscendIndexIVFSQConfig());

    // Construct an empty index
    AscendIndexIVFSQ(int dims, int nlist,
        faiss::ScalarQuantizer::QuantizerType qtype = ScalarQuantizer::QuantizerType::QT_8bit,
        faiss::MetricType metric = MetricType::METRIC_L2, bool encodeResidual = true,
        AscendIndexIVFSQConfig config = AscendIndexIVFSQConfig());

    virtual ~AscendIndexIVFSQ();

    // Initialize ourselves from the given CPU index; will overwrite
    // all data in ourselves
    void copyFrom(const faiss::IndexIVFScalarQuantizer *index);

    // Copy ourselves to the given CPU index; will overwrite all data
    // in the index instance
    void copyTo(faiss::IndexIVFScalarQuantizer *index) const;

    void train(idx_t n, const float *x) override;

    // AscendIndex object is NON-copyable
    AscendIndexIVFSQ(const AscendIndexIVFSQ&) = delete;
    AscendIndexIVFSQ& operator=(const AscendIndexIVFSQ&) = delete;

protected:
    std::shared_ptr<AscendIndexIVFSQImpl> impl_;

    AscendIndexIVFSQ(int dims, int nlist, faiss::MetricType metric, AscendIndexIVFSQConfig config);
};
} // ascend
} // faiss
#endif