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


#ifndef ASCEND_INDEX_IVFSQT_INCLUDED
#define ASCEND_INDEX_IVFSQT_INCLUDED

#include <faiss/IndexScalarQuantizer.h>
#include "ascend/AscendIndexIVFSQ.h"

struct fp16;

namespace faiss {
namespace ascend {
const int64_t IVFSQT_DEFAULT_TEMP_MEM = 0x18000000;

struct AscendIndexIVFSQTConfig : public AscendIndexIVFSQConfig {
    inline AscendIndexIVFSQTConfig() : AscendIndexIVFSQConfig({ 0 }, IVFSQT_DEFAULT_TEMP_MEM)
    {
        SetDefaultIVFSQConfig();
    }

    inline AscendIndexIVFSQTConfig(std::initializer_list<int> devices, int64_t resourceSize = IVFSQT_DEFAULT_TEMP_MEM)
        : AscendIndexIVFSQConfig(devices, resourceSize)
    {
        SetDefaultIVFSQConfig();
    }

    inline AscendIndexIVFSQTConfig(std::vector<int> devices, int64_t resourceSize = IVFSQT_DEFAULT_TEMP_MEM)
        : AscendIndexIVFSQConfig(devices, resourceSize)
    {
        SetDefaultIVFSQConfig();
    }

    inline void SetDefaultIVFSQConfig()
    {
        // increase iteration to 16 for better convergence
        // increase max_points_per_centroid to 512 for getting more data to train
        cp.niter = 16;                    // 16 iterator
        cp.max_points_per_centroid = 512; // 512 points per centroid
    }
};

class AscendIndexIVFSQTImpl;

class AscendIndexIVFSQT : public AscendIndexIVFSQ {
public:
    // Construct an index from CPU IndexIVFSQ
    AscendIndexIVFSQT(const faiss::IndexIVFScalarQuantizer *index,
        AscendIndexIVFSQTConfig config = AscendIndexIVFSQTConfig());

    // Construct an empty index
    AscendIndexIVFSQT(int dimIn, int dimOut, int nlist,
        faiss::ScalarQuantizer::QuantizerType qtype = ScalarQuantizer::QuantizerType::QT_8bit,
        faiss::MetricType metric = MetricType::METRIC_INNER_PRODUCT,
        AscendIndexIVFSQTConfig config = AscendIndexIVFSQTConfig());

    virtual ~AscendIndexIVFSQT();

    // AscendIndex object is NON-copyable
    AscendIndexIVFSQT(const AscendIndexIVFSQT &) = delete;

    AscendIndexIVFSQT &operator = (const AscendIndexIVFSQT &) = delete;

    void copyTo(faiss::IndexIVFScalarQuantizer *index);

    void copyFrom(const faiss::IndexIVFScalarQuantizer *index);

    void train(idx_t n, const float *x) override;

    void setFuzzyK(int value);

    int getFuzzyK() const;

    void setThreshold(float value);

    float getThreshold() const;

    void fineTune(size_t n, const float *x);

    void update(bool cleanData = true);

    void updateTParams(int l2Probe, int l3SegmentNum);

    void setLowerBound(int lowerBound);

    int getLowerBound() const;

    void setMergeThres(int mergeThres);

    int getMergeThres() const;

    void setMemoryLimit(float memoryLimit);

    void setAddTotal(size_t addTotal);

    void setPreciseMemControl(bool preciseMemControl);

    void reset() override;

    void setNumProbes(int nprobes) override;

    size_t remove_ids(const faiss::IDSelector &sel) override;

    float getQMin() const;

    float getQMax() const;

    uint32_t getListLength(int listId) const override;

    void getListCodesAndIds(int listId, std::vector<uint8_t>& codes, std::vector<ascend_idx_t>& ids) const override;

    void setSearchParams(int nprobe, int l2Probe, int l3SegmentNum);

    void setSortMode(int mode);

    void setUseCpuUpdate(int numThreads);

protected:
    std::shared_ptr<AscendIndexIVFSQTImpl> impl_;
};
} // ascend
} // faiss
#endif
