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


#ifndef ASCEND_INDEX_IVFSQFUZZY_IMPL_INCLUDED
#define ASCEND_INDEX_IVFSQFUZZY_IMPL_INCLUDED

#include "faiss/IndexScalarQuantizer.h"
#include "ascend/impl/AscendIndexIVFSQImpl.h"

struct fp16;

namespace faiss {
namespace ascend {

enum FuzzyTypes {
    TYPE_FUZZY = 1,
    TYPE_C = 2,
    TYPE_FAST = 3,
    TYPE_T = 4
};

class AscendIndexFlatAT;

class AscendIndexIVFSQFuzzyImpl : public AscendIndexIVFSQImpl {
public:
    virtual ~AscendIndexIVFSQFuzzyImpl();

    void copyTo(faiss::IndexIVFScalarQuantizer *index);

    void copyFrom(const faiss::IndexIVFScalarQuantizer *index);

    void reset();

    void train(idx_t n, const float *x);

    void setFuzzyK(int value);
    int getFuzzyK() const;

    void setThreshold(float value);
    float getThreshold() const;

    // AscendIndex object is NON-copyable
    AscendIndexIVFSQFuzzyImpl(const AscendIndexIVFSQFuzzyImpl &) = delete;
    AscendIndexIVFSQFuzzyImpl &operator = (const AscendIndexIVFSQFuzzyImpl &) = delete;

protected:
    // Number of Clustering Center in Add stage
    size_t fuzzyK = 3;

    // threshold to choose fuzzy center
    float threshold = 1;

    // total num in index after fuzzy Add
    size_t fuzzyTotal = 0;

    // declare index type, 1 for ivfsqfuzzy, 2 for ivfsqc, 3 for ivfast, 4 for ivfsqt
    int fuzzyType = TYPE_FUZZY;

    // Construct an empty index when AscendIndexIVFSQFuzzy is parent class for other custom classes
    AscendIndexIVFSQFuzzyImpl(AscendIndexIVFSQ *intf, int dims, int nlist, bool dummy,
        faiss::ScalarQuantizer::QuantizerType qtype = ScalarQuantizer::QuantizerType::QT_8bit,
        faiss::MetricType metric = MetricType::METRIC_INNER_PRODUCT,
        AscendIndexIVFSQConfig config = AscendIndexIVFSQConfig());

    void updateDeviceSQTrainedValue();

    // append fuzzy params to sq.trained to clone index
    virtual void appendTrained();

    // split fuzzy params from sq.trained to load from a index
    virtual void splitTrained();

    // get fuzzyTotal when load from a index
    void initFuzzyTotal(const faiss::IndexIVFScalarQuantizer *index);

    virtual void getFuzzyList(size_t n, const float *x, std::vector<idx_t> &resultSearchId) = 0;

private:
    AscendIndexIVFSQConfig ivfsqFuzzyConfig;
};
} // ascend
} // faiss
#endif