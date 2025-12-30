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


#ifndef ASCEND_INDEXIVFSQ_L2_AICPU_INCLUDED
#define ASCEND_INDEXIVFSQ_L2_AICPU_INCLUDED

#include "ascenddaemon/impl/IndexIVFSQ.h"

namespace ascend {
class IndexIVFSQL2Aicpu : public IndexIVFSQ<float> {
public:
    IndexIVFSQL2Aicpu(int numList, int dim, bool encodeResidual, int nprobes, int64_t resourceSize = -1);

    ~IndexIVFSQL2Aicpu();

    APP_ERROR init() override;

    void setNumProbes(int nprobes) override;

private:
    APP_ERROR searchImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels) override;

    APP_ERROR searchImplL1(AscendTensor<float16_t, DIMS_2> &queries,
                           AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndices);

    APP_ERROR searchImplL2(AscendTensor<float16_t, DIMS_2> &queries,
                           AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndices,
                           AscendTensor<float16_t, DIMS_2> &outDists,
                           AscendTensor<int64_t, DIMS_2> &outIndices);
    APP_ERROR searchImplL2Batch(int batch, AscendTensor<float16_t, DIMS_2> &queries,
                                AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndices,
                                AscendTensor<float16_t, DIMS_2> &outDists,
                                AscendTensor<int64_t, DIMS_2> &outIndices);
private:
    uint32_t calMaxBatch() const;
    
    APP_ERROR resetL2TopkOp();

    APP_ERROR resetResidualOp();

    void runL2TopkOp(AscendTensor<float16_t, DIMS_3> &dists,
                     AscendTensor<float16_t, DIMS_3> &vmdists,
                     AscendTensor<int64_t, DIMS_3> &ids,
                     AscendTensor<uint32_t, DIMS_4> &sizes,
                     AscendTensor<uint16_t, DIMS_4> &flags,
                     AscendTensor<int64_t, DIMS_1> &attrs,
                     AscendTensor<float16_t, DIMS_2> &outdists,
                     AscendTensor<int64_t, DIMS_2> &outlabel,
                     aclrtStream stream);

    void runResidualOp(int batch,
                       const std::vector<const AscendTensorBase *> &input,
                       const std::vector<const AscendTensorBase *> &output,
                       aclrtStream stream);

    APP_ERROR resetSqDistOperator();

    void runSqDistOperator(int batch,
                           const std::vector<const AscendTensorBase *> &input,
                           const std::vector<const AscendTensorBase *> &output,
                           aclrtStream stream);

    std::map<int, std::unique_ptr<AscendOperator>> residualOps;
    std::map<int, std::unique_ptr<AscendOperator>> l2DistOps;

private:
    bool byResidual;
    int burstLen;
    int bursts;
};
} // namespace ascend
#endif // ASCEND_INDEXIVFSQ_L2_AICPU_INCLUDED
