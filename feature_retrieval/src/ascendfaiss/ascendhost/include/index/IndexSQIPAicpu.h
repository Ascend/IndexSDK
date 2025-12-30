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


#ifndef ASCEND_INDEXSQ_IP_AICPU_INCLUDED
#define ASCEND_INDEXSQ_IP_AICPU_INCLUDED

#include "ascenddaemon/impl/IndexSQ.h"

namespace ascend {
class IndexSQIPAicpu : public IndexSQ {
public:
    IndexSQIPAicpu(int dim, bool filterable, int64_t resource = -1, int blockSize = SQ_DEFAULT_DIST_COMPUTE_BATCH);

    ~IndexSQIPAicpu();

    APP_ERROR init() override;

    APP_ERROR addVectors(size_t numVecs, const uint8_t *data, const float *preCompute);

private:
    APP_ERROR searchImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels) override;

    APP_ERROR searchFilterImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels, uint8_t *masks,
        uint32_t maskLen) override;

    APP_ERROR searchImpl(std::vector<Index *> indexes, int n, int batchSize, const float16_t *x, int k,
        float16_t *distances, idx_t *labels) override;

    APP_ERROR searchFilterImpl(std::vector<Index *> indexes, int n, int batchSize, const float16_t *x, int k,
        float16_t *distances, idx_t *labels, AscendTensor<uint8_t, DIMS_2, int64_t> &maskData,
        AscendTensor<int, DIMS_1> &maskOffset) override;

    APP_ERROR searchPaged(size_t pageId, size_t pageNum, const float16_t *x,
        AscendTensor<float16_t, DIMS_2> &maxDistances, AscendTensor<int64_t, DIMS_2> &maxIndices);

    APP_ERROR initResult(AscendTensor<float16_t, DIMS_3, size_t> &distances,
        AscendTensor<idx_t, DIMS_3, size_t> &indices) const;

    void runSqDistOperator(int batch,
                           const std::vector<const AscendTensorBase *> &input,
                           const std::vector<const AscendTensorBase *> &output,
                           aclrtStream stream) const;

    void runSqDistMaskOperator(int batch,
                               const std::vector<const AscendTensorBase *> &input,
                               const std::vector<const AscendTensorBase *> &output,
                               aclrtStream stream) const;
};
} // namespace ascend
#endif // ASCEND_INDEXSQ_IP_AICPU_INCLUDED