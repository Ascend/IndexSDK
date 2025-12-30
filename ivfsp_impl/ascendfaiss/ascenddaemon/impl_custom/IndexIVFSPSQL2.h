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


#ifndef ASCEND_INDEX_IVF_SP_SQ_L2_INCLUDED
#define ASCEND_INDEX_IVF_SP_SQ_L2_INCLUDED

#include <vector>

#include <ascenddaemon/impl_custom/IndexIVFSPSQ.h>
#include <ascenddaemon/utils/TopkOp.h>
#include <ascenddaemon/utils/AscendOperator.h>
#include <ascenddaemon/utils/DeviceVector.h>
#include <ascenddaemon/utils/DeviceVectorInl.h>
#include <ascenddaemon/utils/AscendTensor.h>
#include <common/threadpool/AscendThreadPool.h>

namespace ascendSearch {
class IndexIVFSPSQL2 : public IndexIVFSPSQ {
public:
    IndexIVFSPSQL2(int dim, int dim2, int k, int nlist, bool encodeResidual,
                   int nprobes, int searchListSize, int handleBatch, bool filterable,
                   int64_t resourceSize = -1);

    ~IndexIVFSPSQL2();

    APP_ERROR init() override;

    APP_ERROR reset() override;

    inline const std::vector<std::unique_ptr<DeviceVector<float>>>& getNormBase() const
    {
        return normBase;
    }

    int getCodeBookSize() override;

    virtual APP_ERROR getCodeWord(int n, float16_t *feature, float16_t *codeWord, idx_t *labels);

protected:
    void runL1DistOp(AscendTensor<float16_t, DIMS_2>& queryVecs,
                     AscendTensor<float16_t, DIMS_4>& shapedData,
                     AscendTensor<float16_t, DIMS_1>& norms,
                     AscendTensor<float16_t, DIMS_2>& outDists,
                     AscendTensor<uint16_t, DIMS_2>& flag,
                     aclrtStream stream);

    void runSqDistOperator(AscendTensor<float16_t, DIMS_2> &queries,
                           AscendTensor<float16_t, DIMS_4> &book,
                           AscendTensor<uint8_t, DIMS_1> &base,
                           AscendTensor<uint8_t, DIMS_2> &mask,
                           AscendTensor<float, DIMS_1> &norm,
                           AscendTensor<float16_t, DIMS_2> &dm,
                           AscendTensor<uint64_t, DIMS_1> &baseOffset,
                           AscendTensor<uint64_t, DIMS_1> &codebookOffset,
                           AscendTensor<uint64_t, DIMS_1> &normOffset,
                           AscendTensor<uint32_t, DIMS_1> &size,
                           AscendTensor<float16_t, DIMS_2> &result,
                           AscendTensor<float16_t, DIMS_2> &maxResult,
                           AscendTensor<uint16_t, DIMS_2> &flag);

    void runSqDistOperator(AscendTensor<float16_t, DIMS_2> &queries,
                           AscendTensor<float16_t, DIMS_4> &book,
                           AscendTensor<uint8_t, DIMS_1> &base,
                           AscendTensor<float, DIMS_1> &norm,
                           AscendTensor<float16_t, DIMS_1> &diff,
                           AscendTensor<float16_t, DIMS_1> &min,
                           AscendTensor<uint64_t, DIMS_1> &baseOffset,
                           AscendTensor<uint64_t, DIMS_1> &codebookOffset,
                           AscendTensor<uint64_t, DIMS_1> &normOffset,
                           AscendTensor<uint32_t, DIMS_1> &size,
                           AscendTensor<float16_t, DIMS_2> &result,
                           AscendTensor<float16_t, DIMS_2> &maxResult,
                           AscendTensor<uint16_t, DIMS_2> &flag);

protected:
    std::vector<std::unique_ptr<DeviceVector<float>>> normBase;
    std::vector<std::unique_ptr<DeviceVector<float16_t>>> codeBook;
    TopkOp<std::less<float16_t>, std::less_equal<float16_t>, float16_t, false> topKMaxOp;
    TopkOp<std::greater<float16_t>, std::greater_equal<float16_t>, float16_t> topKMinOp;
    std::unique_ptr<AscendOperator> l2FilterDistOps;

    int pageSize;
    int dim2;
    int nCodeBook;
    bool byResidual;
    int codebookNum;

    int distsLen;
    int handleBatch;
    int burstLen;
    int maxesLen;
    int bursts;

private:
    virtual APP_ERROR searchImplL1(AscendTensor<float16_t, DIMS_2> &queries,
                                   AscendTensor<float16_t, DIMS_2> &distances,
                                   aclrtStream stream);

    virtual APP_ERROR searchFilterImplL2(AscendTensor<float16_t, DIMS_2> &queries,
        uint32_t filterSize, uint32_t* filters,
        AscendTensor<float16_t, DIMS_2> &l1Dists,
        AscendTensor<float16_t, DIMS_2> &outDists,
        AscendTensor<idx_t, DIMS_2> &outIndices);

    APP_ERROR resetL1DistOp(int numLists);

    APP_ERROR resetSqDistOperatorFor310P();

    APP_ERROR resetFilterSqDistOperator();
    
    APP_ERROR searchImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels) override;
    APP_ERROR searchImpl(std::vector<Index *> indexes, int n, int batchSize, const float16_t *x, int k,
        float16_t *distances, idx_t *labels) override;

    APP_ERROR searchImpl(AscendTensor<float16_t, DIMS_2> &queries, int k,
                         AscendTensor<float16_t, DIMS_2> &outDistance,
                         AscendTensor<idx_t, DIMS_2> &outIndices);

    APP_ERROR searchImplL2(AscendTensor<float16_t, DIMS_2> &queries,
                           AscendTensor<float16_t, DIMS_2> &l1Distances,
                           AscendTensor<float16_t, DIMS_2> &outDistances,
                           AscendTensor<idx_t, DIMS_2> &outIndices);

    virtual APP_ERROR searchFilterImpl(int n, const float16_t *x, int k, float16_t *distances,
        idx_t *labels, uint32_t filterSize, uint32_t* filters) override;

    virtual APP_ERROR searchFilterImpl(std::vector<Index *> indexes, int n, int batchSize, const float16_t *x, int k,
        float16_t *distances, idx_t *labels, uint32_t filterSize, uint32_t* filters) override;

    void moveVectorForward(idx_t srcIdx, idx_t dstIdx) override;
    void releaseUnusageSpace(int oldTotal, int remove) override;

    size_t calcNormBaseSize(idx_t totalNum);
};
}  // namespace ascendSearch

#endif  // ASCEND_INDEXSP_L2_INCLUDED
