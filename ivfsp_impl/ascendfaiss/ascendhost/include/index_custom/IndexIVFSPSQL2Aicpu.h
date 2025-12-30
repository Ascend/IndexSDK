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


#ifndef ASCENDHOST_INDEXIVFSPSQ_L2_AICPU_INCLUDED
#define ASCENDHOST_INDEXIVFSPSQ_L2_AICPU_INCLUDED


#include <ascenddaemon/impl_custom/IndexIVFSPSQL2.h>
#include <ascenddaemon/utils/AscendTensor.h>

namespace ascendSearch {
class IndexIVFSPSQL2Aicpu : public IndexIVFSPSQL2 {
public:
    IndexIVFSPSQL2Aicpu(int dim, int dim2, int k, int nlist, bool encodeResidual,
                        int nprobes, int searchListSize, int handleBatch, bool filterable,
                        int64_t resourceSize = -1);

    ~IndexIVFSPSQL2Aicpu();

    APP_ERROR init() override;

    APP_ERROR getCodeWord(int n, float *feature, float16_t *codeWord,
                          idx_t *labels);

    void setNumProbes(int nprobes) override;

protected:
    APP_ERROR searchImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels) override;

    APP_ERROR searchImpl(std::vector<Index *> indexes, int n, int batchSize, const float16_t *x,
        int k, float16_t *distances, idx_t *labels) override;

    APP_ERROR searchFilterImpl(int n, const float16_t *x, int k, float16_t *distances,
        idx_t *labels, uint32_t filterSize, uint32_t* filters) override;

    APP_ERROR searchFilterImpl(int n, const float16_t *x, int k, float16_t *distances,
        idx_t *labels, float16_t *l1distances, uint32_t filterSize, uint32_t* filters);

    APP_ERROR searchFilterImpl(std::vector<Index *> indexes, int n, int batchSize, const float16_t *x, int k,
        float16_t *distances, idx_t *labels, uint32_t filterSize, uint32_t* filters);
    APP_ERROR searchFilterImpl(std::vector<Index *> indexes, int n, int batchSize, const float16_t *x, int k,
        float16_t *distances, idx_t *labels, uint32_t filterSize, uint32_t** filters);

private:
    APP_ERROR addBatched(int n, float *feature, float16_t *codeWord, idx_t *labels);

    APP_ERROR addImplL1(AscendTensor<float16_t, DIMS_2> &queries,
                    AscendTensor<int64_t, DIMS_2> &l1TopKIndices);

    APP_ERROR searchImplL1(AscendTensor<float16_t, DIMS_2> &queries,
                           AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost);

    APP_ERROR searchImplL2(AscendTensor<float16_t, DIMS_2> &queries,
                           AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndices,
                           AscendTensor<float16_t, DIMS_2> &outDists,
                           AscendTensor<int64_t, DIMS_2> &outIndices);

    APP_ERROR searchImplL2(std::vector<Index *> indexes,
                            AscendTensor<float16_t, DIMS_2> &queries,
                           AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndices,
                           AscendTensor<float16_t, DIMS_3> &outDists,
                           AscendTensor<int64_t, DIMS_3> &outIndices);

    APP_ERROR searchFilterImplL2(AscendTensor<float16_t, DIMS_2> &queries,
                                 uint32_t filterSize, uint32_t* filters,
                                 AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndices,
                                 AscendTensor<float16_t, DIMS_2> &outDists,
                                 AscendTensor<int64_t, DIMS_2> &outIndices);

    APP_ERROR searchFilterImplL2(std::vector<Index *> indexes,
                                  AscendTensor<float16_t, DIMS_2> &queries,
                                  AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndices,
                                  AscendTensor<float16_t, DIMS_3> &outDists,
                                  AscendTensor<int64_t, DIMS_3> &outIndices,
                                  uint32_t filterSize, uint32_t* filters);

    APP_ERROR searchFilterImplL2(std::vector<Index *> indexes,
                                  AscendTensor<float16_t, DIMS_2> &queries,
                                  AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndices,
                                  AscendTensor<float16_t, DIMS_3> &outDists,
                                  AscendTensor<int64_t, DIMS_3> &outIndices,
                                  uint32_t filterSize, uint32_t** filters);

    APP_ERROR computeMask(int n,
                          int seg,
                          uint32_t* filters,
                          AscendTensor<uint8_t, DIMS_1>& masks,
                          AscendTensor<int, DIMS_1> &listId);

    void runL1TopkOp1(AscendTensor<float16_t, DIMS_2> &dists,
                                  AscendTensor<uint32_t, DIMS_2> &sizes,
                                  AscendTensor<uint16_t, DIMS_2> &flags,
                                  AscendTensor<int64_t, DIMS_1> &attrs,
                                  AscendTensor<float16_t, DIMS_2> &outdists,
                                  AscendTensor<int64_t, DIMS_2> &outlabel,
                                  aclrtStream stream);

    void runL2TopkOp(AscendTensor<float16_t, DIMS_3> &dists,
                  AscendTensor<float16_t, DIMS_3> &vmdists,
                  AscendTensor<idx_t, DIMS_3> &ids,
                  AscendTensor<uint32_t, DIMS_3> &sizes,
                  AscendTensor<uint16_t, DIMS_3> &flags,
                  AscendTensor<int64_t, DIMS_1> &attrs,
                  AscendTensor<float16_t, DIMS_2> &outdists,
                  AscendTensor<int64_t, DIMS_2> &outlabel, aclrtStream stream);

    void runL2TopkMultiSearchOp(AscendTensor<float16_t, DIMS_3> &dists,
                                    AscendTensor<float16_t, DIMS_3> &vmdists,
                                    AscendTensor<idx_t, DIMS_4> &ids,
                                    AscendTensor<uint32_t, DIMS_4> &sizes,
                                    AscendTensor<uint16_t, DIMS_4> &flags,
                                    AscendTensor<int64_t, DIMS_1> &attrs,
                                    AscendTensor<float16_t, DIMS_3> &outdists,
                                    AscendTensor<int64_t, DIMS_3> &outlabel, aclrtStream stream);

    void runL2TopkMultiSearchV2Op(AscendTensor<float16_t, DIMS_3> &dists,
                                    AscendTensor<float16_t, DIMS_3> &vmdists,
                                    AscendTensor<idx_t, DIMS_4> &ids,
                                    AscendTensor<uint32_t, DIMS_4> &sizes,
                                    AscendTensor<uint16_t, DIMS_4> &flags,
                                    AscendTensor<int64_t, DIMS_1> &attrs,
                                    AscendTensor<float16_t, DIMS_3> &outdists,
                                    AscendTensor<int64_t, DIMS_3> &outlabel, aclrtStream stream);

    void runDistCompute(AscendTensor<float16_t, DIMS_2> &queryVecs,
        AscendTensor<float16_t, DIMS_4> &shapedData,
        AscendTensor<uint32_t, DIMS_2> &size,
        AscendTensor<float16_t, DIMS_2> &outDistances,
        AscendTensor<float16_t, DIMS_2> &maxDistances,
        AscendTensor<uint16_t, DIMS_2> &flag, aclrtStream stream);

    APP_ERROR searchPaged(size_t pageId, size_t pageNum, AscendTensor<float16_t, DIMS_2> &queries,
        AscendTensor<float16_t, DIMS_2> &maxDistances, AscendTensor<int64_t, DIMS_2> &maxIndices);

    APP_ERROR resetDistCompOp(int numLists);

    APP_ERROR initL1TopkAttrs();

    APP_ERROR resetL1TopkOp();

    APP_ERROR initL2TopkAttrs();

    APP_ERROR resetL2TopkOp();

    APP_ERROR resetL2TopkMultiSearchOp();

    APP_ERROR resetL2TopkMultiSearchOpV2();

    // aicpu op for topk computation
    std::map<int, std::unique_ptr<::ascendSearch::AscendOperator>> l2TopkOps;
    std::map<int, std::unique_ptr<::ascendSearch::AscendOperator>> l2TopkMultiSearchOps;
    std::map<int, std::unique_ptr<::ascendSearch::AscendOperator>> l2TopkMultiSearchOpsV2;
    std::map<int, std::unique_ptr<::ascendSearch::AscendOperator>> l1TopkOps1;

    AscendTensor<int64_t, DIMS_1> l1Attrs;
    AscendTensor<int64_t, DIMS_1> l2Attrs;
};
}  // namespace ascendSearch

#endif  // ASCENDHOST_INDEXIVFSPSQ_L2_AICPU_INCLUDED
