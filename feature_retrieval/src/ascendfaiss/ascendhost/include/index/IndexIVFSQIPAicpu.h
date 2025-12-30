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


#ifndef ASCEND_INDEXIVFSQ_IP_AICPU_INCLUDED
#define ASCEND_INDEXIVFSQ_IP_AICPU_INCLUDED

#include "ascenddaemon/impl/IndexIVFSQ.h"

namespace ascend {
class IndexIVFSQIPAicpu : public IndexIVFSQ<float> {
public:
    IndexIVFSQIPAicpu(int numList, int dim, bool encodeResidual, int nprobes, int64_t resourceSize = -1);

    ~IndexIVFSQIPAicpu();

    APP_ERROR init() override;

protected:
    int burstLen;
    int distsLen;
    int maxesLen;
    int handleBatch;

    APP_ERROR searchImplL1(AscendTensor<float16_t, DIMS_2> &queries,
                           AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndices);

    APP_ERROR searchImplL2(AscendTensor<float16_t, DIMS_2> &queries,
                           AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndices,
                           AscendTensor<float16_t, DIMS_2> &outDists,
                           AscendTensor<int64_t, DIMS_2> &outIndices);

    APP_ERROR searchImplL2For310(AscendTensor<float16_t, DIMS_2> &queries,
                                 AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndices,
                                 AscendTensor<float16_t, DIMS_2> &outDists,
                                 AscendTensor<int64_t, DIMS_2> &outIndices);

    APP_ERROR searchImplL2For310P(AscendTensor<float16_t, DIMS_2> &queries,
                                  AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndices,
                                  AscendTensor<float16_t, DIMS_2> &outDists,
                                  AscendTensor<int64_t, DIMS_2> &outIndices);

    APP_ERROR searchImplL2For310pBatch(int maxBatch,
                                       AscendTensor<float16_t, DIMS_2> &queries,
                                       AscendTensor<int64_t, DIMS_2> &l1TopNprobeIndicesHost,
                                       AscendTensor<float16_t, DIMS_2> &outDists,
                                       AscendTensor<int64_t, DIMS_2> &outIndices);

    APP_ERROR searchImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels) override;

    struct DistOpTensor {
        inline DistOpTensor(AscendTensor<float16_t, DIMS_2> &queries,
                              AscendTensor<uint64_t, DIMS_3> &listOffset,
                              AscendTensor<uint32_t, DIMS_3> &opSize,
                              AscendTensor<float16_t, DIMS_3> &distResult,
                              AscendTensor<float16_t, DIMS_3> &maxDistResult,
                              AscendTensor<uint16_t, DIMS_3> &opFlag,
                              AscendTensor<int, DIMS_2> &segsHost)
            : queries(queries), listOffset(listOffset), opSize(opSize), distResult(distResult),
              maxDistResult(maxDistResult), opFlag(opFlag), segsHost(segsHost) {};
        
        AscendTensor<float16_t, DIMS_2> &queries;
        AscendTensor<uint64_t, DIMS_3> &listOffset;
        AscendTensor<uint32_t, DIMS_3> &opSize;
        AscendTensor<float16_t, DIMS_3> &distResult;
        AscendTensor<float16_t, DIMS_3> &maxDistResult;
        AscendTensor<uint16_t, DIMS_3> &opFlag;
        AscendTensor<int, DIMS_2> &segsHost;
    };

    struct AccumOpTensor {
        inline AccumOpTensor(AscendTensor<int64_t, DIMS_2> &offsetAddrs,
                             AscendTensor<int64_t, DIMS_2> &offsetAddrsHost,
                             AscendTensor<int64_t, DIMS_2> &opSizeAddrs,
                             AscendTensor<int64_t, DIMS_2> &opSizeAddrsHost,
                             AscendTensor<int64_t, DIMS_2> &distAddrs,
                             AscendTensor<int64_t, DIMS_2> &distAddrsHost,
                             AscendTensor<int64_t, DIMS_2> &maxAddrs,
                             AscendTensor<int64_t, DIMS_2> &maxAddrsHost,
                             AscendTensor<int64_t, DIMS_2> &flagAddrs,
                             AscendTensor<int64_t, DIMS_2> &flagAddrsHost,
                             AscendTensor<uint32_t, DIMS_2> &accumNum,
                             AscendTensor<uint32_t, DIMS_2> &accumNumHost,
                             AscendTensor<uint64_t, DIMS_2> &qOffset,
                             AscendTensor<uint64_t, DIMS_2> &qOffsetHost)
            : offsetAddrs(offsetAddrs), offsetAddrsHost(offsetAddrsHost), opSizeAddrs(opSizeAddrs),
              opSizeAddrsHost(opSizeAddrsHost), distAddrs(distAddrs), distAddrsHost(distAddrsHost),
              maxAddrs(maxAddrs), maxAddrsHost(maxAddrsHost), flagAddrs(flagAddrs), flagAddrsHost(flagAddrsHost),
              accumNum(accumNum), accumNumHost(accumNumHost), queryOffset(qOffset), queryOffsetHost(qOffsetHost) {};

        AscendTensor<int64_t, DIMS_2> &offsetAddrs;
        AscendTensor<int64_t, DIMS_2> &offsetAddrsHost;
        AscendTensor<int64_t, DIMS_2> &opSizeAddrs;
        AscendTensor<int64_t, DIMS_2> &opSizeAddrsHost;
        AscendTensor<int64_t, DIMS_2> &distAddrs;
        AscendTensor<int64_t, DIMS_2> &distAddrsHost;
        AscendTensor<int64_t, DIMS_2> &maxAddrs;
        AscendTensor<int64_t, DIMS_2> &maxAddrsHost;
        AscendTensor<int64_t, DIMS_2> &flagAddrs;
        AscendTensor<int64_t, DIMS_2> &flagAddrsHost;
        AscendTensor<uint32_t, DIMS_2> &accumNum;
        AscendTensor<uint32_t, DIMS_2> &accumNumHost;
        AscendTensor<uint64_t, DIMS_2> &queryOffset;
        AscendTensor<uint64_t, DIMS_2> &queryOffsetHost;
    };

private:
    int accumNum;

    int accumAlign;

    uint32_t actualAccumNum;

    std::vector<int> ivfsqAccumBatchs;

    std::map<int, std::unique_ptr<AscendOperator>> distIvfsqAccumOpsMap;

    uint32_t calMaxBatch() const;

    APP_ERROR resetL2TopkOp();

    APP_ERROR resetSqDistOperator() const;

    APP_ERROR resetSqDistOperatorFor310() const;

    APP_ERROR resetSqDistOperatorFor310P() const;

    void runL2TopkOp(AscendTensor<float16_t, DIMS_3> &dists,
                     AscendTensor<float16_t, DIMS_3> &vmdists,
                     AscendTensor<int64_t, DIMS_3> &ids,
                     AscendTensor<uint32_t, DIMS_3> &sizes,
                     AscendTensor<uint16_t, DIMS_3> &flags,
                     AscendTensor<int64_t, DIMS_1> &attrs,
                     AscendTensor<float16_t, DIMS_2> &outdists,
                     AscendTensor<int64_t, DIMS_2> &outlabel,
                     aclrtStream stream);

    APP_ERROR setAicpuTopkAttr(AscendTensor<int64_t, DIMS_1> &attrs, int k, int tiles, int maxScanSeg) const;

    void runSqDistOperator310P(int batch,
                              const std::vector<const AscendTensorBase *> &input,
                              const std::vector<const AscendTensorBase *> &output,
                              aclrtStream stream) const;
    
    void runSqDistOperator310(int batch,
                              const std::vector<const AscendTensorBase *> &input,
                              const std::vector<const AscendTensorBase *> &output,
                              aclrtStream stream) const;

    APP_ERROR callAccumulateDist(int tiles, int maxScanSeg, DistOpTensor &distOpTensor);
    
    APP_ERROR callSqDistanceOp(DistOpTensor &distOpTensor);

    void fillAccumNum(int actualAccumNum, int &dim0Cnt, int &dim1Cnt, AccumOpTensor &accumOpTen);

    APP_ERROR fillAccumAddr(int actualAccumNum, const DistOpTensor &distOpTen, AccumOpTensor &accumOpTen);

    APP_ERROR runAccumOp(DistOpTensor &distOpTensor, AccumOpTensor &accumOpTensor, int accumBatchs, int accumAlign);

    APP_ERROR resetIvfsqAccumDistOp310P();

    void runIvfsqAccumDistOp(std::vector<const AscendTensorBase *> &input,
                             std::vector<const AscendTensorBase *> &output);

    bool useAccumlateOp(int n, int maxScanSeg);
};
} // namespace ascend
#endif // ASCEND_INDEXIVFSQ_IP_AICPU_INCLUDED
