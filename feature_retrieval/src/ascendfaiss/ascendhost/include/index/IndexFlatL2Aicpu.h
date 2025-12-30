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


#ifndef ASCENDHOST_INDEXFLAT_L2_AICPU_INCLUDED
#define ASCENDHOST_INDEXFLAT_L2_AICPU_INCLUDED

#include "ascenddaemon/impl/IndexFlat.h"
#include "ascenddaemon/utils/AscendTensor.h"

namespace ascend {
struct topkMultisearchParams {
    inline topkMultisearchParams(int batch,
                                AscendTensor<float16_t, DIMS_3, size_t> *distResult,
                                AscendTensor<float16_t, DIMS_3, size_t> *minDistResult,
                                AscendTensor<uint32_t, DIMS_3> *opSize,
                                AscendTensor<uint16_t, DIMS_3> *flag,
                                AscendTensor<int64_t, DIMS_1> *attrsInputs,
                                AscendTensor<uint32_t, DIMS_1> *indexOffset,
                                AscendTensor<uint32_t, DIMS_1> *labelOffset,
                                AscendTensor<uint16_t, DIMS_1> *reorderFlag,
                                AscendTensor<float16_t, DIMS_3, size_t> *minDistances,
                                AscendTensor<idx_t, DIMS_3, size_t> *minIndices)
        : batch(batch), distResult(distResult), minDistResult(minDistResult), opSize(opSize), flag(flag),
          attrsInputs(attrsInputs), indexOffset(indexOffset), labelOffset(labelOffset), reorderFlag(reorderFlag),
          minDistances(minDistances), minIndices(minIndices) {}
    int batch;
    AscendTensor<float16_t, DIMS_3, size_t> *distResult;
    AscendTensor<float16_t, DIMS_3, size_t> *minDistResult;
    AscendTensor<uint32_t, DIMS_3> *opSize;
    AscendTensor<uint16_t, DIMS_3> *flag;
    AscendTensor<int64_t, DIMS_1> *attrsInputs;
    AscendTensor<uint32_t, DIMS_1> *indexOffset;
    AscendTensor<uint32_t, DIMS_1> *labelOffset;
    AscendTensor<uint16_t, DIMS_1> *reorderFlag;
    AscendTensor<float16_t, DIMS_3, size_t> *minDistances;
    AscendTensor<idx_t, DIMS_3, size_t> *minIndices;
};

class IndexFlatL2Aicpu : public IndexFlat {
public:
    IndexFlatL2Aicpu(int dim, int64_t resourceSize = -1);

    ~IndexFlatL2Aicpu();

    virtual APP_ERROR init() override;

    APP_ERROR reset() override;

    virtual APP_ERROR addVectors(AscendTensor<float16_t, DIMS_2> &rawData) override;

protected:
    virtual APP_ERROR searchImpl(AscendTensor<float16_t, DIMS_2>& queries, int k,
        AscendTensor<float16_t, DIMS_2>& outDistance, AscendTensor<idx_t, DIMS_2>& outIndices) override;

    APP_ERROR searchImpl(std::vector<Index *> indexes, int n, int batchSize, const float16_t *x, int k,
        float16_t *distances, idx_t *labels) override;

    APP_ERROR searchPaged(size_t pageId, size_t pageNum, AscendTensor<float16_t, DIMS_2> &queries,
        AscendTensor<float16_t, DIMS_2> &maxDistances, AscendTensor<int64_t, DIMS_2> &maxIndices) override;

private:
    void runDistCompute(int batch,
                        const std::vector<const AscendTensorBase *> &input,
                        const std::vector<const AscendTensorBase *> &output,
                        aclrtStream stream);

    APP_ERROR resetDistCompOp(int numLists);

    void moveVectorForward(idx_t srcIdx, idx_t dstIdx) override;
    void releaseUnusageSpace(int oldTotal, int remove) override;

    size_t calcNormBaseSize(idx_t totalNum) const;

    APP_ERROR computeNormHostBuffer(AscendTensor<float16_t, DIMS_2> &rawData);

    APP_ERROR resetTopkOffline();
    APP_ERROR resetTopkCompOp();

    APP_ERROR resetMultisearchTopkCompOp();

    APP_ERROR resetDistOnlineOp(int batch,
                                std::vector<std::pair<aclDataType, std::vector<int64_t>>> &input,
                                std::vector<std::pair<aclDataType, std::vector<int64_t>>> &output);
    void runMultisearchTopkOnline(topkMultisearchParams &opParams,
                                  const std::vector<const AscendTensorBase *> &inputData,
                                  const std::vector<const AscendTensorBase *> &outputData,
                                  aclrtStream streamAicpu);
    void calculateTopkMultisearch(topkMultisearchParams &opParams, aclrtStream streamAicpu);

    void runTopkCompute(AscendTensor<float16_t, DIMS_3, size_t> &dists,
                        AscendTensor<float16_t, DIMS_3, size_t> &maxdists,
                        AscendTensor<uint32_t, DIMS_3> &sizes,
                        AscendTensor<uint16_t, DIMS_3> &flags,
                        AscendTensor<int64_t, DIMS_1> &attrs,
                        AscendTensor<float16_t, DIMS_2> &outdists,
                        AscendTensor<int64_t, DIMS_2> &outlabel,
                        aclrtStream stream);
    
    APP_ERROR initResult(AscendTensor<float16_t, DIMS_3, size_t> &distances,
        AscendTensor<idx_t, DIMS_3, size_t> &indices) const;
    
    APP_ERROR tryToSychResultAdvanced(int &hasCopiedCount, int &indexDoneCount, int indexId,
        int n, int batchSize, int k,
        float16_t *distances, idx_t *labels,
        AscendTensor<float16_t, DIMS_3, size_t> &minDistances, AscendTensor<idx_t, DIMS_3, size_t> &minIndices);

    // aicpu op for topk computation
    std::map<int, std::unique_ptr<::ascend::AscendOperator>> topkComputeOps;
    std::string distOpName;
    int flagNum{ 0 };
    bool isNeedCleanMinDist{ false };
    std::unordered_map<int, std::vector<aclTensorDesc *>> disL2OpInputDesc;
    std::unordered_map<int, std::vector<aclTensorDesc *>> disL2OpOutputDesc;
    std::unordered_map<int, AscendOpDesc> topkOpDesc;
};
}  // namespace ascend

#endif  // ASCENDHOST_INDEXFLAT_L2_AICPU_INCLUDED
