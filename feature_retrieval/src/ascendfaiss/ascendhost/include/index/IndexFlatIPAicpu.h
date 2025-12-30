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


#ifndef ASCENDHOST_INDEXFLAT_IP_AICPU_INCLUDED
#define ASCENDHOST_INDEXFLAT_IP_AICPU_INCLUDED


#include "ascenddaemon/impl/IndexFlat.h"
#include "ascenddaemon/utils/AscendTensor.h"

namespace ascend {
struct topkFlatIpMultisearchParams {
    inline topkFlatIpMultisearchParams(int batch,
                                       AscendTensor<float16_t, DIMS_3, size_t> &distResult,
                                       AscendTensor<float16_t, DIMS_3, size_t> &maxDistResult,
                                       AscendTensor<uint32_t, DIMS_3> &opSize,
                                       AscendTensor<uint16_t, DIMS_3> &flag,
                                       AscendTensor<int64_t, DIMS_1> &attrsInputs,
                                       AscendTensor<uint32_t, DIMS_1> &indexOffset,
                                       AscendTensor<uint32_t, DIMS_1> &labelOffset,
                                       AscendTensor<uint16_t, DIMS_1> &reorderFlag,
                                       AscendTensor<float16_t, DIMS_3, size_t> &maxDistances,
                                       AscendTensor<idx_t, DIMS_3, size_t> &maxIndices)
        : batch(batch), distResult(distResult), maxDistResult(maxDistResult), opSize(opSize), flag(flag),
          attrsInputs(attrsInputs), indexOffset(indexOffset), labelOffset(labelOffset), reorderFlag(reorderFlag),
          maxDistances(maxDistances), maxIndices(maxIndices) {}
    int batch;
    AscendTensor<float16_t, DIMS_3, size_t> &distResult;
    AscendTensor<float16_t, DIMS_3, size_t> &maxDistResult;
    AscendTensor<uint32_t, DIMS_3> &opSize;
    AscendTensor<uint16_t, DIMS_3> &flag;
    AscendTensor<int64_t, DIMS_1> &attrsInputs;
    AscendTensor<uint32_t, DIMS_1> &indexOffset;
    AscendTensor<uint32_t, DIMS_1> &labelOffset;
    AscendTensor<uint16_t, DIMS_1> &reorderFlag;
    AscendTensor<float16_t, DIMS_3, size_t> &maxDistances;
    AscendTensor<idx_t, DIMS_3, size_t> &maxIndices;
};

class IndexFlatIPAicpu : public IndexFlat {
public:
    IndexFlatIPAicpu(int dim, int64_t resourceSize = -1);

    ~IndexFlatIPAicpu();

    APP_ERROR init() override;

    APP_ERROR addVectors(AscendTensor<float16_t, DIMS_2>& rawData) override;

protected:
    APP_ERROR searchImpl(AscendTensor<float16_t, DIMS_2>& queries, int k,
        AscendTensor<float16_t, DIMS_2>& outDistance, AscendTensor<idx_t, DIMS_2>& outIndices) override;
    APP_ERROR resetTopkCompOp();
    APP_ERROR resetTopkOfflineOp();
    void runTopkCompute(AscendTensor<float16_t, DIMS_3> &dists,
                        AscendTensor<float16_t, DIMS_3> &maxdists,
                        AscendTensor<uint32_t, DIMS_3> &sizes,
                        AscendTensor<uint16_t, DIMS_3> &flags,
                        AscendTensor<int64_t, DIMS_1> &attrs,
                        AscendTensor<float16_t, DIMS_2> &outdists,
                        AscendTensor<int64_t, DIMS_2> &outlabel,
                        aclrtStream stream);

    void runTopkCompute(AscendTensor<float16_t, DIMS_3, size_t> &dists,
                        AscendTensor<float16_t, DIMS_3, size_t> &maxdists,
                        AscendTensor<uint32_t, DIMS_3> &sizes,
                        AscendTensor<uint16_t, DIMS_3> &flags,
                        AscendTensor<int64_t, DIMS_1> &attrs,
                        AscendTensor<float16_t, DIMS_2> &outdists,
                        AscendTensor<int64_t, DIMS_2> &outlabel,
                        aclrtStream stream);

    APP_ERROR resetMultisearchTopkCompOp();

    // aicpu op for topk computation
    std::map<int, std::unique_ptr<::ascend::AscendOperator>> topkComputeOps;

    APP_ERROR searchImpl(std::vector<Index *> indexes, int n, int batchSize, const float16_t *x,
        int k, float16_t *distances, idx_t *labels) override;

    void runDistCompute(int batch,
                        const std::vector<const AscendTensorBase *> &input,
                        const std::vector<const AscendTensorBase *> &output,
                        aclrtStream stream);
    
    APP_ERROR resetDistOnlineOp(int batch,
                                std::vector<std::pair<aclDataType, std::vector<int64_t>>> &input,
                                std::vector<std::pair<aclDataType, std::vector<int64_t>>> &output);

    void runMultisearchTopkOnline(topkFlatIpMultisearchParams &opParams,
                                  const std::vector<const AscendTensorBase *> &input,
                                  const std::vector<const AscendTensorBase *> &output,
                                  aclrtStream streamAicpu);
    void calculateTopkMultisearch(topkFlatIpMultisearchParams &opParams, aclrtStream streamAicpu);

    APP_ERROR resetDistCompOp(int numLists);

    APP_ERROR searchPaged(size_t pageId, size_t pageNum, AscendTensor<float16_t, DIMS_2> &queries,
        AscendTensor<float16_t, DIMS_2> &maxDistances, AscendTensor<int64_t, DIMS_2> &maxIndices) override;

private:
    APP_ERROR initResult(AscendTensor<float16_t, DIMS_3, size_t> &distances,
        AscendTensor<idx_t, DIMS_3, size_t> &indices) const;
    std::unordered_map<int, std::vector<aclTensorDesc *>> disIPOpInputDesc;
    std::unordered_map<int, std::vector<aclTensorDesc *>> disIPOpOutputDesc;
};
}  // namespace ascend

#endif  // ASCEND_INDEXFLAT_IP_INCLUDED
