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


#ifndef ASCENDHOST_INDEXINT8FLAT_Cos_AICPU_INCLUDED
#define ASCENDHOST_INDEXINT8FLAT_Cos_AICPU_INCLUDED


#include "ascenddaemon/impl/IndexInt8Flat.h"
#include "ascenddaemon/utils/AscendTensor.h"

namespace ascend {
class IndexInt8FlatCosAicpu : public IndexInt8Flat<float16_t> {
public:
    IndexInt8FlatCosAicpu(int dim, int64_t resourceSize = -1, int blockSize = FLAT_DEFAULT_DIST_COMPUTE_BATCH);

    ~IndexInt8FlatCosAicpu();

    APP_ERROR init();

    APP_ERROR addVectors(AscendTensor<int8_t, DIMS_2> &rawData);

protected:
    virtual APP_ERROR searchImpl(int n, const int8_t *x, int k, float16_t *distances, idx_t *labels) override;
    APP_ERROR resetDistCompOp(int codeNum) const;
    void runDistCompute(int batch,
                        const std::vector<const AscendTensorBase *> &input,
                        const std::vector<const AscendTensorBase *> &output,
                        aclrtStream stream, uint32_t actualNum = 0) const override;
    APP_ERROR calL2norm(int num, AscendTensor<int8_t, DIMS_2> &rawTensor, AscendTensor<float16_t, 1> &precompData);
    APP_ERROR copyNormByIndice(int64_t startIndice, int64_t length, int64_t normOffset,
                               AscendTensor<float16_t, DIMS_1> &precompData);
private:
    APP_ERROR prepareIndexSearchTensorShare(const HeteroBlockGroupMgr &grpSpliter,
                                           IndexSearchContext &ctx, IndexSearchTensorShare &outCond);
    APP_ERROR searchInGroups(const HeteroBlockGroupMgr &grpSpliter, IndexSearchContext &ctx,
                             AscendTensor<uint8_t, DIMS_2> &mask);
    APP_ERROR searchInGroup(size_t grpId, const HeteroBlockGroupMgr &grpSpliter,
                            IndexSearchContext &ctx, IndexSearchTensorShare &tensorCond,
                            AscendTensor<uint8_t, DIMS_2> &mask);
    APP_ERROR prepareMaskData(int n, const aclrtStream &stream, AscendTensor<uint8_t, DIMS_2> &retMask);

    APP_ERROR searchPaged(int pageId, AscendTensor<int8_t, DIMS_2> &queries, int k,
        AscendTensor<float16_t, DIMS_2> &minDistance, AscendTensor<int64_t, DIMS_2> &minIndices,
        AscendTensor<uint8_t, DIMS_2> &mask, AscendTensor<float16_t, DIMS_1> &queriesNorm);

    APP_ERROR getThreshold(
            AscendTensor<float16_t, DIMS_2>& topKThreshold,
            int nq, int k, int pageId, const AscendTensor<float16_t, DIMS_2> &minDistances) const;

    APP_ERROR groupSearchImpl(int n, const int8_t *x, int k, float16_t *distances, idx_t *labels);

    int calNumThresholds(int n, int dims, int burstLen) const;

    void initSearchResult(int indexesSize, int n, int k, float16_t *distances, idx_t *labels) override;

private:
    int numThresholds = 0;
};
} // namespace ascend

#endif
