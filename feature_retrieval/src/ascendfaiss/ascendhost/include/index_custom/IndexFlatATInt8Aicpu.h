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

 
#ifndef ASCENDHOST_INDEXFLAT_AT_INT8_AICPU_INCLUDED
#define ASCENDHOST_INDEXFLAT_AT_INT8_AICPU_INCLUDED
 
#include "ascenddaemon/impl/Index.h"
#include "ascenddaemon/utils/AscendOperator.h"
#include "ascenddaemon/utils/DeviceVector.h"
#include "index_custom/IndexFlatAT.h"
 
namespace ascend {
class IndexFlatATInt8Aicpu : public IndexFlatAT {
public:
    IndexFlatATInt8Aicpu(int dim, int baseSize, int64_t resourceSize = -1);
 
    ~IndexFlatATInt8Aicpu();
 
    APP_ERROR init() override;

    APP_ERROR addVectors(const AscendTensor<float16_t, DIMS_2> &rawData);

    APP_ERROR reset();

    APP_ERROR getVectors(uint32_t offset, uint32_t num, std::vector<float16_t> &vectors);

    inline int getSize() const
    {
        return ntotal;
    }

    inline int getDim() const
    {
        return dims;
    }

    void clearTmpAscendTensor();

    APP_ERROR searchInt8(idx_t n, const int8_t *x, idx_t k, float16_t *distances, idx_t *labels);

    void updateQMinMax(float16_t qMin, float16_t qMax);

private:
    APP_ERROR saveCodesInt8(const AscendTensor<float16_t, DIMS_2> &rawData);

    APP_ERROR searchBatchedInt8(int n, const int8_t *x, int k, float16_t *distance, idx_t *labels);

    APP_ERROR searchImplInt8(int n, const int8_t *x, int k, float16_t *distances, idx_t *labels);

    APP_ERROR searchBatched(int64_t n, const float16_t* x, int64_t k, float16_t* distance, idx_t* labels) override;

    APP_ERROR searchImpl(int n, const float16_t* x, int k, float16_t* distances, idx_t* labels) override;

    APP_ERROR computeL2Int8(int n, int k,
                            float16_t *distances, idx_t *labels,
                            AscendTensor<int8_t, DIMS_4> &queries,
                            AscendTensor<int32_t, DIMS_1> &queriesNorms,
                            AscendTensor<float16_t, DIMS_3> &distResult,
                            AscendTensor<float16_t, DIMS_3> &minDistResult,
                            AscendTensor<uint16_t, DIMS_3> &opFlag);
 
    void runTopkCompute(AscendTensor<float16_t, DIMS_3> &dists,
                        AscendTensor<float16_t, DIMS_3> &maxdists,
                        AscendTensor<uint32_t, DIMS_3> &sizes,
                        AscendTensor<uint16_t, DIMS_3> &flags,
                        AscendTensor<int64_t, DIMS_1> &attrs,
                        AscendTensor<float16_t, DIMS_2> &outdists,
                        AscendTensor<int64_t, DIMS_2> &outlabel,
                        aclrtStream stream);

    APP_ERROR saveNormsInt8(AscendTensor<int8_t, DIMS_2> &codesQ);

    APP_ERROR computeL2Int8Init(int n, int k,
                                AscendTensor<float16_t, DIMS_3> &distResult,
                                AscendTensor<float16_t, DIMS_3> &minDistResult,
                                AscendTensor<uint16_t, DIMS_3> &opFlag,
                                AscendTensor<int64_t, DIMS_1> &attrsInput);

    void runL2NormTypingInt8Op(const std::vector<const AscendTensorBase *> &input,
                               const std::vector<const AscendTensorBase *> &output,
                               aclrtStream stream) const;

    void runInt8DistCompute(const std::vector<const AscendTensorBase *> &input,
                            const std::vector<const AscendTensorBase *> &output,
                            aclrtStream stream) const;

    APP_ERROR resetL2NormTypingInt8Op() const;

    APP_ERROR resetDistL2MinsInt8AtOp() const;

    APP_ERROR CopyLabelsDeviceToHost(idx_t* hostData, int n, int k, AscendTensor<int64_t, DIMS_3> &deviceData);

    APP_ERROR CopyDisDeviceToHost(float16_t* hostData, int n, int k, AscendTensor<float16_t, DIMS_3> &deviceData);
private:
    int maxBatches = 0;
    int maxK = 0;

    DeviceVector<int8_t, ExpandPolicySlim> codes;
    DeviceVector<int, ExpandPolicySlim> preComputeInt;

    AscendTensor<int32_t, DIMS_2> transferInt32;

    AscendTensor<float16_t, DIMS_3> distResult;
    AscendTensor<float16_t, DIMS_3> minDistResult;
    AscendTensor<uint16_t, DIMS_3> opFlag;
    AscendTensor<float16_t, DIMS_3> minDistances;
    AscendTensor<uint32_t, DIMS_3> minIndices;

    float16_t qMin = 0.0;
    float16_t qMax = 0.0;
};
}  // namespace ascend
 
#endif  // ASCENDHOST_INDEXFLAT_AT_INT8_AICPU_INCLUDED