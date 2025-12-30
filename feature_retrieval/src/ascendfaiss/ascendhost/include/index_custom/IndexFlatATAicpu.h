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


#ifndef ASCENDHOST_INDEXFLAT_AT_AICPU_INCLUDED
#define ASCENDHOST_INDEXFLAT_AT_AICPU_INCLUDED

#include "ascenddaemon/impl/Index.h"
#include "ascenddaemon/utils/DeviceVector.h"
#include "index_custom/IndexFlatAT.h"

namespace ascend {
class IndexFlatATAicpu : public IndexFlatAT {
public:
    IndexFlatATAicpu(int dim, int baseSize, int64_t resourceSize = -1);

    ~IndexFlatATAicpu();

    APP_ERROR init() override;

    APP_ERROR addVectors(const AscendTensor<float16_t, DIMS_2> &rawData);

    APP_ERROR reset();

    APP_ERROR getVectors(uint32_t offset, uint32_t num, std::vector<float16_t> &vectors);

    void clearTmpAscendTensor();

    void setResultCopyBack(bool value);

protected:
    APP_ERROR saveCodes(const AscendTensor<float16_t, DIMS_2> &rawData);

    APP_ERROR saveNorms(const AscendTensor<float16_t, DIMS_2> &rawData);

    APP_ERROR searchBatched(int64_t n, const float16_t* x, int64_t k, float16_t* distance, idx_t* labels) override;

    APP_ERROR searchImpl(int n, const float16_t* x, int k, float16_t* distances, idx_t* labels) override;

    APP_ERROR computeL2(int n, int k,
                        float16_t *distances, idx_t *labels,
                        AscendTensor<float16_t, DIMS_4> &queries,
                        AscendTensor<float, DIMS_1> &queriesNorms,
                        AscendTensor<float16_t, DIMS_3> &distResult,
                        AscendTensor<float16_t, DIMS_3> &minDistResult,
                        AscendTensor<uint16_t, DIMS_3> &opFlag);
private:
    void runTopkCompute(AscendTensor<float16_t, DIMS_3> &dists,
                        AscendTensor<float16_t, DIMS_3> &maxdists,
                        AscendTensor<uint32_t, DIMS_3> &sizes,
                        AscendTensor<uint16_t, DIMS_3> &flags,
                        AscendTensor<int64_t, DIMS_1> &attrs,
                        AscendTensor<float16_t, DIMS_2> &outdists,
                        AscendTensor<int64_t, DIMS_2> &outlabel,
                        aclrtStream stream);

    APP_ERROR computeL2Init(int n, int k,
                            AscendTensor<float16_t, DIMS_3> &distResult,
                            AscendTensor<float16_t, DIMS_3> &minDistResult,
                            AscendTensor<uint16_t, DIMS_3> &opFlag,
                            AscendTensor<int64_t, DIMS_1> &attrsInput);

    void runL2NormOp(const std::vector<const AscendTensorBase *> &input,
                     const std::vector<const AscendTensorBase *> &output,
                     aclrtStream stream) const;

    void runDistCompute(const std::vector<const AscendTensorBase *> &input,
                        const std::vector<const AscendTensorBase *> &output,
                        aclrtStream stream) const;

    APP_ERROR resetL2NormOp() const;

    APP_ERROR resetDistL2MinsAtOp() const;

private:
    bool resultCopyBack = true;

    DeviceVector<float16_t, ExpandPolicySlim> codes;
    DeviceVector<float, ExpandPolicySlim> preCompute;

    AscendTensor<float, DIMS_2> transfer;

    AscendTensor<float16_t, DIMS_3> distResult;
    AscendTensor<float16_t, DIMS_3> minDistResult;
    AscendTensor<uint16_t, DIMS_3> opFlag;
    AscendTensor<float16_t, DIMS_3> minDistances;
    AscendTensor<uint32_t, DIMS_3> minIndices;
};
}  // namespace ascend

#endif  // ASCENDHOST_INDEXFLAT_AT_AICPU_INCLUDED
