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


#ifndef ASCENDHOST_INCLUDE_INDEX_CUSTOM_INDEXFLAT_AT_SUBAICPU
#define ASCENDHOST_INCLUDE_INDEX_CUSTOM_INDEXFLAT_AT_SUBAICPU

#include "ascenddaemon/impl/Index.h"
#include "ascenddaemon/utils/DeviceVector.h"

namespace ascend {
class IndexFlatATSubAicpu : public Index {
public:
    IndexFlatATSubAicpu(int dim, int baseSize, int64_t resourceSize = -1);

    ~IndexFlatATSubAicpu();

    APP_ERROR init();

    APP_ERROR addVectors(int num, int dim, const AscendTensor<float16_t, DIMS_2> &deviceData);

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
    
    APP_ERROR search(
        idx_t n, const uint8_t *x, idx_t k, int ncentroids, uint16_t *labels,
        AscendTensor<float16_t, DIMS_4>& subcentroids,
        AscendTensor<int16_t, DIMS_3>& hassign, AscendTensor<float16_t, DIMS_2>& vDM);

    int searchPage;

private:
    APP_ERROR searchImpl(int n, const uint8_t *x, int k, int ncentroids, uint16_t *labels, AscendTensor<float16_t,
                         DIMS_3>& subcentroids, AscendTensor<int16_t, DIMS_2>& hassign, AscendTensor<float16_t,
                         DIMS_2>& vDM);

    APP_ERROR searchImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels) override;

    APP_ERROR searchBatched(int n, const uint8_t *x, int k, int ncentroids, uint16_t *labels, AscendTensor<float16_t,
                            DIMS_4>& subcentroids, AscendTensor<int16_t, DIMS_3>& hassign, AscendTensor<float16_t,
                            DIMS_2>& vDM);

    void runDistComputeInt8(const std::vector<const AscendTensorBase *> &input,
                            const std::vector<const AscendTensorBase *> &output,
                            aclrtStream stream);

    APP_ERROR resetDistL2AtOp();

    APP_ERROR resetL2NormFlatSubOp();

    void runL2NormFlatSubOp(const std::vector<const AscendTensorBase *> &input,
                            const std::vector<const AscendTensorBase *> &output,
                            aclrtStream stream);

private:
    int baseSize;

    DeviceVector<float16_t, ExpandPolicySlim> codes;
    DeviceVector<float, ExpandPolicySlim> preCompute;

    std::unique_ptr<AscendOperator> flatL2Op;
    std::unique_ptr<AscendOperator> l2NormFlatSub;
};
} // namespace ascend

#endif // ASCENDHOST_INCLUDE_INDEX_CUSTOM_INDEXFLAT_AT_SUBAICPU
