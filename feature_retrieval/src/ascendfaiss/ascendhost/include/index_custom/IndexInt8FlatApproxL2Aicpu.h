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


#ifndef ASCENDHOST_INDEXINT8FLATAPPROX_AICPU_INCLUDED
#define ASCENDHOST_INDEXINT8FLATAPPROX_AICPU_INCLUDED


#include "IndexInt8FlatApproxL2Aicpu.h"
#include "ascenddaemon/impl/IndexInt8Flat.h"
#include "ascenddaemon/utils/AscendTensor.h"

namespace ascend {
namespace {
const int INT8_DEFAULT_DIST_COMPUTE_BATCH = 16384 * 16;
}
class IndexInt8FlatApproxL2Aicpu : public IndexInt8Flat<float16_t> {
public:
    IndexInt8FlatApproxL2Aicpu(int dim, int64_t resourceSize = -1, int blockSize = INT8_DEFAULT_DIST_COMPUTE_BATCH);

    ~IndexInt8FlatApproxL2Aicpu();

    APP_ERROR init() override;

protected:
    void ivecNormsL2sqr(float16_t *nr, const int8_t *x, size_t d, size_t nx);

    float ivecNormL2sqr(const int8_t *x, size_t d) const;

    APP_ERROR computeNorm(AscendTensor<int8_t, DIMS_2> &rawData);
    APP_ERROR addVectors(AscendTensor<int8_t, DIMS_2> &rawData) override;

    float scale;
private:
    APP_ERROR resetDistCompOp(int codeNum) const;

    void runDistCompute(int batch, const std::vector<const AscendTensorBase *> &input,
        const std::vector<const AscendTensorBase *> &output, aclrtStream stream,
        uint32_t actualNum = 0) const override;
};
} // namespace ascend

#endif
