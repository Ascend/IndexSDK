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


#ifndef ASCENDHOST_INDEXINT8FLAT_L2_AICPU_INCLUDED
#define ASCENDHOST_INDEXINT8FLAT_L2_AICPU_INCLUDED


#include "ascenddaemon/impl/IndexInt8Flat.h"
#include "ascenddaemon/utils/AscendTensor.h"

namespace ascend {
class IndexInt8FlatL2Aicpu : public IndexInt8Flat<int32_t> {
public:
    IndexInt8FlatL2Aicpu(int dim, int64_t resourceSize = -1, int blockSize = FLAT_DEFAULT_DIST_COMPUTE_BATCH);

    ~IndexInt8FlatL2Aicpu();

    APP_ERROR init() override;

    APP_ERROR addVectors(AscendTensor<int8_t, DIMS_2> &rawData) override;

protected:
    void runDistCompute(int batch,
                        const std::vector<const AscendTensorBase *> &input,
                        const std::vector<const AscendTensorBase *> &output,
                        aclrtStream stream, uint32_t actualNum = 0) const override;

    APP_ERROR resetDistCompOp(int codeNum);

    void initSearchResult(int indexesSize, int n, int k, float16_t *distances, idx_t *labels) override;
};
} // namespace ascend

#endif
