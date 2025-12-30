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


#ifndef DISTANCE_MATRIX_OP_INCLUDED
#define DISTANCE_MATRIX_OP_INCLUDED

#include <npu/common/AscendFp16.h>

#include "npu/common/AscendTensor.h"

namespace ascendSearchacc {
class DistanceMatrixOp {
public:
    DistanceMatrixOp();
    virtual ~DistanceMatrixOp();

    bool exec(AscendTensor<unsigned char, DIMS_2> &code, AscendTensor<float16_t, DIMS_2> &distTable,
              AscendTensor<float16_t, DIMS_2> &distMatrix);

private:
    bool checkParams(AscendTensor<unsigned char, DIMS_2> &code, AscendTensor<float16_t, DIMS_2> &distTable,
                     AscendTensor<float16_t, DIMS_2> &distMatrix);

    void computePqCodeSize2(AscendTensor<unsigned char, DIMS_2> &code, AscendTensor<float16_t, DIMS_2> &distTable,
                            AscendTensor<float16_t, DIMS_2> &distMatrix);

    void computePqCodeSize4(AscendTensor<unsigned char, DIMS_2> &code, AscendTensor<float16_t, DIMS_2> &distTable,
                            AscendTensor<float16_t, DIMS_2> &distMatrix);

    void computePqCodeSize16(AscendTensor<unsigned char, DIMS_2> &code, AscendTensor<float16_t, DIMS_2> &distTable,
                             AscendTensor<float16_t, DIMS_2> &distMatrix);
};
}  // namespace ascendSearchacc

#endif
