/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

#ifndef GE_OP_DISTANCE_VAL_MASK_GENERATOR_H
#define GE_OP_DISTANCE_VAL_MASK_GENERATOR_H
#include "graph/operator_reg.h"
namespace ge {
REG_OP(DistanceValMaskGenerator)
    .INPUT(x0, TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16,
                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                           DT_COMPLEX64, DT_STRING}))
    .INPUT(x1,
           TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16,
                       DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                       DT_COMPLEX64, DT_STRING}))
    .INPUT(x2,
           TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16,
                       DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                       DT_COMPLEX64, DT_STRING}))
    .INPUT(x3,
           TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16,
                       DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                       DT_COMPLEX64, DT_STRING}))
    .INPUT(x4,
           TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16,
                       DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                       DT_COMPLEX64, DT_STRING}))
    .INPUT(x5,
           TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16,
                       DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                       DT_COMPLEX64, DT_STRING}))
    .INPUT(x6,
           TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16,
                       DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                       DT_COMPLEX64, DT_STRING}))
    .OUTPUT(y0,
            TensorType({DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_INT16,
                        DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                        DT_COMPLEX64, DT_STRING}))
    .OP_END_FACTORY_REG(DistanceValMaskGenerator)
}
#endif

