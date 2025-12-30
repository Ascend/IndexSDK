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


#ifndef GE_OP_DISTANCE_MASKED_IVF_SP_INT_L2_MINS_H
#define GE_OP_DISTANCE_MASKED_IVF_SP_INT_L2_MINS_H

#include "graph/operator_reg.h"

namespace ge {
    REG_OP(DistanceMaskedIVFSpIntL2Mins)
        .INPUT(x0, TensorType({ DT_FLOAT16 }))              /* "First operand." */
         .INPUT(x1, TensorType({ DT_FLOAT16 })) /* "Second operand." */
        .INPUT(x2, TensorType({ DT_UINT8 }))                /* "Second operand." */
        .INPUT(x3, TensorType({ DT_UINT8 }))
        .INPUT(x4, TensorType({ DT_FLOAT }))                /* "Third operand." */
        .INPUT(x5, TensorType({ DT_FLOAT16 }))              /* "Fourth operand." */
        .INPUT(x6, TensorType({ DT_UINT32 }))               /* "Sixth operand." */
        .INPUT(x7, TensorType({ DT_UINT64 }))  /* "Fifth operand." */
        .INPUT(x8, TensorType({ DT_UINT64 }))  /* "Fifth operand." */
        .INPUT(x9, TensorType({ DT_UINT64 }))

        /* "Result, has same element type as three inputs" */
        .OUTPUT(y0, TensorType({ DT_FLOAT16 }))
        .OUTPUT(y1, TensorType({ DT_FLOAT16 }))
        .OUTPUT(y2, TensorType({ DT_UINT16 }))
        .OP_END_FACTORY_REG(DistanceMaskedIVFSpIntL2Mins)
} // namespace ge
#endif // GE_OP_DISTANCE_SQ8_L2_MINS_H