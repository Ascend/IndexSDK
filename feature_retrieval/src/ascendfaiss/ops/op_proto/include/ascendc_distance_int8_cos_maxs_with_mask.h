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


#ifndef GE_OP_ASCENDC_DISTANCE_INT8_COS_MAXS_WITH_MASK_H
#define GE_OP_ASCENDC_DISTANCE_INT8_COS_MAXS_WITH_MASK_H

#include "graph/operator_reg.h"

namespace ge {
    REG_OP(AscendcDistanceInt8CosMaxsWithMask)
        .INPUT(x0, TensorType({ DT_INT8 })) /* "First operand." */
        .INPUT(x1, TensorType({ DT_UINT8 }))   /* "Second operand." */
        .INPUT(x2, TensorType({ DT_INT8 }))   /* "Third operand." */
        .INPUT(x3, TensorType({ DT_FLOAT16 }))   /* "Fourth operand." */
        .INPUT(x4, TensorType({ DT_FLOAT16 })) /* "Fifth operand." */
        .INPUT(x5, TensorType({ DT_UINT32 })) /* "Sixth operand." */
        /* "Result, has same element type as three inputs" */
        .OUTPUT(y0, TensorType({ DT_FLOAT16 }))
        .OUTPUT(y1, TensorType({ DT_FLOAT16 }))
        .OUTPUT(y2, TensorType({ DT_UINT16 }))
        .OP_END_FACTORY_REG(AscendcDistanceInt8CosMaxsWithMask)
} // namespace ge
#endif // GE_OP_ASCENDC_DISTANCE_INT8_COS_MAXS_WITH_MASK_H