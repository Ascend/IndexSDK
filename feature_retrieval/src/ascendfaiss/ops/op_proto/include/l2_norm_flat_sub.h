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


#ifndef GE_OP_L2_NORM_FLAT_SUB_H
#define GE_OP_L2_NORM_FLAT_SUB_H

#include "graph/operator_reg.h"

namespace ge {
    REG_OP(L2NormFlatSub)
        .INPUT(x0, TensorType({ DT_FLOAT16 }))              /* "First operand." */
        /* "Result, has same element type as three inputs" */
        .OUTPUT(y0, TensorType({ DT_FLOAT }))
        .OUTPUT(y1, TensorType({ DT_FLOAT16 }))
        .OP_END_FACTORY_REG(L2NormFlatSub)
} // namespace ge
#endif // GE_OP_L2_NORM_FLAT_SUB_H