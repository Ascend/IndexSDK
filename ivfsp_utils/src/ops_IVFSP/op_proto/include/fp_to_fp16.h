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


#ifndef GE_OP_FP_TO_FP16_H
#define GE_OP_FP_TO_FP16_H

#include "graph/operator_reg.h"

namespace ge {
REG_OP(FpToFp16)
    .INPUT(x0, TensorType({ DT_FLOAT32 })) /* "First operand." */
    .OUTPUT(y0, TensorType({ DT_FLOAT16 }))
    .OUTPUT(y1, TensorType({ DT_UINT16 }))
    .OP_END_FACTORY_REG(FpToFp16)
} // namespace ge

#endif // GE_OP_FP_TO_FP16_H