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

#ifndef GE_OP_ASCENDC_DISTANCE_BATCH_MASK_GENERATOR_WITH_EXTRA_H
#define GE_OP_ASCENDC_DISTANCE_BATCH_MASK_GENERATOR_WITH_EXTRA_H
#include "graph/operator_reg.h"
namespace ge {
REG_OP(AscendcDistanceBatchMaskGeneratorWithExtra)
    .INPUT(query_time_stamp, TensorType({DT_INT32}))
    .INPUT(query_token_set, TensorType({DT_UINT8}))
    .INPUT(db_time_stamp, TensorType({DT_INT32}))
    .INPUT(db_divisor, TensorType({DT_INT32}))
    .INPUT(db_remainder, TensorType({DT_UINT8}))
    .INPUT(extra_mask, TensorType({DT_UINT8}))
    .INPUT(extra_mask_attr, TensorType({DT_INT32}))
    .OUTPUT(distance_mask, TensorType({DT_UINT8}))
    .OP_END_FACTORY_REG(AscendcDistanceBatchMaskGeneratorWithExtra)
}
#endif
