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

#ifndef DISTANCE_IVF_Rabitq_L2_FP32_TILING_H
#define DISTANCE_IVF_Rabitq_L2_FP32_TILING_H
#include "register/tilingdata_base.h"

#define ASCENDC_RETURN_IF_NOT(X, ERRCODE)                                                                   \
    do                                                                                                              \
    {                                                                                                               \
        if (!(X)) {                                                                                                 \
            return ERRCODE;                                                                                         \
        }                                                                                                           \
    } while (false)

namespace optiling {
BEGIN_TILING_DATA_DEF(DistanceIVFRabitqL2FP32TilingData)
    TILING_DATA_FIELD_DEF(int32_t, dimLength);  // dim大小

    TILING_DATA_FIELD_DEF(int32_t, codeTileLength);

    TILING_DATA_FIELD_DEF(int32_t, codeBlockLength); // 每个块的长度

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DistanceIVFRabitqL2FP32, DistanceIVFRabitqL2FP32TilingData)
}
#endif