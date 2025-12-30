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


#ifndef ASCENDC_OP_HOST_DISTANCE_L2_MINS_AT_FP32_TILING_H
#define ASCENDC_OP_HOST_DISTANCE_L2_MINS_AT_FP32_TILING_H
#include "register/tilingdata_base.h"

#define ASCENDC_RETURN_IF_NOT(X, ERRCODE)                                                                   \
    do                                                                                                              \
    {                                                                                                               \
        if (!(X)) {                                                                                                 \
            return ERRCODE;                                                                                         \
        }                                                                                                           \
    } while (false)

namespace optiling {
BEGIN_TILING_DATA_DEF(DistanceFlatL2MinsAtFP32TilingData)
    TILING_DATA_FIELD_DEF(int32_t, formerCoreNum);
    TILING_DATA_FIELD_DEF(int32_t, formerCoreLength);
    TILING_DATA_FIELD_DEF(int32_t, tailCoreNum);
    TILING_DATA_FIELD_DEF(int32_t, tailCoreLength);
    TILING_DATA_FIELD_DEF(int32_t, tileNum);
    TILING_DATA_FIELD_DEF(int32_t, tileLength);
    TILING_DATA_FIELD_DEF(int32_t, lastTileLength);
    TILING_DATA_FIELD_DEF(int32_t, queryNumLength);
    TILING_DATA_FIELD_DEF(int32_t, codesNumLength);
    TILING_DATA_FIELD_DEF(int32_t, dimLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DistanceFlatL2MinsAtFP32, DistanceFlatL2MinsAtFP32TilingData)
}

#endif // ASCENDC_OP_HOST_DISTANCE_L2_MINS_AT_FP32_TILING_H