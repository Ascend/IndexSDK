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
#include "tiling/matrix/matmul_tiling.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DistanceFlatL2MinsAtFP32TilingData)
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, gemm_qb_tiling);
    TILING_DATA_FIELD_DEF(int32_t, queryNumLength);
    TILING_DATA_FIELD_DEF(int32_t, codesNumLength);
    TILING_DATA_FIELD_DEF(int32_t, dimLength);
    TILING_DATA_FIELD_DEF(int32_t, baseCodesNumLength);
    TILING_DATA_FIELD_DEF(int32_t, formerCoreNum);
    TILING_DATA_FIELD_DEF(int32_t, formerCoreLength);
    TILING_DATA_FIELD_DEF(int32_t, formerTileNum);
    TILING_DATA_FIELD_DEF(int32_t, formerTileLength);
    TILING_DATA_FIELD_DEF(int32_t, formerLastTileLength);
    TILING_DATA_FIELD_DEF(int32_t, tailCoreNum);
    TILING_DATA_FIELD_DEF(int32_t, tailCoreLength);
    TILING_DATA_FIELD_DEF(int32_t, tailTileNum);
    TILING_DATA_FIELD_DEF(int32_t, tailTileLength);
    TILING_DATA_FIELD_DEF(int32_t, tailLastTileLength);
    TILING_DATA_FIELD_DEF(int32_t, codesTileNum);
    TILING_DATA_FIELD_DEF(int32_t, codesTileLength);
    TILING_DATA_FIELD_DEF(int32_t, codesLastTileLength);
    TILING_DATA_FIELD_DEF(int32_t, bufferSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DistanceFlatL2MinsAtFP32, DistanceFlatL2MinsAtFP32TilingData)
}

#endif // ASCENDC_OP_HOST_DISTANCE_L2_MINS_AT_FP32_TILING_H