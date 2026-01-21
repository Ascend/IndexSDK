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

#ifndef ROTATE_AND_L2_AT_FP32_TILING_H
#define ROTATE_AND_L2_AT_FP32_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/matrix/matmul_tiling.h"

#define ASCENDC_RETURN_IF_NOT(X, ERRCODE)                                                                   \
    do                                                                                                              \
    {                                                                                                               \
        if (!(X)) {                                                                                                 \
            return ERRCODE;                                                                                         \
        }                                                                                                           \
    } while (false)
    
namespace optiling {
BEGIN_TILING_DATA_DEF(RotateAndL2AtFP32TilingData)
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, gemm_qb_tiling);
    TILING_DATA_FIELD_DEF(int32_t, tileLength);  // vector 单元每次循环处理的向量数量
    TILING_DATA_FIELD_DEF(int32_t, vecNumLength);     // 向量数量
    TILING_DATA_FIELD_DEF(int32_t, dimLength);   // dim大小
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RotateAndL2AtFP32, RotateAndL2AtFP32TilingData)
}

#endif
