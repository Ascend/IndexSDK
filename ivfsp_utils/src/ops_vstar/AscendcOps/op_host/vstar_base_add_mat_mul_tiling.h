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


#ifndef VSTAR_BASE_ADD_MAT_MUL_TILING_H
#define VSTAR_BASE_ADD_MAT_MUL_TILING_H
#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"

namespace optiling {
    BEGIN_TILING_DATA_DEF(VstarBaseAddMatMulTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, nb);
    TILING_DATA_FIELD_DEF(uint32_t, dim);
    TILING_DATA_FIELD_DEF(uint32_t, nList);
    TILING_DATA_FIELD_DEF(uint32_t, subDim);
    TILING_DATA_FIELD_DEF(uint64_t, MMFormatUb1);
    TILING_DATA_FIELD_DEF(uint32_t, aicNum);
    TILING_DATA_FIELD_DEF(uint32_t, aivNum);
    TILING_DATA_FIELD_DEF(uint32_t, each_M_regular);
    TILING_DATA_FIELD_DEF(uint32_t, each_M_extra);
    TILING_DATA_FIELD_DEF(uint32_t, loop_M_regular);
    TILING_DATA_FIELD_DEF(uint32_t, loop_M_extra);
    TILING_DATA_FIELD_DEF(uint32_t, last_M_regular);
    TILING_DATA_FIELD_DEF(uint32_t, last_M_extra);
    TILING_DATA_FIELD_DEF(uint32_t, MTaskCore_regular);
    TILING_DATA_FIELD_DEF(uint32_t, MTaskCore_extra);

    TILING_DATA_FIELD_DEF(uint32_t, each_N_regular);
    TILING_DATA_FIELD_DEF(uint32_t, each_N_extra);
    TILING_DATA_FIELD_DEF(uint32_t, loop_N_regular);
    TILING_DATA_FIELD_DEF(uint32_t, loop_N_extra);
    TILING_DATA_FIELD_DEF(uint32_t, last_N_regular);
    TILING_DATA_FIELD_DEF(uint32_t, last_N_extra);
    TILING_DATA_FIELD_DEF(uint32_t, NTaskCore_regular);
    TILING_DATA_FIELD_DEF(uint32_t, NTaskCore_extra);

    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cube_tiling);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(VstarBaseAddMatMul, VstarBaseAddMatMulTilingData)
}
#endif