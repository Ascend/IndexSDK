
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


#ifndef DISTANCE_FLAT_L2_TILING_H
#define DISTANCE_FLAT_L2_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/matrix/matmul_tilingdata.h"
#include "tiling/matrix/matmul_tiling.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DistanceFlatL2TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, aivNum);
    TILING_DATA_FIELD_DEF(uint32_t, querySize);
    TILING_DATA_FIELD_DEF(uint32_t, codeSize);
    TILING_DATA_FIELD_DEF(uint32_t, dimSize);
    TILING_DATA_FIELD_DEF(uint32_t, queryLoopTimes);
    TILING_DATA_FIELD_DEF(uint32_t, querySizeEachLoop);
    TILING_DATA_FIELD_DEF(uint32_t, querySizeLastLoop);
    TILING_DATA_FIELD_DEF(uint32_t, codeSizeEachLoop);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingIp);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingL2Norm);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DistanceFlatL2, DistanceFlatL2TilingData)
}

#endif