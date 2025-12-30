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

#ifndef ASCENDC_DIST_INT8_FLAT_L2_TILING_H
#define ASCENDC_DIST_INT8_FLAT_L2_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"


namespace optiling {
BEGIN_TILING_DATA_DEF(AscendcDistInt8FlatL2TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, aivNum);
    TILING_DATA_FIELD_DEF(uint32_t, querySize);
    TILING_DATA_FIELD_DEF(uint32_t, codeBlockSize);
    TILING_DATA_FIELD_DEF(uint32_t, dim);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingSquare);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingIp);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AscendcDistInt8FlatL2, AscendcDistInt8FlatL2TilingData)
}
#endif