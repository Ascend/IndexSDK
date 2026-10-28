/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

#ifndef ASCENDC_DIST_INT8_COS_950_TILING_H
#define ASCENDC_DIST_INT8_COS_950_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling
{
BEGIN_TILING_DATA_DEF(AscendcDistInt8FlatCosWith950TilingData)
TILING_DATA_FIELD_DEF(uint32_t, queryNum);
TILING_DATA_FIELD_DEF(uint32_t, dim);
TILING_DATA_FIELD_DEF(uint32_t, baseBlockSize);
TILING_DATA_FIELD_DEF(uint32_t, vecCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, aicNum);
TILING_DATA_FIELD_DEF(uint32_t, onceComputeBaseNum);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AscendcDistInt8FlatCosWith950, AscendcDistInt8FlatCosWith950TilingData)
}  // namespace optiling
#endif  // ASCENDC_DIST_INT8_COS_TILING_H
