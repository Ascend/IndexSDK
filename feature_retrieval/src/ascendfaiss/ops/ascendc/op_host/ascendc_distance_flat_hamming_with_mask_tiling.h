/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
 *
 * IndexSDK is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of Mulan PSL v2.
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

#ifndef ASCENDC_DISTANCE_FLAT_HAMMING_WITH_MASK_TILING_H
#define ASCENDC_DISTANCE_FLAT_HAMMING_WITH_MASK_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling
{
BEGIN_TILING_DATA_DEF(AscendcDistanceFlatHammingWithMaskTilingData)
TILING_DATA_FIELD_DEF(uint32_t, queryNum);
TILING_DATA_FIELD_DEF(uint32_t, codeSize);
TILING_DATA_FIELD_DEF(uint32_t, dim);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, zRegionHeight);
TILING_DATA_FIELD_DEF(uint32_t, burstLen);
TILING_DATA_FIELD_DEF(uint32_t, codeTile);
TILING_DATA_FIELD_DEF(uint32_t, maskRows);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingIp);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AscendcDistanceFlatHammingWithMask, AscendcDistanceFlatHammingWithMaskTilingData)
}  // namespace optiling

#endif  // ASCENDC_DISTANCE_FLAT_HAMMING_WITH_MASK_TILING_H
