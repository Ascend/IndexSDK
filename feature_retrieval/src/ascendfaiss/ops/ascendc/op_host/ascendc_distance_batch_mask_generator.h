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

#ifndef ASCENDC_DISTANCE_BATCH_MASK_GENERATOR_H
#define ASCENDC_DISTANCE_BATCH_MASK_GENERATOR_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AscendcDistanceBatchMaskGeneratorTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, tokenCnt);
TILING_DATA_FIELD_DEF(uint32_t, tileLen);
TILING_DATA_FIELD_DEF(uint32_t, formerNum);
TILING_DATA_FIELD_DEF(uint32_t, formerRepeatNum);
TILING_DATA_FIELD_DEF(uint32_t, tailRepeatNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AscendcDistanceBatchMaskGenerator, AscendcDistanceBatchMaskGeneratorTilingData)
} // namespace optiling
#endif // ASCENDC_DISTANCE_BATCH_MASK_GENERATOR_H