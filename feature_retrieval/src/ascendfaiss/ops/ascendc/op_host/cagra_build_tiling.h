/*
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
 * Licensed under Mulan PSL v2.
 */
#ifndef ASCENDC_CAGRA_BUILD_TILING_H
#define ASCENDC_CAGRA_BUILD_TILING_H

#include <cstdint>

#include "register/tilingdata_base.h"

namespace optiling
{
BEGIN_TILING_DATA_DEF(CagraBuildTilingData)
TILING_DATA_FIELD_DEF(uint32_t, nodeNum);
TILING_DATA_FIELD_DEF(uint32_t, dim);
TILING_DATA_FIELD_DEF(uint32_t, intermediateDegree);
TILING_DATA_FIELD_DEF(uint32_t, sampleDegree);
TILING_DATA_FIELD_DEF(uint32_t, candidateDegree);
TILING_DATA_FIELD_DEF(uint32_t, outputDegree);
TILING_DATA_FIELD_DEF(uint32_t, blockNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CagraNndInit, CagraBuildTilingData)
REGISTER_TILING_DATA_CLASS(CagraNndSampleReverse, CagraBuildTilingData)
REGISTER_TILING_DATA_CLASS(CagraNndLocalJoin, CagraBuildTilingData)
REGISTER_TILING_DATA_CLASS(CagraNndUpdate, CagraBuildTilingData)
REGISTER_TILING_DATA_CLASS(CagraPruneReverse, CagraBuildTilingData)
REGISTER_TILING_DATA_CLASS(CagraMerge, CagraBuildTilingData)
}  // namespace optiling

#endif
