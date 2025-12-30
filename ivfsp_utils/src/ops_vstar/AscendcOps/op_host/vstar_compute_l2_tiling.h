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


#ifndef VSTAR_COMPUTE_L2_TILING_H
#define VSTAR_COMPUTE_L2_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(VstarComputeL2TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, subDim1);
    TILING_DATA_FIELD_DEF(uint32_t, subDim2);
    TILING_DATA_FIELD_DEF(uint32_t, nlist1);
    TILING_DATA_FIELD_DEF(uint32_t, nlist2);
    TILING_DATA_FIELD_DEF(uint32_t, nprobe1);
    TILING_DATA_FIELD_DEF(uint32_t, n);
    TILING_DATA_FIELD_DEF(uint32_t, formerNum);
    TILING_DATA_FIELD_DEF(uint32_t, probePerBlockFormer);
    TILING_DATA_FIELD_DEF(uint32_t, probePerBlockLatter);
    TILING_DATA_FIELD_DEF(uint32_t, queryCodeAlignSize);
    TILING_DATA_FIELD_DEF(uint32_t, queryCodeUsefulByteSizePerProbe);
    TILING_DATA_FIELD_DEF(uint32_t, queryCodeFormerUsefulByteSize);
    TILING_DATA_FIELD_DEF(uint32_t, queryCodeLatterUsefulByteSize);
    TILING_DATA_FIELD_DEF(uint32_t, moveTimesL1PerProbe);
    TILING_DATA_FIELD_DEF(uint32_t, tailSizeL1PerProbe);
    TILING_DATA_FIELD_DEF(uint32_t, moveTimesL0BPerBlockTail);
    TILING_DATA_FIELD_DEF(uint32_t, remainSizeL0BPerBlock);
    TILING_DATA_FIELD_DEF(uint32_t, sizeCodeBookL1BBuffer);
    TILING_DATA_FIELD_DEF(uint32_t, sizeCodeBookL0BBuffer);
    TILING_DATA_FIELD_DEF(uint32_t, cubeAlign);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(VstarComputeL2, VstarComputeL2TilingData)
}

#endif