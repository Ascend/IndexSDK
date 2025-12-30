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


#ifndef VSM3_TILING_H
#define VSM3_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
    BEGIN_TILING_DATA_DEF(VSM3TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, subDim1);
    TILING_DATA_FIELD_DEF(uint32_t, subDim2);
    TILING_DATA_FIELD_DEF(uint32_t, nlist1);
    TILING_DATA_FIELD_DEF(uint32_t, nlist2);
    TILING_DATA_FIELD_DEF(uint32_t, n);

    TILING_DATA_FIELD_DEF(uint32_t, nprobe2);
    TILING_DATA_FIELD_DEF(uint32_t, baseNum);
    TILING_DATA_FIELD_DEF(uint32_t, segmentSize);
    TILING_DATA_FIELD_DEF(uint32_t, segSizeVcMin);
    TILING_DATA_FIELD_DEF(uint32_t, segmentNum);
    TILING_DATA_FIELD_DEF(uint32_t, tmpMaskSize);

    TILING_DATA_FIELD_DEF(uint32_t, formerBlkNum);
    TILING_DATA_FIELD_DEF(uint32_t, probePerBlockFormer);
    TILING_DATA_FIELD_DEF(uint32_t, probePerBlockLatter);
    TILING_DATA_FIELD_DEF(uint32_t, sizeCodeWordUBBuffer);
    TILING_DATA_FIELD_DEF(uint32_t, sizeCodeWordL0BBuffer);
    TILING_DATA_FIELD_DEF(uint32_t, sizeCodeWordL1BBuffer);

    TILING_DATA_FIELD_DEF(uint32_t, cubeAlign);
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(VSM3, VSM3TilingData)
}

#endif

