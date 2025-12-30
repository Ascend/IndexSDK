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


#ifndef VSTAR_COMPUTE_L1_TILING_H
#define VSTAR_COMPUTE_L1_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(VstarComputeL1TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, dim);               // 数据维度
    TILING_DATA_FIELD_DEF(uint32_t, subSpaceDim);       // 子空间维度
    TILING_DATA_FIELD_DEF(uint32_t, nlist);             // 总的粗桶个数, 限制nlist%16==0
    TILING_DATA_FIELD_DEF(uint32_t, blockSize);         // 单个block计算的粗桶个数
    TILING_DATA_FIELD_DEF(uint32_t, nq);
    TILING_DATA_FIELD_DEF(uint32_t, cbTileSizeL1);      // 要求能被subSpaceDim整除
    TILING_DATA_FIELD_DEF(uint32_t, cbTileSizeB2);      // 要求能被16整除
    TILING_DATA_FIELD_DEF(uint32_t, cbLoopsL1);      // 要求能被16整除
    TILING_DATA_FIELD_DEF(uint32_t, cbLoopsB2);      // 要求能被16整除
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(VstarComputeL1, VstarComputeL1TilingData)
}

#endif