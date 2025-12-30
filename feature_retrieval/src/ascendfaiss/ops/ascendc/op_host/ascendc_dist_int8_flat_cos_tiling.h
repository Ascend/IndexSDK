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


#ifndef ASCENDC_DIST_INT8_COS_TILING_H
#define ASCENDC_DIST_INT8_COS_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

/*
 * 输入会传入各个数据地址，但是数据的维度需要通过tiling传递
 * 1. queryNum 查询数量
 * 2. dim 特征维度
 * 3. baseBlockSize 底库容量
 * 4. vecCoreNum AIV数量
 * 5. onceComputeBaseNum 单核单次底库计算量
 */
namespace optiling {
BEGIN_TILING_DATA_DEF(AscendcDistInt8FlatCosTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, queryNum);
    TILING_DATA_FIELD_DEF(uint32_t, dim);
    TILING_DATA_FIELD_DEF(uint32_t, baseBlockSize);
    TILING_DATA_FIELD_DEF(uint32_t, vecCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, onceComputeBaseNum);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AscendcDistInt8FlatCos, AscendcDistInt8FlatCosTilingData)
}
#endif // ASCENDC_DIST_INT8_COS_TILING_H