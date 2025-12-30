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


#ifndef ASCENDC_L2_NORM_TILING_H
#define ASCENDC_L2_NORM_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

/*
 * 输入会传入各个数据地址，但是数据的维度需要通过tiling传递
 * 1. dim 特征维度
 * 2. vecCoreNum AIV数量
 */
namespace optiling {
BEGIN_TILING_DATA_DEF(AscendcL2NormTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, dim);
    TILING_DATA_FIELD_DEF(uint32_t, vecCoreNum);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AscendcL2Norm, AscendcL2NormTilingData)
}
#endif // ASCENDC_L2_NORM_TILING_H