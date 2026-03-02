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

#ifndef DISTANCE_IVF_FLAT_IP_FP32_TILING_H
#define DISTANCE_IVF_FLAT_IP_FP32_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/matrix/matmul_tiling.h"

#define ASCENDC_RETURN_IF_NOT(X, ERRCODE)                                                                   \
    do                                                                                                              \
    {                                                                                                               \
        if (!(X)) {                                                                                                 \
            return ERRCODE;                                                                                         \
        }                                                                                                           \
    } while (false)

namespace optiling {
BEGIN_TILING_DATA_DEF(DistanceIVFFlatIpFP32TilingData)
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, gemm_qb_tiling);
    TILING_DATA_FIELD_DEF(int32_t, dim_len);  // dim大小
    TILING_DATA_FIELD_DEF(int32_t, burst_len); // vcmax/vcmmin长度
    TILING_DATA_FIELD_DEF(int32_t, per_loop_process_len); // 单次循环处理长度
    TILING_DATA_FIELD_DEF(int32_t, code_list_num); // 待处理分桶数
    TILING_DATA_FIELD_DEF(int32_t, code_num); // 底库长度
    TILING_DATA_FIELD_DEF(int32_t, buffer_size); // 核内buffer大小
    TILING_DATA_FIELD_DEF(int32_t, max_res_stride); // ascendc指令单次循环步长
    TILING_DATA_FIELD_DEF(int32_t, max_res_inner_stride); // ascendc指令tiling函数参数
    TILING_DATA_FIELD_DEF(int32_t, aiv_num); // 核数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DistanceIVFFlatIpFP32, DistanceIVFFlatIpFP32TilingData)
}
#endif