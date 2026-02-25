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

#include <limits>
#include "kernel_operator.h"

#include "ascendc_ivfpq_search_distance_topk.h"
#include "ascendc_ivfpq_search_distance_topk_small.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void
ascendc_ivfpq_search_distance(GM_ADDR queryPQ, GM_ADDR codeBase, GM_ADDR codeOffset, GM_ADDR codeSize, GM_ADDR topk,
                              GM_ADDR labelBase, GM_ADDR labelOffset, GM_ADDR distResult, GM_ADDR topkIndex,
                              GM_ADDR topkValue, GM_ADDR topkLabelFinal, GM_ADDR topkValueFinal, GM_ADDR flag,
                              GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe tPipe;
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(0)) {
        IndexOps::AscendcIvfpqSearchDistanceTopK op;
        op.Init(queryPQ, codeBase, codeOffset, codeSize, topk, labelBase, labelOffset, distResult, topkIndex, topkValue,
                topkLabelFinal, topkValueFinal, flag, workspace, &tiling_data, &tPipe);
        op.Process();
        op.ProcessMerge();
    } else if (TILING_KEY_IS(1)) {
        IndexOps::AscendcIvfpqSearchDistanceTopKSmall op;
        op.Init(queryPQ, codeBase, codeOffset, codeSize, topk, labelBase, labelOffset, distResult, topkIndex, topkValue,
                topkLabelFinal, topkValueFinal, flag, workspace, &tiling_data, &tPipe);
        op.Process();
        op.ProcessMerge();
    }
}