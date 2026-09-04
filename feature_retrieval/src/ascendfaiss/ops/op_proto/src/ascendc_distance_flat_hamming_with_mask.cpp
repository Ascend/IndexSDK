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

#include "ascendc_distance_flat_hamming_with_mask.h"

namespace ge
{
IMPLEMT_VERIFIER(AscendcDistanceFlatHammingWithMask, AscendcDistanceFlatHammingWithMaskVerify)
{
    DataType queryType = op.GetInputDescByName("x0").GetDataType();
    DataType baseType = op.GetInputDescByName("x1").GetDataType();
    DataType actualSizeType = op.GetInputDescByName("x2").GetDataType();
    DataType maskType = op.GetInputDescByName("x3").GetDataType();
    DataType baseMaskType = op.GetInputDescByName("x4").GetDataType();
    if (queryType != DT_UINT8 || baseType != DT_UINT8 || actualSizeType != DT_UINT32 || maskType != DT_UINT8 ||
        baseMaskType != DT_UINT8)
    {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AscendcDistanceFlatHammingWithMaskInferShape) { return GRAPH_SUCCESS; }

COMMON_INFER_FUNC_REG(AscendcDistanceFlatHammingWithMask, AscendcDistanceFlatHammingWithMaskInferShape);
VERIFY_FUNC_REG(AscendcDistanceFlatHammingWithMask, AscendcDistanceFlatHammingWithMaskVerify);
}  // namespace ge
