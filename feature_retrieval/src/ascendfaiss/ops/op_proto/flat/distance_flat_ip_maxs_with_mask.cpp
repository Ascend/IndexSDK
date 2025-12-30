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

 
#include "distance_flat_ip_maxs_with_mask.h"
#include "ascend_operator.h"
 
#include <vector>
#include <string>
#include <iostream>
 
namespace ge {
IMPLEMT_VERIFIER(DistanceFlatIPMaxsWithMask, DistanceFlatIPMaxsWithMaskVerify)
{
    DataType inputTypeX0 = op.GetInputDescByName("x0").GetDataType();
    DataType inputTypeX1 = op.GetInputDescByName("x1").GetDataType();
    if ((inputTypeX0 != inputTypeX1)) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}
 
 
// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DistanceFlatIPMaxsWithMaskInferShape)
{
    return GRAPH_SUCCESS;
}
 
// Registered inferfunction
COMMON_INFER_FUNC_REG(DistanceFlatIPMaxsWithMask, DistanceFlatIPMaxsWithMaskInferShape);
 
// Registered verify function
VERIFY_FUNC_REG(DistanceFlatIPMaxsWithMask, DistanceFlatIPMaxsWithMaskVerify);
} // namespace ge