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


//
// Created by mxIndex team on 2023/3/24.
//
#include "distance_flat_ip_maxs_batch.h"
#include "ascend_operator.h"

#include <vector>
#include <string>
#include <iostream>

namespace ge {
IMPLEMT_VERIFIER(DistanceFlatIPMaxsBatch, DistanceFlatIPMaxsBatchVerify)
{
    DataType inputTypeX0 = op.GetInputDescByName("x0").GetDataType();
    DataType inputTypeX1 = op.GetInputDescByName("x1").GetDataType();
    DataType inputTypeX2 = op.GetInputDescByName("x2").GetDataType();

    if ((inputTypeX0 != inputTypeX2) || (inputTypeX1 != DT_UINT8)) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}


// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DistanceFlatIPMaxsBatchInferShape)
{
    DataType inputDtype = op.GetInputDescByName("x0").GetDataType();
    Format inputFormat = op.GetInputDescByName("x0").GetFormat();

    Shape x0_shape = op.GetInputDescByName("x0").GetShape();
    Shape x2_shape = op.GetInputDescByName("x2").GetShape();
    TensorDesc OutputDesc0 = op.GetOutputDescByName("y0");
    TensorDesc OutputDesc1 = op.GetOutputDescByName("y1");
    TensorDesc OutputDesc2 = op.GetOutputDescByName("y2");

    std::vector<int64_t> dimsX0 = x0_shape.GetDims();
    std::vector<int64_t> dimsX2 = x2_shape.GetDims();

    std::vector<int64_t> dimY;
    dimY.push_back(dimsX0[0]);
    dimY.push_back(dimsX2[0] * 16);

    ge::Shape outputShape0 = ge::Shape(dimY);

    OutputDesc0.SetShape(outputShape0);
    OutputDesc0.SetDataType(inputDtype);
    OutputDesc0.SetFormat(inputFormat);
    op.UpdateOutputDesc("y0", OutputDesc0);

    std::vector<int64_t> dim_max_y;
    dim_max_y.push_back(dimsX0[0]);
    dim_max_y.push_back((dimsX2[0] * 16 + 31) / 32 * 2);

    ge::Shape outputShape1 = ge::Shape(dim_max_y);

    OutputDesc1.SetShape(outputShape1);
    OutputDesc1.SetDataType(inputDtype);
    OutputDesc1.SetFormat(inputFormat);
    op.UpdateOutputDesc("y1", OutputDesc1);

    std::vector<int64_t> dimVec2 { CORE_NUM, 16 };
    ge::Shape outputShape2 = ge::Shape(dimVec2);

    OutputDesc2.SetShape(outputShape2);
    OutputDesc2.SetDataType(DT_UINT16);
    OutputDesc2.SetFormat(inputFormat);
    op.UpdateOutputDesc("y2", OutputDesc2);

    return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DistanceFlatIPMaxsBatch, DistanceFlatIPMaxsBatchInferShape);

// Registered verify function
VERIFY_FUNC_REG(DistanceFlatIPMaxsBatch, DistanceFlatIPMaxsBatchVerify);
} // namespace ge