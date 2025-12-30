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


#include "distance_compute_flat_min64.h"

#include <vector>
#include <string>
#include <iostream>

namespace ge {
IMPLEMT_VERIFIER(DistanceComputeFlatMin64, DistanceComputeFlatMin64Verify)
{
    DataType inputTypeX0 = op.GetInputDescByName("x0").GetDataType();
    DataType inputTypeX1 = op.GetInputDescByName("x1").GetDataType();
    DataType inputTypeX2 = op.GetInputDescByName("x2").GetDataType();
    DataType inputTypeX3 = op.GetInputDescByName("x3").GetDataType();
    if ((inputTypeX0 != inputTypeX2) || (inputTypeX0 != inputTypeX3) || (inputTypeX2 != inputTypeX3) ||
        (inputTypeX1 != DT_UINT8)) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}


// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DistanceComputeFlatMin64InferShape)
{
    DataType inputDtype = op.GetInputDescByName("x0").GetDataType();
    Format inputFormat = op.GetInputDescByName("x0").GetFormat();

    Shape x0_shape = op.GetInputDescByName("x0").GetShape();
    Shape x3_shape = op.GetInputDescByName("x3").GetShape();
    TensorDesc OutputDesc0 = op.GetOutputDescByName("y0");
    TensorDesc OutputDesc1 = op.GetOutputDescByName("y1");
    TensorDesc OutputDesc2 = op.GetOutputDescByName("y2");

    std::vector<int64_t> dimsX0 = x0_shape.GetDims();
    std::vector<int64_t> dimsX3 = x3_shape.GetDims();

    std::vector<int64_t> dimY;
    dimY.push_back(dimsX0[0]);
    dimY.push_back(dimsX3[0]);

    ge::Shape outputShape0 = ge::Shape(dimY);

    OutputDesc0.SetShape(outputShape0);
    OutputDesc0.SetDataType(inputDtype);
    OutputDesc0.SetFormat(inputFormat);
    op.UpdateOutputDesc("y0", OutputDesc0);

    std::vector<int64_t> dim_min_y;
    dim_min_y.push_back(dimsX0[0]);
    dim_min_y.push_back((dimsX3[0] + 63) / 64 * 2);

    ge::Shape outputShape1 = ge::Shape(dim_min_y);

    OutputDesc1.SetShape(outputShape1);
    OutputDesc1.SetDataType(inputDtype);
    OutputDesc1.SetFormat(inputFormat);
    op.UpdateOutputDesc("y1", OutputDesc1);

    std::vector<int64_t> dimVec2 { 16, 16 };
    ge::Shape outputShape2 = ge::Shape(dimVec2);

    OutputDesc2.SetShape(outputShape2);
    OutputDesc2.SetDataType(DT_UINT16);
    OutputDesc2.SetFormat(inputFormat);
    op.UpdateOutputDesc("y2", OutputDesc2);

    return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DistanceComputeFlatMin64, DistanceComputeFlatMin64InferShape);

// Registered verify function
VERIFY_FUNC_REG(DistanceComputeFlatMin64, DistanceComputeFlatMin64Verify);
} // namespace ge