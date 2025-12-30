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


#include "distance_table_build.h"

#include <vector>
#include <string>
#include <iostream>

namespace ge {
IMPLEMT_VERIFIER(DistanceTableBuild, DistanceTableBuildVerify)
{
    DataType inputTypeX0 = op.GetInputDescByName("x0").GetDataType();
    DataType inputTypeX1 = op.GetInputDescByName("x1").GetDataType();
    DataType inputTypeX3 = op.GetInputDescByName("x3").GetDataType();
    if ((inputTypeX0 != inputTypeX1) || (inputTypeX0 != inputTypeX3) || (inputTypeX1 != inputTypeX3)) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}


// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DistanceTableBuildInferShape)
{
    DataType inputDtype = op.GetInputDescByName("x0").GetDataType();
    Format inputFormat = op.GetInputDescByName("x0").GetFormat();

    Shape x0Shape = op.GetInputDescByName("x0").GetShape();
    Shape x1Shape = op.GetInputDescByName("x1").GetShape();
    Shape x2Shape = op.GetInputDescByName("x2").GetShape();
    TensorDesc OutputDesc0 = op.GetOutputDescByName("y0");
    TensorDesc OutputDesc1 = op.GetOutputDescByName("y1");

    std::vector<int64_t> dimsX0 = x0Shape.GetDims();
    std::vector<int64_t> dimsX1 = x1Shape.GetDims();
    std::vector<int64_t> dimsX2 = x2Shape.GetDims();

    int64_t dims_m = dimsX1[1] * dimsX1[0] / dimsX0[1];

    std::vector<int64_t> dimY;
    dimY.push_back(dimsX0[0]);
    dimY.push_back(dimsX2[1]);
    dimY.push_back(dimsX1[0]);
    dimY.push_back(dims_m);

    ge::Shape outputShape = ge::Shape(dimY);

    OutputDesc0.SetShape(outputShape);
    OutputDesc0.SetDataType(inputDtype);
    OutputDesc0.SetFormat(inputFormat);
    op.UpdateOutputDesc("y0", OutputDesc0);

    std::vector<int64_t> dimVec1 { 16 };
    ge::Shape outputShape1 = ge::Shape(dimVec1);

    OutputDesc1.SetShape(outputShape1);
    OutputDesc1.SetDataType(DT_UINT16);
    OutputDesc1.SetFormat(inputFormat);
    op.UpdateOutputDesc("y1", OutputDesc1);

    return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DistanceTableBuild, DistanceTableBuildInferShape);

// Registered verify function
VERIFY_FUNC_REG(DistanceTableBuild, DistanceTableBuildVerify);
} // namespace ge