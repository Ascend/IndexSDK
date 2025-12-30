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


#include "distance_flat_subcenters.h"
#include "ascend_operator.h"

#include <vector>
#include <string>
#include <iostream>

namespace ge {
IMPLEMT_VERIFIER(DistanceFlatSubcenters, DistanceFlatSubcentersVerify)
{
    DataType inputTypeX0 = op.GetInputDescByName("x0").GetDataType();
    DataType inputTypeX1 = op.GetInputDescByName("x1").GetDataType();
    DataType inputTypeX2 = op.GetInputDescByName("x2").GetDataType();
    if ((inputTypeX0 != inputTypeX1) || (inputTypeX0 != inputTypeX2) || (inputTypeX1 != inputTypeX2)) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}


// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DistanceFlatSubcentersInferShape)
{
    DataType inputDtype = op.GetInputDescByName("x0").GetDataType();
    Format inputFormat = op.GetInputDescByName("x0").GetFormat();

    Shape x0_shape = op.GetInputDescByName("x0").GetShape();
    Shape x2_shape = op.GetInputDescByName("x2").GetShape();
    Shape x3_shape = op.GetInputDescByName("x3").GetShape();
    TensorDesc outputDesc0 = op.GetOutputDescByName("y0");
    TensorDesc outputDesc1 = op.GetOutputDescByName("y1");

    std::vector<int64_t> dimsX0 = x0_shape.GetDims();
    std::vector<int64_t> dimsX2 = x2_shape.GetDims();
    std::vector<int64_t> dimsX3 = x3_shape.GetDims();

    std::vector<int64_t> dimY;
    dimY.push_back(dimsX0[0]);
    dimY.push_back(dimsX3[1]);
    dimY.push_back(dimsX2[1]);

    ge::Shape outputShape0 = ge::Shape(dimY);

    outputDesc0.SetShape(outputShape0);
    outputDesc0.SetDataType(inputDtype);
    outputDesc0.SetFormat(inputFormat);
    op.UpdateOutputDesc("y0", outputDesc0);

    std::vector<int64_t> dimVec1 { CORE_NUM, 16 };
    ge::Shape outputShape1 = ge::Shape(dimVec1);

    outputDesc1.SetShape(outputShape1);
    outputDesc1.SetDataType(DT_UINT16);
    outputDesc1.SetFormat(inputFormat);
    op.UpdateOutputDesc("y1", outputDesc1);

    return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DistanceFlatSubcenters, DistanceFlatSubcentersInferShape);

// Registered verify function
VERIFY_FUNC_REG(DistanceFlatSubcenters, DistanceFlatSubcentersVerify);
} // namespace ge
