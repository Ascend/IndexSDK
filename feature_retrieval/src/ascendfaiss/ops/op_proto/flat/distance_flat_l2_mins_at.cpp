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


#include "distance_flat_l2_mins_at.h"
#include "ascend_operator.h"

#include <vector>
#include <string>
#include <iostream>

namespace ge {
IMPLEMT_VERIFIER(DistanceFlatL2MinsAt, DistanceFlatL2MinsAtVerify)
{
    DataType inputTypeX0 = op.GetInputDescByName("x0").GetDataType();
    DataType inputTypeX1 = op.GetInputDescByName("x1").GetDataType();
    DataType inputTypeX2 = op.GetInputDescByName("x2").GetDataType();
    DataType inputTypeX3 = op.GetInputDescByName("x3").GetDataType();
    if ((inputTypeX0 != inputTypeX1) || (inputTypeX2 != inputTypeX3)) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}


// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DistanceFlatL2MinsAtInferShape)
{
    DataType inputDtype = op.GetInputDescByName("x0").GetDataType();
    Format inputFormat = op.GetInputDescByName("x0").GetFormat();
    Shape x0_shape = op.GetInputDescByName("x0").GetShape();
    Shape x1_shape = op.GetInputDescByName("x1").GetShape();

    TensorDesc outputDesc0 = op.GetOutputDescByName("y0");
    TensorDesc outputDesc1 = op.GetOutputDescByName("y1");
    TensorDesc outputDesc2 = op.GetOutputDescByName("y2");

    std::vector<int64_t> dimsX0 = x0_shape.GetDims();
    std::vector<int64_t> dimsX1 = x1_shape.GetDims();

    std::vector<int64_t> dimY;
    dimY.push_back(dimsX0[0] * dimsX0[2]);
    dimY.push_back(dimsX1[0] * dimsX1[2]);
    ge::Shape outputShape0 = ge::Shape(dimY);

    outputDesc0.SetShape(outputShape0);
    outputDesc0.SetDataType(inputDtype);
    outputDesc0.SetFormat(inputFormat);
    op.UpdateOutputDesc("y0", outputDesc0);

    std::vector<int64_t> dimMins;
    dimMins.push_back(dimsX0[0] * dimsX0[2]);
    dimMins.push_back(dimsX1[0] * dimsX1[2] / 32);
    ge::Shape outputShape1 = ge::Shape(dimMins);

    outputDesc1.SetShape(outputShape1);
    outputDesc1.SetDataType(inputDtype);
    outputDesc1.SetFormat(inputFormat);
    op.UpdateOutputDesc("y1", outputDesc1);

    std::vector<int64_t> dimVec2 { CORE_NUM, 16 };
    ge::Shape outputShape2 = ge::Shape(dimVec2);

    outputDesc2.SetShape(outputShape2);
    outputDesc2.SetDataType(DT_UINT16);
    outputDesc2.SetFormat(inputFormat);
    op.UpdateOutputDesc("y2", outputDesc2);

    return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DistanceFlatL2MinsAt, DistanceFlatL2MinsAtInferShape);

// Registered verify function
VERIFY_FUNC_REG(DistanceFlatL2MinsAt, DistanceFlatL2MinsAtVerify);
} // namespace ge