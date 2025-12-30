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


#include "distance_int8_l2_mins.h"

#include <vector>
#include <string>
#include <iostream>

namespace ge {
IMPLEMT_VERIFIER(DistanceInt8L2Mins, DistanceInt8L2MinsVerify)
{
    DataType inputTypeX0 = op.GetInputDescByName("x0").GetDataType();
    DataType inputTypeX2 = op.GetInputDescByName("x2").GetDataType();
    DataType inputTypeX3 = op.GetInputDescByName("x3").GetDataType();
    DataType inputTypeX4 = op.GetInputDescByName("x4").GetDataType();
    if ((inputTypeX0 != inputTypeX2) || (inputTypeX3 != DT_INT32) || (inputTypeX4 != DT_UINT32)) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}


// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DistanceInt8L2MinsInferShape)
{
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
    OutputDesc0.SetDataType(DT_FLOAT16);
    OutputDesc0.SetFormat(inputFormat);
    op.UpdateOutputDesc("y0", OutputDesc0);

    std::vector<int64_t> dimMinY;
    dimMinY.push_back(dimsX0[0]);
    dimMinY.push_back((dimsX3[0] + 63) / 64 * 2); // ( ... + 63) / 64 for align, 2 sizeof of fp16

    ge::Shape outputShape1 = ge::Shape(dimMinY);

    OutputDesc1.SetShape(outputShape1);
    OutputDesc1.SetDataType(DT_FLOAT16);
    OutputDesc1.SetFormat(inputFormat);
    op.UpdateOutputDesc("y1", OutputDesc1);

    std::vector<int64_t> dimVec1 { 16, 16 };
    ge::Shape outputShape2 = ge::Shape(dimVec1);

    OutputDesc2.SetShape(outputShape2);
    OutputDesc2.SetDataType(DT_UINT16);
    OutputDesc2.SetFormat(inputFormat);
    op.UpdateOutputDesc("y2", OutputDesc2);

    return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DistanceInt8L2Mins, DistanceInt8L2MinsInferShape);

// Registered verify function
VERIFY_FUNC_REG(DistanceInt8L2Mins, DistanceInt8L2MinsVerify);
} // namespace ge