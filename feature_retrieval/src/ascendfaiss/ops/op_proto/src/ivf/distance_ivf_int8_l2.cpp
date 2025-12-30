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


#include "distance_ivf_int8_l2.h"

#include <vector>
#include <string>
#include <iostream>

namespace ge {
IMPLEMT_VERIFIER(DistanceIVFInt8L2, DistanceIVFInt8L2Verify)
{
    DataType inputTypeX0 = op.GetInputDescByName("x0").GetDataType();
    DataType inputTypeX1 = op.GetInputDescByName("x1").GetDataType();
    DataType inputTypeX2 = op.GetInputDescByName("x2").GetDataType();
    DataType inputTypeX3 = op.GetInputDescByName("x3").GetDataType();
    if ((inputTypeX0 != inputTypeX1) || (inputTypeX2 != DT_INT32) || (inputTypeX3 != DT_UINT32)) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}


// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DistanceIVFInt8L2InferShape)
{
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
    dimY.push_back(dimsX2[0]);

    ge::Shape outputShape0 = ge::Shape(dimY);

    OutputDesc0.SetShape(outputShape0);
    OutputDesc0.SetDataType(DT_FLOAT16);
    OutputDesc0.SetFormat(inputFormat);
    op.UpdateOutputDesc("y0", OutputDesc0);

    std::vector<int64_t> dimMinY;
    dimMinY.push_back(dimsX0[0]);
    dimMinY.push_back((dimsX2[0] + 31) / 32 * 2); // ( ... + 31) / 32 for align, 2 sizeof of fp16

    ge::Shape outputShape1 = ge::Shape(dimMinY);

    OutputDesc1.SetShape(outputShape1);
    OutputDesc1.SetDataType(DT_FLOAT16);
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
COMMON_INFER_FUNC_REG(DistanceIVFInt8L2, DistanceIVFInt8L2InferShape);

// Registered verify function
VERIFY_FUNC_REG(DistanceIVFInt8L2, DistanceIVFInt8L2Verify);
} // namespace ge