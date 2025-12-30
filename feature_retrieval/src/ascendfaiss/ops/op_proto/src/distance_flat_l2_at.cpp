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


#include "distance_flat_l2_at.h"
#include "ascend_operator.h"

#include <vector>
#include <string>
#include <iostream>

namespace ge {
IMPLEMT_VERIFIER(DistanceFlatL2At, DistanceFlatL2AtVerify)
{
    return GRAPH_SUCCESS;
}


// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(DistanceFlatL2AtInferShape)
{
    Format inputFormat = op.GetInputDescByName("x0").GetFormat();
    Shape x0_shape = op.GetInputDescByName("x0").GetShape();
    Shape x1_shape = op.GetInputDescByName("x1").GetShape();

    TensorDesc outputDesc0 = op.GetOutputDescByName("y0");
    TensorDesc outputDesc1 = op.GetOutputDescByName("y1");
    TensorDesc outputDesc2 = op.GetOutputDescByName("y2");
    TensorDesc outputDesc3 = op.GetOutputDescByName("y3");

    std::vector<int64_t> dimsX0 = x0_shape.GetDims();
    std::vector<int64_t> dimsX1 = x1_shape.GetDims();

    std::vector<int64_t> dimVec0 {CORE_NUM, 64, 256};
    ge::Shape outputShape0 = ge::Shape(dimVec0);

    outputDesc0.SetShape(outputShape0);
    outputDesc0.SetDataType(DT_FLOAT16);
    outputDesc0.SetFormat(inputFormat);
    op.UpdateOutputDesc("y0", outputDesc0);

    std::vector<int64_t> dimVec1 { CORE_NUM, 64 };
    ge::Shape outputShape1 = ge::Shape(dimVec1);

    outputDesc1.SetShape(outputShape1);
    outputDesc1.SetDataType(DT_INT16);
    outputDesc1.SetFormat(inputFormat);
    op.UpdateOutputDesc("y1", outputDesc1);

    std::vector<int64_t> dimVec2 { 1, };
    ge::Shape outputShape2 = ge::Shape(dimVec2);

    outputDesc2.SetShape(outputShape2);
    outputDesc2.SetDataType(DT_UINT16);
    outputDesc2.SetFormat(inputFormat);
    op.UpdateOutputDesc("y2", outputDesc2);

    std::vector<int64_t> dimVec3 { CORE_NUM, 16 };
    ge::Shape outputShape3 = ge::Shape(dimVec3);

    outputDesc3.SetShape(outputShape3);
    outputDesc3.SetDataType(DT_UINT16);
    outputDesc3.SetFormat(inputFormat);
    op.UpdateOutputDesc("y3", outputDesc3);

    return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(DistanceFlatL2At, DistanceFlatL2AtInferShape);

// Registered verify function
VERIFY_FUNC_REG(DistanceFlatL2At, DistanceFlatL2AtVerify);
} // namespace ge