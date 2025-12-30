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


#include "linear_transform.h"

#include <vector>
#include <string>
#include <iostream>

namespace ge {
IMPLEMT_VERIFIER(LinearTransform, LinearTransformVerify)
{
    DataType inputTypeX0 = op.GetInputDescByName("x0").GetDataType();
    DataType inputTypeX1 = op.GetInputDescByName("x1").GetDataType();
    if (inputTypeX0 != inputTypeX1) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}


// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(LinearTransformInferShape)
{
    DataType input_dtype = op.GetInputDescByName("x0").GetDataType();
    Format input_format = op.GetInputDescByName("x0").GetFormat();

    Shape x0_shape = op.GetInputDescByName("x0").GetShape();
    Shape x2_shape = op.GetInputDescByName("x2").GetShape();
    TensorDesc OutputDesc = op.GetOutputDescByName("y");

    std::vector<int64_t> dims_x0 = x0_shape.GetDims();
    std::vector<int64_t> dims_x2 = x2_shape.GetDims();

    std::vector<int64_t> dim_y;
    dim_y.push_back(dims_x0[0]);
    dim_y.push_back(dims_x2[0]);

    ge::Shape outputShape = ge::Shape(dim_y);

    OutputDesc.SetShape(outputShape);
    OutputDesc.SetDataType(input_dtype);
    OutputDesc.SetFormat(input_format);
    op.UpdateOutputDesc("y", OutputDesc);

    return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(LinearTransform, LinearTransformInferShape);

// Registered verify function
VERIFY_FUNC_REG(LinearTransform, LinearTransformVerify);
} // namespace ge