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


#include "int8_l2_norm.h"

#include <vector>
#include <string>
#include <iostream>

namespace ge {
    IMPLEMT_VERIFIER(Int8L2Norm, Int8L2NormVerify)
    {
        return GRAPH_SUCCESS;
    }


    // Obtains the processing function of the output tensor description.
    IMPLEMT_COMMON_INFERFUNC(Int8L2NormInferShape)
    {
        DataType x1_dtype = op.GetInputDescByName("x1").GetDataType();
        Format x1_format = op.GetInputDescByName("x1").GetFormat();

        Shape x0_shape = op.GetInputDescByName("x0").GetShape();
        TensorDesc outputDesc0 = op.GetOutputDescByName("y0");

        std::vector<int64_t> dims_x0 = x0_shape.GetDims();

        std::vector<int64_t> dim_y0;
        dim_y0.push_back(dims_x0[0]);

        ge::Shape outputShape0 = ge::Shape(dim_y0);

        outputDesc0.SetShape(outputShape0);
        outputDesc0.SetDataType(x1_dtype);
        outputDesc0.SetFormat(x1_format);
        op.UpdateOutputDesc("y0", outputDesc0);

        return GRAPH_SUCCESS;
    } // namespace ge

    // Registered inferfunction
    COMMON_INFER_FUNC_REG(Int8L2Norm, Int8L2NormInferShape);

    // Registered verify function
    VERIFY_FUNC_REG(Int8L2Norm, Int8L2NormVerify);
} // namespace ge