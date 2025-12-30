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


#include "distance_sq8_l2_mins.h"

#include <vector>
#include <string>
#include <iostream>

namespace ge {
    IMPLEMT_VERIFIER(DistanceSQ8L2Mins, DistanceSQ8L2MinsVerify)
    {
        DataType inputTypeX0 = op.GetInputDescByName("x0").GetDataType();
        DataType inputTypeX4 = op.GetInputDescByName("x4").GetDataType();

        if ((inputTypeX0 != inputTypeX4)) {
            return GRAPH_FAILED;
        }
        return GRAPH_SUCCESS;
    }


    // Obtains the processing function of the output tensor description.
    IMPLEMT_COMMON_INFERFUNC(DistanceSQ8L2MinsInferShape)
    {
        DataType input_dtype = op.GetInputDescByName("x0").GetDataType();
        Format input_format = op.GetInputDescByName("x0").GetFormat();

        Shape x0_shape = op.GetInputDescByName("x0").GetShape();
        Shape x2_shape = op.GetInputDescByName("x2").GetShape();
        TensorDesc OutputDesc0 = op.GetOutputDescByName("y0");
        TensorDesc OutputDesc1 = op.GetOutputDescByName("y1");
        TensorDesc OutputDesc2 = op.GetOutputDescByName("y2");

        std::vector<int64_t> dims_x0 = x0_shape.GetDims();
        std::vector<int64_t> dims_x2 = x2_shape.GetDims();

        std::vector<int64_t> dim_y;
        dim_y.push_back(dims_x0[0]);
        dim_y.push_back(dims_x2[0]);

        ge::Shape outputShape = ge::Shape(dim_y);

        OutputDesc0.SetShape(outputShape);
        OutputDesc0.SetDataType(input_dtype);
        OutputDesc0.SetFormat(input_format);
        op.UpdateOutputDesc("y0", OutputDesc0);

        std::vector<int64_t> dim_min_y;
        dim_min_y.push_back(dims_x0[0]);
        dim_min_y.push_back((dims_x2[0] + 63) / 64 * 2);

        ge::Shape outputShape1 = ge::Shape(dim_min_y);

        OutputDesc1.SetShape(outputShape1);
        OutputDesc1.SetDataType(input_dtype);
        OutputDesc1.SetFormat(input_format);
        op.UpdateOutputDesc("y1", OutputDesc1);


        std::vector<int64_t> dimVec2 { 16, 16 };
        ge::Shape outputShape2 = ge::Shape(dimVec2);

        OutputDesc2.SetShape(outputShape2);
        OutputDesc2.SetDataType(DT_UINT16);
        OutputDesc2.SetFormat(input_format);
        op.UpdateOutputDesc("y2", OutputDesc2);

        return GRAPH_SUCCESS;
    } // namespace ge

    // Registered inferfunction
    COMMON_INFER_FUNC_REG(DistanceSQ8L2Mins, DistanceSQ8L2MinsInferShape);

    // Registered verify function
    VERIFY_FUNC_REG(DistanceSQ8L2Mins, DistanceSQ8L2MinsVerify);
} // namespace ge