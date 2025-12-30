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


#include "distance_ivf_sq8_ip4.h"

#include <vector>
#include <string>
#include <iostream>

namespace ge {
    IMPLEMT_VERIFIER(DistanceIVFSQ8IP4, DistanceIVFSQ8IP4Verify)
    {
        DataType inputTypeX1 = op.GetInputDescByName("x1").GetDataType();
        DataType inputTypeX2 = op.GetInputDescByName("x2").GetDataType();
        DataType inputTypeX3 = op.GetInputDescByName("x3").GetDataType();
        DataType inputTypeX4 = op.GetInputDescByName("x4").GetDataType();

        if ((inputTypeX1 != inputTypeX2) || (inputTypeX3 != inputTypeX4)) {
            return GRAPH_FAILED;
        }
        return GRAPH_SUCCESS;
    }


    // Obtains the processing function of the output tensor description.
    IMPLEMT_COMMON_INFERFUNC(DistanceIVFSQ8IP4InferShape)
    {
        DataType x0_dtype = op.GetInputDescByName("x0").GetDataType();
        Format x0_format = op.GetInputDescByName("x0").GetFormat();

        Shape x0_shape = op.GetInputDescByName("x0").GetShape();
        Shape x1_shape = op.GetInputDescByName("x1").GetShape();
        TensorDesc outputDesc0 = op.GetOutputDescByName("y0");
        TensorDesc outputDesc1 = op.GetOutputDescByName("y1");
        TensorDesc outputDesc2 = op.GetOutputDescByName("y2");

        std::vector<int64_t> dims_x0 = x0_shape.GetDims();
        std::vector<int64_t> dims_x1 = x1_shape.GetDims();

        std::vector<int64_t> dim_y0;
        dim_y0.push_back(4 * dims_x1[0] * 16);

        ge::Shape outputShape0 = ge::Shape(dim_y0);

        outputDesc0.SetShape(outputShape0);
        outputDesc0.SetDataType(x0_dtype);
        outputDesc0.SetFormat(x0_format);
        op.UpdateOutputDesc("y0", outputDesc0);

        std::vector<int64_t> dim_max_y;
        dim_max_y.push_back(4 * dims_x1[0] * 16 / 16 * 2);

        ge::Shape outputShape1 = ge::Shape(dim_max_y);

        outputDesc1.SetShape(outputShape1);
        outputDesc1.SetDataType(x0_dtype);
        outputDesc1.SetFormat(x0_format);
        op.UpdateOutputDesc("y1", outputDesc1);

        std::vector<int64_t> dimVec2 { 2, 32 };
        ge::Shape outputShape2 = ge::Shape(dimVec2);

        outputDesc2.SetShape(outputShape2);
        outputDesc2.SetDataType(DT_UINT16);
        outputDesc2.SetFormat(x0_format);
        op.UpdateOutputDesc("y2", outputDesc2);

        return GRAPH_SUCCESS;
    } // namespace ge

    // Registered inferfunction
    COMMON_INFER_FUNC_REG(DistanceIVFSQ8IP4, DistanceIVFSQ8IP4InferShape);

    // Registered verify function
    VERIFY_FUNC_REG(DistanceIVFSQ8IP4, DistanceIVFSQ8IP4Verify);
} // namespace ge