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


#include "distance_ivf_int8_cos.h"

#include <vector>
#include <string>
#include <iostream>

namespace ge {
    IMPLEMT_VERIFIER(DistanceIVFInt8Cos, DistanceIVFInt8CosVerify)
    {
        DataType inputTypeX0 = op.GetInputDescByName("x0").GetDataType();
        DataType inputTypeX1 = op.GetInputDescByName("x1").GetDataType();

        if ((inputTypeX0 != inputTypeX1)) {
            return GRAPH_FAILED;
        }
        return GRAPH_SUCCESS;
    }


    // Obtains the processing function of the output tensor description.
    IMPLEMT_COMMON_INFERFUNC(DistanceIVFInt8CosInferShape)
    {
        DataType x2_dtype = op.GetInputDescByName("x2").GetDataType();
        Format x2_format = op.GetInputDescByName("x2").GetFormat();

        Shape x0_shape = op.GetInputDescByName("x0").GetShape();
        Shape x3_shape = op.GetInputDescByName("x3").GetShape();
        TensorDesc outputDesc0 = op.GetOutputDescByName("y0");
        TensorDesc outputDesc1 = op.GetOutputDescByName("y1");
        TensorDesc outputDesc2 = op.GetOutputDescByName("y2");

        std::vector<int64_t> dims_x0 = x0_shape.GetDims();
        std::vector<int64_t> dims_x3 = x3_shape.GetDims();

        std::vector<int64_t> dim_y0;
        dim_y0.push_back(dims_x0[0]);
        dim_y0.push_back(dims_x3[0]);

        ge::Shape outputShape0 = ge::Shape(dim_y0);

        outputDesc0.SetShape(outputShape0);
        outputDesc0.SetDataType(x2_dtype);
        outputDesc0.SetFormat(x2_format);
        op.UpdateOutputDesc("y0", outputDesc0);

        std::vector<int64_t> dim_y1;
        dim_y1.push_back(dims_x0[0]);
        dim_y1.push_back((dims_x3[0] + 31) / 32 * 2);

        ge::Shape outputShape1 = ge::Shape(dim_y1);

        outputDesc1.SetShape(outputShape1);
        outputDesc1.SetDataType(x2_dtype);
        outputDesc1.SetFormat(x2_format);
        op.UpdateOutputDesc("y1", outputDesc1);

        std::vector<int64_t> dimVec2 { 16, 16 };
        ge::Shape outputShape2 = ge::Shape(dimVec2);

        outputDesc2.SetShape(outputShape2);
        outputDesc2.SetDataType(DT_UINT16);
        outputDesc2.SetFormat(x2_format);
        op.UpdateOutputDesc("y2", outputDesc2);

        return GRAPH_SUCCESS;
    } // namespace ge

    // Registered inferfunction
    COMMON_INFER_FUNC_REG(DistanceIVFInt8Cos, DistanceIVFInt8CosInferShape);

    // Registered verify function
    VERIFY_FUNC_REG(DistanceIVFInt8Cos, DistanceIVFInt8CosVerify);
} // namespace ge