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


#include "distance_ivf_int8_cos_l1.h"

#include <vector>
#include <string>
#include <iostream>

namespace ge {
    IMPLEMT_VERIFIER(DistanceIVFInt8CosL1, DistanceIVFInt8CosL1Verify)
    {
        DataType inputTypeX0 = op.GetInputDescByName("x0").GetDataType();
        DataType inputTypeX1 = op.GetInputDescByName("x1").GetDataType();

        if ((inputTypeX0 != inputTypeX1)) {
            return GRAPH_FAILED;
        }
        return GRAPH_SUCCESS;
    }


    // Obtains the processing function of the output tensor description.
    IMPLEMT_COMMON_INFERFUNC(DistanceIVFInt8CosL1InferShape)
    {
        DataType x2_dtype = op.GetInputDescByName("x2").GetDataType();
        Format x2_format = op.GetInputDescByName("x2").GetFormat();

        Shape x0_shape = op.GetInputDescByName("x0").GetShape();
        Shape x3_shape = op.GetInputDescByName("x3").GetShape();
        TensorDesc outputDesc0 = op.GetOutputDescByName("y0");
        TensorDesc outputDesc1 = op.GetOutputDescByName("y1");

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

        std::vector<int64_t> dimVec1 { 32 };
        ge::Shape outputShape1 = ge::Shape(dimVec1);

        outputDesc1.SetShape(outputShape1);
        outputDesc1.SetDataType(DT_UINT16);
        outputDesc1.SetFormat(x2_format);
        op.UpdateOutputDesc("y1", outputDesc1);

        return GRAPH_SUCCESS;
    } // namespace ge

    // Registered inferfunction
    COMMON_INFER_FUNC_REG(DistanceIVFInt8CosL1, DistanceIVFInt8CosL1InferShape);

    // Registered verify function
    VERIFY_FUNC_REG(DistanceIVFInt8CosL1, DistanceIVFInt8CosL1Verify);
} // namespace ge