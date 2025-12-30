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


#include "distance_ivf_sq8_l2.h"

#include <vector>
#include <string>
#include <iostream>

namespace ge {
    IMPLEMT_VERIFIER(DistanceIVFSQ8L2, DistanceIVFSQ8L2Verify)
    {
        DataType inputTypeX0 = op.GetInputDescByName("x0").GetDataType();
        DataType inputTypeX3 = op.GetInputDescByName("x3").GetDataType();

        if ((inputTypeX0 != inputTypeX3)) {
            return GRAPH_FAILED;
        }
        return GRAPH_SUCCESS;
    }


    // Obtains the processing function of the output tensor description.
    IMPLEMT_COMMON_INFERFUNC(DistanceIVFSQ8L2InferShape)
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

        int min_batch = 32;
        if (dims_x0[1] > 128) {
            min_batch = 16;
        }

        std::vector<int64_t> dim_min_y;
        dim_min_y.push_back(dims_x0[0]);
        dim_min_y.push_back((dims_x2[0] + min_batch - 1) / min_batch * 2);

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
    COMMON_INFER_FUNC_REG(DistanceIVFSQ8L2, DistanceIVFSQ8L2InferShape);

    // Registered verify function
    VERIFY_FUNC_REG(DistanceIVFSQ8L2, DistanceIVFSQ8L2Verify);
} // namespace ge