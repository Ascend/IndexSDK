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


#include "distance_ivf_sq8_ipx.h"
#include "ascend_operator.h"

#include <vector>
#include <string>
#include <iostream>

namespace ge {
    IMPLEMT_VERIFIER(DistanceIVFSQ8IPX, DistanceIVFSQ8IPXVerify)
    {
        DataType inputTypeX0 = op.GetInputDescByName("x0").GetDataType();
        DataType inputTypeX3 = op.GetInputDescByName("x3").GetDataType();

        if ((inputTypeX0 != inputTypeX3)) {
            return GRAPH_FAILED;
        }
        return GRAPH_SUCCESS;
    }


    // Obtains the processing function of the output tensor description.
    IMPLEMT_COMMON_INFERFUNC(DistanceIVFSQ8IPXInferShape)
    {
        DataType x0_dtype = op.GetInputDescByName("x0").GetDataType();
        Format x0_format = op.GetInputDescByName("x0").GetFormat();

        Shape x0_shape = op.GetInputDescByName("x0").GetShape();
        Shape x2_shape = op.GetInputDescByName("x2").GetShape();

        TensorDesc outputDesc0 = op.GetOutputDescByName("y0");
        TensorDesc outputDesc1 = op.GetOutputDescByName("y1");
        TensorDesc outputDesc2 = op.GetOutputDescByName("y2");

        std::vector<int64_t> dims_x0 = x0_shape.GetDims();
        std::vector<int64_t> dims_x2 = x2_shape.GetDims();

        std::vector<int64_t> dim_y0;

        int segmentSize = 64;
        int maxBatch = 16;
        dim_y0.push_back(dims_x0[0]);
        dim_y0.push_back(dims_x2[1] * segmentSize);

        ge::Shape outputShape0 = ge::Shape(dim_y0);

        outputDesc0.SetShape(outputShape0);
        outputDesc0.SetDataType(x0_dtype);
        outputDesc0.SetFormat(x0_format);
        op.UpdateOutputDesc("y0", outputDesc0);

        std::vector<int64_t> dim_max_y;
        dim_max_y.push_back(dims_x0[0]);
        dim_max_y.push_back(dims_x2[1] * 2 * (segmentSize / maxBatch));

        ge::Shape outputShape1 = ge::Shape(dim_max_y);

        outputDesc1.SetShape(outputShape1);
        outputDesc1.SetDataType(x0_dtype);
        outputDesc1.SetFormat(x0_format);
        op.UpdateOutputDesc("y1", outputDesc1);
        
        std::vector<int64_t> dimVec2 { CORE_NUM, 16 };
        ge::Shape outputShape2 = ge::Shape(dimVec2);

        outputDesc2.SetShape(outputShape2);
        outputDesc2.SetDataType(DT_UINT16);
        outputDesc2.SetFormat(x0_format);
        op.UpdateOutputDesc("y2", outputDesc2);

        return GRAPH_SUCCESS;
    } // namespace ge

    // Registered inferfunction
    COMMON_INFER_FUNC_REG(DistanceIVFSQ8IPX, DistanceIVFSQ8IPXInferShape);

    // Registered verify function
    VERIFY_FUNC_REG(DistanceIVFSQ8IPX, DistanceIVFSQ8IPXVerify);
} // namespace ge