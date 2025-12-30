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

#include "segment_max_ai.h"

#include <vector>
#include <string>
#include <iostream>

namespace ge {
    IMPLEMT_VERIFIER(SegmentMaxAI, SegmentMaxAIVerify)
    {
        return GRAPH_SUCCESS;
    }

    // Obtains the processing function of the output tensor description.
    IMPLEMT_COMMON_INFERFUNC(SegmentMaxAIInferShape)
    {
        DataType x0_dtype = op.GetInputDescByName("x0").GetDataType();
        Format x0_format = op.GetInputDescByName("x0").GetFormat();
        Shape x0_shape = op.GetInputDescByName("x0").GetShape();

        TensorDesc outputDesc0 = op.GetOutputDescByName("y0");
        TensorDesc outputDesc1 = op.GetOutputDescByName("y1");

        std::vector<int64_t> dims_x0 = x0_shape.GetDims();

        std::vector<int64_t> dim_y0;
        dim_y0.push_back(dims_x0[0]);
        dim_y0.push_back(dims_x0[1] / 32 * 2);

        ge::Shape outputShape0 = ge::Shape(dim_y0);

        outputDesc0.SetShape(outputShape0);
        outputDesc0.SetDataType(x0_dtype);
        outputDesc0.SetFormat(x0_format);
        op.UpdateOutputDesc("y0", outputDesc0);

        std::vector<int64_t> dimVec1 { 16, 16 };
        ge::Shape outputShape1 = ge::Shape(dimVec1);

        outputDesc1.SetShape(outputShape1);
        outputDesc1.SetDataType(DT_UINT16);
        outputDesc1.SetFormat(x0_format);
        op.UpdateOutputDesc("y1", outputDesc1);

        return GRAPH_SUCCESS;
    } // namespace ge

    // Registered inferfunction
    COMMON_INFER_FUNC_REG(SegmentMaxAI, SegmentMaxAIInferShape);

    // Registered verify function
    VERIFY_FUNC_REG(SegmentMaxAI, SegmentMaxAIVerify);
} // namespace ge