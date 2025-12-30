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


#include "corr_compute.h"
#include "ascend_operator.h"

#include <vector>
#include <string>
#include <iostream>

namespace ge {
IMPLEMT_VERIFIER(CorrCompute, CorrComputeVerify)
{
    return GRAPH_SUCCESS;
}


// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(CorrComputeInferShape)
{
    DataType inputDtype = op.GetInputDescByName("x0").GetDataType();
    Format inputFormat = op.GetInputDescByName("x0").GetFormat();

    Shape x0_shape = op.GetInputDescByName("x0").GetShape();
    TensorDesc OutputDesc0 = op.GetOutputDescByName("y0");
    TensorDesc OutputDesc1 = op.GetOutputDescByName("y1");

    std::vector<int64_t> dimsX0 = x0_shape.GetDims();

    std::vector<int64_t> dimY;
    dimY.push_back(dimsX0[2]);
    dimY.push_back(dimsX0[2]);

    ge::Shape outputShape0 = ge::Shape(dimY);

    OutputDesc0.SetShape(outputShape0);
    OutputDesc0.SetDataType(inputDtype);
    OutputDesc0.SetFormat(inputFormat);
    op.UpdateOutputDesc("y0", OutputDesc0);

    std::vector<int64_t> dimVec1 { CORE_NUM, 16 };
    ge::Shape outputShape1 = ge::Shape(dimVec1);

    OutputDesc1.SetShape(outputShape1);
    OutputDesc1.SetDataType(DT_UINT16);
    OutputDesc1.SetFormat(inputFormat);
    op.UpdateOutputDesc("y1", OutputDesc1);

    return GRAPH_SUCCESS;
}

// Registered inferfunction
COMMON_INFER_FUNC_REG(CorrCompute, CorrComputeInferShape);

// Registered verify function
VERIFY_FUNC_REG(CorrCompute, CorrComputeVerify);
} // namespace ge
