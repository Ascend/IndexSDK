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


#include "cid_filter.h"
#include "ascend_operator.h"

#include <vector>
#include <string>
#include <iostream>

namespace ge {
    IMPLEMT_VERIFIER(CidFilter, CidFilterVerify)
    {
        return GRAPH_SUCCESS;
    }


    // Obtains the processing function of the output tensor description.
    IMPLEMT_COMMON_INFERFUNC(CidFilterInferShape)
    {
        Format x0_format = op.GetInputDescByName("x0").GetFormat();

        Shape x0_shape = op.GetInputDescByName("x0").GetShape();
        std::vector<int64_t> dims_x0 = x0_shape.GetDims();

        TensorDesc outputDesc0 = op.GetOutputDescByName("y0");
        TensorDesc outputDesc1 = op.GetOutputDescByName("y1");

        std::vector<int64_t> dim_y0 { dims_x0[0] / 16 }; // uint16_t have 16 bits
        ge::Shape outputShape0 = ge::Shape(dim_y0);

        outputDesc0.SetShape(outputShape0);
        outputDesc0.SetDataType(DT_UINT16);
        outputDesc0.SetFormat(x0_format);
        op.UpdateOutputDesc("y0", outputDesc0);

        std::vector<int64_t> dimVec2 { CORE_NUM, 16 };
        ge::Shape outputShape1 = ge::Shape(dimVec2);

        outputDesc1.SetShape(outputShape1);
        outputDesc1.SetDataType(DT_UINT16);
        outputDesc1.SetFormat(x0_format);
        op.UpdateOutputDesc("y1", outputDesc1);

        return GRAPH_SUCCESS;
    } // namespace ge

    // Registered inferfunction
    COMMON_INFER_FUNC_REG(CidFilter, CidFilterInferShape);

    // Registered verify function
    VERIFY_FUNC_REG(CidFilter, CidFilterVerify);
} // namespace ge