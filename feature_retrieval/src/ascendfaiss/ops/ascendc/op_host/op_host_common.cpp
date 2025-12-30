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


#include "op_host_common.h"

namespace ge {
ge::graphStatus ShapeCheck(gert::InferShapeContext *context,
                           const std::vector<size_t> &inputDimShape,
                           const std::vector<size_t> &outputDimShape)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    // 获取算子的输入个数
    size_t inputNums = context->GetComputeNodeInputNum();
    // 检查输入Tensor个数是否正确
    if (inputNums != inputDimShape.size()) {
        return GRAPH_FAILED;
    }
    // 遍历检查输入Tensor的shape
    for (size_t i = 0; i < inputNums; ++i) {
        if ((context->GetInputShape(i) == nullptr) || (context->GetInputShape(i)->GetDimNum() != inputDimShape[i])) {
            return GRAPH_FAILED;
        }
    }

    // 检查输出Tensor个数是否正确
    size_t outputNums = context->GetComputeNodeOutputNum();
    if (outputNums != outputDimShape.size()) {
        return GRAPH_FAILED;
    }
    // 遍历检查输出Tensor的shape
    // context->GetOutputShape(i)->GetDimNum()输出全是0
    return GRAPH_SUCCESS;
}

ge::graphStatus DataTypeCheck(gert::InferDataTypeContext *context,
                              const std::vector<ge::DataType> &inputDataType,
                              const std::vector<ge::DataType> &outputDataType)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    // 获取算子的输入个数
    size_t inputNums = context->GetComputeNodeInputNum();
    // 检查输入Tensor个数是否正确
    if (inputNums != inputDataType.size()) {
        return GRAPH_FAILED;
    }
    // 遍历检查输入Tensor的DataType
    for (size_t i = 0; i < inputNums; ++i) {
        if (context->GetInputDataType(i) != inputDataType[i]) {
            return GRAPH_FAILED;
        }
    }

    // 检查输出Tensor个数是否正确
    size_t outputNums = context->GetComputeNodeOutputNum();
    if (outputNums != outputDataType.size()) {
        return GRAPH_FAILED;
    }
    // 遍历检查输出Tensor的DataType
    for (size_t i = 0; i < outputNums; ++i) {
        if (context->GetOutputDataType(i) != outputDataType[i]) {
            return GRAPH_FAILED;
        }
    }
    return GRAPH_SUCCESS;
}
}
