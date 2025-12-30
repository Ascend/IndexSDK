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


#ifndef ASCENDC_OP_HOST_OP_HOST_COMMON_H
#define ASCENDC_OP_HOST_OP_HOST_COMMON_H

#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace Utils {
constexpr uint32_t CUBE_ALIGN = 16;
constexpr uint32_t KB = 1024;

template<typename T>
constexpr auto Min(T a, T b) -> decltype(a)
{
    return a < b ? a : b;
}

template<typename T>
constexpr auto Max(T a, T b) -> decltype(a)
{
    return a > b ? a : b;
}

template<typename U, typename V>
constexpr auto DivUp(U a, V b) -> decltype(a + b)
{
    return ((a + b - 1) / b);
}

template<typename U, typename V>
constexpr auto RoundUp(U a, V b) -> decltype(a + b)
{
    return DivUp(a, b) * b;
}
}

namespace ge {
ge::graphStatus ShapeCheck(gert::InferShapeContext *context,
                           const std::vector<size_t> &inputDimShape,
                           const std::vector<size_t> &outputDimShape);

ge::graphStatus DataTypeCheck(gert::InferDataTypeContext *context,
                              const std::vector<ge::DataType> &inputDataType,
                              const std::vector<ge::DataType> &outputDataType);
}
#endif // ASCENDC_OP_HOST_OP_HOST_COMMON_H
