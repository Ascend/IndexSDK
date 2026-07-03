/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
 *
 * IndexSDK is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan
 * PSL v2. You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. See the
 * Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */
#include <cmath>
#include <cstdint>

#include "cagra_rabitq_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling
{
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = systemWorkspacesSize;

    auto query_shape = context->GetInputShape(0)->GetOriginShape();
    uint32_t block_number = query_shape.GetDim(0) / 128;
    context->SetBlockDim(block_number);
    context->SetLocalMemorySize(32 * 1024);

    // 从输入 ptr 的 shape 动态获取数据集大小 number_of_base
    auto base_shape = context->GetInputShape(3)->GetOriginShape();
    uint32_t number_of_base = base_shape.GetDim(0) / 128;  // sift1M dataset dim

    CagraRabitqTilingData tiling;
    tiling.set_size(number_of_base);  // sift1M dataset size

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge
{
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    // 校验所有输入形状非空
    for (int i = 0; i < 7; ++i)
    {
        if (context->GetInputShape(i) == nullptr)
        {
            return GRAPH_FAILED;
        }
    }

    // 获取 query 形状并计算 query_num
    const gert::Shape *queryShape = context->GetInputShape(0);
    int64_t query_num = queryShape->GetDim(0) / 128;

    // 计算输出总长度：query_num * 32
    int64_t output_size = query_num * 32;

    // 设置两个输出形状为一维向量
    gert::Shape *distShape = context->GetOutputShape(0);
    *distShape = gert::Shape({output_size});

    gert::Shape *idxShape = context->GetOutputShape(1);
    *idxShape = gert::Shape({output_size});

    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    // 校验输入数据类型
    if (context->GetInputDataType(0) != ge::DT_FLOAT)
    {
        return GRAPH_FAILED;
    }
    if (context->GetInputDataType(1) != ge::DT_UINT32)
    {
        return GRAPH_FAILED;
    }
    if (context->GetInputDataType(2) != ge::DT_UINT32)
    {
        return GRAPH_FAILED;
    }  // visited_hashmap_ptr
    if (context->GetInputDataType(3) != ge::DT_FLOAT)
    {
        return GRAPH_FAILED;
    }  // ptr
    if (context->GetInputDataType(4) != ge::DT_FLOAT)
    {
        return GRAPH_FAILED;
    }  // precompute_all
    if (context->GetInputDataType(5) != ge::DT_UINT8)
    {
        return GRAPH_FAILED;
    }  // code
    if (context->GetInputDataType(6) != ge::DT_UINT8)
    {
        return GRAPH_FAILED;
    }  // rotated_qq_ptr_all

    // 设置输出数据类型
    context->SetOutputDataType(0, ge::DT_FLOAT);   // result_distances_ptr
    context->SetOutputDataType(1, ge::DT_UINT32);  // result_indices_ptr

    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops
{
class CagraRabitq : public OpDef
{
   public:
    explicit CagraRabitq(const char *name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("knn_graph")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("visited_hashmap_ptr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("ptr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("precompute_all")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("code")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("rotated_qq_ptr_all")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("result_distances_ptr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("result_indices_ptr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(CagraRabitq);
}  // namespace ops
