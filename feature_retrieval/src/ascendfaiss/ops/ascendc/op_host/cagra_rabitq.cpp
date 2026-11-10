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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "../op_kernel/cagra_rabitq_limits.h"
#include "cagra_rabitq_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling
{
namespace
{
constexpr uint32_t CAGRA_MIN_INTERNAL_TOPK = 64;
constexpr uint64_t CAGRA_TILING_KEY_SCALE = 10000;

uint32_t RoundUpPowerOfTwo(uint32_t value)
{
    uint32_t result = 1;
    while (result < value)
    {
        result <<= 1;
    }
    return result;
}

bool IsSupportedDim(uint32_t value)
{
    return value == 64 || value == 128 || value == 256 || value == 512 || value == 1024 || value == 1536 ||
           value == 3072;
}

bool IsSupportedGraphDegree(uint32_t value) { return value == 64 || value == 128; }

}  // namespace

static bool IsSupportedTopK(uint32_t value)
{
    return value == 32 || value == 64 || value == 128 || value == 256 || value == 512 || value == 1024;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr ||
        context->GetInputShape(3) == nullptr || context->GetAttrs() == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) return ge::GRAPH_FAILED;
    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    currentWorkspace[0] = systemWorkspacesSize;

    auto query_shape = context->GetInputShape(0)->GetOriginShape();
    auto graph_shape = context->GetInputShape(1)->GetOriginShape();
    auto base_shape = context->GetInputShape(3)->GetOriginShape();
    if (query_shape.GetDimNum() != 2 || graph_shape.GetDimNum() != 2 || base_shape.GetDimNum() != 2 ||
        query_shape.GetDim(0) <= 0 || query_shape.GetDim(0) > 64 || base_shape.GetDim(0) <= 1 ||
        base_shape.GetDim(0) > 1000000000LL || graph_shape.GetDim(0) != base_shape.GetDim(0) ||
        query_shape.GetDim(1) != base_shape.GetDim(1) || query_shape.GetDim(1) <= 0 || query_shape.GetDim(1) > 3072 ||
        graph_shape.GetDim(1) <= 0 || graph_shape.GetDim(1) > 128)
    {
        return ge::GRAPH_FAILED;
    }
    uint32_t block_number = query_shape.GetDim(0);
    uint32_t dim = query_shape.GetDim(1);
    uint32_t graph_degree = graph_shape.GetDim(1);
    const auto *top_k_attr = context->GetAttrs()->GetAttrPointer<int64_t>(0);
    if (top_k_attr == nullptr || *top_k_attr <= 0 || *top_k_attr > CAGRA_MAX_TOPK)
    {
        return ge::GRAPH_FAILED;
    }
    uint32_t top_k = static_cast<uint32_t>(*top_k_attr);
    uint32_t number_of_base = base_shape.GetDim(0);
    if (block_number == 0 || !IsSupportedDim(dim) || !IsSupportedGraphDegree(graph_degree) || number_of_base <= 1 ||
        !IsSupportedTopK(top_k) || top_k > number_of_base)
    {
        return ge::GRAPH_FAILED;
    }
    uint32_t internal_top_k = RoundUpPowerOfTwo(std::max(top_k, CAGRA_MIN_INTERNAL_TOPK));
    for (uint32_t i : {2U, 4U, 5U, 6U})
    {
        if (context->GetInputShape(i) == nullptr) return ge::GRAPH_FAILED;
    }
    const auto visited = context->GetInputShape(2)->GetOriginShape();
    const auto precompute = context->GetInputShape(4)->GetOriginShape();
    const auto code = context->GetInputShape(5)->GetOriginShape();
    const auto rotated = context->GetInputShape(6)->GetOriginShape();
    const int64_t planeBytes = (dim + 7) / 8;
    if (graph_degree >= number_of_base || visited.GetDimNum() != 2 || visited.GetDim(0) < block_number ||
        visited.GetDim(1) != (static_cast<int64_t>(number_of_base) + 31) / 32 || precompute.GetDimNum() != 1 ||
        precompute.GetDim(0) != static_cast<int64_t>(block_number) * 4 || code.GetDimNum() != 1 ||
        code.GetDim(0) != static_cast<int64_t>(number_of_base) * (6 * planeBytes + 16) || rotated.GetDimNum() != 1 ||
        rotated.GetDim(0) != static_cast<int64_t>(block_number) * 6 * planeBytes)
    {
        return ge::GRAPH_FAILED;
    }
    for (uint32_t i = 0; i < 2; ++i)
    {
        if (context->GetOutputShape(i) == nullptr) return ge::GRAPH_FAILED;
        const auto output = context->GetOutputShape(i)->GetOriginShape();
        if (output.GetDimNum() != 2 || output.GetDim(0) != block_number || output.GetDim(1) != top_k)
            return ge::GRAPH_FAILED;
    }
    if (CagraRabitqLocalMemoryBytes(dim, graph_degree, internal_top_k) > CAGRA_LOCAL_MEMORY_SIZE)
    {
        return ge::GRAPH_FAILED;
    }
    // One logical vector block per query; physical cores can execute multiple waves.
    // Keep all query blocks instead of truncating the batch to the physical core count.
    context->SetBlockDim(block_number);
    context->SetTilingKey(static_cast<uint64_t>(graph_degree) * CAGRA_TILING_KEY_SCALE + internal_top_k);
    context->SetLocalMemorySize(CAGRA_LOCAL_MEMORY_SIZE);

    CagraRabitqTilingData tiling;
    tiling.set_size(number_of_base);
    tiling.set_dim(dim);
    tiling.set_graphDegree(graph_degree);
    tiling.set_topK(top_k);
    tiling.set_internalTopK(internal_top_k);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge
{
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetOutputShape(0) == nullptr || context->GetOutputShape(1) == nullptr)
    {
        return GRAPH_FAILED;
    }
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
    if (queryShape->GetDimNum() != 2)
    {
        return GRAPH_FAILED;
    }
    int64_t query_num = queryShape->GetDim(0);
    if (context->GetAttrs() == nullptr)
    {
        return GRAPH_FAILED;
    }
    const auto *topKAttr = context->GetAttrs()->GetAttrPointer<int64_t>(0);
    if (topKAttr == nullptr || *topKAttr <= 0 || *topKAttr > CAGRA_MAX_TOPK ||
        !optiling::IsSupportedTopK(static_cast<uint32_t>(*topKAttr)))
    {
        return GRAPH_FAILED;
    }
    int64_t topK = *topKAttr;

    // The top_k attribute is part of the static model signature.
    gert::Shape *distShape = context->GetOutputShape(0);
    *distShape = gert::Shape({query_num, topK});

    gert::Shape *idxShape = context->GetOutputShape(1);
    *idxShape = gert::Shape({query_num, topK});

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

        this->Attr("top_k").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(CagraRabitq);
}  // namespace ops
