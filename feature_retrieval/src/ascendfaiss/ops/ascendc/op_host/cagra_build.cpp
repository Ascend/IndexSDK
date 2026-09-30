/*
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
 * Licensed under Mulan PSL v2.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "cagra_build_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace
{
constexpr uint32_t kThreadsPerBlock = 512;
constexpr uint32_t kDistanceTeamSize = 8;
constexpr uint32_t kPruneTeamSize = 32;
// Keep the graph-build limits below synchronized with tools/cagra_build_generate_model.py.
constexpr uint32_t kMaxPruneIntermediateDegree = 128;
constexpr uint32_t kMaxBuildDim = 3072;
constexpr uint32_t kMaxSampleDegree = 32;
constexpr uint32_t kSizeEntryCount = 4;
constexpr uint32_t kMaxNodeNum = static_cast<uint32_t>(std::numeric_limits<int32_t>::max());

struct TilingArgs
{
    uint32_t nodeNum = 0;
    uint32_t dim = 0;
    uint32_t intermediateDegree = 0;
    uint32_t sampleDegree = 0;
    uint32_t candidateDegree = 0;
    uint32_t outputDegree = 0;
};

bool ToPositiveUint32(int64_t value, uint32_t &result)
{
    if (value <= 0 || value > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
    {
        return false;
    }
    result = static_cast<uint32_t>(value);
    return true;
}

ge::graphStatus SetTiling(gert::TilingContext *context, const TilingArgs &args,
                          uint32_t rowsPerBlock = kThreadsPerBlock)
{
    if (context == nullptr || context->GetRawTilingData() == nullptr || args.nodeNum < 2 ||
        args.nodeNum > kMaxNodeNum || rowsPerBlock == 0)
    {
        return ge::GRAPH_FAILED;
    }
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aivNum = platform.GetCoreNumAiv();
    if (aivNum == 0)
    {
        return ge::GRAPH_FAILED;
    }
    const uint32_t requiredBlocks = (args.nodeNum - 1) / rowsPerBlock + 1;
    const uint32_t blockNum = std::min(aivNum, requiredBlocks);

    optiling::CagraBuildTilingData tiling;
    tiling.set_nodeNum(args.nodeNum);
    tiling.set_dim(args.dim);
    tiling.set_intermediateDegree(args.intermediateDegree);
    tiling.set_sampleDegree(args.sampleDegree);
    tiling.set_candidateDegree(args.candidateDegree);
    tiling.set_outputDegree(args.outputDegree);
    tiling.set_blockNum(blockNum);
    context->SetBlockDim(blockNum);

    size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = platform.GetLibApiWorkSpaceSize();
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NndInitTiling(gert::TilingContext *context)
{
    if (context == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    const auto *data = context->GetInputShape(0);
    const auto *graph = context->GetOutputShape(0);
    if (data == nullptr || graph == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    TilingArgs args;
    if (!ToPositiveUint32(data->GetStorageShape().GetDim(0), args.nodeNum) ||
        !ToPositiveUint32(data->GetStorageShape().GetDim(1), args.dim) ||
        !ToPositiveUint32(graph->GetStorageShape().GetDim(1), args.intermediateDegree) || args.nodeNum > kMaxNodeNum ||
        args.dim > kMaxBuildDim || args.intermediateDegree > kMaxPruneIntermediateDegree ||
        args.intermediateDegree >= args.nodeNum)
    {
        return ge::GRAPH_FAILED;
    }
    return SetTiling(context, args);
}

ge::graphStatus NndSampleReverseTiling(gert::TilingContext *context)
{
    if (context == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    const auto *graph = context->GetInputShape(0);
    const auto *forwardNew = context->GetOutputShape(0);
    if (graph == nullptr || forwardNew == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    TilingArgs args;
    if (!ToPositiveUint32(graph->GetStorageShape().GetDim(0), args.nodeNum) ||
        !ToPositiveUint32(graph->GetStorageShape().GetDim(1), args.intermediateDegree) ||
        !ToPositiveUint32(forwardNew->GetStorageShape().GetDim(1), args.sampleDegree) || args.nodeNum > kMaxNodeNum ||
        args.intermediateDegree > kMaxPruneIntermediateDegree || args.intermediateDegree >= args.nodeNum ||
        args.sampleDegree > kMaxSampleDegree || args.sampleDegree > args.intermediateDegree)
    {
        return ge::GRAPH_FAILED;
    }
    return SetTiling(context, args);
}

ge::graphStatus NndLocalJoinTiling(gert::TilingContext *context)
{
    if (context == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    const auto *data = context->GetInputShape(0);
    const auto *forwardNew = context->GetInputShape(1);
    const auto *candidates = context->GetOutputShape(0);
    if (data == nullptr || forwardNew == nullptr || candidates == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    TilingArgs args;
    if (!ToPositiveUint32(data->GetStorageShape().GetDim(0), args.nodeNum) ||
        !ToPositiveUint32(data->GetStorageShape().GetDim(1), args.dim) ||
        !ToPositiveUint32(forwardNew->GetStorageShape().GetDim(1), args.sampleDegree) ||
        !ToPositiveUint32(candidates->GetStorageShape().GetDim(1), args.candidateDegree) ||
        args.nodeNum > kMaxNodeNum || args.dim > kMaxBuildDim || args.sampleDegree > kMaxSampleDegree ||
        args.candidateDegree > kMaxPruneIntermediateDegree || args.sampleDegree > args.candidateDegree ||
        args.candidateDegree >= args.nodeNum)
    {
        return ge::GRAPH_FAILED;
    }
    return SetTiling(context, args, kThreadsPerBlock / kDistanceTeamSize);
}

ge::graphStatus NndUpdateTiling(gert::TilingContext *context)
{
    if (context == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    const auto *graph = context->GetInputShape(0);
    const auto *forwardNew = context->GetInputShape(2);
    const auto *candidates = context->GetInputShape(4);
    if (graph == nullptr || forwardNew == nullptr || candidates == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    TilingArgs args;
    if (!ToPositiveUint32(graph->GetStorageShape().GetDim(0), args.nodeNum) ||
        !ToPositiveUint32(graph->GetStorageShape().GetDim(1), args.intermediateDegree) ||
        !ToPositiveUint32(forwardNew->GetStorageShape().GetDim(1), args.sampleDegree) ||
        !ToPositiveUint32(candidates->GetStorageShape().GetDim(1), args.candidateDegree) ||
        args.nodeNum > kMaxNodeNum || args.intermediateDegree > kMaxPruneIntermediateDegree ||
        args.intermediateDegree >= args.nodeNum || args.sampleDegree > kMaxSampleDegree ||
        args.sampleDegree > args.intermediateDegree || args.candidateDegree != args.intermediateDegree)
    {
        return ge::GRAPH_FAILED;
    }
    return SetTiling(context, args);
}

ge::graphStatus PruneReverseTiling(gert::TilingContext *context)
{
    if (context == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    const auto *input = context->GetInputShape(0);
    const auto *output = context->GetOutputShape(0);
    if (input == nullptr || output == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    TilingArgs args;
    if (!ToPositiveUint32(input->GetStorageShape().GetDim(0), args.nodeNum) ||
        !ToPositiveUint32(input->GetStorageShape().GetDim(1), args.intermediateDegree) ||
        !ToPositiveUint32(output->GetStorageShape().GetDim(1), args.outputDegree) || args.nodeNum > kMaxNodeNum ||
        args.outputDegree > args.intermediateDegree || args.outputDegree >= args.nodeNum ||
        args.intermediateDegree >= args.nodeNum || args.intermediateDegree > kMaxPruneIntermediateDegree)
    {
        return ge::GRAPH_FAILED;
    }
    // Sixteen independent 32-thread teams share a 512-thread block; each team owns one row.
    return SetTiling(context, args, kThreadsPerBlock / kPruneTeamSize);
}

ge::graphStatus MergeTiling(gert::TilingContext *context)
{
    if (context == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    const auto *input = context->GetInputShape(0);
    if (input == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    TilingArgs args;
    if (!ToPositiveUint32(input->GetStorageShape().GetDim(0), args.nodeNum) ||
        !ToPositiveUint32(input->GetStorageShape().GetDim(1), args.outputDegree) || args.nodeNum > kMaxNodeNum ||
        args.outputDegree >= args.nodeNum)
    {
        return ge::GRAPH_FAILED;
    }
    return SetTiling(context, args);
}

ge::graphStatus SetMatrixShape(gert::InferShapeContext *context, size_t outputIndex, int64_t rows, int64_t cols)
{
    if (context == nullptr || rows <= 0 || cols <= 0)
    {
        return ge::GRAPH_FAILED;
    }
    auto *output = context->GetOutputShape(outputIndex);
    if (output == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    *output = gert::Shape({rows, cols});
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetVectorShape(gert::InferShapeContext *context, size_t outputIndex, int64_t length)
{
    if (context == nullptr || length <= 0)
    {
        return ge::GRAPH_FAILED;
    }
    auto *output = context->GetOutputShape(outputIndex);
    if (output == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    *output = gert::Shape({length});
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NndInitInferShape(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr || context->GetAttrs() == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    const auto *degree = context->GetAttrs()->GetAttrPointer<int64_t>(0);
    if (degree == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    const int64_t nodeNum = context->GetInputShape(0)->GetDim(0);
    if (*degree <= 0 || *degree >= nodeNum || SetMatrixShape(context, 0, nodeNum, *degree) != ge::GRAPH_SUCCESS)
    {
        return ge::GRAPH_FAILED;
    }
    return SetMatrixShape(context, 1, nodeNum, *degree);
}

ge::graphStatus NndSampleReverseInferShape(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    const auto *graph = context->GetInputShape(0);
    const int64_t nodeNum = graph->GetDim(0);
    const int64_t intermediateDegree = graph->GetDim(1);
    const int64_t sampleDegree = std::min<int64_t>(kMaxSampleDegree, intermediateDegree);
    for (size_t i = 0; i < kSizeEntryCount; ++i)
    {
        if (SetMatrixShape(context, i, nodeNum, sampleDegree) != ge::GRAPH_SUCCESS)
        {
            return ge::GRAPH_FAILED;
        }
    }
    return SetMatrixShape(context, kSizeEntryCount, nodeNum, kSizeEntryCount);
}

ge::graphStatus NndLocalJoinInferShape(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr || context->GetAttrs() == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    const auto *degree = context->GetAttrs()->GetAttrPointer<int64_t>(0);
    if (degree == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    const int64_t nodeNum = context->GetInputShape(0)->GetDim(0);
    if (*degree <= 0 || *degree >= nodeNum || SetMatrixShape(context, 0, nodeNum, *degree) != ge::GRAPH_SUCCESS ||
        SetMatrixShape(context, 1, nodeNum, *degree) != ge::GRAPH_SUCCESS)
    {
        return ge::GRAPH_FAILED;
    }
    return SetVectorShape(context, 2, nodeNum);
}

ge::graphStatus NndUpdateInferShape(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    const auto *graph = context->GetInputShape(0);
    const int64_t nodeNum = graph->GetDim(0);
    const int64_t intermediateDegree = graph->GetDim(1);
    if (SetMatrixShape(context, 0, nodeNum, intermediateDegree) != ge::GRAPH_SUCCESS ||
        SetMatrixShape(context, 1, nodeNum, intermediateDegree) != ge::GRAPH_SUCCESS)
    {
        return ge::GRAPH_FAILED;
    }
    return SetVectorShape(context, 2, 1);
}

ge::graphStatus PruneReverseInferShape(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr || context->GetAttrs() == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    const auto *degree = context->GetAttrs()->GetAttrPointer<int64_t>(0);
    if (degree == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    const auto *input = context->GetInputShape(0);
    const int64_t nodeNum = input->GetDim(0);
    const int64_t intermediateDegree = input->GetDim(1);
    if (*degree <= 0 || *degree > intermediateDegree || *degree >= nodeNum ||
        SetMatrixShape(context, 0, nodeNum, *degree) != ge::GRAPH_SUCCESS ||
        SetMatrixShape(context, 1, nodeNum, *degree) != ge::GRAPH_SUCCESS ||
        SetVectorShape(context, 2, nodeNum) != ge::GRAPH_SUCCESS)
    {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MergeInferShape(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    const auto *input = context->GetInputShape(0);
    return SetMatrixShape(context, 0, input->GetDim(0), input->GetDim(1));
}

ge::graphStatus InitDtype(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_UINT32);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SampleDtype(gert::InferDataTypeContext *context)
{
    for (size_t i = 0; i < kSizeEntryCount + 1; ++i)
    {
        context->SetOutputDataType(i, ge::DT_UINT32);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LocalJoinDtype(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_UINT32);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_UINT32);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpdateDtype(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_UINT32);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_UINT32);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Uint3Dtype(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_UINT32);
    context->SetOutputDataType(1, ge::DT_UINT32);
    context->SetOutputDataType(2, ge::DT_UINT32);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UintDtype(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_UINT32);
    return ge::GRAPH_SUCCESS;
}
}  // namespace

namespace ops
{
class CagraNndInit : public OpDef
{
   public:
    explicit CagraNndInit(const char *name) : OpDef(name)
    {
        this->Input("data").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("graph").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Output("distances").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("intermediate_degree").AttrType(REQUIRED).Int();
        this->SetInferShape(NndInitInferShape);
        this->SetInferDataType(InitDtype);
        this->AICore().SetTiling(NndInitTiling);
        this->AICore().AddConfig("ascend950");
    }
};

class CagraNndSampleReverse : public OpDef
{
   public:
    explicit CagraNndSampleReverse(const char *name) : OpDef(name)
    {
        this->Input("graph").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Output("forward_new").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Output("forward_old").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Output("reverse_new").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Output("reverse_old").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Output("sizes").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->SetInferShape(NndSampleReverseInferShape);
        this->SetInferDataType(SampleDtype);
        this->AICore().SetTiling(NndSampleReverseTiling);
        this->AICore().AddConfig("ascend950");
    }
};

class CagraNndLocalJoin : public OpDef
{
   public:
    explicit CagraNndLocalJoin(const char *name) : OpDef(name)
    {
        this->Input("data").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("forward_new").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Input("forward_old").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Input("reverse_new").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Input("reverse_old").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Input("sizes").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Output("candidate_ids").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Output("candidate_distances").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("candidate_counts").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Attr("candidate_degree").AttrType(REQUIRED).Int();
        this->SetInferShape(NndLocalJoinInferShape);
        this->SetInferDataType(LocalJoinDtype);
        this->AICore().SetTiling(NndLocalJoinTiling);
        this->AICore().AddConfig("ascend950");
    }
};

class CagraNndUpdate : public OpDef
{
   public:
    explicit CagraNndUpdate(const char *name) : OpDef(name)
    {
        this->Input("graph").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Input("distances").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("forward_new").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Input("sizes").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Input("candidate_ids").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Input("candidate_distances").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("candidate_counts").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Output("next_graph").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Output("next_distances").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("update_count").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->SetInferShape(NndUpdateInferShape);
        this->SetInferDataType(UpdateDtype);
        this->AICore().SetTiling(NndUpdateTiling);
        this->AICore().AddConfig("ascend950");
    }
};

class CagraPruneReverse : public OpDef
{
   public:
    explicit CagraPruneReverse(const char *name) : OpDef(name)
    {
        this->Input("intermediate_graph").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Output("pruned_graph").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Output("reverse_graph").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Output("reverse_counts").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Attr("output_degree").AttrType(REQUIRED).Int();
        this->SetInferShape(PruneReverseInferShape);
        this->SetInferDataType(Uint3Dtype);
        this->AICore().SetTiling(PruneReverseTiling);
        this->AICore().AddConfig("ascend950");
    }
};

class CagraMerge : public OpDef
{
   public:
    explicit CagraMerge(const char *name) : OpDef(name)
    {
        this->Input("pruned_graph").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Input("reverse_graph").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Input("reverse_counts").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Output("graph").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->SetInferShape(MergeInferShape);
        this->SetInferDataType(UintDtype);
        this->AICore().SetTiling(MergeTiling);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(CagraNndInit);
OP_ADD(CagraNndSampleReverse);
OP_ADD(CagraNndLocalJoin);
OP_ADD(CagraNndUpdate);
OP_ADD(CagraPruneReverse);
OP_ADD(CagraMerge);
}  // namespace ops
