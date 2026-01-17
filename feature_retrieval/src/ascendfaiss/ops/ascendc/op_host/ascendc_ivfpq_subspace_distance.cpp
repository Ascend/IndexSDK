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

#include "ascendc_ivfpq_subspace_distance_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
    constexpr uint32_t N_BLOCK_BASE_SIZE = 128;
    constexpr uint32_t ONE = 1;
    constexpr uint32_t REDUCE_BASE_SIZE = 64;
    constexpr uint32_t BATCH_MIN = 1;
    constexpr uint32_t BATCH_MAX = 256;
    constexpr uint32_t KSUB_DEFAULT = 256;
}

using namespace matmul_tiling;

namespace optiling {

ge::graphStatus IvfpqSubspaceTiling::GetNpuInfo()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IvfpqSubspaceTiling::ProcessInput()
{
    // InTensor
    auto queryShape = context_->GetInputShape(0)->GetStorageShape(); // 第0个输入 query
    auto queryDimNum = queryShape.GetDimNum();
    auto codeBookShape = context_->GetInputShape(1)->GetStorageShape(); // 第1个输入 codeBook
    auto codeBookDimNum = codeBookShape.GetDimNum();
    if (queryDimNum != 2U || codeBookDimNum != 3U) {
        return ge::GRAPH_FAILED;
    }

    batch_ = static_cast<uint32_t>(queryShape.GetDim(0));
    dim_ = static_cast<uint32_t>(queryShape.GetDim(1));
    subSpaceNum_ = static_cast<uint32_t>(codeBookShape.GetDim(0));
    kSub_ = static_cast<uint32_t>(codeBookShape.GetDim(1));
    dSub_ = static_cast<uint32_t>(codeBookShape.GetDim(2));

    if (batch_ < BATCH_MIN || batch_ > BATCH_MAX || kSub_ != KSUB_DEFAULT) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IvfpqSubspaceTiling::Split()
{
    nBlockTile_ = N_BLOCK_BASE_SIZE;
    nBlockNum_ = kSub_ / nBlockTile_;
    totalTaskNum_ = subSpaceNum_ * nBlockNum_;
    usedAicNum_ = std::max(std::min(totalTaskNum_, aicNum_), ONE);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IvfpqSubspaceTiling::ReduceSumTiling()
{
    auto shape = ge::Shape({ batch_, dSub_ });
    auto codeBookshape = ge::Shape({ nBlockTile_ / 2, dSub_ });

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IvfpqSubspaceTiling::CubeTiling()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    MatmulApiTiling cubeTiling(ascendcPlatform);
    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, true);
    cubeTiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetShape(batch_, nBlockTile_, dSub_);
    cubeTiling.SetOrgShape(batch_, nBlockTile_, dim_, dSub_);
    cubeTiling.SetBufferSpace(-1, -1, -1);
    int ret = cubeTiling.GetTiling(tilingData_->cubeTiling);
    if (ret == -1) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IvfpqSubspaceTiling::FillTilingData()
{
    tilingData_->set_batch(batch_);
    tilingData_->set_subSpaceNum(subSpaceNum_);
    tilingData_->set_dim(dim_);
    tilingData_->set_kSub(kSub_);
    tilingData_->set_dSub(dSub_);
    tilingData_->set_minValue(minValue_);

    tilingData_->set_nBlockTile(nBlockTile_);
    tilingData_->set_nBlockNum(nBlockNum_);
    tilingData_->set_totalTaskNum(totalTaskNum_);

    tilingData_->set_codeBookMinValue(codeBookMinValue_);

    tilingData_->set_usedCoreNum(usedAicNum_);
    tilingData_->set_tilingKey(tilingKey_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IvfpqSubspaceTiling::SetExtraConfig()
{
    context_->SetBlockDim(tilingData_->get_usedCoreNum());
    context_->SetTilingKey(tilingData_->get_tilingKey());
    tilingData_->SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_->GetDataSize());

    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    workSpaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    uint32_t userSize = batch_ * nBlockTile_ * usedAicNum_ * sizeof(float); // M * N * coreNum * dataTypeSize
    currentWorkspace[0] = workSpaceSize_ + userSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IvfpqSubspaceTiling::ProcessTiling(gert::TilingContext* context,
                                                   AscendcIvfpqSubspaceDistanceTilingData& tilingData)
{
    context_ = context;
    tilingData_ = &tilingData;
    if (GetNpuInfo() != ge::GRAPH_SUCCESS ||
        ProcessInput() != ge::GRAPH_SUCCESS ||
        Split() != ge::GRAPH_SUCCESS ||
        ReduceSumTiling() != ge::GRAPH_SUCCESS ||
        CubeTiling() != ge::GRAPH_SUCCESS ||
        FillTilingData() != ge::GRAPH_SUCCESS ||
        SetExtraConfig() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    IvfpqSubspaceTiling ivfpqSubTiling;
    AscendcIvfpqSubspaceDistanceTilingData tilingData;
    auto ret = ivfpqSubTiling.ProcessTiling(context, tilingData);
    if (ret != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* queryShape = context->GetInputShape(0);
    int64_t batch = queryShape->GetDim(0);

    const gert::Shape* codeBookShape = context->GetInputShape(1);
    int64_t subSpaceNum = codeBookShape->GetDim(0);
    int64_t kSub = codeBookShape->GetDim(1);

    gert::Shape* distanceShape = context->GetOutputShape(0);
    *distanceShape = gert::Shape({batch, subSpaceNum, kSub});

    gert::Shape* minDistShape = context->GetOutputShape(1);
    *minDistShape = gert::Shape({batch, subSpaceNum, kSub / REDUCE_BASE_SIZE * 2});

    gert::Shape* flagShape = context->GetOutputShape(2);
    *flagShape = gert::Shape({batch, subSpaceNum, 16});

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class AscendcIvfpqSubspaceDistance : public OpDef {
public:
    explicit AscendcIvfpqSubspaceDistance(const char* name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("codeBook")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("distance")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("minDistResult")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("flag")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910_95");
    }
};

OP_ADD(AscendcIvfpqSubspaceDistance);
}
