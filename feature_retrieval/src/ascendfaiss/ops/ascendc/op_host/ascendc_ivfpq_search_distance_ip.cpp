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

#include "ascendc_ivfpq_search_distance_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t IVFPQ_ONE = 1;
constexpr uint32_t IVFPQ_CODE_BLOCK_SIZE = 16384 * 16;
constexpr uint32_t IVFPQ_NUM_32 = 32;
} // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    IvfpqTiling ivfpqTiling;
    AscendcIvfpqSearchDistanceTopKTilingData tilingData;
    return ivfpqTiling.ProcessTiling(context, tilingData, ReduceMode::IP);
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *queryPQShape = context->GetInputShape(0);
    int64_t batch = queryPQShape->GetDim(0);

    const gert::Shape *codeOffsetShape = context->GetInputShape(2);
    int64_t codeBlockNum = codeOffsetShape->GetDim(1);

    const gert::Shape *topkShape = context->GetInputShape(4);
    int64_t topk = topkShape->GetDim(0);

    gert::Shape *distShape = context->GetOutputShape(0);
    *distShape = gert::Shape({batch, codeBlockNum, IVFPQ_CODE_BLOCK_SIZE});

    gert::Shape *topkIndexShape = context->GetOutputShape(1);
    *topkIndexShape = gert::Shape({batch, codeBlockNum, topk});

    gert::Shape *topkValueShape = context->GetOutputShape(2);
    *topkValueShape = gert::Shape({batch, codeBlockNum, topk});

    gert::Shape *topkLabelFinalShape = context->GetOutputShape(3);
    *topkLabelFinalShape = gert::Shape({batch, topk});

    gert::Shape *topkValueFinalShape = context->GetOutputShape(4);
    *topkValueFinalShape = gert::Shape({batch, topk});

    gert::Shape *flagShape = context->GetOutputShape(5);
    *flagShape = gert::Shape({16});
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class AscendcIvfpqSearchDistanceIP : public OpDef {
public:
    explicit AscendcIvfpqSearchDistanceIP(const char *name) : OpDef(name)
    {
        this->Input("queryPQ")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("codeBase")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("codeOffset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("codeSize")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("topk")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("labelBase")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("labelOffset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("distResult")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("topkIndex")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("topkValue")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("topkLabelFinal")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("topkValueFinal")
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

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(AscendcIvfpqSearchDistanceIP);
} // namespace ops
