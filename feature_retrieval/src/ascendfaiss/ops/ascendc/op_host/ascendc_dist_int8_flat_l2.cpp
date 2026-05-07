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


#include "ascendc_dist_int8_flat_l2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host_common.h"

using namespace matmul_tiling;

namespace {
    constexpr uint32_t QUERY_MAX_SIZE = 48;
    constexpr uint32_t CODE_MAX_SIZE = 256;
    constexpr uint32_t MIN_BATCH = 64;
}

namespace optiling {
using namespace matmul_tiling;
using namespace Utils;

static ge::graphStatus TilingSetInputShapeInfo(gert::TilingContext* context, AscendcDistInt8FlatL2TilingData &tiling)
{
    // 算子输入第0个tensor为查询向量query，第2个tensor为底库向量
    if ((context->GetInputShape(0) == nullptr) || (context->GetInputShape(2) == nullptr)) {
        return ge::GRAPH_FAILED;
    }

    // 算子输入第0个tensor为查询向量query,tensor维度为(querySize, dim)
    auto queryShape = context->GetInputShape(0)->GetStorageShape();
    uint32_t querySize = static_cast<uint32_t>(queryShape.GetDim(0));  // 第0维取querySize
    uint32_t dim = static_cast<uint32_t>(queryShape.GetDim(1));  // 第1维取querySize

    // 算子输入第2个tensor为底库向量,tensor维度为(codeBlockSize / 16, dim / 32, 16, 32)
    auto codeShape = context->GetInputShape(2)->GetStorageShape();
    // 第0维 * 第2维得到codeBlockSize
    uint32_t codeBlockSize = static_cast<uint32_t>(codeShape.GetDim(0)) * static_cast<uint32_t>(codeShape.GetDim(2));

    tiling.set_querySize(querySize);
    tiling.set_dim(dim);
    tiling.set_codeBlockSize(codeBlockSize);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    if (aicNum == 0 || aivNum == 0) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(aicNum);
    tiling.set_aivNum(aivNum);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingSetCubeTiling(gert::TilingContext* context, AscendcDistInt8FlatL2TilingData &tiling)
{
    uint32_t querySize = tiling.get_querySize();
    uint32_t querySizeEachLoop = Min(QUERY_MAX_SIZE, querySize);
    uint32_t codeSizeEachLoop = CODE_MAX_SIZE;

    uint32_t dim = tiling.get_dim();

    // 设置matmul的tiling参数
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    // square tiling参数
    MatmulApiTiling cubeTilingSquare(ascendcPlatform);
    cubeTilingSquare.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8);
    cubeTilingSquare.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, true);
    cubeTilingSquare.SetCType(TPosition::LCM, CubeFormat::ND, DataType::DT_INT32);
    cubeTilingSquare.SetShape(querySizeEachLoop, querySizeEachLoop, dim);
    cubeTilingSquare.SetOrgShape(querySizeEachLoop, querySizeEachLoop, dim);
    cubeTilingSquare.SetBufferSpace(-1, -1, -1);
    int ret = cubeTilingSquare.GetTiling(tiling.cubeTilingSquare);
    if (ret == -1) {
        return ge::GRAPH_FAILED;
    }

    // ip tiling参数
    MatmulApiTiling cubeTilingIp(ascendcPlatform);
    cubeTilingIp.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8);
    cubeTilingIp.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, true);
    cubeTilingIp.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_INT32);
    cubeTilingIp.SetShape(querySizeEachLoop, codeSizeEachLoop, dim);
    cubeTilingIp.SetOrgShape(querySizeEachLoop, codeSizeEachLoop, dim);
    cubeTilingIp.SetBufferSpace(-1, -1, -1);
    ret = cubeTilingIp.GetTiling(tiling.cubeTilingIp);
    if (ret == -1) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    AscendcDistInt8FlatL2TilingData tiling;

    auto ret = TilingSetInputShapeInfo(context, tiling);
    if (ret != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    ret = TilingSetCubeTiling(context, tiling);
    if (ret != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    uint32_t querySize = tiling.get_querySize();
    uint32_t querySizeEachLoop = Min(QUERY_MAX_SIZE, querySize);
    const size_t userWorkspaceSize = querySizeEachLoop * CODE_MAX_SIZE * tiling.get_aivNum() * sizeof(int32_t);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = userWorkspaceSize + sysWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *)
{
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *)
{
    return GRAPH_SUCCESS;
}
}

namespace ops {
class AscendcDistInt8FlatL2 : public OpDef {
public:
    explicit AscendcDistInt8FlatL2(const char* name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("codes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("norm")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("actualSize")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("dist")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("minDist")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("flag")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(AscendcDistInt8FlatL2);
}
