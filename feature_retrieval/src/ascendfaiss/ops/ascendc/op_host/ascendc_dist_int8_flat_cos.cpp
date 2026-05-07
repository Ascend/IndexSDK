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


#include "ascendc_dist_int8_flat_cos_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host_common.h"

using namespace matmul_tiling;
using namespace Utils;

namespace optiling {
static ge::graphStatus TilingFillParam(AscendcDistInt8FlatCosTilingData &tiling, gert::TilingContext *context)
{
    // 算子固定第0个tensor为查询向量，第2个tensor为底库向量
    if ((context->GetInputTensor(0) == nullptr) || (context->GetInputTensor(2) == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &queryShape = context->GetInputTensor(0)->GetStorageShape(); // 算子固定第0个tensor为查询向量
    const gert::Shape &shapedShape = context->GetInputTensor(2)->GetStorageShape(); // 算子固定第2个tensor为底库向量
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    context->SetBlockDim(ascendcPlatform.GetCoreNumAic());

    uint32_t queryNum = static_cast<uint32_t>(queryShape[0]); // 查询向量为[queryNum, dim]，queryNum取第0维
    uint32_t dim = static_cast<uint32_t>(queryShape[1]); // 查询向量为[queryNum, dim]，dim取第1维
    // 底库向量为[blockSize/16, dim/32, 16, 32] 因此blockSize为第0维和第2维的乘积
    uint32_t baseBlockSize = static_cast<uint32_t>(shapedShape[0]) * static_cast<uint32_t>(shapedShape[2]);
    uint32_t vecCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (vecCoreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    tiling.set_queryNum(queryNum);
    tiling.set_dim(dim);
    tiling.set_baseBlockSize(baseBlockSize);
    tiling.set_vecCoreNum(vecCoreNum);
    // 实测512性能最佳；且受限于maxBurst为64得到2个数（极值和索引），以及搬运最小数量为16字节，其最小值为16/2*64=512
    const uint32_t onceComputeBaseNum = 512;
    tiling.set_onceComputeBaseNum(onceComputeBaseNum);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFillCubeTiling(AscendcDistInt8FlatCosTilingData &tiling, gert::TilingContext *context)
{
    auto onceComputeBaseNum = tiling.get_onceComputeBaseNum();
    auto queryNum = tiling.get_queryNum();
    auto dim = tiling.get_dim();
    uint32_t queryNumAlign16 = RoundUp(queryNum, CUBE_ALIGN);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    MatmulApiTiling cubeTiling(ascendcPlatform);
    tiling.cubeTilingData.set_usedCoreNum(1); // 算子内部已经按核tiling，因此这里只需1个核
    cubeTiling.SetAType(TPosition::TSCM, CubeFormat::NZ, matmul_tiling::DataType::DT_INT8);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_INT8);
    cubeTiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    cubeTiling.SetShape(queryNumAlign16, onceComputeBaseNum, dim);
    cubeTiling.SetOrgShape(queryNumAlign16, onceComputeBaseNum, dim);
    cubeTiling.SetBufferSpace(-1, -1, -1);
    cubeTiling.SetDequantType(DequantType::SCALAR);
    int ret = cubeTiling.GetTiling(tiling.cubeTilingData);
    if (ret == -1) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    AscendcDistInt8FlatCosTilingData tiling;
    auto ret = TilingFillParam(tiling, context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    ret = TilingFillCubeTiling(tiling, context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t vecCoreNum = ascendcPlatform.GetCoreNumAiv();
    auto onceComputeBaseNum = tiling.get_onceComputeBaseNum();
    auto queryNum = tiling.get_queryNum();
    uint32_t queryNumAlign16 = RoundUp(queryNum, CUBE_ALIGN);
    const size_t userWorkspaceSize = queryNumAlign16 * onceComputeBaseNum * sizeof(uint16_t) * vecCoreNum;
    const uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1); // 按照算子样例代码设置为1
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
class AscendcDistInt8FlatCos : public OpDef {
public:
    explicit AscendcDistInt8FlatCos(const char* name) : OpDef(name)
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

        this->Input("base")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("queryNorm")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("baseNorm")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("attr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("dist")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("maxDist")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("flag")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(AscendcDistInt8FlatCos);
}
