/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

#include "ascendc_distance_flat_hamming_with_mask_tiling.h"
#include "op_host_common.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

using namespace matmul_tiling;

namespace
{
constexpr uint32_t BINARY_BYTE_BITS = 8;
constexpr uint32_t MAX_QUERY_TILE = 128;
constexpr uint32_t MAX_EXPANDED_ELEMENTS = 32 * 1024;
constexpr uint32_t MAX_BURSTS_PER_TILE = 32;
constexpr uint32_t DIM_1024_CODE_SIZE = 128;
constexpr uint32_t DIM_1024_BURST_LEN = 64;
constexpr uint32_t DEFAULT_BURST_LEN = 128;
}  // namespace

namespace optiling
{
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr || context->GetRawTilingData() == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputTensor(0) == nullptr || context->GetInputTensor(1) == nullptr ||
        context->GetInputTensor(2) == nullptr || context->GetInputTensor(3) == nullptr ||
        context->GetInputTensor(4) == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    if (context->GetPlatformInfo() == nullptr)
    {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &queryShape = context->GetInputTensor(0)->GetStorageShape();
    const gert::Shape &baseShape = context->GetInputTensor(1)->GetStorageShape();
    const gert::Shape &actualSizeShape = context->GetInputTensor(2)->GetStorageShape();
    const gert::Shape &maskShape = context->GetInputTensor(3)->GetStorageShape();

    uint32_t queryNum = static_cast<uint32_t>(queryShape[0]);
    uint32_t codeSize = static_cast<uint32_t>(queryShape[1]);
    uint32_t blockSize = static_cast<uint32_t>(baseShape[0]) * static_cast<uint32_t>(baseShape[2]);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t workspaceCoreNum = static_cast<uint32_t>(actualSizeShape[0]);
    uint32_t aicCoreNum = ascendcPlatform.GetCoreNumAic();
    uint32_t aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (queryNum == 0 || codeSize == 0 || blockSize == 0 || workspaceCoreNum == 0 || aicCoreNum == 0 ||
        aivCoreNum == 0 || workspaceCoreNum > aivCoreNum)
    {
        return ge::GRAPH_FAILED;
    }
    uint32_t blockDim = workspaceCoreNum < aicCoreNum ? workspaceCoreNum : aicCoreNum;

    AscendcDistanceFlatHammingWithMaskTilingData tiling;
    tiling.set_queryNum(queryNum);
    tiling.set_codeSize(codeSize);
    tiling.set_dim(codeSize * BINARY_BYTE_BITS);
    tiling.set_blockSize(blockSize);
    tiling.set_zRegionHeight(static_cast<uint32_t>(baseShape[2]));
    // dim=1024 uses shorter reduction bursts to bound the max-distance scratch line.
    uint32_t burstLen = codeSize == DIM_1024_CODE_SIZE ? DIM_1024_BURST_LEN : DEFAULT_BURST_LEN;
    uint32_t codeTile = burstLen * MAX_BURSTS_PER_TILE;
    tiling.set_burstLen(burstLen);
    tiling.set_codeTile(codeTile);
    tiling.set_maskRows(static_cast<uint32_t>(maskShape[0]));

    uint32_t dim = codeSize * BINARY_BYTE_BITS;
    uint32_t querySizeEachLoop = MAX_EXPANDED_ELEMENTS / dim;
    if (querySizeEachLoop > MAX_QUERY_TILE)
    {
        querySizeEachLoop = MAX_QUERY_TILE;
    }
    if (querySizeEachLoop > queryNum)
    {
        querySizeEachLoop = queryNum;
    }
    if (querySizeEachLoop == 0)
    {
        return ge::GRAPH_FAILED;
    }
    MatmulApiTiling cubeTilingIp(ascendcPlatform);
    cubeTilingIp.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8);
    cubeTilingIp.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_INT8, true);
    cubeTilingIp.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_INT32);
    cubeTilingIp.SetShape(querySizeEachLoop, codeTile, dim);
    cubeTilingIp.SetOrgShape(querySizeEachLoop, codeTile, dim);
    cubeTilingIp.SetBufferSpace(-1, -1, -1);
    int ret = cubeTilingIp.GetTiling(tiling.cubeTilingIp);
    if (ret == -1)
    {
        return ge::GRAPH_FAILED;
    }

    size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    uint64_t queryBytes = static_cast<uint64_t>(queryNum) * dim * sizeof(int8_t);
    uint64_t baseBytes = static_cast<uint64_t>(codeTile) * dim * sizeof(int8_t);
    uint64_t scoreBytes = static_cast<uint64_t>(querySizeEachLoop) * codeTile * sizeof(int32_t);
    uint64_t userWorkspaceSize = static_cast<uint64_t>(blockDim) * (queryBytes + baseBytes + scoreBytes);
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspace[0] = userWorkspaceSize + sysWorkspaceSize;

    context->SetBlockDim(blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge
{
static graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr)
    {
        return GRAPH_FAILED;
    }
    std::vector<size_t> inputDimShape{2, 4, 2, 2, 2};
    std::vector<size_t> outputDimShape{2, 2, 2};
    return ShapeCheck(context, inputDimShape, outputDimShape);
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context == nullptr)
    {
        return GRAPH_FAILED;
    }
    std::vector<DataType> inputDataType{DT_UINT8, DT_UINT8, DT_UINT32, DT_UINT8, DT_UINT8};
    std::vector<DataType> outputDataType{DT_FLOAT16, DT_FLOAT16, DT_UINT16};
    return DataTypeCheck(context, inputDataType, outputDataType);
}
}  // namespace ge

namespace ops
{
static void ConfigureRequiredTensor(OpParamDef &param, ge::DataType dataType)
{
    param.ParamType(REQUIRED);
    param.DataType({dataType});
    param.Format({ge::FORMAT_ND});
    param.UnknownShapeFormat({ge::FORMAT_ND});
}

class AscendcDistanceFlatHammingWithMask : public OpDef
{
   public:
    explicit AscendcDistanceFlatHammingWithMask(const char *name) : OpDef(name)
    {
        ConfigureRequiredTensor(this->Input("query"), ge::DT_UINT8);
        ConfigureRequiredTensor(this->Input("base"), ge::DT_UINT8);
        ConfigureRequiredTensor(this->Input("actualSize"), ge::DT_UINT32);
        ConfigureRequiredTensor(this->Input("mask"), ge::DT_UINT8);
        ConfigureRequiredTensor(this->Input("baseMask"), ge::DT_UINT8);

        ConfigureRequiredTensor(this->Output("dist"), ge::DT_FLOAT16);
        ConfigureRequiredTensor(this->Output("maxDist"), ge::DT_FLOAT16);
        ConfigureRequiredTensor(this->Output("flag"), ge::DT_UINT16);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b").AddConfig("ascend910_93");
    }
};

OP_ADD(AscendcDistanceFlatHammingWithMask);
}  // namespace ops
