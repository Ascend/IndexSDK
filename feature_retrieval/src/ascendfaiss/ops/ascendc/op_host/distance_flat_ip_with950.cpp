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

#include "distance_flat_ip_950_tiling.h"
#include "op_host_common.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace
{
constexpr uint32_t BASE_N = 128;  // N维分块基本单位，与FP32参考路径保持一致
}

namespace optiling
{
using namespace matmul_tiling;

static ge::graphStatus TilingSetInputShapeInfo(gert::TilingContext *context, DistanceFlatIPWith950TilingData &tiling)
{
    if (context == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    // 输入的第0个是query
    if (context->GetInputTensor(0) == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &queriesShape = context->GetInputTensor(0)->GetStorageShape();
    // 输入的第2个是code
    if (context->GetInputTensor(2) == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &shapedShape = context->GetInputTensor(2)->GetStorageShape();

    // query的0维是queryNum
    uint32_t queryNum = static_cast<uint32_t>(queriesShape[0]);
    // query的1维是dim
    uint32_t dim = static_cast<uint32_t>(queriesShape[1]);
    // codeNum即blockSize，等于shapedShape的第0维*第2维
    uint32_t codeNum = static_cast<uint32_t>(shapedShape[0]) * static_cast<uint32_t>(shapedShape[2]);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    // vector core个数
    uint32_t vecCoreNum = ascendcPlatform.GetCoreNumAiv();
    // cube core个数，MIX 1:2下GetBlockIdx()返回0..cubeCoreNum-1
    uint32_t cubeCoreNum = ascendcPlatform.GetCoreNumAic();
    if (vecCoreNum == 0 || cubeCoreNum == 0)
    {
        return ge::GRAPH_FAILED;
    }

    tiling.set_queryNum(queryNum);
    tiling.set_codeNum(codeNum);
    tiling.set_dim(dim);
    tiling.set_vecCoreNum(vecCoreNum);
    tiling.set_cubeCoreNum(cubeCoreNum);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingSetCubeTiling(gert::TilingContext *context, DistanceFlatIPWith950TilingData &tiling,
                                           uint64_t l1Size, uint64_t l0cSize)
{
    tiling.cubeTilingData.set_usedCoreNum(1);

    // query循环量限制最大为128
    constexpr uint32_t queryLoopLimit = 128;
    // 由于burstLen=64，因此512/64=8，这样得到的8个burst大小是32B的倍数。这个值需要考虑burst
    constexpr uint32_t codeNumEachLoop = 512;
    // 1、每次循环matmul的结果不能超过UB的大小，而910B的UB大小为192K。
    // 2、query优先，当前FlatIP最大的batch size为128
    // 3、codeNumEachLoop需要按照512对齐，因为burstLen=64，codeNumEachLoop=512时，正好一个有8个burst，占用32B大小，满足一次DataCopy的最小长度。
    // 因此设计queryNumEachLoop=128，codeNumEachLoop=512。同时实测query优先对性能更好。

    // 限制和对齐queryNumEachLoop
    uint32_t dim = tiling.get_dim();
    uint32_t queryNum = tiling.get_queryNum();
    uint32_t queryNumEachLoop = Utils::Min(queryNum, queryLoopLimit);
    queryNumEachLoop = Utils::DivUp(queryNumEachLoop, Utils::CUBE_ALIGN) * Utils::CUBE_ALIGN;

    // 设置matmul的tiling参数
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    MatmulApiTiling cubeTiling(ascendcPlatform);
    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    // true: B矩阵需转置，与kernel侧MatmulTypeB<..., true>和SetTensorB(..., true)保持一致
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, true);
    // GM输出模式：Cube结果写入GM，AIV从GM搬回UB做后处理，与DistanceFlatL2保持一致
    cubeTiling.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    cubeTiling.SetShape(queryNumEachLoop, codeNumEachLoop, dim);
    cubeTiling.SetOrgShape(queryNumEachLoop, codeNumEachLoop, dim);
    // 固定K轴分块为1，N轴分块不超过BASE_N，避免小维度场景下分块非法
    cubeTiling.SetFixSplit(1, std::min(BASE_N, codeNumEachLoop), -1);
    cubeTiling.SetBias(false);
    // 使用实际L1/L0C大小优化内存分配，避免-1导致的保守分配
    cubeTiling.SetBufferSpace(l1Size, l0cSize);

    int64_t ret = cubeTiling.GetTiling(tiling.cubeTilingData);
    if (ret == -1)
    {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr || context->GetRawTilingData() == nullptr)
    {
        return ge::GRAPH_FAILED;
    }

    DistanceFlatIPWith950TilingData tiling;
    auto ret = TilingSetInputShapeInfo(context, tiling);
    if (ret != ge::GRAPH_SUCCESS)
    {
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t cubeCoreNum = ascendcPlatform.GetCoreNumAic();
    uint32_t vecCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (cubeCoreNum == 0 || vecCoreNum == 0)
    {
        return ge::GRAPH_FAILED;
    }

    // 获取L1和L0C实际大小，用于MatmulApiTiling优化内存分配
    uint64_t l1Size = 0;
    uint64_t l0cSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize);

    ret = TilingSetCubeTiling(context, tiling, l1Size, l0cSize);
    if (ret != ge::GRAPH_SUCCESS)
    {
        return ge::GRAPH_FAILED;
    }

    // 设置使用的cube core的个数：MIX 1:2下SetBlockDim用cubeCoreNum(28)
    context->SetBlockDim(cubeCoreNum);

    // 将tiling序列化保存到TilingContext的上下文
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    // 设置TilingData数据长度。这两步完成tiling的传递
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // GM输出模式下，每个cube组(blockIdx 0..cubeCoreNum-1)分配一段matmul结果暂存空间
    // 同组AIC+2AIV共享同一段workspace，AIC写入，subBlockIdx==1的AIV读取做后处理
    uint32_t queryNum = tiling.get_queryNum();
    constexpr uint32_t queryLoopLimit = 128;
    constexpr uint32_t codeNumEachLoop = 512;
    uint32_t queryNumEachLoop = Utils::Min(queryNum, queryLoopLimit);
    queryNumEachLoop = Utils::DivUp(queryNumEachLoop, Utils::CUBE_ALIGN) * Utils::CUBE_ALIGN;
    const size_t userWorkspaceSize =
        static_cast<size_t>(queryNumEachLoop) * codeNumEachLoop * sizeof(uint16_t) * cubeCoreNum;
    const uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = userWorkspaceSize + sysWorkspaceSize;

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

    std::vector<size_t> inputDimShape{2, 2, 4, 2};  // 2: queries, 2: mask, 4: shaped, 2: actualSize;
    std::vector<size_t> outputDimShape{2, 2, 2};    // 2: dist, 2: maxDist, 2: flag;
    return ShapeCheck(context, inputDimShape, outputDimShape);
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context == nullptr)
    {
        return GRAPH_FAILED;
    }

    std::vector<DataType> inputDataType{DT_FLOAT16, DT_UINT8, DT_FLOAT16, DT_UINT32};
    std::vector<DataType> outputDataType{DT_FLOAT16, DT_FLOAT16, DT_UINT16};
    return DataTypeCheck(context, inputDataType, outputDataType);
}
}  // namespace ge

namespace ops
{
class DistanceFlatIPWith950 : public OpDef
{
   public:
    explicit DistanceFlatIPWith950(const char *name) : OpDef(name)
    {
        this->Input("queries")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("shaped")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
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

        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(DistanceFlatIPWith950);
}  // namespace ops
