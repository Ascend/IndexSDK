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


#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

#include "op_host_common.h"
#include "distance_flat_ip_tiling.h"

namespace optiling {
using namespace matmul_tiling;

static ge::graphStatus TilingSetInputShapeInfo(gert::TilingContext* context, DistanceFlatIPTilingData &tiling)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // 输入的第0个是query
    if (context->GetInputTensor(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &queriesShape = context->GetInputTensor(0)->GetStorageShape();
    // 输入的第2个是code
    if (context->GetInputTensor(2) == nullptr) {
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
    if (vecCoreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    tiling.set_queryNum(queryNum);
    tiling.set_codeNum(codeNum);
    tiling.set_dim(dim);
    tiling.set_vecCoreNum(vecCoreNum);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingSetCubeTiling(gert::TilingContext* context, DistanceFlatIPTilingData &tiling)
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
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    cubeTiling.SetCType(TPosition::VECCALC, CubeFormat::ND, DataType::DT_FLOAT16);
    cubeTiling.SetShape(queryNumEachLoop, codeNumEachLoop, dim);
    cubeTiling.SetOrgShape(queryNumEachLoop, codeNumEachLoop, dim);
    cubeTiling.SetBufferSpace(-1, -1, -1);

    int64_t ret = cubeTiling.GetTiling(tiling.cubeTilingData);
    if (ret == -1) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr || context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    
    DistanceFlatIPTilingData tiling;
    auto ret = TilingSetInputShapeInfo(context, tiling);
    if (ret != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    ret = TilingSetCubeTiling(context, tiling);
    if (ret != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t cubeCoreNum = ascendcPlatform.GetCoreNumAic();
    if (cubeCoreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    // 设置使用的cube core的个数
    context->SetBlockDim(cubeCoreNum);

    // 将tiling序列化保存到TilingContext的上下文
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    // 设置TilingData数据长度。这两步完成tiling的传递
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // 使用了异步的matmul才需要设置Workspace
    const size_t userWorkspaceSize = 0;
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
static graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    std::vector<size_t> inputDimShape {2, 2, 4, 2};  // 2: queries, 2: mask, 4: shaped, 2: actualSize;
    std::vector<size_t> outputDimShape {2, 2, 2};  // 2: dist, 2: maxDist, 2: flag;
    return ShapeCheck(context, inputDimShape, outputDimShape);
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    std::vector<DataType> inputDataType {DT_FLOAT16, DT_UINT8, DT_FLOAT16, DT_UINT32};
    std::vector<DataType> outputDataType {DT_FLOAT16, DT_FLOAT16, DT_UINT16};
    return DataTypeCheck(context, inputDataType, outputDataType);
}
}


namespace ops {
class DistanceFlatIP : public OpDef {
public:
    explicit DistanceFlatIP(const char* name) : OpDef(name)
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

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);

        this->AICore().AddConfig("ascend910b")
            .AddConfig("ascend910_93").AddConfig("ascend950");
    }
};

OP_ADD(DistanceFlatIP);
}
