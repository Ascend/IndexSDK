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


#include "distance_flat_l2_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host_common.h"

namespace {
    constexpr uint32_t INPUT_IDX_QUERY = 0;
    constexpr uint32_t INPUT_IDX_CODE = 2;
    constexpr uint32_t DIM0 = 0;
    constexpr uint32_t DIM1 = 1;
    constexpr uint32_t TOTAL_INPUT_NUM = 4;
    constexpr uint32_t TOTAL_OUTPUT_NUM = 3;
    constexpr uint32_t BYTE_SIZE_16M = 16 * 1024 * 1024;
    constexpr uint32_t QUERY_MAX_SIZE = 48;
    constexpr uint32_t CODE_PROC_PER_LOOP_MAX = 256;
    constexpr uint32_t MIN_BATCH = 64;
    enum class DIR {
        INPUT = 0,
        OUTPUT = 1
    };
}

namespace optiling {

ge::graphStatus TilingGetDimSizeByIndex(gert::TilingContext* context,
    uint32_t index, uint32_t dim, DIR dir, uint32_t &dim_size)
{
    if ((dir == DIR::INPUT && index >= TOTAL_INPUT_NUM) ||
        (dir == DIR::OUTPUT && index >= TOTAL_OUTPUT_NUM)) {
        return ge::GRAPH_FAILED;
    }

    auto shape_ptr = (dir == DIR::INPUT) ? context->GetInputShape(index) : context->GetOutputShape(index);
    if (shape_ptr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto shape_val = shape_ptr->GetStorageShape();
    if (dim > shape_val.GetDimNum()) {
        return ge::GRAPH_FAILED;
    }

    dim_size = static_cast<uint32_t>(shape_val.GetDim(dim));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingBasic(gert::TilingContext* context, DistanceFlatL2TilingData &tiling)
{
    uint32_t querySize = 0;
    uint32_t dimsSize = 0;
    uint32_t codeSize = 0;

    ge::graphStatus ret = ge::GRAPH_FAILED;
    ret = TilingGetDimSizeByIndex(context, INPUT_IDX_QUERY, DIM0, DIR::INPUT, querySize);
    if (ret != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    ret = TilingGetDimSizeByIndex(context, INPUT_IDX_QUERY, DIM1, DIR::INPUT, dimsSize);
    if (ret != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    ret = TilingGetDimSizeByIndex(context, INPUT_IDX_CODE, DIM0, DIR::INPUT, codeSize);
    if (ret != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
  
    tiling.set_querySize(querySize);
    tiling.set_dimSize(dimsSize);
    tiling.set_codeSize(codeSize * Utils::CUBE_ALIGN);
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

// 不涉及actual num的都是静态tiling信息，涉及actual num的需要在kernel侧计算
static ge::graphStatus TilingProcStaticInfo(gert::TilingContext* context, DistanceFlatL2TilingData &tiling)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    if (aicNum == 0 || aivNum == 0) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(aicNum);
    tiling.set_aivNum(aivNum);

    uint32_t querySize = tiling.get_querySize();

    uint32_t querySizeEachLoop = Utils::Min(static_cast<uint32_t>(QUERY_MAX_SIZE), querySize);
    uint32_t queryLoopTimes = Utils::DivUp(querySize, querySizeEachLoop);
    uint32_t querySizeLastLoop = querySize - (queryLoopTimes - 1) * querySizeEachLoop; // 尾部数据
    uint32_t codeSizeEachLoop = CODE_PROC_PER_LOOP_MAX;

    tiling.set_queryLoopTimes(queryLoopTimes);
    tiling.set_querySizeEachLoop(querySizeEachLoop);
    tiling.set_querySizeLastLoop(querySizeLastLoop);
    tiling.set_codeSizeEachLoop(codeSizeEachLoop);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingCube(gert::TilingContext* context, DistanceFlatL2TilingData &tiling)
{
    using namespace matmul_tiling;
    
    uint32_t querySizeEachLoop = tiling.get_querySizeEachLoop();
    uint32_t codeSizeEachLoop = tiling.get_codeSizeEachLoop();
    uint32_t dimSize = tiling.get_dimSize();

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    MatmulApiTiling cubeTilingIp(ascendcPlatform);
    cubeTilingIp.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    cubeTilingIp.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, true);
    cubeTilingIp.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    cubeTilingIp.SetShape(querySizeEachLoop, codeSizeEachLoop, dimSize);
    cubeTilingIp.SetOrgShape(querySizeEachLoop, codeSizeEachLoop, dimSize);
    cubeTilingIp.SetBufferSpace(-1, -1, -1);
    int ret = cubeTilingIp.GetTiling(tiling.cubeTilingIp);
    if (ret == -1) {
        return ge::GRAPH_FAILED;
    }

    MatmulApiTiling cubeTilingL2Norm(ascendcPlatform);
    cubeTilingL2Norm.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    cubeTilingL2Norm.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, true);
    cubeTilingL2Norm.SetCType(TPosition::LCM, CubeFormat::ND, DataType::DT_FLOAT);
    cubeTilingL2Norm.SetShape(querySizeEachLoop, querySizeEachLoop, dimSize);
    cubeTilingL2Norm.SetOrgShape(querySizeEachLoop, querySizeEachLoop, dimSize);
    cubeTilingL2Norm.SetBufferSpace(-1, -1, -1);
    ret = cubeTilingL2Norm.GetTiling(tiling.cubeTilingL2Norm);
    if (ret == -1) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    DistanceFlatL2TilingData tiling;
  
    if (context == nullptr || context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto ret = TilingBasic(context, tiling);
    if (ret != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // tiling处理静态信息
    ret = TilingProcStaticInfo(context, tiling);
    if (ret != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    ret = TilingCube(context, tiling);
    if (ret != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // workdspace
    // 对于InterateAll 异步场景 matmul的结果需要用workspace来缓存这里使用userWorkSpace
    size_t userSize = tiling.get_querySizeEachLoop() * CODE_PROC_PER_LOOP_MAX * tiling.get_aivNum() * sizeof(float);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = BYTE_SIZE_16M + userSize;

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
class DistanceFlatL2 : public OpDef {
public:
    explicit DistanceFlatL2(const char* name) : OpDef(name)
    {
        this->Input("Queries")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("codes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("L2PreNorm")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("ActualNum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("L2Distance")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("L2DistanceMin")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("Flag")
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

OP_ADD(DistanceFlatL2);
}
