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

#include "distance_flat_l2_950_tiling.h"
#include "op_host_common.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace
{
constexpr uint32_t INPUT_IDX_QUERY = 0;
constexpr uint32_t INPUT_IDX_CODE = 2;
constexpr uint32_t DIM0 = 0;
constexpr uint32_t DIM1 = 1;
constexpr uint32_t TOTAL_INPUT_NUM = 4;
constexpr uint32_t TOTAL_OUTPUT_NUM = 3;

// 注意：querySizeEachLoop*codeSizeEachLoop 由 12288 变为 16384，UB 占用增加约 48KB(innerProduct/l2Distance 等)。
constexpr uint32_t QUERY_MAX_SIZE = 32;
// 每轮code粒度。必须为BURST_LEN(64)*8=512：一轮恰好8个burst，极值占8个pair=16个half=32B，
// 正好是一次DataCopy的最小长度，使极值拷贝的写入长度与该轮目的地址步进一致，避免相邻轮次GM地址重叠。
// 【跨文件强约束】本常量必须与 op_kernel/distance_flat_l2_with950.cpp 的 CORE_PROC_CODE_GRAIN 严格相等(均为512)：
// kernel 侧 codeBatchSize=AlignUp(CORE_PROC_CODE_GRAIN/MIN_BATCH, CUBE_ALIGN/2) 与目的地址步进
// codeCopyOutBatchOffset=DivUp(codeSizeEachLoop, MIN_BATCH) 仅在两者同为512时相等(均为8)，
// kernel 的 CopyOutL2DistMin 输出布局依赖该等式；本文件 TilingFunc 中
// userSize(queryAlignAll * CODE_PROC_PER_LOOP_MAX) 同样与之耦合。
// 任一侧单独调整会同时静默破坏极值输出布局与 workspace 大小，必须两侧同步修改。
constexpr uint32_t CODE_PROC_PER_LOOP_MAX = 512;
constexpr uint32_t MIN_BATCH = 64;
enum class DIR
{
    INPUT = 0,
    OUTPUT = 1
};
}  // namespace

namespace optiling
{

ge::graphStatus TilingGetDimSizeByIndex(gert::TilingContext *context, uint32_t index, uint32_t dim, DIR dir,
                                        uint32_t &dim_size)
{
    if ((dir == DIR::INPUT && index >= TOTAL_INPUT_NUM) || (dir == DIR::OUTPUT && index >= TOTAL_OUTPUT_NUM))
    {
        return ge::GRAPH_FAILED;
    }

    auto shape_ptr = (dir == DIR::INPUT) ? context->GetInputShape(index) : context->GetOutputShape(index);
    if (shape_ptr == nullptr)
    {
        return ge::GRAPH_FAILED;
    }

    auto shape_val = shape_ptr->GetStorageShape();
    if (dim > shape_val.GetDimNum())
    {
        return ge::GRAPH_FAILED;
    }

    dim_size = static_cast<uint32_t>(shape_val.GetDim(dim));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingBasic(gert::TilingContext *context, DistanceFlatL2With950TilingData &tiling)
{
    uint32_t querySize = 0;
    uint32_t dimsSize = 0;
    uint32_t codeSize = 0;

    ge::graphStatus ret = ge::GRAPH_FAILED;
    ret = TilingGetDimSizeByIndex(context, INPUT_IDX_QUERY, DIM0, DIR::INPUT, querySize);
    if (ret != ge::GRAPH_SUCCESS)
    {
        return ge::GRAPH_FAILED;
    }

    ret = TilingGetDimSizeByIndex(context, INPUT_IDX_QUERY, DIM1, DIR::INPUT, dimsSize);
    if (ret != ge::GRAPH_SUCCESS)
    {
        return ge::GRAPH_FAILED;
    }

    ret = TilingGetDimSizeByIndex(context, INPUT_IDX_CODE, DIM0, DIR::INPUT, codeSize);
    if (ret != ge::GRAPH_SUCCESS)
    {
        return ge::GRAPH_FAILED;
    }

    tiling.set_querySize(querySize);
    tiling.set_dimSize(dimsSize);
    tiling.set_codeSize(codeSize * Utils::CUBE_ALIGN);
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

// 不涉及actual num的都是静态tiling信息，涉及actual num的需要在kernel侧计算
static ge::graphStatus TilingProcStaticInfo(gert::TilingContext *context, DistanceFlatL2With950TilingData &tiling)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    if (aicNum == 0 || aivNum == 0)
    {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(aicNum);
    tiling.set_aivNum(aivNum);
    tiling.set_aicNum(aicNum);

    uint32_t querySize = tiling.get_querySize();

    uint32_t querySizeEachLoop = Utils::Min(static_cast<uint32_t>(QUERY_MAX_SIZE), querySize);
    uint32_t queryLoopTimes = Utils::DivUp(querySize, querySizeEachLoop);
    uint32_t querySizeLastLoop = querySize - (queryLoopTimes - 1) * querySizeEachLoop;  // 尾部数据
    uint32_t codeSizeEachLoop = CODE_PROC_PER_LOOP_MAX;

    tiling.set_queryLoopTimes(queryLoopTimes);
    tiling.set_querySizeEachLoop(querySizeEachLoop);
    tiling.set_querySizeLastLoop(querySizeLastLoop);
    tiling.set_codeSizeEachLoop(codeSizeEachLoop);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingCube(gert::TilingContext *context, DistanceFlatL2With950TilingData &tiling)
{
    using namespace matmul_tiling;

    uint32_t querySizeEachLoop = tiling.get_querySizeEachLoop();
    uint32_t codeSizeEachLoop = tiling.get_codeSizeEachLoop();
    uint32_t dimSize = tiling.get_dimSize();
    // Ascend950 MIX 1:2：matmul tiling 的 M 必须是 CUBE_ALIGN(16) 的整数倍，
    // 否则 Cube 会按 16 向上取整导致 C 矩阵按填充后的 M 写出，越界污染相邻缓冲区
    // 实际参与计算的 query 行数仍由 kernel 侧的 SetTail 校正为真实值。

    // code 矩阵每个 codeLoop 只从 HBM 读一遍（此前 M 被 UB 限制在 querySizeEachLoop，
    // 导致 code 被重复读取 queryLoopTimes 遍）。后处理仍按 querySizeEachLoop 分块，
    // 故 L2Norm matmul 的 M/N 仍用「分块对齐」尺寸。
    uint32_t queryAlignEpi = Utils::DivUp(querySizeEachLoop, Utils::CUBE_ALIGN) * Utils::CUBE_ALIGN;
    uint32_t queryAlignAll = Utils::DivUp(tiling.get_querySize(), Utils::CUBE_ALIGN) * Utils::CUBE_ALIGN;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    MatmulApiTiling cubeTilingIp(ascendcPlatform);
    cubeTilingIp.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    cubeTilingIp.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, true);
    cubeTilingIp.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    cubeTilingIp.SetShape(queryAlignAll, codeSizeEachLoop, dimSize);
    cubeTilingIp.SetOrgShape(queryAlignAll, codeSizeEachLoop, dimSize);
    cubeTilingIp.SetBufferSpace(-1, -1, -1);
    int ret = cubeTilingIp.GetTiling(tiling.cubeTilingIp);
    if (ret == -1)
    {
        return ge::GRAPH_FAILED;
    }

    MatmulApiTiling cubeTilingL2Norm(ascendcPlatform);
    cubeTilingL2Norm.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    cubeTilingL2Norm.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, true);
    cubeTilingL2Norm.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    cubeTilingL2Norm.SetShape(queryAlignEpi, queryAlignEpi, dimSize);
    cubeTilingL2Norm.SetOrgShape(queryAlignEpi, queryAlignEpi, dimSize);
    cubeTilingL2Norm.SetBufferSpace(-1, -1, -1);
    ret = cubeTilingL2Norm.GetTiling(tiling.cubeTilingL2Norm);
    if (ret == -1)
    {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    DistanceFlatL2With950TilingData tiling;

    if (context == nullptr || context->GetRawTilingData() == nullptr)
    {
        return ge::GRAPH_FAILED;
    }

    auto ret = TilingBasic(context, tiling);
    if (ret != ge::GRAPH_SUCCESS)
    {
        return ge::GRAPH_FAILED;
    }

    // tiling处理静态信息
    ret = TilingProcStaticInfo(context, tiling);
    if (ret != ge::GRAPH_SUCCESS)
    {
        return ge::GRAPH_FAILED;
    }

    ret = TilingCube(context, tiling);
    if (ret != ge::GRAPH_SUCCESS)
    {
        return ge::GRAPH_FAILED;
    }

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // MIX 1:2中两个AIV可能并发发起Matmul请求。分别为每个AIV保留IP和query范数的GM中间结果。
    // (queryAlignEpi) 分配。两者都必须与 kernel 侧 ipElemCnt/normElemCnt 及 matmul tiling 的 M 一致。
    size_t queryAlignEpi = Utils::DivUp(tiling.get_querySizeEachLoop(), Utils::CUBE_ALIGN) * Utils::CUBE_ALIGN;
    size_t queryAlignAll = Utils::DivUp(tiling.get_querySize(), Utils::CUBE_ALIGN) * Utils::CUBE_ALIGN;
    size_t userSize =
        (queryAlignAll * CODE_PROC_PER_LOOP_MAX + queryAlignEpi * queryAlignEpi) * tiling.get_aivNum() * sizeof(float);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    currentWorkspace[0] = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize()) + userSize;

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge
{
static ge::graphStatus InferShape(gert::InferShapeContext *) { return GRAPH_SUCCESS; }

static graphStatus InferDataType(gert::InferDataTypeContext *) { return GRAPH_SUCCESS; }
}  // namespace ge

namespace ops
{
class DistanceFlatL2With950 : public OpDef
{
   public:
    explicit DistanceFlatL2With950(const char *name) : OpDef(name)
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

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(DistanceFlatL2With950);
}  // namespace ops
