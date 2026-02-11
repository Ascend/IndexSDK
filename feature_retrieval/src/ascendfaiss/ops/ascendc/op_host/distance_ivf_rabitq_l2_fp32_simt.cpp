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
#include "distance_ivf_rabitq_l2_fp32_simt_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace {
static const int32_t QUERYLUT = 1;
static const int32_t BASE = 6;
static const int32_t BASE_N = 128;
static const int32_t OFFSET = 6;
static const int32_t FLOAT32_BYTES = 4;
static const int32_t INT8_BYTES = 1;
static const int32_t INT32_BYTES = 4;
static const int32_t GM_ALIGN = 512;  // 全局内存对齐大小
static const uint32_t MAX_SHARE_MEM = 1024 * 216;
}
namespace optiling {

static void SetLutAndCodeTilingInfo(gert::TilingContext *context, DistanceIVFRabitqL2FP32SimtTilingData &tiling,
                                    int32_t ubSize)
{
    if (!context->GetInputTensor(0)) {
        return;
    }
    if (!context->GetInputTensor(QUERYLUT)) {
        return;
    }
    int32_t lutLength = context->GetInputShape(0)->GetStorageShape().GetDim(1) / 8;
    int32_t lutDimLength = context->GetInputShape(QUERYLUT)->GetStorageShape().GetDim(1);  // 一般为固定值256

    int32_t lutTotalSize = lutDimLength * lutLength * FLOAT32_BYTES;

    int32_t remianSize = (ubSize * 9 / 10) - lutTotalSize;
    int32_t lutTileNum = 0;
    int32_t lutTileLength = 0;
    int32_t lastLutTileLength = 0;

    int32_t lut_tile_length = 64;  // 需要优化 每次搬运多少条lut，每条size为256*FLOAT32_BYTES=1KB
    if (lutLength < lut_tile_length) {
        lutTileNum = 1;
        lutTileLength = lutLength;
        lastLutTileLength = lutLength;
    } else if (lutLength % lut_tile_length == 0) {
        lutTileNum = lutLength / lut_tile_length;
        lutTileLength = lut_tile_length;
        lastLutTileLength = lut_tile_length;
    } else {
        lutTileNum = lutLength / lut_tile_length + 1;
        lutTileLength = lut_tile_length;
        lastLutTileLength = lutLength % lut_tile_length;
    }

    tiling.set_lutLength(lutLength);
    tiling.set_lutDimLength(lutDimLength);
    tiling.set_lutTileNum(lutTileNum);
    tiling.set_lutTileLength(lutTileLength);
    tiling.set_lastLutTileLength(lastLutTileLength);
}

static void SetDistTilingInfo(gert::TilingContext *context, DistanceIVFRabitqL2FP32SimtTilingData &tiling,
                              int32_t ubSize)
{
    int32_t codesLength = context->GetInputShape(BASE)->GetStorageShape().GetDim(0);  // indexcode数量
    int32_t dimLength = context->GetInputShape(BASE)->GetStorageShape().GetDim(1);  // indexcode维度 存储类型为int8

    uint64_t remainingSize = (ubSize * 9 / 10);
    // 64 为 minResult 需要的空间，4个1分别表示 sumResult, L1, L2, distResult 占用的空间
    int32_t dist_tile_length = remainingSize / ((64 + 1 + 1 + 1 + 1) * FLOAT32_BYTES) / 16 * 16;

    tiling.set_distTileLength(dist_tile_length);
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    DistanceIVFRabitqL2FP32SimtTilingData tiling;
    ASCENDC_RETURN_IF_NOT(context != nullptr, ge::GRAPH_FAILED);
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t aiCubeNum = ascendcPlatform.GetCoreNumAic();
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t l0cSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize);
    if (aiCubeNum == 0 || ubSize == 0) {
        return ge::GRAPH_FAILED;
    }
    if (!context->GetInputTensor(0)) {
        return ge::GRAPH_FAILED;
    }
    int32_t dimLength = context->GetInputShape(BASE)->GetStorageShape().GetDim(1);  // indexcode维度 存储类型为int8
    tiling.set_dimLength(dimLength);
    SetLutAndCodeTilingInfo(context, tiling, ubSize);
    SetDistTilingInfo(context, tiling, ubSize);

    context->SetLocalMemorySize(MAX_SHARE_MEM);

    int32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    context->SetBlockDim(aivNum);
    int32_t codeBlockLength = context->GetInputShape(BASE)->GetStorageShape().GetDim(0);  // indexcode数量
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);

    int32_t mmoutBufferSize = (codeBlockLength * FLOAT32_BYTES + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN;
    tiling.set_mmoutBufferSize(mmoutBufferSize);
    tiling.set_codeBlockLength(codeBlockLength);

    size_t usrSize = static_cast<size_t>(aivNum * mmoutBufferSize);

    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    // 获取输入形状
    const gert::Shape *input_shape = context->GetInputShape(BASE);

    // 检查输入形状是否合法
    if (input_shape == nullptr || input_shape->GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    // 获取输入形状的维度
    int64_t n = input_shape->GetDim(0);

    // 设置输出形状为 [n, 1]
    gert::Shape *output_shape = context->GetOutputShape(0);
    output_shape->SetDimNum(1);
    output_shape->SetDim(0, n);

    gert::Shape *output_shape1 = context->GetOutputShape(1);
    output_shape1->SetDimNum(1);
    output_shape1->SetDim(0, n / 32);

    gert::Shape *output_shape2 = context->GetOutputShape(2);
    output_shape2->SetDimNum(1);
    output_shape2->SetDim(0, 16);

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    auto data_type = context->GetInputDataType(0);
    context->SetOutputDataType(0, data_type);
    context->SetOutputDataType(1, data_type);
    context->SetOutputDataType(2, ge::DataType::DT_UINT16);
    return GRAPH_SUCCESS;
}
}

/* input:
 *      query: 1 * d
 *      querylut: 1 * d / 4 * 16
 *      centroidslut: nlist * d / 4 * 16
 *      centroidsid: corenum
 *      base: ni * d/8
 *      offset: corenum
 *      actual_size: corenum
 *      indexl2: ni
 *      indexl1: ni
 *      indexesoffset: corenum
 * output:
 *      result: ||query - index||^2 - ||query - centroid||^2 = indexl2 - indexl1 * <query - centroid,
 *              indexcode> -> ni * corenum
 *      max_result: 分块排序
 *      flag:
*/
namespace ops {
class DistanceIVFRabitqL2FP32Simt : public OpDef {
public:
    explicit DistanceIVFRabitqL2FP32Simt(const char *name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("querylut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("centroidslut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("queryid")  // 需要计算的查询的id
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("centroidsid")  // 需要计算的质心的id
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("centroidsl2")  // 需要计算的质心到查询的l2距离
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("base")  // codes地址
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("offset")  // 偏移地址
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("actual_size")  // 计算大小
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("indexl2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("indexl1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("indexl2offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("indexl1offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("result")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("min_result")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("flag")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b").AddConfig("ascend950");
    }
};

OP_ADD(DistanceIVFRabitqL2FP32Simt);
}
