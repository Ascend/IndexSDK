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

#include <iostream>
#include "distance_ivf_rabitq_l2_fp32_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"


namespace {
static const int32_t BASE = 6;
static const int32_t BASE_N = 128;
static const int32_t FLOAT32_BYTES = 4;
static const int32_t INT8_BYTES = 1;
static const int32_t INT32_BYTES = 4;
static const int32_t GM_ALIGN = 512;  // 全局内存对齐大小
}
namespace optiling {

static void SetCodeTilingInfo(gert::TilingContext *context, DistanceIVFRabitqL2FP32TilingData &tiling, int32_t ubSize)
{
    int32_t dimLength = tiling.get_dimLength();
    uint64_t remianingSize = (ubSize * 9 / 10) - 6 * dimLength * FLOAT32_BYTES;
    int32_t code_tile_length =
        remianingSize /
        (dimLength / 8 * INT8_BYTES + dimLength * FLOAT32_BYTES + FLOAT32_BYTES + (64 + 1 + 1 + 1) * FLOAT32_BYTES) /
        8 * 8;
    tiling.set_codeTileLength(code_tile_length);
    std::cout << "code tile length: " << code_tile_length << std::endl;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    DistanceIVFRabitqL2FP32TilingData tiling;
    ASCENDC_RETURN_IF_NOT(context != nullptr, ge::GRAPH_FAILED);
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0) {
        return ge::GRAPH_FAILED;
    }
    if (!context->GetInputTensor(BASE)) {
        return ge::GRAPH_FAILED;
    }
    int32_t dimLength = context->GetInputShape(BASE)->GetStorageShape().GetDim(1);  // indexcode维度 存储类型为int8
    tiling.set_dimLength(dimLength * 8);
    SetCodeTilingInfo(context, tiling, ubSize);
    int32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    context->SetBlockDim(aivNum);
    int32_t codeBlockLength = context->GetInputShape(BASE)->GetStorageShape().GetDim(0);  // indexcode数量
    tiling.set_codeBlockLength(codeBlockLength);
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    size_t usrSize = 0;
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
*               indexcode> -> ni * corenum
 *      max_result: 分块排序
 *      flag:
*/
namespace ops {
class DistanceIVFRabitqL2FP32 : public OpDef {
public:
    explicit DistanceIVFRabitqL2FP32(const char *name) : OpDef(name)
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
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(DistanceIVFRabitqL2FP32);
}
