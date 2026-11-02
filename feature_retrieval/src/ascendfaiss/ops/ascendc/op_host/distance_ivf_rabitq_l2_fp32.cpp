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

namespace
{
static const int32_t METRIC_L2 = 0;
static const int32_t METRIC_INNER_PRODUCT = 1;
static const int32_t QUERYLUT = 1;
static const int32_t BASE = 6;
static const int32_t BASE_N = 128;
static const int32_t FLOAT32_BYTES = 4;
static const int32_t INT8_BYTES = 1;
static const int32_t HALF_BYTES = 2;
static const int32_t INT32_BYTES = 4;
static const int32_t GM_ALIGN = 512;  // 全局内存对齐大小
}  // namespace
namespace optiling
{

static ge::graphStatus SetLutAndCodeTilingInfo(gert::TilingContext *context, DistanceIVFRabitqL2FP32TilingData &tiling,
                                               int32_t ubSize)
{
    int32_t dimLength = tiling.get_dimLength();
    int32_t lutLength = tiling.get_lutLength();
    int32_t lutDimLength = context->GetInputShape(QUERYLUT)->GetStorageShape().GetDim(1);  // 一般为固定值256

    int32_t codeBlockLength = context->GetInputShape(BASE)->GetStorageShape().GetDim(0);
    const int32_t maskLength = 64;
    int32_t burstMaskSize = ((codeBlockLength + maskLength - 1) / maskLength) * sizeof(uint64_t);

    int32_t lutTileNum = 0;
    int32_t lutTileLength = 0;
    int32_t lastLutTileLength = 0;

    int32_t lutTotalSize = lutDimLength * lutLength * FLOAT32_BYTES;
    int32_t remianSize = (ubSize * 9 / 10) - lutTotalSize - burstMaskSize;

    // idx_calc_que 存放gather偏移，占用内存 dimLength / 8 * INT32_BYTES
    // 当前 tile 在第二次 Cast 后释放 codes/half，随后复用同一块 codes_in 预取下一块，
    // 因此预算里仍然只留一份 codes_in，避免把 code tile 缩短。
    // codes_half_que、codes_int32_que 各一份。select_out_que 存放 gather 结果，ip_out_que 存放内积。
    int32_t code_tile_length =
        (remianSize - lutLength * INT32_BYTES) / (lutLength * INT8_BYTES + lutLength * HALF_BYTES +
                                                  lutLength * INT32_BYTES + lutLength * FLOAT32_BYTES + FLOAT32_BYTES);
    // 多比特下 lutLength 随 quantBits 膨胀，极端 dim 可能耗尽 UB 预算：
    // code_tile_length <= 0 会让 kernel 侧 tile 数与 GM 偏移错乱，lut_tile_length 为 0 会让
    // lutLength % lut_tile_length 整数除零，这里直接判定非法并在 tiling 阶段失败。
    if (code_tile_length <= 0)
    {
        return ge::GRAPH_FAILED;
    }
    tiling.set_codeTileLength(code_tile_length);
    std::cout << "code tile length: " << code_tile_length << std::endl;

    // select_out_que 可复用，用以存储分段输入的 centroid LUT
    // select_out_que 大小为 dimLength / 8 * code_tile_lengt 个float，而 LUT 每行为 lutDimLength 个 float
    // 据此可计算每次搬运多少行LUT
    int32_t lut_tile_length = lutLength * code_tile_length / lutDimLength;
    if (lut_tile_length <= 0)
    {
        return ge::GRAPH_FAILED;
    }
    if (lutLength < lut_tile_length)
    {
        lutTileNum = 1;
        lutTileLength = lutLength;
        lastLutTileLength = lutLength;
    }
    else if (lutLength % lut_tile_length == 0)
    {
        lutTileNum = lutLength / lut_tile_length;
        lutTileLength = lut_tile_length;
        lastLutTileLength = lut_tile_length;
    }
    else
    {
        lutTileNum = lutLength / lut_tile_length + 1;
        lutTileLength = lut_tile_length;
        lastLutTileLength = lutLength % lut_tile_length;
    }

    tiling.set_lutLength(lutLength);
    tiling.set_lutDimLength(lutDimLength);
    tiling.set_lutTileNum(lutTileNum);
    tiling.set_lutTileLength(lutTileLength);
    tiling.set_lastLutTileLength(lastLutTileLength);
    return ge::GRAPH_SUCCESS;
}

static void SetDistTilingInfo(gert::TilingContext *context, DistanceIVFRabitqL2FP32TilingData &tiling, int32_t ubSize)
{
    uint64_t remainingSize = (ubSize * 9 / 10);
    // 64 为 minResult 需要的空间，4个1分别表示 sumResult, L1, L2, distResult 占用的空间
    int32_t dist_tile_length = remainingSize / ((64 + 1 + 1 + 1 + 1) * FLOAT32_BYTES) / 16 * 16;

    tiling.set_distTileLength(dist_tile_length);
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    DistanceIVFRabitqL2FP32TilingData tiling;
    ASCENDC_RETURN_IF_NOT(context != nullptr, ge::GRAPH_FAILED);

    const auto *attrs = context->GetAttrs();
    ASCENDC_RETURN_IF_NOT(attrs != nullptr, ge::GRAPH_FAILED);
    const int *metricTypeAttr = attrs->GetAttrPointer<int>(0);
    const int *quantBitsAttr = attrs->GetAttrPointer<int>(1);
    ASCENDC_RETURN_IF_NOT(metricTypeAttr != nullptr && quantBitsAttr != nullptr, ge::GRAPH_FAILED);
    int metricType = *metricTypeAttr;
    if (metricType != METRIC_L2 && metricType != METRIC_INNER_PRODUCT)
    {
        return ge::GRAPH_FAILED;
    }
    tiling.set_metricType(metricType);
    int quantBits = *quantBitsAttr;
    if (quantBits != 1 && quantBits != 2 && quantBits != 4)
    {
        return ge::GRAPH_FAILED;
    }
    tiling.set_quantBits(quantBits);

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0)
    {
        return ge::GRAPH_FAILED;
    }
    if (!context->GetInputTensor(BASE))
    {
        return ge::GRAPH_FAILED;
    }
    int32_t codeLength = context->GetInputShape(BASE)->GetStorageShape().GetDim(1);  // bytes per encoded vector
    tiling.set_lutLength(codeLength);
    tiling.set_dimLength(codeLength * 8 / quantBits);
    ASCENDC_RETURN_IF_NOT(SetLutAndCodeTilingInfo(context, tiling, ubSize) == ge::GRAPH_SUCCESS, ge::GRAPH_FAILED);
    SetDistTilingInfo(context, tiling, ubSize);
    int32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    context->SetBlockDim(aivNum);
    int32_t codeBlockLength = context->GetInputShape(BASE)->GetStorageShape().GetDim(0);  // indexcode数量
    tiling.set_codeBlockLength(codeBlockLength);
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    const int32_t maskLength = 64;
    size_t usrSize = static_cast<size_t>(aivNum) *
                     static_cast<size_t>((codeBlockLength + maskLength - 1) / maskLength) * sizeof(uint64_t);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge
{
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    // 获取输入形状
    const gert::Shape *input_shape = context->GetInputShape(BASE);

    // 检查输入形状是否合法
    if (input_shape == nullptr || input_shape->GetDimNum() != 2)
    {
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
    output_shape2->SetDim(0, 32);

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
}  // namespace ge

/* input:
 *      queryl2: batch
 *      querylut: batch * d / 4 * 16
 *      centroidslut: nlist * d / 4 * 16
 *      centroidsid: corenum
 *      base: ni * (d * quant_bits / 8)
 *      offset: corenum
 *      actual_size: corenum
 *      indexl2: ni
 *      indexl1: ni
 *      indexesoffset: corenum
 * output:
 *      result: ||o - q||^2 = ||o - c||^2 + ||q - c||^2 - 2 <q - c, o - c> =  -> ni (IP: -0.5 * (||o - q||^2 - ||q||^2 -
 * ||o||^2)) max_result: 分块排序 flag:
 */
namespace ops
{
class DistanceIVFRabitqL2FP32 : public OpDef
{
   public:
    explicit DistanceIVFRabitqL2FP32(const char *name) : OpDef(name)
    {
        this->Input("queryl2")
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
        this->Input("ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("filter_attrs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
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

        this->Attr("metric_type").AttrType(REQUIRED).Int();
        this->Attr("quant_bits").AttrType(REQUIRED).Int();
        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b").AddConfig("ascend910_93").AddConfig("ascend950");
    }
};

OP_ADD(DistanceIVFRabitqL2FP32);
}  // namespace ops
