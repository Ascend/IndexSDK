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

#include "ascendc_distance_batch_val_mask_generator.h"

#include "op_host_common.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace
{
constexpr uint32_t QUERY_TIME_STAMP_INPUT_DIM = 0;
constexpr uint32_t QUERY_TOKEN_SET_INPUT_DIM = 1;
constexpr uint32_t DB_TIME_STAMP_INPUT_DIM = 2;
constexpr uint32_t QUERY_TIME_STAMP1_DIM = 1;
constexpr uint32_t DB_TIME_STAMP_SIZE_DIM = 0;
constexpr uint32_t QUERY_TOKEN_SET_BATCH_DIM = 0;
constexpr uint32_t QUERY_TOKEN_SET_TOKEN_CNT_DIM = 1;
constexpr uint32_t TILE_LEN = 8192;
constexpr uint32_t TILE_LEN_SMALL = 4096;
constexpr uint32_t TOKEN_CNT_THRES = 24576;
constexpr uint32_t LEN_OF_EIGHT = 8;
}  // namespace

namespace optiling
{
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0)
    {
        return ge::GRAPH_FAILED;
    }

    const auto queryTimeStampPtr = context->GetInputShape(QUERY_TIME_STAMP_INPUT_DIM);
    const auto queryTokenSetPtr = context->GetInputShape(QUERY_TOKEN_SET_INPUT_DIM);
    const auto dbTimeStampPtr = context->GetInputShape(DB_TIME_STAMP_INPUT_DIM);
    if (queryTimeStampPtr == nullptr || queryTokenSetPtr == nullptr || dbTimeStampPtr == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    const auto queryTimeStampShape = queryTimeStampPtr->GetStorageShape();
    const auto queryTokenSetShape = queryTokenSetPtr->GetStorageShape();
    const auto dbTimeStampShape = dbTimeStampPtr->GetStorageShape();
    if (queryTimeStampShape.GetDim(QUERY_TIME_STAMP1_DIM) != LEN_OF_EIGHT)
    {
        ERROR_LOG("query_time_stamp.shape[1] must be %u.", LEN_OF_EIGHT);
        return ge::GRAPH_FAILED;
    }

    const uint32_t dbLen = dbTimeStampShape.GetDim(DB_TIME_STAMP_SIZE_DIM);
    uint32_t tileLen = TILE_LEN;
    const uint32_t tokenCnt = queryTokenSetShape.GetDim(QUERY_TOKEN_SET_TOKEN_CNT_DIM);
    if (tokenCnt > TOKEN_CNT_THRES)
    {
        tileLen = TILE_LEN_SMALL;
    }
    if (dbLen == 0 || dbLen % tileLen != 0)
    {
        ERROR_LOG("total_db_num must be a nonzero multiple of %u.", tileLen);
        return ge::GRAPH_FAILED;
    }

    const uint32_t totalTileNum = dbLen / tileLen;
    coreNum = totalTileNum < coreNum ? totalTileNum : coreNum;
    const uint32_t formerNum = totalTileNum % coreNum;
    const uint32_t formerRepeatNum = formerNum == 0 ? 0 : totalTileNum / coreNum + 1;
    const uint32_t tailRepeatNum = totalTileNum / coreNum;

    AscendcDistanceBatchValMaskGeneratorTilingData tiling;
    tiling.set_batchSize(queryTokenSetShape.GetDim(QUERY_TOKEN_SET_BATCH_DIM));
    tiling.set_tokenCnt(tokenCnt);
    tiling.set_tileLen(tileLen);
    tiling.set_formerNum(formerNum);
    tiling.set_formerRepeatNum(formerRepeatNum);
    tiling.set_tailRepeatNum(tailRepeatNum);

    context->SetBlockDim(coreNum);
    if (context->GetRawTilingData() == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge
{
static graphStatus InferShape(gert::InferShapeContext*) { return GRAPH_SUCCESS; }
static graphStatus InferDataType(gert::InferDataTypeContext*) { return GRAPH_SUCCESS; }
}  // namespace ge

namespace ops
{
static void ConfigureRequiredTensor(OpParamDef& param, ge::DataType dataType)
{
    param.ParamType(REQUIRED);
    param.DataType({dataType});
    param.Format({ge::FORMAT_ND});
    param.UnknownShapeFormat({ge::FORMAT_ND});
}

class AscendcDistanceBatchValMaskGenerator : public OpDef
{
   public:
    explicit AscendcDistanceBatchValMaskGenerator(const char* name) : OpDef(name)
    {
        ConfigureRequiredTensor(this->Input("query_time_stamp"), ge::DT_INT32);
        ConfigureRequiredTensor(this->Input("query_token_set"), ge::DT_UINT8);
        ConfigureRequiredTensor(this->Input("db_time_stamp"), ge::DT_INT32);
        ConfigureRequiredTensor(this->Input("db_divisor"), ge::DT_INT32);
        ConfigureRequiredTensor(this->Input("db_remainder"), ge::DT_UINT8);
        ConfigureRequiredTensor(this->Input("extra_val_filter"), ge::DT_INT16);
        ConfigureRequiredTensor(this->Input("extra_val_attr"), ge::DT_INT16);
        ConfigureRequiredTensor(this->Output("distance_mask"), ge::DT_UINT8);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b").AddConfig("ascend910_93");
    }
};

OP_ADD(AscendcDistanceBatchValMaskGenerator);
}  // namespace ops
