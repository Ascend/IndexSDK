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

#include "ascendc_distance_batch_mask_generator_with_extra.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host_common.h"

namespace {
const uint32_t QUERY_TOKEN_SET_INPUT_DIM = 1;
const uint32_t DB_TIME_STAMP_INPUT_DIM = 2;

const uint32_t DB_TIME_STAMP_SIZE_DIM = 0;
const uint32_t QUERY_TOKEN_SET_BATCH_DIM = 0;
const uint32_t QUERY_TOKEN_SET_TOKEN_CNT_DIM = 1;

const uint32_t TILE_LEN = 8192;
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    static uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    const auto dbTimeStampPtr = context->GetInputShape(DB_TIME_STAMP_INPUT_DIM);
    const auto queryTokenSetPtr = context->GetInputShape(QUERY_TOKEN_SET_INPUT_DIM);
    if ((dbTimeStampPtr == nullptr) || (queryTokenSetPtr == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    auto dbTimeStampShape = dbTimeStampPtr->GetStorageShape();
    auto queryTokenSetShape = queryTokenSetPtr->GetStorageShape();

    AscendcDistanceBatchMaskGeneratorWithExtraTilingData tiling;
    uint32_t dbLen;
    uint32_t batchSize;
    uint32_t tokenCnt;
    uint32_t totalTileNum;
    uint32_t formerNum;
    uint32_t formerRepeatNum;
    uint32_t tailNum;
    uint32_t tailRepeatNum;
    uint32_t tileLen = TILE_LEN;
    dbLen = dbTimeStampShape.GetDim(DB_TIME_STAMP_SIZE_DIM);
    batchSize = queryTokenSetShape.GetDim(QUERY_TOKEN_SET_BATCH_DIM);
    tokenCnt = queryTokenSetShape.GetDim(QUERY_TOKEN_SET_TOKEN_CNT_DIM);
    if (tokenCnt > 32768) {
        tileLen = 4096;
    }
    totalTileNum = dbLen / tileLen;
    if (totalTileNum <= coreNum) {
        coreNum = totalTileNum;
    }
    formerNum = totalTileNum % coreNum;
    formerRepeatNum = 0;
    if (formerNum != 0) {
        formerRepeatNum = totalTileNum / coreNum + 1;
    }
    tailNum = coreNum - formerNum;
    tailRepeatNum = 0;
    if (tailNum != 0) {
        tailRepeatNum = totalTileNum / coreNum;
    }

    tiling.set_batchSize(batchSize);
    tiling.set_tokenCnt(tokenCnt);
    tiling.set_tileLen(tileLen);
    tiling.set_formerNum(formerNum);
    tiling.set_formerRepeatNum(formerRepeatNum);
    tiling.set_tailRepeatNum(tailRepeatNum);

    context->SetBlockDim(coreNum);
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext*)
{
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext*)
{
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class AscendcDistanceBatchMaskGeneratorWithExtra : public OpDef {
public:
    explicit AscendcDistanceBatchMaskGeneratorWithExtra(const char* name) : OpDef(name)
    {
        this->Input("query_time_stamp")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("query_token_set")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("db_time_stamp")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("db_divisor")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("db_remainder")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("extra_mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("extra_mask_attr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("distance_mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(AscendcDistanceBatchMaskGeneratorWithExtra);
} // namespace ops