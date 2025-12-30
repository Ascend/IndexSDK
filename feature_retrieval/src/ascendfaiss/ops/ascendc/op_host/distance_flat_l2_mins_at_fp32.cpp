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


#include "distance_flat_l2_mins_at_fp32_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
    static const int64_t BLOCK_SIZE = 32;
    static const int64_t FLOAT32_BYTES = 4;
}

namespace optiling {
    static void SetKernelLoopInfo(DistanceFlatL2MinsAtFP32TilingData& tiling, int32_t ubSize, int32_t codesNumLength, int32_t dimLength)
    {
        int32_t tileNum = 0;
        int32_t tileLength = 0;
        int32_t lastTileLength = 0;
        int32_t code_num_length = (ubSize - 2048 - dimLength * FLOAT32_BYTES * 2) / (dimLength * FLOAT32_BYTES * 2 + 64 * FLOAT32_BYTES + 64 * FLOAT32_BYTES + FLOAT32_BYTES * 2) / 16 * 16;
        if (code_num_length < 1) {
            code_num_length = 1;
        }
        if (codesNumLength < code_num_length) {
            tileNum = 1;
            tileLength = codesNumLength;
            lastTileLength = codesNumLength;
        } else if (codesNumLength % code_num_length == 0) {
            tileNum = codesNumLength / code_num_length;
            tileLength = code_num_length;
            lastTileLength = code_num_length;
        } else {
            tileNum = codesNumLength / code_num_length + 1;
            tileLength = code_num_length;
            lastTileLength = codesNumLength % code_num_length;
        }
        tiling.set_tileNum(tileNum);
        tiling.set_tileLength(tileLength);
        tiling.set_lastTileLength(lastTileLength);
    }

    static void SetTilingInfo(gert::TilingContext* context, DistanceFlatL2MinsAtFP32TilingData& tiling, int32_t aivNum, int32_t ubSize)
    {
        int32_t queryNumLength = context->GetInputShape(0)->GetStorageShape().GetDim(0);
        int32_t codesNumLength = context->GetInputShape(1)->GetStorageShape().GetDim(0);
        int32_t dimLength = context->GetInputShape(0)->GetStorageShape().GetDim(1);

        int32_t formerCoreNum = 0;
        int32_t formerCoreLength = 0;
        int32_t tailCoreNum = 0;
        int32_t tailCoreLength = 0;
        if (queryNumLength < aivNum) {
            formerCoreNum = queryNumLength;
            formerCoreLength = 1;
            tailCoreNum = aivNum - formerCoreNum;
            tailCoreLength = 0;
        } else if (queryNumLength % aivNum == 0) {
            formerCoreNum = aivNum;
            formerCoreLength = queryNumLength / aivNum;
            tailCoreNum = 0;
            tailCoreLength = 0;
        } else {
            formerCoreNum = queryNumLength % aivNum;
            formerCoreLength = queryNumLength / aivNum + 1;
            tailCoreNum = aivNum - formerCoreNum;
            tailCoreLength = queryNumLength / aivNum;
        }

        tiling.set_formerCoreNum(formerCoreNum);
        tiling.set_formerCoreLength(formerCoreLength);
        tiling.set_tailCoreNum(tailCoreNum);
        tiling.set_tailCoreLength(tailCoreLength);

        tiling.set_queryNumLength(queryNumLength);
        tiling.set_codesNumLength(codesNumLength);
        tiling.set_dimLength(dimLength);

        context->SetBlockDim(formerCoreNum + tailCoreNum);

        SetKernelLoopInfo(tiling, ubSize, codesNumLength, dimLength);
    }

    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        DistanceFlatL2MinsAtFP32TilingData tiling;

        if (context == nullptr || context->GetRawTilingData() == nullptr) {
            return ge::GRAPH_FAILED;
        }

        const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        auto aivNum = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        if (aivNum == 0 || ubSize == 0 || context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr) {
            return ge::GRAPH_FAILED;
        }

        SetTilingInfo(context, tiling, aivNum, ubSize);
        
        uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (currentWorkspace == nullptr) {
            return ge::GRAPH_FAILED;
        }
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
        const gert::Shape *query_shape = context->GetInputShape(0);
        const gert::Shape *codes_shape = context->GetInputShape(1);

        gert::Shape *dist_result_shape = context->GetOutputShape(0);
        gert::Shape *min_result_shape = context->GetOutputShape(1);
        gert::Shape *flag_shape_shape = context->GetOutputShape(2);

        if (query_shape == nullptr || codes_shape == nullptr || dist_result_shape == nullptr || min_result_shape == nullptr || flag_shape_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }

        dist_result_shape->SetDimNum(2);
        dist_result_shape->SetDim(0, query_shape->GetDim(0));
        dist_result_shape->SetDim(1, codes_shape->GetDim(0));

        min_result_shape->SetDimNum(2);
        min_result_shape->SetDim(0, query_shape->GetDim(0));
        min_result_shape->SetDim(1, codes_shape->GetDim(0) / 64 * 2);

        flag_shape_shape->SetDimNum(2);
        flag_shape_shape->SetDim(0, 40);
        flag_shape_shape->SetDim(1, 16);
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

namespace ops {
class DistanceFlatL2MinsAtFP32 : public OpDef {
public:
    explicit DistanceFlatL2MinsAtFP32(const char* name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("codes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("dist_result")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("min_result")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("flag_shape")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b").AddConfig("ascend910_93");
    }
};

OP_ADD(DistanceFlatL2MinsAtFP32);
}
