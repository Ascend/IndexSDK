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
#include "tiling/tiling_api.h"

namespace {
    static const int64_t BLOCK_SIZE = 32;
    static const int64_t FLOAT32_BYTES = 4;
    static const int32_t BASE_N = 128;
    static const int32_t BASE_CODE = 8192;
    static const int32_t REDUCE_SIZE = 64;
}

namespace optiling {
    ge::graphStatus DoLibApiTiling(DistanceFlatL2MinsAtFP32TilingData &tiling, uint64_t l1_size, uint64_t l0c_size)
    {
        matmul_tiling::MatmulApiTiling gemm_qb_tiling;
        gemm_qb_tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                                matmul_tiling::DataType::DT_FLOAT);
        gemm_qb_tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                                matmul_tiling::DataType::DT_FLOAT, true);
        gemm_qb_tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                                matmul_tiling::DataType::DT_FLOAT);
        gemm_qb_tiling.SetBias(false);
        gemm_qb_tiling.SetOrgShape(tiling.get_formerTileLength(),
                                   tiling.get_baseCodesNumLength(), tiling.get_dimLength());
        gemm_qb_tiling.SetShape(tiling.get_formerTileLength(), tiling.get_baseCodesNumLength(), tiling.get_dimLength());
        gemm_qb_tiling.SetFixSplit(1, std::min(BASE_N, tiling.get_baseCodesNumLength()), -1);
        gemm_qb_tiling.SetBufferSpace(l1_size, l0c_size);
        if (gemm_qb_tiling.GetTiling(tiling.gemm_qb_tiling) == -1) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    static void SetKernelInfo(DistanceFlatL2MinsAtFP32TilingData& tiling, int32_t codesNumLength,
                              int32_t baseCodesNumLength)
    {
        int32_t codesTileNum = 0;
        int32_t codesTileLength = 0;
        int32_t codesLastTileLength = 0;
        if (codesNumLength % baseCodesNumLength == 0) {
            codesTileNum = codesNumLength / baseCodesNumLength;
            codesTileLength = baseCodesNumLength;
            codesLastTileLength = baseCodesNumLength;
        } else {
            codesTileNum = codesNumLength / baseCodesNumLength + 1;
            codesTileLength = baseCodesNumLength;
            codesLastTileLength = codesNumLength % baseCodesNumLength;
        }
        tiling.set_codesTileNum(codesTileNum);
        tiling.set_codesTileLength(codesTileLength);
        tiling.set_codesLastTileLength(codesLastTileLength);
    }

    static int32_t CalculateTileLength(int32_t ub_size, int32_t baseCodesNumLength, int32_t dimLength)
    {
        // 计算每个tile的长度
        return (ub_size - 2048 - 8192 - baseCodesNumLength * FLOAT32_BYTES) / (dimLength * \
                FLOAT32_BYTES + baseCodesNumLength * FLOAT32_BYTES * 2 + REDUCE_SIZE * \
                FLOAT32_BYTES + baseCodesNumLength * FLOAT32_BYTES / 64 * 2);
    }

    static void SetTilingInfo(DistanceFlatL2MinsAtFP32TilingData& tiling, int32_t aivec_num, int32_t ub_size,
                              int32_t queryNumLength, int32_t baseCodesNumLength)
    {
        int32_t formerCoreNum = 0;
        int32_t formerCoreLength = 0;
        int32_t formerTileNum = 0;
        int32_t formerTileLength = 0;
        int32_t formerLastTileLength = 0;
        int32_t tailCoreNum = 0;
        int32_t tailCoreLength = 0;
        int32_t tailTileNum = 0;
        int32_t tailTileLength = 0;
        int32_t tailLastTileLength = 0;
        int32_t tileLength = CalculateTileLength(ub_size, baseCodesNumLength, tiling.get_dimLength());
        if (queryNumLength <= aivec_num) {
            formerCoreNum = queryNumLength;
            formerCoreLength = 1;
            formerTileNum = 1;
            formerTileLength = 1;
            formerLastTileLength = 1;
            tailCoreNum = aivec_num - formerCoreNum;
        } else {
            formerCoreNum = queryNumLength % aivec_num;
            if (formerCoreNum == 0) {
                formerCoreNum = aivec_num;
                formerCoreLength = (queryNumLength + aivec_num - 1) / aivec_num;
                formerTileNum = (formerCoreLength + tileLength - 1) / tileLength;
                formerTileLength = tileLength;
                formerLastTileLength = formerCoreLength - (formerTileNum - 1) * tileLength;
            } else {
                formerCoreLength = (queryNumLength + aivec_num - 1) / aivec_num;
                formerTileNum = (formerCoreLength + tileLength - 1) / tileLength;
                formerTileLength = tileLength;
                formerLastTileLength = formerCoreLength - (formerTileNum - 1) * tileLength;
                tailCoreNum = aivec_num - formerCoreNum;
                tailCoreLength = queryNumLength / aivec_num;
                tailTileNum = (tailCoreLength + tileLength - 1) / tileLength;
                tailTileLength = tileLength;
                tailLastTileLength = tailCoreLength - (tailTileNum - 1) * tileLength;
            }
        }
        tiling.set_formerCoreNum(formerCoreNum);
        tiling.set_formerCoreLength(formerCoreLength);
        tiling.set_formerTileNum(formerTileNum);
        tiling.set_formerTileLength(formerTileLength);
        tiling.set_formerLastTileLength(formerLastTileLength);
        tiling.set_tailCoreNum(tailCoreNum);
        tiling.set_tailCoreLength(tailCoreLength);
        tiling.set_tailTileNum(tailTileNum);
        tiling.set_tailTileLength(tailTileLength);
        tiling.set_tailLastTileLength(tailLastTileLength);
    }

    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        if (context == nullptr|| context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr) {
            return ge::GRAPH_FAILED;
        }
        DistanceFlatL2MinsAtFP32TilingData tiling;
        const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint64_t ub_size = 0;
        uint64_t l1_size = 0;
        uint64_t l0c_size = 0;
        uint32_t aivec_num = ascendcPlatform.GetCoreNumAiv();
        uint32_t aicube_num = ascendcPlatform.GetCoreNumAic();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1_size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c_size);
        int32_t queryNumLength = context->GetInputShape(0)->GetStorageShape().GetDim(0);
        int32_t codesNumLength = context->GetInputShape(1)->GetStorageShape().GetDim(0);
        int32_t dimLength = context->GetInputShape(0)->GetStorageShape().GetDim(1);
        int32_t baseCodesNumLength = std::min(BASE_CODE, codesNumLength);

        tiling.set_queryNumLength(queryNumLength);
        tiling.set_codesNumLength(codesNumLength);
        tiling.set_dimLength(dimLength);
        tiling.set_baseCodesNumLength(baseCodesNumLength);

        SetTilingInfo(tiling, aivec_num, ub_size, queryNumLength, baseCodesNumLength);
        SetKernelInfo(tiling, codesNumLength, baseCodesNumLength);
        auto ret = DoLibApiTiling(tiling, l1_size, l0c_size);
        if (ret != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        context->SetBlockDim(aicube_num);
        uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        
        int32_t bufferSize = tiling.get_formerTileLength() * baseCodesNumLength;
        tiling.set_bufferSize(bufferSize);
        size_t usrSize = static_cast<size_t>(aivec_num) * static_cast<size_t>(bufferSize) * 2 * \
                         static_cast<size_t>(FLOAT32_BYTES);
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

        dist_result_shape->SetDimNum(2);
        dist_result_shape->SetDim(0, query_shape->GetDim(0));
        dist_result_shape->SetDim(1, codes_shape->GetDim(0));

        min_result_shape->SetDimNum(2);
        min_result_shape->SetDim(0, query_shape->GetDim(0));
        min_result_shape->SetDim(1, codes_shape->GetDim(0) / 64);

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
        this->Input("codes_double")
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
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend910_95");
    }
};

OP_ADD(DistanceFlatL2MinsAtFP32);
}
