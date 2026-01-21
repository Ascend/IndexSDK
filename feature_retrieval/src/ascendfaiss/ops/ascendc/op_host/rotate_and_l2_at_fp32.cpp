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
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "rotate_and_l2_at_fp32_tiling.h"

namespace {
    static const int64_t FLOAT32_BYTES = 4;
    static const int32_t MAX_PER_LOOP_PROCESS_LEN = 4096;
}

namespace optiling {
    ge::graphStatus DoLibApiTiling(RotateAndL2AtFP32TilingData &tiling, uint64_t l1_size, uint64_t l0c_size)
    {
        matmul_tiling::MatmulApiTiling gemm_qb_tiling;
        gemm_qb_tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                                matmul_tiling::DataType::DT_FLOAT);
        gemm_qb_tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                                matmul_tiling::DataType::DT_FLOAT, true);
        gemm_qb_tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                                matmul_tiling::DataType::DT_FLOAT);
        gemm_qb_tiling.SetBias(false);
        gemm_qb_tiling.SetOrgShape(tiling.get_vecNumLength(), tiling.get_dimLength(), tiling.get_dimLength());
        gemm_qb_tiling.SetShape(tiling.get_vecNumLength(), tiling.get_dimLength(), tiling.get_dimLength());
        gemm_qb_tiling.SetBufferSpace(l1_size, l0c_size);
        if (gemm_qb_tiling.GetTiling(tiling.gemm_qb_tiling) == -1) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        RotateAndL2AtFP32TilingData tiling;
        ASCENDC_RETURN_IF_NOT(context != nullptr, ge::GRAPH_FAILED);
        const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint64_t ub_size = 0;
        uint64_t l1_size = 0;
        uint64_t l0c_size = 0;
        int32_t aicube_num = static_cast<int32_t>(ascendcPlatform.GetCoreNumAic());
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1_size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c_size);
        int32_t vec_num = context->GetInputShape(0)->GetStorageShape().GetDim(0);
        int32_t dim_len = context->GetInputShape(0)->GetStorageShape().GetDim(1);

        tiling.set_vecNumLength(vec_num);
        tiling.set_dimLength(dim_len);

        int32_t tileLength = (ub_size - 2048) / (dim_len * FLOAT32_BYTES * 2 + FLOAT32_BYTES * 2) / 16 * 16;

        if (tileLength > MAX_PER_LOOP_PROCESS_LEN) {
            tileLength = MAX_PER_LOOP_PROCESS_LEN;
        }
        tiling.set_tileLength(tileLength);
        context->SetBlockDim(aicube_num);
        
        ASCENDC_RETURN_IF_NOT(ge::GRAPH_SUCCESS == DoLibApiTiling(tiling, l1_size, l0c_size), ge::GRAPH_FAILED);
        
        uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        size_t usrSize = 0;
        currentWorkspace[0] = usrSize + sysWorkspaceSize;
        if (context == nullptr || context->GetRawTilingData() == nullptr) {
            return ge::GRAPH_FAILED;
        }
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }
}

namespace ge {
    static ge::graphStatus InferShape(gert::InferShapeContext* context)
    {
        const gert::Shape* shape = context->GetInputShape(0);  // vec*dim
        uint32_t numVec = shape->GetDim(0);
        gert::Shape shape1({numVec});
        gert::Shape *out_shape0 = context->GetOutputShape(0);
        gert::Shape *out_shape1 = context->GetOutputShape(1);
        *out_shape0 = *shape;
        *out_shape1 = shape1;
        return GRAPH_SUCCESS;
    }
    static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
        auto data_type = context->GetInputDataType(0);
        context->SetOutputDataType(0, data_type);
        context->SetOutputDataType(1, data_type);
        return GRAPH_SUCCESS;
    }
}

/* input:
 *      vectors: n * d
 *      matrix: d * d
 * output:
 *      rotate_result: vectors * matrix -> n * d
 *      l2_result: ||vectors||^2 -> n
*/
namespace ops {
class RotateAndL2AtFP32 : public OpDef {
public:
    explicit RotateAndL2AtFP32(const char* name) : OpDef(name)
    {
        this->Input("vectors")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("vectorSize")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("matrix")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("rotate_result")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("l2_result")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b").AddConfig("ascend910_95");
    }
};

OP_ADD(RotateAndL2AtFP32);
}