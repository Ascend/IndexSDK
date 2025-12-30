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

#include "distance_ivf_flat_ip_fp32_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace {
    static const int64_t FLOAT32_BYTES = 4;
    static const int32_t BASE_N = 128;
    static const int32_t GM_ALIGN = 512;
    static const int32_t BASE = 1;
    static const int32_t OFFSET = 2;
    static const int32_t MAX_PER_LOOP_PROCESS_LEN = 4096;
}

namespace optiling {
    ge::graphStatus DoLibApiTiling(DistanceIVFFlatIpFP32TilingData &tiling, uint64_t l1_size, uint64_t l0c_size)
    {
        matmul_tiling::MatmulApiTiling gemm_qb_tiling;
        gemm_qb_tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                                matmul_tiling::DataType::DT_FLOAT);
        gemm_qb_tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                                matmul_tiling::DataType::DT_FLOAT, true);
        gemm_qb_tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                                matmul_tiling::DataType::DT_FLOAT);
        gemm_qb_tiling.SetBias(false);
        gemm_qb_tiling.SetOrgShape(1, tiling.get_per_loop_process_len(), tiling.get_dim_len());
        gemm_qb_tiling.SetShape(1, tiling.get_per_loop_process_len(), tiling.get_dim_len());
        gemm_qb_tiling.SetFixSplit(1, std::min(BASE_N, tiling.get_per_loop_process_len()), -1);
        gemm_qb_tiling.SetBufferSpace(l1_size, l0c_size);
        if (gemm_qb_tiling.GetTiling(tiling.gemm_qb_tiling) == -1) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SetTilingInfo(gert::TilingContext* context, DistanceIVFFlatIpFP32TilingData& tiling, uint64_t ub_size)
    {
        const gert::StorageShape* base_shape = context->GetInputShape(BASE);
        ASCENDC_RETURN_IF_NOT(base_shape != nullptr, ge::GRAPH_FAILED);
        int32_t code_num = base_shape->GetStorageShape().GetDim(0);
        int32_t dim_len = base_shape->GetStorageShape().GetDim(1);
        const gert::StorageShape* offset_shape = context->GetInputShape(OFFSET);
        ASCENDC_RETURN_IF_NOT(offset_shape != nullptr, ge::GRAPH_FAILED);
        int32_t code_list_num = offset_shape->GetStorageShape().GetDim(0);

        int32_t burst_len = 64;
        if (dim_len > 256) {
            burst_len = 32;
        }
        int32_t per_loop_process_len =
            static_cast<int32_t>(ub_size * 9 / 10) / static_cast<int32_t>(FLOAT32_BYTES * 2) / burst_len * burst_len;
        if (code_num < per_loop_process_len) {
            per_loop_process_len = code_num;
        }
        if (per_loop_process_len > MAX_PER_LOOP_PROCESS_LEN) {
            per_loop_process_len = MAX_PER_LOOP_PROCESS_LEN;
        }
        int32_t max_res_stride = code_num / burst_len * 2;
        int32_t max_res_inner_stride = per_loop_process_len / burst_len * 2;
        tiling.set_dim_len(dim_len);
        tiling.set_burst_len(burst_len);
        tiling.set_per_loop_process_len(per_loop_process_len);
        tiling.set_code_list_num(code_list_num);
        tiling.set_code_num(code_num);
        tiling.set_max_res_stride(max_res_stride);
        tiling.set_max_res_inner_stride(max_res_inner_stride);

        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        DistanceIVFFlatIpFP32TilingData tiling;
        ASCENDC_RETURN_IF_NOT(context != nullptr, ge::GRAPH_FAILED);
        const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint64_t ub_size = 0;
        uint64_t l1_size = 0;
        uint64_t l0c_size = 0;
        int32_t aicube_num = static_cast<int32_t>(ascendcPlatform.GetCoreNumAic());
        auto block_dim = aicube_num;
        int32_t use_aiv_num = block_dim * 2;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1_size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c_size);
        ASCENDC_RETURN_IF_NOT(ge::GRAPH_SUCCESS == SetTilingInfo(context, tiling, ub_size), ge::GRAPH_FAILED);
        ASCENDC_RETURN_IF_NOT(ge::GRAPH_SUCCESS == DoLibApiTiling(tiling, l1_size, l0c_size), ge::GRAPH_FAILED);
        context->SetBlockDim(block_dim);
        uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        ASCENDC_RETURN_IF_NOT(currentWorkspace != nullptr, ge::GRAPH_FAILED);
        int32_t buffer_size = (tiling.get_per_loop_process_len() * FLOAT32_BYTES + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN;
        tiling.set_buffer_size(buffer_size);
        size_t usrSize = static_cast<size_t>(use_aiv_num * buffer_size * 2);
        currentWorkspace[0] = usrSize + sysWorkspaceSize;
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }
}


namespace ge {
    static ge::graphStatus InferShape(gert::InferShapeContext* context)
    {
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
class DistanceIVFFlatIpFP32 : public OpDef {
public:
    explicit DistanceIVFFlatIpFP32(const char* name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("base")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("actual_size")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("result")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("max_result")
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

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b").AddConfig("ascend910_93");
    }
};

OP_ADD(DistanceIVFFlatIpFP32);
}
