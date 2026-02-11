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

#include "index_code_and_precompute_tiling.h"

#include <iostream>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace {
    static const int64_t FLOAT32_BYTES = 4;
    static const int64_t UINT8_BYTES = 1;
    static const int32_t BASE_N = 128;   // 分块的基本单位
    static const int32_t GM_ALIGN = 512; // 全局内存对齐大小
    static const int32_t BASE = 1;
    static const int32_t OFFSET = 2;
    static const int32_t MAX_PER_LOOP_PROCESS_LEN = 4096;  // 每个循环处理的最大长度
}

namespace optiling {
    ge::graphStatus DoLibApiTiling(IndexCodeAndPrecomputeTilingData &tiling, uint64_t l1_size, uint64_t l0c_size)
    {
        // 使用MatmulApiTiling，配置gemm的分块策略
        matmul_tiling::MatmulApiTiling gemm_qb_tiling;
        gemm_qb_tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                                matmul_tiling::DataType::DT_FLOAT);
        gemm_qb_tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                                matmul_tiling::DataType::DT_FLOAT, true);
        gemm_qb_tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                                matmul_tiling::DataType::DT_FLOAT);
        gemm_qb_tiling.SetBias(false);
        gemm_qb_tiling.SetOrgShape(tiling.get_vecNumLength(), 1, tiling.get_dimLength());
        gemm_qb_tiling.SetShape(tiling.get_vecNumLength(), 1, tiling.get_dimLength());
        gemm_qb_tiling.SetBufferSpace(l1_size, l0c_size);

        gemm_qb_tiling.SetFixSplit(-1, 1, 128);

        if (gemm_qb_tiling.GetTiling(tiling.gemm_qb_tiling) == -1) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        IndexCodeAndPrecomputeTilingData tiling;
        // 获取平台信息（如核心数、内存大小）
        ASCENDC_RETURN_IF_NOT(context != nullptr, ge::GRAPH_FAILED);
        const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint64_t ub_size = 0;
        uint64_t l1_size = 0;
        uint64_t l0c_size = 0;
        int32_t aicube_num = static_cast<int32_t>(ascendcPlatform.GetCoreNumAic());
        int32_t aivector_num = static_cast<int32_t>(ascendcPlatform.GetCoreNumAiv());
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1_size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c_size);
        // 从context中获取 vec_num、dim_len
        if (!context->GetInputShape(1)) {
            return ge::GRAPH_FAILED;
        }
        int32_t vec_num = context->GetInputShape(1)->GetStorageShape().GetDim(0);
        int32_t dim_len = context->GetInputShape(1)->GetStorageShape().GetDim(1);

        tiling.set_vecNumLength(vec_num);
        tiling.set_dimLength(dim_len);

        context->SetBlockDim(aivector_num);

        // 根据 dim_len 等，确定tileLength：表示每次循环处理的向量数量
        int32_t tileLength = (ub_size - 2048) / (dim_len * FLOAT32_BYTES * 10 * 2 + FLOAT32_BYTES * 2) / 16 * 16;
        int32_t tileLengthStage1 = (ub_size - 2048 - dim_len * FLOAT32_BYTES*2) /
                                   (dim_len * FLOAT32_BYTES*2 + dim_len / 8 * UINT8_BYTES + FLOAT32_BYTES*2) / 16 * 16;
        int32_t tileLengthStage2 = (ub_size - 2048 - FLOAT32_BYTES *2) / (5 * FLOAT32_BYTES*2) / 16 * 16;

        if (tileLengthStage1 > MAX_PER_LOOP_PROCESS_LEN) {
            tileLengthStage1 = MAX_PER_LOOP_PROCESS_LEN;
        }
        if (tileLengthStage2 > MAX_PER_LOOP_PROCESS_LEN) {
            tileLengthStage2 = MAX_PER_LOOP_PROCESS_LEN;
        }

        tiling.set_tileLengthStage1(tileLengthStage1);
        tiling.set_tileLengthStage2(tileLengthStage2);
        tiling.set_tileLength(tileLength);

        // 配置gemm的分块策略
        ASCENDC_RETURN_IF_NOT(ge::GRAPH_SUCCESS == DoLibApiTiling(tiling, l1_size, l0c_size), ge::GRAPH_FAILED);
        
        // 设置工作空间大小和分块数据。
        if (!context->GetWorkspaceSizes(1)) {
            return ge::GRAPH_FAILED;
        }
        uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        size_t usrSize = 0;
        currentWorkspace[0] = usrSize + sysWorkspaceSize;
        if (context == nullptr || context->GetRawTilingData() == nullptr) {
            return ge::GRAPH_FAILED;
        }

        // 添加检查: 写入超出缓冲区，造成缓冲区溢出。
        if (tiling.GetDataSize() > context->GetRawTilingData()->GetCapacity()) {
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
        const gert::Shape* shape = context->GetInputShape(1);  // vec*dim
        uint32_t numCode = shape->GetDim(0);
        uint32_t dim = shape->GetDim(1);
        gert::Shape shape1({numCode, dim / 8});
        gert::Shape shape2({numCode});
        gert::Shape shape3({numCode, dim});

        gert::Shape *out_shape0 = context->GetOutputShape(0);
        gert::Shape *out_shape1 = context->GetOutputShape(1);
        gert::Shape *out_shape2 = context->GetOutputShape(2);
        *out_shape0 = shape1;
        *out_shape1 = shape2;
        *out_shape2 = shape2;

        return GRAPH_SUCCESS;
    }
    static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
        auto data_type = context->GetInputDataType(1);
        context->SetOutputDataType(0, ge::DataType::DT_UINT8);
        context->SetOutputDataType(1, data_type);
        context->SetOutputDataType(2, data_type);

        return GRAPH_SUCCESS;
    }
}

/* input:
 *      indexes: n * d
 *      centroid: 1 * d
 *      centroidl2: 1
 * output:
 *      codes_result: indexes - centroid 取符号 -> n * d/8
 *      l2_result: ||indexes - centroid||^2 = ||indexes||^2 - 2<indexes, centroid> + ||centroid||^2 -> n
 *      l1_result: 2||indexes - centroid||^2 * sqrt(d) / ||indexes - centroid||1 -> n
*/
namespace ops {
class IndexCodeAndPrecompute : public OpDef {
public:
    explicit IndexCodeAndPrecompute(const char* name) : OpDef(name)
    {
        this->Input("vectorNum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("indexes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("indexesl2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("centroid")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("centroidl2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("codes_result")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("l2_result")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("l1_result")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b").AddConfig("ascend950");
    }
};

OP_ADD(IndexCodeAndPrecompute);
}

