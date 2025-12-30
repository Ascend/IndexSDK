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

#include "vstar_base_add_mat_mul_tiling.h"
#include "register/op_def_registry.h"

namespace {
    constexpr uint32_t UB_BLOCK_BYTE_SIZE = 32;
    constexpr uint32_t SEQ_PROC_SLICE = 8u;

}  // namespace

namespace optiling {

    static ge::graphStatus TilingBasic(gert::TilingContext* context, VstarBaseAddMatMulTilingData &tilingData)
    {
        if (context == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const gert::StorageShape* x_shape = context->GetInputShape(0);
        const gert::StorageShape* y_shape = context->GetInputShape(1);
        if ((x_shape == nullptr) || (y_shape == nullptr)) {
            return ge::GRAPH_FAILED;
        }
        auto nb = x_shape->GetStorageShape().GetDim(0);
        auto dim = x_shape->GetStorageShape().GetDim(1);
        auto nList = y_shape->GetStorageShape().GetDim(0);
        auto subDim = y_shape->GetStorageShape().GetDim(1);

        tilingData.set_nb(nb);
        tilingData.set_dim(dim);
        tilingData.set_nList(nList);
        tilingData.set_subDim(subDim);
        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus TilingCore(gert::TilingContext* context, VstarBaseAddMatMulTilingData &tilingData)
    {
        if (context == nullptr) {
            return ge::GRAPH_FAILED;
        }
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        auto aicNum = ascendcPlatform.GetCoreNumAic();
        auto aivNum = ascendcPlatform.GetCoreNumAiv();
        context->SetBlockDim(aicNum);

        auto nb = tilingData.get_nb();
        auto dim = tilingData.get_dim();
        auto nList = tilingData.get_nList();
        auto subDim = tilingData.get_subDim();
        if (aivNum == 0) {
            return ge::GRAPH_FAILED;
        }
        // 均分计算任务到每个核上
        uint32_t nbRegularTaskCore = nb / aivNum;
        uint32_t nbExtraTaskCore = nb / aivNum + 1;
        uint32_t subSpaceRegularTaskCore = nList * subDim / aivNum;
        uint32_t subSpaceExtraTaskCore = nList * subDim / aivNum + 1;

        // 通过计算 UB 大小上限动态计算当前处理的最优左右矩阵tiling大小，其中dim维度不切分为切分的原子大小。
        uint32_t typeSize = sizeof(uint16_t); // 310P 的matmul(MMad与LoadData实际上)只能够处理half类型数据

        uint64_t ub_bw;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_bw);

        // 由于UB需要进行FP32- FP16的转换所以只能按照峰值UB需求来进行最大申请
        if (dim == 0 || typeSize == 0) {
            return ge::GRAPH_FAILED;
        }
        // UB空间用满了会报error，限制使用到3/4
        uint32_t maxVecNumAtUB = (ub_bw * 3 / 4) / (dim * typeSize + dim * sizeof(float));

        uint32_t each_M_regular = std::min(maxVecNumAtUB, nbRegularTaskCore);
        uint32_t each_M_extra = std::min(maxVecNumAtUB, nbExtraTaskCore);
        if (each_M_regular == 0 || each_M_extra == 0) {
            return ge::GRAPH_FAILED;
        }
        uint32_t loop_M_regular = (nbRegularTaskCore + each_M_regular - 1) / each_M_regular;
        uint32_t loop_M_extra = (nbExtraTaskCore + each_M_extra - 1) / each_M_extra;
        uint32_t last_M_regular = nbRegularTaskCore % each_M_regular;
        uint32_t last_M_extra = nbExtraTaskCore % each_M_extra;
        uint32_t each_N_regular = std::min(maxVecNumAtUB, subSpaceRegularTaskCore);
        uint32_t each_N_extra = std::min(maxVecNumAtUB, subSpaceExtraTaskCore);
        if (each_N_regular == 0 || each_N_extra == 0) {
            return ge::GRAPH_FAILED;
        }
        uint32_t loop_N_regular = (subSpaceRegularTaskCore + each_N_regular -1) / each_N_regular;
        uint32_t loop_N_extra = (subSpaceExtraTaskCore + each_N_extra -1) / each_N_extra;
        uint32_t last_N_regular = subSpaceRegularTaskCore % each_N_regular;
        uint32_t last_N_extra = subSpaceExtraTaskCore % each_N_extra;

        tilingData.set_aicNum(aicNum);
        tilingData.set_aivNum(aivNum);
        tilingData.set_each_M_regular(each_M_regular);
        tilingData.set_each_M_extra(each_M_extra);
        tilingData.set_loop_M_regular(loop_M_regular);
        tilingData.set_loop_M_extra(loop_M_extra);
        tilingData.set_last_M_regular(last_M_regular);
        tilingData.set_last_M_extra(last_M_extra);
        tilingData.set_each_N_regular(each_N_regular);
        tilingData.set_each_N_extra(each_N_extra);
        tilingData.set_loop_N_regular(loop_N_regular);
        tilingData.set_loop_N_extra(loop_N_extra);
        tilingData.set_last_N_regular(last_N_regular);
        tilingData.set_last_N_extra(last_N_extra);
        tilingData.set_MTaskCore_regular(nbRegularTaskCore);
        tilingData.set_MTaskCore_extra(nbExtraTaskCore);
        tilingData.set_NTaskCore_regular(subSpaceRegularTaskCore);
        tilingData.set_NTaskCore_extra(subSpaceExtraTaskCore);

        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus TilingCube(gert::TilingContext* context, VstarBaseAddMatMulTilingData &tilingData)
    {
        using namespace matmul_tiling;
        if (context == nullptr) {
            return ge::GRAPH_FAILED;
        }
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        auto aivNum = ascendcPlatform.GetCoreNumAiv();

        MultiCoreMatmulTiling cubeTilingCore(ascendcPlatform);
        auto M = tilingData.get_nb();
        auto N = tilingData.get_nList() * tilingData.get_subDim();
        auto K = tilingData.get_dim();
        cubeTilingCore.SetDim(aivNum);
        cubeTilingCore.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
        cubeTilingCore.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, true);
        cubeTilingCore.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
        cubeTilingCore.SetShape(M, N, K);
        cubeTilingCore.SetOrgShape(M, N, K);
        cubeTilingCore.SetBufferSpace(-1, -1, -1);
        cubeTilingCore.SetBias(false);
        int ret = cubeTilingCore.GetTiling(tilingData.cube_tiling);
        if (ret == -1) {
            printf("cube tiling each core error\n");
            return ge::GRAPH_FAILED;
        }
        matmul_tiling::SysTilingTempBufSize MMFormatUb1;
        MultiCoreMatmulGetTmpBufSize(tilingData.cube_tiling, MMFormatUb1);
        tilingData.set_MMFormatUb1(MMFormatUb1.ubSize + MMFormatUb1.l0cSize);

        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        if (context == nullptr) {
            return ge::GRAPH_FAILED;
        }
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        VstarBaseAddMatMulTilingData tiling;
        auto ret = TilingBasic(context, tiling);
        if (ret != ge::GRAPH_SUCCESS) {
            printf("TilingBasic error\n");
            return ret;
        }

        ret = TilingCube(context, tiling);
        if (ret != ge::GRAPH_SUCCESS) {
            printf("TilingCube error\n");
            return ret;
        }

        ret = TilingCore(context, tiling);
        if (ret != ge::GRAPH_SUCCESS) {
            printf("TilingCore error\n");
            return ret;
        }

        if (context->GetRawTilingData() == nullptr) {
            return ge::GRAPH_FAILED;
        }
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        auto nb = tiling.get_nb();
        auto dim = tiling.get_dim();
        auto nList = tiling.get_nList();
        auto subDim = tiling.get_subDim();
        size_t userWorkspaceSize = (nb * dim + nList * subDim * dim) * sizeof(uint16_t);
        uint32_t systemWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (currentWorkspace == nullptr) {
            printf("work space null\n");
            return ge::GRAPH_FAILED;
        }
        currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;

        return ge::GRAPH_SUCCESS;
    }
}


namespace ge {
    static ge::graphStatus InferShape(gert::InferShapeContext* context)
    {
        if (context == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const gert::Shape* x1_shape = context->GetInputShape(0);
        const gert::Shape* x2_shape = context->GetInputShape(1);
        gert::Shape* y_shape = context->GetOutputShape(0);
        if (x1_shape == nullptr || x2_shape == nullptr || y_shape == nullptr) {
            printf("infershape null ptr check failed\n");
            return GRAPH_FAILED;
        }

        auto nb = x1_shape->GetDim(0);
        auto nList = x2_shape->GetDim(0);
        auto subDim = x2_shape->GetDim(1);
        y_shape->SetDim(0, nb);
        y_shape->SetDim(1, nList);
        y_shape->SetDim(2, subDim);

        return GRAPH_SUCCESS;
    }

    static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
    {
        return GRAPH_SUCCESS;
    }
}


namespace ops {
class VstarBaseAddMatMul : public OpDef {
public:
    explicit VstarBaseAddMatMul(const char* name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310p");
    }
};

OP_ADD(VstarBaseAddMatMul);
}
