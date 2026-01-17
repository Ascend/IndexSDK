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


#include "ascendc_l2_norm_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host_common.h"

using namespace matmul_tiling;
using namespace Utils;

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    context->SetBlockDim(ascendcPlatform.GetCoreNumAic());

    if (context->GetInputTensor(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &queryShape = context->GetInputTensor(0)->GetStorageShape(); // 算子固定第0个tensor为查询向量
    uint32_t dim = static_cast<uint32_t>(queryShape[1]); // 查询向量为[queryNum, dim]，dim取第1维
    uint32_t vecCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (vecCoreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    AscendcL2NormTilingData tiling;
    tiling.set_dim(dim);
    tiling.set_vecCoreNum(vecCoreNum);
    tiling.cubeTilingData.set_usedCoreNum(1); // 算子内部已经按核tiling，因此这里只需1个核
    const uint32_t cubeOnceNum = 128; // 128是为了和TIK算子保持输入一致：transfer矩阵为128维
    MatmulApiTiling cubeTiling(ascendcPlatform);
    // 算子内部使用的是TSCM，但是这里GM换成TSCM时CANN有bug，因此使用GM
    cubeTiling.SetAType(TPosition::GM, CubeFormat::NZ, matmul_tiling::DataType::DT_INT8);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::NZ, matmul_tiling::DataType::DT_INT8, true);
    cubeTiling.SetCType(TPosition::VECOUT, CubeFormat::NZ, matmul_tiling::DataType::DT_FLOAT16);
    cubeTiling.SetShape(cubeOnceNum, cubeOnceNum, dim);
    cubeTiling.SetOrgShape(cubeOnceNum, cubeOnceNum, dim);
    cubeTiling.SetBufferSpace(-1, -1, -1);
    cubeTiling.SetDequantType(DequantType::SCALAR);
    int ret = cubeTiling.GetTiling(tiling.cubeTilingData);
    if (ret == -1) {
        return ge::GRAPH_FAILED;
    }

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    const size_t userWorkspaceSize = 0;
    const uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1); // 按照算子样例代码设置为1
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = userWorkspaceSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *)
{
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *)
{
    return GRAPH_SUCCESS;
}
}

namespace ops {
class AscendcL2Norm : public OpDef {
public:
    explicit AscendcL2Norm(const char* name) : OpDef(name)
    {
        this->Input("feature")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("transfer")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("actualNum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("normResult")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_95");
    }
};

OP_ADD(AscendcL2Norm);
}
