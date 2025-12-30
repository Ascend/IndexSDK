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
#include <cstdint>
#include "vstar_compute_l2_tiling.h"
#include "register/op_def_registry.h"

namespace {
    constexpr uint32_t AI_CORE_USED = 8;
    constexpr uint32_t BUF_NUM = 1;
    constexpr uint32_t CUBE_ALIGN = 16;
    constexpr uint32_t TILE_UNIT = CUBE_ALIGN * BUF_NUM;
    constexpr uint32_t bufferNumL1B = 1;
    constexpr uint32_t bufferNumL0B = 1;

    constexpr uint32_t KB = 1024;
    constexpr uint32_t CODEBOOKL1BSIZE = 512 * KB / bufferNumL1B;
    constexpr uint32_t CODEBOOKL0BSIZE = 64 * KB / bufferNumL0B; // double buffer
    constexpr uint32_t UB_BYTE_SIZE = 256 * KB;
    constexpr uint32_t HALF_SIZE = 2;
    constexpr uint32_t ALIGN_NUM = 16;

    uint32_t DIV_UP(uint32_t x, uint32_t y)
    {
        return (x + y - 1) / y;
    }

    uint32_t ROUND_ALIGN(uint32_t x)
    {
        return (x + (ALIGN_NUM - 1)) / ALIGN_NUM * ALIGN_NUM;
    }
}

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        if (context == nullptr) {
            return ge::GRAPH_FAILED;
        }
        VstarComputeL2TilingData tiling;

        const gert::RuntimeAttrs* attrs = context->GetAttrs();
        if (attrs == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const int subDim1 = *(attrs->GetAttrPointer<int>(0));
        const int subDim2 = *(attrs->GetAttrPointer<int>(1));
        if (subDim1 == 0 || subDim2 == 0) {
            return ge::GRAPH_FAILED;
        }

        const gert::StorageShape* x1_shape = context->GetInputShape(0);
        if (x1_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        int n = x1_shape->GetStorageShape().GetDim(0);
        int nlist1 = x1_shape->GetStorageShape().GetDim(1) / subDim1;

        const gert::StorageShape* x2_shape = context->GetInputShape(1);
        if (x2_shape == nullptr || nlist1 == 0) {
            return ge::GRAPH_FAILED;
        }
        int nlist2 = x2_shape->GetStorageShape().GetDim(0) * 16 / nlist1 / subDim2 ;

        const gert::StorageShape* x3_shape = context->GetInputShape(2);
        if (x3_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        int nprobe1 = x3_shape->GetStorageShape().GetDim(1);

        tiling.set_subDim1(subDim1);
        tiling.set_subDim2(subDim2);
        tiling.set_nlist1(nlist1);
        tiling.set_nlist2(nlist2);
        tiling.set_nprobe1(nprobe1);
        tiling.set_n(n);

        tiling.set_sizeCodeBookL1BBuffer(CODEBOOKL1BSIZE);
        tiling.set_sizeCodeBookL0BBuffer(CODEBOOKL0BSIZE);
        tiling.set_cubeAlign(CUBE_ALIGN);

        if (AI_CORE_USED == 0 || CODEBOOKL1BSIZE == 0 || CODEBOOKL0BSIZE == 0) {
            return ge::GRAPH_FAILED;
        }
        // 将单query的所有nprobe1尽量平均分给所有core
        // nprobe1 = 18 理想情况：前2个core处理6个probe，后6个core处理2个probe。
        // 即formerNum = 2，probePerBlockFormer = 3，probePerBlockLatter = 2。
        uint32_t probePerBlockFormer = DIV_UP(nprobe1, AI_CORE_USED);    // 每个core计算单个query的probePerBlock个nprobe1
        uint32_t formerNum = static_cast<uint32_t>(nprobe1) % AI_CORE_USED;
        // 对于nprobe1=16，probePerBlockFormer和probePerBlockLatter都是2
        uint32_t probePerBlockLatter = (static_cast<uint32_t>(nprobe1) / AI_CORE_USED);

        tiling.set_formerNum(formerNum);
        tiling.set_probePerBlockFormer(probePerBlockFormer);
        tiling.set_probePerBlockLatter(probePerBlockLatter);  // 后(AI_CORE_USED - formerNum)个core处理多少个probe

        // 计算query搬运策略 GM->L1
        // 两层循环，第一层对queryCodeAlignSize循环，第二层对probePerBlockFormer循环，每次搬(1 * subDim1 * HALF_SIZE)
        uint32_t queryCodeAlignSize = ROUND_ALIGN(n);  // 16对齐后query的数量
        // 128 * 64 * 2 / 1024 = 16K
        uint32_t queryCodeUsefulByteSizePerProbe = queryCodeAlignSize * subDim1 * HALF_SIZE;
        // query 一次搬入
        tiling.set_queryCodeAlignSize(queryCodeAlignSize);
        tiling.set_queryCodeUsefulByteSizePerProbe(queryCodeUsefulByteSizePerProbe);
        tiling.set_queryCodeFormerUsefulByteSize(queryCodeUsefulByteSizePerProbe * probePerBlockFormer);
        tiling.set_queryCodeLatterUsefulByteSize(queryCodeUsefulByteSizePerProbe * probePerBlockLatter);

        // codebook搬运策略 GM->L1，目标512K，计算单probe大小，每个probe单独move一次，保证subDim1能16对齐，共搬运probePerBlockFormer个。
        uint32_t codeBookUsefulByteSizePerProbe = static_cast<uint32_t>(subDim1) *
            static_cast<uint32_t>(nlist2) *
            static_cast<uint32_t>(subDim2) *
            HALF_SIZE;  // 64 * 64 * 16 * 2 / 1024 = 128K

        uint32_t moveTimesL1PerProbe = DIV_UP(codeBookUsefulByteSizePerProbe, CODEBOOKL1BSIZE);
        uint32_t tailSizeL1PerProbe = (codeBookUsefulByteSizePerProbe % CODEBOOKL1BSIZE) ?
            (codeBookUsefulByteSizePerProbe % CODEBOOKL1BSIZE) : CODEBOOKL1BSIZE;

        tiling.set_moveTimesL1PerProbe(moveTimesL1PerProbe);
        tiling.set_tailSizeL1PerProbe(tailSizeL1PerProbe);

        // 计算query搬运策略 L1->L0A，左矩阵，限制64K，每次（1 * subDim1），可以累加probe到(1 * subDim1 * probePerBlockFormer)
        // 以probePerBlockFormer为循环全部搬入

        // 计算codebook搬运策略 L1->L0B，右矩阵，限制64K
        // 每个query每个core一次处理的数量(subDim1 * nlist2 * subDim2 * probePerBlock)
        uint32_t codeBookSizeL0BPerBlock = static_cast<uint32_t>(subDim1) *
            static_cast<uint32_t>(nlist2) *
            static_cast<uint32_t>(subDim2) *
            HALF_SIZE; // 64 * 64 * 16 * 2 = 128K
        uint32_t moveTimesL0BPerBlockTail = DIV_UP(codeBookSizeL0BPerBlock, CODEBOOKL0BSIZE);
        uint32_t remainSizeL0BPerBlock = (codeBookSizeL0BPerBlock % CODEBOOKL0BSIZE) ?
            (codeBookSizeL0BPerBlock % CODEBOOKL0BSIZE) : CODEBOOKL0BSIZE;
        tiling.set_moveTimesL0BPerBlockTail(moveTimesL0BPerBlockTail);
        tiling.set_remainSizeL0BPerBlock(remainSizeL0BPerBlock);

        // 计算矩阵乘结果L0C大小，即CO1：(1 * subDim1) * (subDim1 * nlist2 * subDim2) = (1 * nlist2 * subDim2)
        // 在CO2上累积：(1 * nlist2 * subDim2) -> (1 * nlist2 * subDim2)
        // (16 * subDim1) * (subDim1 * nlist2 * subDim2 * probePerBlock) = (1 * nlist2 * subDim2)
        context->SetBlockDim(AI_CORE_USED);
        if (context->GetRawTilingData() == nullptr) {
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
        return GRAPH_SUCCESS;
    }

    static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
    {
        return GRAPH_SUCCESS;
    }
}


namespace ops {
    class VstarComputeL2 : public OpDef {
    public:
        explicit VstarComputeL2(const char* name) : OpDef(name)
        {
            this->Input("queryCode")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT16})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("codebookL2")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT16})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("l1KIndices")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_UINT16})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("outDists")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT16})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("opFlag")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_UINT16})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Attr("subSpaceDim1").Int();
            this->Attr("subSpaceDim2").Int();

            this->SetInferShape(ge::InferShape);
            this->SetInferDataType(ge::InferDataType);

            this->AICore()
                    .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend310p");
        }
    };

    OP_ADD(VstarComputeL2);
}
