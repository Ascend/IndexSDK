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
#include "vsm3_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/select/selectwithbytesmask_tiling.h"


namespace {
    constexpr uint32_t AI_CORE_USED = 8;
    constexpr uint32_t CUBE_ALIGN = 16;

    constexpr uint32_t KB = 1024;
    constexpr uint32_t CODEWORDUBSIZE = 32 * KB;
    constexpr uint32_t CODEWORDL1BSIZE = 64 * KB;
    constexpr uint32_t CODEWORDL0BSIZE = 64 * KB;
    constexpr uint32_t HALF_SIZE = 2;

    constexpr uint32_t CTYPE = 2;
    constexpr uint32_t MASKTYPE = 1;

    constexpr int VCMINDIV = 2;

    uint32_t DIV_UP(uint32_t x, uint32_t y)
    {
        return (x + y - 1) / y;
    }
}

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        if (context == nullptr) {
            return ge::GRAPH_FAILED;
        }
        VSM3TilingData tiling;

        const gert::RuntimeAttrs* attrs = context->GetAttrs();
        if (attrs == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const int nlist1 = *(attrs->GetAttrPointer<int>(0));
        const int nlist2 = *(attrs->GetAttrPointer<int>(1));
        const int segmentNum = *(attrs->GetAttrPointer<int>(2));
        if (nlist1 == 0 || nlist2 == 0 || segmentNum == 0) {
            return ge::GRAPH_FAILED;
        }

        const gert::StorageShape* shapeQueryCode = context->GetInputShape(0);
        if (shapeQueryCode == nullptr) {
            return ge::GRAPH_FAILED;
        }
        int n = shapeQueryCode->GetStorageShape().GetDim(0);
        int subDim1 = shapeQueryCode->GetStorageShape().GetDim(1) / nlist1;

        const gert::StorageShape* shapeCodeWord = context->GetInputShape(1);
        if (shapeCodeWord == nullptr) {
            return ge::GRAPH_FAILED;
        }
        int baseNum = shapeCodeWord->GetStorageShape().GetDim(0) * 16 ;
        int subDim2 = shapeCodeWord->GetStorageShape().GetDim(1) * 16 ;
        if (subDim2 == 0) {
            return ge::GRAPH_FAILED;
        }

        const gert::StorageShape* shapeBucketId = context->GetInputShape(2);
        if (shapeBucketId == nullptr) {
            return ge::GRAPH_FAILED;
        }
        int nprobe2 = shapeBucketId->GetStorageShape().GetDim(1) / 6 ;
        const gert::StorageShape* preComputeShape = context->GetInputShape(5); // preCompute为第5个input(从0开始)
        if (preComputeShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        int segmentSize = preComputeShape->GetStorageShape().GetDim(1);
        int segSizeVcMin = segmentSize;

        std::vector<int64_t> shape0Vec = {1, CODEWORDUBSIZE / subDim2};
        std::vector<int64_t> shape1Vec = {1};
        std::vector<int64_t> mask1Vec = {1, CODEWORDUBSIZE / subDim2};
        ge::Shape src0Shape(shape0Vec);
        ge::Shape src1Shape(shape1Vec);
        ge::Shape maskShape(mask1Vec);
        uint32_t tmpMaskSize = AscendC::GetSelectWithBytesMaskMinTmpSize(src0Shape, src1Shape, CTYPE,
                                                                         maskShape, MASKTYPE, false);

        tiling.set_nlist1(nlist1);
        tiling.set_nlist2(nlist2);
        tiling.set_segmentNum(segmentNum);
        tiling.set_n(n);
        tiling.set_subDim1(subDim1);
        tiling.set_baseNum(baseNum);
        tiling.set_subDim2(subDim2);
        tiling.set_nprobe2(nprobe2);
        tiling.set_segmentSize(segmentSize);
        tiling.set_segSizeVcMin(segSizeVcMin);

        tiling.set_sizeCodeWordUBBuffer(CODEWORDUBSIZE);
        tiling.set_sizeCodeWordL1BBuffer(CODEWORDL1BSIZE);
        tiling.set_sizeCodeWordL0BBuffer(CODEWORDL0BSIZE);
        tiling.set_cubeAlign(CUBE_ALIGN);
        tiling.set_blockDim(AI_CORE_USED);
        tiling.set_tmpMaskSize(tmpMaskSize);

        if (AI_CORE_USED == 0) {
            return ge::GRAPH_FAILED;
        }
        // 将单query的所有nprobe2尽量平均分给所有core
        // nprobe2 = 18 理想情况：前2个core处理6个probe，后6个core处理2个probe。
        // 即formerNum = 2，probePerBlockFormer = 3，probePerBlockLatter = 2。
        uint32_t probePerBlockFormer = DIV_UP(nprobe2, AI_CORE_USED);    // 每个core计算单个query的probePerBlock个nprobe2
        uint32_t formerBlkNum = static_cast<uint32_t>(nprobe2) % AI_CORE_USED;

        // 对于nprobe2=16，probePerBlockFormer和probePerBlockLatter都是2
        uint32_t probePerBlockLatter = (static_cast<uint32_t>(nprobe2) / AI_CORE_USED);

        tiling.set_formerBlkNum(formerBlkNum);
        tiling.set_probePerBlockFormer(probePerBlockFormer);
        tiling.set_probePerBlockLatter(probePerBlockLatter);  // 后(AI_CORE_USED - formerBlkNum)个core处理多少个probe

        if (context->GetRawTilingData() == nullptr) {
            return ge::GRAPH_FAILED;
        }
        context->SetBlockDim(AI_CORE_USED);
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
    class VSM3 : public OpDef {
    public:
        explicit VSM3(const char* name) : OpDef(name)
        {
            this->Input("queryCode")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT16})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("codeWord")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_UINT8})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("l2Indices")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_UINT64})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("diff1")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT16})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("diff2")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT16})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("precompute")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT16})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("mask")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_UINT8})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("attr_nlistl1")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT32})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("attr_nlistl2")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT32})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("attr_segmentl3")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT32})
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
            this->Output("vcMin")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT16})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Attr("nlist1").Int();
            this->Attr("nlist2").Int();
            this->Attr("segmentNum").Int();

            this->SetInferShape(ge::InferShape);
            this->SetInferDataType(ge::InferDataType);

            this->AICore()
                    .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend310p");
        }
    };

    OP_ADD(VSM3);
}
