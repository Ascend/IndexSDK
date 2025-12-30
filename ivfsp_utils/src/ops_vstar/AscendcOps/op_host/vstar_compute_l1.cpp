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

#include "vstar_compute_l1_tiling.h"
#include "register/op_def_registry.h"
#include "chrono"

namespace {
    constexpr uint32_t AI_CORE_USED = 8;
    constexpr uint32_t CUBE_ALIGN = 16;
    constexpr uint32_t TILE_UNIT = CUBE_ALIGN;

    constexpr uint32_t KB = 1024;
    constexpr uint32_t L1_BYTE_SIZE = 1024 * KB;
    constexpr uint32_t CODEBOOK_L1_BYTE_SIZE = 256 * KB * 2;
    constexpr uint32_t A2_BYTE_SIZE = 32 * KB;
    constexpr uint32_t CODEBOOK_B2_BYTE_SIZE = 32 * KB * 2;

    uint32_t DIV_UP(uint32_t x, uint32_t y)
    {
        return (x + y - 1) / y;
    }
}

namespace optiling {
    /**
     * x1: (nq, dim)
     * x2: (nlist * subSpaceDim // 16, dim // 16, 16, 16)
     * @param context
     * @return
     */
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        if (context == nullptr || context->GetAttrs() == nullptr) {
            return ge::GRAPH_FAILED;
        }
        uint32_t blockDim = AI_CORE_USED;
        auto subSpaceDim = static_cast<uint32_t>(*context->GetAttrs()->GetInt(0));
        VstarComputeL1TilingData tiling;
        const gert::StorageShape *x1_shape = context->GetInputShape(0);
        if (x1_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        uint32_t nq = static_cast<uint32_t>(x1_shape->GetStorageShape().GetDim(0));
        uint32_t dim = static_cast<uint32_t>(x1_shape->GetStorageShape().GetDim(1));
        tiling.set_subSpaceDim(subSpaceDim);
        tiling.set_nq(nq);
        tiling.set_dim(dim);
        const gert::StorageShape *x2_shape = context->GetInputShape(1);
        if (x2_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        uint32_t n0 = static_cast<uint32_t>(x2_shape->GetStorageShape().GetDim(0)) * CUBE_ALIGN;
        if (subSpaceDim == 0) {
            return ge::GRAPH_FAILED;
        }
        uint32_t nlist = n0 / subSpaceDim;
        tiling.set_nlist(nlist);                     //  nlist %16 == 0
        /**compute block information*/
        if (blockDim == 0) {
            return ge::GRAPH_FAILED;
        }
        uint32_t blockSize = DIV_UP(nlist, blockDim);
        tiling.set_blockSize(blockSize);
        /** cbByteSizeL1 <= 256KB */
        uint32_t cbByteSize = subSpaceDim * dim * sizeof(uint16_t);
        if (cbByteSize == 0) {
            return ge::GRAPH_FAILED;
        }
        uint32_t cbNumL1 = CODEBOOK_L1_BYTE_SIZE / cbByteSize;
        cbNumL1 = std::max(1U, cbNumL1);
        // 保证 blockSize%cbNumL1==0
        if (blockSize % cbNumL1 != 0) {
            if (cbNumL1 > 16) {
                cbNumL1 = 16;
            } else if (cbNumL1 > 8) {
                cbNumL1 = 8;
            } else if (cbNumL1 > 4) {
                cbNumL1 = 4;
            } else if (cbNumL1 > 2) {
                cbNumL1 = 2;
            } else {
                cbNumL1 = 1;
            }
        }
        uint32_t cbTileSizeL1 = cbNumL1 * subSpaceDim;
        tiling.set_cbTileSizeL1(cbTileSizeL1);
        /** cbByteSizeL0 <= 32KB */
        float cbNumL0 = CODEBOOK_B2_BYTE_SIZE * 1.0 / cbByteSize;
        cbNumL0 = std::min(cbNumL1 * 1.0F, cbNumL0);
        uint32_t cbTileSizeB2 = static_cast<uint32_t>(cbNumL0 * subSpaceDim);
        tiling.set_cbTileSizeB2(cbTileSizeB2);

        if (cbNumL1 == 0 || cbTileSizeB2 == 0) {
            return ge::GRAPH_FAILED;
        }
        tiling.set_cbLoopsL1(blockSize / cbNumL1);
        tiling.set_cbLoopsB2(cbTileSizeL1 / cbTileSizeB2);

        context->SetBlockDim(blockDim);
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
        if (context == nullptr || context->GetAttrs() == nullptr) {
            return ge::GRAPH_FAILED;
        }
        auto subSpaceDim = static_cast<uint32_t>(*context->GetAttrs()->GetInt(0));
        const gert::Shape* x0_shape = context->GetInputShape(0);
        const gert::Shape* x1_shape = context->GetInputShape(1);
        if (x0_shape == nullptr || x1_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        auto nq = x0_shape->GetDim(0);
        auto n0 = x1_shape->GetDim(0) * 16;
        gert::Shape* y0_shape = context->GetOutputShape(0);
        if (y0_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        y0_shape->SetDimNum(2);
        y0_shape->SetDim(0, nq);
        y0_shape->SetDim(1, n0);
        gert::Shape* y1_shape = context->GetOutputShape(1);
        if (y1_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        y1_shape->SetDimNum(2);
        y1_shape->SetDim(0, nq);
        y1_shape->SetDim(1, n0 / subSpaceDim);
        gert::Shape* y2_shape = context->GetOutputShape(2);
        if (y2_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        y2_shape->SetDimNum(2);
        y2_shape->SetDim(0, 8);
        y2_shape->SetDim(1, 16);
        return GRAPH_SUCCESS;
    }

    static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
    {
        return GRAPH_SUCCESS;
    }
}


namespace ops {
    class VstarComputeL1 : public OpDef {
    public:
        explicit VstarComputeL1(const char* name) : OpDef(name)
        {
            this->Input("x0")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("x1")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT16})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("y0")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT16})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("y1")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT16})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("y2")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_UINT16})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Attr("subSpaceDim").Int();

            this->SetInferShape(ge::InferShape);
            this->SetInferDataType(ge::InferDataType);

            this->AICore()
                    .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend310p");
        }
    };

    OP_ADD(VstarComputeL1);
}
