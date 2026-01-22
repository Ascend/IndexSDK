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

#ifndef ASCENDC_OP_HOST_ASCENDC_IVFPQ_SEARCH_DISTANCE_TILING_H
#define ASCENDC_OP_HOST_ASCENDC_IVFPQ_SEARCH_DISTANCE_TILING_H

#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AscendcIvfpqSearchDistanceTopKTilingData)
TILING_DATA_FIELD_DEF(uint32_t, subSpaceNum);
TILING_DATA_FIELD_DEF(uint32_t, ksub);
TILING_DATA_FIELD_DEF(uint32_t, codeBaseSize);
TILING_DATA_FIELD_DEF(uint32_t, codeBlockSize);
TILING_DATA_FIELD_DEF(uint32_t, codeBlockNum);
TILING_DATA_FIELD_DEF(uint32_t, reduceBaseSize);
TILING_DATA_FIELD_DEF(uint32_t, usedAivNum);
TILING_DATA_FIELD_DEF(uint32_t, headAivNum);

TILING_DATA_FIELD_DEF(uint32_t, headAivDealDistResultNum);
TILING_DATA_FIELD_DEF(uint32_t, tailAivDealDistResultNum);
TILING_DATA_FIELD_DEF(uint32_t, reduceMode);
TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
TILING_DATA_FIELD_DEF(uint32_t, minSize);
TILING_DATA_FIELD_DEF(uint32_t, singleCoretotalBlock);
TILING_DATA_FIELD_DEF(uint32_t, minSizeWhole);
TILING_DATA_FIELD_DEF(uint32_t, minSizeSingle);
TILING_DATA_FIELD_DEF(uint32_t, topk);
TILING_DATA_FIELD_DEF(uint32_t, perCoreInnerBlockDealSize);
TILING_DATA_FIELD_DEF(uint32_t, topkOutterNum);

TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topkTilingData);
TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topkTilingDataWhole);
TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topkTilingDataSingle);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AscendcIvfpqSearchDistance, AscendcIvfpqSearchDistanceTopKTilingData)
REGISTER_TILING_DATA_CLASS(AscendcIvfpqSearchDistanceL2, AscendcIvfpqSearchDistanceTopKTilingData)
REGISTER_TILING_DATA_CLASS(AscendcIvfpqSearchDistanceIP, AscendcIvfpqSearchDistanceTopKTilingData)

enum class ReduceMode { L2, IP };
constexpr uint32_t IVFPQ_REDUCE_BASE_SIZE = 64;
constexpr uint32_t IVFPQ_ONE = 1;
constexpr uint32_t IVFPQ_NUM_32 = 32;
constexpr uint32_t IVFPQ_TOPK_OUTTER_NUM = 4096;
constexpr uint32_t IVFPQ_BLOCK_MAX_SIZE = 16384;
constexpr uint32_t IVFPQ_CODE_BLOCK_SIZE = 262144;
constexpr uint32_t IVFPQ_MAX_SHARE_MEM = 1024 * 216;

class IvfpqTiling {
public:
    ge::graphStatus ProcessTiling(
        gert::TilingContext *context, AscendcIvfpqSearchDistanceTopKTilingData &tilingData, ReduceMode reduceMode)
    {
        if (context == nullptr) {
            return ge::GRAPH_FAILED;
        }

        context_ = context;
        tilingData_ = &tilingData;
        if (reduceMode == ReduceMode::L2) {
            reduceMode_ = 0;
        } else {
            reduceMode_ = 1;
        }

        if (GetNpuInfo() != ge::GRAPH_SUCCESS || ProcessInput() != ge::GRAPH_SUCCESS ||
            TopKTiling() != ge::GRAPH_SUCCESS || FillTilingData() != ge::GRAPH_SUCCESS ||
            SetExtraConfig() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        return ge::GRAPH_SUCCESS;
    }

protected:
private:
    ge::graphStatus GetNpuInfo()
    {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        aivNum_ = ascendcPlatform.GetCoreNumAiv();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
        workSpaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus TopKTiling()
    {
        auto platformInfo = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        uint32_t maxSize = 0;
        minSize_ = 0;
        minSizeWhole_ = 0;
        minSizeSingle_ = 0;
        uint32_t blockSize = IVFPQ_TOPK_OUTTER_NUM;
        uint32_t dtypeSize = sizeof(float);

        perCoreInnerBlockDealSize_ = IVFPQ_BLOCK_MAX_SIZE;
        singleCoretotalBlock_ = perCoreInnerBlockDealSize_ / IVFPQ_TOPK_OUTTER_NUM;
        bool isLargest = (reduceMode_ == 1);

        AscendC::TopKTilingFunc(platformInfo,
            blockSize,  // inner
            1,          // outter
            topk_,
            dtypeSize,
            false,
            AscendC::TopKMode::TOPK_NORMAL,
            isLargest,
            tilingData_->topkTilingData);

        AscendC::TopKConfig config;
        config.algo = AscendC::TopKAlgo::RADIX_SELECT;
        config.order = AscendC::TopKOrder::UNSET;
        config.sorted = false;

        AscendC::GetTopKMaxMinTmpSize(blockSize,
            1,
            topk_,
            false,
            false,
            AscendC::TopKMode::TOPK_NORMAL,
            isLargest,
            ge::DataType::DT_FLOAT,
            config,
            maxSize,
            minSize_);

        AscendC::TopKTilingFunc(platformInfo,
            topk_ * (singleCoretotalBlock_ + 1),  // inner
            1,                                    // outter
            topk_,
            dtypeSize,
            true,
            AscendC::TopKMode::TOPK_NORMAL,
            isLargest,
            tilingData_->topkTilingDataWhole);

        config.sorted = true;
        AscendC::GetTopKMaxMinTmpSize(topk_ * (singleCoretotalBlock_ + 1),
            1,
            topk_,
            false,
            true,
            AscendC::TopKMode::TOPK_NORMAL,
            isLargest,
            ge::DataType::DT_FLOAT,
            config,
            maxSize,
            minSizeWhole_);

        AscendC::TopKTilingFunc(platformInfo,
            topk_ * (singleCoretotalBlock_),  // inner
            1,                                // outter
            topk_,
            dtypeSize,
            true,
            AscendC::TopKMode::TOPK_NORMAL,
            isLargest,
            tilingData_->topkTilingDataSingle);

        AscendC::GetTopKMaxMinTmpSize(topk_ * (singleCoretotalBlock_),
            1,
            topk_,
            false,
            true,
            AscendC::TopKMode::TOPK_NORMAL,
            isLargest,
            ge::DataType::DT_FLOAT,
            config,
            maxSize,
            minSizeSingle_);

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ParamDimCheck(const gert::Shape queryPQShape, const gert::Shape codeBaseShape,
        const gert::Shape codeOffsetShape, const gert::Shape codeSizeShape, const gert::Shape topkShape)
    {
        auto queryPQDimNum = queryPQShape.GetDimNum();
        auto codeBaseDimNum = codeBaseShape.GetDimNum();
        auto codeOffsetDimNum = codeOffsetShape.GetDimNum();
        auto codeSizeDimNum = codeSizeShape.GetDimNum();
        auto topkDimNum = topkShape.GetDimNum();
        if (queryPQDimNum != 2U) {
            return ge::GRAPH_FAILED;
        }

        if (codeBaseDimNum != 1U) {
            return ge::GRAPH_FAILED;
        }

        if (codeOffsetDimNum != 1U) {
            return ge::GRAPH_FAILED;
        }

        if (codeSizeDimNum != 1U) {
            return ge::GRAPH_FAILED;
        }

        if (topkDimNum != 2U) {
            return ge::GRAPH_FAILED;
        }

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ParamValueCheck()
    {
        if (subSpaceNum_ != 4) {
            return ge::GRAPH_FAILED;
        }

        if (ksub_ != 256) {
            return ge::GRAPH_FAILED;
        }

        if (codeSizeNum_ != codeBlockNum_) {
            return ge::GRAPH_FAILED;
        }

        if (codeBlockNum_ > 128) {
            return ge::GRAPH_FAILED;
        }

        return ge::GRAPH_SUCCESS;
    }

    virtual ge::graphStatus ProcessInput()
    {
        // InTensor
        auto queryPQShape = context_->GetInputShape(0)->GetStorageShape();     // 第0个输入 queryPQ
        auto codeBaseShape = context_->GetInputShape(1)->GetStorageShape();    // 第1个输入 codeBase
        auto codeOffsetShape = context_->GetInputShape(2)->GetStorageShape();  // 第2个输入 codeOffset
        auto codeSizeShape = context_->GetInputShape(3)->GetStorageShape();    // 第3个输入 codeSize
        auto topkShape = context_->GetInputShape(4)->GetStorageShape();        // 第3个输入 codeSize

        auto ret = ParamDimCheck(queryPQShape, codeBaseShape, codeOffsetShape, codeSizeShape, topkShape);

        subSpaceNum_ = static_cast<uint32_t>(queryPQShape.GetDim(0));
        ksub_ = static_cast<uint32_t>(queryPQShape.GetDim(1));
        codeBlockNum_ = static_cast<uint32_t>(codeOffsetShape.GetDim(0));
        codeSizeNum_ = static_cast<uint32_t>(codeSizeShape.GetDim(0));

        topk_ = static_cast<uint32_t>(topkShape.GetDim(0));
        codeBlockSize_ = IVFPQ_BLOCK_MAX_SIZE;

        if (codeBlockNum_ <= aivNum_) {
            aivNum_ = codeBlockNum_;
        } else {
            aivNum_ = IVFPQ_NUM_32;
        }
        ret = ParamValueCheck();
        return ret;
    }

    ge::graphStatus FillTilingData()
    {
        tilingData_->set_subSpaceNum(subSpaceNum_);
        tilingData_->set_ksub(ksub_);
        tilingData_->set_codeBaseSize(codeBaseSize_);
        tilingData_->set_codeBlockSize(codeBlockSize_);
        tilingData_->set_codeBlockNum(codeBlockNum_);
        tilingData_->set_reduceBaseSize(IVFPQ_REDUCE_BASE_SIZE);
        tilingData_->set_usedAivNum(aivNum_);
        tilingData_->set_headAivNum(headAivNum_);
        tilingData_->set_reduceMode(reduceMode_);
        tilingData_->set_tilingKey(tilingKey_);

        tilingData_->set_singleCoretotalBlock(singleCoretotalBlock_);
        tilingData_->set_minSize(minSize_);
        tilingData_->set_minSizeWhole(minSizeWhole_);
        tilingData_->set_minSizeSingle(minSizeSingle_);
        tilingData_->set_topk(topk_);
        tilingData_->set_perCoreInnerBlockDealSize(perCoreInnerBlockDealSize_);
        tilingData_->set_topkOutterNum(IVFPQ_TOPK_OUTTER_NUM);

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SetExtraConfig()
    {
        context_->SetLocalMemorySize(IVFPQ_MAX_SHARE_MEM);
        context_->SetBlockDim(tilingData_->get_usedAivNum());
        context_->SetTilingKey(tilingData_->get_tilingKey());
        tilingData_->SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(tilingData_->GetDataSize());

        size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
        if (currentWorkspace == nullptr) {
            return ge::GRAPH_FAILED;
        }
        currentWorkspace[0] = workSpaceSize_;

        return ge::GRAPH_SUCCESS;
    }

private:
    gert::TilingContext *context_ = nullptr;
    AscendcIvfpqSearchDistanceTopKTilingData *tilingData_ = nullptr;

    uint32_t aivNum_ = 0;
    uint32_t usedAivNum_ = 0;
    uint32_t libapiSize_ = 0;
    uint32_t workSpaceSize_ = 0;
    uint64_t ubSize_ = 0;
    uint64_t tilingKey_ = 0;

    uint32_t subSpaceNum_ = 0;
    uint32_t ksub_ = 0;
    uint32_t codeBaseSize_ = 0;
    uint32_t codeBlockNum_ = 0;
    uint32_t codeSizeNum_ = 0;

    uint32_t headAivNum_ = 0;
    uint32_t reduceBlockNumPerTailCore_ = 0;
    uint32_t reduceMode_ = 0;

    uint32_t codeBlockSize_ = 0;

    uint32_t minSize_ = 0;
    uint32_t minSizeWhole_ = 0;
    uint32_t minSizeSingle_ = 0;
    uint32_t singleCoretotalBlock_ = 0;

    uint32_t perCoreInnerBlockDealSize_ = 0;

    uint32_t topk_ = 0;
};

}  // namespace optiling
#endif
