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

#include "topk_ivfpq_l3_cpu_kernel.h"

#include <iostream>
#include <algorithm>
#include <string>

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "kernel_tensor.h"
#include "kernel_utils.h"
#include "kernel_shared_def.h"

namespace {
const char* TOPK_IVFPQ_L3 = "TopkIvfpqL3";
}

namespace aicpu {
uint32_t TopkIvfpqL3CpuKernel::Compute(CpuKernelContext &ctx)
{
    Inputs inputs;
    Outputs outputs;
    auto ret = GetInOutAndCheck(ctx, inputs, outputs);
    if (ret != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("Failed to get inputs or outputs");
        return ret;
    }

    ret = CheckInputShapes(inputs);
    if (ret != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("Failed to check input shapes");
        return ret;
    }

    UpdateInOutShape(inputs, outputs);
    InitTopkHeap(outputs);

    auto funcLess = [](float a, float b) -> bool { return a < b; };
    auto funcGreater = [](float a, float b) -> bool { return a > b; };

#ifdef AICPU_UTEST
    uint32_t core = 1;
#else
    uint32_t core = std::min({CpuKernelUtils::GetCPUNum(ctx), static_cast<uint32_t>(nq_)});
#endif

    auto computeFunc = [&](size_t start, size_t end) {
        (void)end; // end is unuseful in this function
        if (asc_ != 0) {
            // put greatest one to top of heap
            DoCompute(start, core, inputs, outputs, funcGreater);
        } else {
            // put least one to top of heap
            DoCompute(start, core, inputs, outputs, funcLess);
        }
    };

#ifdef AICPU_UTEST
    computeFunc(0, 1);
#else
    CpuKernelUtils::ParallelFor(ctx, core, 1, computeFunc);
#endif
    return KERNEL_STATUS_OK;
}

uint32_t TopkIvfpqL3CpuKernel::GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs) const
{
    KERNEL_LOG_INFO("TopkIvfpqL3CpuKernel GetInOutAndCheck begin");

    inputs.topkOffsets = ctx.Input(INPUT_NUM0);
    inputs.topkDists = ctx.Input(INPUT_NUM1);
    inputs.ids = ctx.Input(INPUT_NUM2);
    inputs.size = ctx.Input(INPUT_NUM3);
    inputs.opflag = ctx.Input(INPUT_NUM4);
    inputs.attr = ctx.Input(INPUT_NUM5);
    outputs.outdists = ctx.Output(INPUT_NUM0);
    outputs.outlabels = ctx.Output(INPUT_NUM1);

    KERNEL_CHECK_NULLPTR(inputs.topkOffsets, KERNEL_STATUS_PARAM_INVALID, "Get input[0], name[topkOffsets] failed");
    KERNEL_CHECK_NULLPTR(inputs.topkDists, KERNEL_STATUS_PARAM_INVALID, "Get input[1], name[topkDists] failed");
    KERNEL_CHECK_NULLPTR(inputs.ids, KERNEL_STATUS_PARAM_INVALID, "Get input[2], name[ids] failed");
    KERNEL_CHECK_NULLPTR(inputs.size, KERNEL_STATUS_PARAM_INVALID, "Get input[3], name[size] failed");
    KERNEL_CHECK_NULLPTR(inputs.opflag, KERNEL_STATUS_PARAM_INVALID, "Get input[4], name[opflag] failed");
    KERNEL_CHECK_NULLPTR(inputs.attr, KERNEL_STATUS_PARAM_INVALID, "Get input[5], name[attr] failed");
    KERNEL_CHECK_NULLPTR(outputs.outdists, KERNEL_STATUS_PARAM_INVALID, "Get output[0], name[outdists] failed");
    KERNEL_CHECK_NULLPTR(outputs.outlabels, KERNEL_STATUS_PARAM_INVALID, "Get output[1], name[outlabels] failed");

    KERNEL_LOG_INFO("Shape of input[0][topkOffsets] is %s",
        ShapeToString(inputs.topkOffsets->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[1][topkDists] is %s",
        ShapeToString(inputs.topkDists->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[2][ids] is %s",
        ShapeToString(inputs.ids->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[3][size] is %s",
        ShapeToString(inputs.size->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[4][opflag] is %s",
        ShapeToString(inputs.opflag->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[5][attr] is %s",
        ShapeToString(inputs.attr->GetTensorShape()->GetDimSizes()).c_str());

    return KERNEL_STATUS_OK;
}

uint32_t TopkIvfpqL3CpuKernel::CheckInputShapes(const Inputs &inputs)
{
    KERNEL_LOG_INFO("TopkIvfpqL3CpuKernel CheckInputShapes begin");

    auto shapeTopkOffsets = inputs.topkOffsets->GetTensorShape();
    auto shapeTopkDists = inputs.topkDists->GetTensorShape();
    auto shapeIds = inputs.ids->GetTensorShape();
    auto shapeSize = inputs.size->GetTensorShape();
    auto shapeOpflag = inputs.opflag->GetTensorShape();
    auto shapeAttr = inputs.attr->GetTensorShape();

    KERNEL_CHECK_TRUE(shapeTopkOffsets->GetDims() == INPUT_NUM4,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][topkOffsets] must be 4");
    KERNEL_CHECK_TRUE(shapeTopkDists->GetDims() == INPUT_NUM4,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][topkDists] must be 4");
    KERNEL_CHECK_TRUE(shapeIds->GetDims() == INPUT_NUM3,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][ids] must be 3");
    KERNEL_CHECK_TRUE(shapeSize->GetDims() == INPUT_NUM3,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][size] must be 3");
    KERNEL_CHECK_TRUE(shapeAttr->GetDims() == INPUT_NUM1,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][attr] must be 1");

    auto nq0 = shapeTopkOffsets->GetDimSize(INPUT_NUM0);
    auto nq1 = shapeTopkDists->GetDimSize(INPUT_NUM0);
    KERNEL_CHECK_TRUE(nq0 == nq1, KERNEL_STATUS_PARAM_INVALID, "Nq of inputs must be same");
    nq_ = nq0;

    auto handleBatch0 = shapeTopkOffsets->GetDimSize(INPUT_NUM2);
    auto handleBatch1 = shapeTopkDists->GetDimSize(INPUT_NUM2);
    KERNEL_CHECK_TRUE(handleBatch0 == handleBatch1, KERNEL_STATUS_PARAM_INVALID, "Handle batch of inputs must be same");
    handleBatch_ = handleBatch0;

    auto attrCount = shapeAttr->GetDimSize(INPUT_NUM0);
    KERNEL_CHECK_TRUE(attrCount == TOPK_IVFPQ_L3_ATTR_IDX_COUNT,
        KERNEL_STATUS_PARAM_INVALID, "Num of attrs must be %d", TOPK_IVFPQ_L3_ATTR_IDX_COUNT);

    auto attr = static_cast<int64_t *>(inputs.attr->GetData());
    asc_ = *(attr + TOPK_IVFPQ_L3_ATTR_ASC_IDX);
    k_ = *(attr + TOPK_IVFPQ_L3_ATTR_K_IDX);
    blockNum_ = *(attr + TOPK_IVFPQ_L3_ATTR_BLOCK_NUM_IDX);
    flagNum_ = *(attr + TOPK_IVFPQ_L3_ATTR_FLAG_NUM_IDX);
    KERNEL_CHECK_TRUE(k_ > 0 && asc_ >= 0 && blockNum_ > 0 && flagNum_ > 0,
        KERNEL_STATUS_PARAM_INVALID, "Value of asc, k, blockNum, flagNum must ge 0");

    return KERNEL_STATUS_OK;
}

void TopkIvfpqL3CpuKernel::UpdateInOutShape(Inputs &inputs, Outputs &outputs) const
{
    KERNEL_LOG_INFO("TopkIvfpqL3CpuKernel UpdateInOutShape begin");

    auto shapeTopkOffsets = inputs.topkOffsets->GetTensorShape();
    std::vector<int64_t> dimTopkOffsets = shapeTopkOffsets->GetDimSizes();
    dimTopkOffsets[INPUT_NUM1] = blockNum_;
    shapeTopkOffsets->SetDimSizes(dimTopkOffsets);

    auto shapeTopkDists = inputs.topkDists->GetTensorShape();
    std::vector<int64_t> dimTopkDists = shapeTopkDists->GetDimSizes();
    dimTopkDists[INPUT_NUM1] = blockNum_;
    shapeTopkDists->SetDimSizes(dimTopkDists);

    auto shapeIds = inputs.ids->GetTensorShape();
    std::vector<int64_t> dimIds = shapeIds->GetDimSizes();
    dimIds[INPUT_NUM1] = blockNum_;
    shapeIds->SetDimSizes(dimIds);

    auto shapeSize = inputs.size->GetTensorShape();
    std::vector<int64_t> dimSize = shapeSize->GetDimSizes();
    dimSize[INPUT_NUM1] = blockNum_;
    shapeSize->SetDimSizes(dimSize);

    auto shapeOpFlag = inputs.opflag->GetTensorShape();
    std::vector<int64_t> dimOpFlag = shapeOpFlag->GetDimSizes();
    dimOpFlag[INPUT_NUM1] = blockNum_;
    shapeOpFlag->SetDimSizes(dimOpFlag);

    auto shapeOutdists = outputs.outdists->GetTensorShape();
    std::vector<int64_t> dimOutdists;
    dimOutdists.push_back(nq_);
    dimOutdists.push_back(k_);
    shapeOutdists->SetDimSizes(dimOutdists);

    auto shapeOutlabels = outputs.outlabels->GetTensorShape();
    std::vector<int64_t> dimOutlabels;
    dimOutlabels.push_back(nq_);
    dimOutlabels.push_back(k_);
    shapeOutlabels->SetDimSizes(dimOutlabels);
}

void TopkIvfpqL3CpuKernel::InitTopkHeap(Outputs &outputs) const
{
    float *outdists = static_cast<float *>(outputs.outdists->GetData());
    int64_t *outlabels = static_cast<int64_t *>(outputs.outlabels->GetData());
    std::fill_n(outlabels, nq_ * k_, 0xffffffffffffffff); // 0xffffffffffffffff为无效label
    if (asc_ != 0) {
        std::fill_n(outdists, nq_ * k_, 0x7f7fffff); // 小端排序模式初始化为0x7f7fffff，即最大值
    } else {
        std::fill_n(outdists, nq_ * k_, 0.0001); // 大端排序模式下初始化为0.0001
    }
}

template <typename C>
void TopkIvfpqL3CpuKernel::DoCompute(size_t startQidx, size_t queryCount,
                                     const Inputs &inputs, Outputs &outputs, C &&cmp)
{
    KernelTensor<int32_t> topkOffsets(inputs.topkOffsets);
    KernelTensor<float> topkDists(inputs.topkDists);
    KernelTensor <int64_t> ids(inputs.ids);
    KernelTensor <uint16_t> opflag(inputs.opflag);

    KernelTensor<float> outdists(outputs.outdists);
    KernelTensor <int64_t> outlabels(outputs.outlabels);

    for (int64_t qidx = startQidx; qidx < nq_; qidx += queryCount) {
        for (int64_t bidx = 0; bidx < blockNum_; ++bidx) {
            auto flagPtr = opflag.GetSubTensorDim1(qidx, bidx);
            for (int64_t i = 0; i < flagNum_; ++i) {
                WAITING_FLAG_READY(*(flagPtr + i * flagSize_), TIMEOUT_CHECK_TICK, TIMEOUT_MS);
            }
            for (int64_t i = 0; i < handleBatch_; ++i) {
                bool isLastBlock = (bidx + 1 == blockNum_ && i + 1 == handleBatch_);
                ComputeBlock(qidx, bidx, i, topkOffsets, topkDists, ids, outdists, outlabels, isLastBlock, cmp);
            }
        }
    }
}

template <typename T, typename C>
void TopkIvfpqL3CpuKernel::UpdateHeapByPos(int64_t outdisPos, float *outdists, int64_t indisPos, float *topkDists,
                                           int64_t outlabelPos, T outLabelValue, T *outlabel, int64_t index,
                                           C &&cmp)
{
    outdists[outdisPos] = topkDists[indisPos];
    outlabel[outlabelPos] = static_cast<T>(outLabelValue);
    UpdateHeap<T, C>(outdists, outlabel, k_, index, cmp);
}

template <typename C>
void TopkIvfpqL3CpuKernel::ComputeBlock(int64_t qidx,
                                        int64_t bidx,
                                        int64_t hidx,
                                        KernelTensor<int32_t> &topkOffsetsTensor,
                                        KernelTensor<float> &topkDistsTensor,
                                        KernelTensor<int64_t> &idsTensor,
                                        KernelTensor<float> &outdistsTensor,
                                        KernelTensor<int64_t> &outlabelsTensor,
                                        bool isLastBlock,
                                        C &&cmp)
{
    int32_t *topkOffsets = topkOffsetsTensor.GetSubTensorDim2(qidx, bidx, hidx);
    float *topkDists = topkDistsTensor.GetSubTensorDim2(qidx, bidx, hidx);
    int64_t *ids = idsTensor.GetSubTensorDim2(qidx, bidx, hidx);
    int64_t *id = reinterpret_cast<int64_t *>(*ids);

    float *outdists = outdistsTensor.GetSubTensorDim0(qidx);
    int64_t *outlabel = outlabelsTensor.GetSubTensorDim0(qidx);

    for (int64_t i = 0; i < k_; ++i) {
        if (!cmp(outdists[0], topkDists[i])) {
            break;
        }
        int32_t offset = topkOffsets[i];
        UpdateHeapByPos(0, outdists, i, topkDists, 0,
                        reinterpret_cast<int64_t>(id + offset), outlabel, 0, cmp);
    }

    if (isLastBlock) {
        for (int64_t i = k_ - 1; i >= 1; --i) {
            std::swap(outdists[0], outdists[i]);
            std::swap(outlabel[0], outlabel[i]);
            UpdateHeap(outdists, outlabel, i, 0, cmp);
        }
    }
}

REGISTER_CPU_KERNEL(TOPK_IVFPQ_L3, TopkIvfpqL3CpuKernel);
} // namespace aicpu