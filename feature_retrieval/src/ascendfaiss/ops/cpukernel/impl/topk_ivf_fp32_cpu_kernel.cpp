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


#include "topk_ivf_fp32_cpu_kernel.h"

#include <iostream>
#include <algorithm>
#include <string>

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "kernel_tensor.h"
#include "kernel_utils.h"
#include "kernel_shared_def.h"

namespace {
const char* TOPK_IVF_FP32 = "TopkIvfFp32";
}

namespace aicpu {
uint32_t TopkIvffP32CpuKernel::Compute(CpuKernelContext &ctx)
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
            DoCompute(core, start, inputs, outputs, funcGreater);
        } else {
            // put least one to top of heap
            DoCompute(core, start, inputs, outputs, funcLess);
        }
    };

#ifdef AICPU_UTEST
    computeFunc(0, 1);
#else
    CpuKernelUtils::ParallelFor(ctx, core, 1, computeFunc);
#endif

    return KERNEL_STATUS_OK;
}

uint32_t TopkIvffP32CpuKernel::GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs) const
{
    KERNEL_LOG_INFO("TopkIvffP32CpuKernel GetInOutAndCheck begin");

    inputs.indists = ctx.Input(INPUT_NUM0);
    inputs.vmdists = ctx.Input(INPUT_NUM1);
    inputs.ids = ctx.Input(INPUT_NUM2);
    inputs.size = ctx.Input(INPUT_NUM3);
    inputs.opflag = ctx.Input(INPUT_NUM4);
    inputs.attr = ctx.Input(INPUT_NUM5);
    outputs.outdists = ctx.Output(INPUT_NUM0);
    outputs.outlabels = ctx.Output(INPUT_NUM1);

    KERNEL_CHECK_NULLPTR(inputs.indists, KERNEL_STATUS_PARAM_INVALID, "Get input[0], name[indists] failed");
    KERNEL_CHECK_NULLPTR(inputs.vmdists, KERNEL_STATUS_PARAM_INVALID, "Get input[1], name[vmdists] failed");
    KERNEL_CHECK_NULLPTR(inputs.ids, KERNEL_STATUS_PARAM_INVALID, "Get input[1], name[ids] failed");
    KERNEL_CHECK_NULLPTR(inputs.size, KERNEL_STATUS_PARAM_INVALID, "Get input[2], name[size] failed");
    KERNEL_CHECK_NULLPTR(inputs.opflag, KERNEL_STATUS_PARAM_INVALID, "Get input[3], name[opflag] failed");
    KERNEL_CHECK_NULLPTR(inputs.attr, KERNEL_STATUS_PARAM_INVALID, "Get input[4], name[attr] failed");
    KERNEL_CHECK_NULLPTR(outputs.outdists, KERNEL_STATUS_PARAM_INVALID, "Get output[0], name[outdists] failed");
    KERNEL_CHECK_NULLPTR(outputs.outlabels, KERNEL_STATUS_PARAM_INVALID, "Get output[1], name[outlabels] failed");

    KERNEL_LOG_INFO("Shape of input[0][indists] is %s",
        ShapeToString(inputs.indists->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[1][vmdists] is %s",
        ShapeToString(inputs.vmdists->GetTensorShape()->GetDimSizes()).c_str());
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

uint32_t TopkIvffP32CpuKernel::CheckInputShapes(const Inputs &inputs)
{
    KERNEL_LOG_INFO("TopkIvffP32CpuKernel CheckInputShapes begin");

    auto shapeIndists = inputs.indists->GetTensorShape();
    auto shapeVmdists = inputs.vmdists->GetTensorShape();
    auto shapeIds = inputs.ids->GetTensorShape();
    auto shapeSize = inputs.size->GetTensorShape();
    auto shapeOpflag = inputs.opflag->GetTensorShape();
    auto shapeAttr = inputs.attr->GetTensorShape();

    KERNEL_CHECK_TRUE(shapeIndists->GetDims() == INPUT_NUM4,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][indists] must be 4");
    KERNEL_CHECK_TRUE(shapeVmdists->GetDims() == INPUT_NUM4,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][vmdists] must be 4");
    KERNEL_CHECK_TRUE(shapeIds->GetDims() == INPUT_NUM3,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][ids] must be 3");
    KERNEL_CHECK_TRUE(shapeSize->GetDims() == INPUT_NUM3,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][size] must be 3");
    KERNEL_CHECK_TRUE(shapeAttr->GetDims() == INPUT_NUM1,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][attr] must be 1");

    auto nq0 = shapeIndists->GetDimSize(INPUT_NUM0);
    auto nq1 = shapeVmdists->GetDimSize(INPUT_NUM0);
    KERNEL_CHECK_TRUE(nq0 == nq1, KERNEL_STATUS_PARAM_INVALID, "Nq of inputs must be same");
    nq_ = nq0;

    auto handleBatch0 = shapeIndists->GetDimSize(INPUT_NUM2);
    auto handleBatch1 = shapeVmdists->GetDimSize(INPUT_NUM2);
    KERNEL_CHECK_TRUE(handleBatch0 == handleBatch1, KERNEL_STATUS_PARAM_INVALID, "Handle batch of inputs must be same");
    handleBatch_ = handleBatch0;

    flagSize_ = shapeOpflag->GetDimSize(INPUT_NUM3);

    auto attrCount = shapeAttr->GetDimSize(INPUT_NUM0);
    KERNEL_CHECK_TRUE(attrCount == TOPK_IVF_ATTR_IDX_COUNT,
        KERNEL_STATUS_PARAM_INVALID, "Num of attrs must be %d", TOPK_IVF_ATTR_IDX_COUNT);

    auto attr = static_cast<int64_t *>(inputs.attr->GetData());
    asc_ = *(attr + TOPK_IVF_ATTR_ASC_IDX);
    k_ = *(attr + TOPK_IVF_ATTR_K_IDX);
    burstLen_ = *(attr + TOPK_IVF_ATTR_BURST_LEN_IDX);
    blockNum_ = *(attr + TOPK_IVF_ATTR_BLOCK_NUM_IDX);
    flagNum_ = *(attr + TOPK_IVF_ATTR_FLAG_NUM_IDX);
    KERNEL_CHECK_TRUE(k_ > 0 && burstLen_ > 0 && asc_ >= 0 && blockNum_ > 0 && flagNum_ > 0,
        KERNEL_STATUS_PARAM_INVALID, "Value of asc, k, bustLen, blockNum, flagNum must ge 0");

    return KERNEL_STATUS_OK;
}

void TopkIvffP32CpuKernel::UpdateInOutShape(Inputs &inputs, Outputs &outputs) const
{
    KERNEL_LOG_INFO("TopkIvffP32CpuKernel UpdateInOutShape begin");

    auto shapeIndists = inputs.indists->GetTensorShape();
    std::vector<int64_t> dimIndists = shapeIndists->GetDimSizes();
    dimIndists[INPUT_NUM1] = blockNum_;
    shapeIndists->SetDimSizes(dimIndists);

    auto shapeVmdists = inputs.vmdists->GetTensorShape();
    std::vector<int64_t> dimVmdists = shapeVmdists->GetDimSizes();
    dimVmdists[INPUT_NUM1] = blockNum_;
    shapeVmdists->SetDimSizes(dimVmdists);

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

void TopkIvffP32CpuKernel::InitTopkHeap(Outputs &outputs) const
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
void TopkIvffP32CpuKernel::DoCompute(size_t tcnt, size_t tid, const Inputs &inputs, Outputs &outputs, C &&cmp)
{
    KernelTensor<float> indists(inputs.indists);
    KernelTensor<float> vmdists(inputs.vmdists);
    KernelTensor<int64_t> ids(inputs.ids);
    KernelTensor<uint32_t> size(inputs.size);
    KernelTensor<uint16_t> opflag(inputs.opflag);

    KernelTensor<float> outdists(outputs.outdists);
    KernelTensor<int64_t> outlabels(outputs.outlabels);

    for (int64_t qidx = tid; qidx < nq_; qidx += tcnt) {
        for (int64_t bidx = 0; bidx < blockNum_; ++bidx) {
            uint32_t count = 0;
            for (int64_t hidx = 0; hidx < handleBatch_; ++hidx) {
                count += *(size.GetSubTensorDim2(qidx, bidx, hidx));
            }
            if (count == 0) {
                // no dists to be calculated in this block
                continue;
            }
            auto flagPtr = opflag.GetSubTensorDim1(qidx, bidx);
            for (int64_t i = 0; i < flagNum_; ++i) {
                WAITING_FLAG_READY(*(flagPtr + i * flagSize_), TIMEOUT_CHECK_TICK, TIMEOUT_MS);
            }
            for (int64_t hidx = 0; hidx < handleBatch_; ++hidx) {
                bool reorder = (bidx + 1 == blockNum_ && hidx + 1 == handleBatch_);
                ComputeQuery(qidx, bidx, hidx, indists, vmdists, ids, size, outdists, outlabels, reorder, cmp);
            }
        }
    }
}

template <typename C>
void TopkIvffP32CpuKernel::ComputeQuery(int64_t qidx,
                                        int64_t bidx,
                                        int64_t hidx,
                                        KernelTensor<float> &indistsTensor,
                                        KernelTensor<float> &vmdistsTensor,
                                        KernelTensor<int64_t> &idsTensor,
                                        KernelTensor<uint32_t> &sizeTensor,
                                        KernelTensor<float> &outdistsTensor,
                                        KernelTensor<int64_t> &outlabelsTensor,
                                        bool reorder,
                                        C &&cmp)
{
    float *indists = indistsTensor.GetSubTensorDim2(qidx, bidx, hidx);
    float *vmdists = vmdistsTensor.GetSubTensorDim2(qidx, bidx, hidx);
    int64_t *ids = idsTensor.GetSubTensorDim2(qidx, bidx, hidx);
    uint32_t *size = sizeTensor.GetSubTensorDim2(qidx, bidx, hidx);
    float *outdists = outdistsTensor.GetSubTensorDim0(qidx);
    int64_t *outlabel = outlabelsTensor.GetSubTensorDim0(qidx);

    int64_t ntotal = static_cast<int64_t>(*size);
    int64_t idx = 0;
    int64_t *id = reinterpret_cast<int64_t *>(*ids);

    int64_t burstSize = ntotal / burstLen_;
    for (int64_t i = 0; i < burstSize; ++i) {
        if (!cmp(outdists[0], vmdists[i * 2])) { // vmdists[i*2] is dists, vmdists[i*2+1] is label
            // skip one burst
            idx += burstLen_;
            continue;
        }
        for (int64_t j = 0; j < burstLen_ && idx < ntotal; ++j, ++idx) {
            if (cmp(outdists[0], indists[idx])) {
                outdists[0] = indists[idx];
                outlabel[0] = *(id + idx);
                UpdateHeap(outdists, outlabel, k_, 0, cmp);
            }
        }
    }
    while (idx < ntotal) {
        if (cmp(outdists[0], indists[idx])) {
            outdists[0] = indists[idx];
            outlabel[0] = *(id + idx);
            UpdateHeap(outdists, outlabel, k_, 0, cmp);
        }
        ++idx;
    }
    if (reorder) {
        for (int64_t i = k_ - 1; i >= 1; --i) {
            std::swap(outdists[0], outdists[i]);
            std::swap(outlabel[0], outlabel[i]);
            UpdateHeap(outdists, outlabel, i, 0, cmp);
        }
    }
}

REGISTER_CPU_KERNEL(TOPK_IVF_FP32, TopkIvffP32CpuKernel);
} // namespace aicpu