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


#include "topk_multisearch_ivf_v2_cpu_kernel.h"

#include <iostream>
#include <algorithm>
#include <string>

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "kernel_tensor.h"
#include "kernel_utils.h"
#include "kernel_shared_def.h"

namespace {
const char* TOPK_MULTISEARCH_IVF_V2 = "TopkMultisearchIvfV2";
const uint32_t THREAD_CNT = 6;
}

namespace aicpu {
uint32_t TopkMultisearchIvfV2CpuKernel::Compute(CpuKernelContext &ctx)
{
    Inputs inputsV2;
    Outputs outputsV2;
    auto ret = GetInOutAndCheck(ctx, inputsV2, outputsV2);
    if (ret != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("Failed to get inputs or outputs");
        return ret;
    }

    ret = CheckInputShapes(inputsV2);
    if (ret != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("Failed to check input shapes");
        return ret;
    }

    UpdateInOutShape(inputsV2, outputsV2);

    InitTopkHeap(outputsV2);

    auto funcLessV2 = [](float16_t a, float16_t b) -> bool { return a < b; };
    auto funcGreaterV2 = [](float16_t a, float16_t b) -> bool { return a > b; };

#ifdef AICPU_UTEST
    uint32_t core = 1;
#else
    uint32_t core = std::min({CpuKernelUtils::GetCPUNum(ctx), static_cast<uint32_t>(nq_), THREAD_CNT});
    if (isIndexParall) {
        core = std::min({CpuKernelUtils::GetCPUNum(ctx), static_cast<uint32_t>(indexNum_), THREAD_CNT});
    }
#endif

    auto computeFuncV2 = [&](size_t start, size_t end) {
        (void)end; // end is unuseful in this function
        if (asc_ != 0) {
            // put greatest one to top of heap
            if (isIndexParall) {
                DoComputeParallForIndex(core, start, inputsV2, outputsV2, funcGreaterV2);
            } else {
                DoCompute(core, start, inputsV2, outputsV2, funcGreaterV2);
            }
        } else {
            // put least one to top of heap
            if (isIndexParall) {
                DoComputeParallForIndex(core, start, inputsV2, outputsV2, funcLessV2);
            } else {
                DoCompute(core, start, inputsV2, outputsV2, funcLessV2);
            }
        }
    };

#ifdef AICPU_UTEST
    computeFuncV2(0, 1);
#else
    CpuKernelUtils::ParallelFor(ctx, core, 1, computeFuncV2);
#endif

    return KERNEL_STATUS_OK;
}

uint32_t TopkMultisearchIvfV2CpuKernel::GetInOutAndCheck(const CpuKernelContext &ctx,
    Inputs &inputs, Outputs &outputs) const
{
    KERNEL_LOG_INFO("TopkIvfCpuKernel GetInOutAndCheck begin");

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
    KERNEL_CHECK_NULLPTR(inputs.ids, KERNEL_STATUS_PARAM_INVALID, "Get input[2], name[vmdists] failed");
    KERNEL_CHECK_NULLPTR(inputs.size, KERNEL_STATUS_PARAM_INVALID, "Get input[3], name[size] failed");
    KERNEL_CHECK_NULLPTR(inputs.opflag, KERNEL_STATUS_PARAM_INVALID, "Get input[4], name[opflag] failed");
    KERNEL_CHECK_NULLPTR(inputs.attr, KERNEL_STATUS_PARAM_INVALID, "Get input[5], name[attr] failed");
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

uint32_t TopkMultisearchIvfV2CpuKernel::CheckInputShapes(const Inputs &inputs)
{
    KERNEL_LOG_INFO("TopkIvfCpuKernel CheckInputShapes begin");

    auto shapeIndistsV2 = inputs.indists->GetTensorShape();
    auto shapeVmdistsV2 = inputs.vmdists->GetTensorShape();
    auto shapeIdsV2 = inputs.ids->GetTensorShape();
    auto shapeSizeV2 = inputs.size->GetTensorShape();
    auto shapeOpflagV2 = inputs.opflag->GetTensorShape();
    auto shapeAttrV2 = inputs.attr->GetTensorShape();

    KERNEL_CHECK_TRUE(shapeIndistsV2->GetDims() == INPUT_NUM4,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][indists] must be 4");
    KERNEL_CHECK_TRUE(shapeVmdistsV2->GetDims() == INPUT_NUM4,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][vmdists] must be 4");
    KERNEL_CHECK_TRUE(shapeIdsV2->GetDims() == INPUT_NUM4,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][ids] must be 4");
    KERNEL_CHECK_TRUE(shapeSizeV2->GetDims() == INPUT_NUM4,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][size] must be 4");
    KERNEL_CHECK_TRUE(shapeOpflagV2->GetDims() == INPUT_NUM5,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][opflag] must be 5");
    KERNEL_CHECK_TRUE(shapeAttrV2->GetDims() == INPUT_NUM1,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][attr] must be 1");

    auto nq0 = shapeIndistsV2->GetDimSize(INPUT_NUM0);
    auto nq1 = shapeVmdistsV2->GetDimSize(INPUT_NUM0);
    KERNEL_CHECK_TRUE(nq0 == nq1, KERNEL_STATUS_PARAM_INVALID, "Nq of inputs must be same");

    auto handleBatch0 = shapeIndistsV2->GetDimSize(INPUT_NUM2);
    auto handleBatch1 = shapeVmdistsV2->GetDimSize(INPUT_NUM2);
    KERNEL_CHECK_TRUE(handleBatch0 == handleBatch1, KERNEL_STATUS_PARAM_INVALID, "Handle batch of inputs must be same");
    handleBatch_ = handleBatch0;

    flagSize_ = shapeOpflagV2->GetDimSize(INPUT_NUM4);
    nq_ = shapeOpflagV2->GetDimSize(INPUT_NUM0);
    auto attrCount = shapeAttrV2->GetDimSize(INPUT_NUM0);
    KERNEL_CHECK_TRUE(attrCount == TOPK_MULTISEARCH_IVF_V2_ATTR_IDX_COUNT,
        KERNEL_STATUS_PARAM_INVALID, "Num of attrs must be %d", TOPK_MULTISEARCH_IVF_V2_ATTR_IDX_COUNT);

    auto attr = static_cast<int64_t *>(inputs.attr->GetData());
    asc_ = *(attr + TOPK_MULTISEARCH_IVF_V2_ATTR_ASC_IDX);
    k_ = *(attr + TOPK_MULTISEARCH_IVF_V2_ATTR_K_IDX);
    burstLen_ = *(attr +TOPK_MULTISEARCH_IVF_V2_ATTR_BURST_LEN_IDX);
    blockNum_ = *(attr + TOPK_MULTISEARCH_IVF_V2_ATTR_BLOCK_NUM_IDX);
    flagNum_ = *(attr + TOPK_MULTISEARCH_IVF_V2_ATTR_FLAG_NUM_IDX);
    maxIndexNum_ = *(attr + TOPK_MULTISEARCH_IVF_V2_ATTR_MAX_INDEX_NUM_IDX);
    indexNum_ =  *(attr + TOPK_MULTISEARCH_IVF_V2_ATTR_INDEX_NUM_IDX);
    startqidx_ =  *(attr + TOPK_MULTISEARCH_IVF_V2_ATTR_Q_IDX);
    KERNEL_CHECK_TRUE(k_ > 0 && burstLen_ > 0 && asc_ >= 0 && blockNum_ > 0
        && flagNum_ > 0 && maxIndexNum_ > 0 && indexNum_ > 0,
        KERNEL_STATUS_PARAM_INVALID, "Value of asc, k, bustLen, blockNum, flagNum must ge 0");

    isIndexParall = true;
    return KERNEL_STATUS_OK;
}

void TopkMultisearchIvfV2CpuKernel::UpdateInOutShape(Inputs &inputs, Outputs &outputs) const
{
    KERNEL_LOG_INFO("TopkIvfCpuKernel UpdateInOutShape begin");

    auto shapeIndistsV2 = inputs.indists->GetTensorShape();
    std::vector<int64_t> dimIndistsV2 = shapeIndistsV2->GetDimSizes();
    dimIndistsV2[INPUT_NUM0] = maxIndexNum_;
    dimIndistsV2[INPUT_NUM1] = blockNum_;
    shapeIndistsV2->SetDimSizes(dimIndistsV2);

    auto shapeVmdistsV2 = inputs.vmdists->GetTensorShape();
    std::vector<int64_t> dimVmdistsV2 = shapeVmdistsV2->GetDimSizes();
    dimVmdistsV2[INPUT_NUM0] = maxIndexNum_;
    dimVmdistsV2[INPUT_NUM1] = blockNum_;
    shapeVmdistsV2->SetDimSizes(dimVmdistsV2);

    auto shapeIdsV2 = inputs.ids->GetTensorShape();
    std::vector<int64_t> dimIdsV2 = shapeIdsV2->GetDimSizes();
    dimIdsV2[INPUT_NUM1] = indexNum_;
    dimIdsV2[INPUT_NUM2] = blockNum_;
    shapeIdsV2->SetDimSizes(dimIdsV2);

    auto shapeSizeV2 = inputs.size->GetTensorShape();
    std::vector<int64_t> dimSizeV2 = shapeSizeV2->GetDimSizes();
    dimSizeV2[INPUT_NUM1] = indexNum_;
    dimSizeV2[INPUT_NUM2] = blockNum_;
    shapeSizeV2->SetDimSizes(dimSizeV2);

    auto shapeOpFlagV2 = inputs.opflag->GetTensorShape();
    std::vector<int64_t> dimOpFlagV2 = shapeOpFlagV2->GetDimSizes();
    dimOpFlagV2[INPUT_NUM1] = indexNum_;
    dimOpFlagV2[INPUT_NUM2] = blockNum_;
    shapeOpFlagV2->SetDimSizes(dimOpFlagV2);

    auto shapeOutdistsV2 = outputs.outdists->GetTensorShape();
    std::vector<int64_t> dimOutdistsV2;
    dimOutdistsV2.push_back(indexNum_);
    dimOutdistsV2.push_back(nq_);
    dimOutdistsV2.push_back(k_);
    shapeOutdistsV2->SetDimSizes(dimOutdistsV2);

    auto shapeOutlabelsV2 = outputs.outlabels->GetTensorShape();
    std::vector<int64_t> dimOutlabelsV2;
    dimOutlabelsV2.push_back(indexNum_);
    dimOutlabelsV2.push_back(nq_);
    dimOutlabelsV2.push_back(k_);
    shapeOutlabelsV2->SetDimSizes(dimOutlabelsV2);
}

void TopkMultisearchIvfV2CpuKernel::InitTopkHeap(Outputs &outputs) const
{
    if (startqidx_ == 0) {
        uint16_t *outdists = static_cast<uint16_t *>(outputs.outdists->GetData());
        int64_t *outlabels = static_cast<int64_t *>(outputs.outlabels->GetData());
        std::fill_n(outlabels, nq_ * indexNum_ * k_, 0xffffffffffffffff);
        if (asc_ != 0) {
            std::fill_n(outdists, nq_ * indexNum_ * k_, 0x7bff);
        } else {
            std::fill_n(outdists, nq_ * indexNum_ * k_, 0xfbff);
        }
    }
}

template <typename C>
void TopkMultisearchIvfV2CpuKernel::DoComputeParallForIndex(size_t tcnt, size_t tid,
    const Inputs &inputs, Outputs &outputs, C &&cmp)
{
    KernelTensor<float16_t> indists(inputs.indists);
    KernelTensor<float16_t> vmdists(inputs.vmdists);
    KernelTensor<int64_t> ids(inputs.ids);
    KernelTensor<uint32_t> size(inputs.size);
    KernelTensor<uint16_t> opflag(inputs.opflag);

    KernelTensor<float16_t> outdists(outputs.outdists);
    KernelTensor<int64_t> outlabels(outputs.outlabels);

    for (int64_t qidx = startqidx_; qidx < startqidx_ + 1; qidx++) {
        for (int64_t indexidx = tid; indexidx < indexNum_; indexidx += tcnt) {
            bool reorder = false;
            for (int64_t bidx = 0; bidx < blockNum_; ++bidx) {
                uint32_t count = 0;
                for (int64_t hidx = 0; hidx < handleBatch_; ++hidx) {
                    count += *(size.GetSubTensorDim3(qidx, indexidx, bidx, hidx));
                }
                if (count == 0) {
                    // no dists to be calculated in this block
                    continue;
                }
                auto flagPtr = opflag.GetSubTensorDim2(qidx, indexidx, bidx);
                for (int64_t i = 0; i < flagNum_; ++i) {
                    WAITING_FLAG_READY(*(flagPtr + i * flagSize_), TIMEOUT_CHECK_TICK, TIMEOUT_MS);
                }
                for (int64_t hidx = 0; hidx < handleBatch_; ++hidx) {
                    reorder = (bidx + 1 == blockNum_ && hidx + 1 == handleBatch_);
                    ComputeQuery(qidx, indexidx, bidx, hidx, indists, vmdists, ids, size,
                        outdists, outlabels, reorder, cmp);
                }
            }
            if (!reorder) {
                float16_t *outdists1 = outdists.GetSubTensorDim1(indexidx, qidx);
                int64_t *outlabel1 = outlabels.GetSubTensorDim1(indexidx, qidx);
                for (int64_t i = k_ - 1; i >= 1; --i) {
                    std::swap(outdists1[0], outdists1[i]);
                    std::swap(outlabel1[0], outlabel1[i]);
                    UpdateHeap(outdists1, outlabel1, i, 0, cmp);
                }
            }
        }
    }
}

template <typename C>
void TopkMultisearchIvfV2CpuKernel::DoCompute(size_t tcnt, size_t tid, const Inputs &inputs, Outputs &outputs, C &&cmp)
{
    KernelTensor<float16_t> indists(inputs.indists);
    KernelTensor<float16_t> vmdists(inputs.vmdists);
    KernelTensor<int64_t> ids(inputs.ids);
    KernelTensor<uint32_t> size(inputs.size);
    KernelTensor<uint16_t> opflag(inputs.opflag);

    KernelTensor<float16_t> outdists(outputs.outdists);
    KernelTensor<int64_t> outlabels(outputs.outlabels);

    for (int64_t qidx = tid; qidx < nq_; qidx += tcnt) {
        for (int64_t indexidx = 0; indexidx < indexNum_; indexidx++) {
            bool reorder = false;
            for (int64_t bidx = 0; bidx < blockNum_; ++bidx) {
                uint32_t count = 0;
                for (int64_t hidx = 0; hidx < handleBatch_; ++hidx) {
                    count += *(size.GetSubTensorDim3(qidx, indexidx, bidx, hidx));
                }
                if (count == 0) {
                    // no dists to be calculated in this block
                    continue;
                }
                auto flagPtr = opflag.GetSubTensorDim2(qidx, indexidx, bidx);
                for (int64_t i = 0; i < flagNum_; ++i) {
                    WAITING_FLAG_READY(*(flagPtr + i * flagSize_), TIMEOUT_CHECK_TICK, TIMEOUT_MS);
                }
                for (int64_t hidx = 0; hidx < handleBatch_; ++hidx) {
                    reorder = (bidx + 1 == blockNum_ && hidx + 1 == handleBatch_);
                    ComputeQuery(qidx, indexidx, bidx, hidx, indists, vmdists,
                        ids, size, outdists, outlabels, reorder, cmp);
                }
            }
            if (!reorder) {
                float16_t *outdists1 = outdists.GetSubTensorDim1(indexidx, qidx);
                int64_t *outlabel1 = outlabels.GetSubTensorDim1(indexidx, qidx);
                for (int64_t i = k_ - 1; i >= 1; --i) {
                    std::swap(outdists1[0], outdists1[i]);
                    std::swap(outlabel1[0], outlabel1[i]);
                    UpdateHeap(outdists1, outlabel1, i, 0, cmp);
                }
            }
        }
    }
}

template <typename C>
void TopkMultisearchIvfV2CpuKernel::ComputeQuery(int64_t qidx,
    int64_t indexidx,
    int64_t bidx,
    int64_t hidx,
    KernelTensor<float16_t> &indistsTensor,
    KernelTensor<float16_t> &vmdistsTensor,
    KernelTensor<int64_t> &idsTensor,
    KernelTensor<uint32_t> &sizeTensor,
    KernelTensor<float16_t> &outdistsTensor,
    KernelTensor<int64_t> &outlabelsTensor,
    bool reorder,
    C &&cmp)
{
    float16_t *indists = indistsTensor.GetSubTensorDim2(indexidx%maxIndexNum_, bidx, hidx);
    float16_t *vmdists = vmdistsTensor.GetSubTensorDim2(indexidx%maxIndexNum_, bidx, hidx);
    int64_t *ids = idsTensor.GetSubTensorDim3(qidx, indexidx, bidx, hidx);
    uint32_t *size = sizeTensor.GetSubTensorDim3(qidx, indexidx, bidx, hidx);
    float16_t *outdists = outdistsTensor.GetSubTensorDim1(indexidx, qidx);
    int64_t *outlabel = outlabelsTensor.GetSubTensorDim1(indexidx, qidx);

    int64_t ntotal = static_cast<int64_t>(*size);
    int64_t idxV2 = 0;
    int64_t *id = reinterpret_cast<int64_t *>(*ids);

    int64_t burstSize = ntotal / burstLen_;
    for (int64_t i = 0; i < burstSize; ++i) {
        if (!cmp(outdists[0], vmdists[i * 2])) { // vmdists[i*2] is dists, vmdists[i*2+1] is label
            // skip one burst
            idxV2 += burstLen_;
            continue;
        }
        for (int64_t j = 0; j < burstLen_ && idxV2 < ntotal; ++j, ++idxV2) {
            if (cmp(outdists[0], indists[idxV2])) {
                outdists[0] = indists[idxV2];
                outlabel[0] = *(id + idxV2);
                UpdateHeap(outdists, outlabel, k_, 0, cmp);
            }
        }
    }
    while (idxV2 < ntotal) {
        if (cmp(outdists[0], indists[idxV2])) {
            outdists[0] = indists[idxV2];
            outlabel[0] = *(id + idxV2);
            UpdateHeap(outdists, outlabel, k_, 0, cmp);
        }
        ++idxV2;
    }
    if (reorder) {
        for (int64_t m = k_ - 1; m >= 1; --m) {
            std::swap(outdists[0], outdists[m]);
            std::swap(outlabel[0], outlabel[m]);
            UpdateHeap(outdists, outlabel, m, 0, cmp);
        }
    }
}

REGISTER_CPU_KERNEL(TOPK_MULTISEARCH_IVF_V2, TopkMultisearchIvfV2CpuKernel);
} // namespace aicpu