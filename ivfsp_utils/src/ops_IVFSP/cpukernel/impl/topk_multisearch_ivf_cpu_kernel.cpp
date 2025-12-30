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


#include "topk_multisearch_ivf_cpu_kernel.h"

#include <iostream>
#include <algorithm>
#include <string>

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "kernel_tensor.h"
#include "kernel_utils.h"
#include "kernel_shared_def.h"

namespace {
const char* TOPK_MULTISEARCH_IVF = "TopkMultisearchIvf";
const uint32_t THREAD_CNT = 6;
}

namespace aicpu {
uint32_t TopkMultisearchIvfCpuKernel::Compute(CpuKernelContext &ctx)
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

    auto funcLess = [](float16_t a, float16_t b) -> bool { return a < b; };
    auto funcGreater = [](float16_t a, float16_t b) -> bool { return a > b; };

#ifdef AICPU_UTEST
    uint32_t core = 1;
#else
    uint32_t core = std::min({CpuKernelUtils::GetCPUNum(ctx), static_cast<uint32_t>(nq_), THREAD_CNT});
    if (isIndexParall) {
        core = std::min({CpuKernelUtils::GetCPUNum(ctx), static_cast<uint32_t>(indexNum_), THREAD_CNT});
    }
#endif

    auto computeFunc = [&](size_t start, size_t end) {
        (void)end; // end is unuseful in this function
        if (asc_ != 0) {
            // put greatest one to top of heap
            if (isIndexParall) {
                DoComputeParallForIndex(core, start, inputs, outputs, funcGreater);
            } else {
                DoCompute(core, start, inputs, outputs, funcGreater);
            }
        } else {
            // put least one to top of heap
            if (isIndexParall) {
                DoComputeParallForIndex(core, start, inputs, outputs, funcLess);
            } else {
                DoCompute(core, start, inputs, outputs, funcLess);
            }
        }
    };

#ifdef AICPU_UTEST
    computeFunc(0, 1);
#else
    CpuKernelUtils::ParallelFor(ctx, core, 1, computeFunc);
#endif

    return KERNEL_STATUS_OK;
}

uint32_t TopkMultisearchIvfCpuKernel::GetInOutAndCheck(const CpuKernelContext &ctx,
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
    KERNEL_CHECK_NULLPTR(inputs.ids, KERNEL_STATUS_PARAM_INVALID, "Get input[1], name[vmdists] failed");
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

uint32_t TopkMultisearchIvfCpuKernel::CheckInputShapes(const Inputs &inputs)
{
    KERNEL_LOG_INFO("TopkIvfCpuKernel CheckInputShapes begin");

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
    KERNEL_CHECK_TRUE(shapeIds->GetDims() == INPUT_NUM4,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][ids] must be 3");
    KERNEL_CHECK_TRUE(shapeSize->GetDims() == INPUT_NUM4,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][size] must be 3");
    KERNEL_CHECK_TRUE(shapeOpflag->GetDims() == INPUT_NUM5,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][opflag] must be 4");
    KERNEL_CHECK_TRUE(shapeAttr->GetDims() == INPUT_NUM1,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][attr] must be 1");

    auto nq0 = shapeIndists->GetDimSize(INPUT_NUM0);
    auto nq1 = shapeVmdists->GetDimSize(INPUT_NUM0);
    KERNEL_CHECK_TRUE(nq0 == nq1, KERNEL_STATUS_PARAM_INVALID, "Nq of inputs must be same");

    auto handleBatch0 = shapeIndists->GetDimSize(INPUT_NUM2);
    auto handleBatch1 = shapeVmdists->GetDimSize(INPUT_NUM2);
    KERNEL_CHECK_TRUE(handleBatch0 == handleBatch1, KERNEL_STATUS_PARAM_INVALID, "Handle batch of inputs must be same");
    handleBatch_ = handleBatch0;

    flagSize_ = shapeOpflag->GetDimSize(INPUT_NUM4);
    nq_ = shapeOpflag->GetDimSize(INPUT_NUM0);
    auto attrCount = shapeAttr->GetDimSize(INPUT_NUM0);
    KERNEL_CHECK_TRUE(attrCount == TOPK_MULTISEARCH_IVF_ATTR_IDX_COUNT,
        KERNEL_STATUS_PARAM_INVALID, "Num of attrs must be %d", TOPK_MULTISEARCH_IVF_ATTR_IDX_COUNT);

    auto attr = static_cast<int64_t *>(inputs.attr->GetData());
    asc_ = *(attr + TOPK_MULTISEARCH_IVF_ATTR_ASC_IDX);
    k_ = *(attr + TOPK_MULTISEARCH_IVF_ATTR_K_IDX);
    burstLen_ = *(attr + TOPK_MULTISEARCH_IVF_ATTR_BURST_LEN_IDX);
    blockNum_ = *(attr + TOPK_MULTISEARCH_IVF_ATTR_BLOCK_NUM_IDX);
    flagNum_ = *(attr + TOPK_MULTISEARCH_IVF_ATTR_FLAG_NUM_IDX);
    maxIndexNum_ = *(attr + TOPK_MULTISEARCH_IVF_ATTR_MAX_INDEX_NUM_IDX);
    indexNum_ =  *(attr + TOPK_MULTISEARCH_IVF_ATTR_INDEX_NUM_IDX);
    KERNEL_CHECK_TRUE(k_ > 0 && burstLen_ > 0 && asc_ >= 0 && blockNum_ > 0 &&
        flagNum_ > 0 && maxIndexNum_ > 0 && indexNum_ > 0,
        KERNEL_STATUS_PARAM_INVALID, "Value of asc, k, bustLen, blockNum, flagNum must ge 0");

    isIndexParall = true;
    return KERNEL_STATUS_OK;
}

void TopkMultisearchIvfCpuKernel::UpdateInOutShape(Inputs &inputs, Outputs &outputs) const
{
    KERNEL_LOG_INFO("TopkIvfCpuKernel UpdateInOutShape begin");

    auto shapeIndists = inputs.indists->GetTensorShape();
    std::vector<int64_t> dimIndists = shapeIndists->GetDimSizes();
    dimIndists[INPUT_NUM0] = maxIndexNum_;
    dimIndists[INPUT_NUM1] = blockNum_;
    shapeIndists->SetDimSizes(dimIndists);

    auto shapeVmdists = inputs.vmdists->GetTensorShape();
    std::vector<int64_t> dimVmdists = shapeVmdists->GetDimSizes();
    dimVmdists[INPUT_NUM0] = maxIndexNum_;
    dimVmdists[INPUT_NUM1] = blockNum_;
    shapeVmdists->SetDimSizes(dimVmdists);

    auto shapeIds = inputs.ids->GetTensorShape();
    std::vector<int64_t> dimIds = shapeIds->GetDimSizes();
    dimIds[INPUT_NUM1] = indexNum_;
    dimIds[INPUT_NUM2] = blockNum_;
    shapeIds->SetDimSizes(dimIds);

    auto shapeSize = inputs.size->GetTensorShape();
    std::vector<int64_t> dimSize = shapeSize->GetDimSizes();
    dimSize[INPUT_NUM1] = indexNum_;
    dimSize[INPUT_NUM2] = blockNum_;
    shapeSize->SetDimSizes(dimSize);

    auto shapeOpFlag = inputs.opflag->GetTensorShape();
    std::vector<int64_t> dimOpFlag = shapeOpFlag->GetDimSizes();
    dimOpFlag[INPUT_NUM1] = indexNum_;
    dimOpFlag[INPUT_NUM2] = blockNum_;
    shapeOpFlag->SetDimSizes(dimOpFlag);

    auto shapeOutdists = outputs.outdists->GetTensorShape();
    std::vector<int64_t> dimOutdists;
    dimOutdists.push_back(indexNum_);
    dimOutdists.push_back(nq_);
    dimOutdists.push_back(k_);
    shapeOutdists->SetDimSizes(dimOutdists);

    auto shapeOutlabels = outputs.outlabels->GetTensorShape();
    std::vector<int64_t> dimOutlabels;
    dimOutlabels.push_back(indexNum_);
    dimOutlabels.push_back(nq_);
    dimOutlabels.push_back(k_);
    shapeOutlabels->SetDimSizes(dimOutlabels);
}

void TopkMultisearchIvfCpuKernel::InitTopkHeap(Outputs &outputs) const
{
    uint16_t *outdists = static_cast<uint16_t *>(outputs.outdists->GetData());
    int64_t *outlabels = static_cast<int64_t *>(outputs.outlabels->GetData());
    std::fill_n(outlabels, nq_ * indexNum_ * k_, 0xffffffffffffffff);
    if (asc_ != 0) {
        std::fill_n(outdists, nq_ * indexNum_ * k_, 0x7bff);
    } else {
        std::fill_n(outdists, nq_ * indexNum_ * k_, 0xfbff);
    }
}

template <typename C>
void TopkMultisearchIvfCpuKernel::DoComputeParallForIndex(size_t tcnt, size_t tid,
    const Inputs &inputs, Outputs &outputs, C &&cmp)
{
    KernelTensor<float16_t> indists(inputs.indists);
    KernelTensor<float16_t> vmdists(inputs.vmdists);
    KernelTensor<int64_t> ids(inputs.ids);
    KernelTensor<uint32_t> size(inputs.size);
    KernelTensor<uint16_t> opflag(inputs.opflag);

    KernelTensor<float16_t> outdists(outputs.outdists);
    KernelTensor<int64_t> outlabels(outputs.outlabels);
    for (int64_t qidx = 0; qidx < nq_; qidx++) {
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
void TopkMultisearchIvfCpuKernel::DoCompute(size_t tcnt, size_t tid, const Inputs &inputs, Outputs &outputs, C &&cmp)
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
void TopkMultisearchIvfCpuKernel::ComputeQuery(int64_t qidx,
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
    int64_t idxMulti = 0;
    int64_t *id = reinterpret_cast<int64_t *>(*ids);

    int64_t burstSize = ntotal / burstLen_;
    for (int64_t i = 0; i < burstSize; ++i) {
        if (!cmp(outdists[0], vmdists[i * 2])) { // vmdists[i*2] is dists, vmdists[i*2+1] is label
            // skip one burst
            idxMulti += burstLen_;
            continue;
        }
        for (int64_t j = 0; j < burstLen_ && idxMulti < ntotal; ++j, ++idxMulti) {
            if (cmp(outdists[0], indists[idxMulti])) {
                outdists[0] = indists[idxMulti];
                outlabel[0] = *(id + idxMulti);
                UpdateHeap(outdists, outlabel, k_, 0, cmp);
            }
        }
    }
    while (idxMulti < ntotal) {
        if (cmp(outdists[0], indists[idxMulti])) {
            outdists[0] = indists[idxMulti];
            outlabel[0] = *(id + idxMulti);
            UpdateHeap(outdists, outlabel, k_, 0, cmp);
        }
        ++idxMulti;
    }
    if (reorder) {
        for (int64_t j = k_ - 1; j >= 1; --j) {
            std::swap(outdists[0], outdists[j]);
            std::swap(outlabel[0], outlabel[j]);
            UpdateHeap(outdists, outlabel, j, 0, cmp);
        }
    }
}

REGISTER_CPU_KERNEL(TOPK_MULTISEARCH_IVF, TopkMultisearchIvfCpuKernel);
} // namespace aicpu