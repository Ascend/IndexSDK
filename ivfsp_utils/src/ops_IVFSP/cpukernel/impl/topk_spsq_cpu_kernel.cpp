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


#include "topk_spsq_cpu_kernel.h"

#include <algorithm>
#include <string>
#include <map>
#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "kernel_tensor.h"
#include "kernel_utils.h"
#include "kernel_shared_def.h"

namespace {
const char *TOPK_SPSQ = "TopkSPSQ";
const uint32_t THREAD_CNT = 6;
}

namespace aicpu {
uint32_t TopkSpSqCpuKernel::Compute(CpuKernelContext &ctx)
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

    UpdateOutputsShape(outputs);

    if (pageIdx_ == 0) {
        if (labelType_ == DT_INT64) {
            InitTopkHeap<int64_t>(outputs);
        } else if (labelType_ == DT_UINT16) {
            InitTopkHeap<uint16_t>(outputs);
        } else {
            KERNEL_LOG_ERROR("Invalid datatype");
        }
    }

    auto funcLess = [](const float16_t a, const float16_t b) -> bool const { return a < b; };
    auto funcGreater = [](const float16_t a, const float16_t b) -> bool const { return a > b; };

    auto computeFunc = [&](size_t start, size_t end) {
        if (asc_ != 0) {
            // put greatest one to top of heap
            if (labelType_ == DT_INT64) {
                DoCompute<int64_t>(start, end, inputs, outputs, funcGreater);
            } else if (labelType_ == DT_UINT16) {
                DoCompute<uint16_t>(start, end, inputs, outputs, funcGreater);
            } else {
                KERNEL_LOG_ERROR("Invalid datatype");
            }
        } else {
            // put least one to top of heap
            if (labelType_ == DT_INT64) {
                DoCompute<int64_t>(start, end, inputs, outputs, funcLess);
            } else if (labelType_ == DT_UINT16) {
                DoCompute<uint16_t>(start, end, inputs, outputs, funcLess);
            } else {
                KERNEL_LOG_ERROR("Invalid datatype");
            }
        }
    };
#ifdef AICPU_UTEST
    computeFunc(0, nq_);
#else
    uint32_t core = std::min({ CpuKernelUtils::GetCPUNum(ctx), static_cast<uint32_t>(nq_), THREAD_CNT });
    CpuKernelUtils::ParallelFor(ctx, nq_, nq_ / core, computeFunc);
#endif

    return KERNEL_STATUS_OK;
}

uint32_t TopkSpSqCpuKernel::GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs) const
{
    KERNEL_LOG_INFO("TopkSpSqCpuKernel GetInOutAndCheck begin");

    inputs.indists = ctx.Input(INPUT_NUM0);
    inputs.vmdists = ctx.Input(INPUT_NUM1);
    inputs.size = ctx.Input(INPUT_NUM2);
    inputs.opflag = ctx.Input(INPUT_NUM3);
    inputs.attr = ctx.Input(INPUT_NUM4);
    inputs.offset = ctx.Input(INPUT_NUM5);
    outputs.outdists = ctx.Output(INPUT_NUM0);
    outputs.outlabels = ctx.Output(INPUT_NUM1);

    KERNEL_CHECK_NULLPTR(inputs.indists, KERNEL_STATUS_PARAM_INVALID, "Get input[0], name[indists] failed");
    KERNEL_CHECK_NULLPTR(inputs.vmdists, KERNEL_STATUS_PARAM_INVALID, "Get input[1], name[vmdists] failed");
    KERNEL_CHECK_NULLPTR(inputs.size, KERNEL_STATUS_PARAM_INVALID, "Get input[2], name[size] failed");
    KERNEL_CHECK_NULLPTR(inputs.opflag, KERNEL_STATUS_PARAM_INVALID, "Get input[3], name[opflag] failed");
    KERNEL_CHECK_NULLPTR(inputs.attr, KERNEL_STATUS_PARAM_INVALID, "Get input[4], name[attr] failed");
    KERNEL_CHECK_NULLPTR(inputs.offset, KERNEL_STATUS_PARAM_INVALID, "Get input[5], name[offset] failed");
    KERNEL_CHECK_NULLPTR(outputs.outdists, KERNEL_STATUS_PARAM_INVALID, "Get output[0], name[outdists] failed");
    KERNEL_CHECK_NULLPTR(outputs.outlabels, KERNEL_STATUS_PARAM_INVALID, "Get output[1], name[outlabels] failed");

    KERNEL_LOG_INFO("Shape of input[0][indists] is %s",
        ShapeToString(inputs.indists->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[1][vmdists] is %s",
        ShapeToString(inputs.vmdists->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[2][size] is %s",
        ShapeToString(inputs.size->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[3][opflag] is %s",
        ShapeToString(inputs.opflag->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[4][attr] is %s",
        ShapeToString(inputs.attr->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[5][offset] is %s",
                    ShapeToString(inputs.offset->GetTensorShape()->GetDimSizes()).c_str());

    return KERNEL_STATUS_OK;
}

uint32_t TopkSpSqCpuKernel::CheckInputShapes(const Inputs &inputs)
{
    KERNEL_LOG_INFO("TopkSpSqCpuKernel CheckInputShapes begin");

    auto shapeIndists = inputs.indists->GetTensorShape();
    auto shapeVmdists = inputs.vmdists->GetTensorShape();
    auto shapeSize = inputs.size->GetTensorShape();
    auto shapeOpflag = inputs.opflag->GetTensorShape();
    auto shapeAttr = inputs.attr->GetTensorShape();
    auto shapeOffset = inputs.offset->GetTensorShape();

    KERNEL_CHECK_TRUE(shapeIndists->GetDims() == INPUT_NUM3, KERNEL_STATUS_PARAM_INVALID,
        "Dims of input[0][indists] must be 3");
    KERNEL_CHECK_TRUE(shapeVmdists->GetDims() == INPUT_NUM3, KERNEL_STATUS_PARAM_INVALID,
        "Dims of input[0][vmdists] must be 3");
    KERNEL_CHECK_TRUE(shapeSize->GetDims() == INPUT_NUM3, KERNEL_STATUS_PARAM_INVALID,
        "Dims of input[0][size] must be 3");
    KERNEL_CHECK_TRUE(shapeOpflag->GetDims() == INPUT_NUM3, KERNEL_STATUS_PARAM_INVALID,
        "Dims of input[0][opflag] must be 3");
    KERNEL_CHECK_TRUE(shapeAttr->GetDims() == INPUT_NUM1, KERNEL_STATUS_PARAM_INVALID,
        "Dims of input[0][attr] must be 1");
    KERNEL_CHECK_TRUE(shapeOffset->GetDims() == INPUT_NUM1, KERNEL_STATUS_PARAM_INVALID,
                      "Dims of input[0][offset] must be 1");

    auto nq0 = shapeIndists->GetDimSize(INPUT_NUM1);
    auto nq1 = shapeVmdists->GetDimSize(INPUT_NUM1);
    KERNEL_CHECK_TRUE(nq0 == nq1, KERNEL_STATUS_PARAM_INVALID, "Nq of inputs must be same");
    nq_ = nq0;

    auto coreNum0 = shapeSize->GetDimSize(INPUT_NUM1);

    coreNum_ = coreNum0;

    flagSize_ = shapeOpflag->GetDimSize(INPUT_NUM2);

    auto attrCount = shapeAttr->GetDimSize(INPUT_NUM0);
    KERNEL_CHECK_TRUE(attrCount == TOPK_SPSQ_ATTR_IDX_COUNT, KERNEL_STATUS_PARAM_INVALID, "Num of attrs must be %d",
        TOPK_SPSQ_ATTR_IDX_COUNT);

    auto attr = static_cast<int64_t *>(inputs.attr->GetData());
    asc_ = *(attr + TOPK_SPSQ_ATTR_ASC_IDX);
    k_ = *(attr + TOPK_SPSQ_ATTR_K_IDX);
    burstLen_ = *(attr + TOPK_SPSQ_ATTR_BURST_LEN_IDX);
    blockNum_ = *(attr + TOPK_SPSQ_ATTR_BLOCK_NUM_IDX);
    pageIdx_ = *(attr + TOPK_SPSQ_ATTR_PAGE_IDX);
    pageNum_ = *(attr + TOPK_SPSQ_ATTR_PAGE_NUM_IDX);
    pageSize_ = *(attr + TOPK_SPSQ_ATTR_PAGE_SIZE_IDX);
    quickTopk_ = *(attr + TOPK_SPSQ_ATTR_QUICK_HEAP);
    blockSize_ = *(attr + TOPK_SPSQ_ATTR_BLOCK_SIZE);
    spBiNum_ = *(attr + TOPK_SPSQ_ATTR_BI_NUM_SIZE);
    spBlockNum_ = *(attr + TOPK_SPSQ_ATTR_BLOCK_NUM_SIZE);

    KERNEL_CHECK_TRUE(k_ > 0 && burstLen_ > 0 && asc_ >= 0 && blockNum_ > 0, KERNEL_STATUS_PARAM_INVALID,
        "Value of asc, k, bustLen, blockNum must ge 0");
    KERNEL_CHECK_TRUE(pageIdx_ >= 0 && pageNum_ > pageIdx_ && pageSize_ >= 0, KERNEL_STATUS_PARAM_INVALID,
        "Value of pageIdx, pageNum, pageSize is invalid");

    return KERNEL_STATUS_OK;
}

void TopkSpSqCpuKernel::UpdateOutputsShape(Outputs &outputs)
{
    KERNEL_LOG_INFO("TopkSpSqCpuKernel UpdateOutputsShape begin");

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

    labelType_ = outputs.outlabels->GetDataType();
}

template <typename T, typename C>
void TopkSpSqCpuKernel::DoCompute(size_t start, size_t end, const Inputs &inputs, Outputs &outputs, C &&cmp)
{
    KernelTensor<float16_t> indists(inputs.indists);
    KernelTensor<float16_t> vmdists(inputs.vmdists);
    KernelTensor<uint32_t> size(inputs.size);
    KernelTensor<uint16_t> opflag(inputs.opflag);
    KernelTensor<T> opoffset(inputs.offset);

    KernelTensor<float16_t> outdists(outputs.outdists);
    KernelTensor<T> outlabels(outputs.outlabels);

    for (int64_t i = 0; i < blockNum_; i++) {
        auto flagPtr = opflag.GetSubTensorDim0(i);
        for (int64_t j = 0; j < coreNum_; j++) {
            WAITING_FLAG_READY(*(flagPtr + j * flagSize_), TIMEOUT_CHECK_TICK, TIMEOUT_MS);
        }
        bool reorder = (pageIdx_ + 1 == pageNum_ && i + 1 == blockNum_); // reorder only last page and last block
        for (size_t j = start; j < end; j++) {
            ComputeBlock<T, C>(j, i, indists, vmdists, size, outdists, opoffset, outlabels, reorder, cmp);
        }
    }
}

template<typename T>
void TopkSpSqCpuKernel::InitTopkHeap(Outputs &outputs) const
{
    uint16_t *outdists = static_cast<uint16_t *>(outputs.outdists->GetData());
    T *outlabels = static_cast<T *>(outputs.outlabels->GetData());
    FillDefault(outdists, outlabels, asc_, nq_, k_);
}

template <typename T, typename C>
void TopkSpSqCpuKernel::ComputeBlock(size_t n, int64_t blockIdx, KernelTensor<float16_t> &indistsTensor,
    KernelTensor<float16_t> &vmdistsTensor, KernelTensor<uint32_t> &sizeTensor, KernelTensor<float16_t> &outdistsTensor,
    KernelTensor<T> &offsetTensor, KernelTensor<T> &outlabelsTensor, bool reorder, C &&cmp)
{
    float16_t *indists = indistsTensor.GetSubTensorDim1(blockIdx, n);
    float16_t *vmdists = vmdistsTensor.GetSubTensorDim1(blockIdx, n);
    uint16_t *vmlabel = reinterpret_cast<uint16_t *>(vmdists);
    uint32_t *size = sizeTensor.GetSubTensorDim0(blockIdx);
    float16_t *outdists = outdistsTensor.GetSubTensorDim0(n);
    T *outlabel = outlabelsTensor.GetSubTensorDim0(n);
    T *offset = offsetTensor.GetSubTensorDim0(0);

    int64_t ntotal = static_cast<int64_t>(*size);
    int64_t idx = 0;
    int64_t burstIdx = 0;
    int64_t burstSize = ntotal / burstLen_;

    int64_t pageOffset = 0;
    int offsetIdx = -1;
    if (pageIdx_ > 0) {
        offsetIdx = pageIdx_ * spBiNum_ * spBlockNum_ - 1;
        pageOffset = offset[offsetIdx];
    }
    int64_t totalBaseoffset = pageOffset;
    if (!(offsetIdx == -1 && blockIdx == 0)) {
        totalBaseoffset = offset[offsetIdx + blockIdx * spBiNum_];
    }

    if (!quickTopk_) {
        for (int64_t i = burstIdx; i < burstSize; ++i) {
            if (!cmp(outdists[0], vmdists[i * 2])) { // vmdists[i*2] is dists, vmdists[i*2+1] is label
                // skip one burst
                idx += burstLen_;
                continue;
            }
            for (int64_t j = 0; j < burstLen_ && idx < ntotal; ++j, ++idx) {
                if (cmp(outdists[0], indists[idx])) {
                    outdists[0] = indists[idx];
                    outlabel[0] = static_cast<T>(totalBaseoffset + idx);
                    UpdateHeap<T, C>(outdists, outlabel, k_, 0, cmp);
                }
            }
        }
    } else {
        // Stage one : update heap by vcmin/vcmax
        for (int64_t i = 0; i < burstSize; ++i) {
            if (!cmp(outdists[0], vmdists[i * 2])) { // vmdists[i*2] is dists, vmdists[i*2+1] is label
                continue;
            }
            // update heap by Vcmin/Vcmax, vmdists[i * 2] is dists
            outdists[0] = vmdists[i * 2];
            // vmlabel[i*2+1] is label
            outlabel[0] = static_cast<T>(burstLen_ * i + (vmlabel[i * 2 + 1]) + totalBaseoffset);
            UpdateHeap<T, C>(outdists, outlabel, k_, 0, cmp);
        }

        // Stage two:update heap by Topk burst
        if (blockIdx + 1 == blockNum_) {
            int64_t blockOffset = 0;
            int64_t currentIdx = 0;

            std::vector<std::pair<float16_t, int64_t>> topkBurstIdx;
            std::pair<int64_t, int64_t> currentPostion;

            for (int i = 0; i < k_; ++i) {
                currentIdx = static_cast<int64_t>(outlabel[i]) - pageOffset;
                if (currentIdx >= 0) {
                    topkBurstIdx.emplace_back(outdists[i], currentIdx);
                }
            }
            std::sort(topkBurstIdx.begin(), topkBurstIdx.end(),
                [&cmp](const std::pair<float16_t, int64_t> p1, const std::pair<float16_t, int64_t> p2) -> bool const {
                    return cmp(p2.first, p1.first);
                });

            // The first vaule is BlockIdx, the second vaule BurstIdx
            for (size_t i = 0; i < topkBurstIdx.size(); ++i) {
                // Idx in current block
                blockOffset = topkBurstIdx[i].second % blockSize_;
                // blockIdx in current page
                currentPostion.first = topkBurstIdx[i].second / blockSize_;
                // burst in current block
                currentPostion.second = blockOffset / burstLen_;

                float16_t *indistsSrc = indistsTensor.GetSubTensorDim1(currentPostion.first, n);

                if (!cmp(outdists[0], topkBurstIdx[i].first)) {
                    break;
                }
                // Reset the vcmin/vcmax to avoid update heap twice
                indistsSrc[blockOffset] = outdists[0];
                blockOffset = topkBurstIdx[i].second - blockOffset;
                currentIdx = currentPostion.second * burstLen_;
                for (int64_t j = currentIdx; j < currentIdx + burstLen_; ++j) {
                    if (cmp(outdists[0], indistsSrc[j])) {
                        outdists[0] = indistsSrc[j];
                        outlabel[0] = static_cast<T>(pageOffset + blockOffset + j);
                        UpdateHeap<T, C>(outdists, outlabel, k_, 0, cmp);
                    }
                }
            }
        }
        idx = burstSize * burstLen_;
    }
    // process tail data
    while (idx < ntotal) {
        if (cmp(outdists[0], indists[idx])) {
            outdists[0] = indists[idx];
            outlabel[0] = static_cast<T>(totalBaseoffset + idx);
            UpdateHeap<T, C>(outdists, outlabel, k_, 0, cmp);
        }
        ++idx;
    }
    // reorder dist result in last block
    if (reorder) {
        for (int64_t i = k_ - 1; i >= 1; --i) {
            std::swap(outdists[0], outdists[i]);
            std::swap(outlabel[0], outlabel[i]);
            UpdateHeap<T, C>(outdists, outlabel, i, 0, cmp);
        }
    }
}

REGISTER_CPU_KERNEL(TOPK_SPSQ, TopkSpSqCpuKernel);
} // namespace aicpu