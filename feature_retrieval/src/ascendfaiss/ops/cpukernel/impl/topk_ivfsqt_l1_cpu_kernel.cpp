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


#include "topk_ivfsqt_l1_cpu_kernel.h"

#include <algorithm>
#include <cmath>

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "kernel_tensor.h"
#include "kernel_utils.h"
#include "kernel_shared_def.h"

namespace {
const char* TOPK_IVFSQT_L1 = "TopkIvfsqtL1";
const uint16_t MAX_UINT16 = 0xffff;
const float epsilon = 1e-6;
}

namespace aicpu {
uint32_t TopkIvfsqtL1CpuKernel::Compute(CpuKernelContext &ctx)
{
    Inputs inputs;
    Outputs outputs;
    auto ret = GetInOutAndCheck(ctx, inputs, outputs);
    if (ret != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("Failed to get inputs or outputs");
        return ret;
    }

    ret = CheckInOutShapes(inputs, outputs);
    if (ret != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("Failed to check input shapes");
        return ret;
    }

    UpdateInOutShape(outputs);

    InitTopkHeap(outputs);

    auto funcLess = [](float16_t a, float16_t b) -> bool { return a < b; };
    auto funcGreater = [](float16_t a, float16_t b) -> bool { return a > b; };

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

uint32_t TopkIvfsqtL1CpuKernel::GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs) const
{
    KERNEL_LOG_INFO("TopkIvfsqtL1CpuKernel GetInOutAndCheck begin");

    inputs.indists = ctx.Input(INPUT_NUM0);
    inputs.vmdists = ctx.Input(INPUT_NUM1);
    inputs.opflag = ctx.Input(INPUT_NUM2);
    inputs.attr = ctx.Input(INPUT_NUM3);
    inputs.queryIn = ctx.Input(INPUT_NUM4);
    inputs.compressIndex = ctx.Input(INPUT_NUM5);
    inputs.compressValue = ctx.Input(INPUT_NUM6);

    outputs.outdists = ctx.Output(INPUT_NUM0);
    outputs.outlabels = ctx.Output(INPUT_NUM1);
    outputs.queryOut = ctx.Output(INPUT_NUM2);

    KERNEL_CHECK_NULLPTR(inputs.indists, KERNEL_STATUS_PARAM_INVALID, "Get input[0], name[indists] failed");
    KERNEL_CHECK_NULLPTR(inputs.vmdists, KERNEL_STATUS_PARAM_INVALID, "Get input[1], name[vmdists] failed");
    KERNEL_CHECK_NULLPTR(inputs.opflag, KERNEL_STATUS_PARAM_INVALID, "Get input[2], name[opflag] failed");
    KERNEL_CHECK_NULLPTR(inputs.attr, KERNEL_STATUS_PARAM_INVALID, "Get input[3], name[attr] failed");
    KERNEL_CHECK_NULLPTR(inputs.queryIn, KERNEL_STATUS_PARAM_INVALID, "Get input[4], name[queryIn] failed");
    KERNEL_CHECK_NULLPTR(inputs.compressIndex,
        KERNEL_STATUS_PARAM_INVALID, "Get input[5], name[compressIndex] failed");
    KERNEL_CHECK_NULLPTR(inputs.compressValue,
        KERNEL_STATUS_PARAM_INVALID, "Get input[6], name[compressValue] failed");

    KERNEL_CHECK_NULLPTR(outputs.outdists, KERNEL_STATUS_PARAM_INVALID, "Get output[0], name[outdists] failed");
    KERNEL_CHECK_NULLPTR(outputs.outlabels, KERNEL_STATUS_PARAM_INVALID, "Get output[1], name[outlabels] failed");
    KERNEL_CHECK_NULLPTR(outputs.queryOut, KERNEL_STATUS_PARAM_INVALID, "Get output[2], name[queryOut] failed");

    KERNEL_LOG_INFO("Shape of input[0][indists] is %s",
        ShapeToString(inputs.indists->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[1][vmdists] is %s",
        ShapeToString(inputs.vmdists->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[2][opflag] is %s",
        ShapeToString(inputs.opflag->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[3][attr] is %s",
        ShapeToString(inputs.attr->GetTensorShape()->GetDimSizes()).c_str());

    return KERNEL_STATUS_OK;
}

uint32_t TopkIvfsqtL1CpuKernel::CheckInOutShapes(const Inputs &inputs, const Outputs &outputs)
{
    KERNEL_LOG_INFO("TopkIvfsqtL1CpuKernel CheckInOutShapes begin");

    auto shapeIndists = inputs.indists->GetTensorShape();
    auto shapeVmdists = inputs.vmdists->GetTensorShape();
    auto shapeOpflag = inputs.opflag->GetTensorShape();
    auto shapeAttr = inputs.attr->GetTensorShape();
    auto shapeQueryIn = inputs.queryIn->GetTensorShape(); // nq, dimIn
    auto shapeCompressIndex = inputs.compressIndex->GetTensorShape(); // dimOut, ratio
    auto shapeCompressValue = inputs.compressValue->GetTensorShape(); // ratio, dimOut

    auto shapeOutdists = outputs.outdists->GetTensorShape();
    auto shapeOutlabels = outputs.outlabels->GetTensorShape();
    auto shapeQueryOut = outputs.queryOut->GetTensorShape(); // nq, nimOut

    KERNEL_CHECK_TRUE(shapeIndists->GetDims() == INPUT_NUM2,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][indists] must be 2");
    KERNEL_CHECK_TRUE(shapeVmdists->GetDims() == INPUT_NUM2,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[1][vmdists] must be 2");
    KERNEL_CHECK_TRUE(shapeOpflag->GetDims() == INPUT_NUM3,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[2][opflag] must be 3");
    KERNEL_CHECK_TRUE(shapeAttr->GetDims() == INPUT_NUM1,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[3][attr] must be 1");
    KERNEL_CHECK_TRUE(shapeQueryIn->GetDims() == INPUT_NUM2,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[4][queryIn] must be 2");
    KERNEL_CHECK_TRUE(shapeCompressIndex->GetDims() == INPUT_NUM2,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[5][compressIndex] must be 2");
    KERNEL_CHECK_TRUE(shapeCompressValue->GetDims() == INPUT_NUM2,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[6][compressValue] must be 2");

    KERNEL_CHECK_TRUE(shapeOutdists->GetDims() == INPUT_NUM2,
        KERNEL_STATUS_PARAM_INVALID, "Dims of output[0][shapeOutdists] must be 2");
    KERNEL_CHECK_TRUE(shapeOutlabels->GetDims() == INPUT_NUM2,
        KERNEL_STATUS_PARAM_INVALID, "Dims of output[1][shapeOutlabels] must be 2");
    KERNEL_CHECK_TRUE(shapeQueryOut->GetDims() == INPUT_NUM2,
        KERNEL_STATUS_PARAM_INVALID, "Dims of output[2][queryOut] must be 2");

    auto nq0 = shapeIndists->GetDimSize(INPUT_NUM0);
    auto nq1 = shapeVmdists->GetDimSize(INPUT_NUM0);
    auto nq2 = shapeOpflag->GetDimSize(INPUT_NUM0);
    auto nq3 = shapeQueryIn->GetDimSize(INPUT_NUM0);
    auto nq4 = shapeQueryOut->GetDimSize(INPUT_NUM0);
    KERNEL_CHECK_TRUE(nq0 == nq1 && nq0 == nq2 && nq0 == nq3 && nq0 == nq4,
        KERNEL_STATUS_PARAM_INVALID, "Nq of inputs must be same");
    nq_ = nq0;

    auto dimOut0 = shapeCompressIndex->GetDimSize(INPUT_NUM0);
    auto dimOut1 = shapeCompressValue->GetDimSize(INPUT_NUM1);
    auto dimOut2 = shapeQueryOut->GetDimSize(INPUT_NUM1);
    KERNEL_CHECK_TRUE(dimOut0 == dimOut1 && dimOut0 == dimOut2,
        KERNEL_STATUS_PARAM_INVALID, "Dim of inputs and outputs must be same");
    dimOut_ = dimOut0;

    ratio_ = shapeCompressIndex->GetDimSize(INPUT_NUM1);
    dimIn_ = shapeQueryIn->GetDimSize(INPUT_NUM1);
    KERNEL_CHECK_TRUE(dimOut_ * ratio_ == dimIn_,
        KERNEL_STATUS_PARAM_INVALID, "Ratio * DimOut must equal with DimIn");

    flagNum_ = shapeOpflag->GetDimSize(INPUT_NUM1);
    flagSize_ = shapeOpflag->GetDimSize(INPUT_NUM2);

    auto attrCount = shapeAttr->GetDimSize(INPUT_NUM0);
    KERNEL_CHECK_TRUE(attrCount == TOPK_IVFSQT_L1_ATTR_IDX_COUNT,
        KERNEL_STATUS_PARAM_INVALID, "Num of attrs must be %d", TOPK_IVFSQT_L1_ATTR_IDX_COUNT);

    auto attr = static_cast<int64_t *>(inputs.attr->GetData());
    asc_ = *(attr + TOPK_IVFSQT_L1_ATTR_ASC_IDX);
    k_ = *(attr + TOPK_IVFSQT_L1_ATTR_K_IDX);
    burstLen_ = *(attr + TOPK_IVFSQT_L1_ATTR_BURST_LEN_IDX);
    opSize_ = *(attr + TOPK_IVFSQT_L1_ATTR_OP_SIZE_IDX);
    queryBatchSize_ = *(attr + TOPK_IVFSQT_L1_ATTR_Q_BATCH_SIZE_IDX);
    quickTopk_ = *(attr + TOPK_IVFSQT_L1_ATTR_QUICK_HEAP);

    KERNEL_CHECK_TRUE(k_ > 0 && burstLen_ > 0 && queryBatchSize_ > 0 && opSize_ > 0,
        KERNEL_STATUS_PARAM_INVALID, "Value of attrs must greater than 0");

    return KERNEL_STATUS_OK;
}

void TopkIvfsqtL1CpuKernel::UpdateInOutShape(Outputs &outputs) const
{
    KERNEL_LOG_INFO("TopkIvfsqtL1CpuKernel UpdateInOutShape begin");

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

void TopkIvfsqtL1CpuKernel::InitTopkHeap(const Outputs &outputs) const
{
    KERNEL_LOG_INFO("TopkIvfsqtL1CpuKernel InitTopkHeap begin");
    uint16_t *outdists = static_cast<uint16_t *>(outputs.outdists->GetData());
    uint16_t *outlabels = static_cast<uint16_t *>(outputs.outlabels->GetData());
    std::fill_n(outlabels, nq_ * k_, MAX_UINT16);
    if (asc_ != 0) {
        std::fill_n(outdists, nq_ * k_, 0x7bff);
    } else {
        std::fill_n(outdists, nq_ * k_, 0xfbff);
    }
}

template <typename C>
void TopkIvfsqtL1CpuKernel::DoCompute(size_t tcnt, size_t tid, const Inputs &inputs, const Outputs &outputs, C &&cmp)
{
    KERNEL_LOG_INFO("TopkIvfsqtL1CpuKernel DoCompute begin");
    KernelTensor<float16_t> indists(inputs.indists);
    KernelTensor<float16_t> vmdists(inputs.vmdists);
    KernelTensor<uint16_t> opflag(inputs.opflag);

    KernelTensor<float16_t> outdists(outputs.outdists);
    KernelTensor<uint16_t> outlabels(outputs.outlabels);

    KernelTensor<float16_t> queryIn(inputs.queryIn);
    KernelTensor<int> compressIndex(inputs.compressIndex);
    KernelTensor<float> compressValue(inputs.compressValue);
    KernelTensor<float16_t> queryOut(outputs.queryOut);

    for (int64_t qidx = tid; qidx < nq_; qidx += tcnt) {
        int64_t flagIdx = qidx / queryBatchSize_ * queryBatchSize_;
        auto flagPtr = opflag.GetSubTensorDim0(flagIdx);
        for (int64_t i = 0; i < flagNum_; ++i) {
            WAITING_FLAG_READY(*(flagPtr + i * flagSize_), TIMEOUT_CHECK_TICK, TIMEOUT_MS);
        }
        ComputeQueryBatch(qidx, indists, vmdists, outdists, outlabels,
            queryIn, compressIndex, compressValue, queryOut, cmp);
    }
}

void TopkIvfsqtL1CpuKernel::CompressData(int64_t qidx, KernelTensor<float16_t> &queryInTensor,
    KernelTensor<float16_t> &queryOutTensor, KernelTensor<int> &compressIndexTensor,
    KernelTensor<float> &compressValueTensor)
{
    // compress
    const float16_t *xSlice = queryInTensor.GetSubTensorDim0(qidx);
    float16_t *resSlice = queryOutTensor.GetSubTensorDim0(qidx);
    std::fill_n(resSlice, dimOut_, 0);
    float denominator = 0;
    for (int didx = 0; didx < dimOut_; ++didx) {
        for (int ridx = 0; ridx < ratio_; ++ridx) {
            int dimInIndex = *(compressIndexTensor.GetSubTensorDim1(didx, ridx));
            float value = *(compressValueTensor.GetSubTensorDim1(ridx, didx));
            resSlice[didx] += *(xSlice + dimInIndex) * (1 - value);
        }
        resSlice[didx] /= ratio_;
        denominator += (resSlice[didx] * resSlice[didx]);
    }
    if (denominator > epsilon) {
        const float invNr = 1.0 / sqrt(denominator);
        for (int didx = 0; didx < dimOut_; didx++) {
            resSlice[didx] *= invNr;
        };
    }
}

template <typename C>
void TopkIvfsqtL1CpuKernel::ComputeQueryBatch(int64_t qidx,
                                              KernelTensor<float16_t> &indistsTensor,
                                              KernelTensor<float16_t> &vmdistsTensor,
                                              KernelTensor<float16_t> &outdistsTensor,
                                              KernelTensor<uint16_t> &outlabelsTensor,
                                              KernelTensor<float16_t> &queryInTensor,
                                              KernelTensor<int> &compressIndexTensor,
                                              KernelTensor<float> &compressValueTensor,
                                              KernelTensor<float16_t> &queryOutTensor,
                                              C &&cmp)
{
    KERNEL_LOG_INFO("TopkIvfsqtL1CpuKernel DoCompute begin");
    float16_t *indists = indistsTensor.GetSubTensorDim0(qidx);
    float16_t *vmdists = vmdistsTensor.GetSubTensorDim0(qidx);
    uint16_t *vmlabel = reinterpret_cast<uint16_t *>(vmdists);
    float16_t *outdists = outdistsTensor.GetSubTensorDim0(qidx);
    uint16_t *outlabels = outlabelsTensor.GetSubTensorDim0(qidx);

    int64_t ntotal = opSize_;
    int64_t idx = 0;
    int64_t burstIdx = 0;
    int64_t burstSize = ntotal / burstLen_;

    if (quickTopk_ == 0) {
        for (int64_t i = burstIdx; i < burstSize; ++i) {
            if (!cmp(outdists[0], vmdists[i * 2])) { // vmdists[i*2] is dists, vmdists[i*2+1] is label
                // skip one burst
                idx += burstLen_;
                continue;
            }
            for (int64_t j = 0; j < burstLen_ && idx < ntotal; ++j, ++idx) {
                if (cmp(outdists[0], indists[idx])) {
                    outdists[0] = indists[idx];
                    outlabels[0] = static_cast<uint16_t>(idx);
                    UpdateHeap(outdists, outlabels, k_, 0, cmp);
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
            outlabels[0] = static_cast<uint16_t>(burstLen_ * i + (vmlabel[i * 2 + 1]));
            UpdateHeap(outdists, outlabels, k_, 0, cmp);
            // Reset current dist, vmlabel[i*2+1] is label
            indists[burstLen_ * i + (vmlabel[i * 2 + 1])] = outdists[0];
        }

        uint16_t pageOffset = 0;
        uint16_t blockOffset = 0;
        uint16_t currentIdx = 0;

        std::vector<std::pair<float16_t, uint16_t>> topkBurstIdx;
        std::pair<uint16_t, uint16_t> currentPostion;

        for (int i = 0; i < k_; ++i) {
            currentIdx = outlabels[i] - pageOffset;
            if (currentIdx >= 0 && currentIdx < MAX_UINT16) {
                topkBurstIdx.emplace_back(outdists[i], currentIdx);
            }
        }
        std::sort(topkBurstIdx.begin(), topkBurstIdx.end(),
            [&](const std::pair<float16_t, uint16_t> p1, const std::pair<float16_t, uint16_t> p2) -> bool {
                return cmp(p2.first, p1.first);
            });

        // The first vaule is BlockIdx, the second vaule BurstIdx
        for (size_t i = 0; i < topkBurstIdx.size(); ++i) {
            blockOffset = topkBurstIdx[i].second;
            currentPostion.first = topkBurstIdx[i].second;
            currentPostion.second = blockOffset / burstLen_;

            if (!cmp(outdists[0], topkBurstIdx[i].first)) {
                break;
            }
            blockOffset = topkBurstIdx[i].second - blockOffset;
            currentIdx = currentPostion.second * burstLen_;
            for (int64_t j = currentIdx; j < currentIdx + burstLen_; ++j) {
                if (cmp(outdists[0], indists[j])) {
                    outdists[0] = indists[j];
                    outlabels[0] = static_cast<uint16_t>(pageOffset + blockOffset + j);
                    UpdateHeap(outdists, outlabels, k_, 0, cmp);
                }
            }
        }
        idx = burstSize * burstLen_;
    }

    // process tail data
    while (idx < ntotal) {
        if (cmp(outdists[0], indists[idx])) {
            outdists[0] = indists[idx];
            outlabels[0] = static_cast<uint16_t>(idx);
            UpdateHeap(outdists, outlabels, k_, 0, cmp);
        }
        ++idx;
    }

    CompressData(qidx, queryInTensor, queryOutTensor, compressIndexTensor, compressValueTensor);
}

REGISTER_CPU_KERNEL(TOPK_IVFSQT_L1, TopkIvfsqtL1CpuKernel);
} // namespace aicpu