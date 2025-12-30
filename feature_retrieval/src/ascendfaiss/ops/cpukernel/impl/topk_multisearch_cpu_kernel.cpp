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


#include "topk_multisearch_cpu_kernel.h"

#include <algorithm>
#include <string>
#include <map>
#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "kernel_tensor.h"
#include "kernel_utils.h"
#include "kernel_shared_def.h"

namespace {
const char *TOPK_MULTISEARCH = "TopkMultisearch";
const uint32_t THREAD_CNT = 6;
}

namespace aicpu {
uint32_t TopkMultisearchCpuKernel::Compute(CpuKernelContext &ctx)
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

    auto funcLess = [](const float16_t a, const float16_t b) -> bool { return a < b; };
    auto funcGreater = [](const float16_t a, const float16_t b) -> bool { return a > b; };

    auto computeParallForBatch = [&inputs, &outputs, &funcGreater, &funcLess, this](size_t start, size_t end) {
        if (asc_ != 0) {
            // put greatest one to top of heap
            DoComputeParallForBatch(start, end, inputs, outputs, funcGreater);
        } else {
            // put least one to top of heap
            DoComputeParallForBatch(start, end, inputs, outputs, funcLess);
        }
    };

    uint32_t coreParallForBlcok =
        std::min({ CpuKernelUtils::GetCPUNum(ctx), static_cast<uint32_t>(pageBlockNum_), THREAD_CNT });
    auto computeParallForBlcok = [&inputs, &outputs, &funcGreater, &funcLess, coreParallForBlcok, this]
        (size_t start, size_t end) {
        (void)end; // end is unuseful in this function
        if (asc_ != 0) {
            // put greatest one to top of heap
            DoComputeParallForBlcok(coreParallForBlcok, start, inputs, outputs, funcGreater);
        } else {
            // put least one to top of heap
            DoComputeParallForBlcok(coreParallForBlcok, start, inputs, outputs, funcLess);
        }
    };

    if (!isBlockParall) {
        uint32_t core = std::min({ CpuKernelUtils::GetCPUNum(ctx), static_cast<uint32_t>(nq_), THREAD_CNT });
        CpuKernelUtils::ParallelFor(ctx, nq_, nq_ / core, computeParallForBatch);
    } else {
        CpuKernelUtils::ParallelFor(ctx, coreParallForBlcok, 1, computeParallForBlcok);
    }

    return KERNEL_STATUS_OK;
}

uint32_t TopkMultisearchCpuKernel::GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs) const
{
    KERNEL_LOG_INFO("TopkMultisearchCpuKernel GetInOutAndCheck begin");

    inputs.indists = ctx.Input(INPUT_NUM0);
    inputs.vmdists = ctx.Input(INPUT_NUM1);
    inputs.size = ctx.Input(INPUT_NUM2);
    inputs.opflag = ctx.Input(INPUT_NUM3);
    inputs.attr = ctx.Input(INPUT_NUM4);
    inputs.indexOffset = ctx.Input(INPUT_NUM5);
    inputs.labelOffset = ctx.Input(INPUT_NUM6);
    inputs.reorderFlag = ctx.Input(INPUT_NUM7);
    outputs.outdists = ctx.Output(INPUT_NUM0);
    outputs.outlabels = ctx.Output(INPUT_NUM1);

    KERNEL_CHECK_NULLPTR(inputs.indists, KERNEL_STATUS_PARAM_INVALID, "Get input[0], name[indists] failed");
    KERNEL_CHECK_NULLPTR(inputs.vmdists, KERNEL_STATUS_PARAM_INVALID, "Get input[1], name[vmdists] failed");
    KERNEL_CHECK_NULLPTR(inputs.size, KERNEL_STATUS_PARAM_INVALID, "Get input[2], name[size] failed");
    KERNEL_CHECK_NULLPTR(inputs.opflag, KERNEL_STATUS_PARAM_INVALID, "Get input[3], name[opflag] failed");
    KERNEL_CHECK_NULLPTR(inputs.attr, KERNEL_STATUS_PARAM_INVALID, "Get input[4], name[attr] failed");
    KERNEL_CHECK_NULLPTR(inputs.indexOffset, KERNEL_STATUS_PARAM_INVALID, "Get input[5], name[indexOffset] failed");
    KERNEL_CHECK_NULLPTR(inputs.labelOffset, KERNEL_STATUS_PARAM_INVALID, "Get input[6], name[labelOffset] failed");
    KERNEL_CHECK_NULLPTR(inputs.reorderFlag, KERNEL_STATUS_PARAM_INVALID, "Get input[7], name[reorderFlag] failed");
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
    KERNEL_LOG_INFO("Shape of input[5][indexOffset] is %s",
        ShapeToString(inputs.indexOffset->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[6][labelOffset] is %s",
        ShapeToString(inputs.labelOffset->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[7][reorderFlag] is %s",
        ShapeToString(inputs.reorderFlag->GetTensorShape()->GetDimSizes()).c_str());

    return KERNEL_STATUS_OK;
}

uint32_t TopkMultisearchCpuKernel::CheckInputShapes(const Inputs &inputs)
{
    KERNEL_LOG_INFO("TopkMultisearchCpuKernel CheckInputShapes begin");

    auto shapeIndists = inputs.indists->GetTensorShape();
    auto shapeVmdists = inputs.vmdists->GetTensorShape();
    auto shapeSize = inputs.size->GetTensorShape();
    auto shapeOpflag = inputs.opflag->GetTensorShape();
    auto shapeAttr = inputs.attr->GetTensorShape();
    auto shapeIndexOffset = inputs.indexOffset->GetTensorShape();
    auto shapeLabelOffset = inputs.labelOffset->GetTensorShape();
    auto shapeReorderFlag = inputs.reorderFlag->GetTensorShape();

    KERNEL_CHECK_TRUE(shapeIndists->GetDims() == INPUT_NUM3, KERNEL_STATUS_PARAM_INVALID,
        "Dims of input[0][indists] must be 3");
    KERNEL_CHECK_TRUE(shapeVmdists->GetDims() == INPUT_NUM3, KERNEL_STATUS_PARAM_INVALID,
        "Dims of input[1][vmdists] must be 3");
    KERNEL_CHECK_TRUE(shapeSize->GetDims() == INPUT_NUM3, KERNEL_STATUS_PARAM_INVALID,
        "Dims of input[2][size] must be 3");
    KERNEL_CHECK_TRUE(shapeOpflag->GetDims() == INPUT_NUM3, KERNEL_STATUS_PARAM_INVALID,
        "Dims of input[3][opflag] must be 3");
    KERNEL_CHECK_TRUE(shapeAttr->GetDims() == INPUT_NUM1, KERNEL_STATUS_PARAM_INVALID,
        "Dims of input[4][attr] must be 1");
    KERNEL_CHECK_TRUE(shapeIndexOffset->GetDims() == INPUT_NUM1, KERNEL_STATUS_PARAM_INVALID,
        "Dims of input[5][indexOffset] must be 1");
    KERNEL_CHECK_TRUE(shapeLabelOffset->GetDims() == INPUT_NUM1, KERNEL_STATUS_PARAM_INVALID,
        "Dims of input[6][labelOffset] must be 1");
    KERNEL_CHECK_TRUE(shapeReorderFlag->GetDims() == INPUT_NUM1, KERNEL_STATUS_PARAM_INVALID,
        "Dims of input[7][reorderFlag] must be 1");

    auto nq0 = shapeIndists->GetDimSize(INPUT_NUM1);
    auto nq1 = shapeVmdists->GetDimSize(INPUT_NUM1);
    KERNEL_CHECK_TRUE(nq0 == nq1, KERNEL_STATUS_PARAM_INVALID, "Nq of inputs must be same");
    nq_ = nq0;

    auto coreNum0 = shapeSize->GetDimSize(INPUT_NUM1);

    coreNum_ = coreNum0;

    flagSize_ = shapeOpflag->GetDimSize(INPUT_NUM2);

    auto attrCount = shapeAttr->GetDimSize(INPUT_NUM0);
    KERNEL_CHECK_TRUE(attrCount == TOPK_MULTISEARCH_ATTR_IDX_COUNT, KERNEL_STATUS_PARAM_INVALID,
        "Num of attrs must be %d", TOPK_MULTISEARCH_ATTR_IDX_COUNT);

    auto attr = static_cast<int64_t *>(inputs.attr->GetData());
    asc_ = *(attr + TOPK_MULTISEARCH_ATTR_ASC_IDX);
    k_ = *(attr + TOPK_MULTISEARCH_ATTR_K_IDX);
    burstLen_ = *(attr + TOPK_MULTISEARCH_ATTR_BURST_LEN_IDX);
    pageBlockNum_ = *(attr + TOPK_MULTISEARCH_ATTR_PAGE_BLOCK_NUM_IDX);
    indexNum_ = *(attr + TOPK_MULTISEARCH_ATTR_INDEX_NUM_IDX);
    quickTopk_ = *(attr + TOPK_MULTISEARCH_ATTR_QUICK_HEAP);
    blockSize_ = *(attr + TOPK_MULTISEARCH_ATTR_BLOCK_SIZE);
    KERNEL_CHECK_TRUE(k_ > 0 && burstLen_ > 0 && asc_ >= 0 && pageBlockNum_ > 0 && indexNum_ > 0,
        KERNEL_STATUS_PARAM_INVALID, "Value of asc, k, bustLen, reuseBlockNum, indexNum must ge 0");

    SetParallMode(inputs);

    return KERNEL_STATUS_OK;
}

void TopkMultisearchCpuKernel::SetParallMode(const Inputs &inputs)
{
    // only all block is last block of indexes, it can multi-core parallel by block
    auto reorderFlag = static_cast<uint16_t *>(inputs.reorderFlag->GetData());
    bool isAbleBlockParall = true;
    for (int i = 0; i < pageBlockNum_; ++i) {
        if (*(reorderFlag + i) == 0) {
            isAbleBlockParall = false;
            break;
        }
    }

    // only small batch(<= 2) and big blockNums to multi-core parallel by block
    if (isAbleBlockParall && pageBlockNum_ > nq_ && nq_ <= 2) {
        isBlockParall = true;
    }
}

void TopkMultisearchCpuKernel::UpdateOutputsShape(Outputs &outputs) const
{
    KERNEL_LOG_INFO("TopkMultisearchCpuKernel UpdateOutputsShape begin");

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

template <typename C>
void TopkMultisearchCpuKernel::DoComputeParallForBatch(size_t start, size_t end, const Inputs &inputs, Outputs &outputs,
    C &&cmp)
{
    KernelTensor<float16_t> indists(inputs.indists);
    KernelTensor<float16_t> vmdists(inputs.vmdists);
    KernelTensor<uint32_t> size(inputs.size);
    KernelTensor<uint16_t> opflag(inputs.opflag);
    KernelTensor<uint32_t> indexOffset(inputs.indexOffset);
    KernelTensor<uint32_t> labelOffset(inputs.labelOffset);
    KernelTensor<uint16_t> reorderFlag(inputs.reorderFlag);

    KernelTensor<float16_t> outdists(outputs.outdists);
    KernelTensor<int64_t> outlabels(outputs.outlabels);

    for (int64_t i = 0; i < pageBlockNum_; i++) {
        auto flagPtr = opflag.GetSubTensorDim0(i);
        for (int64_t j = 0; j < coreNum_; j++) {
            WAITING_FLAG_READY(*(flagPtr + j * flagSize_), TIMEOUT_CHECK_TICK, TIMEOUT_MS);
        }
        for (size_t j = start; j < end; j++) {
            ComputeBlock(j, i, indists, vmdists, size, indexOffset, labelOffset, reorderFlag, outdists, outlabels, cmp);
        }
    }
}

template <typename C>
void TopkMultisearchCpuKernel::DoComputeParallForBlcok(size_t tcnt, size_t tid, const Inputs &inputs, Outputs &outputs,
    C &&cmp)
{
    KernelTensor<float16_t> indists(inputs.indists);
    KernelTensor<float16_t> vmdists(inputs.vmdists);
    KernelTensor<uint32_t> size(inputs.size);
    KernelTensor<uint16_t> opflag(inputs.opflag);
    KernelTensor<uint32_t> indexOffset(inputs.indexOffset);
    KernelTensor<uint32_t> labelOffset(inputs.labelOffset);
    KernelTensor<uint16_t> reorderFlag(inputs.reorderFlag);

    KernelTensor<float16_t> outdists(outputs.outdists);
    KernelTensor<int64_t> outlabels(outputs.outlabels);

    for (int64_t bidx = tid; bidx < pageBlockNum_; bidx += tcnt) {
        auto flagPtr = opflag.GetSubTensorDim0(bidx);
        for (int64_t j = 0; j < coreNum_; j++) {
            WAITING_FLAG_READY(*(flagPtr + j * flagSize_), TIMEOUT_CHECK_TICK, TIMEOUT_MS);
        }
        for (int64_t i = 0; i < nq_; i++) {
            ComputeBlock(i, bidx, indists, vmdists, size, indexOffset, labelOffset, reorderFlag, outdists, outlabels,
                cmp);
        }
    }
}

template <typename C>
void TopkMultisearchCpuKernel::ReorderLastBlock(float16_t *outdists, int64_t *outlabel, C &&cmp)
{
    for (int64_t i = k_ - 1; i >= 1; --i) {
        std::swap(outdists[0], outdists[i]);
        std::swap(outlabel[0], outlabel[i]);
        UpdateHeap(outdists, outlabel, i, 0, cmp);
    }
}

template <typename C>
void TopkMultisearchCpuKernel::SortTopkBurst(std::vector<std::pair<float16_t, int64_t>>& topkBurstIdx,
    int64_t *outlabel, int64_t pageOffset, float16_t *outdists, C &&cmp)
{
    for (int i = 0; i < k_; ++i) {
        int64_t currentIdx = outlabel[i] - pageOffset;
        if (currentIdx >= 0) {
            topkBurstIdx.emplace_back(outdists[i], currentIdx);
        }
    }
    std::sort(topkBurstIdx.begin(), topkBurstIdx.end(),
        [&cmp](const std::pair<float16_t, int64_t> p1, const std::pair<float16_t, int64_t> p2) -> bool {
            return cmp(p2.first, p1.first);
        });
}

template <typename C>
void TopkMultisearchCpuKernel::ComputeBlock(size_t n, int64_t blockIdx, KernelTensor<float16_t> &indistsTensor,
    KernelTensor<float16_t> &vmdistsTensor, KernelTensor<uint32_t> &sizeTensor,
    KernelTensor<uint32_t> &indexOffsetTensor, KernelTensor<uint32_t> &labelOffsetTensor,
    KernelTensor<uint16_t> &reorderFlagTensor, KernelTensor<float16_t> &outdistsTensor,
    KernelTensor<int64_t> &outlabelsTensor, C &&cmp)
{
    float16_t *indists = indistsTensor.GetSubTensorDim1(blockIdx, n);
    float16_t *vmdists = vmdistsTensor.GetSubTensorDim1(blockIdx, n);
    uint16_t *vmlabel = reinterpret_cast<uint16_t *>(vmdists);
    uint32_t *size = sizeTensor.GetSubTensorDim0(blockIdx);
    uint32_t *indexOffset = indexOffsetTensor.GetSubTensorDim0(blockIdx);
    uint32_t *labelOffset = labelOffsetTensor.GetSubTensorDim0(blockIdx);
    uint32_t *baseLabelOffset = labelOffsetTensor.GetSubTensorDim0(0);
    uint16_t *reorderFlag = reorderFlagTensor.GetSubTensorDim0(blockIdx);
    float16_t *outdists = outdistsTensor.GetSubTensorDim1(*indexOffset, n);
    int64_t *outlabel = outlabelsTensor.GetSubTensorDim1(*indexOffset, n);

    int64_t baseOffset = static_cast<int64_t>(*labelOffset) * blockSize_;
    int64_t ntotal = static_cast<int64_t>(*size);
    int64_t idx = 0;
    int64_t burstIdx = 0;
    int64_t burstSize = ntotal / burstLen_;

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
                    outlabel[0] = baseOffset + idx;
                    UpdateHeap(outdists, outlabel, k_, 0, cmp);
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
            outlabel[0] = burstLen_ * i + (vmlabel[i * 2 + 1]) + baseOffset;
            UpdateHeap(outdists, outlabel, k_, 0, cmp);
        }

        // Stage two:update heap by Topk burst
        if (static_cast<bool >(*reorderFlag) || (blockIdx + 1 == pageBlockNum_)) {
            int64_t blockOffset = 0;
            int64_t currentIdx = 0;

            // index's blocks more than one page
            int indistsOffset = 0;
            int64_t pageOffset = (*baseLabelOffset) * blockSize_;
            if (blockIdx - (*labelOffset) >= 0) {
                // index's blocks no more than one page
                indistsOffset = blockIdx - (*labelOffset);
                pageOffset = 0;
            }

            std::vector<std::pair<float16_t, int64_t>> topkBurstIdx;
            std::pair<int64_t, int64_t> currentPostion;

            SortTopkBurst(topkBurstIdx, outlabel, pageOffset, outdists, cmp);

            // The first vaule is BlockIdx, the second vaule BurstIdx
            for (size_t i = 0; i < topkBurstIdx.size(); ++i) {
                // Idx in current block
                blockOffset = topkBurstIdx[i].second % blockSize_;
                // blockIdx in current page
                currentPostion.first = topkBurstIdx[i].second / blockSize_;
                // burst in current block
                currentPostion.second = blockOffset / burstLen_;

                float16_t *indistsSrc = indistsTensor.GetSubTensorDim1(currentPostion.first + indistsOffset, n);

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
                        outlabel[0] = pageOffset + blockOffset + j;
                        UpdateHeap(outdists, outlabel, k_, 0, cmp);
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
            outlabel[0] = baseOffset + idx;
            UpdateHeap(outdists, outlabel, k_, 0, cmp);
        }
        ++idx;
    }

    // reorder dist result in last block
    if (*reorderFlag) {
        ReorderLastBlock<C>(outdists, outlabel, cmp);
    }
}

template <typename C>
void TopkMultisearchCpuKernel::UpdateHeap(float16_t *dists, int64_t *label, int64_t len, int64_t index, C &&cmp) const
{
    int64_t l = 0;
    int64_t r = 0;
    int64_t m = 0;
    while (true) {
        l = 2 * index + 1; // 2 * index + 1 to find left subnode
        r = 2 * index + 2; // 2 * index + 2 to find right subnode
        m = index;
        if (l < len && cmp(dists[l], dists[m])) {
            m = l;
        }
        if (r < len && cmp(dists[r], dists[m])) {
            m = r;
        }
        if (m != index) {
            std::swap(dists[m], dists[index]);
            std::swap(label[m], label[index]);
            index = m;
        } else {
            break;
        }
    }
}

REGISTER_CPU_KERNEL(TOPK_MULTISEARCH, TopkMultisearchCpuKernel);
} // namespace aicpu