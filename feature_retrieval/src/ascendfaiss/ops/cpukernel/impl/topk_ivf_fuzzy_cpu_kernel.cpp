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


#include "topk_ivf_fuzzy_cpu_kernel.h"

#include <algorithm>
#include <string>

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "kernel_tensor.h"
#include "kernel_utils.h"
#include "kernel_shared_def.h"
#include "securec.h"

namespace {
const char* TOPK_IVF_FUZZY = "TopkIvfFuzzy";
const int INDEX_OFFSET_BIT = 32;
const int64_t INT_64_MAX = 0xffffffffffffffff;
}

namespace aicpu {
uint32_t TopkIvfFuzzyCpuKernel::Compute(CpuKernelContext &ctx)
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

    popQ_.resize(nq_, k_ / kBufRatio_);

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

uint32_t TopkIvfFuzzyCpuKernel::GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs) const
{
    KERNEL_LOG_INFO("TopkIvfFuzzyCpuKernel GetInOutAndCheck begin");

    inputs.indists = ctx.Input(INPUT_NUM0);
    inputs.vmdists = ctx.Input(INPUT_NUM1);
    inputs.ids = ctx.Input(INPUT_NUM2);
    inputs.size = ctx.Input(INPUT_NUM3);
    inputs.opflag = ctx.Input(INPUT_NUM4);
    inputs.attr = ctx.Input(INPUT_NUM5);
    outputs.outdists = ctx.Output(INPUT_NUM0);
    outputs.outlabels = ctx.Output(INPUT_NUM1);
    outputs.popdists = ctx.Output(INPUT_NUM2);
    outputs.poplabels = ctx.Output(INPUT_NUM3);

    KERNEL_CHECK_NULLPTR(inputs.indists, KERNEL_STATUS_PARAM_INVALID, "Get input[0], name[indists] failed");
    KERNEL_CHECK_NULLPTR(inputs.vmdists, KERNEL_STATUS_PARAM_INVALID, "Get input[1], name[vmdists] failed");
    KERNEL_CHECK_NULLPTR(inputs.ids, KERNEL_STATUS_PARAM_INVALID, "Get input[1], name[vmdists] failed");
    KERNEL_CHECK_NULLPTR(inputs.size, KERNEL_STATUS_PARAM_INVALID, "Get input[2], name[size] failed");
    KERNEL_CHECK_NULLPTR(inputs.opflag, KERNEL_STATUS_PARAM_INVALID, "Get input[3], name[opflag] failed");
    KERNEL_CHECK_NULLPTR(inputs.attr, KERNEL_STATUS_PARAM_INVALID, "Get input[4], name[attr] failed");
    KERNEL_CHECK_NULLPTR(outputs.outdists, KERNEL_STATUS_PARAM_INVALID, "Get output[0], name[outdists] failed");
    KERNEL_CHECK_NULLPTR(outputs.outlabels, KERNEL_STATUS_PARAM_INVALID, "Get output[1], name[outlabels] failed");
    KERNEL_CHECK_NULLPTR(outputs.popdists, KERNEL_STATUS_PARAM_INVALID, "Get output[2], name[popdists] failed");
    KERNEL_CHECK_NULLPTR(outputs.poplabels, KERNEL_STATUS_PARAM_INVALID, "Get output[3], name[poplabels] failed");

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

uint32_t TopkIvfFuzzyCpuKernel::CheckInputShapes(const Inputs &inputs)
{
    KERNEL_LOG_INFO("TopkIvfFuzzyCpuKernel CheckInputShapes begin");

    auto shapeIndists = inputs.indists->GetTensorShape();
    auto shapeVmdists = inputs.vmdists->GetTensorShape();
    auto shapeIds = inputs.ids->GetTensorShape();
    auto shapeSize = inputs.size->GetTensorShape();
    auto shapeOpflag = inputs.opflag->GetTensorShape();
    auto shapeAttr = inputs.attr->GetTensorShape();

    KERNEL_CHECK_TRUE(shapeIndists->GetDims() == INPUT_NUM2,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][indists] must be 2");
    KERNEL_CHECK_TRUE(shapeVmdists->GetDims() == INPUT_NUM2,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][vmdists] must be 2");
    KERNEL_CHECK_TRUE(shapeIds->GetDims() == INPUT_NUM2,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][ids] must be 2");
    KERNEL_CHECK_TRUE(shapeSize->GetDims() == INPUT_NUM2,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][size] must be 2");
    KERNEL_CHECK_TRUE(shapeOpflag->GetDims() == INPUT_NUM3,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][opflag] must be 3");
    KERNEL_CHECK_TRUE(shapeAttr->GetDims() == INPUT_NUM1,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][attr] must be 1");

    auto nq0 = shapeIndists->GetDimSize(INPUT_NUM0);
    auto nq1 = shapeVmdists->GetDimSize(INPUT_NUM0);
    auto nq2 = shapeIds->GetDimSize(INPUT_NUM0);
    auto nq3 = shapeSize->GetDimSize(INPUT_NUM0);
    auto nq4 = shapeOpflag->GetDimSize(INPUT_NUM0);
    KERNEL_CHECK_TRUE(nq0 == nq1 && nq0 == nq2 && nq0 == nq3 && nq0 == nq4,
        KERNEL_STATUS_PARAM_INVALID, "Nq of inputs must be same");
    nq_ = nq0;

    flagNum_ = shapeOpflag->GetDimSize(INPUT_NUM1);
    flagSize_ = shapeOpflag->GetDimSize(INPUT_NUM2);

    auto attrCount = shapeAttr->GetDimSize(INPUT_NUM0);
    KERNEL_CHECK_TRUE(attrCount == TOPK_IVF_FUZZY_ATTR_IDX_COUNT,
        KERNEL_STATUS_PARAM_INVALID, "Num of attrs must be %d", TOPK_IVF_FUZZY_ATTR_IDX_COUNT);

    auto attr = static_cast<int64_t *>(inputs.attr->GetData());
    asc_ = *(attr + TOPK_IVF_FUZZY_ATTR_ASC_IDX);
    k_ = *(attr + TOPK_IVF_FUZZY_ATTR_K_IDX);
    burstLen_ = *(attr + TOPK_IVF_FUZZY_ATTR_BURST_LEN_IDX);
    l3SegNum_ = *(attr + TOPK_IVF_FUZZY_ATTR_L3_SEG_NUM_IDX);
    l3SegSize_ = *(attr + TOPK_IVF_FUZZY_ATTR_L3_SEG_SIZE_IDX);
    kHeapRatio_ = *(attr + TOPK_IVF_FUZZY_ATTR_K_HEAP_RATIO_IDX);
    kBufRatio_ = *(attr + TOPK_IVF_FUZZY_ATTR_K_BUF_RATIO_IDX);
    queryBatchSize_ = *(attr + TOPK_IVF_FUZZY_ATTR_Q_BATCH_SIZE_IDX);
    ivfFuzzyTopkMode_ = *(attr + TOPK_IVF_FUZZY_ATTR_SORT_MODE);
    KERNEL_CHECK_TRUE(k_ > 0 && burstLen_ > 0 &&
        l3SegNum_ > 0 && l3SegSize_ > 0 && kHeapRatio_ > 0 && kBufRatio_ > 0 && queryBatchSize_ > 0,
        KERNEL_STATUS_PARAM_INVALID, "Value of attrs must greater than 0");

    return KERNEL_STATUS_OK;
}

void TopkIvfFuzzyCpuKernel::UpdateInOutShape(Inputs &inputs, Outputs &outputs) const
{
    KERNEL_LOG_INFO("TopkIvfFuzzyCpuKernel UpdateInOutShape begin");

    auto shapeIndists = inputs.indists->GetTensorShape();
    std::vector<int64_t> dimIndists = shapeIndists->GetDimSizes();
    dimIndists[INPUT_NUM1] = l3SegNum_ * l3SegSize_;
    shapeIndists->SetDimSizes(dimIndists);

    auto shapeVmdists = inputs.vmdists->GetTensorShape();
    std::vector<int64_t> dimVmdists = shapeVmdists->GetDimSizes();
    dimVmdists[INPUT_NUM1] = l3SegNum_ * l3SegSize_ / burstLen_ * 2; // 1 vmdists has 2 values: [dist label]
    shapeVmdists->SetDimSizes(dimVmdists);

    auto shapeIds = inputs.ids->GetTensorShape();
    std::vector<int64_t> dimIds = shapeIds->GetDimSizes();
    dimIds[INPUT_NUM1] = l3SegNum_;
    shapeIds->SetDimSizes(dimIds);

    auto shapeSize = inputs.size->GetTensorShape();
    std::vector<int64_t> dimSize = shapeSize->GetDimSizes();
    dimSize[INPUT_NUM1] = l3SegNum_;
    shapeSize->SetDimSizes(dimSize);

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

    auto shapePopdists = outputs.popdists->GetTensorShape();
    std::vector<int64_t> dimPopdists;
    dimPopdists.push_back(nq_);
    dimPopdists.push_back(k_ / kBufRatio_);
    shapePopdists->SetDimSizes(dimPopdists);

    auto shapePoplabels = outputs.poplabels->GetTensorShape();
    std::vector<int64_t> dimPoplabels;
    dimPoplabels.push_back(nq_);
    dimPoplabels.push_back(k_ / kBufRatio_);
    shapePoplabels->SetDimSizes(dimPoplabels);
}

void TopkIvfFuzzyCpuKernel::InitTopkHeap(Outputs &outputs) const
{
    uint16_t *outdists = static_cast<uint16_t *>(outputs.outdists->GetData());
    int64_t *outlabels = static_cast<int64_t *>(outputs.outlabels->GetData());
    std::fill_n(outlabels, nq_ * k_, INT_64_MAX);
    if (asc_ != 0) {
        std::fill_n(outdists, nq_ * k_, 0x7bff);
    } else {
        std::fill_n(outdists, nq_ * k_, 0xfbff);
    }
}

template <typename C>
void TopkIvfFuzzyCpuKernel::DoCompute(size_t tcnt, size_t tid, const Inputs &inputs, Outputs &outputs, C &&cmp)
{
    KernelTensor<float16_t> indists(inputs.indists);
    KernelTensor<float16_t> vmdists(inputs.vmdists);
    KernelTensor<int64_t> ids(inputs.ids);
    KernelTensor<uint32_t> size(inputs.size);
    KernelTensor<uint16_t> opflag(inputs.opflag);

    KernelTensor<float16_t> outdists(outputs.outdists);
    KernelTensor<int64_t> outlabels(outputs.outlabels);
    KernelTensor<float16_t> popdists(outputs.popdists);
    KernelTensor<int64_t> poplabels(outputs.poplabels);

    std::vector<std::pair<float16_t, uint32_t>> distIndexPairs(l3SegNum_ * l3SegSize_ / burstLen_);
    for (int64_t qidx = tid; qidx < nq_; qidx += tcnt) {
        int64_t flagIdx = qidx / queryBatchSize_ * queryBatchSize_;
        auto flagPtr = opflag.GetSubTensorDim0(flagIdx);
        for (int64_t i = 0; i < flagNum_; ++i) {
            WAITING_FLAG_READY(*(flagPtr + i * flagSize_), TIMEOUT_CHECK_TICK, TIMEOUT_MS);
        }
        ComputeQueryBatch(qidx, indists, vmdists, ids, size,
            outdists, outlabels, popdists, poplabels, distIndexPairs, cmp);
    }
}

int64_t TopkIvfFuzzyCpuKernel::GetRealLabelByIndex(const int64_t *ids, int64_t index) const
{
    const uint32_t segIdx = index >> INDEX_OFFSET_BIT;
    const uint32_t idx = static_cast<uint32_t>(index) % l3SegSize_;
    int64_t *id = reinterpret_cast<int64_t *>(*(ids + segIdx));
    return id[idx];
}

template <typename C>
void TopkIvfFuzzyCpuKernel::FineRanking(uint32_t beginIdx, uint32_t endIdx, int64_t *outlabels, float16_t *outdists,
    float16_t *indists, int64_t qidx, uint32_t segIdx, int kHeap, C &&cmp)
{
    for (uint32_t idx = beginIdx; idx < endIdx; ++idx) {
        if (cmp(outdists[0], indists[idx])) {
            if (outlabels[0] != INT_64_MAX) {
                popQ_[qidx].push(outdists[0], outlabels[0]);
            }
            outdists[0] = indists[idx];
            outlabels[0] = ((static_cast<uint64_t>(segIdx)) << INDEX_OFFSET_BIT) | idx;
            UpdateHeap(outdists, outlabels, kHeap, 0, cmp);
        }
    }
}

template <typename C>
void TopkIvfFuzzyCpuKernel::ComputeQueryBatch(int64_t qidx,
                                              KernelTensor<float16_t> &indistsTensor,
                                              KernelTensor<float16_t> &vmdistsTensor,
                                              KernelTensor<int64_t> &idsTensor,
                                              KernelTensor<uint32_t> &sizeTensor,
                                              KernelTensor<float16_t> &outdistsTensor,
                                              KernelTensor<int64_t> &outlabelsTensor,
                                              KernelTensor<float16_t> &popdistsTensor,
                                              KernelTensor<int64_t> &poplabelsTensor,
                                              std::vector<std::pair<float16_t, uint32_t>> &distIndexPairs,
                                              C &&cmp)
{
    float16_t *indists = indistsTensor.GetSubTensorDim0(qidx);
    float16_t *vmdists = vmdistsTensor.GetSubTensorDim0(qidx);
    int64_t *ids = idsTensor.GetSubTensorDim0(qidx);
    uint32_t *size = sizeTensor.GetSubTensorDim0(qidx);
    float16_t *outdists = outdistsTensor.GetSubTensorDim0(qidx);
    int64_t *outlabels = outlabelsTensor.GetSubTensorDim0(qidx);
    float16_t *popdists = popdistsTensor.GetSubTensorDim0(qidx);
    int64_t *poplabels = poplabelsTensor.GetSubTensorDim0(qidx);

    int64_t distIndPairsLen = ComputeDistIndexPairs(distIndexPairs, vmdists, size, cmp);

    const int kHeap = k_ / kHeapRatio_;
    int heapPos = kHeap;

    auto iterEnd = distIndexPairs.begin() + std::min(k_, distIndPairsLen);
    if (ivfFuzzyTopkMode_ == 1) {
        iterEnd = distIndexPairs.begin() + distIndPairsLen;
    }
    for (auto iter = distIndexPairs.begin(); iter != iterEnd; ++iter) {
        const uint32_t beginIdx = (*iter).second;
        const uint32_t segIdx = beginIdx / l3SegSize_;
        const uint32_t endIdx = std::min(beginIdx + burstLen_, segIdx * l3SegSize_ + size[segIdx]);

        if (ivfFuzzyTopkMode_ == 1 || iter - distIndexPairs.begin() < kHeap) {
            FineRanking(beginIdx, endIdx, outlabels, outdists, indists, qidx, segIdx, kHeap, cmp);
            continue;
        }

        for (uint32_t idx = beginIdx; idx < endIdx; ++idx) {
            if (heapPos >= k_) {
                break;
            }
            if (cmp(outdists[0], indists[idx])) {
                outdists[heapPos] = indists[idx];
                outlabels[heapPos] = (((uint64_t)segIdx) << INDEX_OFFSET_BIT) | idx;
                ++heapPos;
            }
        }
    }

    TransIndexToLabel(qidx, ids, outlabels);

    (void)CopyPopToOutput(qidx, popdists, poplabels);
}

template <typename C>
int TopkIvfFuzzyCpuKernel::ComputeDistIndexPairs(
    std::vector<std::pair<float16_t, uint32_t>> &distIndexPairs, float16_t *vmdists, uint32_t *size, C &&cmp)
{
    const int extremeLen = 2 * (l3SegSize_ / burstLen_);
    int distIndPairsLen = 0;
    for (int64_t segIdx = 0; segIdx < l3SegNum_; ++segIdx) {
        int indPointer = segIdx * l3SegSize_;
        const int pairSize = (size[segIdx] + burstLen_ - 1) / burstLen_;
        for (int i = 0; i < pairSize; ++i) {
            int vmdistsIdx = segIdx * extremeLen + i * 2;  // each pairSize has 2 values
            float16_t dis = vmdists[vmdistsIdx];
            distIndexPairs[distIndPairsLen].first = dis;
            distIndexPairs[distIndPairsLen].second = indPointer;
            indPointer += burstLen_;
            ++distIndPairsLen;
        }
    }
    // use nth_element to move the top k elements to the front, using the opposite of "compare" as comparator
    if (k_ < distIndPairsLen) {
        std::nth_element(
            distIndexPairs.begin(), distIndexPairs.begin() + k_ - 1, distIndexPairs.begin() + distIndPairsLen,
            [&cmp] (const std::pair<float16_t, uint32_t> &item1, const std::pair<float16_t, uint32_t> &item2) {
                return cmp(item2.first, item1.first);
            });
    }

    return distIndPairsLen;
}

void TopkIvfFuzzyCpuKernel::TransIndexToLabel(int64_t qidx, int64_t *ids, int64_t *outlabels)
{
    for (int i = 0; i < k_; ++i) {
        int64_t index = outlabels[i];
        if (index == INT_64_MAX) {
            continue;
        }
        outlabels[i] = GetRealLabelByIndex(ids, index);
    }

    int bufferPoppedBegin = popQ_[qidx].getBegin();
    int bufferPoppedEnd = popQ_[qidx].getEnd();
    int bufferPoppedSize = popQ_[qidx].getSize();
    int64_t *bufferPoppedIds = popQ_[qidx].getIds();

    if (bufferPoppedBegin < bufferPoppedEnd) {
        for (int i = bufferPoppedBegin; i < bufferPoppedEnd; ++i) {
            int64_t index = *(bufferPoppedIds + i);
            *(bufferPoppedIds + i) = GetRealLabelByIndex(ids, index);
        }
    } else if (bufferPoppedBegin > bufferPoppedEnd) {
        for (int i = bufferPoppedBegin; i < bufferPoppedSize + 1; ++i) {
            int64_t index = *(bufferPoppedIds + i);
            *(bufferPoppedIds + i) = GetRealLabelByIndex(ids, index);
        }
        for (int i = 0; i < bufferPoppedEnd; ++i) {
            int64_t index = *(bufferPoppedIds + i);
            *(bufferPoppedIds + i) = GetRealLabelByIndex(ids, index);
        }
    }
}

uint32_t TopkIvfFuzzyCpuKernel::CopyPopToOutput(int64_t qidx, float16_t *popdists, int64_t *poplabels)
{
    int bufferPoppedBegin = popQ_[qidx].getBegin();
    int bufferPoppedEnd = popQ_[qidx].getEnd();
    int bufferPoppedSize = popQ_[qidx].getSize();
    float16_t* bufferPoppedDists = popQ_[qidx].getDists();
    int64_t* bufferPoppedIds = popQ_[qidx].getIds();

    if (bufferPoppedBegin > bufferPoppedEnd) {
        int latterSize = bufferPoppedSize - bufferPoppedBegin + 1;
        auto err = memcpy_s(popdists,
                            latterSize * sizeof(float16_t),
                            bufferPoppedDists + bufferPoppedBegin,
                            latterSize * sizeof(float16_t));
        KERNEL_CHECK_TRUE(err == EOK, KERNEL_STATUS_INNER_ERROR, "memcpy distancePopped former err, err=%d", err);
        err = memcpy_s(poplabels,
                       latterSize * sizeof(int64_t),
                       bufferPoppedIds + bufferPoppedBegin,
                       latterSize * sizeof(int64_t));
        KERNEL_CHECK_TRUE(err == EOK, KERNEL_STATUS_INNER_ERROR, "memcpy indexPopped former err, err=%d", err);
        if (bufferPoppedEnd > 0) {
            err = memcpy_s(popdists + latterSize,
                           bufferPoppedEnd * sizeof(float16_t),
                           bufferPoppedDists,
                           bufferPoppedEnd * sizeof(float16_t));
            KERNEL_CHECK_TRUE(err == EOK, KERNEL_STATUS_INNER_ERROR, "memcpy distancePopped latter err, err=%d", err);
            err = memcpy_s(poplabels + latterSize,
                           bufferPoppedEnd * sizeof(int64_t),
                           bufferPoppedIds,
                           bufferPoppedEnd * sizeof(int64_t));
            KERNEL_CHECK_TRUE(err == EOK, KERNEL_STATUS_INNER_ERROR, "memcpy indexPopped latter err, err=%d", err);
        }
    } else if (bufferPoppedBegin < bufferPoppedEnd) {
        int cpySize = bufferPoppedEnd - bufferPoppedBegin;
        int beginPos = k_ / kBufRatio_ - cpySize;
        auto err = memcpy_s(popdists + beginPos,
                            cpySize * sizeof(float16_t),
                            bufferPoppedDists + bufferPoppedBegin,
                            cpySize * sizeof(float16_t));
        KERNEL_CHECK_TRUE(err == EOK, KERNEL_STATUS_INNER_ERROR, "memcpy distancePopped whole err, err=%d", err);
        err = memcpy_s(poplabels + beginPos,
                       cpySize * sizeof(int64_t),
                       bufferPoppedIds + bufferPoppedBegin,
                       cpySize * sizeof(int64_t));
        KERNEL_CHECK_TRUE(err == EOK, KERNEL_STATUS_INNER_ERROR, "memcpy indexPopped whole err, err=%d", err);
    }
    return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(TOPK_IVF_FUZZY, TopkIvfFuzzyCpuKernel);
} // namespace aicpu