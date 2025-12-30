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


#include "topk_ivfsqt_l2_cpu_kernel.h"

#include <algorithm>
#include <cmath>

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "kernel_tensor.h"
#include "kernel_utils.h"
#include "kernel_shared_def.h"

namespace {
const char* TOPK_IVFSQT_L2 = "TopkIvfsqtL2";
}

namespace aicpu {
uint32_t TopkIvfsqtL2CpuKernel::Compute(CpuKernelContext &ctx)
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

    UpdateInOutShape(inputs, outputs);

#ifdef AICPU_UTEST
    uint32_t core = 1;
#else
    uint32_t core = std::min({CpuKernelUtils::GetCPUNum(ctx), static_cast<uint32_t>(nq_)});
#endif

    auto computeFunc = [&](size_t start, size_t end) {
        (void)end; // end is unuseful in this function
        DoCompute(core, start, inputs, outputs);
    };

#ifdef AICPU_UTEST
    computeFunc(0, 1);
#else
    CpuKernelUtils::ParallelFor(ctx, core, 1, computeFunc);
#endif

    return KERNEL_STATUS_OK;
}

uint32_t TopkIvfsqtL2CpuKernel::GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs) const
{
    KERNEL_LOG_INFO("TopkIvfsqtL2CpuKernel GetInOutAndCheck begin");

    inputs.indists = ctx.Input(INPUT_NUM0);
    inputs.opflag = ctx.Input(INPUT_NUM1);
    inputs.attr = ctx.Input(INPUT_NUM2);
    inputs.subListSegNum = ctx.Input(INPUT_NUM3);  // int
    inputs.subListOffset = ctx.Input(INPUT_NUM4);  // uint64_t
    inputs.subListIndicesOffset = ctx.Input(INPUT_NUM5);  // int64_t
    inputs.subListSizes = ctx.Input(INPUT_NUM6);  // uint32_t
    inputs.l1KIndices = ctx.Input(INPUT_NUM7);  // uint16_t

    outputs.subListOffsetL3 = ctx.Output(INPUT_NUM0);  // uint64_t
    outputs.idResult = ctx.Output(INPUT_NUM1);  // int64_t
    outputs.opSize = ctx.Output(INPUT_NUM2);  // uint32_t

    KERNEL_CHECK_NULLPTR(inputs.indists, KERNEL_STATUS_PARAM_INVALID, "Get input[0], name[indists] failed");
    KERNEL_CHECK_NULLPTR(inputs.opflag, KERNEL_STATUS_PARAM_INVALID, "Get input[1], name[opflag] failed");
    KERNEL_CHECK_NULLPTR(inputs.attr, KERNEL_STATUS_PARAM_INVALID, "Get input[2], name[attr] failed");
    KERNEL_CHECK_NULLPTR(inputs.subListSegNum, KERNEL_STATUS_PARAM_INVALID,
        "Get input[3], name[subListSegNum] failed");
    KERNEL_CHECK_NULLPTR(inputs.subListOffset, KERNEL_STATUS_PARAM_INVALID,
        "Get input[4], name[subListOffset] failed");
    KERNEL_CHECK_NULLPTR(inputs.subListIndicesOffset, KERNEL_STATUS_PARAM_INVALID,
        "Get input[5], name[subListIndicesOffset] failed");
    KERNEL_CHECK_NULLPTR(inputs.subListSizes, KERNEL_STATUS_PARAM_INVALID, "Get input[6], name[subListSizes] failed");
    KERNEL_CHECK_NULLPTR(inputs.l1KIndices, KERNEL_STATUS_PARAM_INVALID, "Get input[7], name[l1KIndices] failed");

    KERNEL_CHECK_NULLPTR(outputs.subListOffsetL3, KERNEL_STATUS_PARAM_INVALID,
        "Get output[0], name[subListOffsetL3] failed");
    KERNEL_CHECK_NULLPTR(outputs.idResult, KERNEL_STATUS_PARAM_INVALID, "Get output[1], name[idResult] failed");
    KERNEL_CHECK_NULLPTR(outputs.opSize, KERNEL_STATUS_PARAM_INVALID, "Get output[2], name[opSize] failed");

    KERNEL_LOG_INFO("Shape of input[0][indists] is %s",
        ShapeToString(inputs.indists->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[1][opflag] is %s",
        ShapeToString(inputs.opflag->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[2][attr] is %s",
        ShapeToString(inputs.attr->GetTensorShape()->GetDimSizes()).c_str());

    return KERNEL_STATUS_OK;
}

uint32_t TopkIvfsqtL2CpuKernel::CheckInOutShapes(const Inputs &inputs, const Outputs &outputs)
{
    KERNEL_LOG_INFO("TopkIvfsqtL2CpuKernel CheckInOutShapes begin");

    auto shapeIndists = inputs.indists->GetTensorShape();
    auto shapeOpflag = inputs.opflag->GetTensorShape();
    auto shapeAttr = inputs.attr->GetTensorShape();

    auto shapeSubListSegNum = inputs.subListSegNum->GetTensorShape();  // numLists * SUBCENTER_NUM
    auto shapeSubListOffset = inputs.subListOffset->GetTensorShape();  // numLists * SUBCENTER_NUM
    auto shapeSubListIndicesOffset = inputs.subListIndicesOffset->GetTensorShape();  // numLists * SUBCENTER_NUM
    auto shapeSubListSizes = inputs.subListSizes->GetTensorShape();  // numLists * SUBCENTER_NUM
    auto shapel1KIndices = inputs.l1KIndices->GetTensorShape();  // n, nprobe
    auto shapeOutSubListOffsetL3 = outputs.subListOffsetL3->GetTensorShape();  // n, l3SegmentNum
    auto shapeOutIdResult = outputs.idResult->GetTensorShape();  // n, l3SegmentNum
    auto shapeOutOpSize = outputs.opSize->GetTensorShape();  // n, l3SegmentNum

    KERNEL_CHECK_TRUE(shapeIndists->GetDims() == INPUT_NUM2,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][indists] must be 2");
    KERNEL_CHECK_TRUE(shapeOpflag->GetDims() == INPUT_NUM3,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[2][opflag] must be 3");
    KERNEL_CHECK_TRUE(shapeAttr->GetDims() == INPUT_NUM1,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[3][attr] must be 1");
    KERNEL_CHECK_TRUE(shapeSubListSegNum->GetDims() == INPUT_NUM1,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[4][subListSegNum] must be 1");
    KERNEL_CHECK_TRUE(shapeSubListOffset->GetDims() == INPUT_NUM1,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[5][subListOffset] must be 1");
    KERNEL_CHECK_TRUE(shapeSubListIndicesOffset->GetDims() == INPUT_NUM1,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[6][subListIndicesOffset] must be 1");
    KERNEL_CHECK_TRUE(shapeSubListSizes->GetDims() == INPUT_NUM1,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[7][subListSizes] must be 1");
    KERNEL_CHECK_TRUE(shapel1KIndices->GetDims() == INPUT_NUM2,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[8][shapel1KIndices] must be 2");

    KERNEL_CHECK_TRUE(shapeOutSubListOffsetL3->GetDims() == INPUT_NUM2,
        KERNEL_STATUS_PARAM_INVALID, "Dims of output[2][subListOffsetL3] must be 2");
    KERNEL_CHECK_TRUE(shapeOutIdResult->GetDims() == INPUT_NUM2,
        KERNEL_STATUS_PARAM_INVALID, "Dims of output[3][idResult] must be 2");
    KERNEL_CHECK_TRUE(shapeOutOpSize->GetDims() == INPUT_NUM2,
        KERNEL_STATUS_PARAM_INVALID, "Dims of output[4][opSize] must be 2");

    auto nq0 = shapeIndists->GetDimSize(INPUT_NUM0);
    auto nq1 = shapeOpflag->GetDimSize(INPUT_NUM0);
    auto nq2 = shapeOutSubListOffsetL3->GetDimSize(INPUT_NUM0);
    auto nq3 = shapeOutIdResult->GetDimSize(INPUT_NUM0);
    auto nq4 = shapeOutOpSize->GetDimSize(INPUT_NUM0);
    auto nq5 = shapel1KIndices->GetDimSize(INPUT_NUM0);
    KERNEL_CHECK_TRUE(nq0 == nq1 && nq0 == nq2 && nq0 == nq3 && nq0 == nq4 && nq0 == nq5,
        KERNEL_STATUS_PARAM_INVALID, "Nq of inputs must be same");
    nq_ = nq0;

    flagNum_ = shapeOpflag->GetDimSize(INPUT_NUM1);
    flagSize_ = shapeOpflag->GetDimSize(INPUT_NUM2);

    auto attrCount = shapeAttr->GetDimSize(INPUT_NUM0);
    KERNEL_CHECK_TRUE(attrCount == TOPK_IVFSQT_L2_ATTR_IDX_COUNT,
        KERNEL_STATUS_PARAM_INVALID, "Num of attrs must be %d", TOPK_IVFSQT_L2_ATTR_IDX_COUNT);

    auto attr = static_cast<int64_t *>(inputs.attr->GetData());
    k_ = *(attr + TOPK_IVFSQT_L2_ATTR_K_IDX);
    subcenterNum_ = *(attr + TOPK_IVFSQT_L2_ATTR_SUBCENTER_NUM_IDX);
    l3SegNum_ = *(attr + TOPK_IVFSQT_L2_ATTR_L3_SEG_NUM_IDX);
    l3SegSize_ = *(attr + TOPK_IVFSQT_L2_ATTR_L3_SEG_SIZE_IDX);
    l1NProbe_ = *(attr + TOPK_IVFSQT_L2_ATTR_L1_NPROBE_IDX);
    pageShapedDataOffsetStep_ = *(attr + TOPK_IVFSQT_L2_ATTR_PAGE_SHAPED_DATA_OFFSET_STEP_IDX);
    queryBatchSize_ = *(attr + TOPK_IVFSQT_L2_ATTR_Q_BATCH_SIZE_IDX);

    KERNEL_CHECK_TRUE(k_ > 0 && queryBatchSize_ > 0,
        KERNEL_STATUS_PARAM_INVALID, "Value of attrs must greater than 0");

    auto len0 = shapeSubListSegNum->GetDimSize(INPUT_NUM0);
    auto len1 = shapeSubListOffset->GetDimSize(INPUT_NUM0);
    auto len2 = shapeSubListIndicesOffset->GetDimSize(INPUT_NUM0);
    auto len3 = shapeSubListSizes->GetDimSize(INPUT_NUM0);
    KERNEL_CHECK_TRUE(len0 == len1 && len0 == len2 && len0 == len3,
        KERNEL_STATUS_PARAM_INVALID, "len of inputs postprocess subList data must be same");

    return KERNEL_STATUS_OK;
}

void TopkIvfsqtL2CpuKernel::UpdateInOutShape(Inputs &inputs, Outputs &outputs) const
{
    KERNEL_LOG_INFO("TopkIvfsqtL2CpuKernel UpdateInOutShape begin");

    auto shapel1KIndices = inputs.l1KIndices->GetTensorShape();
    std::vector<int64_t> dimOutl1KIndices;
    dimOutl1KIndices.push_back(nq_);
    dimOutl1KIndices.push_back(l1NProbe_);
    shapel1KIndices->SetDimSizes(dimOutl1KIndices);

    auto shapeOutSubListOffsetL3 = outputs.subListOffsetL3->GetTensorShape();
    std::vector<int64_t> dimOutSubListOffsetL3;
    dimOutSubListOffsetL3.push_back(nq_);
    dimOutSubListOffsetL3.push_back(l3SegNum_);
    shapeOutSubListOffsetL3->SetDimSizes(dimOutSubListOffsetL3);

    auto shapeOutIdResult = outputs.idResult->GetTensorShape();
    std::vector<int64_t> dimOutSubOutIdResult;
    dimOutSubOutIdResult.push_back(nq_);
    dimOutSubOutIdResult.push_back(l3SegNum_);
    shapeOutIdResult->SetDimSizes(dimOutSubOutIdResult);

    auto shapeOutOpSize = outputs.opSize->GetTensorShape();
    std::vector<int64_t> dimOutOpSize;
    dimOutOpSize.push_back(nq_);
    dimOutOpSize.push_back(l3SegNum_);
    shapeOutOpSize->SetDimSizes(dimOutOpSize);
}

void TopkIvfsqtL2CpuKernel::DoCompute(size_t tcnt, size_t tid, const Inputs &inputs, const Outputs &outputs)
{
    KERNEL_LOG_INFO("TopkIvfsqtL2CpuKernel DoCompute begin");
    KernelTensor<float16_t> indists(inputs.indists);
    KernelTensor<uint16_t> opflag(inputs.opflag);
    KernelTensor<int> subListSegNum(inputs.subListSegNum);
    KernelTensor<uint64_t> subListOffset(inputs.subListOffset);
    KernelTensor<int64_t> subListIndicesOffset(inputs.subListIndicesOffset);
    KernelTensor<uint32_t> subListSizes(inputs.subListSizes);
    KernelTensor<uint16_t> l1KIndices(inputs.l1KIndices);

    KernelTensor<uint64_t> subListOffsetL3(outputs.subListOffsetL3);
    KernelTensor<int64_t> idResult(outputs.idResult);
    KernelTensor<uint32_t> opSize(outputs.opSize);

    for (int64_t qidx = tid; qidx < nq_; qidx += tcnt) {
        int64_t flagIdx = qidx / queryBatchSize_ * queryBatchSize_;
        auto flagPtr = opflag.GetSubTensorDim0(flagIdx);
        for (int64_t i = 0; i < flagNum_; ++i) {
            WAITING_FLAG_READY(*(flagPtr + i * flagSize_), TIMEOUT_CHECK_TICK, TIMEOUT_MS);
        }
        ComputeQueryBatch(qidx, indists, subListSegNum, subListOffset, subListIndicesOffset, subListSizes,
            l1KIndices, subListOffsetL3, idResult, opSize);
    }
}

void TopkIvfsqtL2CpuKernel::ComputeQueryBatch(int64_t qidx,
                                              KernelTensor<float16_t> &indistsTensor,
                                              KernelTensor<int> &subListSegNumTensor,
                                              KernelTensor<uint64_t> &subListOffsetTensor,
                                              KernelTensor<int64_t> &subListIndicesOffsetTensor,
                                              KernelTensor<uint32_t> &subListSizesTensors,
                                              KernelTensor<uint16_t> &l1KIndicesTensors,
                                              KernelTensor<uint64_t> &subListOffsetL3Tensor,
                                              KernelTensor<int64_t> &idResultTensor,
                                              KernelTensor<uint32_t> &opSizeTensor)
{
    KERNEL_LOG_INFO("TopkIvfsqtL2CpuKernel DoCompute begin");
    float16_t *indists = indistsTensor.GetSubTensorDim0(qidx);

    std::vector<std::pair<float16_t, int64_t>> p(l1NProbe_ * subcenterNum_);
    uint16_t *l1KIndices = l1KIndicesTensors.GetSubTensorDim0(qidx);
    int distIndPairsLen = 0;
    for (int i = 0; i < l1NProbe_; i++) {
        for (int j = 0; j < subcenterNum_; j++) {
            p[distIndPairsLen].first = *(indists + distIndPairsLen);
            p[distIndPairsLen].second = static_cast<int64_t>(l1KIndices[i]) * subcenterNum_ + j;
            ++distIndPairsLen;
        }
    }

    std::partial_sort(p.begin(), p.begin() + k_, p.end());

    // postProcess
    int *subListSegNum = subListSegNumTensor.GetSubTensorDim0(0);
    uint64_t *subListOffset = subListOffsetTensor.GetSubTensorDim0(0);
    int64_t *subListIndicesOffset = subListIndicesOffsetTensor.GetSubTensorDim0(0);
    uint32_t *subListSizes = subListSizesTensors.GetSubTensorDim0(0);

    uint64_t *subListOffsetL3 = subListOffsetL3Tensor.GetSubTensorDim0(qidx);
    int64_t *idResult = idResultTensor.GetSubTensorDim0(qidx);
    uint32_t *opSize = opSizeTensor.GetSubTensorDim0(qidx);
    std::fill_n(subListOffsetL3, l3SegNum_, 0);
    std::fill_n(idResult, l3SegNum_, 0);
    std::fill_n(opSize, l3SegNum_, l3SegSize_);

    int segIdx = 0;
    bool flagFinish = false;
    for (int subIdx = 0; subIdx < k_ && !flagFinish; subIdx++) {
        int64_t subListId =  p[subIdx].second;
        int subSegs = subListSegNum[subListId];
        uint64_t subListOffsetBase = subListOffset[subListId];
        uint64_t* subListIndicesOffsetBase = reinterpret_cast<uint64_t*>(subListIndicesOffset[subListId]);
        for (int subSegIdx = 0; subSegIdx < subSegs; ++subSegIdx, subListOffsetBase += pageShapedDataOffsetStep_,
            subListIndicesOffsetBase += l3SegSize_) {
            subListOffsetL3[segIdx] = subListOffsetBase;
            idResult[segIdx] = reinterpret_cast<int64_t>(subListIndicesOffsetBase);

            if (subSegIdx == (subSegs - 1)) {
                opSize[segIdx] = subListSizes[subListId];
            }
            ++segIdx;
            if (segIdx == l3SegNum_) {
                flagFinish = true;
                break;
            }
        }
    }
    if (segIdx < l3SegNum_) {
        std::fill_n(opSize + segIdx, l3SegNum_ - segIdx, 0);
    }
}

REGISTER_CPU_KERNEL(TOPK_IVFSQT_L2, TopkIvfsqtL2CpuKernel);
} // namespace aicpu