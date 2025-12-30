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

#include "ivf_sp_topk_l1_kernels.h"
#include <algorithm>
#include "utils/cpu_kernel_utils.h"
#include "utils/kernel_tensor.h"
#include "utils/kernel_utils.h"
#include "utils/kernel_shared_def.h"

namespace {
    const char *IVF_SP_TOPK_L1 = "IvfSpTopkL1";
    const uint32_t THREAD_CNT = 6;
}

namespace aicpu {
    uint32_t IvfSpTopkL1CpuKernel::Compute(CpuKernelContext &ctx)
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
        uint32_t core = std::min({CpuKernelUtils::GetCPUNum(ctx), static_cast<uint32_t>(nq_), THREAD_CNT});
        CpuKernelUtils::ParallelFor(ctx, nq_, nq_ / core, computeFunc);
#endif

        return 0;
    }


    uint32_t IvfSpTopkL1CpuKernel::GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs) const
    {
        KERNEL_LOG_INFO("TopkIvfSpL1CpuKernel GetInOutAndCheck begin");

        inputs.indists = ctx.Input(INPUT_NUM0);
        inputs.size = ctx.Input(INPUT_NUM1);
        inputs.opflag = ctx.Input(INPUT_NUM2);
        inputs.attr = ctx.Input(INPUT_NUM3);
        outputs.outdists = ctx.Output(INPUT_NUM0);
        outputs.outlabels = ctx.Output(INPUT_NUM1);

        KERNEL_CHECK_NULLPTR(inputs.indists, KERNEL_STATUS_PARAM_INVALID, "Get input[0], name[indists] failed");
        KERNEL_CHECK_NULLPTR(inputs.size, KERNEL_STATUS_PARAM_INVALID, "Get input[1], name[size] failed");
        KERNEL_CHECK_NULLPTR(inputs.opflag, KERNEL_STATUS_PARAM_INVALID, "Get input[2], name[opflag] failed");
        KERNEL_CHECK_NULLPTR(inputs.attr, KERNEL_STATUS_PARAM_INVALID, "Get input[3], name[attr] failed");
        KERNEL_CHECK_NULLPTR(outputs.outdists, KERNEL_STATUS_PARAM_INVALID, "Get output[0], name[outdists] failed");
        KERNEL_CHECK_NULLPTR(outputs.outlabels, KERNEL_STATUS_PARAM_INVALID, "Get output[1], name[outlabels] failed");

        KERNEL_LOG_INFO("Shape of input[0][indists] is %s",
                        ShapeToString(inputs.indists->GetTensorShape()->GetDimSizes()).c_str());
        KERNEL_LOG_INFO("Shape of input[1][size] is %s",
                        ShapeToString(inputs.size->GetTensorShape()->GetDimSizes()).c_str());
        KERNEL_LOG_INFO("Shape of input[2][opflag] is %s",
                        ShapeToString(inputs.opflag->GetTensorShape()->GetDimSizes()).c_str());
        KERNEL_LOG_INFO("Shape of input[3][attr] is %s",
                        ShapeToString(inputs.attr->GetTensorShape()->GetDimSizes()).c_str());

        return KERNEL_STATUS_OK;
    }

    uint32_t IvfSpTopkL1CpuKernel::CheckInputShapes(const Inputs &inputs)
    {
        KERNEL_LOG_INFO("TopkIvfSpL1CpuKernel CheckInputShapes begin");

        auto shapeIndists = inputs.indists->GetTensorShape();
        auto shapeSize = inputs.size->GetTensorShape();
        auto shapeOpflag = inputs.opflag->GetTensorShape();
        auto shapeAttr = inputs.attr->GetTensorShape();

        KERNEL_CHECK_TRUE(shapeIndists->GetDims() == INPUT_NUM2, KERNEL_STATUS_PARAM_INVALID,
                          "Dims of input[0][indists] must be 2");
        KERNEL_CHECK_TRUE(shapeSize->GetDims() == INPUT_NUM2, KERNEL_STATUS_PARAM_INVALID,
                          "Dims of input[0][size] must be 2");
        KERNEL_CHECK_TRUE(shapeOpflag->GetDims() == INPUT_NUM2, KERNEL_STATUS_PARAM_INVALID,
                          "Dims of input[0][opflag] must be 2");
        KERNEL_CHECK_TRUE(shapeAttr->GetDims() == INPUT_NUM1, KERNEL_STATUS_PARAM_INVALID,
                          "Dims of input[0][attr] must be 1");

        auto nq0 = shapeIndists->GetDimSize(INPUT_NUM0);
        nq_ = nq0;

        auto coreNum0 = shapeSize->GetDimSize(INPUT_NUM0);

        coreNum_ = coreNum0;

        flagSize_ = shapeOpflag->GetDimSize(INPUT_NUM1);

        auto attrCount = shapeAttr->GetDimSize(INPUT_NUM0);
        KERNEL_CHECK_TRUE(attrCount == TOPK_IVFSP_L1_ATTR_IDX_COUNT, KERNEL_STATUS_PARAM_INVALID,
                          "Num of attrs must be %d",
                          TOPK_IVFSP_L1_ATTR_IDX_COUNT);

        auto attr = static_cast<int64_t *>(inputs.attr->GetData());
        asc_ = *(attr + TOPK_IVFSP_L1_ATTR_ASC_IDX);
        k_ = *(attr + TOPK_IVFSP_L1_ATTR_K_IDX);
        quickTopk_ = *(attr + TOPK_IVFSP_L1_ATTR_QUICK_HEAP);
        blockNum_ = 1;
        pageIdx_ = 0;

        KERNEL_CHECK_TRUE(k_ > 0 && asc_ >= 0, KERNEL_STATUS_PARAM_INVALID,
                          "Value of asc, k must ge 0");

        return KERNEL_STATUS_OK;
    }

    void IvfSpTopkL1CpuKernel::UpdateOutputsShape(Outputs &outputs)
    {
        KERNEL_LOG_INFO("TopkIvfSpL1CpuKernel UpdateOutputsShape begin");

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

    template<typename T, typename C>
    void IvfSpTopkL1CpuKernel::DoCompute(size_t start, size_t end, const Inputs &inputs, Outputs &outputs, C &&cmp)
    {
        KernelTensor<float16_t> indists(inputs.indists);
        KernelTensor<uint32_t> size(inputs.size);
        KernelTensor<uint16_t> opflag(inputs.opflag);

        KernelTensor<float16_t> outdists(outputs.outdists);
        KernelTensor<T> outlabels(outputs.outlabels);

        for (int64_t i = 0; i < blockNum_; i++) {
            auto flagPtr = opflag.GetSubTensorDim0(i);
            for (int64_t j = 0; j < coreNum_; j++) {
                WAITING_FLAG_READY(*(flagPtr + j * flagSize_), TIMEOUT_CHECK_TICK, TIMEOUT_MS);
            }

            for (size_t j = start; j < end; j++) {
                ComputeBlock<T, C>(j, i, indists, size, outdists, outlabels, cmp);
            }
        }
    }

    template<typename T>
    void IvfSpTopkL1CpuKernel::InitTopkHeap(Outputs &outputs) const
    {
        uint16_t *outdists = static_cast<uint16_t *>(outputs.outdists->GetData());
        T *outlabels = static_cast<T *>(outputs.outlabels->GetData());
        FillDefault(outdists, outlabels, asc_, nq_, k_);
    }

    template<typename T, typename C>
    void IvfSpTopkL1CpuKernel::ComputeBlock(size_t n, int64_t blockIdx, KernelTensor<float16_t> &indistsTensor,
                                            KernelTensor<uint32_t> &sizeTensor, KernelTensor<float16_t> &outdistsTensor,
                                            KernelTensor<T> &outlabelsTensor, C &&cmp)
    {
        float16_t *indists = indistsTensor.GetSubTensorDim0(n);
        uint32_t *size = sizeTensor.GetSubTensorDim0(blockIdx);
        float16_t *outdists = outdistsTensor.GetSubTensorDim0(n);
        T *outlabel = outlabelsTensor.GetSubTensorDim0(n);

        int64_t ntotal = static_cast<int64_t>(*size);
        int64_t idx = 0;

        // process tail data
        while (idx < ntotal) {
            if (cmp(outdists[0], indists[idx])) {
                UpdateHeap<T, C>(outdists, outlabel, k_, indists[idx], idx, cmp);
            }
            ++idx;
        }
    }

    template<typename T, typename C>
    void IvfSpTopkL1CpuKernel::UpdateHeap(float16_t *dists, T *label, int64_t len, float16_t pushDist, int64_t index,
                                          C &&cmp)
    {
        size_t i = UpdateHeapImpl(dists, label, len, pushDist, cmp);
        dists[i] = pushDist;
        label[i] = index;
    }

    REGISTER_CPU_KERNEL(IVF_SP_TOPK_L1, IvfSpTopkL1CpuKernel
    );
} // namespace aicpu
