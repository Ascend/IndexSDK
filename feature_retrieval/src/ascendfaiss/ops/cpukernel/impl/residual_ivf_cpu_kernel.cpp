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


#include "residual_ivf_cpu_kernel.h"

#include <algorithm>
#include <string>

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "kernel_tensor.h"
#include "kernel_utils.h"
#include "kernel_shared_def.h"

namespace {
const char *RESIDUAL_IVF = "ResidualIvf";
}

namespace aicpu {
uint32_t ResidualIvfCpuKernel::Compute(CpuKernelContext &ctx)
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

    KernelTensor<float16_t> query(inputs.query);
    KernelTensor<float16_t> coarseCentroids(inputs.coarseCentroids);
    KernelTensor<uint64_t> l1TopNprobeIndices(inputs.l1TopNprobeIndices);
    KernelTensor<float16_t> residuals(outputs.residuals);

    auto computeFunc = [&](size_t start, size_t end) {
        DoCompute(start, end, query, coarseCentroids, l1TopNprobeIndices, residuals);
    };
#ifdef AICPU_UTEST
    computeFunc(0, nq_);
#else
    uint32_t core = std::min({CpuKernelUtils::GetCPUNum(ctx), static_cast<uint32_t>(nq_)});
    CpuKernelUtils::ParallelFor(ctx, nq_, nq_ / core, computeFunc);
#endif

    return KERNEL_STATUS_OK;
}

uint32_t ResidualIvfCpuKernel::GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs) const
{
    KERNEL_LOG_INFO("ResidualIvfCpuKernel GetInOutAndCheck begin");

    inputs.query = ctx.Input(INPUT_NUM0);
    inputs.coarseCentroids = ctx.Input(INPUT_NUM1);
    inputs.l1TopNprobeIndices = ctx.Input(INPUT_NUM2);
    outputs.residuals = ctx.Output(INPUT_NUM0);

    KERNEL_CHECK_NULLPTR(inputs.query,
        KERNEL_STATUS_PARAM_INVALID, "Get input[0], name[query] failed");
    KERNEL_CHECK_NULLPTR(inputs.coarseCentroids,
        KERNEL_STATUS_PARAM_INVALID, "Get input[1], name[coarseCentroids] failed");
    KERNEL_CHECK_NULLPTR(inputs.l1TopNprobeIndices,
        KERNEL_STATUS_PARAM_INVALID, "Get input[2], name[l1TopNprobeIndices] failed");
    KERNEL_CHECK_NULLPTR(outputs.residuals,
        KERNEL_STATUS_PARAM_INVALID, "Get output[0], name[residuals] failed");

    return KERNEL_STATUS_OK;
}

uint32_t ResidualIvfCpuKernel::CheckInOutShapes(const Inputs &inputs, const Outputs &outputs)
{
    KERNEL_LOG_INFO("ResidualIvfCpuKernel CheckInputShapes begin");

    auto shapeQuery = inputs.query->GetTensorShape(); // nq, dim
    auto shapeCoarseCentroids = inputs.coarseCentroids->GetTensorShape(); // nlist, dim
    auto shapeL1TopNprobeIndices = inputs.l1TopNprobeIndices->GetTensorShape(); // nq, nprobe
    auto shapeResiduals = outputs.residuals->GetTensorShape(); // nq, nprobe, dim

    KERNEL_CHECK_TRUE(shapeQuery->GetDims() == INPUT_NUM2,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][query] must be 2");
    KERNEL_CHECK_TRUE(shapeCoarseCentroids->GetDims() == INPUT_NUM2,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[1][shapeCoarseCentroids] must be 2");
    KERNEL_CHECK_TRUE(shapeL1TopNprobeIndices->GetDims() == INPUT_NUM2,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[2][l1TopNprobeIndices] must be 2");
    KERNEL_CHECK_TRUE(shapeResiduals->GetDims() == INPUT_NUM3,
        KERNEL_STATUS_PARAM_INVALID, "Dims of output[0][residuals] must be 3");

    auto nq0 = shapeQuery->GetDimSize(INPUT_NUM0);
    auto nq1 = shapeL1TopNprobeIndices->GetDimSize(INPUT_NUM0);
    auto nq2 = shapeResiduals->GetDimSize(INPUT_NUM0);
    KERNEL_CHECK_TRUE(nq0 == nq1 && nq0 == nq2,
        KERNEL_STATUS_PARAM_INVALID, "Nq of inputs and outputs must be same");
    nq_ = nq0;

    auto dim0 = shapeQuery->GetDimSize(INPUT_NUM1);
    auto dim1 = shapeCoarseCentroids->GetDimSize(INPUT_NUM1);
    auto dim2 = shapeResiduals->GetDimSize(INPUT_NUM2);
    KERNEL_CHECK_TRUE(dim0 == dim1 && dim0 == dim2,
        KERNEL_STATUS_PARAM_INVALID, "Dim of inputs and outputs must be same");
    dim_ = dim0;

    auto nprobe0 = shapeL1TopNprobeIndices->GetDimSize(INPUT_NUM1);
    auto nprobe1 = shapeResiduals->GetDimSize(INPUT_NUM1);
    KERNEL_CHECK_TRUE(nprobe0 == nprobe1,
        KERNEL_STATUS_PARAM_INVALID, "Nprobe of inputs and outputs must be same");
    nprobe_ = nprobe0;

    return KERNEL_STATUS_OK;
}

void ResidualIvfCpuKernel::DoCompute(size_t start, size_t end,
                                     KernelTensor<float16_t> &queryTensor,
                                     KernelTensor<float16_t> &coarseCentroidsTensor,
                                     KernelTensor<uint64_t> &l1TopNprobeIndicesTensor,
                                     KernelTensor<float16_t> &residualsTensor)
{
    for (size_t qidx = start; qidx < end; ++qidx) {
        for (int64_t pidx = 0; pidx < nprobe_; ++pidx) {
            uint64_t *listId = l1TopNprobeIndicesTensor.GetSubTensorDim1(qidx, pidx);
            for (int64_t didx = 0; didx < dim_; ++didx) {
                float16_t *residuals = residualsTensor.GetSubTensorDim2(qidx, pidx, didx);
                float16_t *query = queryTensor.GetSubTensorDim1(qidx, didx);
                float16_t *coarseCentroids = coarseCentroidsTensor.GetSubTensorDim1(*listId, didx);
                *residuals = *query - *coarseCentroids;
            }
        }
    }
}

REGISTER_CPU_KERNEL(RESIDUAL_IVF, ResidualIvfCpuKernel);
} // namespace aicpu