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


#include "vec_l2sqr_sp_cpu_kernel.h"

#include <algorithm>
#include <string>

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "kernel_tensor.h"
#include "kernel_utils.h"
#include "kernel_shared_def.h"

namespace {
const char* VEC_L2SQR = "VecL2SqrSp";
const uint32_t THREAD_CNT = 6;
}

namespace aicpu {
uint32_t VecL2sqrSpCpuKernel::Compute(CpuKernelContext &ctx)
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

    DataType inputType = inputs.invecs->GetDataType();
    DataType outputType = outputs.outl2sqr->GetDataType();

    auto computeFunc = [&](size_t start, size_t end) {
        if (inputType == DT_FLOAT16 && outputType == DT_FLOAT16) {
            DoCompute<float16_t, float16_t>(start, end, inputs, outputs);
        } else if (inputType == DT_FLOAT16 && outputType == DT_FLOAT) {
            DoCompute<float16_t, float>(start, end, inputs, outputs);
        } else if (inputType == DT_INT8 && outputType == DT_INT32) {
            DoCompute<int8_t, int>(start, end, inputs, outputs);
        } else {
            KERNEL_LOG_ERROR("Invalid datatype");
        }
    };

#ifdef AICPU_UTEST
    computeFunc(0, nq_);
#else
    uint32_t core = std::min({CpuKernelUtils::GetCPUNum(ctx), static_cast<uint32_t>(nq_), THREAD_CNT});
    CpuKernelUtils::ParallelFor(ctx, nq_, nq_ / core, computeFunc);
#endif

    return KERNEL_STATUS_OK;
}

uint32_t VecL2sqrSpCpuKernel::GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs) const
{
    KERNEL_LOG_INFO("VecL2sqrSpCpuKernel GetInOutAndCheck begin");

    inputs.invecs = ctx.Input(INPUT_NUM0);
    outputs.outl2sqr = ctx.Output(INPUT_NUM0);

    KERNEL_CHECK_NULLPTR(inputs.invecs, KERNEL_STATUS_PARAM_INVALID, "Get input[0], name[invecs] failed");
    KERNEL_CHECK_NULLPTR(outputs.outl2sqr, KERNEL_STATUS_PARAM_INVALID, "Get output[0], name[outl2sqr] failed");

    KERNEL_LOG_INFO("Shape of input[0][invecs] is %s",
        ShapeToString(inputs.invecs->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of output[0][outl2sqr] is %s",
        ShapeToString(outputs.outl2sqr->GetTensorShape()->GetDimSizes()).c_str());

    return KERNEL_STATUS_OK;
}

uint32_t VecL2sqrSpCpuKernel::CheckInOutShapes(const Inputs &inputs, const Outputs &outputs)
{
    KERNEL_LOG_INFO("VecL2sqrSpCpuKernel CheckInputShapes begin");

    auto shapeInvecs = inputs.invecs->GetTensorShape();
    auto shapeOutl2sqr = outputs.outl2sqr->GetTensorShape();

    KERNEL_CHECK_TRUE(shapeInvecs->GetDims() == INPUT_NUM2,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][invecs] must be 2");
    KERNEL_CHECK_TRUE(shapeOutl2sqr->GetDims() == INPUT_NUM1 || shapeOutl2sqr->GetDims() == INPUT_NUM2,
        KERNEL_STATUS_PARAM_INVALID, "Dims of output[0][outl2sqr] must be 1 or 2");

    auto nq0 = shapeInvecs->GetDimSize(INPUT_NUM0);
    auto nq1 = shapeOutl2sqr->GetDimSize(INPUT_NUM0);
    KERNEL_CHECK_TRUE(nq0 == nq1, KERNEL_STATUS_PARAM_INVALID, "Nq of inputs and outputs must be same");
    nq_ = nq0;

    dim_ = shapeInvecs->GetDimSize(INPUT_NUM1);

    if (shapeOutl2sqr->GetDims() == INPUT_NUM2)
        outputCubeAlign_ = shapeOutl2sqr->GetDimSize(INPUT_NUM1);

    return KERNEL_STATUS_OK;
}

template<typename inputType, typename outputType>
void VecL2sqrSpCpuKernel::DoCompute(size_t start, size_t end, const Inputs &inputs, Outputs &outputs)
{
    KernelTensor<inputType> invecsTensor(inputs.invecs);
    KernelTensor<outputType> outl2sqrTensor(outputs.outl2sqr);
    for (size_t i = start; i < end; ++i) {
        inputType *invecs = invecsTensor.GetSubTensorDim0(i);
        outputType *outdists = outl2sqrTensor.GetSubTensorDim0(i);
        double res = 0.0;
        for (int64_t d = 0; d < dim_; ++d) {
            res += static_cast<outputType>(invecs[d]) * static_cast<outputType>(invecs[d]);
        }
        for (int c = 0; c < outputCubeAlign_; ++c) {
            *(outdists + c) = static_cast<outputType>(res);
        }
    }
}

REGISTER_CPU_KERNEL(VEC_L2SQR, VecL2sqrSpCpuKernel);
} // namespace aicpu