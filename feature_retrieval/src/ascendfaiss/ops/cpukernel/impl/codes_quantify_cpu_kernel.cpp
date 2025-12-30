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


#include "codes_quantify_cpu_kernel.h"

#include <algorithm>
#include <string>

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "kernel_tensor.h"
#include "kernel_utils.h"
#include "kernel_shared_def.h"

namespace {
const char *g_codesQuantify = "CodesQuantify";
const int INT8_LOWER_BOUND = -128;
const int INT8_UPPER_BOUND = 127;
const int UINT8_UPPER_BOUND = 255;
}

namespace aicpu {
uint32_t CodesQuantifyCpuKernel::Compute(CpuKernelContext &ctx)
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

    KernelTensor<float16_t> codes(inputs.codes);
    KernelTensor<int8_t> codesQ(outputs.codesQ);

    auto computeFunc = [&codes, &codesQ, this](size_t start, size_t end) { DoCompute(start, end, codes, codesQ); };
#ifdef AICPU_UTEST
    computeFunc(0, nq_);
#else
    uint32_t core = std::min({ CpuKernelUtils::GetCPUNum(ctx), static_cast<uint32_t>(nq_) });
    CpuKernelUtils::ParallelFor(ctx, nq_, nq_ / core, computeFunc);
#endif

    return KERNEL_STATUS_OK;
}

uint32_t CodesQuantifyCpuKernel::GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs) const
{
    KERNEL_LOG_INFO("CodesQuantifyCpuKernel GetInOutAndCheck begin");

    inputs.codes = ctx.Input(INPUT_NUM0);
    inputs.attrs = ctx.Input(INPUT_NUM1);
    outputs.codesQ = ctx.Output(INPUT_NUM0);

    KERNEL_CHECK_NULLPTR(inputs.codes, KERNEL_STATUS_PARAM_INVALID, "Get input[0], name[codes] failed");
    KERNEL_CHECK_NULLPTR(inputs.attrs, KERNEL_STATUS_PARAM_INVALID, "Get input[1], name[attrs] failed");
    KERNEL_CHECK_NULLPTR(outputs.codesQ, KERNEL_STATUS_PARAM_INVALID, "Get output[0], name[codesQ] failed");

    KERNEL_LOG_INFO("Shape of input[0][codes] is %s",
        ShapeToString(inputs.codes->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[1][attrs] is %s",
        ShapeToString(inputs.attrs->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of output[0][codesQ] is %s",
        ShapeToString(outputs.codesQ->GetTensorShape()->GetDimSizes()).c_str());

    return KERNEL_STATUS_OK;
}

uint32_t CodesQuantifyCpuKernel::CheckInOutShapes(const Inputs &inputs, const Outputs &outputs)
{
    KERNEL_LOG_INFO("CodesQuantifyCpuKernel CheckInputShapes begin");

    auto shapeCodes = inputs.codes->GetTensorShape();
    auto shapeAttrs = inputs.attrs->GetTensorShape();
    auto shapeCodesQ = outputs.codesQ->GetTensorShape();

    KERNEL_CHECK_TRUE(shapeCodes->GetDims() == INPUT_NUM2, KERNEL_STATUS_PARAM_INVALID,
        "Dims of input[0][inshaped] must be 2");
    KERNEL_CHECK_TRUE(shapeAttrs->GetDims() == INPUT_NUM1, KERNEL_STATUS_PARAM_INVALID,
        "Dims of input[1][attrs] must be 1");
    KERNEL_CHECK_TRUE(shapeCodesQ->GetDims() == INPUT_NUM2, KERNEL_STATUS_PARAM_INVALID,
        "Dims of output[0][outvecs] must be 2");

    auto nq0 = shapeCodes->GetDimSize(INPUT_NUM0);
    auto nq1 = shapeCodesQ->GetDimSize(INPUT_NUM0);
    KERNEL_CHECK_TRUE(nq0 == nq1, KERNEL_STATUS_PARAM_INVALID, "Nq of inputs and outputs must be same");
    nq_ = nq0;

    auto dim0 = shapeCodes->GetDimSize(INPUT_NUM1);
    auto dim1 = shapeCodesQ->GetDimSize(INPUT_NUM1);
    KERNEL_CHECK_TRUE(dim0 == dim1, KERNEL_STATUS_PARAM_INVALID, "Dim of inputs and outputs must be same");
    dim_ = dim0;

    auto attrCount = shapeAttrs->GetDimSize(INPUT_NUM0);
    KERNEL_CHECK_TRUE(attrCount == CODES_QUANTIFY_ATTR_IDX_COUNT, KERNEL_STATUS_PARAM_INVALID,
        "Num of attrs must be %d", CODES_QUANTIFY_ATTR_IDX_COUNT);

    auto attr = static_cast<float16_t *>(inputs.attrs->GetData());
    qMax_ = *(attr + CODES_QUANTIFY_ATTR_QMAX_IDX);
    qMin_ = *(attr + CODES_QUANTIFY_ATTR_QMIN_IDX);

    return KERNEL_STATUS_OK;
}

void CodesQuantifyCpuKernel::DoCompute(size_t start, size_t end, KernelTensor<float16_t> &codesTensor,
    KernelTensor<int8_t> &codesQTensor)
{
    for (size_t i = start; i < end; ++i) {
        float16_t *codes = codesTensor.GetSubTensorDim0(i);
        int8_t *codesQ = codesQTensor.GetSubTensorDim0(i);
        for (int64_t d = 0; d < dim_; ++d) {
            if (codes[d] < qMin_) {
                codesQ[d] = static_cast<int8_t>(INT8_LOWER_BOUND);
            } else if (codes[d] > qMax_) {
                codesQ[d] = static_cast<int8_t>(INT8_UPPER_BOUND);
            } else {
                // 1e-6 is a minimum value to prevent division by 0
                codesQ[d] = static_cast<int8_t>(((codes[d] - qMin_) / (qMax_ - qMin_ + 1e-6) * UINT8_UPPER_BOUND) +
                    INT8_LOWER_BOUND);
            }
        }
    }
}

REGISTER_CPU_KERNEL(g_codesQuantify, CodesQuantifyCpuKernel);
} // namespace aicpu