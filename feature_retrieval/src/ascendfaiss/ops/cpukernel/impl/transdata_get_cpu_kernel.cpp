/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

#include "transdata_get_cpu_kernel.h"

#include <algorithm>
#include <string>

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "kernel_tensor.h"
#include "kernel_utils.h"
#include "kernel_shared_def.h"
#include "securec.h"

namespace {
const char* TRANSDATA_GET = "TransdataGet";
}

namespace aicpu {
uint32_t TransdataGetCpuKernel::Compute(CpuKernelContext &ctx)
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

    auto computeFunc = [this, &inputs, &outputs] (size_t start, size_t end) {
        DoCompute(start, end, inputs, outputs);
    };
#ifdef AICPU_UTEST
    computeFunc(0, nq_);
#else
    uint32_t core = std::min({CpuKernelUtils::GetCPUNum(ctx), static_cast<uint32_t>(nq_)});
    CpuKernelUtils::ParallelFor(ctx, nq_, nq_ / core, computeFunc);
#endif

    return KERNEL_STATUS_OK;
}

uint32_t TransdataGetCpuKernel::GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs) const
{
    KERNEL_LOG_INFO("TransdataGetCpuKernel GetInOutAndCheck begin");

    inputs.inshaped = ctx.Input(INPUT_NUM0);
    inputs.attrs = ctx.Input(INPUT_NUM1);
    outputs.outvecs = ctx.Output(INPUT_NUM0);

    KERNEL_CHECK_NULLPTR(inputs.inshaped, KERNEL_STATUS_PARAM_INVALID, "Get input[0], name[inshaped] failed");
    KERNEL_CHECK_NULLPTR(inputs.attrs, KERNEL_STATUS_PARAM_INVALID, "Get input[1], name[attrs] failed");
    KERNEL_CHECK_NULLPTR(outputs.outvecs, KERNEL_STATUS_PARAM_INVALID, "Get output[0], name[outvecs] failed");

    KERNEL_LOG_INFO("Shape of input[0][inshaped] is %s",
        ShapeToString(inputs.inshaped->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[1][attrs] is %s",
        ShapeToString(inputs.attrs->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of output[0][outvecs] is %s",
        ShapeToString(outputs.outvecs->GetTensorShape()->GetDimSizes()).c_str());

    return KERNEL_STATUS_OK;
}

uint32_t TransdataGetCpuKernel::CheckInOutShapes(const Inputs &inputs, const Outputs &outputs)
{
    KERNEL_LOG_INFO("TransdataGetCpuKernel CheckInputShapes begin");

    auto shapeInshaped = inputs.inshaped->GetTensorShape();
    auto shapeAttrs = inputs.attrs->GetTensorShape();
    auto shapeOutvecs = outputs.outvecs->GetTensorShape();

    KERNEL_CHECK_TRUE(shapeInshaped->GetDims() == INPUT_NUM4,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][inshaped] must be 4");
    KERNEL_CHECK_TRUE(shapeAttrs->GetDims() == INPUT_NUM1,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[1][attrs] must be 1");
    KERNEL_CHECK_TRUE(shapeOutvecs->GetDims() == INPUT_NUM2,
        KERNEL_STATUS_PARAM_INVALID, "Dims of output[0][outvecs] must be 2");

    nq_ = shapeOutvecs->GetDimSize(INPUT_NUM0);
    dim_ = shapeOutvecs->GetDimSize(INPUT_NUM1);

    dimAlignedNum_ = shapeInshaped->GetDimSize(INPUT_NUM1);
    nqAlign_ = shapeInshaped->GetDimSize(INPUT_NUM2);
    dimAlign_ = shapeInshaped->GetDimSize(INPUT_NUM3);

    return KERNEL_STATUS_OK;
}

void TransdataGetCpuKernel::DoCompute(size_t start, size_t end, const Inputs &inputs, Outputs &outputs)
{
    KernelTensor<float16_t> inshapedTensor(inputs.inshaped);
    KernelTensor<float16_t> outvecsTensor(outputs.outvecs);
    for (size_t i = start; i < end; ++i) {
        float16_t *outvecs = outvecsTensor.GetSubTensorDim0(0);
        float16_t *inshaped = inshapedTensor.GetSubTensorDim0(0);
        auto attr = static_cast<uint32_t *>(inputs.attrs->GetData());
        offset_ = *(attr + i);
        float16_t *inputData = inshaped + offset_ / nqAlign_ * dimAlignedNum_ * (nqAlign_ * dimAlign_) +
            offset_ % nqAlign_ * dimAlign_;

        for (int64_t j = 0; j < dimAlignedNum_; j++) {
            size_t getOffset = i * static_cast<size_t>(dim_) + j * dimAlign_;
            size_t cpyNum = (j == dimAlignedNum_ - 1) ? (dim_ - j * dimAlign_) : dimAlign_;
            auto err = memcpy_s(outvecs + getOffset, cpyNum * sizeof(float16_t),
                                inputData + j * dimAlign_ * nqAlign_, cpyNum * sizeof(float16_t));
            if (err != EOK) {
                KERNEL_LOG_ERROR("Copy data shaped error %d", err);
            }
        }
    }
}

REGISTER_CPU_KERNEL(TRANSDATA_GET, TransdataGetCpuKernel);
} // namespace aicpu