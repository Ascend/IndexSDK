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

#include "transdata_dist_cpu_kernel.h"

#include <algorithm>
#include <string>

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "kernel_tensor.h"
#include "kernel_utils.h"
#include "kernel_shared_def.h"
#include "securec.h"

namespace {
const char* TRANSDATA_DIST = "TransdataDist";
}

namespace aicpu {
uint32_t TransdataDistCpuKernel::Compute(CpuKernelContext &ctx)
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
    computeFunc(0, n);
#else
    uint32_t core = std::min({CpuKernelUtils::GetCPUNum(ctx), static_cast<uint32_t>(n)});
    CpuKernelUtils::ParallelFor(ctx, n, n / core, computeFunc);
#endif

    return KERNEL_STATUS_OK;
}

uint32_t TransdataDistCpuKernel::GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs) const
{
    KERNEL_LOG_INFO("TransdataDistCpuKernel GetInOutAndCheck begin");

    inputs.inshaped = ctx.Input(INPUT_NUM0);
    outputs.outvecs = ctx.Output(INPUT_NUM0);

    KERNEL_CHECK_NULLPTR(inputs.inshaped, KERNEL_STATUS_PARAM_INVALID, "Get input[0], name[inshaped] failed");
    KERNEL_CHECK_NULLPTR(outputs.outvecs, KERNEL_STATUS_PARAM_INVALID, "Get output[0], name[outvecs] failed");

    KERNEL_LOG_INFO("Shape of input[0][inshaped] is %s",
        ShapeToString(inputs.inshaped->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of output[0][outvecs] is %s",
        ShapeToString(outputs.outvecs->GetTensorShape()->GetDimSizes()).c_str());

    return KERNEL_STATUS_OK;
}

uint32_t TransdataDistCpuKernel::CheckInOutShapes(const Inputs &inputs, const Outputs &outputs)
{
    KERNEL_LOG_INFO("TransdataDistCpuKernel CheckInputShapes begin");

    auto shapeInshaped = inputs.inshaped->GetTensorShape();
    auto shapeOutvecs = outputs.outvecs->GetTensorShape();

    KERNEL_CHECK_TRUE(shapeInshaped->GetDims() == INPUT_NUM3,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][inshaped] must be 3");
    KERNEL_CHECK_TRUE(shapeOutvecs->GetDims() == INPUT_NUM2,
        KERNEL_STATUS_PARAM_INVALID, "Dims of output[0][outvecs] must be 2");

    idxSliceNum = shapeInshaped->GetDimSize(INPUT_NUM0);
    n = shapeInshaped->GetDimSize(INPUT_NUM1);
    burstLen = shapeInshaped->GetDimSize(INPUT_NUM2);

    maxNum = shapeOutvecs->GetDimSize(INPUT_NUM1);

    return KERNEL_STATUS_OK;
}

void TransdataDistCpuKernel::DoCompute(size_t start, size_t end, const Inputs &inputs, Outputs &outputs)
{
    KernelTensor<float> inshapedTensor(inputs.inshaped);
    KernelTensor<float> outvecsTensor(outputs.outvecs);

    int idxCopyNum = 0;
    for (int i = 0; i < idxSliceNum; i++) {
        for (int j = start; j < end; j++) {
            float *outvecs = outvecsTensor.GetSubTensorDim0(j);
            float *inshaped = inshapedTensor.GetSubTensorDim1(i, j);
            idxCopyNum = (i == idxSliceNum - 1) ? (maxNum - i * burstLen) : burstLen;
            auto err = memcpy_s(outvecs + i * burstLen, idxCopyNum * sizeof(float),
                inshaped, idxCopyNum * sizeof(float));
            if (err != EOK) {
                KERNEL_LOG_ERROR("memcpy_s data error %d", err);
            }
        }
    }
}

REGISTER_CPU_KERNEL(TRANSDATA_DIST, TransdataDistCpuKernel);
} // namespace aicpu
