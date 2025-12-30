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


#include "transdata_raw_sp_cpu_kernel.h"

#include <algorithm>
#include <string>

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "kernel_tensor.h"
#include "kernel_utils.h"
#include "kernel_shared_def.h"
#include "securec.h"

namespace {
const char* TRANSDATA_RAW = "TransdataRawSp";
}

namespace aicpu {
uint32_t TransdataRawSpCpuKernel::Compute(CpuKernelContext &ctx)
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

    DataType dataType = inputs.inshaped->GetDataType();

    auto computeFunc = [&](size_t start, size_t end) {
        switch (dataType) {
            case DT_FLOAT16:
                DoCompute<float16_t>(start, end, inputs, outputs);
                break;
            case DT_INT8:
            case DT_UINT8:
                DoCompute<uint8_t>(start, end, inputs, outputs);
                break;
            default:
                KERNEL_LOG_ERROR("Invalid datatype");
        }
    };
#ifdef AICPU_UTEST
    computeFunc(0, nq_);
#else
    uint32_t core = std::min({CpuKernelUtils::GetCPUNum(ctx), static_cast<uint32_t>(nq_)});
    CpuKernelUtils::ParallelFor(ctx, nq_, nq_ / core, computeFunc);
#endif

    return KERNEL_STATUS_OK;
}

uint32_t TransdataRawSpCpuKernel::GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs) const
{
    KERNEL_LOG_INFO("TransdataRawSpCpuKernel GetInOutAndCheck begin");

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

uint32_t TransdataRawSpCpuKernel::CheckInOutShapes(const Inputs &inputs, const Outputs &outputs)
{
    KERNEL_LOG_INFO("TransdataRawSpCpuKernel CheckInputShapes begin");

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

    auto attrCount = shapeAttrs->GetDimSize(INPUT_NUM0);
    KERNEL_CHECK_TRUE(attrCount == TRANSDATA_RAW_SP_ATTR_IDX_COUNT,
        KERNEL_STATUS_PARAM_INVALID, "Num of attrs must be %d", TRANSDATA_RAW_SP_ATTR_IDX_COUNT);

    auto attr = static_cast<int64_t *>(inputs.attrs->GetData());
    offset_ = *(attr + TRANSDATA_RAW_SP_ATTR_OFFSET_IDX);

    return KERNEL_STATUS_OK;
}

template<typename T>
void TransdataRawSpCpuKernel::DoCompute(size_t start, size_t end, const Inputs &inputs, Outputs &outputs)
{
    KernelTensor<T> inshapedTensor(inputs.inshaped);
    KernelTensor<T> outvecsTensor(outputs.outvecs);
    for (size_t i = start; i < end; ++i) {
        T *outvecs = outvecsTensor.GetSubTensorDim0(i);
        int64_t total = offset_ + static_cast<int64_t>(i);
        int64_t idx = total / nqAlign_;
        int64_t offset = total % nqAlign_;
        T *inshaped = inshapedTensor.GetSubTensorDim3(idx, 0, 0, 0) + offset * dimAlign_;
        for (int64_t j = 0; j < dimAlignedNum_; j++) {
            auto err = memcpy_s(outvecs + j * dimAlign_, dimAlign_ * sizeof(T),
                                inshaped + j * dimAlign_ * nqAlign_, dimAlign_ * sizeof(T));
            if (err != EOK) {
                KERNEL_LOG_ERROR("Copy data shaped error %d", err);
            }
        }
    }
}

REGISTER_CPU_KERNEL(TRANSDATA_RAW, TransdataRawSpCpuKernel);
} // namespace aicpu