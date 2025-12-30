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


#include "transdata_shaped_cpu_kernel.h"

#include <algorithm>
#include <string>

#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "kernel_tensor.h"
#include "kernel_utils.h"
#include "kernel_shared_def.h"
#include "securec.h"

namespace {
const char* TRANSDATA_SHAPED = "TransdataShaped";
}

namespace aicpu {
uint32_t TransdataShapedCpuKernel::Compute(CpuKernelContext &ctx)
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

    DataType dataType = inputs.invecs->GetDataType();

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

uint32_t TransdataShapedCpuKernel::GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs) const
{
    KERNEL_LOG_INFO("TransdataShapedCpuKernel GetInOutAndCheck begin");

    inputs.invecs = ctx.Input(INPUT_NUM0);
    inputs.attrs = ctx.Input(INPUT_NUM1);
    outputs.outshaped = ctx.Output(INPUT_NUM0);

    KERNEL_CHECK_NULLPTR(inputs.invecs, KERNEL_STATUS_PARAM_INVALID, "Get input[0], name[indists] failed");
    KERNEL_CHECK_NULLPTR(inputs.attrs, KERNEL_STATUS_PARAM_INVALID, "Get input[1], name[attrs] failed");
    KERNEL_CHECK_NULLPTR(outputs.outshaped, KERNEL_STATUS_PARAM_INVALID, "Get output[0], name[outshaped] failed");

    KERNEL_LOG_INFO("Shape of input[0][invecs] is %s",
        ShapeToString(inputs.invecs->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[1][attrs] is %s",
        ShapeToString(inputs.attrs->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of output[0][outshaped] is %s",
        ShapeToString(outputs.outshaped->GetTensorShape()->GetDimSizes()).c_str());

    return KERNEL_STATUS_OK;
}

uint32_t TransdataShapedCpuKernel::CheckInOutShapes(const Inputs &inputs, const Outputs &outputs)
{
    KERNEL_LOG_INFO("TransdataShapedCpuKernel CheckInputShapes begin");

    auto shapeInvecs = inputs.invecs->GetTensorShape();
    auto shapeAttrs = inputs.attrs->GetTensorShape();
    auto shapeOutshaped = outputs.outshaped->GetTensorShape();

    KERNEL_CHECK_TRUE(shapeInvecs->GetDims() == INPUT_NUM2,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][invecs] must be 2");
    KERNEL_CHECK_TRUE(shapeAttrs->GetDims() == INPUT_NUM1,
                      KERNEL_STATUS_PARAM_INVALID, "Dims of input[1][attrs] must be 1");
    KERNEL_CHECK_TRUE(shapeOutshaped->GetDims() == INPUT_NUM4,
        KERNEL_STATUS_PARAM_INVALID, "Dims of output[0][outshaped] must be 4");

    nq_ = shapeInvecs->GetDimSize(INPUT_NUM0);
    dim_ = shapeInvecs->GetDimSize(INPUT_NUM1);

    dimAlignedNum_ = shapeOutshaped->GetDimSize(INPUT_NUM1);
    nqAlign_ = shapeOutshaped->GetDimSize(INPUT_NUM2);
    dimAlign_ = shapeOutshaped->GetDimSize(INPUT_NUM3);

    auto attrCount = shapeAttrs->GetDimSize(INPUT_NUM0);
    KERNEL_CHECK_TRUE(attrCount == TRANSDATA_SHAPED_ATTR_IDX_COUNT,
        KERNEL_STATUS_PARAM_INVALID, "Num of attrs must be %d", TRANSDATA_SHAPED_ATTR_IDX_COUNT);

    auto attr = static_cast<int64_t *>(inputs.attrs->GetData());
    ntotal_ = *(attr + TRANSDATA_SHAPED_ATTR_NTOTAL_IDX);

    return KERNEL_STATUS_OK;
}

template<typename T>
void TransdataShapedCpuKernel::DoCompute(size_t start, size_t end, const Inputs &inputs, Outputs &outputs)
{
    KernelTensor<T> invecsTensor(inputs.invecs);
    KernelTensor<T> outshapedTensor(outputs.outshaped);
    for (size_t i = start; i < end; ++i) {
        T *invecs = invecsTensor.GetSubTensorDim0(i);
        int64_t total = ntotal_ + static_cast<int64_t>(i);
        int64_t idx = total / nqAlign_;
        int64_t offset = total % nqAlign_;
        T *outshaped = outshapedTensor.GetSubTensorDim3(idx, 0, 0, 0) + offset * dimAlign_;
        for (int64_t j = 0; j < dimAlignedNum_; j++) {
            auto err = memcpy_s(outshaped + j * dimAlign_ * nqAlign_, dimAlign_ * sizeof(T),
                                invecs + j * dimAlign_, dimAlign_ * sizeof(T));
            if (err != EOK) {
                KERNEL_LOG_ERROR("Copy data shaped error %d", err);
            }
        }
    }
}

REGISTER_CPU_KERNEL(TRANSDATA_SHAPED, TransdataShapedCpuKernel);
} // namespace aicpu
