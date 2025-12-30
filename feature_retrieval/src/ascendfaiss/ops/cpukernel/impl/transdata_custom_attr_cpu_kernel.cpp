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


#include "transdata_custom_attr_cpu_kernel.h"

#include <algorithm>
#include <string>

#include "DataType.h"
#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "kernel_tensor.h"
#include "kernel_utils.h"
#include "kernel_shared_def.h"
#include "securec.h"

namespace {
const char *TRANSDATA_CUSTOM_ATTR = "TransdataCustomAttr";
}

namespace aicpu {
uint32_t TransdataCustomAttrCpuKernel::Compute(CpuKernelContext &ctx)
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
    auto computeFunc = [inputs, outputs, dataType, this](size_t start, size_t end) {
        switch (dataType) {
            case faiss::ascend::UINT8:
                DoCompute<uint8_t>(start, end, inputs, outputs);
                break;
            default:
                KERNEL_LOG_ERROR("Invalid datatype");
        }
    };
#ifdef AICPU_UTEST
    computeFunc(0, nq_);
#else
    uint32_t core = std::min({ CpuKernelUtils::GetCPUNum(ctx), static_cast<uint32_t>(nq_) });
    CpuKernelUtils::ParallelFor(ctx, nq_, nq_ / core, computeFunc);
#endif

    return KERNEL_STATUS_OK;
}

uint32_t TransdataCustomAttrCpuKernel::GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs,
    Outputs &outputs) const
{
    KERNEL_LOG_INFO("TransdataCustomAttrCpuKernel GetInOutAndCheck begin");

    inputs.invecs = ctx.Input(INPUT_NUM0);
    inputs.attrs = ctx.Input(INPUT_NUM1);
    outputs.outvecs = ctx.Output(INPUT_NUM0);

    KERNEL_CHECK_NULLPTR(inputs.invecs, KERNEL_STATUS_PARAM_INVALID, "Get input[0], name[indists] failed");
    KERNEL_CHECK_NULLPTR(inputs.attrs, KERNEL_STATUS_PARAM_INVALID, "Get input[1], name[attrs] failed");
    KERNEL_CHECK_NULLPTR(outputs.outvecs, KERNEL_STATUS_PARAM_INVALID, "Get output[0], name[outvecs] failed");

    KERNEL_LOG_INFO("Shape of input[0][invecs] is %s",
        ShapeToString(inputs.invecs->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of input[1][attrs] is %s",
        ShapeToString(inputs.attrs->GetTensorShape()->GetDimSizes()).c_str());
    KERNEL_LOG_INFO("Shape of output[0][outvecs] is %s",
        ShapeToString(outputs.outvecs->GetTensorShape()->GetDimSizes()).c_str());

    return KERNEL_STATUS_OK;
}

uint32_t TransdataCustomAttrCpuKernel::CheckInOutShapes(const Inputs &inputs, const Outputs &outputs)
{
    KERNEL_LOG_INFO("RemovedataAttrCpuKernel CheckInputShapes begin");

    auto shapeInvecs = inputs.invecs->GetTensorShape();
    auto shapeAttrs = inputs.attrs->GetTensorShape();
    auto shapeOutvecs = outputs.outvecs->GetTensorShape();

    KERNEL_CHECK_TRUE(shapeInvecs->GetDims() == INPUT_NUM2, KERNEL_STATUS_PARAM_INVALID,
        "Dims of input[0][invecs] must be 2");
    KERNEL_CHECK_TRUE(shapeAttrs->GetDims() == INPUT_NUM1, KERNEL_STATUS_PARAM_INVALID,
        "Dims of input[1][attrs] must be 1");
    KERNEL_CHECK_TRUE(shapeOutvecs->GetDims() == INPUT_NUM2, KERNEL_STATUS_PARAM_INVALID,
        "Dims of output[0][outvecs] must be 2");

    nq_ = shapeInvecs->GetDimSize(INPUT_NUM0);
    customAttrLen_ = shapeInvecs->GetDimSize(INPUT_NUM1);

    auto attrCount = shapeAttrs->GetDimSize(INPUT_NUM0);
    KERNEL_CHECK_TRUE(attrCount == TRANSDATA_CUSTOM_ATTR_IDX_COUNT, KERNEL_STATUS_PARAM_INVALID,
        "Num of attrs must be %d", TRANSDATA_CUSTOM_ATTR_IDX_COUNT);

    auto attr = static_cast<int64_t *>(inputs.attrs->GetData());
    blockOffset_ = *(attr + TRANSDATA_CUSTOM_ATTR_NTOTAL_IDX);

    return KERNEL_STATUS_OK;
}

template <typename T>
void TransdataCustomAttrCpuKernel::DoCompute(size_t start, size_t end, const Inputs &inputs, const Outputs &outputs)
{
    KernelTensor<T> invecsTensor(inputs.invecs);
    KernelTensor<T> outvecsTensor(outputs.outvecs);

    for (size_t i = start; i < end; ++i) {
        T *srcDataPtr = invecsTensor.GetSubTensorDim0(i);
        for (int64_t j = 0; j < customAttrLen_; ++j) {
            T *dstDataPtr = outvecsTensor.GetSubTensorDim1(j, blockOffset_ + i);
            auto err = memcpy_s(dstDataPtr, sizeof(T), srcDataPtr + j, sizeof(T));
            if (err != EOK) {
                KERNEL_LOG_ERROR("Copy data attr error %d", err);
            }
        }
    }
}

REGISTER_CPU_KERNEL(TRANSDATA_CUSTOM_ATTR, TransdataCustomAttrCpuKernel);
} // namespace aicpu
