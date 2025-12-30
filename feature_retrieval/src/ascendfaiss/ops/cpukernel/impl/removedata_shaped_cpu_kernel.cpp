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


#include "removedata_shaped_cpu_kernel.h"

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
const char* REMOVEDATA_SHAPED = "RemovedataShaped";
}

namespace aicpu {
uint32_t RemovedataShapedCpuKernel::Compute(CpuKernelContext &ctx)
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

    auto computeFunc = [&](size_t start, size_t end) {
        switch (dataType_) {
            case faiss::ascend::UINT8:
            case faiss::ascend::INT8:
                DoCompute<uint8_t>(start, end, inputs, outputs);
                break;
            case faiss::ascend::FLOAT16:
                DoCompute<float16_t>(start, end, inputs, outputs);
                break;
            default:
                KERNEL_LOG_ERROR("Invalid datatype");
        }
    };
    computeFunc(0, nq_);
    return KERNEL_STATUS_OK;
}

uint32_t RemovedataShapedCpuKernel::GetInOutAndCheck(const CpuKernelContext &ctx,
    Inputs &inputs, Outputs &outputs) const
{
    KERNEL_LOG_INFO("RemovedataShapedCpuKernel GetInOutAndCheck begin");

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

uint32_t RemovedataShapedCpuKernel::CheckInOutShapes(const Inputs &inputs, const Outputs &outputs)
{
    KERNEL_LOG_INFO("RemovedataShapedCpuKernel CheckInputShapes begin");

    auto shapeInvecs = inputs.invecs->GetTensorShape();
    auto shapeAttrs = inputs.attrs->GetTensorShape();
    auto shapeOutvecs = outputs.outvecs->GetTensorShape();

    KERNEL_CHECK_TRUE(shapeInvecs->GetDims() == INPUT_NUM1,
        KERNEL_STATUS_PARAM_INVALID, "Dims of input[0][invecs] must be 1");
    KERNEL_CHECK_TRUE(shapeAttrs->GetDims() == INPUT_NUM1,
                      KERNEL_STATUS_PARAM_INVALID, "Dims of input[1][attrs] must be 1");
    KERNEL_CHECK_TRUE(shapeOutvecs->GetDims() == INPUT_NUM1,
        KERNEL_STATUS_PARAM_INVALID, "Dims of output[0][outvecs] must be 1");

    nq_ = shapeInvecs->GetDimSize(INPUT_NUM0);

    auto attrCount = shapeAttrs->GetDimSize(INPUT_NUM0);
    KERNEL_CHECK_TRUE(attrCount == REMOVEDATA_SHAPED_ATTR_IDX_COUNT,
        KERNEL_STATUS_PARAM_INVALID, "Num of attrs must be %d", REMOVEDATA_SHAPED_ATTR_IDX_COUNT);

    auto attr = static_cast<int64_t *>(inputs.attrs->GetData());
    dataType_ = *(attr + REMOVEDATA_SHAPED_ATTR_DATA_TYPE);
    dimAlignedNum_ = *(attr + REMOVEDATA_SHAPED_ATTR_DIM_ALIGN_NUM);
    zRegionHeight_ = *(attr + REMOVEDATA_SHAPED_ATTR_ZREGION_HEIGHT);
    dimCubeAlign_ = *(attr + REMOVEDATA_SHAPED_ATTR_CUBE_ALIGN);
    dimAlignLen_ = zRegionHeight_ * dimCubeAlign_;

    return KERNEL_STATUS_OK;
}

template<typename T>
void RemovedataShapedCpuKernel::DoCompute(size_t start, size_t end, const Inputs &inputs, const Outputs &outputs)
{
    KernelTensor<uint64_t> invecsTensor(inputs.invecs);
    KernelTensor<uint64_t> outvecsTensor(outputs.outvecs);

    for (size_t i = start; i < end; ++i) {
        T *srcDataPtr = reinterpret_cast<T*>(*invecsTensor.GetSubTensorDim0(i));
        T *dstDataPtr = reinterpret_cast<T*>(*outvecsTensor.GetSubTensorDim0(i));
        
        for (int j = 0; j < dimAlignedNum_; j++) {
            auto err = memcpy_s(dstDataPtr, dimCubeAlign_ * sizeof(T), srcDataPtr, dimCubeAlign_ * sizeof(T));
            if (err != EOK) {
                KERNEL_LOG_ERROR("Copy data shaped error %d", err);
            }
            dstDataPtr += dimAlignLen_;
            srcDataPtr += dimAlignLen_;
        }
    }
}

REGISTER_CPU_KERNEL(REMOVEDATA_SHAPED, RemovedataShapedCpuKernel);
} // namespace aicpu
