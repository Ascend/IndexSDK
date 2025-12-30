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


#ifndef AICPU_VEC_L2SQR_CPU_KERNEL_H
#define AICPU_VEC_L2SQR_CPU_KERNEL_H

#include <arm_fp16.h>
#include <sys/time.h>

#include "cpu_kernel.h"
#include "kernel_tensor.h"

namespace aicpu {
class VecL2sqrCpuKernel : public CpuKernel {
struct Inputs {
    Tensor *invecs = nullptr;
};

struct Outputs {
    Tensor *outl2sqr = nullptr;
};

public:
    VecL2sqrCpuKernel() = default;

    ~VecL2sqrCpuKernel() override = default;

    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    uint32_t GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs) const;

    uint32_t CheckInOutShapes(const Inputs &inputs, const Outputs &outputs);

    template<typename inputType, typename outputType>
    void DoCompute(size_t start, size_t end, const Inputs &inputs, Outputs &outputs);

    int64_t nq_ = 0;
    int64_t dim_ = 0;
    int64_t outputCubeAlign_ = 1;
};
} // namespace aicpu
#endif