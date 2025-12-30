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


#ifndef AICPU_KM_UPDATE_CENTROIDS_CPU_KERNEL_H
#define AICPU_KM_UPDATE_CENTROIDS_CPU_KERNEL_H

#include <arm_fp16.h>
#include <sys/time.h>

#include "cpu_kernel.h"
#include "kernel_tensor.h"

namespace aicpu {
class KmUpdateCentroidsCpuKernel : public CpuKernel {
struct Inputs {
    Tensor *codes = nullptr;
    Tensor *assign = nullptr;
};

struct Outputs {
    Tensor *centroids = nullptr;
};

public:
    KmUpdateCentroidsCpuKernel() = default;

    ~KmUpdateCentroidsCpuKernel() override = default;

    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    uint32_t GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs) const;

    uint32_t CheckInOutShapes(const Inputs &inputs, const Outputs &outputs);

    void DoCompute(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs);

    void DoAccumulate(size_t start, size_t end, size_t kFrozen, std::vector<size_t> &hassign,
    float16_t *centroids, KernelTensor<float16_t> &codes, KernelTensor<uint64_t> &assign);

    void GetCentroids(size_t start, size_t end, float16_t *centroids, std::vector<size_t> &hassign);

    void RepalceCentroids(std::vector<size_t> &hassign, float16_t *centroids);

    int64_t nq_ = 0;
    int64_t dim_ = 0;
    int64_t k_ = 0;
};
} // namespace aicpu
#endif