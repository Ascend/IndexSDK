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

#ifndef AICPU_TOPK_IVFPQ_L3_CPU_KERNEL_H
#define AICPU_TOPK_IVFPQ_L3_CPU_KERNEL_H

#include <arm_fp16.h>

#include "cpu_kernel.h"
#include "kernel_tensor.h"

namespace aicpu {
class TopkIvfpqL3CpuKernel : public CpuKernel {
struct Inputs {
    Tensor *topkOffsets = nullptr;
    Tensor *topkDists = nullptr;
    Tensor *ids = nullptr;
    Tensor *size = nullptr;
    Tensor *opflag = nullptr;
    Tensor *attr = nullptr;
};

struct Outputs {
    Tensor *outdists = nullptr;
    Tensor *outlabels = nullptr;
};

public:
    TopkIvfpqL3CpuKernel() = default;

    ~TopkIvfpqL3CpuKernel() override = default;

    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    uint32_t GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs) const;

    uint32_t CheckInputShapes(const Inputs &inputs);

    void UpdateInOutShape(Inputs &inputs, Outputs &outputs) const;

    void InitTopkHeap(Outputs &outputs) const;

    template <typename C>
    void DoCompute(size_t startQidx, size_t queryCount, const Inputs &inputs, Outputs &outputs, C &&cmp);

    template <typename C>
    void ComputeBlock(int64_t qidx,
                      int64_t bidx,
                      int64_t hidx,
                      KernelTensor<int32_t> &topkOffsetsTensor,
                      KernelTensor<float> &topkDistsTensor,
                      KernelTensor<int64_t> &idsTensor,
                      KernelTensor<float> &outdistsTensor,
                      KernelTensor<int64_t> &outlabelsTensor,
                      bool isLastBlock,
                      C &&cmp);

    template <typename T, typename C>
    void UpdateHeapByPos(int64_t outdisPos, float *outdists, int64_t indisPos, float *topkDists,
                         int64_t outlabelPos, T outLabelValue, T *outlabel, int64_t index,
                         C &&cmp);

private:
    int64_t nq_ = 0;
    int64_t handleBatch_ = 32;
    int64_t flagSize_ = 16;

    int64_t asc_ = 1;
    int64_t k_ = 0;
    int64_t blockNum_ = 0;
    int64_t flagNum_ = 32;
};
} // namespace aicpu
#endif