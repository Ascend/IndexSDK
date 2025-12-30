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


#ifndef AICPU_TOPK_IVFSQT_L1_CPU_KERNEL_H
#define AICPU_TOPK_IVFSQT_L1_CPU_KERNEL_H

#include <arm_fp16.h>

#include "cpu_kernel.h"
#include "kernel_tensor.h"

namespace aicpu {
class TopkIvfsqtL1CpuKernel : public CpuKernel {
struct Inputs {
    Tensor *indists = nullptr;
    Tensor *vmdists = nullptr;
    Tensor *opflag = nullptr;
    Tensor *attr = nullptr;
    Tensor *queryIn = nullptr;
    Tensor *compressIndex = nullptr;
    Tensor *compressValue = nullptr;
};

struct Outputs {
    Tensor *outdists = nullptr;
    Tensor *outlabels = nullptr;
    Tensor *queryOut = nullptr;
};

public:
    TopkIvfsqtL1CpuKernel() = default;

    ~TopkIvfsqtL1CpuKernel() override = default;

    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    uint32_t GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs) const;

    uint32_t CheckInOutShapes(const Inputs &inputs, const Outputs &outputs);

    void UpdateInOutShape(Outputs &outputs) const;

    void InitTopkHeap(const Outputs &outputs) const;

    void CompressData(int64_t qidx, KernelTensor<float16_t> &queryInTensor, KernelTensor<float16_t> &queryOutTensor,
        KernelTensor<int> &compressIndexTensor, KernelTensor<float> &compressValueTensor);

    template <typename C>
    void DoCompute(size_t start, size_t end, const Inputs &inputs, const Outputs &outputs, C &&cmp);

    template <typename C>
    void ComputeQueryBatch(int64_t qidx,
                           KernelTensor<float16_t> &indistsTensor,
                           KernelTensor<float16_t> &vmdistsTensor,
                           KernelTensor<float16_t> &outdistsTensor,
                           KernelTensor<uint16_t> &outlabelsTensor,
                           KernelTensor<float16_t> &queryInTensor,
                           KernelTensor<int> &compressIndexTensor,
                           KernelTensor<float> &compressValueTensor,
                           KernelTensor<float16_t> &queryOutTensor,
                           C &&cmp);

private:
    int64_t nq_ = 0;
    int64_t flagNum_ = 0;
    int64_t flagSize_ = 0;

    int64_t asc_ = 1;
    int64_t k_ = 0;
    int64_t burstLen_ = 0;
    int64_t opSize_ = 0;
    int64_t queryBatchSize_ = 0;
    int64_t quickTopk_ = 0;

    int64_t dimIn_ = 0;
    int64_t dimOut_ = 0;
    int64_t ratio_ = 0;
};
} // namespace aicpu
#endif