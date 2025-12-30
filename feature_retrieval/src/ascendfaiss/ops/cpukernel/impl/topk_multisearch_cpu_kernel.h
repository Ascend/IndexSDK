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


#ifndef AICPU_TOPK_MULTISEARCH_CPU_KERNEL_H
#define AICPU_TOPK_MULTISEARCH_CPU_KERNEL_H

#include <arm_fp16.h>
#include <sys/time.h>

#include "cpu_kernel.h"
#include "kernel_tensor.h"

namespace aicpu {
class TopkMultisearchCpuKernel : public CpuKernel {
struct Inputs {
    Tensor *indists = nullptr;
    Tensor *vmdists = nullptr;
    Tensor *size = nullptr;
    Tensor *opflag = nullptr;
    Tensor *attr = nullptr;
    Tensor *indexOffset = nullptr;
    Tensor *labelOffset = nullptr;
    Tensor *reorderFlag = nullptr;
};

struct Outputs {
    Tensor *outdists = nullptr;
    Tensor *outlabels = nullptr;
};

public:
    TopkMultisearchCpuKernel() = default;

    ~TopkMultisearchCpuKernel() override = default;

    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    uint32_t GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs) const;

    uint32_t CheckInputShapes(const Inputs &inputs);

    void UpdateOutputsShape(Outputs &outputs) const;

    void InitTopkHeap(Outputs &outputs);

    void SetParallMode(const Inputs &inputs);

    template <typename C>
    void ReorderLastBlock(float16_t *outdists, int64_t *outlabel, C &&cmp);

    template <typename C>
    void DoComputeParallForBatch(size_t start, size_t end, const Inputs &inputs, Outputs &outputs, C &&cmp);

    template <typename C>
    void DoComputeParallForBlcok(size_t tcnt, size_t tid, const Inputs &inputs, Outputs &outputs, C &&cmp);

    template <typename C>
    void SortTopkBurst(std::vector<std::pair<float16_t, int64_t>>& topkBurstIdx,
        int64_t *outlabel, int64_t pageOffset, float16_t *outdists, C &&cmp);

    template <typename C>
    void ComputeBlock(size_t n,
                      int64_t blockIdx,
                      KernelTensor<float16_t> &indistsTensor,
                      KernelTensor<float16_t> &vmdistsTensor,
                      KernelTensor<uint32_t> &sizeTensor,
                      KernelTensor<uint32_t> &indexOffsetTensor,
                      KernelTensor<uint32_t> &labelOffsetTensor,
                      KernelTensor<uint16_t> &reorderFlagTensor,
                      KernelTensor<float16_t> &outdistsTensor,
                      KernelTensor<int64_t> &outlabelsTensor,
                      C &&cmp);

    template <typename C>
    void UpdateHeap(float16_t *dists, int64_t *label, int64_t len, int64_t index, C &&cmp) const;

    int64_t nq_ = 0;
    int64_t blockSize_ = 0;
    int64_t coreNum_ = 0;
    int64_t flagSize_ = 0;

    int64_t asc_ = 1;
    int64_t k_ = 0;
    int64_t burstLen_ = 0;
    int64_t pageBlockNum_ = 0;
    int64_t indexNum_ = 0;
    int64_t quickTopk_ = 0;

    bool isBlockParall = false;
};
} // namespace aicpu
#endif