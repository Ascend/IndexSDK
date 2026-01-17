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


#ifndef AICPU_TOPK_IVF_FP32_CPU_KERNEL_H
#define AICPU_TOPK_IVF_FP32_CPU_KERNEL_H

#include <arm_fp16.h>

#include "cpu_kernel.h"
#include "kernel_tensor.h"

namespace aicpu {
    class TopkIvffP32CpuKernel : public CpuKernel {
        struct Inputs {
            Tensor *indists = nullptr;
            Tensor *vmdists = nullptr;
            Tensor *ids = nullptr;
            Tensor *size = nullptr;
            Tensor *opflag = nullptr;
            Tensor *attr = nullptr;
        };

        struct TopkBurstPos {
            int64_t tileIdx = 0;
            int64_t blockIdx = 0;
            int64_t burstIdx = 0;
            int64_t offset = 0;
        };

        struct Outputs {
            Tensor *outdists = nullptr;
            Tensor *outlabels = nullptr;
        };

    public:
        TopkIvffP32CpuKernel() = default;

        ~TopkIvffP32CpuKernel() override = default;

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
                          TopkBurstPos* valueIndices,
                          KernelTensor<float> &indistsTensor,
                          KernelTensor<float> &vmdistsTensor,
                          KernelTensor<int64_t> &idsTensor,
                          KernelTensor<uint32_t> &sizeTensor,
                          KernelTensor<float> &outdistsTensor,
                          KernelTensor<int64_t> &outlabelsTensor,
                          bool isLastBlock,
                          C &&cmp);

        template <typename T, typename C>
        void UpdateHeapByPos(int64_t outdisPos, float *outdists, int64_t indisPos, float *indists,
                             int64_t outlabelPos, T outLabelValue, T *outlabel, int64_t index,
                             C &&cmp);
        template <typename C>
        void UpdateHeapByBurst(int64_t qidx, int64_t tileIdx, int64_t blockIdx,  TopkBurstPos* valueIndices,
                               int64_t burstSize, float *outdists, float *vmdists, int64_t *outlabel,
                               uint32_t *vmlabel, bool isLastBlock, KernelTensor<float> &indistsTensor,
                               KernelTensor<int64_t> &idsTensor, C &&cmp);

        template <typename T, typename C>
        void UpdateHeapWithPos(float *dists, T *label, TopkBurstPos* valueIndices, int64_t len, int64_t index, C &&cmp);

    private:
        int64_t nq_ = 0;
        int64_t handleBatch_ = 0;
        int64_t flagSize_ = 0;

        int64_t asc_ = 1;
        int64_t k_ = 0;
        int64_t burstLen_ = 0;
        int64_t blockNum_ = 0;
        int64_t flagNum_ = 0;
        int64_t quickTopk_ = 0;
    };
} // namespace aicpu
#endif