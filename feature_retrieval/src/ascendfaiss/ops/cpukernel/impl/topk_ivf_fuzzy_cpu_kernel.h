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


#ifndef AICPU_TOPK_IVF_FUZZY_CPU_KERNEL_H
#define AICPU_TOPK_IVF_FUZZY_CPU_KERNEL_H

#include <arm_fp16.h>

#include "CircularQueue.h"

#include "cpu_kernel.h"
#include "kernel_tensor.h"

namespace aicpu {
class TopkIvfFuzzyCpuKernel : public CpuKernel {
struct Inputs {
    Tensor *indists = nullptr;
    Tensor *vmdists = nullptr;
    Tensor *ids = nullptr;
    Tensor *size = nullptr;
    Tensor *opflag = nullptr;
    Tensor *attr = nullptr;
};

struct Outputs {
    Tensor *outdists = nullptr;
    Tensor *outlabels = nullptr;
    Tensor *popdists = nullptr;
    Tensor *poplabels = nullptr;
};

public:
    TopkIvfFuzzyCpuKernel() = default;

    ~TopkIvfFuzzyCpuKernel() override = default;

    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    uint32_t GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs) const;

    uint32_t CheckInputShapes(const Inputs &inputs);

    void UpdateInOutShape(Inputs &inputs, Outputs &outputs) const;

    void InitTopkHeap(Outputs &outputs) const;

    template <typename C>
    void DoCompute(size_t start, size_t end, const Inputs &inputs, Outputs &outputs, C &&cmp);

    int64_t GetRealLabelByIndex(const int64_t *ids, int64_t index) const;

    template <typename C>
    void ComputeQueryBatch(int64_t qidx,
                           KernelTensor<float16_t> &indistsTensor,
                           KernelTensor<float16_t> &vmdistsTensor,
                           KernelTensor<int64_t> &idsTensor,
                           KernelTensor<uint32_t> &sizeTensor,
                           KernelTensor<float16_t> &outdistsTensor,
                           KernelTensor<int64_t> &outlabelsTensor,
                           KernelTensor<float16_t> &popdistsTensor,
                           KernelTensor<int64_t> &poplabelsTensor,
                           std::vector<std::pair<float16_t, uint32_t>> &distIndexPairs,
                           C &&cmp);

    template <typename C>
    int ComputeDistIndexPairs(
        std::vector<std::pair<float16_t, uint32_t>> &distIndexPairs, float16_t *vmdists, uint32_t *size, C &&cmp);

    void TransIndexToLabel(int64_t qidx, int64_t *ids, int64_t *outlabels);

    uint32_t CopyPopToOutput(int64_t qidx, float16_t *popdists, int64_t *poplabels);

    template <typename C>
    void FineRanking(uint32_t beginIdx, uint32_t endIdx, int64_t *outlabels, float16_t *outdists,
        float16_t *indists, int64_t qidx, uint32_t segIdx, int kHeap, C &&cmp);

private:
    int64_t nq_ = 0;
    int64_t flagNum_ = 0;
    int64_t flagSize_ = 0;

    int64_t asc_ = 1;
    int64_t k_ = 0;
    int64_t burstLen_ = 0;
    int64_t l3SegNum_ = 0;
    int64_t l3SegSize_ = 0;
    int64_t kHeapRatio_ = 0;
    int64_t kBufRatio_ = 0;
    int64_t queryBatchSize_ = 0;
    int64_t ivfFuzzyTopkMode_ = 0;

    std::vector<ascend::CircularQueue<float16_t, int64_t>> popQ_;
};
} // namespace aicpu
#endif