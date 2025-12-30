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


#ifndef AICPU_TOPK_FLAT_FP32_CPU_KERNEL_H
#define AICPU_TOPK_FLAT_FP32_CPU_KERNEL_H

#define AICPU_RETURN_IF_NOT_LOG(X, ERRCODE, MSG)                                                                   \
    do                                                                                                              \
    {                                                                                                               \
        if (!(X)) {                                                                                                 \
            KERNEL_LOG_ERROR(MSG);                                                                                  \
            return ERRCODE;                                                                                         \
        }                                                                                                           \
    } while (false)


#include <arm_fp16.h>
#include <sys/time.h>

#include "cpu_kernel.h"
#include "kernel_tensor.h"

namespace aicpu {
class TopkFlatCpuKernelFp32 : public CpuKernel {
struct Inputs {
    Tensor *indists = nullptr;
    Tensor *vmdists = nullptr;
    Tensor *size = nullptr;
    Tensor *opflag = nullptr;
    Tensor *attr = nullptr;
};

struct Outputs {
    Tensor *outdists = nullptr;
    Tensor *outlabels = nullptr;
};

public:
    TopkFlatCpuKernelFp32() = default;

    ~TopkFlatCpuKernelFp32() override = default;

    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    uint32_t GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs) const;

    uint32_t CheckInputShapes(const Inputs &inputs);
    uint32_t GetAndCheckInOut(CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs);

    void UpdateOutputsShape(Outputs &outputs);

    template<typename T>
    void InitTopkHeap(Outputs &outputs) const;

    template <typename T, typename C>
    void DoCompute(size_t start, size_t end, const Inputs &inputs, Outputs &outputs, C &&cmp);

    template <typename T, typename C>
    void ReorderLastBlock(float *outdists, T *outlabel, C &&cmp);

    template <typename T, typename C>
    void updateHeapByBurst(size_t n, float *outdists, T *outlabel, int64_t blockIdx,
                      C &&cmp, KernelTensor<float> &indistsTensor);

    template <typename T, typename C>
    void ComputeBlock(size_t n,
                      int64_t blockIdx,
                      KernelTensor<float> &indistsTensor,
                      KernelTensor<float> &vmdistsTensor,
                      KernelTensor<uint32_t> &sizeTensor,
                      KernelTensor<float> &outdistsTensor,
                      KernelTensor<T> &outlabelsTensor,
                      bool reorder,
                      C &&cmp);
    template <typename T, typename C>
    void UpdateHeapByBurst(size_t n, int64_t blockIdx, int64_t burstSize, int64_t totalBaseoffset,
                           float *outdists, float *vmdists, T *outlabel, uint32_t *vmlabel,
                           KernelTensor<float> &indistsTensor, C &&cmp);
    template <typename T, typename C>
    void UpdateHeapByPos(int64_t outdisPos, float *outdists, int64_t indisPos, float *indists,
                         int64_t outlabelPos, T outLabelValue, T *outlabel, int64_t index,
                         C &&cmp);
    int64_t nq_ = 0;
    int64_t blockSize_ = 0;
    int64_t coreNum_ = 0;
    int64_t flagSize_ = 0;

    int64_t asc_ = 1;
    int64_t k_ = 0;
    int64_t burstLen_ = 0;
    int64_t blockNum_ = 0;
    int64_t pageIdx_ = 0;
    int64_t pageNum_ = 0;
    int64_t pageSize_ = 0;
    int64_t quickTopk_ = 0;

    DataType labelType_ = DT_INT64;
};
} // namespace aicpu
#endif