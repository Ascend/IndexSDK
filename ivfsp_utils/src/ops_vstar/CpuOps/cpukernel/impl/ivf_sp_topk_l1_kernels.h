
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


#ifndef IVF_SP_TOPK_L1_KERNELS_H_
#define IVF_SP_TOPK_L1_KERNELS_H_

#include "cpu_kernel.h"
#include "utils/kernel_tensor.h"
#include "arm_fp16.h"

namespace aicpu {
class IvfSpTopkL1CpuKernel : public CpuKernel {
    struct Inputs {
        Tensor *indists = nullptr;
        Tensor *size = nullptr;
        Tensor *opflag = nullptr;
        Tensor *attr = nullptr;
    };

    struct Outputs {
        Tensor *outdists = nullptr;
        Tensor *outlabels = nullptr;
    };

public:
    IvfSpTopkL1CpuKernel() = default;
    ~IvfSpTopkL1CpuKernel() = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    uint32_t GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs) const;

    uint32_t CheckInputShapes(const Inputs &inputs);

    void UpdateOutputsShape(Outputs &outputs);

    template<typename T>
    void InitTopkHeap(Outputs &outputs) const;

    template <typename T, typename C>
    void DoCompute(size_t start, size_t end, const Inputs &inputs, Outputs &outputs, C &&cmp);

    template <typename T, typename C>
    void ComputeBlock(size_t n, int64_t blockIdx, KernelTensor<float16_t> &indistsTensor,
                      KernelTensor<uint32_t> &sizeTensor, KernelTensor<float16_t> &outdistsTensor,
                      KernelTensor<T> &outlabelsTensor, C &&cmp);

    template <typename T, typename C>
    void UpdateHeap(float16_t *dists, T *label, int64_t len, float16_t pushDist, int64_t index, C &&cmp);
    int64_t nq_ = 0;
    int64_t coreNum_ = 0;
    int64_t flagSize_ = 0;

    int64_t asc_ = 1;
    int64_t k_ = 0;
    int64_t blockNum_ = 0;
    int64_t pageIdx_ = 0;
    int64_t quickTopk_ = 0;

    DataType labelType_ = DT_INT64;
};
} // namespace aicpu
#endif
