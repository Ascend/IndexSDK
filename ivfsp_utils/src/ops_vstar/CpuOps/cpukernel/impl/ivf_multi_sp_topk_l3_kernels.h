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


#ifndef IVFSP_IVF_MULTI_SP_TOPK_L3_KERNELS_H
#define IVFSP_IVF_MULTI_SP_TOPK_L3_KERNELS_H
#include <sys/time.h>

#include "cpu_kernel.h"
#include "utils/kernel_tensor.h"
#include "arm_fp16.h"

namespace aicpu {
    class IvfMultiSpTopkL3CpuKernel : public CpuKernel {
        struct Inputs {
            Tensor *indists = nullptr;
            Tensor *vmdists = nullptr;
            Tensor *idaddress = nullptr;
            Tensor *opflag = nullptr;
            Tensor *attr = nullptr;
            Tensor *l2indices = nullptr;
        };

        struct Outputs {
            Tensor *outdists = nullptr;
            Tensor *outlabels = nullptr;
        };

    public:
        IvfMultiSpTopkL3CpuKernel() = default;

        ~IvfMultiSpTopkL3CpuKernel() = default;

        uint32_t Compute(CpuKernelContext &ctx) override;

    private:
        uint32_t GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs) const;  // done

        uint32_t CheckInputShapes(const Inputs &inputs); // done

        void UpdateOutputsShape(Outputs &outputs); // done

        template<typename T>
        void InitTopkHeap(Outputs &outputs) const; // done

        template <typename T, typename C>
        void DoCompute(size_t start, size_t end, const Inputs &inputs, Outputs &outputs, C &&cmp);

        template <typename T, typename C>
        void ComputeBlock(int64_t indexidx,
                          size_t n,
                          KernelTensor<float16_t> &indistsTensor,
                          KernelTensor<float16_t> &vmdistsTensor,
                          KernelTensor<uint64_t> &idaddressTensor,
                          KernelTensor<uint64_t> &l2indicesTensor,
                          KernelTensor<float16_t> &outdistsTensor,
                          KernelTensor<T> &outlabelsTensor,
                          bool reorder,
                          C &&cmp);

        int64_t nq_ = 0;
        int64_t indexNum_ = 0;
        int64_t blockSize_ = 0;
        int64_t coreNum_ = 0;
        int64_t flagSize_ = 0;

        int64_t asc_ = 0;  // 0 for put greatest one to top of heap, 1 for put least one to top of heap
        int64_t k_ = 0;
        int64_t burstLen_ = 0;
        int64_t l3SegNum_ = 0;
        int64_t l3SegSize_ = 0;
        int64_t nprobeL2_ = 0;
        int64_t realNprobeL2_ = 0;
        int64_t quickTopk_ = 0;

        DataType labelType_ = DT_INT64;
    };
} // namespace aicpu
#endif  // IVFSP_IVF_MULTI_SP_TOPK_L3_KERNELS_H
