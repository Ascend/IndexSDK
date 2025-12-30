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


#ifndef IVFSP_IVF_MULTI_SP_TOPK_L2_KERNELS_H
#define IVFSP_IVF_MULTI_SP_TOPK_L2_KERNELS_H
#include <arm_fp16.h>

#include "cpu_kernel.h"
#include "utils/kernel_tensor.h"

namespace aicpu {
    class IvfMultiSpTopkL2CpuKernel : public CpuKernel {
        struct Inputs {
            Tensor *dists = nullptr;
            Tensor *L1Indices = nullptr;
            Tensor *opFlag = nullptr;
            Tensor *attr = nullptr;
        };

        struct Outputs {
            Tensor *distsRes = nullptr;
            Tensor *indicesL2 = nullptr;
        };

    public:
        IvfMultiSpTopkL2CpuKernel() = default;
        ~IvfMultiSpTopkL2CpuKernel() = default;
        uint32_t Compute(CpuKernelContext &ctx) override;

    private:
        uint32_t GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs) const;
        uint32_t CheckInputShapes(const Inputs &inputs);
        void UpdateOutputsShape(Outputs &outputs);

        void InitTopkHeap(Outputs &outputs) const;

        template <typename C>
        void DoCompute(size_t start, size_t end, const Inputs &inputs, Outputs &outputs, C &&cmp);

        template <typename C>
        void ComputeBlock(size_t n,
                          KernelTensor<float16_t> &inDistsTensor,
                          KernelTensor<uint16_t> &l1IndicesTensor,
                          KernelTensor<float16_t> &outDistsTensor,
                          KernelTensor<uint64_t> &outIndicesL2Tensor,
                          C &&cmp);
        template <typename C>
        void UpdateHeap(float16_t *dists, uint64_t *label, int64_t len, float16_t pushDist, uint64_t index, C &&cmp);
        int64_t nq_ = 0;
        int64_t nsize_ = 0;
        int64_t coreNum_ = 0;

        int64_t flagSize_ = 0;
        int64_t asc_ = 0; // 0 for IP Distance, 1 for L2 Distance
        int64_t nListL2_ = 0;
        int64_t threadCnt_ = 0;
        int64_t nProbeL2 = 0;
    };
} // namespace aicpu
#endif  // IVFSP_IVF_MULTI_SP_TOPK_L2_KERNELS_H
