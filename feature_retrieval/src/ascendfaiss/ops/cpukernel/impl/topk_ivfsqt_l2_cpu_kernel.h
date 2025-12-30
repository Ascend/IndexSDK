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


#ifndef AICPU_TOPK_IVFSQT_L2_CPU_KERNEL_H
#define AICPU_TOPK_IVFSQT_L2_CPU_KERNEL_H

#include <arm_fp16.h>

#include "cpu_kernel.h"
#include "kernel_tensor.h"

namespace aicpu {
class TopkIvfsqtL2CpuKernel : public CpuKernel {
struct Inputs {
    Tensor *indists = nullptr;
    Tensor *opflag = nullptr;
    Tensor *attr = nullptr;
    
    Tensor *subListSegNum = nullptr;
    Tensor *subListOffset = nullptr;
    Tensor *subListIndicesOffset = nullptr;
    Tensor *subListSizes = nullptr;
    Tensor *l1KIndices = nullptr;
};

struct Outputs {
    Tensor *subListOffsetL3 = nullptr;
    Tensor *idResult = nullptr;
    Tensor *opSize = nullptr;
};

public:
    TopkIvfsqtL2CpuKernel() = default;

    ~TopkIvfsqtL2CpuKernel() override = default;

    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    uint32_t GetInOutAndCheck(const CpuKernelContext &ctx, Inputs &inputs, Outputs &outputs) const;

    uint32_t CheckInOutShapes(const Inputs &inputs, const Outputs &outputs);

    void UpdateInOutShape(Inputs &inputs, Outputs &outputs) const;

    void DoCompute(size_t start, size_t end, const Inputs &inputs, const Outputs &outputs);

    void ComputeQueryBatch(int64_t qidx,
                           KernelTensor<float16_t> &indistsTensor,
                           KernelTensor<int> &subListSegNumTensor,
                           KernelTensor<uint64_t> &subListOffsetTensor,
                           KernelTensor<int64_t> &subListIndicesOffsetTensor,
                           KernelTensor<uint32_t> &subListSizesTensor,
                           KernelTensor<uint16_t> &l1KIndicesTensors,
                           KernelTensor<uint64_t> &subListOffsetL3Tensor,
                           KernelTensor<int64_t> &idResultTensor,
                           KernelTensor<uint32_t> &opSizeTensor);

private:
    int64_t nq_ = 0;
    int64_t flagNum_ = 0;
    int64_t flagSize_ = 0;

    int64_t k_ = 0;
    int64_t queryBatchSize_ = 0;
    int64_t subcenterNum_ = 0;
    int64_t l3SegNum_ = 0;
    int64_t l3SegSize_ = 0;
    int64_t l1NProbe_ = 0;
    int64_t pageShapedDataOffsetStep_ = 0;
};
} // namespace aicpu
#endif