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


#ifndef AICPU_KERNEL_TENSOR_H
#define AICPU_KERNEL_TENSOR_H

#include <cstdint>
#include "cpu_kernel.h"

namespace aicpu {
template <typename T>
class KernelTensor {
public:
    KernelTensor() = delete;

    explicit KernelTensor(Tensor *tensor)
    {
        data_ = static_cast<T *>(tensor->GetData());
        dims_ = tensor->GetTensorShape()->GetDimSizes(); // dims = [dim0_size, dim1_size, dim2_size ... dimn_size]
        dim_ = dims_.size();
        strides_.resize(dim_);
        int64_t stride = 1;
        for (int i = dim_ - 1; i >= 0; i--) {
            strides_[i] = stride;
            stride *= dims_[i];
        }
    }

    T *GetSubTensorDim0(int64_t dim0)
    {
        if (dim_ < 1) { // must has least 1 dim
            return data_;
        }
        // return pointer to Tensor[dim0]
        return data_ + dim0 * strides_[0];
    }

    T *GetSubTensorDim1(int64_t dim0, int64_t dim1)
    {
        if (dim_ < 2) { // must has least 2 dim
            return data_;
        }
        // return pointer to Tensor[dim0][dim1]
        return data_ + dim0 * strides_[0] + dim1 * strides_[1];
    }

    T *GetSubTensorDim2(int64_t dim0, int64_t dim1, int64_t dim2)
    {
        if (dim_ < 3) { // must has least 3 dim
            return data_;
        }
        // return pointer to Tensor[dim0][dim1][dim2]
        return data_ + dim0 * strides_[0] + dim1 * strides_[1] + dim2 * strides_[2];
    }

    T *GetSubTensorDim3(int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3)
    {
        if (dim_ < 4) { // must has least 4 dim
            return data_;
        }
        // return pointer to Tensor[dim0][dim1][dim2][dim3]
        return data_ + dim0 * strides_[0] + dim1 * strides_[1] + dim2 * strides_[2] + dim3 * strides_[3];
    }

private:
// public:
    T *data_;

    size_t dim_;
    std::vector<int64_t> dims_;
    std::vector<int64_t> strides_;
};
} // namespace aicpu

#endif