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


#ifndef ASCENDC_OP_KERNEL_COMMON_H
#define ASCENDC_OP_KERNEL_COMMON_H

#ifndef ALLOC_LOCAL_TENSOR
#define ALLOC_LOCAL_TENSOR(localTensor, TensorSize, TensorDtype, position) \
    TBuf<QuePosition::position> localTensor##Buf;                          \
    pipe.InitBuffer(localTensor##Buf, (TensorSize) * sizeof(TensorDtype)); \
    auto (localTensor) = localTensor##Buf.Get<TensorDtype>()
#endif

namespace Utils {
constexpr uint32_t CUBE_ALIGN = 16;
constexpr uint32_t BURST_BLOCK_RATIO = 2;
constexpr uint32_t MASK_BLOCK_OFFSET_IDX = 1;
constexpr uint32_t MASK_LEN_IDX = 2;
constexpr uint32_t MASK_FLAG_IDX = 3;
constexpr uint32_t MASK_BIT_NUM = 8;
constexpr uint64_t VIC_HALF_FULL_MASK = 128;
constexpr half HALF_MAX = 65504.0;
constexpr half HALF_MIN = -65504.0;
constexpr uint32_t SELECT_REPEAT_TIME = 224;  // 要求<=255(uint8_t)，且是32B的整数倍
constexpr uint32_t BLOCK_HALF_NUM = 16;
constexpr uint32_t ALIGN_16 = 16;

template<typename U, typename V>
__aicore__ constexpr auto DivUp(U a, V b) -> decltype(a + b)
{
    return ((a + b - 1) / b);
}

template<typename U, typename V>
__aicore__ constexpr auto RoundUp(U a, V b) -> decltype(a + b)
{
    return DivUp(a, b) * b;
}

template<typename T>
__aicore__ constexpr auto Min(T a, T b) -> decltype(a)
{
    return a < b ? a : b;
}

template<typename T>
__aicore__ inline auto Max(T a, T b) -> decltype(a)
{
    return (a > b) ? a : b;
}
}

#endif // ASCENDC_OP_KERNEL_COMMON_H
