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


#ifndef ASCEND_STATICUTILS_H
#define ASCEND_STATICUTILS_H

#include <sys/time.h>
#include <limits>

namespace ascendSearchacc {
namespace utils {
template <typename U, typename V>
constexpr auto divDown(U a, V b) -> decltype(a + b)
{
    return (b == 0) ? std::numeric_limits<U>::max() : (a / b);
}

template <typename U, typename V>
constexpr auto divUp(U a, V b) -> decltype(a + b)
{
    using CommonType = std::common_type_t<U, V>;
    return (b == 0) ? std::numeric_limits<CommonType>::max() : ((static_cast<CommonType>(a) + static_cast<CommonType>(b) - 1) / static_cast<CommonType>(b));
}

template <typename U, typename V>
constexpr auto roundDown(U a, V b) -> decltype(a + b)
{
    return divDown(a, b) * b;
}

template <typename U, typename V>
constexpr auto roundUp(U a, V b) -> decltype(a + b)
{
    return divUp(a, b) * b;
}

template <class T>
constexpr T pow(T n, T power)
{
    return (power > 0) ? (n * pow(n, power - 1)) : 1;
}

template <class T>
constexpr T pow2(T n)
{
    const int power = 2;
    return pow(power, (T)n);
}

template <typename T>
constexpr int log2(T n, int p = 0)
{
    return (n <= 1) ? p : log2(n / 2, p + 1);  // 2 means divisor
}

template <typename T>
constexpr bool isPowerOf2(T v)
{
    return (v && !(v & (v - 1)));
}

template <typename T>
constexpr T nextHighestPowerOf2(T v)
{
    return (isPowerOf2(v) ? (T)2 * v : ((T)1 << (log2(v) + 1)));  // 2 means scale
}

inline double getMillisecs()
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    const double sec2msec = 1e3;
    const double usec2msec = 1e-3;
    return tv.tv_sec * sec2msec + tv.tv_usec * usec2msec;
}
}  // namespace utils
}  // namespace ascendSearchacc
#endif  // ASCEND_STATICUTILS_H
