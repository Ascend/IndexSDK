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


#ifndef OCK_MEMORY_BRIDGE_OCK_SAFE_UTILS_H
#define OCK_MEMORY_BRIDGE_OCK_SAFE_UTILS_H
#include <cstdint>
#include <limits>
#include <cmath>
#include <vector>
#include <type_traits>
namespace ock {
namespace utils {

inline bool SafeFloatEqual(const double lhs, const double rhs, const double precision = 0.00000001)
{
    return fabs(lhs - rhs) < precision;
}
namespace traits {
struct LeftDoubleTag {};
struct RightDoubleTag {};
struct NoDoubleTag {};
struct AllDoubleTag {};
template <typename A, typename B>
struct DoubleTagDispatchTrait {
    using Tag = NoDoubleTag;
};
template <typename T>
struct DoubleTagDispatchTrait<double, T> {
    using Tag = LeftDoubleTag;
};
template <typename T>
struct DoubleTagDispatchTrait<T, double> {
    using Tag = RightDoubleTag;
};
template <>
struct DoubleTagDispatchTrait<double, double> {
    using Tag = AllDoubleTag;
};
template <typename LT, typename RT>
bool SafeEqual(const LT &lhs, const RT &rhs, traits::NoDoubleTag)
{
    if (std::is_signed<LT>::value) {
        if (!std::is_signed<RT>::value) {
            if (lhs < 0) {
                return false;
            }
            return (typename std::make_unsigned<LT>::type)(lhs) == rhs;
        } else {
            return lhs == rhs;
        }
    } else {
        if (std::is_signed<RT>::value) {
            if (rhs < 0) {
                return false;
            } else {
                return lhs == (typename std::make_unsigned<RT>::type)(rhs);
            }
        } else {
            return lhs == rhs;
        }
    }
}
template <typename LT, typename RT>
bool SafeLessThan(const LT &lhs, const RT &rhs, traits::NoDoubleTag)
{
    if (std::is_signed<LT>::value) {
        if (!std::is_signed<RT>::value) {
            if (lhs < 0) {
                return true;
            }
            return (typename std::make_unsigned<LT>::type)(lhs) < rhs;
        } else {
            return lhs < rhs;
        }
    } else {
        if (std::is_signed<RT>::value) {
            if (rhs < 0) {
                return false;
            } else {
                return lhs < (typename std::make_unsigned<RT>::type)(rhs);
            }
        } else {
            return lhs < rhs;
        }
    }
}
template <typename LT, typename RT>
bool SafeEqual(const LT &lhs, const RT &rhs, traits::AllDoubleTag)
{
    return SafeFloatEqual(lhs, rhs);
}
template <typename LT, typename RT>
bool SafeEqual(const LT &lhs, const RT &rhs, traits::LeftDoubleTag)
{
    return SafeFloatEqual(lhs, rhs);
}
template <typename LT, typename RT>
bool SafeEqual(const LT &lhs, const RT &rhs, traits::RightDoubleTag)
{
    return SafeFloatEqual(lhs, rhs);
}
template <typename LT, typename RT>
bool SafeLessThan(const LT &lhs, const RT &rhs, traits::AllDoubleTag)
{
    return lhs < rhs;
}
template <typename LT, typename RT>
bool SafeLessThan(const LT &lhs, const RT &rhs, traits::LeftDoubleTag)
{
    return lhs < rhs;
}
template <typename LT, typename RT>
bool SafeLessThan(const LT &lhs, const RT &rhs, traits::RightDoubleTag)
{
    return lhs < rhs;
}
}  // namespace traits

template <typename LT, typename RT>
bool SafeEqual(const LT &lhs, const RT &rhs)
{
    return traits::SafeEqual(lhs, rhs, typename traits::DoubleTagDispatchTrait<LT, RT>::Tag());
}
template <typename LT, typename RT>
bool SafeLessThan(const LT &lhs, const RT &rhs)
{
    return traits::SafeLessThan(lhs, rhs, typename traits::DoubleTagDispatchTrait<LT, RT>::Tag());
}
template <typename U, typename V>
constexpr auto SafeDivDown(U a, V b) -> decltype(a + b)
{
    return (b == 0) ? std::numeric_limits<decltype(a + b)>::max() : (a / b);
}

template <typename U, typename V>
constexpr auto SafeDivUp(U a, V b) -> decltype(a + b)
{
    return (b == 0) ? std::numeric_limits<decltype(a + b)>::max() : ((a + b - 1) / b);
}

template <typename U, typename V>
constexpr auto SafeDiv(U a, V b) -> decltype(a + b)
{
    return (b == 0) ? std::numeric_limits<decltype(a + b)>::max() : (a / b);
}

template <typename U, typename V>
constexpr auto SafeMod(U a, V b) -> decltype(a + b)
{
    return (b == 0) ? a : a % b;
}

template <typename U, typename V>
constexpr auto SafeRoundUp(U a, V b) -> decltype(a + b)
{
    return SafeDivUp(a, b) * b;
}

template <typename U, typename V>
constexpr auto SafeRoundDown(U a, V b) -> decltype(a + b)
{
    return SafeDivDown(a, b) * b;
}

}  // namespace utils
}  // namespace ock
#endif