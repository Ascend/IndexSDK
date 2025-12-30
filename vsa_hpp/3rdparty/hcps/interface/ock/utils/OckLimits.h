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


#ifndef OCK_HCPS_PIER_LIMITS_H
#define OCK_HCPS_PIER_LIMITS_H
#include <type_traits>
#include <limits>
#include "ock/utils/OckTypeTraits.h"
namespace ock {
namespace utils {

namespace impl {
template <typename T, bool IsNumberTemp>
struct LimitsImpl {
    static T Max(void)
    {
        return T::Max();
    }
    static T Min(void)
    {
        return T::Min();
    }
};
template <typename T>
struct LimitsImpl<T, true> {
    static T Max(void)
    {
        return std::numeric_limits<T>::max();
    }
    static T Min(void)
    {
        return std::numeric_limits<T>::min();
    }
};
}  // namespace impl
template <typename T>
struct Limits : public impl::LimitsImpl<T, traits::Disjunction<std::is_integral<T>, std::is_floating_point<T>>::value> {
    using BaseT = impl::LimitsImpl<T, traits::Disjunction<std::is_integral<T>, std::is_floating_point<T>>::value>;
    static T Max(void)
    {
        return BaseT::Max();
    }
    static T Min(void)
    {
        return BaseT::Min();
    }
};

}  // namespace utils
}  // namespace ock
#endif