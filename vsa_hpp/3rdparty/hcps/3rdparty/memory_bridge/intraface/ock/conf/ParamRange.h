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


#ifndef MEMORY_BRIDGE_OCK_HMM_PARAM_RANGE_CONF_H
#define MEMORY_BRIDGE_OCK_HMM_PARAM_RANGE_CONF_H

#include <cstdint>
#include <ostream>

namespace ock {
namespace conf {

template<typename T>
struct ParamRange {
    ParamRange(T minimumValue, T maximumValue) : minValue(minimumValue), maxValue(maximumValue) {}

    bool IsIn(const T &value) const
    {
        return minValue <= value && value <= maxValue;
    }
    bool NotIn(const T &value) const
    {
        return value < minValue || value > maxValue;
    }

    T minValue;
    T maxValue;
};

template<typename T>
bool operator==(const ParamRange<T> &lhs, const ParamRange<T> &rhs)
{
    return lhs.minValue == rhs.minValue && lhs.maxValue == rhs.maxValue;
}

template<typename T>
bool operator!=(const ParamRange<T> &lhs, const ParamRange<T> &rhs)
{
    return !(lhs == rhs);
}

template<typename T>
std::ostream &operator<<(std::ostream &os, const ParamRange<T> &data)
{
    return os << "[" << data.minValue << ", " << data.maxValue << "]";
}

}  // namespace conf
}  // namespace ock
#endif