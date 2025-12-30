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


#ifndef OCK_MEMORY_BRIDGE_OCK_ADAPTER_STR_UTILS_H
#define OCK_MEMORY_BRIDGE_OCK_ADAPTER_STR_UTILS_H
#include <sstream>
#include <vector>
#include <string>
#include "ock/utils/OstreamUtils.h"
namespace ock {
namespace utils {
void Trim(::std::string &strSrc, const ::std::string &strTotrim);

template <typename T>
std::string ToString(const T &data)
{
    std::ostringstream osStr;
    osStr << data;
    return osStr.str();
}
namespace detail {
template <typename T>
bool FromString(T &result, const std::string &data)
{
    try {
        std::istringstream is(data);
        is >> result;
        return !is.fail() && !is.bad() && is.eof();
    } catch (...) {
        return false;
    }
}

}  // namespace detail
template <typename T>
bool FromString(T &result, const std::string &data)
{
    return detail::FromString(result, data);
}
bool FromStringEx(uint8_t &result, const std::string &data);
bool FromStringEx(uint16_t &result, const std::string &data);
bool FromStringEx(uint32_t &result, const std::string &data);
bool FromStringEx(uint64_t &result, const std::string &data);
bool FromStringEx(bool &result, const std::string &data);
bool FromStringEx(int8_t &result, const std::string &data);
bool FromStringEx(int16_t &result, const std::string &data);
bool FromStringEx(int32_t &result, const std::string &data);
bool FromStringEx(int64_t &result, const std::string &data);
template <>
inline bool FromString<uint8_t>(uint8_t &result, const std::string &data)
{
    return FromStringEx(result, data);
}
template <>
inline bool FromString<uint16_t>(uint16_t &result, const std::string &data)
{
    return FromStringEx(result, data);
}
template <>
inline bool FromString<uint32_t>(uint32_t &result, const std::string &data)
{
    return FromStringEx(result, data);
}
template <>
inline bool FromString<uint64_t>(uint64_t &result, const std::string &data)
{
    return FromStringEx(result, data);
}
template <>
inline bool FromString<bool>(bool &result, const std::string &data)
{
    return FromStringEx(result, data);
}
template <>
inline bool FromString<int8_t>(int8_t &result, const std::string &data)
{
    return FromStringEx(result, data);
}
template <>
inline bool FromString<int16_t>(int16_t &result, const std::string &data)
{
    return FromStringEx(result, data);
}
template <>
inline bool FromString<int32_t>(int32_t &result, const std::string &data)
{
    return FromStringEx(result, data);
}
template <>
inline bool FromString<int64_t>(int64_t &result, const std::string &data)
{
    return FromStringEx(result, data);
}
/*
@brief 字符串切割，连续的delimiter也被认为是一个分割
*/
void Split(const ::std::string &text, const ::std::string &delimiter, ::std::vector< ::std::string> &tokens);

std::string ToUpper(const std::string &data);

}  // namespace utils
}  // namespace ock
#endif