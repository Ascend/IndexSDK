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

#include <cctype>
#include "ock/utils/StrUtils.h"
namespace ock {
namespace utils {

void Trim(::std::string &strSrc, const ::std::string &strTotrim)
{
    auto iBegin = strSrc.find_first_not_of(strTotrim);
    if (::std::string::npos == iBegin) {
        strSrc = "";
    } else {
        auto iEnd = strSrc.find_last_not_of(strTotrim);
        strSrc = strSrc.substr(iBegin, iEnd - iBegin + 1);
    }
}
void Split(const ::std::string &text, const ::std::string &delimiter, ::std::vector< ::std::string> &tokens)
{
    ::std::string::size_type iPos = 0;
    ::std::string sToSplit = text;

    Trim(sToSplit, delimiter);
    iPos = sToSplit.find(delimiter);
    while (::std::string::npos != iPos) {
        tokens.push_back(sToSplit.substr(0, iPos));
        sToSplit = sToSplit.substr(iPos, sToSplit.length() - iPos);
        Trim(sToSplit, delimiter);
        iPos = sToSplit.find(delimiter);
    }
    tokens.push_back(sToSplit);
}

std::string ToUpper(const std::string &inData)
{
    std::string data = inData;
    for (auto &chr : data) {
        if (chr >= 'a' && chr <= 'z') {
            chr = static_cast<char>(std::toupper(chr));
        }
    }
    return data;
}

template <typename T>
bool UnSingedNumericFromString(T &result, ::std::string const &data)
{
    if (data.empty()) {
        return false;
    } else if (data.find('-') != std::string::npos) {
        return false;
    } else if (data.length() > 2U && data[0] == '0' && (data[1U] == 'x' || data[1U] == 'X')) {
        ::std::istringstream is(data.substr(2U));
        is >> ::std::hex >> result;
        return !is.fail() && !is.bad() && is.eof();
    } else if (data.length() > 3U && data[0] == '+' && data[1U] == '0' && (data[2U] == 'x' || data[2U] == 'X')) {
        ::std::istringstream is(data.substr(3U));
        is >> ::std::hex >> result;
        return !is.fail() && !is.bad() && is.eof();
    }
    return detail::FromString(result, data);
}
template <typename T>
bool SingedNumericFromString(T &result, ::std::string const &data)
{
    if (detail::FromString<T>(result, data)) {
        return true;
    }
    if (data.length() > 2U && data[0] == '0' && (data[1U] == 'x' || data[1U] == 'X')) {
        ::std::istringstream is(data.substr(2U));
        is >> ::std::hex >> result;
        return !is.fail() && !is.bad() && is.eof();
    } else if (data.length() > 3U && data[0] == '+' && data[1U] == '0' && (data[2U] == 'x' || data[2U] == 'X')) {
        ::std::istringstream is(data.substr(3U));
        is >> ::std::hex >> result;
        return !is.fail() && !is.bad() && is.eof();
    } else if (data.length() > 3U && data[0] == '-' && data[1U] == '0' && (data[2U] == 'x' || data[2U] == 'X')) {
        ::std::istringstream is(data.substr(3U));
        is >> ::std::hex >> result;
        if (!is.fail() && !is.bad() && is.eof()) {
            result = static_cast<T>(-result);
            return true;
        }
    }
    return false;
}
bool FromStringEx(uint8_t &result, const std::string &data)
{
    uint32_t tmpValue = {0};
    if (UnSingedNumericFromString(tmpValue, data)) {
        if (tmpValue > uint32_t{UINT8_MAX}) {
            return false;
        }
        result = static_cast<int8_t>(tmpValue);
        return true;
    } else {
        return false;
    }
}
bool FromStringEx(uint16_t &result, const std::string &data)
{
    return UnSingedNumericFromString(result, data);
}
bool FromStringEx(uint32_t &result, const std::string &data)
{
    return UnSingedNumericFromString(result, data);
}
bool FromStringEx(uint64_t &result, const std::string &data)
{
    return UnSingedNumericFromString(result, data);
}
bool FromStringEx(bool &result, const std::string &data)
{
    if (data == "1" || data == "Y" || data == "YES" || data == "yes" || data == "Yes" || data == "T" ||
        data == "true" || data == "True" || data == "TRUE") {
        result = true;
        return true;
    }
    if (data == "0" || data == "N" || data == "NO" || data == "No" || data == "no" || data == "F" || data == "false" ||
        data == "False" || data == "FALSE") {
        result = false;
        return true;
    }
    return false;
}
bool FromStringEx(int8_t &result, const std::string &data)
{
    int32_t tmpValue = {0};
    if (SingedNumericFromString(tmpValue, data)) {
        if (tmpValue > int32_t{INT8_MAX} || tmpValue < int32_t{INT8_MIN}) {
            return false;
        }
        result = static_cast<int8_t>(tmpValue);
        return true;
    } else {
        return false;
    }
}
bool FromStringEx(int16_t &result, const std::string &data)
{
    return SingedNumericFromString(result, data);
}
bool FromStringEx(int32_t &result, const std::string &data)
{
    return SingedNumericFromString(result, data);
}
bool FromStringEx(int64_t &result, const std::string &data)
{
    return SingedNumericFromString(result, data);
}
}  // namespace utils
}  // namespace ock