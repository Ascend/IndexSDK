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


#ifndef OCK_HCPS_PIER_OSTREAM_UTILS_H
#define OCK_HCPS_PIER_OSTREAM_UTILS_H
#include <list>
#include <vector>
#include <set>
#include <map>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <ostream>
namespace ock {
namespace utils {

template <typename DataT>
std::ostream &operator<<(std::ostream &os, const std::shared_ptr<DataT> &data)
{
    if (data.get() == nullptr) {
        return os << "nullptr";
    }
    return os << *data;
}
template <typename ContainerT>
std::ostream &PrintContainer(std::ostream &os, const ContainerT &data, uint64_t maxPrintElemCount = 15ULL,
    const std::string &splitFlag = "")
{
    uint64_t curIndex = 0;
    os << "[";
    for (auto iter = data.begin(); iter != data.end() && curIndex < maxPrintElemCount; ++iter, ++curIndex) {
        if (iter != data.begin()) {
            os << ",";
        }
        os << *iter << splitFlag;
    }
    if (curIndex < data.size()) {
        os << ", left " << (data.size() - curIndex) << " items ...]";
    } else {
        os << "]";
    }
    return os;
}
template <typename T>
std::ostream &PrintArray(std::ostream &os, uint32_t count, const T *feature, uint32_t maxPrintCount = 20UL)
{
    uint32_t i = 0;
    for (i = 0; i < count && i < maxPrintCount; ++i) {
        if (i != 0) {
            os << ",";
        }
        os << feature[i];
    }
    if (i < count) {
        os << ", ...(left " << (count - i) << " datas)";
    }
    return os;
}
template <typename FirstT, typename SecondT>
std::ostream &operator<<(std::ostream &os, const std::pair<FirstT, SecondT> &data)
{
    return os << "{" << data.first << ":" << data.second << "}";
}
template <typename DataT>
std::ostream &operator<<(std::ostream &os, const std::list<DataT> &data)
{
    return PrintContainer(os, data);
}
template <typename DataT>
std::ostream &operator<<(std::ostream &os, const std::vector<DataT> &data)
{
    return PrintContainer(os, data);
}
template <typename DataT>
std::ostream &operator<<(std::ostream &os, const std::set<DataT> &data)
{
    return PrintContainer(os, data);
}
template <typename DataT>
std::ostream &operator<<(std::ostream &os, const std::unordered_set<DataT> &data)
{
    return PrintContainer(os, data);
}
template <typename KeyT, typename ValueT>
std::ostream &operator<<(std::ostream &os, const std::unordered_map<KeyT, ValueT> &data)
{
    return PrintContainer(os, data);
}
template <typename KeyT, typename ValueT>
std::ostream &operator<<(std::ostream &os, const std::map<KeyT, ValueT> &data)
{
    return PrintContainer(os, data);
}
template <typename DataT>
std::ostream &Print(std::ostream &os, const DataT &data)
{
    return os << data;
}

}  // namespace utils
}  // namespace ock
#endif