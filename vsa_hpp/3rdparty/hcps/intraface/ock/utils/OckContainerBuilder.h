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

#ifndef OCK_HCPS_PIER_CONTAINER_BUILD_H
#define OCK_HCPS_PIER_CONTAINER_BUILD_H
#include <cstdint>
#include <type_traits>
#include <vector>
namespace ock {
namespace utils {

template <typename _InContainerT, typename _OutContainerT>
void BuildPtrContainer(_InContainerT &from, _OutContainerT &to)
{
    static_assert(std::is_same<typename _InContainerT::value_type,
                      typename std::remove_pointer<typename _OutContainerT::value_type>::type>::value,
        "The value_type of 'to' must be a pointer of value_type of 'from'");
    for (auto iter = from.begin(); iter != from.end(); ++iter) {
        to.push_back(&(*iter));
    }
}
template <typename _InContainerT, typename _OutContainerT>
void CopyContainerData(_InContainerT &from, _OutContainerT &to)
{
    static_assert(std::is_same<typename _InContainerT::value_type, typename _OutContainerT::value_type>::value,
        "The value_type of 'to' must be same of value_type of 'from'");
    for (auto iter = from.begin(); iter != from.end(); ++iter) {
        to.push_back(*iter);
    }
}

template<typename _PtrContainerT>
void BuildFixLengthPtrContainer(uintptr_t addr, _PtrContainerT &inOutDatas)
{
    auto curPtr = reinterpret_cast<typename _PtrContainerT::value_type *>(addr);
    for (size_t i = 0; i < inOutDatas.size(); ++ i) {
        inOutDatas[i] = curPtr;
        curPtr ++;
    }
}
template<typename _PtrIteratorT>
void BuildFixLengthPtrContainerExt(uintptr_t addr, _PtrIteratorT begin, _PtrIteratorT end)
{
    auto curPtr = reinterpret_cast<typename _PtrIteratorT::value_type *>(addr);
    for (auto iter = begin; iter != end; ++ iter) {
        *iter = curPtr;
        curPtr ++;
    }
}
template <typename _InContainerT, typename _OutContainerT>
void BuildFixVectorPtrContainer(_InContainerT &from, _OutContainerT &to)
{
    static_assert(std::is_same<typename _InContainerT::value_type,
                          typename std::remove_pointer<typename _OutContainerT::value_type>::type>::value,
                  "The value_type of 'to' must be a pointer of value_type of 'from'");
    for (uint64_t i = 0; i < from.size() && i < to.size(); ++i) {
        to[i] = &from[i];
    }
}
}  // namespace utils
}  // namespace ock

#endif