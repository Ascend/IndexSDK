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


#ifndef OCK_HCPS_PIER_CONTAINER_INFO_UTILS_H
#define OCK_HCPS_PIER_CONTAINER_INFO_UTILS_H
#include <cstdint>
#include <type_traits>
#include <iterator>
namespace ock {
namespace utils {

template <typename _Iterator>
struct OckContainerInfo {
    using IteratorT = _Iterator;
    OckContainerInfo(_Iterator beginPos, _Iterator endPos);
    
    uint64_t Size(void) const;
    bool Empty(void) const;
    _Iterator Middle(void);
    
    void CopyFrom(const OckContainerInfo<_Iterator> &other);

    _Iterator begin;
    _Iterator end;
};
template <typename _Iterator>
OckContainerInfo<_Iterator>::OckContainerInfo(_Iterator beginPos, _Iterator endPos) : begin(beginPos), end(endPos)
{}
template <typename _Iterator>
inline uint64_t OckContainerInfo<_Iterator>::Size(void) const
{
    return std::distance(begin, end);
}
template <typename _Iterator>
inline bool OckContainerInfo<_Iterator>::Empty(void) const
{
    return begin == end;
}
template <typename _Iterator>
inline _Iterator OckContainerInfo<_Iterator>::Middle(void)
{
    auto ret = begin;
    std::advance(ret, std::distance(begin, end) / 2ULL);
    return ret;
}
template <typename _Iterator>
inline void OckContainerInfo<_Iterator>::CopyFrom(const OckContainerInfo<_Iterator> &other)
{
    for (auto iter = begin, otherIter = other.begin; iter != end && otherIter != other.end; ++ iter, ++ otherIter) {
        *iter = *otherIter;
    }
}
template <typename T>
struct OckContainerInfoTraits {
public:
    template <typename U>
    static auto TestBegin(U *p) -> decltype(p->begin, std::true_type{});
    template <typename U>
    static auto TestEnd(U *p) -> decltype(p->end, std::true_type{});
    template <typename U>
    static auto TestSize(U *p) -> decltype(p->Size(), std::true_type{});
    template <typename U>
    static auto TestEmpty(U *p) -> decltype(p->Empty(), std::true_type{});
    template <typename U>
    static auto TestMiddle(U *p) -> decltype(p->Middle(), std::true_type{});

    template <typename U>
    static std::false_type TestBegin(...);
    template <typename U>
    static std::false_type TestEnd(...);
    template <typename U>
    static std::false_type TestSize(...);
    template <typename U>
    static std::false_type TestEmpty(...);
    template <typename U>
    static std::false_type TestMiddle(...);

    static constexpr bool value = decltype(TestBegin<T>(nullptr))::value && decltype(TestEnd<T>(nullptr))::value &&
                                  decltype(TestSize<T>(nullptr))::value && decltype(TestEmpty<T>(nullptr))::value &&
                                  decltype(TestMiddle<T>(nullptr))::value;
};

template <typename _Iterator>
OckContainerInfo<_Iterator> MakeContainerInfo(_Iterator begin, _Iterator end)
{
    return OckContainerInfo<_Iterator>(begin, end);
}

}  // namespace utils
}  // namespace ock
#endif