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


#ifndef OCK_HCPS_PIER_TYPE_TRAITS_H
#define OCK_HCPS_PIER_TYPE_TRAITS_H
#include <type_traits>
#include <tuple>
#include <ostream>
#include <iterator>
#include <vector>
namespace ock {
namespace utils {
namespace traits {

// C17开始std支持disjunction
template <typename... _Bn>
struct Disjunction : std::__or_<_Bn...> {};
template <typename... _Bn>
struct Conjunction : std::true_type {};
template <typename _T, typename... _Bn>
struct Conjunction<_T, _Bn...> : std::conditional<bool(_T::value), Conjunction<_Bn...>, std::false_type> {};

template <typename _Tuple, size_t NTemp>
struct IsSameType : public Conjunction<std::is_same<typename std::tuple_element<0UL, _Tuple>::type,
                                           typename std::tuple_element<NTemp - 1UL, _Tuple>::type>,
                        typename IsSameType<_Tuple, NTemp - 1UL>::type> {};
template <typename _Tuple>
struct IsSameType<_Tuple, 1UL> : public std::true_type {};

template <typename _Tuple>
struct IsSameType<_Tuple, 2UL> : public std::is_same<typename std::tuple_element<0UL, _Tuple>::type,
                                     typename std::tuple_element<1UL, _Tuple>::type> {};
namespace impl {
template <typename _Tuple, size_t N>
struct TuplePrinterImpl {
    static std::ostream &Print(std::ostream &os, const _Tuple &data)
    {
        TuplePrinterImpl<_Tuple, N - 1ULL>::Print(os, data);
        return os << "," << std::get<N - 1ULL>(data);
    }
};
template <typename _Tuple>
struct TuplePrinterImpl<_Tuple, 1ULL> {
    static std::ostream &Print(std::ostream &os, const _Tuple &data)
    {
        return os << (std::get<0ULL>(data));
    }
};
template <typename _Tuple, size_t N>
struct TupleBeforeSizeImpl {
    static const size_t value =
        sizeof(typename std::tuple_element<N - 1ULL, _Tuple>::type) + TupleBeforeSizeImpl<_Tuple, N - 1ULL>::value;
};
template <typename _Tuple>
struct TupleBeforeSizeImpl<_Tuple, 0ULL> {
    static const size_t value = 0ULL;
};

template <typename _InputIterator, bool IsIntergerTemp>
struct DistanceImpl {
    static int32_t Value(_InputIterator begin, _InputIterator end)
    {
        return std::distance(begin, end);
    }
};
template <typename _InputIterator>
struct DistanceImpl<_InputIterator, true> {
    static int32_t Value(_InputIterator begin, _InputIterator end)
    {
        return static_cast<int32_t>(end - begin);
    }
};
}  // namespace impl
template <typename _Tuple>
struct TuplePrinter {
    static std::ostream &Print(std::ostream &os, const _Tuple &data)
    {
        return impl::TuplePrinterImpl<_Tuple, std::tuple_size<_Tuple>::value>::Print(os, data);
    }
};
template <size_t N, typename _Tuple>
struct TupleParse {
    static const typename std::tuple_element<N, _Tuple>::type &Parse(const uint8_t *addr)
    {
        return *reinterpret_cast<const typename std::tuple_element<N, _Tuple>::type *>(
            &(addr[impl::TupleBeforeSizeImpl<_Tuple, N>::value]));
    }
    static typename std::tuple_element<N, _Tuple>::type &Parse(uint8_t *addr)
    {
        return *reinterpret_cast<typename std::tuple_element<N, _Tuple>::type *>(
            &(addr[impl::TupleBeforeSizeImpl<_Tuple, N>::value]));
    }
    static typename std::tuple_element<N, _Tuple>::type &Parse(uintptr_t addr)
    {
        return *reinterpret_cast<typename std::tuple_element<N, _Tuple>::type *>(
            addr + impl::TupleBeforeSizeImpl<_Tuple, N>::value);
    }
};
template <typename _InputIterator>
int32_t Distance(_InputIterator begin, _InputIterator end)
{
    return impl::DistanceImpl<_InputIterator, std::is_integral<_InputIterator>::value>::Value(begin, end);
};

template <typename T>
struct IsIterator {
    template <typename U, typename = typename std::iterator_traits<U>::value_type>
    static std::true_type test(U *);
    template <typename>
    static std::false_type test(...);
public:
    static constexpr bool value = decltype(test<T>(nullptr))::value;
};
}  // namespace traits
}  // namespace utils
}  // namespace ock
#endif