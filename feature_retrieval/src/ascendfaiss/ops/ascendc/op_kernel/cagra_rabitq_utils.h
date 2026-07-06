/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

#ifndef CAGRA_RABITQ_UTILS_H
#define CAGRA_RABITQ_UTILS_H
#include <cstdint>
#include <limits>
#include <type_traits>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

using namespace AscendC;

constexpr float kFloatMax = std::numeric_limits<float>::max();

template <class T>
struct remove_cv
{
    using type = T;
};
template <class T>
struct remove_cv<const T>
{
    using type = T;
};
template <class T>
struct remove_cv<volatile T>
{
    using type = T;
};
template <class T>
struct remove_cv<const volatile T>
{
    using type = T;
};

template <class T, T v>
struct integral_constant
{
    static constexpr T value = v;
    typedef T value_type;
    typedef integral_constant<T, v> type;
    constexpr operator T() const noexcept { return v; }
    constexpr T operator()() const noexcept { return v; }
};

template <typename T, typename U>
struct is_same
{
    static const bool value = false;
};
template <typename T>
struct is_same<T, T>
{
    static const bool value = true;
};
template <class T, class U>
inline constexpr bool is_same_v = is_same<T, U>::value;

template <bool _Test, class _Ty = void>
struct enable_if
{
};
template <class _Ty>
struct enable_if<true, _Ty>
{
    using type = _Ty;
};
template <bool _Test, class _Ty = void>
using enable_if_t = typename enable_if<_Test, _Ty>::type;

template <class T>
struct is_floating_point : integral_constant<bool, is_same<float, typename remove_cv<T>::type>::value ||
                                                       is_same<double, typename remove_cv<T>::type>::value ||
                                                       is_same<long double, typename remove_cv<T>::type>::value>
{
};
template <class T>
inline constexpr bool is_floating_point_v = is_floating_point<T>::value;

template <class T>
constexpr unsigned size_of();
template <>
constexpr unsigned size_of<int8_t>()
{
    return 1;
}
template <>
constexpr unsigned size_of<uint8_t>()
{
    return 1;
}
template <>
constexpr unsigned size_of<uint16_t>()
{
    return 2;
}
template <>
constexpr unsigned size_of<uint32_t>()
{
    return 4;
}
template <>
constexpr unsigned size_of<uint64_t>()
{
    return 8;
}
template <>
constexpr unsigned size_of<uint4>()
{
    return 16;
}
template <>
constexpr unsigned size_of<ulonglong4>()
{
    return 32;
}
template <>
constexpr unsigned size_of<float>()
{
    return 4;
}
template <>
constexpr unsigned size_of<half>()
{
    return 2;
}
template <>
constexpr unsigned size_of<half2>()
{
    return 4;
}

template <class IdxT>
struct gen_index_msb_1_mask
{
    static constexpr IdxT value = static_cast<IdxT>(1u) << (size_of<IdxT>() * 8 - 1);
};

// max values for data types
template <class BS_T, class FP_T>
union fp_conv
{
    BS_T bs;
    FP_T fp;
};
template <class T>
__forceinline__[aicore] T get_max_value();

template <>
__forceinline__[aicore] float get_max_value<float>()
{
    return kFloatMax;
}

template <>
__forceinline__[aicore] half get_max_value<half>()
{
    return fp_conv<uint16_t, half>{.bs = 0x7bff}.fp;
}

template <>
__forceinline__[aicore] uint32_t get_max_value<uint32_t>()
{
    return 0xffffffffu;
}

template <>
__forceinline__[aicore] uint64_t get_max_value<uint64_t>()
{
    return 0xfffffffffffffffflu;
}

__forceinline__[aicore] uint64_t xorshift64(uint64_t u)
{
    u ^= u >> 12;
    u ^= u << 25;
    u ^= u >> 27;
    return u * 0x2545F4914F6CDD1DULL;
}

template <typename T>
__forceinline__[aicore] void swap(T &val1, T &val2)
{
    const T val0 = val1;
    val1 = val2;
    val2 = val0;
}

template <typename K>
__forceinline__[aicore] bool swap_if_needed(K &key1, K &key2)
{
    if (key1 > key2)
    {
        swap<K>(key1, key2);
        return true;
    }
    return false;
}

template <typename K, typename V>
__forceinline__[aicore] bool swap_if_needed(K &key1, K &key2, V &val1, V &val2)
{
    if (key1 > key2)
    {
        swap<K>(key1, key2);
        swap<V>(val1, val2);
        return true;
    }
    return false;
}

template <typename K, typename V>
__forceinline__[aicore] bool swap_if_needed(K &key1, K &key2, V &val1, V &val2, bool ascending)
{
    if (key1 == key2)
    {
        return false;
    }
    if ((key1 > key2) == ascending)
    {
        swap<K>(key1, key2);
        swap<V>(val1, val2);
        return true;
    }
    return false;
}

template <class K, class V>
__forceinline__[aicore] void swap_if_needed(K &k0, V &v0, K &k1, V &v1, const bool asc)
{
    if ((k0 != k1) && ((k0 < k1) != asc))
    {
        const auto tmp_k = k0;
        k0 = k1;
        k1 = tmp_k;
        const auto tmp_v = v0;
        v0 = v1;
        v1 = tmp_v;
    }
}

template <class K, class V>
__forceinline__[aicore] void swap_if_needed(K &k0, V &v0, const unsigned lane_offset, const bool asc)
{
    auto k1 = __shfl_xor(k0, lane_offset, 32);
    auto v1 = __shfl_xor(v0, lane_offset, 32);
    if ((k0 != k1) && ((k0 < k1) != asc))
    {
        k0 = k1;
        v0 = v1;
    }
}

template <class K, class V, unsigned N, unsigned warp_size = 32>
struct warp_merge_core
{
    __forceinline__[aicore] void operator()(K k[N], V v[N], const uint32_t range, const bool asc)
    {
        const auto lane_id = threadIdx.x % warp_size;

        if (range == 1)
        {
            for (uint32_t b = 2; b <= N; b <<= 1)
            {
                for (uint32_t c = b / 2; c >= 1; c >>= 1)
                {
#pragma unroll
                    for (uint32_t i = 0; i < N; i++)
                    {
                        uint32_t j = i ^ c;
                        if (i >= j) continue;
                        const auto line_id = i + (N * lane_id);
                        const auto p = static_cast<bool>(line_id & b) == static_cast<bool>(line_id & c);
                        swap_if_needed(k[i], v[i], k[j], v[j], (p == asc));
                    }
                }
            }
            return;
        }

        const uint32_t b = range;
        for (uint32_t c = b / 2; c >= 1; c >>= 1)
        {
            const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
#pragma unroll
            for (uint32_t i = 0; i < N; i++)
            {
                swap_if_needed(k[i], v[i], c, (p == asc));
            }
        }
        const auto p = ((lane_id & b) == 0);
        for (uint32_t c = N / 2; c >= 1; c >>= 1)
        {
#pragma unroll
            for (uint32_t i = 0; i < N; i++)
            {
                uint32_t j = i ^ c;
                if (i >= j) continue;
                swap_if_needed(k[i], v[i], k[j], v[j], (p == asc));
            }
        }
    }
};

template <class K, class V, unsigned warp_size>
struct warp_merge_core<K, V, 6, warp_size>
{
    __forceinline__[aicore] void operator()(K k[6], V v[6], const uint32_t range, const bool asc)
    {
        constexpr unsigned N = 6;
        const auto lane_id = threadIdx.x % warp_size;

        if (range == 1)
        {
            for (uint32_t i = 0; i < N; i += 3)
            {
                const auto p = (i == 0);
                swap_if_needed(k[0 + i], v[0 + i], k[1 + i], v[1 + i], (p == asc));
                swap_if_needed(k[1 + i], v[1 + i], k[2 + i], v[2 + i], (p == asc));
                swap_if_needed(k[0 + i], v[0 + i], k[1 + i], v[1 + i], (p == asc));
            }
            const auto p = ((lane_id & 1) == 0);
            for (uint32_t i = 0; i < 3; i++)
            {
                uint32_t j = i + 3;
                swap_if_needed(k[i], v[i], k[j], v[j], (p == asc));
            }
            for (uint32_t i = 0; i < N; i += 3)
            {
                swap_if_needed(k[0 + i], v[0 + i], k[1 + i], v[1 + i], (p == asc));
                swap_if_needed(k[1 + i], v[1 + i], k[2 + i], v[2 + i], (p == asc));
                swap_if_needed(k[0 + i], v[0 + i], k[1 + i], v[1 + i], (p == asc));
            }
            return;
        }

        const uint32_t b = range;
        for (uint32_t c = b / 2; c >= 1; c >>= 1)
        {
            const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
#pragma unroll
            for (uint32_t i = 0; i < N; i++)
            {
                swap_if_needed(k[i], v[i], c, (p == asc));
            }
        }
        const auto p = ((lane_id & b) == 0);
        for (uint32_t i = 0; i < 3; i++)
        {
            uint32_t j = i + 3;
            swap_if_needed(k[i], v[i], k[j], v[j], (p == asc));
        }
        for (uint32_t i = 0; i < N; i += N / 2)
        {
            swap_if_needed(k[0 + i], v[0 + i], k[1 + i], v[1 + i], (p == asc));
            swap_if_needed(k[1 + i], v[1 + i], k[2 + i], v[2 + i], (p == asc));
            swap_if_needed(k[0 + i], v[0 + i], k[1 + i], v[1 + i], (p == asc));
        }
    }
};

template <class K, class V, unsigned warp_size>
struct warp_merge_core<K, V, 3, warp_size>
{
    __forceinline__[aicore] void operator()(K k[3], V v[3], const uint32_t range, const bool asc)
    {
        constexpr unsigned N = 3;
        const auto lane_id = threadIdx.x % warp_size;

        if (range == 1)
        {
            const auto p = ((lane_id & 1) == 0);
            swap_if_needed(k[0], v[0], k[1], v[1], (p == asc));
            swap_if_needed(k[1], v[1], k[2], v[2], (p == asc));
            swap_if_needed(k[0], v[0], k[1], v[1], (p == asc));
            return;
        }

        const uint32_t b = range;
        for (uint32_t c = b / 2; c >= 1; c >>= 1)
        {
            const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
#pragma unroll
            for (uint32_t i = 0; i < N; i++)
            {
                swap_if_needed(k[i], v[i], c, (p == asc));
            }
        }
        const auto p = ((lane_id & b) == 0);
        swap_if_needed(k[0], v[0], k[1], v[1], (p == asc));
        swap_if_needed(k[1], v[1], k[2], v[2], (p == asc));
        swap_if_needed(k[0], v[0], k[1], v[1], (p == asc));
    }
};

template <class K, class V, unsigned warp_size>
struct warp_merge_core<K, V, 2, warp_size>
{
    __forceinline__[aicore] void operator()(K k[2], V v[2], const uint32_t range, const bool asc)
    {
        constexpr unsigned N = 2;
        const auto lane_id = threadIdx.x % warp_size;

        if (range == 1)
        {
            const auto p = ((lane_id & 1) == 0);
            swap_if_needed(k[0], v[0], k[1], v[1], (p == asc));
            return;
        }

        const uint32_t b = range;
        for (uint32_t c = b / 2; c >= 1; c >>= 1)
        {
            const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
#pragma unroll
            for (uint32_t i = 0; i < N; i++)
            {
                swap_if_needed(k[i], v[i], c, (p == asc));
            }
        }
        const auto p = ((lane_id & b) == 0);
        swap_if_needed(k[0], v[0], k[1], v[1], (p == asc));
    }
};

template <class K, class V, unsigned warp_size>
struct warp_merge_core<K, V, 1, warp_size>
{
    __forceinline__[aicore] void operator()(K k[1], V v[1], const uint32_t range, const bool asc)
    {
        const auto lane_id = threadIdx.x % warp_size;
        const uint32_t b = range;
        for (uint32_t c = b / 2; c >= 1; c >>= 1)
        {
            const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
            swap_if_needed(k[0], v[0], c, (p == asc));
        }
    }
};

template <class K, class V, unsigned N, unsigned warp_size = 32>
__forceinline__[aicore] void warp_merge(K k[N], V v[N], unsigned range, const bool asc = true)
{
    warp_merge_core<K, V, N, warp_size>{}(k, v, range, asc);
}

template <class K, class V, unsigned N, unsigned warp_size = 32>
__forceinline__[aicore] void warp_sort(K k[N], V v[N], const bool asc = true)
{
    for (uint32_t range = 1; range <= warp_size; range <<= 1)
    {
        warp_merge<K, V, N, warp_size>(k, v, range, asc);
    }
}

constexpr __forceinline__[aicore] uint32_t get_size(const uint32_t bitlen) { return 1U << bitlen; }

template <class IdxT>
__forceinline__[aicore] uint32_t hashmap_insert(__gm__ IdxT *const table, const uint32_t hash_table_size,
                                                const IdxT key)
{
    uint32_t index = key / 32;
    uint32_t bits = key % 32;
    if (index >= hash_table_size) return 0;

    if (atomicOr(&table[index], 1u << bits) & (1u << bits))
    {
        return 0;
    }
    return 1;
}

template <class IdxT>
__forceinline__[aicore] void init(__gm__ IdxT *const table, const uint32_t hash_table_size, unsigned FIRST_TID = 0)
{
    if (threadIdx.x < FIRST_TID) return;
    for (unsigned i = threadIdx.x - FIRST_TID; i < hash_table_size; i += blockDim.x - FIRST_TID)
    {
        table[i] = 0;
    }
}

#define DI __forceinline__[aicore]
#define HDI __forceinline__[aicore]

/**
 * @brief Provide a ceiling division operation ie. ceil(a / b)
 * @tparam IntType supposed to be only integers for now!
 */
template <typename IntType>
constexpr HDI IntType ceildiv(IntType a, IntType b)
{
    return (a / b) + ((a % b) != 0 ? 1 : 0);
}

template <typename math_, int VecLen>
struct IOType
{
};
template <>
struct IOType<bool, 1>
{
    static_assert(sizeof(bool) == sizeof(int8_t), "IOType bool size assumption failed");
    typedef int8_t Type;
};
template <>
struct IOType<bool, 2>
{
    typedef int16_t Type;
};
template <>
struct IOType<bool, 4>
{
    typedef int32_t Type;
};
template <>
struct IOType<bool, 8>
{
    typedef int2 Type;
};
template <>
struct IOType<bool, 16>
{
    typedef int4 Type;
};
template <>
struct IOType<int8_t, 1>
{
    typedef int8_t Type;
};
template <>
struct IOType<int8_t, 2>
{
    typedef int16_t Type;
};
template <>
struct IOType<int8_t, 4>
{
    typedef int32_t Type;
};
template <>
struct IOType<int8_t, 8>
{
    typedef int2 Type;
};
template <>
struct IOType<int8_t, 16>
{
    typedef int4 Type;
};
template <>
struct IOType<uint8_t, 1>
{
    typedef uint8_t Type;
};
template <>
struct IOType<uint8_t, 2>
{
    typedef uint16_t Type;
};
template <>
struct IOType<uint8_t, 4>
{
    typedef uint32_t Type;
};
template <>
struct IOType<uint8_t, 8>
{
    typedef uint2 Type;
};
template <>
struct IOType<uint8_t, 16>
{
    typedef uint4 Type;
};
template <>
struct IOType<int16_t, 1>
{
    typedef int16_t Type;
};
template <>
struct IOType<int16_t, 2>
{
    typedef int32_t Type;
};
template <>
struct IOType<int16_t, 4>
{
    typedef int2 Type;
};
template <>
struct IOType<int16_t, 8>
{
    typedef int4 Type;
};
template <>
struct IOType<uint16_t, 1>
{
    typedef uint16_t Type;
};
template <>
struct IOType<uint16_t, 2>
{
    typedef uint32_t Type;
};
template <>
struct IOType<uint16_t, 4>
{
    typedef uint2 Type;
};
template <>
struct IOType<uint16_t, 8>
{
    typedef uint4 Type;
};
template <>
struct IOType<half, 1>
{
    typedef half Type;
};
template <>
struct IOType<half, 2>
{
    typedef half2 Type;
};
template <>
struct IOType<half, 4>
{
    typedef uint2 Type;
};
template <>
struct IOType<half, 8>
{
    typedef uint4 Type;
};
template <>
struct IOType<half2, 1>
{
    typedef half2 Type;
};
template <>
struct IOType<half2, 2>
{
    typedef uint2 Type;
};
template <>
struct IOType<half2, 4>
{
    typedef uint4 Type;
};
template <>
struct IOType<int32_t, 1>
{
    typedef int32_t Type;
};
template <>
struct IOType<int32_t, 2>
{
    typedef uint2 Type;
};
template <>
struct IOType<int32_t, 4>
{
    typedef uint4 Type;
};
template <>
struct IOType<uint32_t, 1>
{
    typedef uint32_t Type;
};
template <>
struct IOType<uint32_t, 2>
{
    typedef uint2 Type;
};
template <>
struct IOType<uint32_t, 4>
{
    typedef uint4 Type;
};
template <>
struct IOType<float, 1>
{
    typedef float Type;
};
template <>
struct IOType<float, 2>
{
    typedef float2 Type;
};
template <>
struct IOType<float, 4>
{
    typedef float4 Type;
};
template <>
struct IOType<int64_t, 1>
{
    typedef int64_t Type;
};
template <>
struct IOType<int64_t, 2>
{
    typedef uint4 Type;
};
template <>
struct IOType<uint64_t, 1>
{
    typedef uint64_t Type;
};
template <>
struct IOType<uint64_t, 2>
{
    typedef uint4 Type;
};
template <>
struct IOType<unsigned long long, 1>
{
    typedef unsigned long long Type;
};
template <>
struct IOType<unsigned long long, 2>
{
    typedef uint4 Type;
};
template <>
struct IOType<double, 1>
{
    typedef float2 Type;
};
template <>
struct IOType<double, 2>
{
    typedef float4 Type;
};

template <typename math_, int veclen_>
struct TxN_t
{
    typedef math_ math_t;
    typedef typename IOType<math_t, veclen_>::Type io_t;
    static const int Ratio = veclen_;

    struct alignas(io_t)
    {
        math_t data[Ratio];
    } val;

    __forceinline__[aicore] auto *vectorized_data() { return reinterpret_cast<io_t *>(val.data); }

    DI void fill(math_t _val)
    {
#pragma unroll
        for (int i = 0; i < Ratio; ++i)
        {
            val.data[i] = _val;
        }
    }

    template <typename idx_t = int>
    DI void load(__gm__ const math_t *ptr, idx_t idx)
    {
        __gm__ const io_t *bptr = reinterpret_cast<__gm__ const io_t *>(&ptr[idx]);
        *vectorized_data() = __ldg(bptr);
    }

    template <typename idx_t = int>
    DI void load(__gm__ math_t *ptr, idx_t idx)
    {
        __gm__ io_t *bptr = reinterpret_cast<__gm__ io_t *>(&ptr[idx]);
        *vectorized_data() = *bptr;
    }

    template <typename idx_t = int>
    DI void store(__gm__ math_t *ptr, idx_t idx)
    {
        __gm__ io_t *bptr = reinterpret_cast<__gm__ io_t *>(&ptr[idx]);
        *bptr = *vectorized_data();
    }
};

using LOAD_64BIT_T = uint64_t;

template <class T, unsigned X_MAX = 1024>
__forceinline__[aicore] T swizzling(T x)
{
    // return x;
    if constexpr (X_MAX <= 1024)
    {
        return (x) ^ ((x) >> 5);
    }
    else
    {
        return (x) ^ (((x) >> 5) & 0x1f);
    }
}

template <class LOAD_T, class DATA_T>
__forceinline__[aicore] constexpr unsigned get_vlen()
{
    return sizeof(LOAD_T) / sizeof(DATA_T);
}

template <typename T>
struct config
{
};

template <>
struct config<float>
{
    using value_t = float;
    static constexpr float kDivisor = 1.0;
};
template <>
struct config<double>
{
    using value_t = double;
    static constexpr double kDivisor = 1.0;
};
template <>
struct config<half>
{
    using value_t = half;
    static constexpr float kDivisor = 1.0;
};

template <>
struct config<int8_t>
{
    using value_t = int32_t;
    static constexpr float kDivisor = 128.0;
};

template <typename T>
struct mapping
{
    template <typename S>
    HDI constexpr auto operator()(const S &x) const -> enable_if_t<is_same_v<S, T>, T>
    {
        return x;
    };

    template <typename S>
    HDI constexpr auto operator()(const S &x) const -> enable_if_t<!is_same_v<S, T>, T>
    {
        constexpr double kMult = config<T>::kDivisor / config<S>::kDivisor;
        if constexpr (is_floating_point_v<S>)
        {
            return static_cast<T>(x * static_cast<S>(kMult));
        }
        if constexpr (is_floating_point_v<T>)
        {
            return static_cast<T>(x) * static_cast<T>(kMult);
        }
        return static_cast<T>(static_cast<float>(x) * static_cast<float>(kMult));
    };
};

using QUERY_T = float;
using INDEX_T = uint32_t;

template <uint32_t DATASET_BLOCK_DIM>
__forceinline__[aicore] void copy_query(__gm__ const float *const dmem_query_ptr, __ubuf__ float *const smem_query_ptr,
                                        const uint32_t query_smem_buffer_length, const uint32_t dim,
                                        const uint32_t first_tid = 0)
{
    if (threadIdx.x < first_tid) return;
    for (unsigned i = threadIdx.x - first_tid; i < query_smem_buffer_length; i += blockDim.x - first_tid)
    {
        unsigned j = swizzling(i) & (query_smem_buffer_length - 1);
        if (i < dim)
        {
            smem_query_ptr[j] = QUERY_T(dmem_query_ptr[i]);
        }
        else
        {
            smem_query_ptr[j] = 0.0f;
        }
    }
}

#endif  // CAGRA_RABITQ_UTILS_H
