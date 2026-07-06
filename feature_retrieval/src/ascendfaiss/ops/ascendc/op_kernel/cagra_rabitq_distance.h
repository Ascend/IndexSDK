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

#ifndef CAGRA_RABITQ_DISTANCE_H
#define CAGRA_RABITQ_DISTANCE_H
#include "cagra_rabitq_utils.h"

using namespace AscendC;

template <class T>
__forceinline__[aicore] void set_smem_ptr(__ubuf__ T *const) {};

template <uint32_t DATASET_BLOCK_DIM, uint32_t TEAM_SIZE, typename DISTANCE_T = float>
__forceinline__[aicore] DISTANCE_T compute_similarity_rabitq(__gm__ uint8_t *code, uint32_t dataset_i, bool valid_i,
                                                             int dim,
                                                             __ubuf__ const uint8_t *const rearranged_rotated_qq,
                                                             float qr_to_c_L2sqr, int qb, float sum_q_quant,
                                                             float scale_q, float shift_query)
{
    if (!valid_i) return get_max_value<DISTANCE_T>();
    if (qb <= 0 || qb > 8) return get_max_value<DISTANCE_T>();

    const uint32_t CODE_UINT32_LENGTH = 4;
    const uint32_t NUM_PLANES = qb;
    const uint32_t PLANE_BYTES = 16;
    const uint32_t BLOCK_OFFSET = NUM_PLANES * PLANE_BYTES;
    const uint32_t BLOCK_BYTES = BLOCK_OFFSET + 16;

    const unsigned lane_id = threadIdx.x % TEAM_SIZE;
    __gm__ uint8_t *vec_ptr = code + dataset_i * BLOCK_BYTES;
    __ubuf__ const uint32_t *q_plane = reinterpret_cast<__ubuf__ const uint32_t *>(rearranged_rotated_qq);

    __gm__ float *base_precompute = reinterpret_cast<__gm__ float *>(vec_ptr + BLOCK_OFFSET);
    float or_c_l2sqr = base_precompute[0];
    float shift_base = base_precompute[1];
    float scale_x = base_precompute[2];
    float sum_x_quant = base_precompute[3];

    uint64_t dot_qb = 0;  // 改为 64 位，消除溢出隐患

    for (uint32_t k = lane_id; k < CODE_UINT32_LENGTH; k += TEAM_SIZE)
    {
        uint32_t qv[8];
#pragma unroll
        for (int j = 0; j < qb; ++j)
        {
            qv[j] = q_plane[j * CODE_UINT32_LENGTH + k];
        }
#pragma unroll
        for (int p = 0; p < NUM_PLANES; ++p)
        {
            __gm__ uint32_t *base_plane = reinterpret_cast<__gm__ uint32_t *>(vec_ptr + p * PLANE_BYTES);
            uint32_t base_word = base_plane[k];
#pragma unroll
            for (int m = 0; m < qb; ++m)
            {
                // 保持 64 位累加，安全左移
                dot_qb += (uint64_t)__popc(qv[m] & base_word) << (m + p);
            }
        }
    }

    // 将 64 位点积拆分为两个 32 位，通过 shuffle 完成 team 内规约
    uint32_t dot_lo = static_cast<uint32_t>(dot_qb & 0xFFFFFFFF);
    uint32_t dot_hi = static_cast<uint32_t>(dot_qb >> 32);

    for (uint32_t offset = TEAM_SIZE / 2; offset > 0; offset >>= 1)
    {
        dot_lo += __shfl_xor(dot_lo, offset, TEAM_SIZE);
        dot_hi += __shfl_xor(dot_hi, offset, TEAM_SIZE);
    }

    // 合并回 64 位
    dot_qb = (static_cast<uint64_t>(dot_hi) << 32) | dot_lo;

    float dot_product =
        scale_x * scale_q *
        ((float)dot_qb - shift_base * sum_q_quant - shift_query * sum_x_quant + shift_base * shift_query * (float)dim);
    return or_c_l2sqr + qr_to_c_L2sqr - 2.0f * dot_product;
}

struct none_cagra_sample_filter
{
    __forceinline__[aicore] bool operator()(const uint32_t query_ix, const uint32_t sample_ix) const { return true; }
};

template <unsigned TOPK_BY_BITONIC_SORT, class INDEX_T>
__forceinline__[aicore] void pickup_next_parents(__ubuf__ uint32_t *const terminate_flag,
                                                 __ubuf__ INDEX_T *const next_parent_offsets,
                                                 __ubuf__ INDEX_T *const internal_topk_indices,
                                                 const size_t internal_topk_size, const size_t dataset_size,
                                                 const uint32_t search_width, const unsigned first_tid)
{
    if (threadIdx.x < first_tid || threadIdx.x >= first_tid + 32) return;
    constexpr INDEX_T index_msb_1_mask = gen_index_msb_1_mask<INDEX_T>::value;
    const unsigned thread_id = threadIdx.x - first_tid;
    for (uint32_t i = thread_id; i < search_width; i += 32)
    {
        next_parent_offsets[i] = get_max_value<INDEX_T>();
    }
    uint32_t itopk_max = internal_topk_size;
    if (itopk_max % 32)
    {
        itopk_max += 32 - (itopk_max % 32);
    }
    uint32_t num_new_parents = 0;
    for (uint32_t j = thread_id; j < itopk_max; j += 32)
    {
        INDEX_T index;
        int new_parent = 0;
        if (j < internal_topk_size)
        {
            index = internal_topk_indices[j];
            if ((index & index_msb_1_mask) == 0)
            {
                new_parent = 1;
            }
        }
        const uint32_t ballot_mask = __ballot(new_parent);
        if (new_parent)
        {
            const auto i = __popc(ballot_mask & ((1 << thread_id) - 1)) + num_new_parents;
            if (i < search_width)
            {
                next_parent_offsets[i] = j;
            }
        }
        num_new_parents += __popc(ballot_mask);
        if (num_new_parents >= search_width)
        {
            break;
        }
    }
    if (thread_id == 0 && (num_new_parents == 0))
    {
        *terminate_flag = 1;
    }
}

template <unsigned CODE_SIZE, unsigned TEAM_SIZE, unsigned DATASET_BLOCK_DIM, class QUERY_T, class DISTANCE_T,
          class INDEX_T, class DATA_T>
__forceinline__[aicore] void compute_distance_to_random_nodes_rabitq(
    __ubuf__ INDEX_T *const result_indices_ptr, __ubuf__ DISTANCE_T *const result_distances_ptr,
    __gm__ DATA_T *base_data_code, __ubuf__ const QUERY_T *const rearranged_rotated_qq, const size_t num_pickup,
    const unsigned num_distillation, const uint64_t rand_xor_mask, __gm__ const INDEX_T *const seed_ptr,
    const uint32_t num_seeds, __gm__ INDEX_T *const visited_hash_ptr, const uint32_t hash_table_size, const size_t ld,
    const size_t size, const uint32_t dim, float qr_to_c_L2sqr, int qb, float sum_q_quant, float scale_q,
    float shift_query, const uint32_t block_id = 0, const uint32_t num_blocks = 1)
{
    uint32_t max_i = num_pickup;

    for (uint32_t i = threadIdx.x; i < max_i; i += blockDim.x)
    {
        bool valid_i = (i < max_i);

        INDEX_T best_index_team_local = get_max_value<INDEX_T>();
        DISTANCE_T best_norm2_team_local = get_max_value<DISTANCE_T>();
        for (uint32_t j = 0; j < num_distillation; j++)
        {
            INDEX_T seed_index = 0;
            if (valid_i)
            {
                uint32_t gid = block_id + (num_blocks * (i + (num_pickup * j)));
                if (seed_ptr && (gid < num_seeds))
                {
                    seed_index = seed_ptr[gid];
                }
                else
                {
                    seed_index = xorshift64(gid ^ rand_xor_mask);
                    seed_index = static_cast<INDEX_T>(seed_index % size);
                }
            }

            if (seed_index >= size) valid_i = false;
            const auto norm2 = compute_similarity_rabitq<DATASET_BLOCK_DIM, TEAM_SIZE>(
                base_data_code, seed_index, valid_i, dim, rearranged_rotated_qq, qr_to_c_L2sqr, qb, sum_q_quant,
                scale_q, shift_query);

            if (valid_i && (norm2 < best_norm2_team_local))
            {
                best_norm2_team_local = norm2;
                best_index_team_local = seed_index;
            }
        }

        if (valid_i)
        {
            if (hashmap_insert(visited_hash_ptr, hash_table_size, best_index_team_local))
            {
                result_distances_ptr[i] = best_norm2_team_local;
                result_indices_ptr[i] = best_index_team_local;
            }
            else
            {
                result_distances_ptr[i] = get_max_value<DISTANCE_T>();
                result_indices_ptr[i] = get_max_value<INDEX_T>();
            }
        }
    }
}

template <unsigned TEAM_SIZE, unsigned DATASET_BLOCK_DIM, class DISTANCE_T, class INDEX_T>
__forceinline__[aicore] void compute_distance_to_child_nodes_rabitq(
    __ubuf__ INDEX_T *const result_child_indices_ptr, __ubuf__ DISTANCE_T *const result_child_distances_ptr,
    __ubuf__ const uint8_t *const rearranged_rotated_qq, __gm__ uint8_t *base_data_code, const uint32_t dim,
    const uint32_t knn_k, const uint32_t search_width, float qr_to_c_L2sqr, int qb, float sum_q_quant, float scale_q,
    float shift_query)
{
    const INDEX_T invalid_index = get_max_value<INDEX_T>();
    uint32_t thread_id = threadIdx.x - 32;
    uint32_t lane_id = thread_id % TEAM_SIZE;
    uint32_t max_i = ceildiv(knn_k * (blockDim.x - 32), blockDim.x) * search_width;

    if (max_i % (32 / TEAM_SIZE))
    {
        max_i += (32 / TEAM_SIZE) - (max_i % (32 / TEAM_SIZE));
    }

    for (uint32_t tid = thread_id; tid < max_i * TEAM_SIZE; tid += blockDim.x - 32)
    {
        const auto i = tid / TEAM_SIZE;
        bool valid_i = (i < max_i);

        INDEX_T child_id = result_child_indices_ptr[i];
        const auto norm2 = compute_similarity_rabitq<DATASET_BLOCK_DIM, TEAM_SIZE>(
            base_data_code, child_id, child_id != invalid_index, dim, rearranged_rotated_qq, qr_to_c_L2sqr, qb,
            sum_q_quant, scale_q, shift_query);

        if (valid_i && lane_id == 0)
        {
            if (child_id != invalid_index)
            {
                result_child_distances_ptr[i] = norm2;
            }
            else
            {
                result_child_distances_ptr[i] = get_max_value<DISTANCE_T>();
            }
        }
    }
}

template <class INDEX_T>
__forceinline__[aicore] void copy_buffer_distances(__ubuf__ INDEX_T *candidates_indices,
                                                   __ubuf__ float *candidates_distances,
                                                   __ubuf__ float *temp_candidates_distances,
                                                   __ubuf__ INDEX_T *temp_candidates_indices,
                                                   const uint32_t candidates_size)
{
    const INDEX_T invalid_index = get_max_value<INDEX_T>();

    for (uint32_t i = threadIdx.x; i < candidates_size; i += blockDim.x)
    {
        INDEX_T child_id = temp_candidates_indices[i];
        candidates_indices[i] = child_id;
        float child_distance = temp_candidates_distances[i];
        if (child_id == invalid_index)
        {
            child_distance = get_max_value<float>();
        }
        candidates_distances[i] = child_distance;
    }
}

template <class INDEX_T>
__forceinline__[aicore] void set_parents(__ubuf__ INDEX_T *itopk_indices, const unsigned itopk,
                                         const __ubuf__ INDEX_T *parents_indices, const unsigned first_tid)
{
    if (threadIdx.x < first_tid) return;
    for (int i = threadIdx.x - first_tid; i < itopk; i += blockDim.x - first_tid)
    {
        if (parents_indices[0] == itopk_indices[i])
        {
            itopk_indices[i] |= gen_index_msb_1_mask<INDEX_T>::value;
        }
    }
}

template <class INDEX_T>
__forceinline__[aicore] void pickChildren(__gm__ const INDEX_T *const knn_graph, const uint32_t knn_k,
                                          __ubuf__ INDEX_T *const result_child_indices_ptr,
                                          __ubuf__ const INDEX_T *const parent_indices, const uint32_t search_width,
                                          __gm__ INDEX_T *const visited_hashmap_ptr, const uint32_t hash_table_size)
{
    const INDEX_T invalid_index = get_max_value<INDEX_T>();
    for (uint32_t i = threadIdx.x; i < knn_k * search_width; i += blockDim.x)
    {
        const INDEX_T smem_parent_id = parent_indices[i / knn_k];
        INDEX_T child_id = invalid_index;
        if (smem_parent_id != invalid_index)
        {
            child_id = knn_graph[(i % knn_k) + (static_cast<int64_t>(knn_k) * smem_parent_id)];
        }

        if (child_id != invalid_index)
        {
            if (hashmap_insert(visited_hashmap_ptr, hash_table_size, child_id) == 0)
            {
                child_id = invalid_index;
            }
        }
        result_child_indices_ptr[i] = child_id;
    }
}

template <unsigned MAX_CANDIDATES, class IdxT = void>
__forceinline__[aicore] void topk_by_bitonic_sort_1st(__ubuf__ float *candidate_distances,
                                                      __ubuf__ IdxT *candidate_indices, const uint32_t num_candidates,
                                                      const uint32_t num_itopk, unsigned MULTI_WARPS = 0)
{
    const unsigned lane_id = threadIdx.x % 32;
    const unsigned warp_id = threadIdx.x / 32;
    if (MULTI_WARPS == 0)
    {
        if (warp_id > 0)
        {
            return;
        }
        constexpr unsigned N = (MAX_CANDIDATES + 31) / 32;
        float key[N];
        IdxT val[N];
        for (unsigned i = 0; i < N; i++)
        {
            unsigned j = lane_id + (32 * i);
            if (j < num_candidates)
            {
                key[i] = candidate_distances[j];
                val[i] = candidate_indices[j];
            }
            else
            {
                key[i] = get_max_value<float>();
                val[i] = get_max_value<IdxT>();
            }
        }
        warp_sort<float, IdxT, N>(key, val);
        for (unsigned i = 0; i < N; i++)
        {
            unsigned j = (N * lane_id) + i;
            if (j < num_candidates && j < num_itopk)
            {
                candidate_distances[swizzling(j)] = key[i];
                candidate_indices[swizzling(j)] = val[i];
            }
        }
    }
    else
    {
        constexpr unsigned max_candidates_per_warp = (MAX_CANDIDATES + 1) / 2;
        constexpr unsigned N = (max_candidates_per_warp + 31) / 32;
        float key[N];
        IdxT val[N];
        if (warp_id < 2)
        {
            for (unsigned i = 0; i < N; i++)
            {
                unsigned jl = lane_id + (32 * i);
                unsigned j = jl + (max_candidates_per_warp * warp_id);
                if (j < num_candidates)
                {
                    key[i] = candidate_distances[j];
                    val[i] = candidate_indices[j];
                }
                else
                {
                    key[i] = get_max_value<float>();
                    val[i] = get_max_value<IdxT>();
                }
            }
            warp_sort<float, IdxT, N>(key, val);
            for (unsigned i = 0; i < N; i++)
            {
                unsigned jl = (N * lane_id) + i;
                unsigned j = jl + (max_candidates_per_warp * warp_id);
                if (j < num_candidates && jl < num_itopk)
                {
                    candidate_distances[swizzling(j)] = key[i];
                    candidate_indices[swizzling(j)] = val[i];
                }
            }
        }
        __sync_workitems();

        unsigned num_warps_used = (num_itopk + max_candidates_per_warp - 1) / max_candidates_per_warp;
        if (warp_id < num_warps_used)
        {
            for (unsigned i = 0; i < N; i++)
            {
                unsigned jl = (N * lane_id) + i;
                unsigned kl = max_candidates_per_warp - 1 - jl;
                unsigned j = jl + (max_candidates_per_warp * warp_id);
                unsigned k = MAX_CANDIDATES - 1 - j;
                if (j >= num_candidates || k >= num_candidates || kl >= num_itopk) continue;
                float temp_key = candidate_distances[swizzling(k)];
                if (key[i] == temp_key) continue;
                if ((warp_id == 0) == (key[i] > temp_key))
                {
                    key[i] = temp_key;
                    val[i] = candidate_indices[swizzling(k)];
                }
            }
        }
        if (num_warps_used > 1)
        {
            __sync_workitems();
        }
        if (warp_id < num_warps_used)
        {
            warp_merge<float, IdxT, N>(key, val, 32);
            for (unsigned i = 0; i < N; i++)
            {
                unsigned jl = (N * lane_id) + i;
                unsigned j = jl + (max_candidates_per_warp * warp_id);
                if (j < num_candidates && j < num_itopk)
                {
                    candidate_distances[swizzling(j)] = key[i];
                    candidate_indices[swizzling(j)] = val[i];
                }
            }
        }
        if (num_warps_used > 1)
        {
            __sync_workitems();
        }
    }
}

__forceinline__[aicore] uint32_t convert(uint32_t x)
{
    if (x & 0x80000000)
    {
        return x ^ 0xffffffff;
    }
    else
    {
        return x ^ 0x80000000;
    }
}

__forceinline__[aicore] uint16_t convert(uint16_t x)
{
    if (x & 0x8000)
    {
        return x ^ 0xffff;
    }
    else
    {
        return x ^ 0x8000;
    }
}

struct u32_vector
{
    uint1 x1;
    uint2 x2;
    uint4 x4;
    ulonglong4 x8;
};

struct u16_vector
{
    ushort1 x1;
    ushort2 x2;
    ushort4 x4;
    uint4 x8;
};

template <int vecLen>
__forceinline__[aicore] void load_u32_vector(struct u32_vector &vec, __ubuf__ const uint32_t *x, int i)
{
    if constexpr (vecLen == 1)
    {
        vec.x1 = ((__ubuf__ uint1 *)(x + i))[0];
    }
    else if constexpr (vecLen == 2)
    {
        vec.x2 = ((__ubuf__ uint2 *)(x + i))[0];
    }
    else if constexpr (vecLen == 4)
    {
        vec.x4 = ((__ubuf__ uint4 *)(x + i))[0];
    }
    else if constexpr (vecLen == 8)
    {
        vec.x8 = ((__ubuf__ ulonglong4 *)(x + i))[0];
    }
}

template <int vecLen>
__forceinline__[aicore] void load_u16_vector(struct u16_vector &vec, __ubuf__ const uint16_t *x, int i)
{
    if constexpr (vecLen == 1)
    {
        vec.x1 = ((__ubuf__ ushort1 *)(x + i))[0];
    }
    else if constexpr (vecLen == 2)
    {
        vec.x2 = ((__ubuf__ ushort2 *)(x + i))[0];
    }
    else if constexpr (vecLen == 4)
    {
        vec.x4 = ((__ubuf__ ushort4 *)(x + i))[0];
    }
    else if constexpr (vecLen == 8)
    {
        vec.x8 = ((__ubuf__ uint4 *)(x + i))[0];
    }
}

template <int vecLen>
__forceinline__[aicore] uint32_t get_element_from_u32_vector(struct u32_vector &vec, int i)
{
    uint32_t xi;
    if constexpr (vecLen == 1)
    {
        xi = convert(vec.x1.x);
    }
    else if constexpr (vecLen == 2)
    {
        switch (i)
        {
            case 0:
                xi = convert(vec.x2.x);
                break;
            case 1:
                xi = convert(vec.x2.y);
                break;
            default:
                xi = 0;
                break;  // 调用方保证 i 有效，兜底避免未初始化
        }
    }
    else if constexpr (vecLen == 4)
    {
        switch (i)
        {
            case 0:
                xi = convert(vec.x4.x);
                break;
            case 1:
                xi = convert(vec.x4.y);
                break;
            case 2:
                xi = convert(vec.x4.z);
                break;
            case 3:
                xi = convert(vec.x4.w);
                break;
            default:
                xi = 0;
                break;
        }
    }
    else if constexpr (vecLen == 8)
    {
        switch (i)
        {
            case 0:
                xi = convert((uint32_t)(vec.x8.x & 0xffffffff));
                break;
            case 1:
                xi = convert((uint32_t)(vec.x8.x >> 32));
                break;
            case 2:
                xi = convert((uint32_t)(vec.x8.y & 0xffffffff));
                break;
            case 3:
                xi = convert((uint32_t)(vec.x8.y >> 32));
                break;
            case 4:
                xi = convert((uint32_t)(vec.x8.z & 0xffffffff));
                break;
            case 5:
                xi = convert((uint32_t)(vec.x8.z >> 32));
                break;
            case 6:
                xi = convert((uint32_t)(vec.x8.w & 0xffffffff));
                break;
            case 7:
                xi = convert((uint32_t)(vec.x8.w >> 32));
                break;
            default:
                xi = 0;
                break;
        }
    }
    return xi;
}

template <int vecLen>
__forceinline__[aicore] uint16_t get_element_from_u16_vector(struct u16_vector &vec, int i)
{
    uint16_t xi;
    if constexpr (vecLen == 1)
    {
        xi = convert(vec.x1.x);
    }
    else if constexpr (vecLen == 2)
    {
        switch (i)
        {
            case 0:
                xi = convert(vec.x2.x);
                break;
            case 1:
                xi = convert(vec.x2.y);
                break;
            default:
                xi = 0;
                break;
        }
    }
    else if constexpr (vecLen == 4)
    {
        switch (i)
        {
            case 0:
                xi = convert(vec.x4.x);
                break;
            case 1:
                xi = convert(vec.x4.y);
                break;
            case 2:
                xi = convert(vec.x4.z);
                break;
            case 3:
                xi = convert(vec.x4.w);
                break;
            default:
                xi = 0;
                break;
        }
    }
    else if constexpr (vecLen == 8)
    {
        switch (i)
        {
            case 0:
                xi = convert((uint16_t)(vec.x8.x & 0xffff));
                break;
            case 1:
                xi = convert((uint16_t)(vec.x8.x >> 16));
                break;
            case 2:
                xi = convert((uint16_t)(vec.x8.y & 0xffff));
                break;
            case 3:
                xi = convert((uint16_t)(vec.x8.y >> 16));
                break;
            case 4:
                xi = convert((uint16_t)(vec.x8.z & 0xffff));
                break;
            case 5:
                xi = convert((uint16_t)(vec.x8.z >> 16));
                break;
            case 6:
                xi = convert((uint16_t)(vec.x8.w & 0xffff));
                break;
            case 7:
                xi = convert((uint16_t)(vec.x8.w >> 16));
                break;
            default:
                xi = 0;
                break;
        }
    }
    return xi;
}

template <typename T, int stateBitLen, int vecLen>
__forceinline__[aicore] void update_histogram(int itr, uint32_t thread_id, uint32_t num_threads, uint32_t hint,
                                              uint32_t threshold, uint32_t &num_bins, uint32_t &shift,
                                              __ubuf__ const T *x, uint32_t nx, __ubuf__ uint32_t *hist,
                                              __ubuf__ uint8_t *state, __ubuf__ uint32_t *output,
                                              __ubuf__ uint32_t *output_count)
{
    if (sizeof(T) == 4)
    {
        if (itr == 0)
        {
            shift = 21;
            num_bins = 2048;
        }
        else if (itr == 1)
        {
            shift = 10;
            num_bins = 2048;
        }
        else
        {
            shift = 0;
            num_bins = 1024;
        }
    }
    else if (sizeof(T) == 2)
    {
        if (itr == 0)
        {
            shift = 8;
            num_bins = 256;
        }
        else
        {
            shift = 0;
            num_bins = 256;
        }
    }
    else
    {
        return;
    }
    if (itr > 0)
    {
        for (int i = threadIdx.x; i < num_bins; i += blockDim.x)
        {
            hist[i] = 0;
        }
        __sync_workitems();
    }

    int ii = 0;
    for (int i = thread_id * vecLen; i < nx; i += num_threads * max(vecLen, stateBitLen), ii++)
    {
        uint8_t iState = 0;
        if ((stateBitLen == 8) && (itr > 0))
        {
            iState = state[thread_id + (num_threads * ii)];
            if (iState == (uint8_t)0xff) continue;
        }
#pragma unroll
        for (int v = 0; v < max(vecLen, stateBitLen); v += vecLen)
        {
            const int iv = i + (num_threads * v);
            if (iv >= nx) break;
            struct u32_vector x_u32_vec;
            struct u16_vector x_u16_vec;
            if (sizeof(T) == 4)
            {
                load_u32_vector<vecLen>(x_u32_vec, (__ubuf__ const uint32_t *)x, iv);
            }
            else
            {
                load_u16_vector<vecLen>(x_u16_vec, (__ubuf__ const uint16_t *)x, iv);
            }
#pragma unroll
            for (int u = 0; u < vecLen; u++)
            {
                const int ivu = iv + u;
                if (ivu >= nx) break;
                uint8_t mask = (uint8_t)0x1 << (v + u);
                if ((stateBitLen == 8) && (iState & mask)) continue;
                uint32_t xi;
                if (sizeof(T) == 4)
                {
                    xi = get_element_from_u32_vector<vecLen>(x_u32_vec, u);
                }
                else
                {
                    xi = get_element_from_u16_vector<vecLen>(x_u16_vec, u);
                }
                if ((xi > hint) && (itr == 0))
                {
                    if (stateBitLen == 8)
                    {
                        iState |= mask;
                    }
                }
                else if (xi < threshold)
                {
                    if (stateBitLen == 8)
                    {
                        output[atomicAdd(output_count, 1)] = ivu;
                        iState |= mask;
                    }
                }
                else
                {
                    const uint32_t k = (xi - threshold) >> shift;
                    if (k >= num_bins)
                    {
                        if (stateBitLen == 8)
                        {
                            iState |= mask;
                        }
                    }
                    else if (k + 1 < num_bins)
                    {
                        atomicAdd(&(hist[k + 1]), 1);
                    }
                }
            }
        }
        if (stateBitLen == 8)
        {
            state[thread_id + (num_threads * ii)] = iState;
        }
    }
    __sync_workitems();
}

template <typename T>
__forceinline__ __simt_callee__[aicore] T ScanWarp(T val)
{
    const int lane = threadIdx.x & 31;
    T tmp;
    tmp = __shfl_up(val, 1, 32);
    if (lane >= 1) val += tmp;
    tmp = __shfl_up(val, 2, 32);
    if (lane >= 2) val += tmp;
    tmp = __shfl_up(val, 4, 32);
    if (lane >= 4) val += tmp;
    tmp = __shfl_up(val, 8, 32);
    if (lane >= 8) val += tmp;
    tmp = __shfl_up(val, 16, 32);
    if (lane >= 16) val += tmp;
    return val;
}

template <typename T>
__forceinline__ __simt_callee__[aicore] void ScanBlockInclusive(T &val)
{
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    __ubuf__ T *warp_sum = (__ubuf__ T *)get_imm(0x0);

    if (warp_id == 0) warp_sum[lane] = 0;
    val = ScanWarp(val);
    __sync_workitems();
    if (lane == 31) warp_sum[warp_id] = val;
    __sync_workitems();
    if (warp_id == 0) warp_sum[lane] = ScanWarp(warp_sum[lane]);
    __sync_workitems();
    if (warp_id > 0) val += warp_sum[warp_id - 1];
    __sync_workitems();
}

template <typename T>
__forceinline__ __simt_callee__[aicore] void ScanBlockExclusive(T &val)
{
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    __ubuf__ T *warp_sum = (__ubuf__ T *)get_imm(0x0);

    T old_val = val;
    if (warp_id == 0) warp_sum[lane] = 0;
    val = ScanWarp(val);
    __sync_workitems();
    if (lane == 31) warp_sum[warp_id] = val;
    __sync_workitems();
    if (warp_id == 0) warp_sum[lane] = ScanWarp(warp_sum[lane]);
    __sync_workitems();
    if (warp_id > 0) val += warp_sum[warp_id - 1];
    val -= old_val;
    __sync_workitems();
}

template <typename T, int ITEMS_PER_THREAD>
__forceinline__[aicore] void inclusive_block_scan(T (&input)[ITEMS_PER_THREAD])
{
    for (int i = 1; i < ITEMS_PER_THREAD; ++i)
    {
        input[i] += input[i - 1];
    }

    T thread_prefix = input[ITEMS_PER_THREAD - 1];

    ScanBlockExclusive(thread_prefix);

    for (int i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        input[i] += thread_prefix;
    }
}

#endif  // CAGRA_RABITQ_DISTANCE_H
