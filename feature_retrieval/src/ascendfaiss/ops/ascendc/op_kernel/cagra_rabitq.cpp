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

#include "cagra_rabitq_topk.h"
#include "utils/debug/asc_assert.h"

__simt_vf__ __aicore__ LAUNCH_BOUND(32) inline void CagraRabitqUnsupportedTilingKey() { __trap(); }

template <unsigned Capacity, class IndexT>
__forceinline__ __simt_callee__[aicore] void SortRadixCandidates(uint32_t count, uint32_t topk,
                                                                 __ubuf__ float *distances, __ubuf__ IndexT *indices,
                                                                 __ubuf__ uint32_t *hints, __ubuf__ uint32_t *scratch)
{
    // The histogram's convert() float-flips raw FP32 bits; final sorting compares FP32 values.
    // Preserve negative RaBitQ estimates rather than clamping or transforming the input twice.
    if (threadIdx.x == 0) hints[0] = ~0U;
    __sync_workitems();
    auto *keys = reinterpret_cast<__ubuf__ uint32_t *>(distances);
    topk_by_radix_sort<Capacity, IndexT>{}(topk, 56, count, keys, indices, keys, indices, nullptr, hints, true,
                                           scratch);
    __sync_workitems();
}

template <int TEAM_SIZE, int DATASET_BLOCK_DIM, int MAX_ITOPK, int MAX_CANDIDATES, bool TOPK_BY_BITONIC_SORT,
          typename DATA_T, typename INDEX_T, typename DISTANCE_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void Simtsearch(
    __gm__ float *precompute_all, __gm__ uint8_t *code, __gm__ uint8_t *rotated_qq_ptr_all, const uint32_t qb,
    const uint32_t rotated_qq_size, __ubuf__ uint8_t *rotated_qq_buffer, __ubuf__ DISTANCE_T *distance_flag,
    __gm__ const float *const queries_ptr, const uint32_t dim, __ubuf__ DATA_T *query_buffer,
    const uint32_t query_smem_buffer_length, __ubuf__ uint32_t *terminate_flag, __ubuf__ uint32_t *topk_ws,
    const uint32_t hash_table_size, __gm__ const INDEX_T *seed_ptr, const uint32_t num_seeds,
    __ubuf__ INDEX_T *result_indices_buffer, __ubuf__ DISTANCE_T *result_distances_buffer,
    const uint32_t result_buffer_size, const unsigned num_distillation, const uint64_t rand_xor_mask, __gm__ float *ptr,
    const uint64_t ld, const uint64_t size, __gm__ uint32_t *const knn_graph, const uint32_t graph_degree,
    __ubuf__ uint32_t *count_flag, __ubuf__ DISTANCE_T *candidate_distances_buffer,
    __ubuf__ INDEX_T *candidate_indices_buffer, __ubuf__ INDEX_T *parent_offsets_buffer,
    __ubuf__ INDEX_T *temp_indices_buffer, __ubuf__ DISTANCE_T *temp_distances_buffer, const uint32_t patience,
    const uint32_t internal_topk, const uint32_t search_width, const uint32_t max_iteration,
    const uint32_t min_iteration, const uint32_t top_k, __gm__ DISTANCE_T *result_distances_ptr,
    __gm__ INDEX_T *result_indices_ptr, __gm__ uint32_t *const num_executed_iterations,
    __ubuf__ uint32_t *smem_work_ptr, const uint32_t adding_pref, __ubuf__ uint32_t *filter_flag,
    __gm__ uint32_t *const visited_hashmap_ptr)

{
    none_cagra_sample_filter sample_filter;
    const auto query_id = blockIdx.x;
    __gm__ const DATA_T *const query_ptr = queries_ptr + query_id * dim;
    __gm__ uint8_t *rotated_qq_ptr = rotated_qq_ptr_all + query_id * rotated_qq_size;
    __gm__ float *precompute = precompute_all + 4 * query_id;
    {
        copy_query<DATASET_BLOCK_DIM>(query_ptr, query_buffer, query_smem_buffer_length, dim);
        uint32_t first_tid = blockDim.x > ceildiv<int>(dim, 32) ? ceildiv<int>(dim, 32) : 0;
        for (unsigned i = threadIdx.x - first_tid; i < rotated_qq_size; i += blockDim.x - first_tid)
        {
            rotated_qq_buffer[i] = rotated_qq_ptr[i];
        }
    }

    float qr_to_c_L2sqr = precompute[0];
    float shift_query = precompute[1];
    float scale_q = precompute[2];
    float sum_q_quant = precompute[3];

    if (threadIdx.x == 0)
    {
        terminate_flag[0] = 0;
        distance_flag[0] = get_max_value<DISTANCE_T>();
        topk_ws[0] = ~0u;
    }

    __gm__ INDEX_T *local_visited_hashmap_ptr;
    local_visited_hashmap_ptr = visited_hashmap_ptr + (hash_table_size * query_id);
    init(local_visited_hashmap_ptr, hash_table_size, 0);
    __sync_workitems();

    __gm__ const INDEX_T *const local_seed_ptr = seed_ptr ? seed_ptr + (num_seeds * query_id) : nullptr;
    compute_distance_to_random_nodes_rabitq<TEAM_SIZE, DATASET_BLOCK_DIM>(
        result_indices_buffer, result_distances_buffer, code, rotated_qq_buffer, result_buffer_size, num_distillation,
        rand_xor_mask, local_seed_ptr, num_seeds, local_visited_hashmap_ptr, hash_table_size, ld, size, dim,
        qr_to_c_L2sqr, qb, sum_q_quant, scale_q, shift_query, query_id * result_buffer_size, 1,
        size <= result_buffer_size);

    for (int i = threadIdx.x; i < search_width * graph_degree; i += blockDim.x)
    {
        temp_distances_buffer[i] = get_max_value<DATA_T>();
        temp_indices_buffer[i] = get_max_value<INDEX_T>();
    }

    if constexpr (!TOPK_BY_BITONIC_SORT)
    {
        __sync_workitems();
        SortRadixCandidates<MAX_ITOPK>(result_buffer_size, internal_topk, result_distances_buffer,
                                       result_indices_buffer, topk_ws, smem_work_ptr);
        // Selected entries can still occur in the unsorted tail. Discard that tail before expansion.
        for (uint32_t i = internal_topk + threadIdx.x; i < result_buffer_size; i += blockDim.x)
        {
            result_distances_buffer[i] = get_max_value<DISTANCE_T>();
            result_indices_buffer[i] = get_max_value<INDEX_T>();
        }
        __sync_workitems();
    }
    uint32_t iter = 0;
    std::uint32_t not_add_count = 0;

    while (iter < max_iteration && (TOPK_BY_BITONIC_SORT ? not_add_count <= patience : size > internal_topk))
    {
        if (threadIdx.x == 0)
        {
            count_flag[0] = 0;
        }

        if constexpr (TOPK_BY_BITONIC_SORT)
        {
            const unsigned multi_warps_1 = ((blockDim.x >= 64) && (MAX_CANDIDATES > 128)) ? 1 : 0;
            // Split 256-entry lists across two warps to keep per-lane register pressure bounded on A5.
            const unsigned multi_warps_2 = ((blockDim.x >= 64) && (MAX_ITOPK >= 256)) ? 1 : 0;

            __sync_workitems();

            get_min_k(result_distances_buffer + internal_topk, result_indices_buffer + internal_topk,
                      candidate_distances_buffer, candidate_indices_buffer, search_width * graph_degree);

            {
                const unsigned first_tid = ((blockDim.x <= 32) ? 0 : 32);
                pickup_next_parents<TOPK_BY_BITONIC_SORT, INDEX_T>(terminate_flag, parent_offsets_buffer,
                                                                   result_indices_buffer, internal_topk, size,
                                                                   search_width, first_tid);
            }

            __sync_workitems();
            if (threadIdx.x == 0)
            {
                const INDEX_T invalid_index = get_max_value<INDEX_T>();
                const INDEX_T parent_offset = parent_offsets_buffer[0];
                const bool topk_parent_valid = parent_offset != invalid_index;
                const bool candidate_parent_valid = candidate_indices_buffer[0] != invalid_index;
                const float topk_parent_distance =
                    topk_parent_valid ? result_distances_buffer[parent_offset] : get_max_value<DISTANCE_T>();
                if (candidate_parent_valid &&
                    (!topk_parent_valid || candidate_distances_buffer[0] < topk_parent_distance))
                {
                    parent_offsets_buffer[0] = candidate_indices_buffer[0];
                }
                else if (topk_parent_valid)
                {
                    parent_offsets_buffer[0] = result_indices_buffer[parent_offset];
                }
                else
                {
                    parent_offsets_buffer[0] = invalid_index;
                }
                if (parent_offsets_buffer[0] != invalid_index)
                {
                    terminate_flag[0] = 0;
                }
            }

            __sync_workitems();

            pickChildren(knn_graph, graph_degree, temp_indices_buffer, parent_offsets_buffer, search_width,
                         local_visited_hashmap_ptr, hash_table_size, size);

            __sync_workitems();

            if (threadIdx.x >= 32)
            {
                compute_distance_to_child_nodes_rabitq<TEAM_SIZE, DATASET_BLOCK_DIM>(
                    temp_indices_buffer, temp_distances_buffer, rotated_qq_buffer, code, dim, graph_degree,
                    search_width, qr_to_c_L2sqr, qb, sum_q_quant, scale_q, shift_query);
            }
            __sync_workitems();

            if constexpr (is_same<decltype(sample_filter), none_cagra_sample_filter>::value)
            {
                topk_by_bitonic_sort<MAX_ITOPK, MAX_CANDIDATES>(result_distances_buffer, result_indices_buffer,
                                                                internal_topk, result_distances_buffer + internal_topk,
                                                                result_indices_buffer + internal_topk,
                                                                search_width * graph_degree, topk_ws, (iter == 0),
                                                                multi_warps_1, multi_warps_2, count_flag, adding_pref);
            }
            else if (*filter_flag == 0)
            {
                topk_by_bitonic_sort<MAX_ITOPK, MAX_CANDIDATES>(result_distances_buffer, result_indices_buffer,
                                                                internal_topk, result_distances_buffer + internal_topk,
                                                                result_indices_buffer + internal_topk,
                                                                search_width * graph_degree, topk_ws, (iter == 0),
                                                                multi_warps_1, multi_warps_2, count_flag, adding_pref);
            }
            else
            {
                topk_by_bitonic_sort_1st<MAX_ITOPK + MAX_CANDIDATES>(result_distances_buffer, result_indices_buffer,
                                                                     internal_topk + search_width * graph_degree,
                                                                     internal_topk, false);
                if (threadIdx.x == 0)
                {
                    *terminate_flag = 0;
                }
            }
        }
        else
        {
            // The radix prefix already contains all candidates from the previous expansion.
            // Select from the full prefix, rather than stopping when its first 26 entries settle.
            {
                const unsigned first_tid = ((blockDim.x <= 32) ? 0 : 32);
                pickup_next_parents<TOPK_BY_BITONIC_SORT, INDEX_T>(terminate_flag, parent_offsets_buffer,
                                                                   result_indices_buffer, internal_topk, size,
                                                                   search_width, first_tid);
            }

            __sync_workitems();
            if (threadIdx.x == 0)
            {
                const INDEX_T invalid_index = get_max_value<INDEX_T>();
                const INDEX_T parent_offset = parent_offsets_buffer[0];
                if (parent_offset != invalid_index)
                {
                    parent_offsets_buffer[0] = result_indices_buffer[parent_offset];
                    result_indices_buffer[parent_offset] |= gen_index_msb_1_mask<INDEX_T>::value;
                    terminate_flag[0] = 0;
                }
                else
                {
                    terminate_flag[0] = 1;
                }
            }

            __sync_workitems();
            if (*terminate_flag && iter >= min_iteration)
            {
                break;
            }

            pickChildren(knn_graph, graph_degree, temp_indices_buffer, parent_offsets_buffer, search_width,
                         local_visited_hashmap_ptr, hash_table_size, size);

            __sync_workitems();

            if (threadIdx.x >= 32)
            {
                compute_distance_to_child_nodes_rabitq<TEAM_SIZE, DATASET_BLOCK_DIM>(
                    temp_indices_buffer, temp_distances_buffer, rotated_qq_buffer, code, dim, graph_degree,
                    search_width, qr_to_c_L2sqr, qb, sum_q_quant, scale_q, shift_query);
            }
            __sync_workitems();

            // Merge this expansion before testing the iteration limit, including the final round.
            copy_buffer_distances(result_indices_buffer + internal_topk, result_distances_buffer + internal_topk,
                                  temp_distances_buffer, temp_indices_buffer, search_width * graph_degree);
            __sync_workitems();
            SortRadixCandidates<MAX_ITOPK>(result_buffer_size, internal_topk, result_distances_buffer,
                                           result_indices_buffer, topk_ws, smem_work_ptr);
        }
        __sync_workitems();

        if constexpr (TOPK_BY_BITONIC_SORT)
        {
            if (count_flag[0] == 1)
            {
                not_add_count++;
            }
            else
            {
                not_add_count = 0;
            }
        }
        if (iter + 1 == max_iteration)
        {
            break;
        }
        if (*terminate_flag && iter >= min_iteration)
        {
            break;
        }

        if constexpr (TOPK_BY_BITONIC_SORT)
        {
            copy_buffer_distances(result_indices_buffer + internal_topk, result_distances_buffer + internal_topk,
                                  temp_distances_buffer, temp_indices_buffer, search_width * graph_degree);
            unsigned first_tid = blockDim.x <= search_width * graph_degree ? 0 : search_width * graph_degree;
            set_parents(result_indices_buffer, internal_topk, parent_offsets_buffer, first_tid);
        }

        __sync_workitems();

        if constexpr (!is_same<decltype(sample_filter), none_cagra_sample_filter>::value)
        {
            if (threadIdx.x == 0)
            {
                *filter_flag = 0;
            }
            __sync_workitems();

            constexpr INDEX_T index_msb_1_mask = gen_index_msb_1_mask<INDEX_T>::value;
            const INDEX_T invalid_index = get_max_value<INDEX_T>();

            for (unsigned p = threadIdx.x; p < search_width; p += blockDim.x)
            {
                if (parent_offsets_buffer[p] != invalid_index)
                {
                    const auto parent_id = result_indices_buffer[parent_offsets_buffer[p]] & ~index_msb_1_mask;
                    if (!sample_filter(query_id, parent_id))
                    {
                        result_distances_buffer[parent_offsets_buffer[p]] = get_max_value<DISTANCE_T>();
                        result_indices_buffer[parent_offsets_buffer[p]] = invalid_index;
                        *filter_flag = 1;
                    }
                }
            }
            __sync_workitems();
        }

        iter++;
    }

    if constexpr (!is_same<decltype(sample_filter), none_cagra_sample_filter>::value)
    {
        constexpr INDEX_T index_msb_1_mask = gen_index_msb_1_mask<INDEX_T>::value;
        const INDEX_T invalid_index = get_max_value<INDEX_T>();

        for (unsigned i = threadIdx.x; i < internal_topk + search_width * graph_degree; i += blockDim.x)
        {
            const auto node_id = result_indices_buffer[i] & ~index_msb_1_mask;
            if (node_id != (invalid_index & ~index_msb_1_mask) && !sample_filter(query_id, node_id))
            {
                result_distances_buffer[i] = get_max_value<DISTANCE_T>();
                result_indices_buffer[i] = invalid_index;
            }
        }

        __sync_workitems();
        topk_by_bitonic_sort_1st<MAX_ITOPK + MAX_CANDIDATES>(result_distances_buffer, result_indices_buffer,
                                                             internal_topk + search_width * graph_degree, top_k, false);
        __sync_workitems();
    }

    for (uint32_t i = threadIdx.x; i < top_k; i += blockDim.x)
    {
        unsigned j = i + (top_k * query_id);
        unsigned ii = i;
        if (TOPK_BY_BITONIC_SORT)
        {
            ii = swizzling(i);
        }
        if (result_distances_ptr != nullptr)
        {
            result_distances_ptr[j] = result_distances_buffer[ii];
        }
        constexpr INDEX_T index_msb_1_mask = gen_index_msb_1_mask<INDEX_T>::value;

        result_indices_ptr[j] = result_indices_buffer[ii] & ~index_msb_1_mask;
    }
    if (threadIdx.x == 0 && num_executed_iterations != nullptr)
    {
        num_executed_iterations[query_id] = iter + 1;
    }
}

extern "C" __global__ __aicore__ void cagra_rabitq(GM_ADDR queries_ptr_gm, GM_ADDR knn_graph_gm,
                                                   GM_ADDR visited_hashmap_ptr_gm, GM_ADDR ptr_gm,
                                                   GM_ADDR precompute_all_gm, GM_ADDR code_gm,
                                                   GM_ADDR rotated_qq_ptr_all_gm, GM_ADDR result_distances_ptr_gm,
                                                   GM_ADDR result_indices_ptr_gm, GM_ADDR workspace, GM_ADDR tiling)
{
    __gm__ float *queries_ptr = (__gm__ float *)queries_ptr_gm;
    __gm__ uint32_t *knn_graph = (__gm__ uint32_t *)knn_graph_gm;
    __gm__ uint32_t *visited_hashmap_ptr = (__gm__ uint32_t *)visited_hashmap_ptr_gm;
    __gm__ float *ptr = (__gm__ float *)ptr_gm;
    __gm__ float *precompute_all = (__gm__ float *)precompute_all_gm;
    __gm__ uint8_t *code = (__gm__ uint8_t *)code_gm;
    __gm__ uint8_t *rotated_qq_ptr_all = (__gm__ uint8_t *)rotated_qq_ptr_all_gm;
    __gm__ float *result_distances_ptr = (__gm__ float *)result_distances_ptr_gm;
    __gm__ uint32_t *result_indices_ptr = (__gm__ uint32_t *)result_indices_ptr_gm;
    GET_TILING_DATA(tiling_data, tiling);
    const uint32_t size = tiling_data.size;
    const uint32_t dim = tiling_data.dim;
    const uint32_t graph_degree = tiling_data.graphDegree;
    const uint32_t top_k = tiling_data.topK;
    const uint32_t internal_topk = tiling_data.internalTopK;

    const uint64_t ld = dim;
    constexpr uint64_t rand_xor_mask = 0x0000000000128394;

    constexpr uint32_t num_distillation = 1;
    constexpr uint32_t num_seeds = 0;
    constexpr uint32_t search_width = 1;
    constexpr uint32_t min_iteration = 0;
    uint32_t max_iteration = 64;
    if (internal_topk >= 512)
    {
        // Single-path budget, following cuVS single-CTA planning. Small bitonic searches keep
        // their existing limit. uint64_t keeps the reachable-node estimate from overflowing.
        max_iteration = internal_topk / search_width;
        const uint64_t branching = graph_degree / 2 > 2 ? graph_degree / 2 : 2;
        for (uint64_t reachable = 1; reachable < size; reachable *= branching)
        {
            ++max_iteration;
        }
    }

    std::uint32_t adding_pref = 25;
    std::uint32_t patience = 16;
    const uint32_t qb = 6;
    const uint32_t rotated_qq_size = (dim + 7) / 8 * qb;

    constexpr uint32_t TEAM_SIZE = 4;
    constexpr uint32_t DATASET_BLOCK_DIM = 128;
    static_assert(CAGRA_MAX_LOCAL_MEMORY_BYTES <= CAGRA_LOCAL_MEMORY_SIZE, "CAGRA UB overflow");

    none_cagra_sample_filter sample_filter;

    using LOAD_T = uint4;
    using DATA_T = float;
    using INDEX_T = uint32_t;
    using DISTANCE_T = float;
    using QUERY_T = float;

    __ubuf__ uint32_t *smem = (__ubuf__ uint32_t *)get_imm(0x0);
    __gm__ const uint32_t *seed_ptr = nullptr;
    __gm__ uint32_t *const num_executed_iterations = nullptr;

    uint32_t result_buffer_size = internal_topk + (search_width * graph_degree);
    uint32_t result_buffer_size_32 = result_buffer_size;
    if (result_buffer_size % 32)
    {
        result_buffer_size_32 += 32 - (result_buffer_size % 32);
    }

    const uint32_t hash_table_size = (size + 31) / 32;
    const auto query_smem_buffer_length = ((dim + DATASET_BLOCK_DIM - 1) / DATASET_BLOCK_DIM) * DATASET_BLOCK_DIM;
    __ubuf__ QUERY_T *query_buffer = reinterpret_cast<__ubuf__ QUERY_T *>(smem);
    __ubuf__ uint8_t *rotated_qq_buffer = reinterpret_cast<__ubuf__ uint8_t *>(query_buffer + query_smem_buffer_length);
    __ubuf__ INDEX_T *result_indices_buffer = reinterpret_cast<__ubuf__ INDEX_T *>(rotated_qq_buffer + rotated_qq_size);

    __ubuf__ DISTANCE_T *result_distances_buffer =
        reinterpret_cast<__ubuf__ DISTANCE_T *>(result_indices_buffer + result_buffer_size_32);

    __ubuf__ INDEX_T *temp_indices_buffer =
        reinterpret_cast<__ubuf__ INDEX_T *>(result_distances_buffer + result_buffer_size_32);

    __ubuf__ DISTANCE_T *temp_distances_buffer =
        reinterpret_cast<__ubuf__ DISTANCE_T *>(temp_indices_buffer + (search_width * graph_degree));

    __ubuf__ INDEX_T *parent_offsets_buffer =
        reinterpret_cast<__ubuf__ INDEX_T *>(temp_distances_buffer + (search_width * graph_degree));

    __ubuf__ INDEX_T *candidate_indices_buffer =
        reinterpret_cast<__ubuf__ INDEX_T *>(parent_offsets_buffer + search_width);

    __ubuf__ DISTANCE_T *candidate_distances_buffer =
        reinterpret_cast<__ubuf__ DISTANCE_T *>(candidate_indices_buffer + (search_width * graph_degree + 31) / 32);
    __ubuf__ uint8_t *distance_work_buffer_ptr =
        reinterpret_cast<__ubuf__ uint8_t *>(candidate_distances_buffer + (search_width * graph_degree + 31) / 32);
    __ubuf__ uint32_t *topk_ws = reinterpret_cast<__ubuf__ uint32_t *>(distance_work_buffer_ptr);
    __ubuf__ uint32_t *terminate_flag = reinterpret_cast<__ubuf__ uint32_t *>(topk_ws + 3);
    __ubuf__ uint32_t *count_flag = reinterpret_cast<__ubuf__ uint32_t *>(terminate_flag + 1);
    __ubuf__ DISTANCE_T *distance_flag = reinterpret_cast<__ubuf__ DISTANCE_T *>(count_flag + 1);
    __ubuf__ uint32_t *smem_work_ptr = reinterpret_cast<__ubuf__ uint32_t *>(distance_flag + 1);

    __ubuf__ uint32_t *filter_flag = terminate_flag;

#define LAUNCH_CAGRA_RABITQ(MAX_TOPK, MAX_CANDIDATES)                                                                  \
    Simt::VF_CALL<                                                                                                     \
        Simtsearch<TEAM_SIZE, DATASET_BLOCK_DIM, MAX_TOPK, MAX_CANDIDATES, (MAX_TOPK < 512), float, uint32_t, float>>( \
        dim3{256, 1, 1}, precompute_all, code, rotated_qq_ptr_all, qb, rotated_qq_size, rotated_qq_buffer,             \
        distance_flag, queries_ptr, dim, query_buffer, query_smem_buffer_length, terminate_flag, topk_ws,              \
        hash_table_size, seed_ptr, num_seeds, result_indices_buffer, result_distances_buffer, result_buffer_size,      \
        num_distillation, rand_xor_mask, ptr, ld, size, knn_graph, graph_degree, count_flag,                           \
        candidate_distances_buffer, candidate_indices_buffer, parent_offsets_buffer, temp_indices_buffer,              \
        temp_distances_buffer, patience, internal_topk, search_width, max_iteration, min_iteration, top_k,             \
        result_distances_ptr, result_indices_ptr, num_executed_iterations, smem_work_ptr, adding_pref, filter_flag,    \
        visited_hashmap_ptr)

    if (TILING_KEY_IS(640064))
    {
        LAUNCH_CAGRA_RABITQ(64, 64);
    }
    else if (TILING_KEY_IS(640128))
    {
        LAUNCH_CAGRA_RABITQ(128, 64);
    }
    else if (TILING_KEY_IS(640256))
    {
        LAUNCH_CAGRA_RABITQ(256, 64);
    }
    else if (TILING_KEY_IS(640512))
    {
        LAUNCH_CAGRA_RABITQ(512, 64);
    }
    else if (TILING_KEY_IS(641024))
    {
        LAUNCH_CAGRA_RABITQ(1024, 64);
    }
    else if (TILING_KEY_IS(1280064))
    {
        LAUNCH_CAGRA_RABITQ(64, 128);
    }
    else if (TILING_KEY_IS(1280128))
    {
        LAUNCH_CAGRA_RABITQ(128, 128);
    }
    else if (TILING_KEY_IS(1280256))
    {
        LAUNCH_CAGRA_RABITQ(256, 128);
    }
    else if (TILING_KEY_IS(1280512))
    {
        LAUNCH_CAGRA_RABITQ(512, 128);
    }
    else if (TILING_KEY_IS(1281024))
    {
        LAUNCH_CAGRA_RABITQ(1024, 128);
    }
    else
    {
        Simt::VF_CALL<CagraRabitqUnsupportedTilingKey>(dim3{32, 1, 1});
    }
#undef LAUNCH_CAGRA_RABITQ
}
