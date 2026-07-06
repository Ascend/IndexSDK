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
#ifndef CAGRA_RABITQ_TOPK_H
#define CAGRA_RABITQ_TOPK_H
#include "cagra_rabitq_distance.h"

template <unsigned MAX_ITOPK, class IdxT = void>
__forceinline__[aicore] void topk_by_bitonic_sort_2nd(__ubuf__ float *itopk_distances, __ubuf__ IdxT *itopk_indices,
                                                      const uint32_t num_itopk, __ubuf__ float *candidate_distances,
                                                      __ubuf__ IdxT *candidate_indices, const uint32_t num_candidates,
                                                      __ubuf__ uint32_t *work_buf, const bool first,
                                                      unsigned MULTI_WARPS = 0)
{
    const unsigned lane_id = threadIdx.x % 32;
    const unsigned warp_id = threadIdx.x / 32;
    if (MULTI_WARPS == 0)
    {
        if (warp_id > 0)
        {
            return;
        }
        constexpr unsigned N = (MAX_ITOPK + 31) / 32;
        float key[N];
        IdxT val[N];
        if (first)
        {
            for (unsigned i = 0; i < N; i++)
            {
                unsigned j = lane_id + (32 * i);
                if (j < num_itopk)
                {
                    key[i] = itopk_distances[j];
                    val[i] = itopk_indices[j];
                }
                else
                {
                    key[i] = get_max_value<float>();
                    val[i] = get_max_value<IdxT>();
                }
            }
            warp_sort<float, IdxT, N>(key, val);
        }
        else
        {
            for (unsigned i = 0; i < N; i++)
            {
                unsigned j = (N * lane_id) + i;
                if (j < num_itopk)
                {
                    key[i] = itopk_distances[swizzling(j)];
                    val[i] = itopk_indices[swizzling(j)];
                }
                else
                {
                    key[i] = get_max_value<float>();
                    val[i] = get_max_value<IdxT>();
                }
            }
        }

        for (unsigned i = 0; i < N; i++)
        {
            unsigned j = (N * lane_id) + i;
            unsigned k = MAX_ITOPK - 1 - j;
            if (k >= num_itopk || k >= num_candidates) continue;
            float candidate_key = candidate_distances[swizzling(k)];
            if (key[i] > candidate_key)
            {
                key[i] = candidate_key;
                val[i] = candidate_indices[swizzling(k)];
            }
        }

        warp_merge<float, IdxT, N>(key, val, 32);

        for (unsigned i = 0; i < N; i++)
        {
            unsigned j = (N * lane_id) + i;
            if (j < num_itopk)
            {
                itopk_distances[swizzling(j)] = key[i];
                itopk_indices[swizzling(j)] = val[i];
            }
        }
    }
    else
    {
        constexpr unsigned max_itopk_per_warp = (MAX_ITOPK + 1) / 2;
        constexpr unsigned N = (max_itopk_per_warp + 31) / 32;
        float key[N];
        IdxT val[N];
        if (first)
        {
            if (warp_id < 2)
            {
                for (unsigned i = 0; i < N; i++)
                {
                    unsigned j = lane_id + (32 * i) + (max_itopk_per_warp * warp_id);
                    if (j < num_itopk)
                    {
                        key[i] = itopk_distances[j];
                        val[i] = itopk_indices[j];
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
                    unsigned j = (N * threadIdx.x) + i;
                    if (j >= num_itopk) continue;
                    itopk_distances[swizzling(j)] = key[i];
                    itopk_indices[swizzling(j)] = val[i];
                }
            }
            __sync_workitems();
            if (warp_id < 2)
            {
                for (unsigned i = 0; i < N; i++)
                {
                    unsigned j = (N * threadIdx.x) + i;
                    unsigned k = MAX_ITOPK - 1 - j;
                    if (k >= num_itopk) continue;
                    float temp_key = itopk_distances[swizzling(k)];
                    if (key[i] == temp_key) continue;
                    if ((warp_id == 0) == (key[i] > temp_key))
                    {
                        key[i] = temp_key;
                        val[i] = itopk_indices[swizzling(k)];
                    }
                }
                warp_merge<float, IdxT, N>(key, val, 32);
            }
            __sync_workitems();

            if (warp_id < 2)
            {
                for (unsigned i = 0; i < N; i++)
                {
                    unsigned j = (N * threadIdx.x) + i;
                    if (j >= num_itopk) continue;
                    itopk_distances[swizzling(j)] = key[i];
                    itopk_indices[swizzling(j)] = val[i];
                }
            }
        }
        const uint32_t num_itopk_div2 = num_itopk / 2;
        if (threadIdx.x < 3)
        {
            work_buf[threadIdx.x] = num_itopk_div2;
        }
        __sync_workitems();
        for (unsigned k = threadIdx.x; k < min(num_candidates, num_itopk); k += blockDim.x)
        {
            const unsigned j = num_itopk - 1 - k;
            const float itopk_key = itopk_distances[swizzling(j)];
            const float candidate_key = candidate_distances[swizzling(k)];
            if (itopk_key > candidate_key)
            {
                itopk_distances[swizzling(j)] = candidate_key;
                itopk_indices[swizzling(j)] = candidate_indices[swizzling(k)];
                if (j < num_itopk_div2)
                {
                    atomicMin(work_buf + 2, j);
                }
                else
                {
                    atomicMin(work_buf + 1, j - num_itopk_div2);
                }
            }
        }
        __sync_workitems();
        for (unsigned j = threadIdx.x; j < num_itopk_div2; j += blockDim.x)
        {
            const unsigned k = j + num_itopk_div2;
            float key_0 = itopk_distances[swizzling(j)];
            float key_1 = itopk_distances[swizzling(k)];
            if (key_0 > key_1)
            {
                itopk_distances[swizzling(j)] = key_1;
                itopk_distances[swizzling(k)] = key_0;
                IdxT val_0 = itopk_indices[swizzling(j)];
                IdxT val_1 = itopk_indices[swizzling(k)];
                itopk_indices[swizzling(j)] = val_1;
                itopk_indices[swizzling(k)] = val_0;
                atomicMin(work_buf + 0, j);
            }
        }
        if (threadIdx.x == blockDim.x - 1)
        {
            if (work_buf[2] < num_itopk_div2)
            {
                work_buf[1] = work_buf[2];
            }
        }
        __sync_workitems();

        if (warp_id < 2)
        {
            const uint32_t turning_point = work_buf[warp_id];
            for (unsigned i = 0; i < N; i++)
            {
                unsigned k = num_itopk;
                unsigned j = (N * lane_id) + i;
                if (j < turning_point)
                {
                    k = j + (num_itopk_div2 * warp_id);
                }
                else if (j >= (MAX_ITOPK / 2 - num_itopk_div2))
                {
                    j -= (MAX_ITOPK / 2 - num_itopk_div2);
                    if ((turning_point <= j) && (j < num_itopk_div2))
                    {
                        k = j + (num_itopk_div2 * warp_id);
                    }
                }
                if (k < num_itopk)
                {
                    key[i] = itopk_distances[swizzling(k)];
                    val[i] = itopk_indices[swizzling(k)];
                }
                else
                {
                    key[i] = get_max_value<float>();
                    val[i] = get_max_value<IdxT>();
                }
            }
            warp_merge<float, IdxT, N>(key, val, 32);
            for (unsigned i = 0; i < N; i++)
            {
                const unsigned j = (N * lane_id) + i;
                if (j < num_itopk_div2)
                {
                    unsigned k = j + (num_itopk_div2 * warp_id);
                    itopk_distances[swizzling(k)] = key[i];
                    itopk_indices[swizzling(k)] = val[i];
                }
            }
        }
    }
}

template <unsigned MAX_ITOPK, unsigned MAX_CANDIDATES, class IdxT>
__forceinline__[aicore] void topk_by_bitonic_sort(__ubuf__ float *itopk_distances, __ubuf__ IdxT *itopk_indices,
                                                  const uint32_t num_itopk, __ubuf__ float *candidate_distances,
                                                  __ubuf__ IdxT *candidate_indices, const uint32_t num_candidates,
                                                  __ubuf__ uint32_t *work_buf, const bool first,
                                                  const unsigned MULTI_WARPS_1, const unsigned MULTI_WARPS_2,
                                                  __ubuf__ std::uint32_t *count_flag, std::uint32_t add_perf)
{
    if (num_candidates == 0)
    {
        return;
    }
    if (add_perf >= num_itopk)
    {
        return;
    }
    topk_by_bitonic_sort_1st<MAX_CANDIDATES, IdxT>(candidate_distances, candidate_indices, num_candidates, num_itopk,
                                                   MULTI_WARPS_1);
    if (threadIdx.x == 0 && candidate_distances[0] >= itopk_distances[add_perf])
    {
        count_flag[0] = 1;
    }
    topk_by_bitonic_sort_2nd<MAX_ITOPK, IdxT>(itopk_distances, itopk_indices, num_itopk, candidate_distances,
                                              candidate_indices, num_candidates, work_buf, first, MULTI_WARPS_2);
}

template <class IdxT>
__forceinline__[aicore] void get_min_k(__ubuf__ float *candidate_distances, __ubuf__ IdxT *candidate_indices,
                                       __ubuf__ float *min_k_distances, __ubuf__ IdxT *min_k_indices,
                                       const uint32_t num_candidates, const uint32_t first_tid = 0)
{
    if (threadIdx.x < first_tid || threadIdx.x >= first_tid + 32) return;

    const unsigned lane_id = threadIdx.x % 32;
    constexpr unsigned N = 2;
    float value[N];
    for (unsigned i = 0; i < N; ++i)
    {
        unsigned j = lane_id + (32 * i);
        if (j < num_candidates)
        {
            value[i] = candidate_distances[j];
        }
        else
        {
            value[i] = get_max_value<float>();
        }
    }

    if (value[0] > value[1])
    {
        value[0] = value[1];
    }

    value[0] = __reduce_min(value[0]);

    if (threadIdx.x == first_tid)
    {
        IdxT index = get_max_value<IdxT>();
        for (unsigned i = 0; i < num_candidates; ++i)
        {
            if (candidate_distances[i] == value[0])
            {
                min_k_indices[0] = candidate_indices[i];
                break;
            }
        }
        min_k_distances[0] = value[0];
    }
}

template <unsigned num_bins, unsigned blockDim_x>
__forceinline__[aicore] void select_best_index_helper(uint32_t &my_index, uint32_t &my_csum,
                                                      __ubuf__ const uint32_t *const hist,
                                                      const uint32_t nx_below_threshold, const uint32_t max_threshold,
                                                      const uint32_t threshold, const uint32_t shift,
                                                      const uint32_t topk)
{
    if constexpr (blockDim_x <= num_bins)
    {
        constexpr int n_data = num_bins / blockDim_x;
        uint32_t csum[n_data];
        for (int i = 0; i < n_data; i++)
        {
            csum[i] = hist[i + (n_data * threadIdx.x)];
        }
        inclusive_block_scan(csum);
        for (int i = n_data - 1; i >= 0; i--)
        {
            if (nx_below_threshold + csum[i] > topk) continue;
            const uint32_t index = i + (n_data * threadIdx.x);
            if (threshold + (index << shift) > max_threshold) continue;
            my_index = index;
            my_csum = csum[i];
            break;
        }
    }
}

template <unsigned blockDim_x>
__forceinline__[aicore] void select_best_index_for_next_threshold_core(
    uint32_t &my_index, uint32_t &my_csum, const unsigned num_bins, __ubuf__ const uint32_t *const hist,
    const uint32_t nx_below_threshold, const uint32_t max_threshold, const uint32_t threshold, const uint32_t shift,
    const uint32_t topk)
{
    if (num_bins == 2048)
    {
        select_best_index_helper<2048, blockDim_x>(my_index, my_csum, hist, nx_below_threshold, max_threshold,
                                                   threshold, shift, topk);
    }
    else if (num_bins == 1024)
    {
        select_best_index_helper<1024, blockDim_x>(my_index, my_csum, hist, nx_below_threshold, max_threshold,
                                                   threshold, shift, topk);
    }
    else if (num_bins == 256)
    {
        select_best_index_helper<256, blockDim_x>(my_index, my_csum, hist, nx_below_threshold, max_threshold, threshold,
                                                  shift, topk);
    }
}

__forceinline__[aicore] void select_best_index_for_next_threshold(const uint32_t topk, const uint32_t threshold,
                                                                  const uint32_t max_threshold,
                                                                  const uint32_t nx_below_threshold,
                                                                  const uint32_t num_bins, const uint32_t shift,
                                                                  __ubuf__ const uint32_t *const hist,  // [num_bins]
                                                                  __ubuf__ uint32_t *const best_index,
                                                                  __ubuf__ uint32_t *const best_csum)
{
    uint32_t my_index = 0xffffffff;
    uint32_t my_csum = 0;
    if (num_bins <= blockDim.x)
    {
        uint32_t csum = 0;
        if (threadIdx.x < num_bins)
        {
            csum = hist[threadIdx.x];
        }
        ScanBlockInclusive(csum);

        if (threadIdx.x < num_bins)
        {
            const uint32_t index = threadIdx.x;
            if ((nx_below_threshold + csum <= topk) && (threshold + (index << shift) <= max_threshold))
            {
                my_index = index;
                my_csum = csum;
            }
        }
    }
    else
    {
        switch (blockDim.x)
        {
            case 32:
                select_best_index_for_next_threshold_core<32>(my_index, my_csum, num_bins, hist, nx_below_threshold,
                                                              max_threshold, threshold, shift, topk);
                break;
            case 64:
                select_best_index_for_next_threshold_core<64>(my_index, my_csum, num_bins, hist, nx_below_threshold,
                                                              max_threshold, threshold, shift, topk);
                break;
            case 128:
                select_best_index_for_next_threshold_core<128>(my_index, my_csum, num_bins, hist, nx_below_threshold,
                                                               max_threshold, threshold, shift, topk);
                break;
            case 256:
                select_best_index_for_next_threshold_core<256>(my_index, my_csum, num_bins, hist, nx_below_threshold,
                                                               max_threshold, threshold, shift, topk);
                break;
            case 512:
                select_best_index_for_next_threshold_core<512>(my_index, my_csum, num_bins, hist, nx_below_threshold,
                                                               max_threshold, threshold, shift, topk);
                break;
            case 1024:
                select_best_index_for_next_threshold_core<1024>(my_index, my_csum, num_bins, hist, nx_below_threshold,
                                                                max_threshold, threshold, shift, topk);
                break;
            default:
                break;
        }
    }
    if (threadIdx.x < num_bins)
    {
        const int laneid = 31 - __clz((int)__ballot((my_index != 0xffffffff)));
        if ((threadIdx.x & 0x1f) == laneid)
        {
            const uint32_t old_index = atomicMax(best_index, my_index);
            if (old_index < my_index)
            {
                atomicMax(best_csum, my_csum);
            }
        }
    }
    __sync_workitems();
}

template <typename T, int stateBitLen, int vecLen>
__forceinline__[aicore] void output_index_below_threshold(const uint32_t topk, const uint32_t thread_id,
                                                          const uint32_t num_threads, const uint32_t threshold,
                                                          const uint32_t nx_below_threshold,
                                                          __ubuf__ const T *const x,  // [nx,]
                                                          const uint32_t nx, __ubuf__ const uint8_t *state,
                                                          __ubuf__ uint32_t *const output,  // [topk]
                                                          __ubuf__ uint32_t *const output_count,
                                                          __ubuf__ uint32_t *const output_count_eq)
{
    int ii = 0;
    for (int i = thread_id * vecLen; i < nx; i += num_threads * max(vecLen, stateBitLen), ii++)
    {
        uint8_t iState = 0;
        if (stateBitLen == 8)
        {
            iState = state[thread_id + (num_threads * ii)];
            if (iState == (uint8_t)0xff) continue;
        }
#pragma unroll
        for (int v = 0; v < max(vecLen, stateBitLen); v += vecLen)
        {
            const int iv = i + (num_threads * v);
            if (iv >= nx) break;

            struct u32_vector u32_vec;
            struct u16_vector u16_vec;
            if (sizeof(T) == 4)
            {
                load_u32_vector<vecLen>(u32_vec, (__ubuf__ const uint32_t *)x, iv);
            }
            else
            {
                load_u16_vector<vecLen>(u16_vec, (__ubuf__ const uint16_t *)x, iv);
            }
#pragma unroll
            for (int u = 0; u < vecLen; u++)
            {
                const int ivu = iv + u;
                if (ivu >= nx) break;

                const uint8_t mask = (uint8_t)0x1 << (v + u);
                if ((stateBitLen == 8) && (iState & mask)) continue;

                uint32_t xi;
                if (sizeof(T) == 4)
                {
                    xi = get_element_from_u32_vector<vecLen>(u32_vec, u);
                }
                else
                {
                    xi = get_element_from_u16_vector<vecLen>(u16_vec, u);
                }
                if (xi < threshold)
                {
                    output[atomicAdd(output_count, 1)] = ivu;
                }
                else if (xi == threshold)
                {
                    if (nx_below_threshold + atomicAdd(output_count_eq, 1) < topk)
                    {
                        output[atomicAdd(output_count, 1)] = ivu;
                    }
                }
            }
        }
    }
}

template <int maxTopk, int numSortThreads, class ValT>
__forceinline__[aicore] void sort_topk_results(uint32_t topk, __ubuf__ const uint32_t *x, __ubuf__ const ValT *in_vals,
                                               __ubuf__ uint32_t *y, __ubuf__ ValT *out_vals,
                                               __ubuf__ uint32_t *smem_out_vals, __ubuf__ uint32_t *_smem)
{
    const uint32_t thread_id = threadIdx.x;
    constexpr int numTopkPerThread = maxTopk / numSortThreads;

    float my_keys[numTopkPerThread];
    ValT my_vals[numTopkPerThread];

    if (thread_id < numSortThreads)
    {
        for (int i = 0; i < numTopkPerThread; i++)
        {
            const int k = thread_id + (numSortThreads * i);
            if (k < topk)
            {
                const int j = smem_out_vals[k];
                my_keys[i] = ((__ubuf__ float *)x)[j];
                if (in_vals)
                {
                    my_vals[i] = in_vals[j];
                }
                else
                {
                    my_vals[i] = j;
                }
            }
            else
            {
                my_keys[i] = kFloatMax;
                my_vals[i] = ~static_cast<ValT>(0);
            }
        }
    }

    uint32_t mask = 1;
    if (thread_id < numSortThreads)
    {
        const bool ascending = ((thread_id & mask) == 0);
        if (numTopkPerThread == 3)
        {
            swap_if_needed<float, ValT>(my_keys[0], my_keys[1], my_vals[0], my_vals[1], ascending);
            swap_if_needed<float, ValT>(my_keys[0], my_keys[2], my_vals[0], my_vals[2], ascending);
            swap_if_needed<float, ValT>(my_keys[1], my_keys[2], my_vals[1], my_vals[2], ascending);
        }
        else
        {
            for (int j = 0; j < numTopkPerThread / 2; j += 1)
            {
#pragma unroll
                for (int i = 0; i < numTopkPerThread; i += 2)
                {
                    swap_if_needed<float, ValT>(my_keys[i], my_keys[i + 1], my_vals[i], my_vals[i + 1], ascending);
                }
#pragma unroll
                for (int i = 1; i < numTopkPerThread - 1; i += 2)
                {
                    swap_if_needed<float, ValT>(my_keys[i], my_keys[i + 1], my_vals[i], my_vals[i + 1], ascending);
                }
            }
        }
    }

    while (mask < numSortThreads)
    {
        uint32_t next_mask = mask << 1;

        for (uint32_t curr_mask = mask; curr_mask > 0; curr_mask >>= 1)
        {
            const bool ascending = ((thread_id & curr_mask) == 0) == ((thread_id & next_mask) == 0);
            if (curr_mask >= 32)
            {
                __ubuf__ ValT *const smem_vals = reinterpret_cast<__ubuf__ ValT *>(_smem);
                __ubuf__ float *const smem_keys = reinterpret_cast<__ubuf__ float *>(smem_vals + maxTopk);
                __sync_workitems();
                if (thread_id < numSortThreads)
                {
#pragma unroll
                    for (int i = 0; i < numTopkPerThread; i++)
                    {
                        smem_keys[thread_id + (numSortThreads * i)] = my_keys[i];
                        smem_vals[thread_id + (numSortThreads * i)] = my_vals[i];
                    }
                }
                __sync_workitems();
                if (thread_id < numSortThreads)
                {
#pragma unroll
                    for (int i = 0; i < numTopkPerThread; i++)
                    {
                        float opp_key = smem_keys[(thread_id ^ curr_mask) + (numSortThreads * i)];
                        ValT opp_val = smem_vals[(thread_id ^ curr_mask) + (numSortThreads * i)];
                        swap_if_needed<float, ValT>(my_keys[i], opp_key, my_vals[i], opp_val, ascending);
                    }
                }
            }
            else
            {
                if (thread_id < numSortThreads)
                {
#pragma unroll
                    for (int i = 0; i < numTopkPerThread; i++)
                    {
                        float opp_key = __shfl_xor(my_keys[i], curr_mask, 32);
                        ValT opp_val = __shfl_xor(my_vals[i], curr_mask, 32);
                        swap_if_needed<float, ValT>(my_keys[i], opp_key, my_vals[i], opp_val, ascending);
                    }
                }
            }
        }

        if (thread_id < numSortThreads)
        {
            const bool ascending = ((thread_id & next_mask) == 0);
            if (numTopkPerThread == 3)
            {
                swap_if_needed<float, ValT>(my_keys[0], my_keys[1], my_vals[0], my_vals[1], ascending);
                swap_if_needed<float, ValT>(my_keys[0], my_keys[2], my_vals[0], my_vals[2], ascending);
                swap_if_needed<float, ValT>(my_keys[1], my_keys[2], my_vals[1], my_vals[2], ascending);
            }
            else
            {
#pragma unroll
                for (uint32_t curr_mask = numTopkPerThread / 2; curr_mask > 0; curr_mask >>= 1)
                {
#pragma unroll
                    for (int i = 0; i < numTopkPerThread; i++)
                    {
                        const int j = i ^ curr_mask;
                        if (i > j) continue;
                        swap_if_needed<float, ValT>(my_keys[i], my_keys[j], my_vals[i], my_vals[j], ascending);
                    }
                }
            }
        }
        mask = next_mask;
    }

    if (thread_id < numSortThreads)
    {
        for (int i = 0; i < numTopkPerThread; i++)
        {
            const int k = i + (numTopkPerThread * thread_id);
            if (k < topk)
            {
                if (y)
                {
                    y[k] = reinterpret_cast<uint32_t *>(my_keys)[i];
                }
                if (out_vals)
                {
                    out_vals[k] = my_vals[i];
                }
            }
        }
    }
}

template <int stateBitLen, int vecLen, int maxTopk, int numSortThreads, class ValT>
__forceinline__[aicore] void topk_cta_11_core(uint32_t topk, uint32_t len_x, __ubuf__ const uint32_t *_x,
                                              __ubuf__ const ValT *_in_vals, __ubuf__ uint32_t *_y,
                                              __ubuf__ ValT *_out_vals, __ubuf__ uint8_t *_state,
                                              __ubuf__ uint32_t *_hint, bool sort, __ubuf__ uint32_t *_smem)
{
    __ubuf__ uint32_t *const smem_out_vals = _smem;
    __ubuf__ uint32_t *const hist = &(_smem[2 * maxTopk]);
    __ubuf__ uint32_t *const best_index = &(_smem[2 * maxTopk + 2048]);
    __ubuf__ uint32_t *const best_csum = &(_smem[2 * maxTopk + 2048 + 3]);

    const uint32_t num_threads = blockDim.x;
    const uint32_t thread_id = threadIdx.x;
    uint32_t nx = len_x;
    __ubuf__ const uint32_t *const x = _x;
    __ubuf__ const ValT *in_vals = NULL;
    if (_in_vals)
    {
        in_vals = _in_vals;
    }
    __ubuf__ uint32_t *y = NULL;
    if (_y)
    {
        y = _y;
    }
    __ubuf__ ValT *out_vals = NULL;
    if (_out_vals)
    {
        out_vals = _out_vals;
    }
    __ubuf__ uint8_t *state = _state;
    const uint32_t hint = (_hint == NULL ? ~0u : *_hint);

    for (int i = 2 * maxTopk + thread_id; i < 2 * maxTopk + 2048 + 8; i += num_threads)
    {
        _smem[i] = 0;
    }
    __ubuf__ uint32_t *const output_count = &(_smem[2 * maxTopk + 2048 + 6]);
    __ubuf__ uint32_t *const output_count_eq = &(_smem[2 * maxTopk + 2048 + 7]);
    uint32_t threshold = 0;
    uint32_t nx_below_threshold = 0;
    __sync_workitems();

#pragma unroll
    for (int j = 0; j < 3; j += 1)
    {
        uint32_t num_bins;
        uint32_t shift;

        update_histogram<uint32_t, stateBitLen, vecLen>(j, thread_id, num_threads, hint, threshold, num_bins, shift, x,
                                                        nx, hist, state, smem_out_vals, output_count);
        select_best_index_for_next_threshold(topk, threshold, hint, nx_below_threshold, num_bins, shift, hist,
                                             best_index + j, best_csum + j);

        threshold += (best_index[j] << shift);
        nx_below_threshold += best_csum[j];
        if (nx_below_threshold == topk) break;
    }

    if ((_hint != NULL) && (thread_id == 0))
    {
        *_hint = min(threshold, hint);
    }

    output_index_below_threshold<uint32_t, stateBitLen, vecLen>(topk, thread_id, num_threads, threshold,
                                                                nx_below_threshold, x, nx, state, smem_out_vals,
                                                                output_count, output_count_eq);
    __sync_workitems();

    if (!sort)
    {
        for (int k = thread_id; k < topk; k += blockDim.x)
        {
            const uint32_t i = smem_out_vals[k];
            if (y)
            {
                y[k] = x[i];
            }
            if (out_vals)
            {
                if (in_vals)
                {
                    out_vals[k] = in_vals[i];
                }
                else
                {
                    out_vals[k] = i;
                }
            }
        }
        return;
    }

    sort_topk_results<maxTopk, numSortThreads, ValT>(topk, x, in_vals, y, out_vals, smem_out_vals, _smem);
}

template <unsigned MAX_INTERNAL_TOPK>
struct topk_by_radix_sort_base
{
    static constexpr uint32_t smem_size = MAX_INTERNAL_TOPK * 2 + 2048 + 8;
    static constexpr uint32_t state_bit_length = 0;
    static constexpr uint32_t vecLen = 2;
};
template <unsigned MAX_INTERNAL_TOPK, class IdxT, class = void>
struct topk_by_radix_sort : topk_by_radix_sort_base<MAX_INTERNAL_TOPK>
{
};

template <unsigned MAX_INTERNAL_TOPK, class IdxT>
struct topk_by_radix_sort<MAX_INTERNAL_TOPK, IdxT, enable_if_t<((MAX_INTERNAL_TOPK <= 64))>>
    : topk_by_radix_sort_base<MAX_INTERNAL_TOPK>
{
    __forceinline__[aicore] void operator()(uint32_t topk, uint32_t batch_size, uint32_t len_x,
                                            __ubuf__ const uint32_t *_x, __ubuf__ const IdxT *_in_vals,
                                            __ubuf__ uint32_t *_y, __ubuf__ IdxT *_out_vals, __ubuf__ uint32_t *work,
                                            __ubuf__ uint32_t *_hints, bool sort, __ubuf__ uint32_t *_smem)
    {
        __ubuf__ uint8_t *const state = reinterpret_cast<__ubuf__ uint8_t *>(work);
        topk_cta_11_core<topk_by_radix_sort_base<MAX_INTERNAL_TOPK>::state_bit_length,
                         topk_by_radix_sort_base<MAX_INTERNAL_TOPK>::vecLen, 64, 32, IdxT>(
            topk, len_x, _x, _in_vals, _y, _out_vals, state, _hints, sort, _smem);
    }
};

#define TOP_FUNC_PARTIAL_SPECIALIZATION(V)                                                                     \
    template <unsigned MAX_INTERNAL_TOPK, class IdxT>                                                          \
    struct topk_by_radix_sort<MAX_INTERNAL_TOPK, IdxT,                                                         \
                              enable_if_t<((MAX_INTERNAL_TOPK <= V) && (2 * MAX_INTERNAL_TOPK > V))>>          \
        : topk_by_radix_sort_base<MAX_INTERNAL_TOPK>                                                           \
    {                                                                                                          \
        __forceinline__[aicore] void operator()(uint32_t topk, uint32_t batch_size, uint32_t len_x,            \
                                                __ubuf__ const uint32_t *_x, __ubuf__ const IdxT *_in_vals,    \
                                                __ubuf__ uint32_t *_y, __ubuf__ IdxT *_out_vals,               \
                                                __ubuf__ uint32_t *work, __ubuf__ uint32_t *_hints, bool sort, \
                                                __ubuf__ uint32_t *_smem)                                      \
        {                                                                                                      \
            __ubuf__ uint8_t *state = (__ubuf__ uint8_t *)work;                                                \
            topk_cta_11_core<topk_by_radix_sort_base<MAX_INTERNAL_TOPK>::state_bit_length,                     \
                             topk_by_radix_sort_base<MAX_INTERNAL_TOPK>::vecLen, V, V / 4, IdxT>(              \
                topk, len_x, _x, _in_vals, _y, _out_vals, state, _hints, sort, _smem);                         \
        }                                                                                                      \
    }
TOP_FUNC_PARTIAL_SPECIALIZATION(128);
TOP_FUNC_PARTIAL_SPECIALIZATION(256);
TOP_FUNC_PARTIAL_SPECIALIZATION(512);
TOP_FUNC_PARTIAL_SPECIALIZATION(1024);

template <class INDEX_T>
__forceinline__[aicore] void copy_buffer_indices(__ubuf__ INDEX_T *candidates_indices,
                                                 __ubuf__ INDEX_T *temp_candidates_indices,
                                                 const uint32_t candidates_size)
{
    for (uint32_t i = threadIdx.x; i < candidates_size; i += blockDim.x)
    {
        INDEX_T child_id = temp_candidates_indices[i];
        candidates_indices[i] = child_id;
    }
}

#endif  // CAGRA_RABITQ_TOPK_H
