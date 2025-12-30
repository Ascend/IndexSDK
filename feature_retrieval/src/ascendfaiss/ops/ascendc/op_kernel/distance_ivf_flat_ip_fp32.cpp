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

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/matmul.h"

using namespace AscendC;

namespace kernels {
    template <typename T>
    class DistanceIVFFlatIpFP32 {
    public:
        __aicore__ inline DistanceIVFFlatIpFP32(TPipe *pipe_in) {pipe_ = pipe_in;};
        __aicore__ inline void Init(GM_ADDR query, GM_ADDR base, GM_ADDR offset, GM_ADDR actual_size,
        GM_ADDR result, GM_ADDR max_result, GM_ADDR flag, GM_ADDR usrWorkspace,
        const DistanceIVFFlatIpFP32TilingData *tiling_data) {
            core_id = GetBlockIdx();
            tiling_ = *tiling_data;

            query_gm_.SetGlobalBuffer((__gm__ T *)query);
            offset_gm_.SetGlobalBuffer((__gm__ uint64_t *)offset);
            actual_size_gm_.SetGlobalBuffer((__gm__ uint32_t *)actual_size);
            uint64_t code_offset = offset_gm_.GetValue(core_id);
            actual_size_val = actual_size_gm_.GetValue(core_id);
            code_gm_.SetGlobalBuffer((__gm__ T *)(base + code_offset));
            result_gm_.SetGlobalBuffer((__gm__ T *)result);
            max_res_gm_.SetGlobalBuffer((__gm__ T *)max_result);
            flag_gm_.SetGlobalBuffer((__gm__ uint16_t *)flag);
            mm_out_gm_.SetGlobalBuffer((__gm__ T *)(usrWorkspace + tiling_.buffer_size * 2 * core_id));

            pipe_->InitBuffer(in_que, 1, tiling_.per_loop_process_len * sizeof(T));
            pipe_->InitBuffer(out_que, 1, tiling_.per_loop_process_len * sizeof(T));
            pipe_->InitBuffer(flag_queue, 1, 32);
            
            tile_length = tiling_.per_loop_process_len;
            tile_num = (actual_size_val + tile_length - 1) / tile_length;
            tile_last_length = actual_size_val % tile_length;
            if (tile_last_length == 0) {
                tile_last_length = tile_length;
            }
        }

        __aicore__ inline void Process() {
            if (tile_num > 0) {
                uint32_t loop_idxs[2] = {0, 0};
                uint32_t ping_pong_ids[2] = {0, 0};
                for (size_t i = 0; i < tile_num + 1; i++) {
                    uint32_t idx_0 = i % 2;
                    uint32_t idx_1 = (i - 1) % 2;
                    ping_pong_ids[idx_0] = idx_0;
                    loop_idxs[idx_0] = i;

                    if (i > 0) {
                        WaitGemmQB();
                    }
                    if (i < tile_num) {
                        LaunchGemmQB(loop_idxs[idx_0], ping_pong_ids[idx_0]);
                    }
                    if (i > 0) {
                        VecProcess(loop_idxs[idx_1], ping_pong_ids[idx_1]);
                    }
                }
            }

            LocalTensor<uint16_t> flag_local = flag_queue.AllocTensor<uint16_t>();
            uint16_t pad_value(0);
            Duplicate(flag_local, pad_value, 16);
            flag_local.SetValue(0, 1);
            flag_queue.EnQue(flag_local);
            flag_local = flag_queue.DeQue<uint16_t>();
            DataCopy(flag_gm_[core_id * 16], flag_local, 16);
            flag_queue.FreeTensor(flag_local);
        }
    public:
        TPipe *pipe_;
        DistanceIVFFlatIpFP32TilingData tiling_;
        using QueryType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>;
        using BaseType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T, true>;
        using OutType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>;
        using BiasType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>;
        using mm_qb = matmul::Matmul<QueryType, BaseType, OutType, BiasType>;
        mm_qb gemm_qb;
    private:
        __aicore__ inline void VecProcess(int32_t loop_idx, int32_t ping_pong_id) {
            int32_t code_len = (loop_idx == tile_num - 1) ? tile_last_length : tile_length;
            int32_t align_code_len = (code_len + 8 - 1) / 8 * 8;
            LocalTensor<T> mm_res_local = in_que.AllocTensor<T>();
            GlobalTensor<T> src_gm_ = mm_out_gm_[ping_pong_id * tiling_.per_loop_process_len];
            DataCopyParams copy_params;
            copy_params.blockCount = 1;
            copy_params.blockLen = static_cast<uint16_t>(code_len * sizeof(T));
            copy_params.srcStride = 0;
            copy_params.dstStride = 0;
            DataCopyPadParams pad_params{true, 0, static_cast<uint8_t>(align_code_len - code_len), 0};
            DataCopyPad(mm_res_local, src_gm_, copy_params, pad_params);

            in_que.EnQue(mm_res_local);
            mm_res_local = in_que.DeQue<T>();
            LocalTensor<T> res_local = out_que.AllocTensor<T>();

            DataCopy(res_local, mm_res_local, align_code_len);
            out_que.EnQue(res_local);
            res_local = out_que.DeQue<T>();
            int32_t res_dst_offset = core_id * tiling_.code_num + loop_idx * tile_length;
            DataCopyPad(result_gm_[res_dst_offset], res_local, copy_params);

            auto event_mte3_v = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
            SetFlag<AscendC::HardEvent::MTE3_V>(event_mte3_v);
            WaitFlag<AscendC::HardEvent::MTE3_V>(event_mte3_v);
            int32_t burst_len = tiling_.burst_len;
            int32_t align_burst_len = (align_code_len + burst_len - 1) / burst_len * burst_len;
            int32_t pad_length = align_burst_len - align_code_len;
            if (pad_length > 0) {
                T pad_value(-65536.0);
                Duplicate(mm_res_local[align_code_len], pad_value, pad_length);
                PipeBarrier<PIPE_V>();
            }

            int32_t rep_times = align_burst_len / burst_len;
            int32_t dst_max_res_offset = core_id * tiling_.max_res_stride + loop_idx * tiling_.max_res_inner_stride;
            WholeReduceMax(res_local, mm_res_local, burst_len, rep_times, 1, 1, burst_len / 8);
            out_que.EnQue(res_local);
            res_local = out_que.DeQue<T>();
            DataCopyParams copy_max_res_params;
            copy_max_res_params.blockCount = 1;
            copy_max_res_params.blockLen = static_cast<uint16_t>(rep_times * 2 * sizeof(T));
            copy_params.srcStride = 0;
            copy_params.dstStride = 0;
            DataCopyPad(max_res_gm_[dst_max_res_offset], res_local, copy_max_res_params);
            in_que.FreeTensor(mm_res_local);
            out_que.FreeTensor(res_local);
        }

        __aicore__ inline void LaunchGemmQB(int32_t loop_idx, int32_t ping_pong_id) {
            uint32_t code_offset = loop_idx * tile_length * tiling_.dim_len;
            int32_t code_len = (loop_idx == tile_num - 1) ? tile_last_length : tile_length;
            gemm_qb.SetTail(1, code_len, tiling_.dim_len);
            gemm_qb.SetTensorA(query_gm_);
            gemm_qb.SetTensorB(code_gm_[code_offset], true);
            gemm_qb.template IterateAll<false>(mm_out_gm_[ping_pong_id  * tiling_.per_loop_process_len], 0, false, true);
        }

        __aicore__ inline void WaitGemmQB() {
            gemm_qb.WaitIterateAll();
            gemm_qb.End();
        }

    private:
        TQue<QuePosition::VECIN, 1> in_que;
        TQue<QuePosition::VECOUT, 1> out_que, flag_queue;

        GlobalTensor<T> query_gm_;
        GlobalTensor<T> code_gm_;
        GlobalTensor<uint64_t> offset_gm_;
        GlobalTensor<uint32_t> actual_size_gm_;
        GlobalTensor<T> result_gm_;
        GlobalTensor<T> max_res_gm_;
        GlobalTensor<uint16_t> flag_gm_;
        GlobalTensor<T> mm_out_gm_;

        int32_t core_id;
        uint32_t actual_size_val;
        int32_t tile_num;
        int32_t tile_length;
        int32_t tile_last_length;
    };
}

extern "C" __global__ __aicore__ void distance_ivf_flat_ip_fp32(GM_ADDR query, GM_ADDR base, GM_ADDR offset, GM_ADDR actual_size, GM_ADDR result, GM_ADDR max_result, GM_ADDR flag, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    TPipe pipe;
    kernels::DistanceIVFFlatIpFP32<float32_t> op(&pipe);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGIST_MATMUL_OBJ(op.pipe_, GetSysWorkSpacePtr(), op.gemm_qb, &tiling_data.gemm_qb_tiling);
    op.Init(query, base, offset, actual_size, result, max_result, flag, usrWorkspace,
        &tiling_data);
    op.Process();
}