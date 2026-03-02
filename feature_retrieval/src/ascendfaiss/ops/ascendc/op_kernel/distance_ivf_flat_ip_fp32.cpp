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

const int32_t CORE_MAX_NUM = 56; // 数组开辟最大值到核数上限，实际使用取决于适配服务器或推理卡的核数
const int32_t MES_LENGTH = 2; // [0] = 长度 [1] = 偏移

using namespace AscendC;

namespace kernels {
    template <typename T>
    class DistanceIVFFlatIpFP32 {
    public:
        __aicore__ inline DistanceIVFFlatIpFP32(TPipe *pipe_in) {pipe_ = pipe_in;};
        __aicore__ inline void InitBuffer(GM_ADDR query, GM_ADDR base, GM_ADDR offset,
                                          GM_ADDR actual_size, GM_ADDR result,
                                          GM_ADDR max_result, GM_ADDR flag, GM_ADDR usrWorkspace,
                                          const DistanceIVFFlatIpFP32TilingData *tiling_data) {
            this->core_id = GetBlockIdx();
            this->tiling_ = *tiling_data;
            this->aiv_num = tiling_.aiv_num;

            query_gm_.SetGlobalBuffer((__gm__ T *)query);
            offset_gm_.SetGlobalBuffer((__gm__ uint64_t *)offset);
            actual_size_gm_.SetGlobalBuffer((__gm__ uint32_t *)actual_size, this->aiv_num);
            result_gm_.SetGlobalBuffer((__gm__ T *)result);
            max_res_gm_.SetGlobalBuffer((__gm__ T *)max_result);
            flag_gm_.SetGlobalBuffer((__gm__ uint16_t *)flag);
            mm_out_gm_.SetGlobalBuffer((__gm__ T *)(usrWorkspace + tiling_.buffer_size * 2 * this->core_id));
            pipe_->InitBuffer(in_que, 1, tiling_.per_loop_process_len * sizeof(T));
            pipe_->InitBuffer(out_que, 1, tiling_.per_loop_process_len * sizeof(T));
            pipe_->InitBuffer(flag_queue, 1, this->aiv_num * 16 * sizeof(uint16_t));
        }

        __aicore__ inline void InitCoreLength() {
            for (int32_t i = 0; i < this->aiv_num; i++) {
                this->all_codes = this->all_codes + actual_size_gm_.GetValue(i);
            }
            this->tile_length = tiling_.per_loop_process_len;
            this->core_length = (this->all_codes / this->aiv_num + this->tiling_.burst_len - 1) /
                                this->tiling_.burst_len * this->tiling_.burst_len;

            for (int32_t i = 0; i < this->aiv_num; i++) {
                int32_t actual_size_val = actual_size_gm_.GetValue(i);
                InitlengthArr(actual_size_val);
            }

            if (this->length_arr_block_length > 0) {
                this->length_arr[this->length_arr_core_idx][0] =
                    this->length_arr[this->length_arr_core_idx][0] + this->length_arr_length;
                this->length_arr[this->length_arr_core_idx][1] =
                    this->length_arr[this->length_arr_core_idx - 1][1] +
                    this->length_arr[this->length_arr_core_idx - 1][0];
            }

            PipeBarrier<PIPE_ALL>();
        }

        __aicore__ inline void Process(GM_ADDR base) {
            SetProcessMes();
            if (this->tile_core_length > 0) {
                while (this->cube_flag) {
                    code_gm_.SetGlobalBuffer((__gm__ T *)(base + code_offset));
                    ProcessCube();

                    if (pingpong_idx > 1) {
                        WaitGemmQB();
                        VecProcess(this->offset_arr[2][this->pingpong_idx % 2],
                                   this->offset_arr[0][this->pingpong_idx % 2],
                                   this->offset_arr[1][this->pingpong_idx % 2],
                                   this->pingpong_idx - 1);
                    }
                    this->core_offset = this->core_offset + this->length;
                }
                WaitGemmQB();
                VecProcess(this->length,
                           this->offset_arr[0][(this->pingpong_idx - 1) % 2],
                           this->offset_arr[1][(this->pingpong_idx - 1) % 2],
                           this->pingpong_idx);
            }

            SyncAll();
            if (this->core_id == 0) {
                LocalTensor<uint16_t> flag_local = flag_queue.AllocTensor<uint16_t>();
                uint16_t pad_value(0);
                Duplicate(flag_local, pad_value, this->aiv_num * 16);
                for (int32_t i = 0; i < this->aiv_num; i++) {
                    flag_local.SetValue(i * 16, this->uint16_one);
                }
                flag_queue.EnQue(flag_local);
                flag_local = flag_queue.DeQue<uint16_t>();
                DataCopy(flag_gm_, flag_local, this->aiv_num * 16);
                flag_queue.FreeTensor(flag_local);
            }
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
        __aicore__ inline void InitlengthArr(int32_t actual_size_val) {
            if (isLast(actual_size_val)) {
                return;
            }
            int32_t actual_size_val_block = (actual_size_val + this->tiling_.burst_len - 1) /
                                            this->tiling_.burst_len * this->tiling_.burst_len;
            if (this->length_arr_block_length + actual_size_val_block < this->core_length) {
                this->length_arr_length = this->length_arr_length + actual_size_val;
                this->length_arr_block_length = this->length_arr_block_length + actual_size_val_block;
                return;
            } else if (this->length_arr_block_length + actual_size_val_block == this->core_length) {
                this->length_arr[this->length_arr_core_idx][0] = this->length_arr_length + actual_size_val;
                SetCoreOffset();
                ResetZero();
                this->length_arr_core_idx++;
                return;
            } else {
                int32_t last_length = this->core_length - this->length_arr_block_length;
                int32_t next_length = actual_size_val - last_length;
                int32_t block_next_length = (next_length + this->tiling_.burst_len - 1) /
                                            this->tiling_.burst_len * this->tiling_.burst_len;
                this->length_arr[this->length_arr_core_idx][0] = this->length_arr_length + last_length;
                SetCoreOffset();
                this->length_arr_core_idx++;
                while (block_next_length >= this->core_length) {
                    SetCoreOffset();
                    if (this->length_arr_core_idx == this->aiv_num - 1) {
                        this->length_arr[this->length_arr_core_idx][0] =
                            this->length_arr[this->length_arr_core_idx][0] + next_length;
                        ResetZero();
                        return;
                    }
                    if (next_length < this->core_length) {
                        this->length_arr[this->length_arr_core_idx][0] = next_length;
                    } else {
                        this->length_arr[this->length_arr_core_idx][0] = this->core_length;
                    }
                    ResetZero();
                    this->length_arr_core_idx++;
                    next_length = next_length - this->core_length;
                    block_next_length = block_next_length - this->core_length;
                }
                if (block_next_length == 0) {
                    ResetZero();
                    return;
                }
                this->length_arr_length = next_length;
                this->length_arr_block_length = block_next_length;
            }
        }

        __aicore__ inline bool isLast(int32_t actual_size_val) {
            if (this->length_arr_core_idx == this->aiv_num - 1) {
                this->length_arr[this->length_arr_core_idx][0] =
                    this->length_arr[this->length_arr_core_idx][0] + this->length_arr_length + actual_size_val;
                if (this->length_arr[this->length_arr_core_idx][1] == 0) {
                    this->length_arr[this->length_arr_core_idx][1] =
                        this->length_arr[this->length_arr_core_idx - 1][1] +
                        this->length_arr[this->length_arr_core_idx - 1][0];
                    ResetZero();
                }
                return true;
            }
            return false;
        }

        __aicore__ inline void SetCoreOffset() {
            if (this->length_arr_core_idx > 0) {
                this->length_arr[this->length_arr_core_idx][1] =
                    this->length_arr[this->length_arr_core_idx - 1][1] +
                    this->length_arr[this->length_arr_core_idx - 1][0];
            }
        }

        __aicore__ inline void ResetZero() {
            this->length_arr_length = 0;
            this->length_arr_block_length = 0;
        }

        __aicore__ inline void VecProcess(int32_t code_len, int32_t copyout_offset,
                                          int32_t dst_max_res_offset, int32_t pingpong_idx) {
            int32_t align_code_len = (code_len + 8 - 1) / 8 * 8;

            LocalTensor<T> mm_res_local = in_que.AllocTensor<T>();
            GlobalTensor<T> src_gm_ = mm_out_gm_[((pingpong_idx - 1) % 2) * tiling_.per_loop_process_len];
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
            // 搬出距离结果
            DataCopyPad(result_gm_[copyout_offset], res_local, copy_params);

            // 计算burst max结果
            auto event_mte3_v = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
            SetFlag<AscendC::HardEvent::MTE3_V>(event_mte3_v);
            WaitFlag<AscendC::HardEvent::MTE3_V>(event_mte3_v);
            int32_t align_burst_len = (align_code_len + this->tiling_.burst_len - 1) /
                                      this->tiling_.burst_len * this->tiling_.burst_len;
            int32_t pad_length = align_burst_len - align_code_len;
            if (pad_length > 0) {
                T pad_value(-65536.0);
                Duplicate(mm_res_local[align_code_len], pad_value, pad_length);
                PipeBarrier<PIPE_V>();
            }
            int32_t rep_times = align_burst_len / this->tiling_.burst_len;
            WholeReduceMax(res_local, mm_res_local, this->tiling_.burst_len,
                           rep_times, 1, 1, this->tiling_.burst_len / 8);
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

        __aicore__ inline void LaunchGemmQB(int32_t length, int32_t offset, int32_t pingpong_idx) {
            gemm_qb.SetTail(1, length, tiling_.dim_len);
            gemm_qb.SetTensorA(query_gm_);
            gemm_qb.SetTensorB(code_gm_[offset * tiling_.dim_len], true);
            gemm_qb.template IterateAll<false>(mm_out_gm_[(pingpong_idx % 2) * tiling_.per_loop_process_len],
                                               0, false, true);
        }

        __aicore__ inline void WaitGemmQB() {
            gemm_qb.WaitIterateAll();
            gemm_qb.End();
        }

        __aicore__ inline void SetProcessMes() {
            this->tile_core_length = this->length_arr[this->core_id][0];
            this->core_offset = this->length_arr[this->core_id][1];
            this->end_offset = this->core_offset + this->tile_core_length;
            if (this->tile_core_length > 0) {
                for (int32_t i = 0; i < this->aiv_num; i++) {
                    if (this->actual_size_pro_val <= this->core_offset &&
                        this->core_offset < this->actual_size_pro_val + actual_size_gm_.GetValue(i)) {
                        this->code_offset = offset_gm_.GetValue(i);
                        this->code_idx = i;
                        this->offset = this->core_offset - this->actual_size_pro_val;
                        this->offset_arr[0][0] = i * this->tiling_.code_num + this->offset;
                        this->offset_arr[1][0] = i * this->tiling_.max_res_stride +
                                                 this->offset / this->tiling_.burst_len * 2;
                        this->actual_size_pro_val = this->actual_size_pro_val + actual_size_gm_.GetValue(i);
                        break;
                    }
                    this->actual_size_pro_val = this->actual_size_pro_val + actual_size_gm_.GetValue(i);
                }
            }
        }

        __aicore__ inline void ProcessCube() {
            if (this->pingpong_idx > 0) {
                this->offset_arr[2][(this->pingpong_idx - 1) % 2] = this->length;
                if (this->offset == 0) {
                    this->offset_arr[0][this->pingpong_idx % 2] = this->code_idx * this->tiling_.code_num;
                    this->offset_arr[1][this->pingpong_idx % 2] = this->code_idx * this->tiling_.max_res_stride;
                } else {
                    this->offset_arr[0][this->pingpong_idx % 2] =
                        this->offset_arr[0][(this->pingpong_idx - 1) % 2] + this->length;
                    this->offset_arr[1][this->pingpong_idx % 2] =
                        this->offset_arr[1][(this->pingpong_idx - 1) % 2] + this->length / this->tiling_.burst_len * 2;
                }
            }

            if (this->core_offset + this->tile_length < this->end_offset) {
                if (this->core_offset + this->tile_length < this->actual_size_pro_val) {
                    this->length = this->tile_length;
                    LaunchGemmQB(this->length, this->offset, this->pingpong_idx);
                    this->pingpong_idx++;
                    this->offset = this->offset + this->length;
                } else {
                    this->length = this->actual_size_pro_val - this->core_offset;
                    LaunchGemmQB(this->length, this->offset, this->pingpong_idx);
                    this->pingpong_idx++;
                    this->code_idx++;
                    this->actual_size_pro_val = this->actual_size_pro_val + actual_size_gm_.GetValue(code_idx);
                    this->code_offset = offset_gm_.GetValue(code_idx);
                    this->offset = 0;
                }
            } else {
                if (this->end_offset > this->actual_size_pro_val) {
                    this->length = this->actual_size_pro_val - this->core_offset;
                    LaunchGemmQB(this->length, this->offset, this->pingpong_idx);
                    this->pingpong_idx++;
                    this->code_idx++;
                    this->actual_size_pro_val = this->actual_size_pro_val + actual_size_gm_.GetValue(code_idx);
                    this->code_offset = offset_gm_.GetValue(code_idx);
                    this->offset = 0;
                } else {
                    this->length = this->end_offset - this->core_offset;
                    LaunchGemmQB(this->length, this->offset, this->pingpong_idx);
                    this->pingpong_idx++;
                    this->cube_flag = false;
                }
            }
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

        uint32_t all_codes = 0;
        int32_t aiv_num = 0;
        int32_t core_id = 0;
        int32_t core_length = 0;
        int32_t tile_core_length = 0;
        int32_t tile_length = 0;
        int32_t core_offset = 0;
        int32_t length_arr[CORE_MAX_NUM][MES_LENGTH] = {0};
        int32_t length_arr_core_idx = 0;
        int32_t length_arr_length = 0;
        int32_t length_arr_block_length = 0;
        bool cube_flag = true;
        int32_t end_offset = 0;
        int64_t code_offset = 0;
        int32_t code_idx = 0;
        int32_t actual_size_pro_val = 0;
        int32_t offset = 0;
        int32_t length = 0;
        int32_t pingpong_idx = 0;
        int32_t offset_arr[3][2] = {0};
        uint16_t uint16_one = 1;
    };
}

extern "C" __global__ __aicore__ void distance_ivf_flat_ip_fp32(GM_ADDR query, GM_ADDR base, GM_ADDR offset, GM_ADDR actual_size, GM_ADDR result, GM_ADDR max_result, GM_ADDR flag, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    TPipe pipe;
    kernels::DistanceIVFFlatIpFP32<float32_t> op(&pipe);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGIST_MATMUL_OBJ(op.pipe_, GetSysWorkSpacePtr(), op.gemm_qb, &tiling_data.gemm_qb_tiling);
    op.InitBuffer(query, base, offset, actual_size, result, max_result, flag, usrWorkspace,
        &tiling_data);
    op.InitCoreLength();
    op.Process(base);
}