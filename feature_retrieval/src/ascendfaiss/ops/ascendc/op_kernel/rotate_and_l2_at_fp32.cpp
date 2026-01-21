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

namespace kernels
{
    template <typename T>
    class RotateAndL2AtFP32
    {
    public:
        __aicore__ inline RotateAndL2AtFP32(TPipe *pipe_in) {pipe_ = pipe_in;};
        __aicore__ inline void Init(GM_ADDR vectors, GM_ADDR vectorSize, GM_ADDR matrix,
                                    GM_ADDR rotate_result, GM_ADDR l2_result, GM_ADDR usrWorkspace,
                                    const RotateAndL2AtFP32TilingData *tiling_data) {
            vectorSize_gm_.SetGlobalBuffer((__gm__ int32_t *)vectorSize, 1);
            TilingFunc();

            tiling_ = *tiling_data;
            this->dimLength = tiling_.dimLength;
            this->maskNum = this->dimLength / this->mask;

            vec_gm_.SetGlobalBuffer((__gm__ T *)vectors + this->vecOffset * this->dimLength,
                                            this->vecLength * this->dimLength);
            matrix_gm_.SetGlobalBuffer((__gm__ T *)matrix);
            result_gm_.SetGlobalBuffer((__gm__ T *)rotate_result + this->vecOffset * this->dimLength,
                                            this->vecLength * this->dimLength);
            l2_res_gm_.SetGlobalBuffer((__gm__ T *)l2_result + this->vecOffset, this->vecLength);

            this->tileLength = tiling_.tileLength;
            if (this->vecLength == 0) {
                this->tileNum = 0;
                this->tileLength = 0;
                this->lastTileLength = 0;
            } else if (this->vecLength < this->tileLength) {
                this->tileNum = 1;
                this->tileLength = this->vecLength;
                this->lastTileLength = this->vecLength;
            } else if (this->vecLength % this->tileLength == 0) {
                this->tileNum = this->vecLength / this->tileLength;
                this->lastTileLength = this->tileLength;
            } else {
                this->tileNum = this->vecLength / this->tileLength + 1;
                this->lastTileLength = this->vecLength % this->tileLength;
            }

            pipe_->InitBuffer(in_que, 2, RoundUp(this->tileLength * this->dimLength * sizeof(T), 32));
            pipe_->InitBuffer(out_que, 2, RoundUp(this->tileLength * sizeof(T), 32));
        }

        __aicore__ inline int32_t RoundUp(int32_t length, int32_t align) {
            return (length + align - 1) / align * align;
        }

        __aicore__ inline void Process() {
            if (this->vecLength > 0) {
                LaunchGemmQB();
                for (int32_t i = 0; i < this->tileNum; i++) {
                    int32_t copyLength = this->tileLength;
                    if (i == this->tileNum - 1)
                    {
                        copyLength = this->lastTileLength;
                    }
                    CopyIn(i, copyLength);
                    PipeBarrier<PIPE_ALL>();
                    Compute(i, copyLength);
                    CopyOut(i, copyLength);
                }
                WaitGemmQB();
            }
        }

    public:
        TPipe *pipe_;
        RotateAndL2AtFP32TilingData tiling_;
        using QueryType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>;
        using BaseType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T, true>;
        using OutType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>;
        using BiasType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>;
        using mm_qb = matmul::Matmul<QueryType, BaseType, OutType, BiasType>;
        mm_qb gemm_qb;
    private:
        __aicore__ inline void TilingFunc() {
            int32_t vec_num = vectorSize_gm_.GetValue(0);  // 表示当前的实际vector数量
            int32_t formerCoreNum = 0;    // 非尾核数量
            int32_t formerCoreLength = 0; // 每个非尾核处理的向量数量
            int32_t tailCoreNum = 0;      // 尾核数量
            int32_t tailCoreLength = 0;   // 每个尾核处理的向量数量
            // 根据 vec_num 和 block_num，确定上述四个参数（分核：根据向量数量分核）
            int32_t blockNum = get_block_num();
            if (vec_num < blockNum) {
                formerCoreNum = vec_num;
                formerCoreLength = 1;
                tailCoreNum = blockNum - formerCoreNum;
                tailCoreLength = 0;
            } else if (vec_num % blockNum == 0) {
                formerCoreNum = blockNum;
                formerCoreLength = vec_num / blockNum;
                tailCoreNum = 0;
                tailCoreLength = 0;
            } else {
                formerCoreNum = vec_num % blockNum;
                formerCoreLength = vec_num / blockNum + 1;
                tailCoreNum = blockNum - formerCoreNum;
                tailCoreLength = vec_num / blockNum;
            }
            TilingAcrossCores(formerCoreNum, formerCoreLength, tailCoreLength);
        }

        __aicore__ inline void TilingAcrossCores(int32_t formerCoreNum, int32_t formerCoreLength,
                                                 int32_t tailCoreLength) {
            // 根据core_id判断block负责的向量区间
            int32_t block_id = get_block_idx();
            int32_t sub_block_id = get_subblockid();
            if (block_id < formerCoreNum) {
                if ASCEND_IS_AIC {
                    this->vecLength = formerCoreLength;
                    this->vecOffset = block_id * formerCoreLength;
                }
                if ASCEND_IS_AIV {
                    if (formerCoreLength % 2 == 0) {
                        this->vecLength = formerCoreLength / 2;
                        this->vecOffset = block_id * formerCoreLength +
                                          sub_block_id * formerCoreLength / 2;  // 当前core负责的向量偏移
                    } else {
                        this->vecLength = (1 - sub_block_id) + (int32_t)(formerCoreLength / 2);  // 第一个vector多处理一个；
                        this->vecOffset = block_id * formerCoreLength +
                                          sub_block_id * formerCoreLength / 2 + sub_block_id;  // 第二个vector多偏移一个
                    }
                }
            } else {
                if ASCEND_IS_AIC {
                    this->vecLength = tailCoreLength;
                    this->vecOffset = formerCoreNum * formerCoreLength + (block_id - formerCoreNum) * tailCoreLength;
                }
                if ASCEND_IS_AIV {
                    if (tailCoreLength % 2 == 0) {
                        this->vecLength = tailCoreLength / 2;
                        this->vecOffset = formerCoreNum * formerCoreLength +
                                          (block_id - formerCoreNum) * tailCoreLength +
                                          sub_block_id * tailCoreLength / 2;
                    } else {
                        this->vecLength = (1 - sub_block_id) + tailCoreLength / 2;  // 第一个vector多处理一个；
                        this->vecOffset = formerCoreNum * formerCoreLength +
                                          (block_id - formerCoreNum) * tailCoreLength +
                                          sub_block_id * tailCoreLength / 2 + sub_block_id;  // 第二个vector多偏移一个
                    }
                }
            }
        }

        __aicore__ inline void LaunchGemmQB() {
            gemm_qb.SetTail(this->vecLength, this->dimLength, this->dimLength);
            gemm_qb.SetTensorA(vec_gm_);
            gemm_qb.SetTensorB(matrix_gm_);
            gemm_qb.template IterateAll<false>(result_gm_, 0, false, true);
        }

        __aicore__ inline void WaitGemmQB() {
            gemm_qb.WaitIterateAll();
            gemm_qb.End();
        }
        
        __aicore__ inline void CopyIn(int32_t progress, int32_t copyLength)
        {
            LocalTensor<T> vecsLocal = in_que.AllocTensor<T>();
            DataCopy(vecsLocal, vec_gm_[progress * this->tileLength * this->dimLength], copyLength * this->dimLength);
            in_que.EnQue(vecsLocal);
        }

        __aicore__ inline void Compute(int32_t progress, int32_t copyLength)
        {
            LocalTensor<T> vecsLocal = in_que.DeQue<T>();
            LocalTensor<T> l2ResultLocal = out_que.AllocTensor<T>();
            Mul(vecsLocal, vecsLocal, vecsLocal, copyLength * this->dimLength);
            PipeBarrier<PIPE_V>();

            if (this->maskNum >= 1) {
                for (int32_t i = 0; i < copyLength; i++) {
                    Add(vecsLocal[i * this->dimLength], vecsLocal[i * this->dimLength + this->mask],
                        vecsLocal[i * this->dimLength], this->mask, this->maskNum - 1, {1, 1, 1, 0, 8, 0});
                }
            }
            PipeBarrier<PIPE_V>();

            int32_t repStride = 8 * this->maskNum;
            for (int32_t i = 0; i < copyLength; i += 255) {
                int32_t len = copyLength - i;
                if (len > 255) len = 255;
                WholeReduceSum<T>(l2ResultLocal[i], vecsLocal[i * this->dimLength], this->mask, len, 1, 1, repStride);
            }

            int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);

            out_que.EnQue(l2ResultLocal);
            in_que.FreeTensor(vecsLocal);
        }
        
        __aicore__ inline void CopyOut(int32_t progress, int32_t copyLength)
        {
            LocalTensor<T> l2ResultLocal = out_que.DeQue<T>();
            DataCopyPad(l2_res_gm_[progress * this->tileLength], l2ResultLocal,
                        {1, (uint32_t)(copyLength * sizeof(T)), 0, 0, 0});
            out_que.FreeTensor(l2ResultLocal);
        }
    private:
        TQue<QuePosition::VECIN, 1> in_que;
        TQue<QuePosition::VECOUT, 1> out_que;

        GlobalTensor<int32_t> vectorSize_gm_;
        GlobalTensor<T> vec_gm_;
        GlobalTensor<T> matrix_gm_;
        GlobalTensor<T> result_gm_;
        GlobalTensor<T> l2_res_gm_;

        int32_t tileNum;
        int32_t tileLength;
        int32_t lastTileLength;

        int32_t dimLength;
        int32_t vecLength;
        int32_t vecOffset;
        int32_t mask = 64;
        int32_t maskNum;
    };
}

extern "C" __global__ __aicore__ void rotate_and_l2_at_fp32(GM_ADDR vectors, GM_ADDR vectorSize,
                                                            GM_ADDR matrix, GM_ADDR rotate_result,
                                                            GM_ADDR l2_result, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    TPipe pipe;
    kernels::RotateAndL2AtFP32<float32_t> op(&pipe);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGIST_MATMUL_OBJ(op.pipe_, GetSysWorkSpacePtr(), op.gemm_qb, &tiling_data.gemm_qb_tiling);
    op.Init(vectors, vectorSize, matrix, rotate_result, l2_result, usrWorkspace, &tiling_data);
    op.Process();
}