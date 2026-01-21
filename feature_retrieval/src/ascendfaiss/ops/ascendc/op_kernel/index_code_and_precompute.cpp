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
    class IndexCodeAndPrecompute {
    public:
        __aicore__ inline IndexCodeAndPrecompute(TPipe *pipe_in) {pipe_ = pipe_in;};
        __aicore__ inline void Init(GM_ADDR vectorSize, GM_ADDR indexes, GM_ADDR indexesl2,
                                    GM_ADDR centroid, GM_ADDR centroidl2, GM_ADDR codes_result,
                                    GM_ADDR l2_result, GM_ADDR l1_result, GM_ADDR usrWorkspace,
                                    const IndexCodeAndPrecomputeTilingData *tiling_data) {
            vectorSize_gm_.SetGlobalBuffer((__gm__ int32_t *)vectorSize, 1);
            int32_t vec_num = vectorSize_gm_.GetValue(0);  // 表示当前add_vector的实际vector数量

            int32_t formerCoreNum = 0;    // 非尾核数量
            int32_t formerCoreLength = 0; // 每个非尾核处理的向量数量
            int32_t tailCoreNum = 0;      // 尾核数量
            int32_t tailCoreLength = 0;   // 每个尾核处理的向量数量
            
            // 根据 vec_num 和 blockNum，确定上述四个参数（分核：根据向量数量分核）
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

            int32_t core_id = get_block_idx();
            tiling_ = *tiling_data;
            this->dimLength = tiling_.dimLength;    // 向量维度
            this->maskNum = this->dimLength / this->mask;

            // 根据core_id判断block负责的向量区间
            if (core_id < formerCoreNum) {
                this->vecLength = formerCoreLength;    // 当前core负责的向量数量
                this->vecOffset = core_id * formerCoreLength;   // 当前core负责的向量偏移
            } else {
                this->vecLength = tailCoreLength;
                this->vecOffset = formerCoreNum * formerCoreLength + (core_id - formerCoreNum) * tailCoreLength;
            }

            // 根据block负责的向量区间，构建GlobalTensor（起始地址+大小）
            int32_t offset = this->vecOffset * this->dimLength;
            int32_t length = this->vecLength * this->dimLength;

            indexes_gm_.SetGlobalBuffer((__gm__ T *)indexes + offset, length); // indexes: n * d
            indexesl2_gm_.SetGlobalBuffer((__gm__ T *)indexesl2 + this->vecOffset, this->vecLength); // indexes: n * 1
            centroid_gm_.SetGlobalBuffer((__gm__ T *)centroid, this->dimLength);      // centroid: 1 * d
            centroidl2_gm_.SetGlobalBuffer((__gm__ T *)centroidl2, 1);  // centroidl2: 1

            // 补充考虑dim非8位对齐的场景
            codes_res_gm_.SetGlobalBuffer((__gm__ uint8_t *)codes_result + offset/8, length/8); //  n * d/8
            l2_res_gm_.SetGlobalBuffer((__gm__ T *)l2_result + this->vecOffset, this->vecLength);  // l2_result: n
            l1_res_gm_.SetGlobalBuffer((__gm__ T *)l1_result + this->vecOffset, this->vecLength);  // l1_result: n

            // InitBuffer：根据block负责的vecLength，确定tileNum、tileLength、lastTileLength
            this->tileLengthStage1 = tiling_.tileLengthStage1;  // vector 单元第一次循环处理的向量数量
            this->tileLengthStage2 = tiling_.tileLengthStage2;  // vector 单元第二次循环处理的向量数量
            
            FirstInitBuffer();
        }

        __aicore__ inline void FirstInitBuffer() {
            this->tileLength = this->tileLengthStage1;

            if (this->vecLength == 0) {
                this->tileNum = 0;
                this->tileLength = 0;
                this->lastTileLength = 0;
            } else if (this->vecLength < this->tileLength) {
                this->tileNum = 1;   // vector 单元循环次数
                this->tileLength = this->vecLength;   // vector 单元每次循环处理的向量数量
                this->lastTileLength = this->vecLength;   // vector 单元最后一次循环处理的向量数量
            } else if (this->vecLength % this->tileLength == 0) {
                this->tileNum = this->vecLength / this->tileLength;
                this->lastTileLength = this->tileLength;
            } else {
                this->tileNum = this->vecLength / this->tileLength + 1;
                this->lastTileLength = this->vecLength % this->tileLength;
            }

            pipe_->InitBuffer(in_indexes_que, 1, RoundUp(this->tileLength * this->dimLength * sizeof(T), 64));
            pipe_->InitBuffer(in_center_que, 1, RoundUp(this->dimLength * sizeof(T), 64));
            pipe_->InitBuffer(out_l1_que, 1, RoundUp(this->tileLength * sizeof(T), 64));
            pipe_->InitBuffer(out_l2_que, 1, RoundUp(this->tileLength * sizeof(T), 64));
            pipe_->InitBuffer(out_codes_que, 1, RoundUp(this->tileLength * this->dimLength / 8 * sizeof(uint8_t), 64));

            pipe_->InitBuffer(tmp_que, 1, RoundUp(this->tileLength * this->dimLength * sizeof(T), 64));
        }

        __aicore__ inline void SecondInitBuffer() {
            // SecondInit Buffer：根据 tileLengthStage2，确定tileNum、tileLength、lastTileLength
            this->tileLength = this->tileLengthStage2;  // vector 单元每次循环处理的向量数量

            if (this->vecLength == 0) {
                this->tileNum = 0;
                this->tileLength = 0;
                this->lastTileLength = 0;
            } else if (this->vecLength < this->tileLength) {
                this->tileNum = 1;   // vector 单元循环次数
                this->tileLength = this->vecLength;   // vector 单元每次循环处理的向量数量
                this->lastTileLength = this->vecLength;   // vector 单元最后一次循环处理的向量数量
            } else if (this->vecLength % this->tileLength == 0) {
                this->tileNum = this->vecLength / this->tileLength;
                this->lastTileLength = this->tileLength;
            } else {
                this->tileNum = this->vecLength / this->tileLength + 1;
                this->lastTileLength = this->vecLength % this->tileLength;
            }

            pipe_->Reset();
            pipe_->InitBuffer(in_indexesl2_que, 1, RoundUp(this->tileLength * sizeof(T), 64));
            pipe_->InitBuffer(in_index_sub_center_que, 1, RoundUp(this->tileLength * sizeof(T), 64));
            pipe_->InitBuffer(in_innerProductLocal_que, 1, RoundUp(this->tileLength * sizeof(T), 64));

            pipe_->InitBuffer(out_l1_que, 1, RoundUp(this->tileLength * sizeof(T), 64));
            pipe_->InitBuffer(out_l2_que, 1, RoundUp(this->tileLength * sizeof(T), 64));

            pipe_->InitBuffer(in_centerl2_que, 1, RoundUp(sizeof(T), 64));
        }

        __aicore__ inline int32_t RoundUp(int32_t length, int32_t align) {
            return (length + align - 1) / align * align;
        }

        __aicore__ inline void Process() {
            if (this->vecLength > 0) {
                // 只需搬运一次聚类中心
                centerLocal = in_center_que.AllocTensor<T>();
                DataCopy(centerLocal, centroid_gm_, this->dimLength);
                in_center_que.EnQue(centerLocal);
                PipeBarrier<PIPE_ALL>();
                centerLocal = in_center_que.DeQue<T>();

                for (int32_t i = 0; i < this->tileNum; i++) {
                    int32_t copyVecLength = (i == this->tileNum - 1)? this->lastTileLength : this->tileLength;

                    CopyInForCode(i, copyVecLength);
                    PipeBarrier<PIPE_ALL>();
                    VecComputeForCode(i, copyVecLength);
                    CopyOutForCode(i, copyVecLength);
                }

                PipeBarrier<PIPE_ALL>();
                in_center_que.FreeTensor(centerLocal);

                PipeBarrier<PIPE_ALL>();   // AscendC::SyncAll();

                SecondInitBuffer();

                this->centerl2Localval = centroidl2_gm_.GetValue(0);

                for (int32_t i = 0; i < this->tileNum; i++) {
                    int32_t copyVecLength = (i == this->tileNum - 1)? this->lastTileLength : this->tileLength;

                    CopyInForPreComp(i, copyVecLength);
                    PipeBarrier<PIPE_ALL>();
                    VecComputeForPreComp(i, copyVecLength);
                    CopyOutForPreComp(i, copyVecLength);
                }

                PipeBarrier<PIPE_ALL>();
            }
        }
    
    public:
        TPipe *pipe_;
        IndexCodeAndPrecomputeTilingData tiling_;

    private:
        __aicore__ inline void CopyInForCode(int32_t progress, int32_t copyLength)
        {
            // 将当前progress的tile端搬运到 indexesLocal，并入队
            LocalTensor<T> indexesLocal = in_indexes_que.AllocTensor<T>();
            DataCopy(indexesLocal, indexes_gm_[progress * this->tileLength * this->dimLength],
                     copyLength * this->dimLength);
            in_indexes_que.EnQue(indexesLocal);
        }
        __aicore__ inline void VecComputeForCode(int32_t progress, int32_t copyLength)
        {
            LocalTensor<T> indexesLocal = in_indexes_que.DeQue<T>();

            LocalTensor<uint8_t> codesResultLocal = out_codes_que.AllocTensor<uint8_t>();
            LocalTensor<T> l1ResultLocal = out_l1_que.AllocTensor<T>();  // l1因子的GM暂存l1结果
            LocalTensor<T> l2ResultLocal = out_l2_que.AllocTensor<T>();  // l2的GM暂存内积结果

            LocalTensor<T> tmp_Local = tmp_que.AllocTensor<T>();

            for (int32_t i = 0; i < copyLength; i++) {
                Mul(tmp_Local[i * this->dimLength], indexesLocal[i * this->dimLength],
                    this->centerLocal, this->dimLength);
            }
            PipeBarrier<PIPE_V>();

            for (int32_t i = 0; i < copyLength; i++) { // dimLength不是64的倍数时？
                Add(tmp_Local[i * this->dimLength], tmp_Local[i * this->dimLength + this->mask],
                    tmp_Local[i * this->dimLength], this->mask, this->maskNum - 1, {1, 1, 1, 0, 8, 0});
            }
            PipeBarrier<PIPE_V>();

            int32_t repStride0 = 8 * this->maskNum;
            for (int32_t i = 0; i < copyLength; i += 255) {
                int32_t len = copyLength - i;
                if (len > 255) len = 255;
                WholeReduceSum<T>(l2ResultLocal[i], tmp_Local[i * this->dimLength], this->mask, len, 1, 1, repStride0);
            }
            PipeBarrier<PIPE_V>();

            // index-centor:未实现32B对齐，待修改为高级API
            for (int32_t i = 0; i < copyLength; i++) {
                Sub(indexesLocal[i * this->dimLength], indexesLocal[i * this->dimLength],
                  this->centerLocal, this->dimLength);
            }
            PipeBarrier<PIPE_V>();

            // 双值量化
            CompareScalar(codesResultLocal, indexesLocal, (float32_t)0, AscendC::CMPMODE::GE,
                          this->dimLength * copyLength);
            PipeBarrier<PIPE_V>();

            for (int32_t i = 0; i < copyLength; i++) {
                // Abs + ReduceSum
                Abs(indexesLocal[i * this->dimLength], indexesLocal[i * this->dimLength], this->dimLength);
                PipeBarrier<PIPE_V>();

                // dimLength不是64的倍数时？
                Add(indexesLocal[i * this->dimLength], indexesLocal[i * this->dimLength + this->mask],
                    indexesLocal[i * this->dimLength], this->mask, this->maskNum - 1, {1, 1, 1, 0, 8, 0});
                PipeBarrier<PIPE_V>();
            }

            int32_t repStride = 8 * this->maskNum;
            for (int32_t i = 0; i < copyLength; i += 255) {
                int32_t len = copyLength - i;
                if (len > 255) len = 255;
                WholeReduceSum<T>(l1ResultLocal[i], indexesLocal[i * this->dimLength],
                                  this->mask, len, 1, 1, repStride);
            }

            int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);

            out_codes_que.EnQue<uint8_t>(codesResultLocal);
            out_l1_que.EnQue(l1ResultLocal);
            out_l2_que.EnQue(l2ResultLocal);

            in_indexes_que.FreeTensor(indexesLocal);
            tmp_que.FreeTensor(tmp_Local);
        }
        __aicore__ inline void CopyOutForCode(int32_t progress, int32_t copyLength)
        {
            // 将当前progress的result搬运到vecsLocal，并enque到in_que
            LocalTensor<uint8_t> codesResultLocal = out_codes_que.DeQue<uint8_t>();
            LocalTensor<T> l1ResultLocal = out_l1_que.DeQue<T>();
            LocalTensor<T> l2ResultLocal = out_l2_que.DeQue<T>();
            
            DataCopyExtParams copyParams1{1, (uint32_t)(copyLength * this->dimLength / 8 * sizeof(uint8_t)), 0, 0, 0};
            DataCopyPad(codes_res_gm_[progress * this->tileLength * this->dimLength / 8],
                        codesResultLocal, copyParams1);
            
            DataCopyExtParams copyParams2{1, (uint32_t)(copyLength * sizeof(T)), 0, 0, 0};
            DataCopyPad(l1_res_gm_[progress * this->tileLength], l1ResultLocal, copyParams2);
            DataCopyPad(l2_res_gm_[progress * this->tileLength], l2ResultLocal, copyParams2);
        
            out_codes_que.FreeTensor(codesResultLocal);
            out_l1_que.FreeTensor(l1ResultLocal);
            out_l2_que.FreeTensor(l2ResultLocal);
        }
        
        __aicore__ inline void CopyInForPreComp(int32_t progress, int32_t copyLength)
        {
            // 将当前progress的tile端搬运到 indexesLocal，并入队
            LocalTensor<T> indexesl2Local = in_indexesl2_que.AllocTensor<T>();
            LocalTensor<T> indexSubCenterLocal = in_index_sub_center_que.AllocTensor<T>();
            LocalTensor<T> innerProductLocal = in_innerProductLocal_que.AllocTensor<T>();

            // 搬运一次
            DataCopyExtParams copyParams{1, (uint32_t)(copyLength * sizeof(T)), 0, 0, 0};
            uint8_t paddingNum = (RoundUp(copyLength * sizeof(T), 32) - copyLength * sizeof(T)) / 4;
            DataCopyPadExtParams<float> padParams{true, 0, paddingNum, 0};

            DataCopyPad(indexesl2Local, indexesl2_gm_[progress * this->tileLength], copyParams, padParams);
            // L1的结果放在l1_res_gm_中
            DataCopyPad(indexSubCenterLocal, l1_res_gm_[progress * this->tileLength], copyParams, padParams);
            // 内积的结果放在l2_res_gm_中
            DataCopyPad(innerProductLocal, l2_res_gm_[progress * this->tileLength], copyParams, padParams);

            in_indexesl2_que.EnQue(indexesl2Local);
            in_index_sub_center_que.EnQue(indexSubCenterLocal);
            in_innerProductLocal_que.EnQue(innerProductLocal);
        }
        __aicore__ inline void VecComputeForPreComp(int32_t progress, int32_t copyLength)
        {
            LocalTensor<T> indexesl2Local = in_indexesl2_que.DeQue<T>();
            LocalTensor<T> indexSubCenterLocal = in_index_sub_center_que.DeQue<T>();
            LocalTensor<T> innerProductLocal = in_innerProductLocal_que.DeQue<T>();

            LocalTensor<T> l2ResultLocal = out_l2_que.AllocTensor<T>();
            LocalTensor<T> l1ResultLocal = out_l1_que.AllocTensor<T>();

            // ||indexes||^2 - 2<indexes, centroid> + ||centroid||^2  未实现32B对齐，待修改为高级API
            Adds(l2ResultLocal, indexesl2Local, this->centerl2Localval, copyLength);
            PipeBarrier<PIPE_V>();
            Sub(l2ResultLocal, l2ResultLocal, innerProductLocal, copyLength);
            PipeBarrier<PIPE_V>();
            Sub(l2ResultLocal, l2ResultLocal, innerProductLocal, copyLength);
            PipeBarrier<PIPE_V>();

            // l1因子
            float32_t para_dim = 2 * sqrt((float)this->dimLength);
            Div(l1ResultLocal, l2ResultLocal, indexSubCenterLocal, copyLength);
            PipeBarrier<PIPE_V>();
            Muls(l1ResultLocal, l1ResultLocal, para_dim, copyLength);

            int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);

            // 结果入队
            out_l2_que.EnQue(l2ResultLocal);
            out_l1_que.EnQue(l1ResultLocal);

            in_indexesl2_que.FreeTensor(indexesl2Local);
            in_index_sub_center_que.FreeTensor(indexSubCenterLocal);
            in_innerProductLocal_que.FreeTensor(innerProductLocal);
        }
        __aicore__ inline void CopyOutForPreComp(int32_t progress, int32_t copyLength)
        {
            // 将当前progress的result搬运到vecsLocal，并enque到in_que
            LocalTensor<T> l2ResultLocal = out_l2_que.DeQue<T>();
            LocalTensor<T> l1ResultLocal = out_l1_que.DeQue<T>();

            DataCopyExtParams copyParams{1, (uint32_t)(copyLength * sizeof(T)), 0, 0, 0};
            DataCopyPad(l2_res_gm_[progress * this->tileLength], l2ResultLocal, copyParams);
            DataCopyPad(l1_res_gm_[progress * this->tileLength], l1ResultLocal, copyParams);

            out_l2_que.FreeTensor(l2ResultLocal);
            out_l1_que.FreeTensor(l1ResultLocal);
        }

    private:
        TQue<QuePosition::VECIN, 1> in_indexes_que, in_indexesl2_que;
        TQue<QuePosition::VECIN, 1> in_center_que, in_centerl2_que;
        TQue<QuePosition::VECIN, 1> in_index_sub_center_que, in_innerProductLocal_que;
        TQue<QuePosition::VECOUT, 1> out_l1_que, out_l2_que, out_codes_que;
        TQue<QuePosition::VECOUT, 1> out_l11_que, out_l22_que;

        TQue<QuePosition::VECCALC, 1> tmp_que;

        GlobalTensor<int32_t> vectorSize_gm_;
        GlobalTensor<T> indexes_gm_;
        GlobalTensor<T> indexesl2_gm_;
        GlobalTensor<T> centroid_gm_;
        GlobalTensor<T> centroidl2_gm_;

        GlobalTensor<uint8_t> codes_res_gm_;
        GlobalTensor<T> l2_res_gm_;
        GlobalTensor<T> l1_res_gm_;

        LocalTensor<T> centerLocal;

        float32_t centerl2Localval;

        int32_t dimLength;  // 向量维度
        int32_t vecLength;  // 当前core负责的向量数量
        int32_t vecOffset;  // 当前core负责的向量偏移

        int32_t tileNum;          // vector 单元循环次数
        int32_t tileLength;       // vector 单元每次循环处理的向量数量
        int32_t lastTileLength;   // vector 单元最后一次循环处理的向量数量

        int32_t tileLengthStage1;
        int32_t tileLengthStage2;

        int32_t mask = 64;
        int32_t maskNum;
    };
}

extern "C" __global__ __aicore__ void index_code_and_precompute(GM_ADDR vectorSize, GM_ADDR indexes, GM_ADDR indexesl2,
                                                                GM_ADDR centroid, GM_ADDR centroidl2,
                                                                GM_ADDR codes_result, GM_ADDR l2_result,
                                                                GM_ADDR l1_result, GM_ADDR workspace, GM_ADDR tiling) {

    GET_TILING_DATA(tiling_data, tiling);  // 获取算子的Tiling信息
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);  // 获取用户使用的workspace指针
    TPipe pipe;  // 统一管理Device端内存等资源
    kernels::IndexCodeAndPrecompute<float32_t> op(&pipe);  // 算子注册
    // 调用Init和Process函数，分别进行初始化和算子实现
    op.Init(vectorSize, indexes, indexesl2, centroid, centroidl2,
            codes_result, l2_result, l1_result, usrWorkspace, &tiling_data);
    op.Process();
}