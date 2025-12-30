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

using namespace AscendC;

namespace AscendC
{
    template <typename T>
    class DistanceFlatL2MinsAtFP32
    {
    public:
        __aicore__ inline DistanceFlatL2MinsAtFP32() {}
        __aicore__ inline void Init(GM_ADDR query, GM_ADDR codes, GM_ADDR dist_result, GM_ADDR min_result,
            GM_ADDR flag_shape, GM_ADDR workspace, int32_t formerCoreNum, int32_t formerCoreLength, int32_t tailCoreNum,
            int32_t tailCoreLength, int32_t tileNum, int32_t tileLength, int32_t lastTileLength, int32_t queryNumLength,
            int32_t codesNumLength, int32_t dimLength)
        {
            this->blockIdx = GetBlockIdx();
            this->formerCoreNum = formerCoreNum;
            this->formerCoreLength = formerCoreLength;
            this->tailCoreNum = tailCoreNum;
            this->tailCoreLength = tailCoreLength;

            this->tileNum = tileNum;
            this->tileLength = tileLength;
            this->lastTileLength = lastTileLength;

            this->queryNumLength = queryNumLength;
            this->codesNumLength = codesNumLength;
            this->dimLength = dimLength;
            if (this->blockIdx < this->formerCoreNum) {
                this->copyQueryLength = this->formerCoreLength;
                this->queryOffset = this->blockIdx * this->formerCoreLength;
                queryGlobal.SetGlobalBuffer((__gm__ T *)query + this->queryOffset * this->dimLength,
                                            this->formerCoreLength * this->dimLength);
            } else {
                this->copyQueryLength = this->tailCoreLength;
                this->queryOffset = this->formerCoreNum * this->formerCoreLength +
                                    (this->blockIdx - this->formerCoreNum) * this->tailCoreLength;
                queryGlobal.SetGlobalBuffer((__gm__ T *)query + this->queryOffset * this->dimLength,
                                            this->tailCoreLength * this->dimLength);
            }
            codesGlobal.SetGlobalBuffer((__gm__ T *)codes, this->codesNumLength * this->dimLength);
            distResultGlobal.SetGlobalBuffer((__gm__ T *)dist_result, this->queryNumLength * this->codesNumLength);
            minResultGlobal.SetGlobalBuffer((__gm__ T *)min_result, this->queryNumLength *
                                            (this->codesNumLength + 63) / this->mask * 2);
            flagShapeGlobal.SetGlobalBuffer((__gm__ uint16_t *)flag_shape,
                                            (this->formerCoreNum + this->tailCoreNum) * 16);

            pipe.InitBuffer(inQueueQuery, 1, this->dimLength * sizeof(T));
            pipe.InitBuffer(inQueueCodes, 1, this->tileLength * this->dimLength * sizeof(T));
            pipe.InitBuffer(inQueueFlag, 1, 16 * sizeof(uint16_t));
            pipe.InitBuffer(distResultQueue, 1, this->tileLength * sizeof(T));
            pipe.InitBuffer(minResultQueue, 1, this->tileLength * this->mask * sizeof(T));
            pipe.InitBuffer(flagShapeQueue, 1, 16 * sizeof(uint16_t));
            pipe.InitBuffer(processBuf, this->tileLength * this->dimLength * sizeof(T));
            pipe.InitBuffer(process64AddBuf, this->tileLength * this->mask * sizeof(T));
        }

        __aicore__ inline void Process()
        {
            for (int32_t i = 0; i < this->copyQueryLength; i++) {
                LocalTensor<T> queryLocal = inQueueQuery.AllocTensor<T>();
                DataCopy(queryLocal, queryGlobal[i * this->dimLength], this->dimLength);
                inQueueQuery.EnQue(queryLocal);
                LocalTensor<T> queryDeQueLocal = inQueueQuery.DeQue<T>();
                LocalTensor<T> minResultLocal;
                PipeBarrier<PIPE_V>();
                this->minNum = 0;
                for (int32_t j = 0; j < this->tileNum; j++)
                {
                    if (j % 64 == 0) {
                        minResultLocal = minResultQueue.AllocTensor<T>();
                    }
                    uint16_t copyCodesLength = this->tileLength;
                    if (j == this->tileNum - 1)
                    {
                        copyCodesLength = this->lastTileLength;
                    }
                    CopyIn(j, copyCodesLength);
                    Compute(j, copyCodesLength, i, queryDeQueLocal, minResultLocal);
                    CopyOut(j, copyCodesLength, i);
                }
                inQueueQuery.FreeTensor(queryDeQueLocal);
            }
            LocalTensor<uint16_t> flagLocal = inQueueFlag.AllocTensor<uint16_t>();
            inQueueFlag.EnQue(flagLocal);
            flagLocal = inQueueFlag.DeQue<uint16_t>();
            Duplicate(flagLocal, this->zero_uint16, 16);
            flagLocal.SetValue(0, 1);
            LocalTensor<uint16_t> flagOutLocal = flagShapeQueue.AllocTensor<uint16_t>();
            DataCopy(flagOutLocal, flagLocal, 16);
            flagShapeQueue.EnQue(flagOutLocal);
            inQueueFlag.FreeTensor(flagLocal);
            flagOutLocal = flagShapeQueue.DeQue<uint16_t>();
            DataCopy(flagShapeGlobal[this->blockIdx * 16], flagOutLocal, 16);
            flagShapeQueue.FreeTensor(flagOutLocal);
        }

    private:
        __aicore__ inline void CopyIn(int32_t progress, int32_t copyCodesLength)
        {
            LocalTensor<T> codesLocal = inQueueCodes.AllocTensor<T>();
            DataCopy(codesLocal, codesGlobal[progress * this->tileLength * this->dimLength],
                     copyCodesLength * this->dimLength);
            inQueueCodes.EnQue(codesLocal);
        }

        __aicore__ inline void Compute(int32_t progress, int32_t copyCodesLength, int32_t queryLength,
                                       LocalTensor<T> queryLocal, LocalTensor<T> minResultLocal)
        {
            LocalTensor<T> codesLocal = inQueueCodes.DeQue<T>();
            LocalTensor<T> processLocal = processBuf.Get<T>();
            LocalTensor<T> process64AddLocal = process64AddBuf.Get<T>();
            LocalTensor<T> distResultLocal = distResultQueue.AllocTensor<T>();
            int32_t maskNum = this->dimLength / this->mask;
            for (int32_t i = 0; i < maskNum; i++) {
                uint8_t repStride = 8 * maskNum;
                Sub(processLocal[i * this->mask], queryLocal[i * this->mask],
                    codesLocal[i * this->mask], this->mask, copyCodesLength, {1, 1, 1, repStride, 0, repStride});
            }
            PipeBarrier<PIPE_V>();
            Mul(processLocal, processLocal, processLocal, copyCodesLength * this->dimLength);
            PipeBarrier<PIPE_V>();
            Duplicate(process64AddLocal, this->zero_float, copyCodesLength * this->mask);
            PipeBarrier<PIPE_V>();
            for (int32_t i = 0; i < copyCodesLength; i++) {
                Add(process64AddLocal[i * this->mask], processLocal[i * this->dimLength],
                    process64AddLocal[i * this->mask], this->mask, maskNum, {1, 1, 1, 0, 8, 0});
            }
            PipeBarrier<PIPE_V>();
            WholeReduceSum<T>(distResultLocal, process64AddLocal, this->mask, copyCodesLength, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            DataCopy(minResultLocal[(progress % this->mask) * this->tileLength], distResultLocal, copyCodesLength);
            if (progress == this->tileNum - 1) {
                int32_t minLength = (this->codesNumLength - this->minNum * this->mask *
                                     this->tileLength + this->mask - 1) / this->mask;
                PipeBarrier<PIPE_V>();
                WholeReduceMin(minResultLocal, minResultLocal, this->mask, minLength, 1, 1, 8);
                minResultQueue.EnQue(minResultLocal);
                CopyOutMin(progress, queryLength, minLength);
            } else if ((progress + 1) % this->mask == 0) {
                int32_t minLength = this->tileLength;
                PipeBarrier<PIPE_V>();
                WholeReduceMin(minResultLocal, minResultLocal, this->mask, this->tileLength, 1, 1, 8);
                minResultQueue.EnQue(minResultLocal);
                CopyOutMin(progress, queryLength, minLength);
            }
            distResultQueue.EnQue(distResultLocal);
            inQueueCodes.FreeTensor(codesLocal);
        }

        __aicore__ inline void CopyOut(int32_t progress, uint16_t copyCodesLength, int32_t queryLength)
        {
            LocalTensor<T> distResultLocal = distResultQueue.DeQue<T>();
            DataCopy(distResultGlobal[(this->queryOffset + queryLength) *
                     this->codesNumLength + progress * this->tileLength], distResultLocal, copyCodesLength);
            distResultQueue.FreeTensor(distResultLocal);
        }

        __aicore__ inline void CopyOutMin(int32_t progress, int32_t queryLength, int32_t minLength)
        {
            LocalTensor<T> minResultLocal = minResultQueue.DeQue<T>();
            int32_t minOffset = (this->codesNumLength + this->mask - 1) / this->mask;
            DataCopy(minResultGlobal[(this->queryOffset + queryLength) * minOffset * 2 +
                     this->minNum * this->tileLength * 2], minResultLocal, minLength * 2);
            minResultQueue.FreeTensor(minResultLocal);
            this->minNum = this->minNum + 1;
        }

    private:
        TPipe pipe;
        TQue<QuePosition::VECIN, 1> inQueueQuery, inQueueCodes, inQueueFlag;
        TQue<QuePosition::VECOUT, 1> distResultQueue, minResultQueue, flagShapeQueue;
        TBuf<TPosition::VECCALC> processBuf;
        TBuf<TPosition::VECCALC> process64AddBuf;

        GlobalTensor<T> queryGlobal;
        GlobalTensor<T> codesGlobal;
        GlobalTensor<T> distResultGlobal;
        GlobalTensor<T> minResultGlobal;
        GlobalTensor<uint16_t> flagShapeGlobal;

        int32_t blockIdx = 0;
        int32_t formerCoreNum;
        int32_t formerCoreLength;
        int32_t tailCoreNum;
        int32_t tailCoreLength;
        int32_t tileNum;
        int32_t tileLength;
        int32_t lastTileLength;

        int32_t queryNumLength;
        int32_t codesNumLength;
        int32_t dimLength;
        int32_t copyQueryLength;
        int32_t queryOffset;
        int32_t mask = 64;
        float32_t zero_float = 0.0;
        uint16_t zero_uint16 = 0;
        int32_t minNum;
    };
}

extern "C" __global__ __aicore__ void distance_flat_l2_mins_at_fp32(GM_ADDR query, GM_ADDR codes,
    GM_ADDR dist_result, GM_ADDR min_result, GM_ADDR flag_shape, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    DistanceFlatL2MinsAtFP32<float32_t> op;
    op.Init(query, codes, dist_result, min_result, flag_shape, workspace,
        tiling_data.formerCoreNum,
        tiling_data.formerCoreLength,
        tiling_data.tailCoreNum,
        tiling_data.tailCoreLength,
        tiling_data.tileNum,
        tiling_data.tileLength,
        tiling_data.lastTileLength,
        tiling_data.queryNumLength,
        tiling_data.codesNumLength,
        tiling_data.dimLength);
    op.Process();
}