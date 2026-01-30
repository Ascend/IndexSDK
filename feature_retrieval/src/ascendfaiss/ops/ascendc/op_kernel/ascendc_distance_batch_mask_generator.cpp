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
#include "kernel_utils.h"

using namespace AscendC;

namespace IndexOps {
class AscendcDistanceBatchMaskGenerator {
public:
    __aicore__ inline AscendcDistanceBatchMaskGenerator() = default;

    __aicore__ inline void Init(
        GM_ADDR queryTimeStamp, GM_ADDR queryTokenSet, GM_ADDR dbTimeStamp, GM_ADDR dbDivisor, GM_ADDR dbRemainder,
        GM_ADDR distanceMask, GM_ADDR workspace, const AscendcDistanceBatchMaskGeneratorTilingData* tilingData,
        TPipe* pipe)
    {
        this->coreNum = GetBlockNum();
        this->coreId = GetBlockIdx();
        if (this->coreId >= this->coreNum) {
            return;
        }
        InitTiling(tilingData);
        this->queryTimeStampGm.SetGlobalBuffer((__gm__ int32_t*)queryTimeStamp);
        this->queryTokenSetGm.SetGlobalBuffer((__gm__ uint8_t*)queryTokenSet);
        if (this->coreId < this->formerNum) {
            this->offset = this->coreId * this->tileLen * this->formerRepeatNum;
        } else {
            this->offset = this->formerNum * this->tileLen * this->formerRepeatNum +
                           (this->coreId - this->formerNum) * this->tileLen * this->tailRepeatNum;
        }
        this->m = (this->formerNum * this->tileLen * this->formerRepeatNum +
                   (this->coreNum - this->formerNum) * this->tileLen * this->tailRepeatNum) /
                  8;
        this->dbTimeStampGm.SetGlobalBuffer((__gm__ int32_t*)dbTimeStamp + this->offset);
        this->dbDivisorGm.SetGlobalBuffer((__gm__ int32_t*)dbDivisor + this->offset);
        this->dbRemainderGm.SetGlobalBuffer((__gm__ uint8_t*)dbRemainder + this->offset * 2);
        this->distanceMaskGm.SetGlobalBuffer((__gm__ uint8_t*)distanceMask + this->offset / 8);

        pipe->InitBuffer(this->queryTimeStampBuf, this->queryTimeStampLenAlign * sizeof(int32_t));
        pipe->InitBuffer(this->queryTokenSetBuf, this->queryTokenSetLenAlign * sizeof(uint8_t));
        pipe->InitBuffer(this->dbTimeStampBuf, this->dbTimeStampLenAlign * sizeof(int32_t));
        pipe->InitBuffer(this->dbTimeStampFloat32Buf, this->dbTimeStampLenAlign * sizeof(float32_t));
        pipe->InitBuffer(this->dbDivisorBuf, this->dbDivisorLenAlign * sizeof(int32_t));
        pipe->InitBuffer(this->dbRemainderBuf, this->dbRemainderLenAlign * sizeof(uint8_t));
        pipe->InitBuffer(this->distanceMaskBuf, this->distanceMaskLenAlign * sizeof(uint8_t));
        pipe->InitBuffer(this->timeStampCmpResBuf, 3 * this->distanceMaskLenAlign * sizeof(uint8_t));
        pipe->InitBuffer(this->tmpRemainderBuf, this->dbDivisorLenAlign * sizeof(int16_t));
        pipe->InitBuffer(this->resRemainderBuf, this->dbDivisorLenAlign * sizeof(int16_t));
        pipe->InitBuffer(this->tokenCmpResBuf, this->dbDivisorLenAlign * sizeof(uint8_t));

        this->queryTimeStampLocal = this->queryTimeStampBuf.Get<int32_t>();
        this->queryTokenSetLocal = this->queryTokenSetBuf.Get<uint8_t>();
        this->dbTimeStampLocal = this->dbTimeStampBuf.Get<int32_t>();
        this->dbTimeStampFloat32Local = this->dbTimeStampFloat32Buf.Get<float32_t>();
        this->dbDivisorLocal = this->dbDivisorBuf.Get<int32_t>();
        this->dbRemainderLocal = this->dbRemainderBuf.Get<uint8_t>();
        this->distanceMaskLocal = this->distanceMaskBuf.Get<uint8_t>();
        this->timeStampCmpResLocal = this->timeStampCmpResBuf.Get<uint8_t>();
        this->timeStampCmpResTmpLocal1 = this->timeStampCmpResLocal[this->distanceMaskLenAlign];
        this->timeStampCmpResTmpLocal2 = this->timeStampCmpResLocal[2 * this->distanceMaskLenAlign];
        this->tmpRemainderLocal = this->tmpRemainderBuf.Get<int16_t>();
        this->resRemainderLocal = this->resRemainderBuf.Get<int16_t>();
        this->tokenCmpResLocal = this->tokenCmpResBuf.Get<uint8_t>();
    }

    __aicore__ inline void Process()
    {
        if (this->coreId >= this->coreNum) {
            return;
        }
        DataCopy(this->queryTimeStampLocal, this->queryTimeStampGm, this->queryTimeStampLenAlign);
        for (uint32_t bid = 0; bid < this->batchSize; ++bid) {
            DataCopyPad(
                this->queryTokenSetLocal, this->queryTokenSetGm[bid * this->queryTokenSetLen],
                {1, static_cast<uint32_t>(this->queryTokenSetLen), 0, 0, 0}, {false, 0, 0, 0});
            SetFlag<HardEvent::V_MTE2>(0);
            SetFlag<HardEvent::V_MTE2>(1);
            SetFlag<HardEvent::MTE3_V>(2);
            for (uint32_t loop = 0; loop < this->repeatNum; ++loop) {
                WaitFlag<HardEvent::V_MTE2>(0);
                DataCopy(
                    this->dbTimeStampLocal, this->dbTimeStampGm[loop * this->dbTimeStampLenAlign],
                    this->dbTimeStampLenAlign);
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::V_MTE2>(1);
                DataCopy(
                    this->dbDivisorLocal, this->dbDivisorGm[loop * this->dbDivisorLenAlign], this->dbDivisorLenAlign);
                DataCopy(
                    this->dbRemainderLocal, this->dbRemainderGm[loop * this->dbRemainderLenAlign],
                    this->dbRemainderLenAlign);
                CompareTimeStamp(bid);
                SetFlag<HardEvent::V_MTE2>(0);
                CompareTokenId();
                SetFlag<HardEvent::V_MTE2>(1);
                WaitFlag<HardEvent::MTE3_V>(2);
                LocalTensor<int16_t> distanceMaskInt16Local = this->distanceMaskLocal.ReinterpretCast<int16_t>();
                And(distanceMaskInt16Local, this->timeStampCmpResInt16Local, this->tokenCmpResInt16Local,
                    this->distanceMaskLenAlign / 2);
                SetFlag<HardEvent::V_MTE3>(2);
                WaitFlag<HardEvent::V_MTE3>(2);
                DataCopy(
                    this->distanceMaskGm[bid * this->m + loop * this->distanceMaskLenAlign], this->distanceMaskLocal,
                    this->distanceMaskLenAlign);
                SetFlag<HardEvent::MTE3_V>(2);
            }
            WaitFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(1);
            WaitFlag<HardEvent::MTE3_V>(2);
        }
    }

private:
    GlobalTensor<int32_t> queryTimeStampGm;
    GlobalTensor<uint8_t> queryTokenSetGm;
    GlobalTensor<int32_t> dbTimeStampGm;
    GlobalTensor<int32_t> dbDivisorGm;
    GlobalTensor<uint8_t> dbRemainderGm;
    GlobalTensor<uint8_t> distanceMaskGm;

    TBuf<TPosition::VECCALC> queryTimeStampBuf;
    TBuf<TPosition::VECCALC> queryTokenSetBuf;
    TBuf<TPosition::VECCALC> dbTimeStampBuf;
    TBuf<TPosition::VECCALC> dbTimeStampFloat32Buf;
    TBuf<TPosition::VECCALC> dbDivisorBuf;
    TBuf<TPosition::VECCALC> dbRemainderBuf;
    TBuf<TPosition::VECCALC> distanceMaskBuf;
    TBuf<TPosition::VECCALC> timeStampCmpResBuf;
    TBuf<TPosition::VECCALC> tmpRemainderBuf;
    TBuf<TPosition::VECCALC> resRemainderBuf;
    TBuf<TPosition::VECCALC> tokenCmpResBuf;

    LocalTensor<int32_t> queryTimeStampLocal;
    LocalTensor<uint8_t> queryTokenSetLocal;
    LocalTensor<int32_t> dbTimeStampLocal;
    LocalTensor<float32_t> dbTimeStampFloat32Local;
    LocalTensor<int32_t> dbDivisorLocal;
    LocalTensor<uint8_t> dbRemainderLocal;
    LocalTensor<uint8_t> distanceMaskLocal;
    LocalTensor<uint8_t> timeStampCmpResLocal;
    LocalTensor<uint8_t> timeStampCmpResTmpLocal1;
    LocalTensor<uint8_t> timeStampCmpResTmpLocal2;
    LocalTensor<int16_t> timeStampCmpResInt16Local;
    LocalTensor<int16_t> tmpRemainderLocal;
    LocalTensor<int16_t> resRemainderLocal;
    LocalTensor<uint8_t> tokenCmpResLocal;
    LocalTensor<int16_t> tokenCmpResInt16Local;

    uint32_t coreNum;
    uint32_t coreId;
    uint32_t batchSize;
    uint32_t tokenCnt;
    uint32_t formerNum;
    uint32_t formerRepeatNum;
    uint32_t tailRepeatNum;
    uint32_t repeatNum;
    uint32_t tileLen;
    int32_t offset;
    uint32_t m;

    uint32_t queryTimeStampLen;
    uint32_t queryTokenSetLen;
    uint32_t dbTimeStampLen;
    uint32_t dbDivisorLen;
    uint32_t dbRemainderLen;
    uint32_t distanceMaskLen;
    uint32_t queryTimeStampLenAlign;
    uint32_t queryTokenSetLenAlign;
    uint32_t dbTimeStampLenAlign;
    uint32_t dbDivisorLenAlign;
    uint32_t dbRemainderLenAlign;
    uint32_t distanceMaskLenAlign;

private:
    __aicore__ inline void InitTiling(const AscendcDistanceBatchMaskGeneratorTilingData* tilingData)
    {
        this->batchSize = tilingData->batchSize;
        this->tokenCnt = tilingData->tokenCnt;
        this->formerNum = tilingData->formerNum;
        this->tileLen = tilingData->tileLen;
        if (this->coreId < this->formerNum) {
            this->formerRepeatNum = tilingData->formerRepeatNum;
            this->repeatNum = tilingData->formerRepeatNum;
        } else {
            this->tailRepeatNum = tilingData->tailRepeatNum;
            this->repeatNum = tilingData->tailRepeatNum;
        }
        this->queryTimeStampLen = 8 * this->batchSize;
        this->queryTokenSetLen = this->tokenCnt;
        this->dbTimeStampLen = this->tileLen;
        this->dbDivisorLen = this->tileLen;
        this->dbRemainderLen = 2 * this->tileLen;
        this->distanceMaskLen = this->tileLen / 8;

        this->queryTimeStampLenAlign = AlignUp(this->queryTimeStampLen, 8);
        this->queryTokenSetLenAlign = AlignUp(this->queryTokenSetLen, 32);
        this->dbTimeStampLenAlign = AlignUp(this->dbTimeStampLen, 8);
        this->dbDivisorLenAlign = AlignUp(this->dbDivisorLen, 8);
        this->dbRemainderLenAlign = AlignUp(this->dbRemainderLen, 32);
        this->distanceMaskLenAlign = AlignUp(this->distanceMaskLen, 32);
    }

    __aicore__ inline void CompareTimeStamp(uint32_t bid)
    {
        SetFlag<HardEvent::MTE2_S>(0);
        WaitFlag<HardEvent::MTE2_S>(0);
        int32_t startTime = this->queryTimeStampLocal.GetValue(bid * 8);
        int32_t endTime = this->queryTimeStampLocal.GetValue(bid * 8 + 1);
        WaitFlag<HardEvent::MTE2_V>(0);
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
        this->timeStampCmpResInt16Local = this->timeStampCmpResLocal.ReinterpretCast<int16_t>();
        if (startTime == 0 && endTime == -INT32_MAX) {
            Duplicate(this->timeStampCmpResInt16Local, (int16_t)(-1), this->distanceMaskLenAlign / 2);
            return;
        }
        if (startTime < endTime) {
            Duplicate(this->timeStampCmpResInt16Local, (int16_t)(0), this->distanceMaskLenAlign / 2);
            return;
        }
        Adds(this->dbTimeStampLocal, this->dbTimeStampLocal, startTime, this->dbTimeStampLenAlign);
        Cast(this->dbTimeStampFloat32Local,
            this->dbTimeStampLocal,
            AscendC::RoundMode::CAST_NONE,
            this->dbTimeStampLenAlign);
        CompareScalar(this->timeStampCmpResTmpLocal1,
            this->dbTimeStampFloat32Local,
            (float32_t)(0.0),
            AscendC::CMPMODE::GE,
            this->dbTimeStampLenAlign);
        Adds(this->dbTimeStampLocal, this->dbTimeStampLocal, (endTime - startTime), this->dbTimeStampLenAlign);
        Cast(this->dbTimeStampFloat32Local,
            this->dbTimeStampLocal,
            AscendC::RoundMode::CAST_NONE,
            this->dbTimeStampLenAlign);
        CompareScalar(
            this->timeStampCmpResTmpLocal2, this->dbTimeStampFloat32Local, (float32_t)(0.0), AscendC::CMPMODE::LE,
            this->dbTimeStampLenAlign);
        LocalTensor<int16_t> timeStampCmpResTmpInt16Local1 = this->timeStampCmpResTmpLocal1.ReinterpretCast<int16_t>();
        LocalTensor<int16_t> timeStampCmpResTmpInt16Local2 = this->timeStampCmpResTmpLocal2.ReinterpretCast<int16_t>();
        And(this->timeStampCmpResInt16Local, timeStampCmpResTmpInt16Local1, timeStampCmpResTmpInt16Local2,
            this->distanceMaskLenAlign / 2);
    }

    __aicore__ inline void CompareTokenId()
    {
        LocalTensor<int16_t> queryTokenSetInt16Local = this->queryTokenSetLocal.ReinterpretCast<int16_t>();
        SetFlag<HardEvent::MTE2_V>(1);
        WaitFlag<HardEvent::MTE2_V>(1);
        LocalTensor<uint32_t> dbDivisorUint32Local = this->dbDivisorLocal.ReinterpretCast<uint32_t>();
        Gather(this->tmpRemainderLocal, queryTokenSetInt16Local, dbDivisorUint32Local, 0, this->dbDivisorLenAlign);
        LocalTensor<int16_t> dbRemainderInt16Local = this->dbRemainderLocal.ReinterpretCast<int16_t>();
        And(this->resRemainderLocal, this->tmpRemainderLocal, dbRemainderInt16Local, this->dbDivisorLenAlign);
        LocalTensor<float16_t> resRemainderFloat16Local = this->resRemainderLocal.ReinterpretCast<float16_t>();
        CompareScalar(
            this->tokenCmpResLocal, resRemainderFloat16Local, (float16_t)(2.0), AscendC::CMPMODE::GT,
            this->dbDivisorLenAlign);
        this->tokenCmpResInt16Local = this->tokenCmpResLocal.ReinterpretCast<int16_t>();
    }
};
} // namespace IndexOps

extern "C" __global__ __aicore__ void ascendc_distance_batch_mask_generator(
    GM_ADDR queryTimeStamp, GM_ADDR queryTokenSet, GM_ADDR dbTimeStamp, GM_ADDR dbDivisor, GM_ADDR dbRemainder,
    GM_ADDR distanceMask, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    IndexOps::AscendcDistanceBatchMaskGenerator op;
    op.Init(
        queryTimeStamp, queryTokenSet, dbTimeStamp, dbDivisor, dbRemainder, distanceMask, workspace, &tilingData,
        &pipe);
    op.Process();
}