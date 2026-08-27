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

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"

using namespace AscendC;

namespace IndexOps
{
class AscendcDistanceBatchValMaskGenerator
{
   public:
    __aicore__ inline AscendcDistanceBatchValMaskGenerator() = default;

    __aicore__ inline void Init(GM_ADDR queryTimeStamp, GM_ADDR queryTokenSet, GM_ADDR dbTimeStamp, GM_ADDR dbDivisor,
                                GM_ADDR dbRemainder, GM_ADDR extraValFilter, GM_ADDR extraValAttr, GM_ADDR distanceMask,
                                const AscendcDistanceBatchValMaskGeneratorTilingData* tilingData, TPipe* pipe)
    {
        this->coreNum = GetBlockNum();
        this->coreId = GetBlockIdx();
        if (this->coreId >= this->coreNum)
        {
            return;
        }
        InitTiling(tilingData);
        this->queryTimeStampGm.SetGlobalBuffer((__gm__ int32_t*)queryTimeStamp);
        this->queryTokenSetGm.SetGlobalBuffer((__gm__ uint8_t*)queryTokenSet);
        this->extraValFilterGm.SetGlobalBuffer((__gm__ int16_t*)extraValFilter);
        if (this->coreId < this->formerNum)
        {
            this->offset = this->coreId * this->tileLen * this->formerRepeatNum;
        }
        else
        {
            this->offset = this->formerNum * this->tileLen * this->formerRepeatNum +
                           (this->coreId - this->formerNum) * this->tileLen * this->tailRepeatNum;
        }
        this->maskStride = (this->formerNum * this->tileLen * this->formerRepeatNum +
                            (this->coreNum - this->formerNum) * this->tileLen * this->tailRepeatNum) /
                           8;
        this->dbTimeStampGm.SetGlobalBuffer((__gm__ int32_t*)dbTimeStamp + this->offset);
        this->dbDivisorGm.SetGlobalBuffer((__gm__ int32_t*)dbDivisor + this->offset);
        this->dbRemainderGm.SetGlobalBuffer((__gm__ uint8_t*)dbRemainder + this->offset * 2);
        this->extraValAttrGm.SetGlobalBuffer((__gm__ int16_t*)extraValAttr + this->offset);
        this->distanceMaskGm.SetGlobalBuffer((__gm__ uint8_t*)distanceMask + this->offset / 8);

        pipe->InitBuffer(this->queryTimeStampBuf, this->queryTimeStampLenAlign * sizeof(int32_t));
        pipe->InitBuffer(this->queryTokenSetBuf, this->queryTokenSetLenAlign * sizeof(uint8_t));
        pipe->InitBuffer(this->extraValFilterBuf, EXTRA_VAL_FILTER_LEN * sizeof(int16_t));
        pipe->InitBuffer(this->dbTimeStampBuf, this->dbTimeStampLenAlign * sizeof(int32_t));
        pipe->InitBuffer(this->dbTimeStampFloat32Buf, this->dbTimeStampLenAlign * sizeof(float32_t));
        pipe->InitBuffer(this->dbDivisorBuf, this->dbDivisorLenAlign * sizeof(int32_t));
        pipe->InitBuffer(this->dbRemainderBuf, this->dbRemainderLenAlign * sizeof(uint8_t));
        pipe->InitBuffer(this->distanceMaskBuf, this->distanceMaskLenAlign * sizeof(uint8_t));
        pipe->InitBuffer(this->extraValMaskBuf, this->distanceMaskLenAlign * sizeof(uint8_t));
        pipe->InitBuffer(this->timeStampCmpResBuf, 3 * this->distanceMaskLenAlign * sizeof(uint8_t));
        pipe->InitBuffer(this->tmpRemainderBuf, this->dbDivisorLenAlign * sizeof(int16_t));
        pipe->InitBuffer(this->resRemainderBuf, this->dbDivisorLenAlign * sizeof(int16_t));
        pipe->InitBuffer(this->tokenCmpResBuf, this->dbDivisorLenAlign * sizeof(uint8_t));

        this->queryTimeStampLocal = this->queryTimeStampBuf.Get<int32_t>();
        this->queryTokenSetLocal = this->queryTokenSetBuf.Get<uint8_t>();
        this->extraValFilterLocal = this->extraValFilterBuf.Get<int16_t>();
        this->dbTimeStampLocal = this->dbTimeStampBuf.Get<int32_t>();
        this->dbTimeStampFloat32Local = this->dbTimeStampFloat32Buf.Get<float32_t>();
        this->dbDivisorLocal = this->dbDivisorBuf.Get<int32_t>();
        this->dbRemainderLocal = this->dbRemainderBuf.Get<uint8_t>();
        this->distanceMaskLocal = this->distanceMaskBuf.Get<uint8_t>();
        this->extraValMaskLocal = this->extraValMaskBuf.Get<uint8_t>();
        this->timeStampCmpResLocal = this->timeStampCmpResBuf.Get<uint8_t>();
        this->timeStampCmpResTmpLocal1 = this->timeStampCmpResLocal[this->distanceMaskLenAlign];
        this->timeStampCmpResTmpLocal2 = this->timeStampCmpResLocal[2 * this->distanceMaskLenAlign];
        this->tmpRemainderLocal = this->tmpRemainderBuf.Get<int16_t>();
        this->resRemainderLocal = this->resRemainderBuf.Get<int16_t>();
        this->tokenCmpResLocal = this->tokenCmpResBuf.Get<uint8_t>();
    }

    __aicore__ inline void Process()
    {
        if (this->coreId >= this->coreNum)
        {
            return;
        }
        DataCopy(this->queryTimeStampLocal, this->queryTimeStampGm, this->queryTimeStampLenAlign);
        for (uint32_t bid = 0; bid < this->batchSize; ++bid)
        {
            DataCopyPad(this->queryTokenSetLocal, this->queryTokenSetGm[bid * this->queryTokenSetLen],
                        {1, this->queryTokenSetLen, 0, 0, 0}, {false, 0, 0, 0});
            DataCopy(this->extraValFilterLocal, this->extraValFilterGm[bid * EXTRA_VAL_FILTER_LEN],
                     EXTRA_VAL_FILTER_LEN);
            SetFlag<HardEvent::MTE2_S>(FILTER_EVENT_ID);
            WaitFlag<HardEvent::MTE2_S>(FILTER_EVENT_ID);
            const int16_t filterValue = this->extraValFilterLocal.GetValue(0);
            const int16_t matchMode = this->extraValFilterLocal.GetValue(1);

            SetFlag<HardEvent::V_MTE2>(TIME_EVENT_ID);
            SetFlag<HardEvent::V_MTE2>(TOKEN_EVENT_ID);
            SetFlag<HardEvent::MTE3_V>(OUTPUT_EVENT_ID);
            for (uint32_t loop = 0; loop < this->repeatNum; ++loop)
            {
                WaitFlag<HardEvent::V_MTE2>(TIME_EVENT_ID);
                DataCopy(this->dbTimeStampLocal, this->dbTimeStampGm[loop * this->dbTimeStampLenAlign],
                         this->dbTimeStampLenAlign);
                SetFlag<HardEvent::MTE2_V>(TIME_EVENT_ID);
                WaitFlag<HardEvent::V_MTE2>(TOKEN_EVENT_ID);
                DataCopy(this->dbDivisorLocal, this->dbDivisorGm[loop * this->dbDivisorLenAlign],
                         this->dbDivisorLenAlign);
                DataCopy(this->dbRemainderLocal, this->dbRemainderGm[loop * this->dbRemainderLenAlign],
                         this->dbRemainderLenAlign);
                CompareTimeStamp(bid);
                SetFlag<HardEvent::V_MTE2>(TIME_EVENT_ID);
                CompareTokenId();
                SetFlag<HardEvent::V_MTE2>(TOKEN_EVENT_ID);
                WaitFlag<HardEvent::MTE3_V>(OUTPUT_EVENT_ID);
                LocalTensor<int16_t> distanceMaskInt16Local = this->distanceMaskLocal.ReinterpretCast<int16_t>();
                And(distanceMaskInt16Local, this->timeStampCmpResInt16Local, this->tokenCmpResInt16Local,
                    this->distanceMaskLenAlign / 2);
                CompareExtraVal(loop, filterValue, matchMode);
                LocalTensor<int16_t> extraValMaskInt16Local = this->extraValMaskLocal.ReinterpretCast<int16_t>();
                And(distanceMaskInt16Local, distanceMaskInt16Local, extraValMaskInt16Local,
                    this->distanceMaskLenAlign / 2);
                SetFlag<HardEvent::V_MTE3>(OUTPUT_EVENT_ID);
                WaitFlag<HardEvent::V_MTE3>(OUTPUT_EVENT_ID);
                DataCopy(this->distanceMaskGm[bid * this->maskStride + loop * this->distanceMaskLenAlign],
                         this->distanceMaskLocal, this->distanceMaskLenAlign);
                SetFlag<HardEvent::MTE3_V>(OUTPUT_EVENT_ID);
            }
            WaitFlag<HardEvent::V_MTE2>(TIME_EVENT_ID);
            WaitFlag<HardEvent::V_MTE2>(TOKEN_EVENT_ID);
            WaitFlag<HardEvent::MTE3_V>(OUTPUT_EVENT_ID);
        }
    }

   private:
    static constexpr uint32_t EXTRA_VAL_FILTER_LEN = 16;
    static constexpr uint32_t TIME_EVENT_ID = 0;
    static constexpr uint32_t TOKEN_EVENT_ID = 1;
    static constexpr uint32_t OUTPUT_EVENT_ID = 2;
    static constexpr uint32_t FILTER_EVENT_ID = 3;
    static constexpr float TOKEN_MATCH_THRESHOLD = 2.0F;

    GlobalTensor<int32_t> queryTimeStampGm;
    GlobalTensor<uint8_t> queryTokenSetGm;
    GlobalTensor<int16_t> extraValFilterGm;
    GlobalTensor<int32_t> dbTimeStampGm;
    GlobalTensor<int32_t> dbDivisorGm;
    GlobalTensor<uint8_t> dbRemainderGm;
    GlobalTensor<int16_t> extraValAttrGm;
    GlobalTensor<uint8_t> distanceMaskGm;

    TBuf<TPosition::VECCALC> queryTimeStampBuf;
    TBuf<TPosition::VECCALC> queryTokenSetBuf;
    TBuf<TPosition::VECCALC> extraValFilterBuf;
    TBuf<TPosition::VECCALC> dbTimeStampBuf;
    TBuf<TPosition::VECCALC> dbTimeStampFloat32Buf;
    TBuf<TPosition::VECCALC> dbDivisorBuf;
    TBuf<TPosition::VECCALC> dbRemainderBuf;
    TBuf<TPosition::VECCALC> distanceMaskBuf;
    TBuf<TPosition::VECCALC> extraValMaskBuf;
    TBuf<TPosition::VECCALC> timeStampCmpResBuf;
    TBuf<TPosition::VECCALC> tmpRemainderBuf;
    TBuf<TPosition::VECCALC> resRemainderBuf;
    TBuf<TPosition::VECCALC> tokenCmpResBuf;

    LocalTensor<int32_t> queryTimeStampLocal;
    LocalTensor<uint8_t> queryTokenSetLocal;
    LocalTensor<int16_t> extraValFilterLocal;
    LocalTensor<int32_t> dbTimeStampLocal;
    LocalTensor<float32_t> dbTimeStampFloat32Local;
    LocalTensor<int32_t> dbDivisorLocal;
    LocalTensor<uint8_t> dbRemainderLocal;
    LocalTensor<uint8_t> distanceMaskLocal;
    LocalTensor<uint8_t> extraValMaskLocal;
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
    uint32_t maskStride;

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

    __aicore__ inline void InitTiling(const AscendcDistanceBatchValMaskGeneratorTilingData* tilingData)
    {
        this->batchSize = tilingData->batchSize;
        this->tokenCnt = tilingData->tokenCnt;
        this->formerNum = tilingData->formerNum;
        this->tileLen = tilingData->tileLen;
        this->formerRepeatNum = tilingData->formerRepeatNum;
        this->tailRepeatNum = tilingData->tailRepeatNum;
        if (this->coreId < this->formerNum)
        {
            this->repeatNum = this->formerRepeatNum;
        }
        else
        {
            this->repeatNum = this->tailRepeatNum;
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
        SetFlag<HardEvent::MTE2_S>(TIME_EVENT_ID);
        WaitFlag<HardEvent::MTE2_S>(TIME_EVENT_ID);
        const int32_t startTime = this->queryTimeStampLocal.GetValue(bid * 8);
        const int32_t endTime = this->queryTimeStampLocal.GetValue(bid * 8 + 1);
        WaitFlag<HardEvent::MTE2_V>(TIME_EVENT_ID);
        SetFlag<HardEvent::S_V>(TIME_EVENT_ID);
        WaitFlag<HardEvent::S_V>(TIME_EVENT_ID);
        this->timeStampCmpResInt16Local = this->timeStampCmpResLocal.ReinterpretCast<int16_t>();
        if (startTime == 0 && endTime == -INT32_MAX)
        {
            Duplicate(this->timeStampCmpResInt16Local, static_cast<int16_t>(-1), this->distanceMaskLenAlign / 2);
            return;
        }
        if (startTime < endTime)
        {
            Duplicate(this->timeStampCmpResInt16Local, static_cast<int16_t>(0), this->distanceMaskLenAlign / 2);
            return;
        }
        Adds(this->dbTimeStampLocal, this->dbTimeStampLocal, startTime, this->dbTimeStampLenAlign);
        Cast(this->dbTimeStampFloat32Local, this->dbTimeStampLocal, RoundMode::CAST_NONE, this->dbTimeStampLenAlign);
        CompareScalar(this->timeStampCmpResTmpLocal1, this->dbTimeStampFloat32Local, 0.0F, CMPMODE::GE,
                      this->dbTimeStampLenAlign);
        Adds(this->dbTimeStampLocal, this->dbTimeStampLocal, endTime - startTime, this->dbTimeStampLenAlign);
        Cast(this->dbTimeStampFloat32Local, this->dbTimeStampLocal, RoundMode::CAST_NONE, this->dbTimeStampLenAlign);
        CompareScalar(this->timeStampCmpResTmpLocal2, this->dbTimeStampFloat32Local, 0.0F, CMPMODE::LE,
                      this->dbTimeStampLenAlign);
        auto cmp1 = this->timeStampCmpResTmpLocal1.ReinterpretCast<int16_t>();
        auto cmp2 = this->timeStampCmpResTmpLocal2.ReinterpretCast<int16_t>();
        And(this->timeStampCmpResInt16Local, cmp1, cmp2, this->distanceMaskLenAlign / 2);
    }

    __aicore__ inline void CompareTokenId()
    {
        auto queryTokenSetInt16Local = this->queryTokenSetLocal.ReinterpretCast<int16_t>();
        SetFlag<HardEvent::MTE2_V>(TOKEN_EVENT_ID);
        WaitFlag<HardEvent::MTE2_V>(TOKEN_EVENT_ID);
        auto dbDivisorUint32Local = this->dbDivisorLocal.ReinterpretCast<uint32_t>();
        Gather(this->tmpRemainderLocal, queryTokenSetInt16Local, dbDivisorUint32Local, 0, this->dbDivisorLenAlign);
        auto dbRemainderInt16Local = this->dbRemainderLocal.ReinterpretCast<int16_t>();
        And(this->resRemainderLocal, this->tmpRemainderLocal, dbRemainderInt16Local, this->dbDivisorLenAlign);
        auto resRemainderFloat16Local = this->resRemainderLocal.ReinterpretCast<float16_t>();
        // Preserve the legacy token encoding rule: values greater than 2 indicate a token match.
        CompareScalar(this->tokenCmpResLocal, resRemainderFloat16Local, static_cast<float16_t>(TOKEN_MATCH_THRESHOLD),
                      CMPMODE::GT, this->dbDivisorLenAlign);
        this->tokenCmpResInt16Local = this->tokenCmpResLocal.ReinterpretCast<int16_t>();
    }

    __aicore__ inline void CompareExtraVal(uint32_t loop, int16_t filterValue, int16_t matchMode)
    {
        SetFlag<HardEvent::V_MTE2>(FILTER_EVENT_ID);
        WaitFlag<HardEvent::V_MTE2>(FILTER_EVENT_ID);
        DataCopy(this->tmpRemainderLocal, this->extraValAttrGm[loop * this->dbDivisorLenAlign],
                 this->dbDivisorLenAlign);
        SetFlag<HardEvent::MTE2_V>(FILTER_EVENT_ID);
        WaitFlag<HardEvent::MTE2_V>(FILTER_EVENT_ID);
        Duplicate(this->resRemainderLocal, filterValue, this->dbDivisorLenAlign);
        And(this->resRemainderLocal, this->resRemainderLocal, this->tmpRemainderLocal, this->dbDivisorLenAlign);
        if (matchMode == 0)
        {
            Duplicate(this->tmpRemainderLocal, filterValue, this->dbDivisorLenAlign);
            Sub(this->resRemainderLocal, this->resRemainderLocal, this->tmpRemainderLocal, this->dbDivisorLenAlign);
        }
        auto matchedValueLocal = this->dbTimeStampFloat32Local.ReinterpretCast<float16_t>();
        Cast(matchedValueLocal, this->resRemainderLocal, RoundMode::CAST_NONE, this->dbDivisorLenAlign);
        CompareScalar(this->extraValMaskLocal, matchedValueLocal, static_cast<float16_t>(0.0),
                      matchMode == 0 ? CMPMODE::EQ : CMPMODE::GT, this->dbDivisorLenAlign);
    }
};
}  // namespace IndexOps

extern "C" __global__ __aicore__ void ascendc_distance_batch_val_mask_generator(
    GM_ADDR queryTimeStamp, GM_ADDR queryTokenSet, GM_ADDR dbTimeStamp, GM_ADDR dbDivisor, GM_ADDR dbRemainder,
    GM_ADDR extraValFilter, GM_ADDR extraValAttr, GM_ADDR distanceMask, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    IndexOps::AscendcDistanceBatchValMaskGenerator op;
    op.Init(queryTimeStamp, queryTokenSet, dbTimeStamp, dbDivisor, dbRemainder, extraValFilter, extraValAttr,
            distanceMask, &tilingData, &pipe);
    op.Process();
}
