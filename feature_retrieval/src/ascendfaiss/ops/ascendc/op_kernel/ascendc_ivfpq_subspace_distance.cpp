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

#define ASCENDC_CUBE_ONLY

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

namespace
{
constexpr uint32_t CONST_ONE = 1;
constexpr uint32_t CONST_TWO = 2;
constexpr uint32_t FLOAT_DATA_BLOCK_SIZE = 8;
constexpr uint16_t IP_READY_ID = 1;
constexpr float SCALAR = -2.0;
constexpr uint8_t SYNC_MODE2 = 2;
constexpr uint32_t FLAG_ALIGN = 32;

__aicore__ inline uint32_t AlignUpFloatBlock(uint32_t value)
{
    return (value + FLOAT_DATA_BLOCK_SIZE - 1) / FLOAT_DATA_BLOCK_SIZE * FLOAT_DATA_BLOCK_SIZE;
}
}  // namespace

namespace IndexOps
{

class AscendcIvfpqSubspaceDistance
{
   public:
    __aicore__ inline AscendcIvfpqSubspaceDistance(){};
    __aicore__ inline ~AscendcIvfpqSubspaceDistance(){};
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR codeBook, GM_ADDR distance, GM_ADDR workspace,
                                const AscendcIvfpqSubspaceDistanceTilingData *__restrict tilingData, TPipe *tPipe);
    __aicore__ inline void InitPad(GM_ADDR query, GM_ADDR codeBook, GM_ADDR distance,
                                   const AscendcIvfpqSubspaceDistanceTilingData *__restrict tilingData, TPipe *tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessPad();

   private:
    __aicore__ inline void ParseTilingData(const AscendcIvfpqSubspaceDistanceTilingData *__restrict tilingData);
    __aicore__ inline void ComputeQueryL2Norm(uint32_t subSpaceIdx);
    __aicore__ inline void ComputeQueryL2NormPad(uint32_t subSpaceIdx);
    __aicore__ inline void ComputeDistance(uint32_t taskIdx, uint32_t subSpaceIdx, uint32_t baseBlockIdx);
    __aicore__ inline void ComputeDistancePad(uint32_t taskIdx, uint32_t subSpaceIdx, uint32_t baseBlockIdx);
    __aicore__ inline void ComputeCodeBookL2Norm(uint32_t codeBookOffset);
    __aicore__ inline void ComputeCodeBookL2NormPad(uint32_t codeBookOffset);
    __aicore__ inline void ComputeInnerProductPad();
    __aicore__ inline void ComputeIpMuls(uint32_t ipOffset);
    __aicore__ inline void AddDistance();
    __aicore__ inline void CopyOutDistance(uint32_t nBlockOffset, uint32_t subSpaceIdx);
    __aicore__ inline void ComputeL2Norm(LocalTensor<float> tensor, uint32_t nums);

   private:
    TPipe pipe;

    GlobalTensor<float> queryGm;
    GlobalTensor<float> codeBookGm;
    GlobalTensor<float> distanceGm;
    GlobalTensor<float> ipGm;

    LocalTensor<float> codeBookNormBrcbUb;  // UB上NBlockTile条codeBook广播后的L2Norm数据[batch, nBlockTile/2]
    LocalTensor<float> queryNormBrcbUb;     // UB上batch条query广播后的L2Norm数据[batch, nBlockTile/2]
    LocalTensor<float> ipUb;                // UB上每个vector核的内积结果[batch, nBlockTile/2]
    LocalTensor<float> distanceEnqueUb;     // UB上每个vector核计算的的L2距离结果[batch, nBlockTile/2]
    LocalTensor<float> queryPadUb;          // pad 路径：query [batch, dSubPad]
    LocalTensor<float> codeBookPadUb;       // pad 路径：codebook [nBlockTilePerVec, dSubPad]
    LocalTensor<float> ipMulTmpUb;          // pad 路径：逐元素乘临时缓冲

    // 输入query的队列
    TQue<AscendC::TPosition::VECIN, CONST_ONE> queryQueue;
    TQue<AscendC::TPosition::VECIN, CONST_ONE> codeBookQueue;
    TQue<AscendC::TPosition::VECIN, CONST_ONE> ipQueue;
    TQue<AscendC::TPosition::VECOUT, CONST_ONE> distResultQueue;

    TBuf<AscendC::TPosition::VECCALC> queryNormBuf;
    TBuf<AscendC::TPosition::VECCALC> queryNormBrcbBuf;

    TBuf<AscendC::TPosition::VECCALC> codeBookNormBuf;
    TBuf<AscendC::TPosition::VECCALC> codeBookNormBrcbBuf;
    TBuf<AscendC::TPosition::VECCALC> queryPadBuf;
    TBuf<AscendC::TPosition::VECCALC> codeBookPadBuf;
    TBuf<AscendC::TPosition::VECCALC> ipMulTmpBuf;
    TBuf<AscendC::TPosition::VECCALC> ipResultBuf;

    int64_t blockIdx = 0;
    uint32_t batch = 0;
    uint32_t dim = 0;
    uint32_t subSpaceNum = 0;
    uint32_t kSub = 0;
    uint32_t dSub = 0;
    uint32_t dSubPad = 0;
    uint32_t minValue = 0;
    uint32_t codeBookMinValue = 0;

    uint32_t nBlockTile = 0;
    uint32_t nBlockTilePerVec = 0;
    uint32_t nBlockNum = 0;
    uint32_t totalTaskNum = 0;
    uint32_t usedCoreNum = 0;
};

__aicore__ inline void AscendcIvfpqSubspaceDistance::Init(
    GM_ADDR query, GM_ADDR codeBook, GM_ADDR distance, GM_ADDR workspace,
    const AscendcIvfpqSubspaceDistanceTilingData *__restrict tilingData, TPipe *tPipe)
{
    this->blockIdx = GetBlockIdx();
    this->pipe = *tPipe;
    ParseTilingData(tilingData);
    this->nBlockTilePerVec = this->nBlockTile / CONST_TWO;
    this->dSubPad = AlignUpFloatBlock(this->dSub);

    this->queryGm.SetGlobalBuffer((__gm__ float *)query);
    this->codeBookGm.SetGlobalBuffer((__gm__ float *)codeBook);
    this->distanceGm.SetGlobalBuffer((__gm__ float *)distance);

    pipe.InitBuffer(queryQueue, CONST_ONE, this->batch * this->dSub * sizeof(float));
    pipe.InitBuffer(codeBookQueue, CONST_ONE, this->nBlockTilePerVec * this->dSub * sizeof(float));
    pipe.InitBuffer(ipQueue, CONST_ONE, this->batch * this->nBlockTilePerVec * sizeof(float));
    pipe.InitBuffer(distResultQueue, CONST_ONE, this->batch * this->nBlockTilePerVec * sizeof(float));

    pipe.InitBuffer(queryNormBuf, this->batch * sizeof(float));
    pipe.InitBuffer(queryNormBrcbBuf, this->batch * this->nBlockTilePerVec * sizeof(float));

    pipe.InitBuffer(codeBookNormBuf, this->nBlockTilePerVec * sizeof(float));

    queryNormBrcbUb = queryNormBrcbBuf.Get<float>();
    ipGm.SetGlobalBuffer((__gm__ float *)workspace);
}

__aicore__ inline void AscendcIvfpqSubspaceDistance::InitPad(
    GM_ADDR query, GM_ADDR codeBook, GM_ADDR distance,
    const AscendcIvfpqSubspaceDistanceTilingData *__restrict tilingData, TPipe *tPipe)
{
    this->blockIdx = GetBlockIdx();
    this->pipe = *tPipe;
    ParseTilingData(tilingData);
    this->nBlockTilePerVec = this->nBlockTile / CONST_TWO;
    this->dSubPad = AlignUpFloatBlock(this->dSub);

    this->queryGm.SetGlobalBuffer((__gm__ float *)query);
    this->codeBookGm.SetGlobalBuffer((__gm__ float *)codeBook);
    this->distanceGm.SetGlobalBuffer((__gm__ float *)distance);

    pipe.InitBuffer(distResultQueue, CONST_ONE, this->batch * this->nBlockTilePerVec * sizeof(float));
    pipe.InitBuffer(queryNormBuf, AlignUpFloatBlock(this->batch) * sizeof(float));
    pipe.InitBuffer(queryNormBrcbBuf, this->batch * this->nBlockTilePerVec * sizeof(float));
    pipe.InitBuffer(codeBookNormBuf, AlignUpFloatBlock(this->nBlockTilePerVec) * sizeof(float));
    pipe.InitBuffer(queryPadBuf, this->batch * this->dSubPad * sizeof(float));
    pipe.InitBuffer(codeBookPadBuf, this->nBlockTilePerVec * this->dSubPad * sizeof(float));
    // shared tmp for query/codebook sq-norms and (optionally) IP expand; need max row count
    uint32_t ipMulRows = this->batch > this->nBlockTilePerVec ? this->batch : this->nBlockTilePerVec;
    pipe.InitBuffer(ipMulTmpBuf, ipMulRows * this->dSubPad * sizeof(float));
    pipe.InitBuffer(ipResultBuf, this->batch * this->nBlockTilePerVec * sizeof(float));

    queryNormBrcbUb = queryNormBrcbBuf.Get<float>();
    queryPadUb = queryPadBuf.Get<float>();
    codeBookPadUb = codeBookPadBuf.Get<float>();
    ipMulTmpUb = ipMulTmpBuf.Get<float>();
    ipUb = ipResultBuf.Get<float>();
}

__aicore__ inline void AscendcIvfpqSubspaceDistance::ParseTilingData(
    const AscendcIvfpqSubspaceDistanceTilingData *__restrict tilingData)
{
    this->subSpaceNum = tilingData->subSpaceNum;
    this->batch = tilingData->batch;
    this->dim = tilingData->dim;
    this->kSub = tilingData->kSub;
    this->dSub = tilingData->dSub;
    this->minValue = tilingData->minValue;
    this->nBlockTile = tilingData->nBlockTile;
    this->nBlockNum = tilingData->nBlockNum;
    this->totalTaskNum = tilingData->totalTaskNum;
    this->usedCoreNum = tilingData->usedCoreNum;
    this->codeBookMinValue = tilingData->codeBookMinValue;
}

__aicore__ inline void AscendcIvfpqSubspaceDistance::Process()
{
    for (size_t taskIdx = blockIdx; taskIdx < totalTaskNum * CONST_TWO; taskIdx += usedCoreNum * CONST_TWO)
    {
        uint32_t curAicIdx = taskIdx / CONST_TWO;
        uint32_t subSpaceIdx = curAicIdx / nBlockNum;  // 当前AICORE处理的子空间idx
        uint32_t nBlockIdx = curAicIdx % nBlockNum;    // 当前AICORE处理的码本基本块idx
        // Vector: 将nBlockTile条codeBook拆成两份, 每个Vector负责处理 nBlockTile/2 条码本数据的计算
        ComputeDistance(taskIdx, subSpaceIdx, nBlockIdx);
    }
    AscendC::SyncAll();
}

__aicore__ inline void AscendcIvfpqSubspaceDistance::ProcessPad()
{
    for (size_t taskIdx = blockIdx; taskIdx < totalTaskNum * CONST_TWO; taskIdx += usedCoreNum * CONST_TWO)
    {
        uint32_t curAicIdx = taskIdx / CONST_TWO;
        uint32_t subSpaceIdx = curAicIdx / nBlockNum;
        uint32_t nBlockIdx = curAicIdx % nBlockNum;
        ComputeDistancePad(taskIdx, subSpaceIdx, nBlockIdx);
    }
    AscendC::SyncAll();
}

__aicore__ inline void AscendcIvfpqSubspaceDistance::ComputeDistance(uint32_t taskIdx, uint32_t subSpaceIdx,
                                                                     uint32_t baseBlockIdx)
{
    // AI Core上Vector核的ID
    uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
    uint32_t codeBookOffset = subSpaceIdx * kSub * dSub + baseBlockIdx * nBlockTile * dSub;
    uint32_t curAicIdx = taskIdx / CONST_TWO;
    uint32_t ipOffset = curAicIdx * batch * nBlockTile;
    uint32_t nBlockOffset = baseBlockIdx * nBlockTile;
    if (subBlockIdx == CONST_ONE)
    {
        codeBookOffset += nBlockTilePerVec * dSub;
        ipOffset += nBlockTilePerVec;
        nBlockOffset += nBlockTilePerVec;
    }

    // step1: 将当前子空间的batch条query搬入UB, 并计算L2Norm, 且broadcast
    ComputeQueryL2Norm(subSpaceIdx);  // -> queryNormBrcbUb[batch, nBlockTilePerVec]
    AscendC::PipeBarrier<PIPE_V>();

    // step2: 将当前子空间的NBlockTile/2 条码本搬入UB, 并计算L2Norm, 且broadcast
    ComputeCodeBookL2Norm(codeBookOffset);  // -> codeBookNormBrcbUb[batch, nBlockTilePerVec]
    AscendC::PipeBarrier<PIPE_V>();

    // step3: 等待cube计算完ipGm [batch, nBlockTile]
    AscendC::CrossCoreWaitFlag(IP_READY_ID);

    // step4: 将当前vec负责的一半内积结果搬入UB, 并逐元素乘-2
    ComputeIpMuls(ipOffset);  // -> ipUb[batch, nBlockTilePerVec]
    AscendC::PipeBarrier<PIPE_V>();

    // step5: 计算Add: codeBookNormBrcbUb + queryNormBrcbUb + ipUb
    AddDistance();  // -> distanceUb[batch, nBlockTilePerVec]

    // step6: 将计算出来的 distance[batch, nBlockTile]的结果, 按照目标格式, 搬出到GM上
    CopyOutDistance(nBlockOffset, subSpaceIdx);  // -> distanceGm
}

__aicore__ inline void AscendcIvfpqSubspaceDistance::ComputeDistancePad(uint32_t taskIdx, uint32_t subSpaceIdx,
                                                                        uint32_t baseBlockIdx)
{
    (void)taskIdx;
    uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
    uint32_t codeBookOffset = subSpaceIdx * kSub * dSub + baseBlockIdx * nBlockTile * dSub;
    uint32_t nBlockOffset = baseBlockIdx * nBlockTile;
    if (subBlockIdx == CONST_ONE)
    {
        codeBookOffset += nBlockTilePerVec * dSub;
        nBlockOffset += nBlockTilePerVec;
    }

    ComputeQueryL2NormPad(subSpaceIdx);
    AscendC::PipeBarrier<PIPE_V>();

    ComputeCodeBookL2NormPad(codeBookOffset);
    AscendC::PipeBarrier<PIPE_V>();

    ComputeInnerProductPad();
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Muls(ipUb, ipUb, SCALAR, this->batch * nBlockTilePerVec);
    AscendC::PipeBarrier<PIPE_V>();

    AddDistance();
    CopyOutDistance(nBlockOffset, subSpaceIdx);
}

__aicore__ inline void AscendcIvfpqSubspaceDistance::ComputeQueryL2Norm(uint32_t subSpaceIdx)
{
    // copyin
    LocalTensor<float> queryEnqueUb = queryQueue.AllocTensor<float>();
    uint16_t blockCount = static_cast<uint16_t>(this->batch);
    uint16_t blockLen = static_cast<uint16_t>(this->dSub / FLOAT_DATA_BLOCK_SIZE);
    uint16_t srcGap = static_cast<uint16_t>((this->dim - this->dSub) / FLOAT_DATA_BLOCK_SIZE);
    uint16_t dstGap = 0;
    DataCopyParams repeatParams{blockCount, blockLen, srcGap, dstGap};
    AscendC::DataCopy(queryEnqueUb, queryGm[subSpaceIdx * this->dSub], repeatParams);
    AscendC::PipeBarrier<PIPE_ALL>();
    queryQueue.EnQue(queryEnqueUb);

    // 计算L2Norm
    LocalTensor<float> queryDequeUb = queryQueue.DeQue<float>();

    // Mul
    AscendC::Mul(queryDequeUb, queryDequeUb, queryDequeUb, this->batch * this->dSub);
    AscendC::PipeBarrier<PIPE_V>();

    // ReduceSum
    AscendC::LocalTensor<float> reduceSumResult = queryNormBuf.Get<float>();
    AscendC::WholeReduceSum<float>(reduceSumResult,                      // dst
                                   queryDequeUb,                         // src
                                   this->dSub,                           // mask
                                   this->batch,                          // repeatTime
                                   1,                                    // dstRepStride
                                   1,                                    // srcBlkStride
                                   this->dSub / FLOAT_DATA_BLOCK_SIZE);  // srcRepStride
    AscendC::PipeBarrier<PIPE_V>();

    // Broadcast
    const uint32_t dstShape_[] = {this->batch, nBlockTilePerVec};
    const uint32_t srcShape_[] = {this->batch, CONST_ONE};
    AscendC::Broadcast<float, CONST_TWO, CONST_ONE>(queryNormBrcbUb, reduceSumResult, dstShape_, srcShape_);
    AscendC::PipeBarrier<PIPE_V>();
    queryQueue.FreeTensor(queryDequeUb);
}

__aicore__ inline void AscendcIvfpqSubspaceDistance::ComputeQueryL2NormPad(uint32_t subSpaceIdx)
{
    AscendC::Duplicate(queryPadUb, 0.0f, this->batch * this->dSubPad);
    AscendC::PipeBarrier<PIPE_V>();

    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(this->batch);
    copyParams.blockLen = static_cast<uint32_t>(this->dSub * sizeof(float));
    copyParams.srcStride = static_cast<uint32_t>((this->dim - this->dSub) * sizeof(float));
    copyParams.dstStride = 0;
    DataCopyPadExtParams<float> padParams{true, 0, static_cast<uint8_t>(this->dSubPad - this->dSub), 0.0f};
    AscendC::DataCopyPad(queryPadUb, queryGm[subSpaceIdx * this->dSub], copyParams, padParams);
    AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::Mul(ipMulTmpUb, queryPadUb, queryPadUb, this->batch * this->dSubPad);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::LocalTensor<float> reduceSumResult = queryNormBuf.Get<float>();
    AscendC::WholeReduceSum<float>(reduceSumResult, ipMulTmpUb, this->dSubPad, this->batch, 1, 1,
                                   this->dSubPad / FLOAT_DATA_BLOCK_SIZE);
    AscendC::PipeBarrier<PIPE_V>();

    const uint32_t dstShape_[] = {this->batch, nBlockTilePerVec};
    const uint32_t srcShape_[] = {this->batch, CONST_ONE};
    AscendC::Broadcast<float, CONST_TWO, CONST_ONE>(queryNormBrcbUb, reduceSumResult, dstShape_, srcShape_);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void AscendcIvfpqSubspaceDistance::ComputeCodeBookL2Norm(uint32_t codeBookOffset)
{
    // copyin
    LocalTensor<float> codeBookEnqueUb = codeBookQueue.AllocTensor<float>();
    AscendC::DataCopy(codeBookEnqueUb, codeBookGm[codeBookOffset], nBlockTilePerVec * dSub);
    AscendC::PipeBarrier<PIPE_ALL>();
    codeBookQueue.EnQue(codeBookEnqueUb);

    // 计算L2Norm
    LocalTensor<float> codeBookDequeUb = codeBookQueue.DeQue<float>();

    // Mul
    AscendC::Mul(codeBookDequeUb, codeBookDequeUb, codeBookDequeUb, nBlockTilePerVec * dSub);
    AscendC::PipeBarrier<PIPE_V>();

    // ReduceSum
    AscendC::LocalTensor<float> codeBookNormResult = codeBookNormBuf.Get<float>();
    AscendC::WholeReduceSum<float>(codeBookNormResult,                   // dst
                                   codeBookDequeUb,                      // src
                                   this->dSub,                           // mask
                                   this->nBlockTilePerVec,               // repeatTime
                                   1,                                    // dstRepStride
                                   1,                                    // srcBlkStride
                                   this->dSub / FLOAT_DATA_BLOCK_SIZE);  // srcRepStride
    AscendC::PipeBarrier<PIPE_V>();

    // Broadcast
    const uint32_t dstShape_[] = {this->batch, nBlockTilePerVec};
    const uint32_t srcShape_[] = {CONST_ONE, nBlockTilePerVec};
    distanceEnqueUb = distResultQueue.AllocTensor<float>();
    AscendC::Broadcast<float, CONST_TWO, 0>(distanceEnqueUb, codeBookNormResult, dstShape_, srcShape_);
    AscendC::PipeBarrier<PIPE_V>();
    codeBookQueue.FreeTensor(codeBookDequeUb);
}

__aicore__ inline void AscendcIvfpqSubspaceDistance::ComputeCodeBookL2NormPad(uint32_t codeBookOffset)
{
    AscendC::Duplicate(codeBookPadUb, 0.0f, nBlockTilePerVec * this->dSubPad);
    AscendC::PipeBarrier<PIPE_V>();

    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint16_t>(nBlockTilePerVec);
    copyParams.blockLen = static_cast<uint32_t>(this->dSub * sizeof(float));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<float> padParams{true, 0, static_cast<uint8_t>(this->dSubPad - this->dSub), 0.0f};
    AscendC::DataCopyPad(codeBookPadUb, codeBookGm[codeBookOffset], copyParams, padParams);
    AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::LocalTensor<float> codeBookNormResult = codeBookNormBuf.Get<float>();
    AscendC::Mul(ipMulTmpUb, codeBookPadUb, codeBookPadUb, nBlockTilePerVec * this->dSubPad);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum<float>(codeBookNormResult, ipMulTmpUb, this->dSubPad, this->nBlockTilePerVec, 1, 1,
                                   this->dSubPad / FLOAT_DATA_BLOCK_SIZE);
    AscendC::PipeBarrier<PIPE_V>();

    const uint32_t dstShape_[] = {this->batch, nBlockTilePerVec};
    const uint32_t srcShape_[] = {CONST_ONE, nBlockTilePerVec};
    distanceEnqueUb = distResultQueue.AllocTensor<float>();
    AscendC::Broadcast<float, CONST_TWO, 0>(distanceEnqueUb, codeBookNormResult, dstShape_, srcShape_);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void AscendcIvfpqSubspaceDistance::ComputeInnerProductPad()
{
    AscendC::Duplicate(ipUb, 0.0f, this->batch * nBlockTilePerVec);
    AscendC::PipeBarrier<PIPE_ALL>();
    for (uint32_t b = 0; b < this->batch; b++)
    {
        for (uint32_t c = 0; c < nBlockTilePerVec; c++)
        {
            float sum = 0.0f;
            for (uint32_t j = 0; j < this->dSub; j++)
            {
                sum += queryPadUb.GetValue(b * this->dSubPad + j) * codeBookPadUb.GetValue(c * this->dSubPad + j);
            }
            ipUb.SetValue(b * nBlockTilePerVec + c, sum);
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void AscendcIvfpqSubspaceDistance::ComputeIpMuls(uint32_t ipOffset)
{
    // copyin
    LocalTensor<float> ipEnqueUb = ipQueue.AllocTensor<float>();
    AscendC::DataCopyParams copyParams{
        static_cast<uint16_t>(this->batch), static_cast<uint16_t>(nBlockTilePerVec / FLOAT_DATA_BLOCK_SIZE),
        static_cast<uint16_t>((this->nBlockTile - nBlockTilePerVec) / FLOAT_DATA_BLOCK_SIZE), 0};
    AscendC::DataCopy(ipEnqueUb, ipGm[ipOffset], copyParams);
    AscendC::PipeBarrier<PIPE_ALL>();
    ipQueue.EnQue(ipEnqueUb);
    // muls -2
    ipUb = ipQueue.DeQue<float>();
    AscendC::Muls(ipUb, ipUb, SCALAR, this->batch * nBlockTilePerVec);
    AscendC::PipeBarrier<PIPE_V>();
    ipQueue.FreeTensor(ipEnqueUb);
}

__aicore__ inline void AscendcIvfpqSubspaceDistance::AddDistance()
{
    AscendC::Add(distanceEnqueUb, distanceEnqueUb, queryNormBrcbUb, nBlockTilePerVec * this->batch);
    AscendC::PipeBarrier<PIPE_V>();
    // + ipUb
    AscendC::Add(distanceEnqueUb, distanceEnqueUb, ipUb, nBlockTilePerVec * this->batch);
    AscendC::PipeBarrier<PIPE_V>();
    distResultQueue.EnQue(distanceEnqueUb);
}

__aicore__ inline void AscendcIvfpqSubspaceDistance::CopyOutDistance(uint32_t nBlockOffset, uint32_t subSpaceIdx)
{
    uint32_t distanceOffset = nBlockOffset + subSpaceIdx * this->kSub;
    LocalTensor<float> distanceDequeUb = distResultQueue.DeQue<float>();
    AscendC::DataCopyParams copyParams{
        static_cast<uint16_t>(this->batch),                               // blockCount
        static_cast<uint16_t>(nBlockTilePerVec / FLOAT_DATA_BLOCK_SIZE),  // blockLen
        0,                                                                // srcGap
        static_cast<uint16_t>((this->subSpaceNum * this->kSub - nBlockTilePerVec) / FLOAT_DATA_BLOCK_SIZE)};  // dstGap
    AscendC::DataCopy(distanceGm[distanceOffset], distanceDequeUb, copyParams);
    AscendC::PipeBarrier<PIPE_ALL>();
    distResultQueue.FreeTensor(distanceDequeUb);
}

class IvfpqSubspaceDistanceMatmul
{
   public:
    __aicore__ inline IvfpqSubspaceDistanceMatmul(){};
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR codeBook, GM_ADDR distance, GM_ADDR workspace,
                                const AscendcIvfpqSubspaceDistanceTilingData *__restrict tilingData,
                                const TCubeTiling &tiling, TPipe *tPipe);
    __aicore__ inline void Process(AscendC::TPipe *pipe);
    __aicore__ inline void ParseTilingData(const AscendcIvfpqSubspaceDistanceTilingData *__restrict tilingData);
    __aicore__ inline void ComputeInnerProduct(uint32_t taskIdx, uint32_t subSpaceIdx, uint32_t nBlockIdx);

    using AType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using BType = MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
    using CType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    Matmul<AType, BType, CType> matmulObj;

    TPipe pipe;

    GlobalTensor<float> queryGm;
    GlobalTensor<float> codeBookGm;
    GlobalTensor<float> distanceGm;
    GlobalTensor<float> mmResGm;

    TCubeTiling cubeTiling;

    int64_t blockIdx = 0;
    uint32_t batch = 0;
    uint32_t dim = 0;
    uint32_t subSpaceNum = 0;
    uint32_t kSub = 0;
    uint32_t dSub = 0;
    uint32_t minValue = 0;

    uint32_t nBlockTile = 0;
    uint32_t nBlockNum = 0;
    uint32_t totalTaskNum = 0;
    uint32_t usedCoreNum = 0;
};

__aicore__ inline void IvfpqSubspaceDistanceMatmul::Init(
    GM_ADDR query, GM_ADDR codeBook, GM_ADDR distance, GM_ADDR workspace,
    const AscendcIvfpqSubspaceDistanceTilingData *__restrict tilingData, const TCubeTiling &tiling, TPipe *tPipe)
{
    this->blockIdx = GetBlockIdx();
    this->pipe = *tPipe;
    this->cubeTiling = tiling;
    ParseTilingData(tilingData);

    this->queryGm.SetGlobalBuffer((__gm__ float *)query);
    this->codeBookGm.SetGlobalBuffer((__gm__ float *)codeBook);
    this->distanceGm.SetGlobalBuffer((__gm__ float *)distance);
    this->mmResGm.SetGlobalBuffer((__gm__ float *)workspace);

    matmulObj.SetOrgShape(this->batch, this->nBlockTile, this->dim, this->dSub);
}

__aicore__ inline void IvfpqSubspaceDistanceMatmul::ParseTilingData(
    const AscendcIvfpqSubspaceDistanceTilingData *__restrict tilingData)
{
    this->subSpaceNum = tilingData->subSpaceNum;
    this->batch = tilingData->batch;
    this->dim = tilingData->dim;
    this->kSub = tilingData->kSub;
    this->dSub = tilingData->dSub;
    this->minValue = tilingData->minValue;
    this->nBlockTile = tilingData->nBlockTile;
    this->nBlockNum = tilingData->nBlockNum;
    this->totalTaskNum = tilingData->totalTaskNum;
    this->usedCoreNum = tilingData->usedCoreNum;
}

__aicore__ inline void IvfpqSubspaceDistanceMatmul::Process(AscendC::TPipe *pipe)
{
    for (size_t taskIdx = blockIdx; taskIdx < totalTaskNum; taskIdx += usedCoreNum)
    {
        uint32_t subSpaceIdx = taskIdx / nBlockNum;  // 当前AICORE处理的子空间idx
        uint32_t nBlockIdx = taskIdx % nBlockNum;    // 当前AICORE处理的码本基本块idx
        ComputeInnerProduct(taskIdx, subSpaceIdx, nBlockIdx);
        AscendC::CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(IP_READY_ID);
    }
    matmulObj.End();
}

__aicore__ inline void IvfpqSubspaceDistanceMatmul::ComputeInnerProduct(uint32_t taskIdx, uint32_t subSpaceIdx,
                                                                        uint32_t nBlockIdx)
{
    uint32_t queryGmOffset = subSpaceIdx * this->dSub;
    uint32_t codeBookGmOffset = taskIdx * this->nBlockTile * this->dSub;

    matmulObj.SetSingleShape(this->batch, this->nBlockTile, this->dSub);
    matmulObj.SetTensorA(queryGm[queryGmOffset]);
    matmulObj.SetTensorB(codeBookGm[codeBookGmOffset], true);
    matmulObj.template IterateAll(mmResGm[taskIdx * this->batch * this->nBlockTile]);
}

}  // namespace IndexOps

extern "C" __global__ __aicore__ void ascendc_ivfpq_subspace_distance(GM_ADDR query, GM_ADDR codeBook, GM_ADDR distance,
                                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe tPipe;
    GET_TILING_DATA(tiling_data, tiling);
    TCubeTiling tCubeTiling;

    if (TILING_KEY_IS(0))
    {
        const bool usePadPath = (tiling_data.dSub < FLOAT_DATA_BLOCK_SIZE);
        if ASCEND_IS_AIC
        {
            if (!usePadPath)
            {
                tCubeTiling = tiling_data.cubeTiling;
                IndexOps::IvfpqSubspaceDistanceMatmul matmulOp;
                matmulOp.Init(query, codeBook, distance, workspace, &tiling_data, tCubeTiling, &tPipe);
                REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), matmulOp.matmulObj, &matmulOp.cubeTiling);
                matmulOp.Process(&tPipe);
            }
            else
            {
                AscendC::SyncAll();
            }
        }
        if ASCEND_IS_AIV
        {
            IndexOps::AscendcIvfpqSubspaceDistance op;
            if (usePadPath)
            {
                op.InitPad(query, codeBook, distance, &tiling_data, &tPipe);
                op.ProcessPad();
            }
            else
            {
                op.Init(query, codeBook, distance, workspace, &tiling_data, &tPipe);
                op.Process();
            }
        }
    }
}
