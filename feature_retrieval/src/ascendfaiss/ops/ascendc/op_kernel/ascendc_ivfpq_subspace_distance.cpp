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

namespace {
constexpr uint32_t CONST_ONE = 1;
constexpr uint32_t CONST_TWO = 2;
constexpr uint32_t FLOAT_DATA_BLOCK_SIZE = 8;
constexpr uint16_t IP_READY_ID = 1;
constexpr float SCALAR = -2.0;
constexpr uint8_t SYNC_MODE2 = 2;
constexpr uint32_t MIN_DIST_RESULT_BASE_SIZE = 2;
constexpr uint32_t REDUCE_BASE_SIZE = 64;
constexpr uint32_t FLAG_ALIGN = 32;
}  // namespace

namespace IndexOps {

class AscendcIvfpqSubspaceDistance {
public:
    __aicore__ inline AscendcIvfpqSubspaceDistance(){};
    __aicore__ inline ~AscendcIvfpqSubspaceDistance(){};
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR codeBook, GM_ADDR distance, GM_ADDR minDistResult, GM_ADDR flag,
                                GM_ADDR workspace,
                                const AscendcIvfpqSubspaceDistanceTilingData *__restrict tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const AscendcIvfpqSubspaceDistanceTilingData *__restrict tilingData);
    __aicore__ inline void ComputeQueryL2Norm(uint32_t subSpaceIdx);
    __aicore__ inline void ComputeDistance(uint32_t subSpaceIdx, uint32_t baseBlockIdx);
    __aicore__ inline void ComputeCodeBookL2Norm(uint32_t codeBookOffset);
    __aicore__ inline void ComputeIpMuls(uint32_t ipOffset);
    __aicore__ inline void AddDistance();
    __aicore__ inline void CopyOutDistance(uint32_t nBlockOffset, uint32_t subSpaceIdx, uint32_t minDistBlockOffset);
    __aicore__ inline void ComputeL2Norm(LocalTensor<float> tensor, uint32_t nums);
    __aicore__ inline void ComputeFlag();

private:
    TPipe pipe;

    GlobalTensor<float> queryGm;
    GlobalTensor<float> codeBookGm;
    GlobalTensor<float> distanceGm;
    GlobalTensor<float> minDistResultGm;
    GlobalTensor<uint16_t> flagGm;
    GlobalTensor<float> ipGm;

    LocalTensor<float> codeBookNormBrcbUb;  // UB上NBlockTile条codeBook广播后的L2Norm数据[batch, nBlockTile/2]
    LocalTensor<float> queryNormBrcbUb;     // UB上batch条query广播后的L2Norm数据[batch, nBlockTile/2]
    LocalTensor<float> ipUb;                // UB上每个vector核的内积结果[batch, nBlockTile/2]
    LocalTensor<float> distanceEnqueUb;     // UB上每个vector核计算的的L2距离结果[batch, nBlockTile/2]

    // 输入query的队列
    TQue<AscendC::TPosition::VECIN, CONST_ONE> queryQueue;
    TQue<AscendC::TPosition::VECIN, CONST_ONE> codeBookQueue;
    TQue<AscendC::TPosition::VECIN, CONST_ONE> ipQueue;
    TQue<AscendC::TPosition::VECOUT, CONST_ONE> distResultQueue;
    TQue<AscendC::TPosition::VECOUT, CONST_ONE> minDistResultQueue;

    TBuf<AscendC::TPosition::VECCALC> queryNormTmpBuf;
    TBuf<AscendC::TPosition::VECCALC> queryNormBuf;
    TBuf<AscendC::TPosition::VECCALC> queryNormBrcbBuf;

    TBuf<AscendC::TPosition::VECCALC> codeBookNormTmpBuf;
    TBuf<AscendC::TPosition::VECCALC> codeBookNormBuf;
    TBuf<AscendC::TPosition::VECCALC> codeBookNormBrcbBuf;

    int64_t blockIdx = 0;
    uint32_t batch = 0;
    uint32_t dim = 0;
    uint32_t subSpaceNum = 0;
    uint32_t kSub = 0;
    uint32_t dSub = 0;
    uint32_t minValue = 0;
    uint32_t codeBookMinValue = 0;

    uint32_t nBlockTile = 0;
    uint32_t nBlockTilePerVec = 0;
    uint32_t nBlockNum = 0;
    uint32_t totalTaskNum = 0;
    uint32_t usedCoreNum = 0;
    uint32_t curReduceRepeatTimePerBatch = 0;
};

__aicore__ inline void AscendcIvfpqSubspaceDistance::Init(GM_ADDR query, GM_ADDR codeBook, GM_ADDR distance,
    GM_ADDR minDistResult, GM_ADDR flag, GM_ADDR workspace,
    const AscendcIvfpqSubspaceDistanceTilingData *__restrict tilingData, TPipe *tPipe)
{
    this->blockIdx = GetBlockIdx();
    this->pipe = *tPipe;
    ParseTilingData(tilingData);
    this->nBlockTilePerVec = this->nBlockTile / CONST_TWO;

    this->queryGm.SetGlobalBuffer((__gm__ float *)query);
    this->codeBookGm.SetGlobalBuffer((__gm__ float *)codeBook);
    this->distanceGm.SetGlobalBuffer((__gm__ float *)distance);
    this->minDistResultGm.SetGlobalBuffer((__gm__ float *)minDistResult);
    this->flagGm.SetGlobalBuffer((__gm__ uint16_t *)flag);

    pipe.InitBuffer(queryQueue, CONST_ONE, this->batch * this->dSub * sizeof(float));
    pipe.InitBuffer(codeBookQueue, CONST_ONE, this->nBlockTilePerVec * this->dSub * sizeof(float));
    pipe.InitBuffer(ipQueue, CONST_ONE, this->batch * this->nBlockTilePerVec * sizeof(float));
    pipe.InitBuffer(distResultQueue, CONST_ONE, this->batch * this->nBlockTilePerVec * sizeof(float));
    pipe.InitBuffer(minDistResultQueue, CONST_ONE,
                    this->batch * this->nBlockTilePerVec /
                        (REDUCE_BASE_SIZE / FLOAT_DATA_BLOCK_SIZE) * sizeof(float));
    pipe.InitBuffer(queryNormTmpBuf, this->minValue);
    pipe.InitBuffer(queryNormBuf, this->batch * sizeof(float));
    pipe.InitBuffer(queryNormBrcbBuf, this->batch * this->nBlockTilePerVec * sizeof(float));

    pipe.InitBuffer(codeBookNormTmpBuf, this->codeBookMinValue);
    pipe.InitBuffer(codeBookNormBuf, this->nBlockTilePerVec * sizeof(float));

    queryNormBrcbUb = queryNormBrcbBuf.Get<float>();
    ipGm.SetGlobalBuffer((__gm__ float *)workspace);

    curReduceRepeatTimePerBatch = nBlockTilePerVec / REDUCE_BASE_SIZE;
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
    for (size_t taskIdx = blockIdx; taskIdx < totalTaskNum * CONST_TWO; taskIdx += usedCoreNum * CONST_TWO) {
        uint32_t curAicIdx = taskIdx / CONST_TWO;
        uint32_t subSpaceIdx = curAicIdx / nBlockNum;  // 当前AICORE处理的子空间idx
        uint32_t nBlockIdx = curAicIdx % nBlockNum;    // 当前AICORE处理的码本基本块idx
        // Vector: 将nBlockTile条codeBook拆成两份, 每个Vector负责处理 nBlockTile/2 条码本数据的计算
        ComputeDistance(subSpaceIdx, nBlockIdx);
    }
    AscendC::SyncAll();
    ComputeFlag();
}

__aicore__ inline void AscendcIvfpqSubspaceDistance::ComputeDistance(uint32_t subSpaceIdx, uint32_t baseBlockIdx)
{
    // AI Core上Vector核的ID
    uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
    uint32_t codeBookOffset = subSpaceIdx * kSub * dSub + baseBlockIdx * nBlockTile * dSub;
    uint32_t curAicIdx = blockIdx / CONST_TWO;
    uint32_t ipOffset = curAicIdx * batch * nBlockTile;
    uint32_t nBlockOffset = baseBlockIdx * nBlockTile;
    uint32_t minDistBlockOffset = baseBlockIdx * nBlockTile / REDUCE_BASE_SIZE * MIN_DIST_RESULT_BASE_SIZE;
    if (subBlockIdx == CONST_ONE)
    {
        codeBookOffset += nBlockTilePerVec * dSub;
        ipOffset += nBlockTilePerVec;
        nBlockOffset += nBlockTilePerVec;
        minDistBlockOffset += nBlockTilePerVec / REDUCE_BASE_SIZE * MIN_DIST_RESULT_BASE_SIZE;
    }

    // step1: 将当前子空间的batch条query搬入UB, 并计算L2Norm, 且broadcast
    ComputeQueryL2Norm(subSpaceIdx);        // -> queryNormBrcbUb[batch, nBlockTilePerVec]
    AscendC::PipeBarrier<PIPE_V>();

    // step2: 将当前子空间的NBlockTile/2 条码本搬入UB, 并计算L2Norm, 且broadcast
    ComputeCodeBookL2Norm(codeBookOffset); // -> codeBookNormBrcbUb[batch, nBlockTilePerVec]
    AscendC::PipeBarrier<PIPE_V>();

    // step3: 等待cube计算完ipGm [batch, nBlockTile]
    AscendC::CrossCoreWaitFlag(IP_READY_ID);

    // step4: 将当前vec负责的一半内积结果搬入UB, 并逐元素乘-2
    ComputeIpMuls(ipOffset);  // -> ipUb[batch, nBlockTilePerVec]
    // AscendC::PipeBarrier<PIPE_V>();

    // step5: 计算Add: codeBookNormBrcbUb + queryNormBrcbUb + ipUb
    AddDistance();      // -> distanceUb[batch, nBlockTilePerVec]

    // step6: 将计算出来的 distance[batch, nBlockTile]的结果, 按照目标格式, 搬出到GM上
    CopyOutDistance(nBlockOffset, subSpaceIdx, minDistBlockOffset);    // -> distanceGm
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
    queryQueue.EnQue(queryEnqueUb);

    // 计算L2Norm
    LocalTensor<float> queryDequeUb = queryQueue.DeQue<float>();

    // Mul
    AscendC::Mul(queryDequeUb, queryDequeUb, queryDequeUb, this->batch * this->dSub);
    AscendC::PipeBarrier<PIPE_V>();

    // ReduceSum
    AscendC::LocalTensor<uint8_t> reduceSumTmp = queryNormTmpBuf.Get<uint8_t>();
    AscendC::LocalTensor<float> reduceSumResult = queryNormBuf.Get<float>();
    uint32_t shape[] = {this->batch, this->dSub};
    constexpr bool isReuse = true;
    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR,
                       isReuse>(reduceSumResult, queryDequeUb, reduceSumTmp, shape, true);
    AscendC::PipeBarrier<PIPE_V>();

    // Broadcast
    const uint32_t dstShape_[] = {this->batch, nBlockTilePerVec};
    const uint32_t srcShape_[] = {this->batch, CONST_ONE};
    AscendC::Broadcast<float, CONST_TWO, CONST_ONE>(queryNormBrcbUb, reduceSumResult, dstShape_, srcShape_);
    queryQueue.FreeTensor(queryDequeUb);
}

__aicore__ inline void AscendcIvfpqSubspaceDistance::ComputeCodeBookL2Norm(uint32_t codeBookOffset)
{
    // copyin
    LocalTensor<float> codeBookEnqueUb = codeBookQueue.AllocTensor<float>();
    AscendC::DataCopy(codeBookEnqueUb, codeBookGm[codeBookOffset], nBlockTilePerVec * dSub);
    codeBookQueue.EnQue(codeBookEnqueUb);

    // 计算L2Norm
    LocalTensor<float> codeBookDequeUb = codeBookQueue.DeQue<float>();

    // Mul
    AscendC::Mul(codeBookDequeUb, codeBookDequeUb, codeBookDequeUb, nBlockTilePerVec * dSub);
    AscendC::PipeBarrier<PIPE_V>();

    // ReduceSum
    AscendC::LocalTensor<uint8_t> codeBookNormTmp = codeBookNormTmpBuf.Get<uint8_t>();
    AscendC::LocalTensor<float> codeBookNormResult = codeBookNormBuf.Get<float>();
    uint32_t shape[] = {nBlockTilePerVec, dSub};
    constexpr bool isReuse = true;
    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, isReuse>(
        codeBookNormResult, codeBookDequeUb, codeBookNormTmp, shape, true);
    AscendC::PipeBarrier<PIPE_V>();

    // Broadcast
    const uint32_t dstShape_[] = {this->batch, nBlockTilePerVec};
    const uint32_t srcShape_[] = {CONST_ONE, nBlockTilePerVec};
    distanceEnqueUb = distResultQueue.AllocTensor<float>();
    AscendC::Broadcast<float, CONST_TWO, 0>(distanceEnqueUb, codeBookNormResult, dstShape_, srcShape_);
    codeBookQueue.FreeTensor(codeBookDequeUb);
}

__aicore__ inline void AscendcIvfpqSubspaceDistance::ComputeIpMuls(uint32_t ipOffset)
{
    // copyin
    LocalTensor<float> ipEnqueUb = ipQueue.AllocTensor<float>();
    AscendC::DataCopyParams copyParams{
        static_cast<uint16_t>(this->batch),
        static_cast<uint16_t>(nBlockTilePerVec / FLOAT_DATA_BLOCK_SIZE),
        static_cast<uint16_t>((this->nBlockTile - nBlockTilePerVec) / FLOAT_DATA_BLOCK_SIZE),
        0};
    AscendC::DataCopy(ipEnqueUb, ipGm[ipOffset], copyParams);
    ipQueue.EnQue(ipEnqueUb);
    // muls -2
    ipUb = ipQueue.DeQue<float>();
    AscendC::Muls(ipUb, ipUb, SCALAR, this->batch * nBlockTilePerVec);
    ipQueue.FreeTensor(ipEnqueUb);
}

__aicore__ inline void AscendcIvfpqSubspaceDistance::AddDistance()
{
    AscendC::Add(distanceEnqueUb, distanceEnqueUb, queryNormBrcbUb, nBlockTilePerVec * this->batch);
    AscendC::PipeBarrier<PIPE_V>();
    // + ipUb
    AscendC::Add(distanceEnqueUb, distanceEnqueUb, ipUb, nBlockTilePerVec * this->batch);
    AscendC::PipeBarrier<PIPE_V>();

    LocalTensor<float> minDistResultEnqueUb = minDistResultQueue.AllocTensor<float>();
    AscendC::WholeReduceMin<float>(minDistResultEnqueUb,         // dst
                                   distanceEnqueUb,             // src
                                   REDUCE_BASE_SIZE,       // mask
                                   this->batch * curReduceRepeatTimePerBatch,    // repeatTime
                                   4,                      // dstRepStride
                                   1,                      // srcBlkStride
                                   8);                     // srcRepStride
    distResultQueue.EnQue(distanceEnqueUb);
    minDistResultQueue.EnQue(minDistResultEnqueUb);
}

__aicore__ inline void AscendcIvfpqSubspaceDistance::CopyOutDistance(uint32_t nBlockOffset,
                                                                     uint32_t subSpaceIdx,
                                                                     uint32_t minDistBlockOffset)
{
    uint32_t distanceOffset = nBlockOffset + subSpaceIdx * this->kSub;
    LocalTensor<float> distanceDequeUb = distResultQueue.DeQue<float>();
    AscendC::DataCopyParams copyParams{
        static_cast<uint16_t>(this->batch),     // blockCount
        static_cast<uint16_t>(nBlockTilePerVec / FLOAT_DATA_BLOCK_SIZE),  // blockLen
        0,                                      // srcGap
        static_cast<uint16_t>((this->subSpaceNum * this->kSub - nBlockTilePerVec) / FLOAT_DATA_BLOCK_SIZE)}; // dstGap
    AscendC::DataCopy(distanceGm[distanceOffset], distanceDequeUb, copyParams);
    minDistResultQueue.FreeTensor(distanceDequeUb);

    uint32_t dstStride = this->subSpaceNum * this->kSub / REDUCE_BASE_SIZE * MIN_DIST_RESULT_BASE_SIZE -
                         curReduceRepeatTimePerBatch * MIN_DIST_RESULT_BASE_SIZE;
    uint32_t minDistResultOffset = subSpaceIdx * this->kSub / REDUCE_BASE_SIZE *
                                       MIN_DIST_RESULT_BASE_SIZE + minDistBlockOffset;
    LocalTensor<float> minDistResultDequeUb = minDistResultQueue.DeQue<float>();
    AscendC::DataCopyParams minDistCopyParams{
        static_cast<uint16_t>(this->batch),      // blockCount this->batch
        static_cast<uint16_t>(curReduceRepeatTimePerBatch * MIN_DIST_RESULT_BASE_SIZE * sizeof(float)),  // blockLen
        0,                                       // srcStride
        static_cast<uint16_t>((this->subSpaceNum * this->kSub / REDUCE_BASE_SIZE * MIN_DIST_RESULT_BASE_SIZE -
                               curReduceRepeatTimePerBatch * MIN_DIST_RESULT_BASE_SIZE) * sizeof(float))};  // dstStride
    AscendC::DataCopyPad(minDistResultGm[minDistResultOffset], minDistResultDequeUb, minDistCopyParams);
    minDistResultQueue.FreeTensor(minDistResultDequeUb);
}

__aicore__ inline void AscendcIvfpqSubspaceDistance::ComputeFlag() {
    AscendC::InitGlobalMemory(flagGm, static_cast<uint64_t>(this->batch * this->subSpaceNum * FLAG_ALIGN),
                              (uint16_t) 1);
}

class IvfpqSubspaceDistanceMatmul {
public:
    __aicore__ inline IvfpqSubspaceDistanceMatmul(){};
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR codeBook, GM_ADDR distance, GM_ADDR workspace,
        const AscendcIvfpqSubspaceDistanceTilingData *__restrict tilingData, const TCubeTiling &tiling, TPipe *tPipe);
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


__aicore__ inline void IvfpqSubspaceDistanceMatmul::Init(GM_ADDR query, GM_ADDR codeBook, GM_ADDR distance,
                                                         GM_ADDR workspace,
                                                         const AscendcIvfpqSubspaceDistanceTilingData
                                                             *__restrict tilingData,
                                                         const TCubeTiling &tiling, TPipe *tPipe)
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
    for (size_t taskIdx = blockIdx; taskIdx < totalTaskNum; taskIdx += usedCoreNum) {
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
    matmulObj.template IterateAll(mmResGm[blockIdx * this->batch * this->nBlockTile]);
}

}  // namespace IndexOps

extern "C" __global__ __aicore__ void ascendc_ivfpq_subspace_distance(GM_ADDR query, GM_ADDR codeBook,
        GM_ADDR distance, GM_ADDR minDistResult, GM_ADDR flag, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe tPipe;
    GET_TILING_DATA(tiling_data, tiling);
    TCubeTiling tCubeTiling;

    if (TILING_KEY_IS(0)) {
        if ASCEND_IS_AIC {
            tCubeTiling = tiling_data.cubeTiling;
            IndexOps::IvfpqSubspaceDistanceMatmul matmulOp;
            matmulOp.Init(query, codeBook, distance, workspace, &tiling_data, tCubeTiling, &tPipe);
            REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), matmulOp.matmulObj, &matmulOp.cubeTiling);
            matmulOp.Process(&tPipe);
        }
        if ASCEND_IS_AIV {
            IndexOps::AscendcIvfpqSubspaceDistance op;
            op.Init(query, codeBook, distance, minDistResult, flag, workspace, &tiling_data, &tPipe);
            op.Process();
        }
    }
}