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

namespace AscendC
{
template <typename T>
class DistanceFlatL2MinsAtFP32
{
public:
__aicore__ inline DistanceFlatL2MinsAtFP32() {}
__aicore__ inline void InitTilingInfo(const DistanceFlatL2MinsAtFP32TilingData &tiling_data)
{
    this->blockIdx = GetBlockIdx();
    this->formerCoreNum = tiling_data.formerCoreNum;
    this->formerCoreLength = tiling_data.formerCoreLength;
    this->formerTileNum = tiling_data.formerTileNum;
    this->formerTileLength = tiling_data.formerTileLength;
    this->formerLastTileLength = tiling_data.formerLastTileLength;
    this->tailCoreNum = tiling_data.tailCoreNum;
    this->tailCoreLength = tiling_data.tailCoreLength;
    this->tailTileNum = tiling_data.tailTileNum;
    this->tailTileLength = tiling_data.tailTileLength;
    this->tailLastTileLength = tiling_data.tailLastTileLength;
    this->codesTileNum = tiling_data.codesTileNum;
    this->codesTileLength = tiling_data.codesTileLength;
    this->codesLastTileLength = tiling_data.codesLastTileLength;
    this->queryNumLength = tiling_data.queryNumLength;
    this->codesNumLength = tiling_data.codesNumLength;
    this->dimLength = tiling_data.dimLength;
    this->bufferSize = tiling_data.bufferSize;
}

__aicore__ inline void InitBuffer(GM_ADDR query, GM_ADDR codes, GM_ADDR codes_double, GM_ADDR dist_result,
                                  GM_ADDR min_result, GM_ADDR flag_shape, GM_ADDR usrWorkspace)
{
    if (this->blockIdx < this->formerCoreNum) {
        this->tileNum = this->formerTileNum;
        this->tileLength = this->formerTileLength;
        this->lastTileLength = this->formerLastTileLength;
        this->queryCoreOffset = this->blockIdx * this->formerCoreLength;
        queryGlobal.SetGlobalBuffer((__gm__ T *)query + this->queryCoreOffset * this->dimLength,
                                    this->formerCoreLength * this->dimLength);
    } else {
        this->tileNum = this->tailTileNum;
        this->tileLength = this->tailTileLength;
        this->lastTileLength = this->tailLastTileLength;
        this->queryCoreOffset = this->formerCoreNum * this->formerCoreLength +
                                (this->blockIdx - this->formerCoreNum) * this->tailCoreLength;
        queryGlobal.SetGlobalBuffer((__gm__ T *)query + this->queryCoreOffset * this->dimLength,
                                    this->tailCoreLength * this->dimLength);
    }
    codesGlobal.SetGlobalBuffer((__gm__ T *)codes, this->codesNumLength * this->dimLength);
    codesDoubleGlobal.SetGlobalBuffer((__gm__ T *)codes_double, this->codesNumLength);
    distResultGlobal.SetGlobalBuffer((__gm__ T *)dist_result, this->queryNumLength * this->codesNumLength);
    minResultGlobal.SetGlobalBuffer((__gm__ T *)min_result,
                                    this->queryNumLength * this->codesNumLength / this->mask * 2);
    flagShapeGlobal.SetGlobalBuffer((__gm__ uint16_t *)flag_shape + this->blockIdx * 16, 16);
    workspaceGlobal.SetGlobalBuffer((__gm__ T *)usrWorkspace + this->bufferSize * 2 * this->blockIdx,
                                    this->bufferSize * 2);

    pipe.InitBuffer(inQueueQuery, 1, this->formerTileLength * this->dimLength * sizeof(T));
    pipe.InitBuffer(inQueueCodesDouble, 1, this->codesTileLength * sizeof(T));
    pipe.InitBuffer(inQueueWorkspace, 1, this->formerTileLength * this->codesTileLength * sizeof(T));
    pipe.InitBuffer(distResultQueue, 1, this->formerTileLength * this->codesTileLength * sizeof(T));
    pipe.InitBuffer(minResultQueue, 1, this->formerTileLength * this->codesTileLength / 64 * 2 * sizeof(T));
    pipe.InitBuffer(flagShapeQueue, 1, 16 * sizeof(uint16_t));
    
    pipe.InitBuffer(processBuf, this->formerTileLength * this->mask * sizeof(T));
    pipe.InitBuffer(inQueueFlag, 1, 16 * sizeof(uint16_t));
}

__aicore__ inline void ProcessCore(int pIdx)
{
    LocalTensor<T> processLocal;
    SetProcessLength(pIdx - 1);
    if ((pIdx - 1) % this->codesTileNum == 0) {
        LocalTensor<T> queryLocal = inQueueQuery.AllocTensor<T>();
        DataCopy(queryLocal, queryGlobal[(pIdx - 1) /
                                         this->codesTileNum * this->formerTileLength * this->dimLength],
                 this->queryLength * this->dimLength);
        inQueueQuery.EnQue(queryLocal);
        queryLocal = inQueueQuery.DeQue<T>();
        processLocal = processBuf.Get<T>();
        Mul(queryLocal, queryLocal, queryLocal, this->queryLength * this->dimLength);
        PipeBarrier<PIPE_V>();
        Duplicate(processLocal, this->zero_float, this->queryLength * this->mask);
        PipeBarrier<PIPE_V>();
        if (this->dimLength > 64) {
            int32_t maskNum = this->dimLength / this->mask;
            for (int32_t j = 0; j < this->queryLength; j++) {
                Add(processLocal[j * this->mask], queryLocal[j * this->dimLength], processLocal[j * this->mask],
                  this->mask, maskNum, {1, 1, 1, 0, 8, 0});
            }
            PipeBarrier<PIPE_V>();
            WholeReduceSum<T>(processLocal, processLocal, this->mask, this->queryLength, 1, 1, 8);
            PipeBarrier<PIPE_V>();
        } else {
            WholeReduceSum<T>(processLocal, queryLocal, this->dimLength, this->queryLength, 1, 1, this->dimLength / 8);
            PipeBarrier<PIPE_V>();
        }
        inQueueQuery.FreeTensor(queryLocal);
    }
    WaitGemmQB();
    VecProcess((pIdx - 1) / this->codesTileNum, (pIdx -1) % this->codesTileNum, processLocal, (pIdx -1) % 2);
}

__aicore__ inline void Process()
{
    for (int32_t i = 0; i < this->tileNum * this->codesTileNum + 1; i++) {
        if (i < this->tileNum * this->codesTileNum) {
            SetProcessLength(i);
            LaunchGemmQB(i / this->codesTileNum, i % this->codesTileNum, i % 2);
        }
        if (i > 0) {
            ProcessCore(i);
        }
    }
    LocalTensor<uint16_t> flagOutLocal = flagShapeQueue.AllocTensor<uint16_t>();
    Duplicate(flagOutLocal, this->zero_uint16, 16);
    flagOutLocal.SetValue(0, 1);
    flagShapeQueue.EnQue(flagOutLocal);
    flagOutLocal = flagShapeQueue.DeQue<uint16_t>();
    DataCopy(flagShapeGlobal, flagOutLocal, 16);
    flagShapeQueue.FreeTensor(flagOutLocal);
}

public:
TPipe pipe;
using QueryType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>;
using BaseType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T, true>;
using OutType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>;
using BiasType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>;
using mm_qb = matmul::Matmul<QueryType, BaseType, OutType, BiasType>;
mm_qb gemm_qb;

private:
__aicore__ inline void VecProcess(int32_t queryProcess, int32_t codesProcess,
                                    LocalTensor<T> processLocal, int32_t pingPongId)
{
    CopyIn(queryProcess, codesProcess, pingPongId);
    Compute(processLocal);
    CopyOut(queryProcess, codesProcess);
}

__aicore__ inline void CopyIn(int32_t queryProcess, int32_t codesProcess, int32_t pingPongId)
{
    LocalTensor<T> codesDoubleLocal = inQueueCodesDouble.AllocTensor<T>();
    DataCopy(codesDoubleLocal, codesDoubleGlobal[codesProcess * this->codesTileLength], this->codesLength);
    inQueueCodesDouble.EnQue(codesDoubleLocal);
    LocalTensor<T> workspaceLocal = inQueueWorkspace.AllocTensor<T>();
    DataCopy(workspaceLocal, workspaceGlobal[pingPongId * this->bufferSize], this->queryLength * this->codesLength);
    inQueueWorkspace.EnQue(workspaceLocal);
}

__aicore__ inline void Compute(LocalTensor<T> processLocal)
{
    LocalTensor<T> codesDoubleLocal = inQueueCodesDouble.DeQue<T>();
    LocalTensor<T> workspaceLocal = inQueueWorkspace.DeQue<T>();
    LocalTensor<T> distResultLocal = distResultQueue.AllocTensor<T>();
    LocalTensor<T> minResultLocal = minResultQueue.AllocTensor<T>();
    uint32_t nShape[2] = {this->queryLength, 1};
    uint32_t nBroadShape[2] = {this->queryLength, this->codesLength};
    BroadCast<T, 2, 1>(distResultLocal, processLocal, nBroadShape, nShape);
    PipeBarrier<PIPE_V>();
    for (int32_t i = 0; i < this->queryLength; i++) {
        Add(distResultLocal[i * this->codesLength], codesDoubleLocal, distResultLocal[i * this->codesLength],
            this->codesLength);
    }
    PipeBarrier<PIPE_V>();
    Sub(distResultLocal, distResultLocal, workspaceLocal, this->queryLength * this->codesLength);
    PipeBarrier<PIPE_V>();
    Sub(distResultLocal, distResultLocal, workspaceLocal, this->queryLength * this->codesLength);
    PipeBarrier<PIPE_V>();
    for (int32_t i = 0; i < this->queryLength; i++) {
        WholeReduceMin(minResultLocal[i * this->codesLength / this->mask * 2],
                       distResultLocal[i * this->codesLength], this->mask, this->codesLength / this->mask, 1, 1, 8);
    }
    distResultQueue.EnQue(distResultLocal);
    minResultQueue.EnQue(minResultLocal);
    inQueueCodesDouble.FreeTensor(codesDoubleLocal);
    inQueueWorkspace.FreeTensor(workspaceLocal);
}

__aicore__ inline void CopyOut(int32_t queryProcess, int32_t codesProcess)
{
    LocalTensor<T> distResultLocal = distResultQueue.DeQue<T>();
    ub2gm_datacopy_aligned(this->queryLength, this->codesLength, this->codesLength,
                           this->codesNumLength, distResultLocal,
                           distResultGlobal[(this->queryCoreOffset + queryProcess * this->formerTileLength) *
                                                this->codesNumLength + codesProcess * this->codesTileLength]);
    distResultQueue.FreeTensor(distResultLocal);
    LocalTensor<T> minResultLocal = minResultQueue.DeQue<T>();
    ub2gm_datacopy_aligned(this->queryLength, this->codesLength / this->mask * 2, this->codesLength / this->mask * 2,
                           this->codesNumLength / this->mask * 2, minResultLocal,
                           minResultGlobal[(this->queryCoreOffset + queryProcess * this->formerTileLength) *
                                               this->codesNumLength / this->mask * 2 + codesProcess *
                                               this->codesTileLength / this->mask * 2]);
    minResultQueue.FreeTensor(minResultLocal);
}

__aicore__ inline void LaunchGemmQB(int32_t queryProcess, int32_t codesProcess, int32_t pingPongId)
{
    gemm_qb.SetTail(this->queryLength, this->codesLength, this->dimLength);
    gemm_qb.SetTensorA(queryGlobal[queryProcess * this->formerTileLength * this->dimLength]);
    gemm_qb.SetTensorB(codesGlobal[codesProcess * this->codesTileLength * this->dimLength], true);
    gemm_qb.template IterateAll<false>(workspaceGlobal[pingPongId * this->bufferSize], 0, false, true);
}

__aicore__ inline void WaitGemmQB()
{
    gemm_qb.WaitIterateAll();
    gemm_qb.End();
}

__aicore__ inline void SetProcessLength(int32_t process)
{
    this->queryLength = this->formerTileLength;
    if (process / this->codesTileNum == this->tileNum - 1) {
        this->queryLength = this->lastTileLength;
    }
    this->codesLength = this->codesTileLength;
    if (process % this->codesTileNum == this->codesTileNum - 1) {
        this->codesLength = this->codesLastTileLength;
    }
}

__aicore__ inline void ub2gm_datacopy_aligned(const uint32_t &rows, const uint32_t &cols, const uint32_t &src_offset,
                                              const uint32_t &dst_offset, const AscendC::LocalTensor<T> &src_tensor,
                                              const AscendC::GlobalTensor<T> &dst_tensor)
{
    constexpr uint32_t aligned_block = 32 / sizeof(T);
    if ((cols % aligned_block == 0) && (src_offset % aligned_block == 0) && (dst_offset % aligned_block == 0)) {
        AscendC::DataCopyParams ub2gm_datacopy_params;
        ub2gm_datacopy_params.blockCount = rows;
        ub2gm_datacopy_params.blockLen = cols / aligned_block;
        ub2gm_datacopy_params.srcStride = (src_offset - cols) / aligned_block;
        ub2gm_datacopy_params.dstStride = (dst_offset - cols) / aligned_block;
        DataCopy(dst_tensor, src_tensor, ub2gm_datacopy_params);
    } else {
        AscendC::DataCopyExtParams ub2gm_datacopy_params;
        ub2gm_datacopy_params.blockCount = rows;
        ub2gm_datacopy_params.blockLen = cols * sizeof(T);
        ub2gm_datacopy_params.srcStride = (src_offset - cols) / aligned_block;
        ub2gm_datacopy_params.dstStride = (dst_offset - cols) * sizeof(T);
        ub2gm_datacopy_params.rsv = 0;
        DataCopyPad(dst_tensor, src_tensor, ub2gm_datacopy_params);
    }
}

private:
GlobalTensor<T> queryGlobal;
GlobalTensor<T> codesGlobal;
GlobalTensor<T> codesDoubleGlobal;
GlobalTensor<T> distResultGlobal;
GlobalTensor<T> minResultGlobal;
GlobalTensor<uint16_t> flagShapeGlobal;
GlobalTensor<T> workspaceGlobal;

TQue<QuePosition::VECIN, 1> inQueueQuery, inQueueCodesDouble, inQueueWorkspace, inQueueFlag;
TQue<QuePosition::VECOUT, 1> distResultQueue, minResultQueue, flagShapeQueue;

TBuf<TPosition::VECCALC> processBuf;

int32_t blockIdx = 0;
int32_t formerCoreNum;
int32_t formerCoreLength;
int32_t formerTileNum;
int32_t formerTileLength;
int32_t formerLastTileLength;
int32_t tailCoreNum;
int32_t tailCoreLength;
int32_t tailTileNum;
int32_t tailTileLength;
int32_t tailLastTileLength;
int32_t codesTileNum;
int32_t codesTileLength;
int32_t codesLastTileLength;
int32_t tileNum;
int32_t tileLength;
int32_t lastTileLength;

int32_t queryNumLength;
int32_t codesNumLength;
int32_t dimLength;
int32_t bufferSize;
int32_t queryCoreOffset;
uint32_t queryLength;
uint32_t codesLength;

float32_t zero_float = 0.0;
uint16_t zero_uint16 = 0;
int32_t mask = 64;
};
}

extern "C" __global__ __aicore__ void distance_flat_l2_mins_at_fp32(GM_ADDR query, GM_ADDR codes, GM_ADDR codes_double,
    GM_ADDR dist_result, GM_ADDR min_result, GM_ADDR flag_shape, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    DistanceFlatL2MinsAtFP32<float32_t> op;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.gemm_qb, &tiling_data.gemm_qb_tiling);
    op.InitTilingInfo(tiling_data);
    op.InitBuffer(query, codes, codes_double, dist_result, min_result, flag_shape, usrWorkspace);
    op.Process();
}