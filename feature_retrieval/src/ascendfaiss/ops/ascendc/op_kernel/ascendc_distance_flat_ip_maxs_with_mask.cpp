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
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/matmul.h"

#include "op_kernel_common.h"

using namespace AscendC;
using namespace matmul;

namespace {
constexpr uint32_t BURST_LEN_HIGH = 64;
}

namespace IndexOps {
class AscendcDistanceFlatIPMaxsWithMask {
public:
    __aicore__ inline AscendcDistanceFlatIPMaxsWithMask(const AscendcDistanceFlatIPMaxsWithMaskTilingData &tilingData)
        : queryNum(tilingData.queryNum), codeNum(tilingData.codeNum), dim(tilingData.dim),
        vecCoreNum(tilingData.vecCoreNum), blockIdx(GetBlockIdx()), tiling(tilingData.cubeTilingData) {}

    __aicore__ inline void Init(GM_ADDR query,
                                GM_ADDR mask,
                                GM_ADDR shaped,
                                GM_ADDR actualSize,
                                GM_ADDR dist,
                                GM_ADDR maxDist,
                                GM_ADDR flag,
                                GM_ADDR workspace);

    __aicore__ inline void Process();

private:
    __aicore__ inline void GetActualSize();

    __aicore__ inline void SetFlag(uint32_t blockIdx);

    __aicore__ inline void ComputeLoopParameters();

    __aicore__ inline void CopyDist2GM(const LocalTensor<half> &distTensor,
                                       uint32_t codeMoveOffset,
                                       uint32_t codeMoveNum,
                                       uint32_t queryMoveOffset,
                                       uint32_t queryMoveNum);

    __aicore__ inline void DistanceComputeLoop(uint32_t aicoreMoveOffset,
                                               uint32_t aicoreCodeNum,
                                               uint32_t queryMoveOffset,
                                               uint32_t queryMoveNum,
                                               bool isTail);

    __aicore__ inline void CubeComputeLoop(uint32_t codeMoveOffset,
                                           uint32_t codeMoveNum,
                                           uint32_t queryMoveOffset,
                                           uint32_t queryMoveNum,
                                           bool isTail);

    __aicore__ inline void ComputeExtremum(const LocalTensor<half> &dist,
                                           uint32_t codeMoveNum,
                                           uint32_t queryMoveNum,
                                           LocalTensor<half> &distExtremum);

    __aicore__ inline void CopyDistExtremum2GM(const LocalTensor<half> &distExtremum,
                                               uint32_t codeMoveOffset,
                                               uint32_t codeMoveNum,
                                               uint32_t queryMoveOffset,
                                               uint32_t queryMoveNum);

    __aicore__ inline void DoMask(uint32_t codeMoveOffset,
                                  uint32_t codeMoveNum,
                                  uint32_t queryMoveOffset,
                                  uint32_t queryMoveNum,
                                  LocalTensor<half> &dist);

private:
    uint32_t queryNum;
    uint32_t codeNum;
    uint32_t dim;
    uint32_t vecCoreNum;
    uint32_t blockIdx;
    TCubeTiling tiling;

    uint32_t burstSizeOfBlock {0};
    uint32_t burstSizeEachLoop {0};

    uint32_t actualCodeNum {0};
    uint32_t burstLen {0};
    uint32_t codeNumEachCore {0};
    uint32_t aicoreCodeMoveOffset {0};
    uint32_t queryNumEachLoop {0};
    uint32_t codeNumEachLoop {0};
    uint32_t maskLenEachLoop {0};
    uint32_t maskLen {0};
    uint32_t selectLoopTime {0};
    uint8_t selectRemainder {0};

    // A/B矩阵均从GM输入，op_host中我们设置每次循环的计算量给singleCore的参数
    // cube的tiling的具体参数会自动计算，它自己控制L1等空间的使用
    using MatmulTypeA = MatmulType<TPosition::GM, CubeFormat::ND, half>;
    // true:B矩阵需要转置
    using MatmulTypeB = MatmulType<TPosition::GM, CubeFormat::ND, half, true>;
    using MatmulTypeC = MatmulType<TPosition::VECCALC, CubeFormat::ND, half>;
    Matmul<MatmulTypeA, MatmulTypeB, MatmulTypeC> matmulObj;

    GlobalTensor<half> queryGM;
    GlobalTensor<uint8_t> maskGM;
    GlobalTensor<half> codeGM;
    GlobalTensor<half> distGM;
    GlobalTensor<uint32_t> actualSizeGM;
    GlobalTensor<uint16_t> flagGM;
    GlobalTensor<half> distMaxGM;

    TPipe pipe;

    TSCM<TPosition::GM, 1> querySCM;
    TQue<QuePosition::VECIN, 1> maskQue;
    TQue<QuePosition::VECOUT, 1> distQueue;
    TQue<QuePosition::VECOUT, 1> distExtremumQueue;
};

__aicore__ inline void AscendcDistanceFlatIPMaxsWithMask::SetFlag(uint32_t blockIdx)
{
    // 需要计算的dist和vmdist都搬运到GM（MET3）之后再开始设置flag
    pipe_barrier(PIPE_MTE3);

    TBuf<> flagBuf;
    pipe.InitBuffer(flagBuf, DEFAULT_C0_SIZE);
    LocalTensor<uint16_t> flagLocal = flagBuf.Get<uint16_t>(DEFAULT_C0_SIZE / sizeof(uint16_t));
    // flag只需要设置第0个元素为1即可
    flagLocal.SetValue(0, 1);
    // 需要等到SetValue（PIPE_S）结束才能开始DataCopy搬运flag到GM（MET3）
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);

    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    DataCopy(flagGM[blockIdx * Utils::CUBE_ALIGN], flagLocal, flagLocal.GetSize());
}

__aicore__ inline void AscendcDistanceFlatIPMaxsWithMask::GetActualSize()
{
    TBuf<> actualSizeBuf;
    pipe.InitBuffer(actualSizeBuf, DEFAULT_C0_SIZE);
    // actualSize仅第0个数有效，只需要搬运32B
    LocalTensor<uint32_t> actualSizeLocal = actualSizeBuf.Get<uint32_t>(DEFAULT_C0_SIZE / sizeof(uint32_t));
    DataCopy(actualSizeLocal, actualSizeGM, actualSizeLocal.GetSize());
    // 搬运actualSize为MTE2数据流，需要等它搬运到UB之后，才能下发GetValue为PIPE_S计算流水线获取值
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    actualCodeNum = actualSizeLocal.GetValue(0);
}

__aicore__ inline void AscendcDistanceFlatIPMaxsWithMask::ComputeLoopParameters()
{
    // 实际要计算的底库数量
    GetActualSize();

    burstLen = BURST_LEN_HIGH;
    // 从tiling中获取query和code分块的参数，在op_host中设置的
    queryNumEachLoop = tiling.M;
    codeNumEachLoop = tiling.N;
    maskLen = codeNum / Utils::MASK_BIT_NUM;
    // 按照每次循环计算量，来尽可能均分底库。尾块给最后一个core处理
    uint32_t totalLoopTime = actualCodeNum / codeNumEachLoop;
    uint32_t leftCode = actualCodeNum % codeNumEachLoop;
    uint32_t eachCoreLoopTime = totalLoopTime / vecCoreNum;
    uint32_t leftLoopTime = totalLoopTime % vecCoreNum;
    if (leftLoopTime != 0) {
        if (blockIdx < leftLoopTime) {
            // 如果有剩余的循环次数，则均分给前面的core
            codeNumEachCore = (eachCoreLoopTime + 1) * codeNumEachLoop; // blockIdx: [0, leftLoopTime)
            aicoreCodeMoveOffset = blockIdx * codeNumEachCore;
        } else {
            codeNumEachCore = eachCoreLoopTime * codeNumEachLoop;  // blockIdx: [leftLoopTime, 47)
            aicoreCodeMoveOffset = leftLoopTime * (eachCoreLoopTime + 1) *
                codeNumEachLoop + (blockIdx - leftLoopTime) * codeNumEachCore;
        }
    } else {
        // 循环次数均分给所有core
        codeNumEachCore = eachCoreLoopTime * codeNumEachLoop;  // blockIdx: [leftLoopTime, 47)
        aicoreCodeMoveOffset = blockIdx * codeNumEachCore;
    }
    // 尾块数据交给最后一个core处理
    if (blockIdx == vecCoreNum - 1) {
        codeNumEachCore = codeNumEachCore + leftCode;  // blockIdx: [48]
        aicoreCodeMoveOffset = actualCodeNum - codeNumEachCore;
    }

    // cpp申请空间的时候保证对齐，codeNum能被burstLen整除
    // 每个block对应的burst的长度，1个burst包含2个half数据，value和index
    burstSizeOfBlock = codeNum / burstLen * Utils::BURST_BLOCK_RATIO;
    // 每次循环计算时burst的长度
    burstSizeEachLoop = codeNumEachLoop / burstLen * Utils::BURST_BLOCK_RATIO;

    // 每次循环计算时，mask的长度
    maskLenEachLoop = codeNumEachLoop / Utils::MASK_BIT_NUM;
    // 每次循环计算时，select的重复次数
    uint32_t validResultVecRepeats = queryNumEachLoop * codeNumEachLoop / Utils::VIC_HALF_FULL_MASK;
    selectLoopTime = validResultVecRepeats / Utils::SELECT_REPEAT_TIME;
    selectRemainder = static_cast<uint8_t>(validResultVecRepeats % Utils::SELECT_REPEAT_TIME);
}

__aicore__ inline void AscendcDistanceFlatIPMaxsWithMask::Init(GM_ADDR query, GM_ADDR mask, GM_ADDR shaped,
    GM_ADDR actualSize, GM_ADDR dist, GM_ADDR maxDist,
    GM_ADDR flag, GM_ADDR workspace)
{
    actualSizeGM.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(actualSize), DEFAULT_C0_SIZE);

    ComputeLoopParameters();

    queryGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(query), queryNum * dim);
    // mask是已经在searchPaged中对query偏移后的。maskBlockOffset是当前searchPaged中，已经计算过的底库的偏移量
    maskGM.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(mask));
    codeGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(shaped), dim * codeNum);
    distGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(dist), queryNum * codeNum);
    flagGM.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(flag), vecCoreNum * Utils::CUBE_ALIGN);
    distMaxGM.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(maxDist), queryNum * burstSizeOfBlock);
}

__aicore__ inline void AscendcDistanceFlatIPMaxsWithMask::Process()
{
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulObj);

    // 底库较少时，分给某一些core的计算量可能为0
    if (codeNumEachCore == 0) {
        SetFlag(blockIdx);
        return;
    }

    // 按照整块的大小去申请空间，每次循环计算最大占用UB的空间
    // 每次循环计算出来dist数量为queryNumEachLoop * codeNumEachLoop
    // 每次循环计算出来的vmDist数量为queryNumEachLoop * burstSizeEachLoop
    pipe.InitBuffer(distQueue, 1, queryNumEachLoop * codeNumEachLoop * sizeof(half));
    pipe.InitBuffer(distExtremumQueue, 1, queryNumEachLoop * burstSizeEachLoop * sizeof(half));
    pipe.InitBuffer(maskQue, 1, queryNumEachLoop * maskLenEachLoop * sizeof(uint8_t));

    matmulObj.Init(&tiling);

    // query分块的循环计算次数
    uint32_t queryLoopTime = queryNum / queryNumEachLoop;
    // 循环处理query整块
    for (uint32_t queryLoopIdx = 0; queryLoopIdx < queryLoopTime; queryLoopIdx++) {
        DistanceComputeLoop(aicoreCodeMoveOffset,
                            codeNumEachCore,
                            queryLoopIdx * queryNumEachLoop,
                            queryNumEachLoop,
                            false);
    }

    // 处理query尾块
    uint32_t queryLastNum = queryNum % queryNumEachLoop;
    if (queryLastNum > 0) {
        DistanceComputeLoop(aicoreCodeMoveOffset,
                            codeNumEachCore,
                            queryLoopTime * queryNumEachLoop,
                            queryLastNum,
                            true);
    }

    matmulObj.End();

    SetFlag(blockIdx);
}

__aicore__ inline void AscendcDistanceFlatIPMaxsWithMask::DistanceComputeLoop(uint32_t aicoreMoveOffset,
    uint32_t aicoreCodeNum, uint32_t queryMoveOffset, uint32_t queryMoveNum,
    bool isTail)
{
    // 根据query的偏移本次计算的query的起始位置
    GlobalTensor<half> curQueryGM = queryGM[queryMoveOffset * dim];
    matmulObj.SetTensorA(curQueryGM);

    // code分块的循环计算次数
    uint32_t codeLoopTime = aicoreCodeNum / codeNumEachLoop;
    // 循环处理code整块
    for (uint32_t codeLoopIdx = 0; codeLoopIdx < codeLoopTime; codeLoopIdx++) {
        CubeComputeLoop(aicoreMoveOffset + codeLoopIdx * codeNumEachLoop,
                        codeNumEachLoop,
                        queryMoveOffset,
                        queryMoveNum,
                        isTail);
    }

    // 处理code尾块
    uint32_t codeLastNum = aicoreCodeNum % codeNumEachLoop;
    if (codeLastNum > 0) {
        CubeComputeLoop(aicoreMoveOffset + codeLoopTime * codeNumEachLoop,
                        codeLastNum,
                        queryMoveOffset,
                        queryMoveNum,
                        true);
    }
}

__aicore__ inline void AscendcDistanceFlatIPMaxsWithMask::CopyDist2GM(const LocalTensor<half> &distTensor,
    uint32_t codeMoveOffset, uint32_t codeMoveNum, uint32_t queryMoveOffset,
    uint32_t queryMoveNum)
{
    uint32_t startOffset = queryMoveOffset * codeNum + codeMoveOffset;
    uint16_t nBlock = static_cast<uint16_t>(Utils::DivUp(codeMoveNum, Utils::CUBE_ALIGN));
    DataCopyParams copyParam = { static_cast<uint16_t>(queryMoveNum),
                                 nBlock,
                                 static_cast<uint16_t>((codeNumEachLoop / Utils::CUBE_ALIGN) - nBlock),
                                 static_cast<uint16_t>((codeNum - codeMoveNum) / Utils::CUBE_ALIGN) };
    DataCopy(distGM[startOffset], distTensor, copyParam);
}

__aicore__ inline void AscendcDistanceFlatIPMaxsWithMask::ComputeExtremum(const LocalTensor<half> &dist,
    uint32_t codeMoveNum, uint32_t queryMoveNum,
    LocalTensor<half> &distExtremum)
{
    half zero = 0;
    Duplicate(distExtremum, zero, distExtremum.GetSize());
    uint32_t srcRepStride = burstLen * sizeof(half) / DEFAULT_C0_SIZE;
    uint32_t repeatTimes = codeMoveNum / burstLen;
    if (repeatTimes > 0) {
        for (uint32_t j = 0; j < queryMoveNum; j++) {
            WholeReduceMax(distExtremum[j * burstSizeEachLoop],
                           dist[j * codeNumEachLoop],
                           burstLen,
                           repeatTimes,
                           1,
                           1,
                           srcRepStride);
        }
    }

    uint32_t lastNum = codeMoveNum % burstLen;
    if (lastNum > 0) {
        for (uint32_t j = 0; j < queryMoveNum; j++) {
            WholeReduceMax(distExtremum[j * burstSizeEachLoop + repeatTimes * Utils::BURST_BLOCK_RATIO],
                           dist[j * codeNumEachLoop + repeatTimes * burstLen],
                           lastNum,
                           1,
                           1,
                           1,
                           srcRepStride);
        }
    }
}

__aicore__ inline void AscendcDistanceFlatIPMaxsWithMask::CopyDistExtremum2GM(const LocalTensor<half> &distExtremum,
    uint32_t codeMoveOffset, uint32_t codeMoveNum, uint32_t queryMoveOffset,
    uint32_t queryMoveNum)
{
    uint32_t dstOffset = queryMoveOffset * burstSizeOfBlock + codeMoveOffset / burstLen * Utils::BURST_BLOCK_RATIO;
    uint32_t burstNum = Utils::DivUp(codeMoveNum, burstLen);
    uint32_t blocks = Utils::DivUp(burstNum * Utils::BURST_BLOCK_RATIO, Utils::CUBE_ALIGN);
    DataCopyParams copyParam = { static_cast<uint16_t>(queryMoveNum),
                                 static_cast<uint16_t>(blocks),
                                 static_cast<uint16_t>(burstSizeEachLoop / Utils::CUBE_ALIGN - blocks),
                                 static_cast<uint16_t>(burstSizeOfBlock / Utils::CUBE_ALIGN - blocks) };
    DataCopy(distMaxGM[dstOffset], distExtremum, copyParam);
}

__aicore__ inline void AscendcDistanceFlatIPMaxsWithMask::DoMask(uint32_t codeMoveOffset,
    uint32_t codeMoveNum, uint32_t queryMoveOffset, uint32_t queryMoveNum,
    LocalTensor<half> &dist)
{
    auto maskLocal = maskQue.AllocTensor<uint8_t>();
    // 本次算子已经计算好的query的对应的mask偏移:queryMoveOffset * maskLen
    // 本次算子已经计算好的code的对应的mask偏移:codeMoveOffset / Utils::MASK_BIT_NUM
    uint64_t maskOffset = queryMoveOffset * static_cast<uint64_t>(maskLen) + codeMoveOffset / Utils::MASK_BIT_NUM;

    for (uint32_t i = 0; i < queryMoveNum; i++) {
        DataCopy(maskLocal[i * maskLenEachLoop], maskGM[i * static_cast<uint64_t>(maskLen) + maskOffset],
            maskLenEachLoop);
    }
    maskQue.EnQue(maskLocal);
    maskLocal = maskQue.DeQue<uint8_t>();
    // dstBlkStride,src0BlkStride,src1BlkStride 1
    // dstRepStride,src0RepStride,src1RepStride 8
    BinaryRepeatParams param {
        1,
        1,
        1,
        8,
        8,
        8
    };

    const uint32_t distOffset = Utils::SELECT_REPEAT_TIME * Utils::VIC_HALF_FULL_MASK;
    const uint32_t maskRepateOffset = distOffset / Utils::MASK_BIT_NUM;
    for (uint32_t i = 0; i < selectLoopTime; i++) {
        Select(dist[i * distOffset], maskLocal[i * maskRepateOffset], dist[i * distOffset], Utils::HALF_MIN,
            SELMODE::VSEL_TENSOR_SCALAR_MODE, Utils::VIC_HALF_FULL_MASK, Utils::SELECT_REPEAT_TIME, param);
    }
    if (selectRemainder != 0) {
        Select(dist[selectLoopTime * distOffset], maskLocal[selectLoopTime * maskRepateOffset],
            dist[selectLoopTime * distOffset], Utils::HALF_MIN, SELMODE::VSEL_TENSOR_SCALAR_MODE,
            Utils::VIC_HALF_FULL_MASK, selectRemainder, param);
    }

    maskQue.FreeTensor(maskLocal);
}

__aicore__ inline void AscendcDistanceFlatIPMaxsWithMask::CubeComputeLoop(uint32_t codeMoveOffset,
    uint32_t codeMoveNum, uint32_t queryMoveOffset, uint32_t queryMoveNum,
    bool isTail)
{
    LocalTensor<half> distTensor = distQueue.AllocTensor<half>();
    LocalTensor<half> distExtremumensor = distExtremumQueue.AllocTensor<half>();

    GlobalTensor<half> curLoopCodeGM = codeGM[codeMoveOffset * dim];
    matmulObj.SetTensorB(curLoopCodeGM, true);

    // 如果是尾块的话，query的尾块或者code的尾块，此时queryMoveNum可能不是设置的queryNumEachLoop大小，
    // codeMoveNum也可能不是设置的codeNumEachLoop大小，需要重新设置singleCoreM\singleCoreN
    if (isTail) {
        matmulObj.SetTail(queryMoveNum, codeMoveNum, dim);
    }

    matmulObj.IterateAll(distTensor);

    distQueue.EnQue(distTensor);
    distTensor = distQueue.DeQue<half>();

    DoMask(codeMoveOffset, codeMoveNum, queryMoveOffset, queryMoveNum, distTensor);

    ComputeExtremum(distTensor, codeMoveNum, queryMoveNum, distExtremumensor);

    // 目的流水线Cast对应PIPE_V，源流水线DataCopy对应PIPE_MTE3
    // Cast和ReduceMaX都是vector指令，是串行执行的
    // 将dist和maxDist搬运到GM，需要等待这两个vector执行完毕
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    CopyDist2GM(distTensor, codeMoveOffset, codeMoveNum, queryMoveOffset, queryMoveNum);
    CopyDistExtremum2GM(distExtremumensor, codeMoveOffset, codeMoveNum, queryMoveOffset, queryMoveNum);

    distQueue.FreeTensor(distTensor);
    distExtremumQueue.FreeTensor(distExtremumensor);

    // 目的流水线DataCopy对应PIPE_V，源流水线DataCopy对应PIPE_MTE3
    // matmul将数据从GM搬运进来是MTE2，将结果数据搬出去对应MTE3，避免之前的结果没有搬运，而matmul计算太快的导致结果覆盖了
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
}
}

extern "C" __global__ __aicore__ void ascendc_distance_flat_ip_maxs_with_mask(GM_ADDR query, GM_ADDR mask,
    GM_ADDR shaped, GM_ADDR actualSize, GM_ADDR dist,
    GM_ADDR maxDist, GM_ADDR flag, GM_ADDR workspace,
    GM_ADDR tiling)
{
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    IndexOps::AscendcDistanceFlatIPMaxsWithMask op(tilingData);
    if ASCEND_IS_AIV {
        op.Init(query, mask, shaped, actualSize, dist, maxDist, flag, workspace);
    }
    op.Process();
}
