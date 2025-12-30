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
using namespace AscendC;

namespace {
    // 所有matrix(CUBE)计算以16字节为单位, 因此所有矩阵乘和对应的矩阵分型均以16为单位
    constexpr uint32_t CUBE_ALIGN = 16;
}

class KernelDistanceComputeL2 {
public:
    __aicore__ inline KernelDistanceComputeL2() {
    }
    __aicore__ inline void Init(GM_ADDR queryCode,
                                GM_ADDR codebookL2,
                                GM_ADDR l1KIndices,
                                GM_ADDR outDists,
                                GM_ADDR opFlag,
                                uint32_t subDim1,
                                uint32_t subDim2,
                                uint32_t nlist1,
                                uint32_t nlist2,
                                uint32_t nprobe1,
                                uint32_t n,
                                uint32_t formerNum,
                                uint32_t probePerBlockFormer,
                                uint32_t probePerBlockLatter,
                                uint32_t moveTimesL1PerProbe,
                                uint32_t tailSizeL1PerProbe,
                                uint32_t moveTimesL0BPerBlockTail,
                                uint32_t remainSizeL0BPerBlock,
                                uint32_t sizeCodeBookL1BBuffer,
                                uint32_t sizeCodeBookL0BBuffer,
                                uint32_t cubeAlign)
    {
        this->n = n;
        this->nlist1 = nlist1;
        this->nlist2 = nlist2;
        this->subDim1 = subDim1;
        this->subDim2 = subDim2;
        this->nprobe1 = nprobe1;
        this->formerNum = formerNum;
        this->probePerBlockFormer = probePerBlockFormer;
        this->probePerBlockLatter = probePerBlockFormer;

        this->moveTimesL1PerProbe = moveTimesL1PerProbe;
        this->tailSizeL1PerProbe = tailSizeL1PerProbe;
        this->moveTimesL0BPerBlockTail = moveTimesL0BPerBlockTail;
        this->remainSizeL0BPerBlock = remainSizeL0BPerBlock;

        this->sizeCodeBookL1BBuffer = sizeCodeBookL1BBuffer ;
        this->sizeCodeBookL0BBuffer = sizeCodeBookL0BBuffer ;

        this->blkIdx = GetBlockIdx();
        // 单个query需要由当前核处理的probe数量
        this->probePerBlock = (blkIdx < formerNum ? probePerBlockFormer : probePerBlockLatter);

        this->cubeAlign = cubeAlign;

        queryCodeGM.SetGlobalBuffer((__gm__ half*)queryCode);
        codebookL2GM.SetGlobalBuffer((__gm__ half*)codebookL2);
        l1KIndicesGM.SetGlobalBuffer((__gm__ uint16_t*)l1KIndices);
        outDistsGM.SetGlobalBuffer((__gm__ half*)outDists);
        opFlagGM.SetGlobalBuffer((__gm__ uint16_t*)opFlag + blkIdx * 16);

        pipe.InitBuffer(inQueueA1, 1, cubeAlign * subDim1 * sizeof(half));
        pipe.InitBuffer(inQueueA2, 1, cubeAlign * subDim1 * sizeof(half));
        pipe.InitBuffer(inQueueB1, bufferNumL1B, this->sizeCodeBookL1BBuffer);
        pipe.InitBuffer(inQueueB2, bufferNumL0B, this->sizeCodeBookL0BBuffer);
        pipe.InitBuffer(outQueueCO1,
                        bufferNumL0B, cubeAlign * ((this->sizeCodeBookL0BBuffer/subDim1/sizeof(half))) * sizeof(half));
        pipe.InitBuffer(outQueueCO2, 1, nlist2*subDim2*sizeof(half));
        pipe.InitBuffer(indicesQueue, 1, (n * nprobe1 * sizeof(uint16_t) / 32 + 1) * 32);

        pipe.InitBuffer(inQueueVec1, 1, nlist2*subDim2*sizeof(half));
        pipe.InitBuffer(outQueueVec1, 1, nlist2*subDim2*sizeof(half)/subDim2);
        pipe.InitBuffer(outQueueFlag, 1, 16 * sizeof(uint16_t));
    }

    __aicore__ inline void Process()
    {
// 将Indices矩阵放入VECIN
        CopyInIndices();
// 当前核需要处理的总probe数量
        int32_t loopCount = n * probePerBlock;
        LocalTensor<uint16_t> indicesUb = indicesQueue.DeQue<uint16_t>();
// 每次循环处理一个probe
        for (int32_t loop = 0; loop < loopCount ; loop++) {
            int32_t indicesIdx = GetIndicesIdx(loop);
//           从当前indices下标中取到nlist1的id
            int32_t nlist1Idx = indicesUb.GetValue(indicesIdx);
            int32_t queryIdx = loop / probePerBlock;
            CopyInA(queryIdx, nlist1Idx);
            SplitA();
            LocalTensor <half> a2Local = inQueueA2.DeQue<half>();
            LocalTensor <half> c2Local = outQueueCO2.AllocTensor<half>();
//          多次搬运将一个probe对应的B矩阵搬进L1
            for (int32_t moveTimesL1 = 0; moveTimesL1 < moveTimesL1PerProbe; moveTimesL1++) {
//              通过已搬运次数判断当前应搬L1的数据大小
                int32_t curSizeL1B = (moveTimesL1 == moveTimesL1PerProbe-1) ?
                    tailSizeL1PerProbe : (sizeCodeBookL1BBuffer);
                CopyInB(nlist1Idx, moveTimesL1, curSizeL1B);
                LocalTensor <half> b1Local = inQueueB1.DeQue<half>();
//              当前L1需要搬运几次到L0B
                int32_t moveTimesL0BPerL1 = DivUp(curSizeL1B, sizeCodeBookL0BBuffer);
//              多次搬运将一个L1里的B1矩阵搬进L0B
                for (int32_t moveTimesL0B = 0; moveTimesL0B < moveTimesL0BPerL1; moveTimesL0B++) {
                    SplitB(b1Local, moveTimesL0B);
                    MatMulCompute(a2Local);
//                  每次搬进以及计算的大小均为sizeCodeBookL0BBuffer，但是最后一次只需要搬出remainSizeL0BPerBlock
                    Aggregate(c2Local, moveTimesL1, moveTimesL0B, moveTimesL0BPerL1);
                }
                inQueueB1.FreeTensor(b1Local);
            }
            inQueueA2.FreeTensor(a2Local);
            outQueueCO2.EnQue<half>(c2Local);
//            C02内部元素求平方
            SquareCompute();
//            每subdim2个元素累加
            ReduceSumCompute();
//            通过当前处理的indices的下标判断结果应该放在什么位置
            CopyOut(indicesIdx);
        }
        CopyFlagOut();
        indicesQueue.FreeTensor(indicesUb);
    }

private:
// 向上整除接口
    __aicore__ inline int32_t DivUp(int32_t x, int32_t y)
    {
        return (x + y - 1) / y;
    }
// 采用二级接口将indices全部搬运进indicesLocal（VECIN）
    __aicore__ inline void CopyInIndices()
    {
        LocalTensor<uint16_t> indicesUb = indicesQueue.AllocTensor<uint16_t>();
//      32B对齐
        DataCopy(indicesUb[0], l1KIndicesGM[0], (n * nprobe1 * sizeof(uint16_t) / 32 + 1) * 32);
        indicesQueue.EnQue(indicesUb);
    }
// 通过当前循环次数和core得到应取得的indices下标，通过此下标可取得nlist1_id
    __aicore__ inline int32_t GetIndicesIdx(int32_t loop)
    {
        int32_t line = loop / probePerBlock;
        int32_t column = loop % probePerBlock;
        if (blkIdx < formerNum) {
            return line*nprobe1 + blkIdx * probePerBlockFormer + column;
        }
        return line * nprobe1 + formerNum * probePerBlockFormer + (blkIdx - formerNum) * probePerBlockLatter + column;
    }

// 将一个query搬到L1中并转换成zN格式（同时也是zZ格式）
    __aicore__ inline void CopyInA(int32_t queryIdx, int32_t nlist1Idx)
    {
        LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
        DataCopy(a1Local[0], queryCodeGM[queryIdx * subDim1 * nlist1 + nlist1Idx * subDim1],
                 { static_cast<uint16_t>(subDim1/16), 1, 0, 15 });

        inQueueA1.EnQue(a1Local);
    }

    __aicore__ inline void SplitA()
    {
        LocalTensor<half> a1Local = inQueueA1.DeQue<half>();
        LocalTensor<half> a2Local = inQueueA2.AllocTensor<half>();

        LoadData2dParams loadDataParams;
        loadDataParams.repeatTimes = subDim1/16 ;
        loadDataParams.srcStride = 1;
        loadDataParams.ifTranspose = false;
        LoadData(a2Local[0], a1Local[0], loadDataParams);
        inQueueA2.EnQue<half>(a2Local);
        inQueueA1.FreeTensor(a1Local);
    }

    __aicore__ inline void CopyInB(int32_t nlist1Idx, int32_t moveTimesL1, int32_t curSizeL1B)
    {
        LocalTensor<half> b1Local = inQueueB1.AllocTensor<half>();
        int32_t srcOffset = nlist1Idx * (subDim1 * nlist2 * subDim2) +
            moveTimesL1 * (sizeCodeBookL1BBuffer / sizeof(half));
        DataCopy(b1Local[0], codebookL2GM[srcOffset], { 1, static_cast<uint16_t>(curSizeL1B/32), 0, 0 });

        inQueueB1.EnQue(b1Local);
    }

    __aicore__ inline void SplitB(const LocalTensor<half>& b1Local, const int32_t moveTimesL0B)
    {
        int32_t srcOffset = moveTimesL0B * sizeCodeBookL0BBuffer / sizeof(half);
        int32_t dstOffset = 0;
        int32_t columnBlock = sizeCodeBookL0BBuffer / sizeof(half) / subDim1 / CUBE_ALIGN;
        LocalTensor<half> b2Local = inQueueB2.AllocTensor<half>();

        for (int32_t i = 0; i < subDim1 / CUBE_ALIGN; ++i) {
            LoadData2dParams loadDataParams;
            loadDataParams.repeatTimes = columnBlock;
            loadDataParams.srcStride = subDim1 / CUBE_ALIGN;
            loadDataParams.ifTranspose = false;

            LoadData(b2Local[dstOffset], b1Local[srcOffset], loadDataParams);
            srcOffset += CUBE_ALIGN * CUBE_ALIGN;
            dstOffset += CUBE_ALIGN * CUBE_ALIGN * columnBlock;
        }

        inQueueB2.EnQue<half>(b2Local);
    }

    __aicore__ inline void MatMulCompute(const LocalTensor<half>& a2Local)
    {
        uint16_t mComputeSize = 16;
        uint16_t nComputeSize = sizeCodeBookL0BBuffer / sizeof(half) / subDim1;
        uint16_t kComputeSize = subDim1;
        LocalTensor<half> b2Local = inQueueB2.DeQue<half>();
        LocalTensor<half> c1Local = outQueueCO1.AllocTensor<half>();

        Mmad(c1Local, a2Local, b2Local, { mComputeSize, nComputeSize, kComputeSize, false, 0, false, false, false });

        outQueueCO1.EnQue<half>(c1Local);
        inQueueB2.FreeTensor(b2Local);
    }

    __aicore__ inline void Aggregate(LocalTensor<half>& c2Local, const int32_t moveTimesL1, const int32_t moveTimesL0B,
                                     const int32_t moveTimesL0BPerL1)
    {
        LocalTensor<half> c1Local = outQueueCO1.DeQue<half>();
        int32_t columnBlock = 0;
        int32_t dstOffset = moveTimesL1 * (sizeCodeBookL1BBuffer / sizeof(half) / subDim1) +
            moveTimesL0B * (sizeCodeBookL0BBuffer / sizeof(half) / subDim1);
        if ((moveTimesL1 == moveTimesL1PerProbe-1)&&(moveTimesL0B == moveTimesL0BPerL1-1)) {
            columnBlock=remainSizeL0BPerBlock/sizeof(half)/subDim1/16;
        } else {
            columnBlock=sizeCodeBookL0BBuffer/sizeof(half)/subDim1/16;
        }

        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = columnBlock;
        DataCopyEnhancedParams enhancedParams;
        enhancedParams.blockMode = BlockMode::BLOCK_MODE_VECTOR;
        DataCopy(c2Local[dstOffset], c1Local, dataCopyParams, enhancedParams);
        outQueueCO1.FreeTensor(c1Local);
    }

    __aicore__ inline void SquareCompute()
    {
        LocalTensor <half> c2Local = outQueueCO2.DeQue<half>();
        LocalTensor<half> invec1Local = inQueueVec1.AllocTensor<half>();
        Mul(invec1Local, c2Local, c2Local, nlist2*subDim2);

        inQueueVec1.EnQue<half>(invec1Local);
        outQueueCO2.FreeTensor(c2Local);
    }

    __aicore__ inline void ReduceSumCompute()
    {
        LocalTensor <half> invec1Local = inQueueVec1.DeQue<half>();
        LocalTensor<half> outvec1Local = outQueueVec1.AllocTensor<half>();
        WholeReduceSum<half>(outvec1Local, invec1Local, subDim2, nlist2, 1, 1, subDim2*sizeof(half)/32);

        outQueueVec1.EnQue<half>(outvec1Local);
        inQueueVec1.FreeTensor(invec1Local);
    }

    __aicore__ inline void CopyOut(const int32_t indicesIdx)
    {
        LocalTensor<half> outvec1Local = outQueueVec1.DeQue<half>();
        DataCopy(outDistsGM[indicesIdx * nlist2], outvec1Local,
                 { 1, static_cast<uint16_t>(nlist2*sizeof(half)/32), 0, 0});
        outQueueVec1.FreeTensor(outvec1Local);
    }

    __aicore__ inline void CopyFlagOut()
    {
        LocalTensor<uint16_t> outflagUb = outQueueFlag.AllocTensor<uint16_t>();
        outflagUb.SetValue(0, 1);
        DataCopy(opFlagGM, outflagUb, 16);
        outQueueFlag.FreeTensor(outflagUb);
    }

private:
    static const uint32_t bufferNumL1B = 1;
    static const uint32_t bufferNumL0B = 1;
    TPipe pipe;

    TQue<QuePosition::A1, 1> inQueueA1;
    TQue<QuePosition::A2, 1> inQueueA2;
    TQue<QuePosition::B1, bufferNumL1B> inQueueB1;
    TQue<QuePosition::B2, bufferNumL0B> inQueueB2;
    // dst queue
    TQue<QuePosition::CO1, bufferNumL0B> outQueueCO1;
    TQue<QuePosition::CO2, 1> outQueueCO2;

    TQue<QuePosition::VECIN, 1> indicesQueue;

    TQue<QuePosition::VECIN, 1> inQueueVec1;
    TQue<QuePosition::VECOUT, 1> outQueueVec1;
    TQue<QuePosition::VECIN, 1> outQueueFlag;

    GlobalTensor<half> queryCodeGM;
    GlobalTensor<half> codebookL2GM;
    GlobalTensor<uint16_t> l1KIndicesGM;
    GlobalTensor<half> outDistsGM;
    GlobalTensor<uint16_t> opFlagGM;

    uint32_t subDim1;
    uint32_t subDim2;
    uint32_t nlist1;
    uint32_t nlist2;
    uint32_t nprobe1;
    uint32_t n;
    uint32_t formerNum;
    uint32_t probePerBlockFormer;
    uint32_t probePerBlockLatter;
    uint32_t moveTimesL1PerProbe;
    uint32_t tailSizeL1PerProbe;
    uint32_t moveTimesL0BPerBlockTail;
    uint32_t remainSizeL0BPerBlock;

    uint32_t probePerBlock;
    uint32_t sizeCodeBookL1BBuffer;
    uint32_t sizeCodeBookL0BBuffer;
    uint32_t cubeAlign;
    uint32_t blkIdx;
};

#ifndef __CCE_KT_TEST__
extern "C" __global__ __aicore__ void vstar_compute_l2(GM_ADDR queryCode, GM_ADDR codebookL2, GM_ADDR l1KIndices,
                                                       GM_ADDR outDists, GM_ADDR opFlag, GM_ADDR workspace,
                                                       GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelDistanceComputeL2 op;
    op.Init(queryCode,
            codebookL2,
            l1KIndices,
            outDists,
            opFlag,
            tiling_data.subDim1,
            tiling_data.subDim2,
            tiling_data.nlist1,
            tiling_data.nlist2,
            tiling_data.nprobe1,
            tiling_data.n,
            tiling_data.formerNum,
            tiling_data.probePerBlockFormer,
            tiling_data.probePerBlockLatter,
            tiling_data.moveTimesL1PerProbe,
            tiling_data.tailSizeL1PerProbe,
            tiling_data.moveTimesL0BPerBlockTail,
            tiling_data.remainSizeL0BPerBlock,
            tiling_data.sizeCodeBookL1BBuffer,
            tiling_data.sizeCodeBookL0BBuffer,
            tiling_data.cubeAlign);

    op.Process();
}

#else
extern "C" __global__ __aicore__ void vstar_compute_l2(GM_ADDR queryCode, GM_ADDR codebookL2, GM_ADDR l1KIndices,
                                                       GM_ADDR outDists, GM_ADDR opFlag) {
    KernelDistanceComputeL2 op;
    op.Init(queryCode, codebookL2, l1KIndices, outDists, opFlag,
            64, // subDim1
            16, // subDim2
            1024, // nlist1
            64, // nlist2
            16, // nprobe1
            128, // n
            0, // formerNum
            2, // probePerBlockFormer
            2, // probePerBlockLatter
            1, // moveTimesL1PerProbe
            128 * 1024, // tailSizeL1PerProbe
            2, // moveTimesL0BPerBlockTail
            64 * 1024,  // remainSizeL0BPerBlock
            512 *1024, // uint32_t sizeCodeBookL1BBuffer,
            64 * 1024, // uint32_t sizeCodeBookL0BBuffer,
            16);
    op.Process();
}
#endif