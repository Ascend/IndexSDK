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

class VSM3 {
public:
    __aicore__ inline VSM3()
    {
    }

    __aicore__ inline void Init(GM_ADDR queryCode,
                                GM_ADDR codeWord,
                                GM_ADDR l2Indices,
                                GM_ADDR diff1, GM_ADDR diff2, GM_ADDR precompute,
                                GM_ADDR mask, GM_ADDR outDists, GM_ADDR opFlag, GM_ADDR vcMin,
                                uint32_t subDim1,
                                uint32_t subDim2,
                                uint32_t nlist1,
                                uint32_t nlist2,
                                uint32_t n,
                                uint32_t nprobe2,
                                uint32_t segmentSize,
                                uint32_t segSizeVcMin,
                                uint32_t tmpMaskSize,
                                uint32_t segmentNum,
                                uint32_t formerBlkNum,
                                uint32_t probePerBlockFormer,
                                uint32_t probePerBlockLatter,
                                uint32_t sizeCodeWordUBBuffer,
                                uint32_t sizeCodeWordL0BBuffer,
                                uint32_t sizeCodeWordL1BBuffer,
                                uint32_t cubeAlign,
                                uint32_t blockDim)
    {
        this->n = n;
        this->nlist1 = nlist1;
        this->nlist2 = nlist2;
        this->subDim1 = subDim1;
        this->subDim2 = subDim2;
        this->nprobe2 = nprobe2;
        this->segmentSize = segmentSize;
        this->segSizeVcMin = segSizeVcMin;
        this->segmentNum = segmentNum;
        this->formerBlkNum = formerBlkNum;
        this->probePerBlockFormer = probePerBlockFormer;
        this->probePerBlockLatter = probePerBlockLatter;
        this->sizeCodeWordUBBuffer = sizeCodeWordUBBuffer;
        this->sizeCodeWordL0BBuffer = sizeCodeWordL0BBuffer / bufferNumL0B;
        ;
        this->sizeCodeWordL1BBuffer = sizeCodeWordL1BBuffer;

        this->blkIdx = GetBlockIdx();
        this->probePerBlock = (blkIdx < formerBlkNum ? probePerBlockFormer : probePerBlockLatter);
        this->cubeAlign = cubeAlign;
        this->blockDim = blockDim;

        this->tmpMaskSize = tmpMaskSize;

        queryCodeGM.SetGlobalBuffer((__gm__ half *)queryCode);
        codeWordGM.SetGlobalBuffer((__gm__ uint8_t *)codeWord);
        // 存储信息顺序为：bucketId、AccuSegmentNum、codeWordOffset、precomputeOffset
        l2IndicesGM.SetGlobalBuffer((__gm__ uint64_t *)l2Indices);

        outDistsGM.SetGlobalBuffer((__gm__ half *)outDists);
        diff1GM.SetGlobalBuffer((__gm__ half *)diff1);
        diff2GM.SetGlobalBuffer((__gm__ half *)diff2);
        precomputeGM.SetGlobalBuffer((__gm__ half *)precompute);

        opFlagGM.SetGlobalBuffer((__gm__ uint16_t *)opFlag + blkIdx * 16);
        vcMinGM.SetGlobalBuffer((__gm__ half *)vcMin);
        maskGlobal.SetGlobalBuffer((__gm__ uint8_t *)mask);

        pipe.InitBuffer(inQueueA1, 1, cubeAlign * subDim2 * sizeof(half));
        pipe.InitBuffer(inQueueA2, 1, cubeAlign * subDim2 * sizeof(half));
        pipe.InitBuffer(inQueueB1, 1, sizeCodeWordL1BBuffer);
        pipe.InitBuffer(inQueueB2, bufferNumL0B, sizeCodeWordL0BBuffer);
        pipe.InitBuffer(outQueueCO1, bufferNumL0B,
                        cubeAlign * ((sizeCodeWordL0BBuffer / subDim2 / sizeof(half))) * sizeof(half));
        pipe.InitBuffer(outQueueCO2, 1, ((sizeCodeWordL0BBuffer / subDim2 / sizeof(half))) * sizeof(half));

        pipe.InitBuffer(inQueueDiff, 1, 2 * subDim2 * sizeof(half));
        pipe.InitBuffer(inQueueQuery, 1, 2 * subDim2 * sizeof(half));
        pipe.InitBuffer(inQueueQueryMulDiff2, 1, subDim2 * sizeof(half));

        pipe.InitBuffer(inQueuePrecompute, 1, ((sizeCodeWordL0BBuffer / subDim2 / sizeof(half))) * sizeof(half));
        pipe.InitBuffer(outQueueCompute, 1, ((sizeCodeWordL0BBuffer / subDim2 / sizeof(half))) * sizeof(half));
        pipe.InitBuffer(inQueueCodeWordU8, 1, sizeCodeWordUBBuffer);
        pipe.InitBuffer(inQueueCodeWordFp16, 1, 2 * sizeCodeWordUBBuffer);
        pipe.InitBuffer(inQueueL2Indices, 1, 6 * nprobe2 * sizeof(uint64_t));
        pipe.InitBuffer(outQueueFlag, 1, 16 * sizeof(uint16_t));

        pipe.InitBuffer(outQueueVcMin, 1, 2 * segmentNum * segmentSize * sizeof(half) / segSizeVcMin);

        pipe.InitBuffer(maskQueue, 1, ((sizeCodeWordL0BBuffer / subDim2 / sizeof(half))) * sizeof(uint8_t));
        pipe.InitBuffer(maskTmpQueue, 1, tmpMaskSize);
    }

    __aicore__ inline void Process()
    {
        CopyDiff1Diff2();
        LocalTensor<half> diffLocal = inQueueDiff.DeQue<half>();
        for (int32_t queryIdx = 0; queryIdx < n; queryIdx++) {
            int32_t loopCount = probePerBlock;
            int32_t endFlag = 0;

            Copyl2Indices(queryIdx);
            pipe_barrier(PIPE_ALL);
            LocalTensor<uint64_t> l2IndicesLocal = inQueueL2Indices.DeQue<uint64_t>();
            for (int32_t loop = 0; loop < loopCount; loop++) {
                if (endFlag == 1)
                    break;
                int32_t l2IndicesIdx = blockDim * loop + blkIdx;

                int32_t bucketIdx = l2IndicesLocal.GetValue(l2IndicesIdx) / nlist2; // 粗桶id
                uint64_t codeWordBeginOffset = l2IndicesLocal.GetValue(nprobe2 * 2 + 2 * l2IndicesIdx);
                uint64_t codeWordEndOffset = l2IndicesLocal.GetValue(nprobe2 * 2 + 2 * l2IndicesIdx + 1);
                uint64_t precomputeBeginOffset = l2IndicesLocal.GetValue(nprobe2 * 4 + 2 * l2IndicesIdx);
                int32_t preL2IndicesIdxAccuSegmentNum = l2IndicesIdx > 0 ?
                    l2IndicesLocal.GetValue(nprobe2 + l2IndicesIdx - 1) : 0;
                int32_t curL2IndicesIdxAccuSegmentNum = l2IndicesLocal.GetValue(nprobe2 + l2IndicesIdx);
                //              判断是否是此query在该核上应处理的最后一个子桶
                if (loop == loopCount - 1 || curL2IndicesIdxAccuSegmentNum >= segmentNum ||
                    preL2IndicesIdxAccuSegmentNum >= segmentNum) {
                    inQueueL2Indices.FreeTensor(l2IndicesLocal);
                    if (preL2IndicesIdxAccuSegmentNum >= segmentNum)
                        break;
                    endFlag = 1;
                    if (curL2IndicesIdxAccuSegmentNum >= segmentNum) {
                        codeWordEndOffset = (segmentNum - preL2IndicesIdxAccuSegmentNum) * segmentSize * subDim2 +
                            codeWordBeginOffset;
                    }
                }
                if (codeWordEndOffset == codeWordBeginOffset)
                    continue;
                //              基于前面的计算得到了真正需要算的endOffset和beginOffset以及粗桶id bucketIdx
                int32_t codeWordNum = (codeWordEndOffset - codeWordBeginOffset) / subDim2;
                int32_t codeWordSize = codeWordNum * subDim2 * sizeof(uint8_t);
                int32_t moveTimesCurProbe = DivUp(codeWordSize, sizeCodeWordUBBuffer);
                int32_t tailSize = codeWordSize % sizeCodeWordUBBuffer;
                if (tailSize == 0)
                    tailSize = sizeCodeWordUBBuffer;

                pipe_barrier(PIPE_ALL);
                CopyQueryInVec(queryIdx, bucketIdx);
                pipe_barrier(PIPE_ALL);
                MatQueryDiff1Diff2(diffLocal);
                pipe_barrier(PIPE_ALL);
                SumQueryDiff2();
                pipe_barrier(PIPE_ALL);
                LocalTensor<half> queryMulDiff2Local = inQueueQueryMulDiff2.DeQue<half>();
                pipe_barrier(PIPE_ALL);
                CopyInA();
                pipe_barrier(PIPE_ALL);
                LocalTensor<half> a1Local = inQueueA1.DeQue<half>();

                for (int32_t moveTimes = 0; moveTimes < moveTimesCurProbe; moveTimes++) {
                    int32_t curCopySize = (moveTimes == moveTimesCurProbe - 1 ? tailSize : sizeCodeWordUBBuffer);
                    int32_t curCopyNum = curCopySize / subDim2 / sizeof(uint8_t);
                    int32_t curComputeSize = 2 * curCopySize;

                    CopyPrecompute(precomputeBeginOffset, moveTimes, curCopyNum);
                    pipe_barrier(PIPE_ALL);
                    uint64_t maskBeginOffset = precomputeBeginOffset;
                    CopyMask(maskBeginOffset, moveTimes, curCopyNum);
                    pipe_barrier(PIPE_ALL);

                    PrecomputeAddScalar(queryMulDiff2Local, curCopyNum);
                    pipe_barrier(PIPE_ALL);
                    CopyCodeWord(codeWordBeginOffset, moveTimes, curCopySize);
                    pipe_barrier(PIPE_ALL);
                    CastCodeWordFp16(curCopySize);
                    pipe_barrier(PIPE_ALL);

                    SplitA(a1Local);
                    pipe_barrier(PIPE_ALL);
                    CopyInB(curComputeSize);
                    pipe_barrier(PIPE_ALL);
                    SplitB(curComputeSize);
                    pipe_barrier(PIPE_ALL);

                    MatMulCompute(curComputeSize);
                    pipe_barrier(PIPE_ALL);
                    Aggregate(curComputeSize);
                    pipe_barrier(PIPE_ALL);
                    AddPrecompute(curCopyNum);
                    pipe_barrier(PIPE_ALL);

                    uint64_t dstOffset = (queryIdx * segmentNum + preL2IndicesIdxAccuSegmentNum) * segmentSize +
                        moveTimes * (sizeCodeWordUBBuffer / subDim2);
                    // 外部申请了2倍内存，这里queryIdx对应的segment需要同步修改偏移两倍
                    uint64_t vmDstOffset = (queryIdx * segmentNum * 2 + preL2IndicesIdxAccuSegmentNum) * segmentSize +
                        moveTimes * (sizeCodeWordUBBuffer / subDim2);
                    ComputeAndCopyOut(dstOffset, vmDstOffset, curCopyNum);
                    pipe_barrier(PIPE_ALL);
                }
                inQueueQueryMulDiff2.FreeTensor(queryMulDiff2Local);
                inQueueA1.FreeTensor(a1Local);
            }
        }
        CopyFlagOut();
        inQueueDiff.FreeTensor(diffLocal);
    }

private:
    __aicore__ inline int32_t DivUp(int32_t x, int32_t y)
    {
        return (x + y - 1) / y;
    }

    __aicore__ inline void CopyDiff1Diff2()
    {
        LocalTensor<half> diffLocal = inQueueDiff.AllocTensor<half>();
        DataCopy(diffLocal[0], diff1GM[0], subDim2);
        DataCopy(diffLocal[subDim2], diff2GM[0], subDim2);
        inQueueDiff.EnQue(diffLocal);
    }

    __aicore__ inline void Copyl2Indices(const int32_t queryIdx)
    {
        LocalTensor<uint64_t> l2IndicesLocal = inQueueL2Indices.AllocTensor<uint64_t>();
        DataCopy(l2IndicesLocal, l2IndicesGM[queryIdx * nprobe2 * 6], nprobe2 * 6);
        inQueueL2Indices.EnQue(l2IndicesLocal);
    }

    __aicore__ inline void CopyQueryInVec(const int32_t queryIdx, const int32_t bucketIdx)
    {
        LocalTensor<half> queryLocal = inQueueQuery.AllocTensor<half>();
        DataCopy(queryLocal, queryCodeGM[queryIdx * subDim1 * nlist1 + bucketIdx * subDim1], subDim2);
        inQueueQuery.EnQue(queryLocal);
    }

    __aicore__ inline void MatQueryDiff1Diff2(LocalTensor<half> &diffLocal)
    {
        LocalTensor<half> queryLocal = inQueueQuery.DeQue<half>();
        LocalTensor<half> queryMulDiff2Local = inQueueQueryMulDiff2.AllocTensor<half>();
        Mul(queryMulDiff2Local, queryLocal[0], diffLocal[subDim2], subDim2);
        inQueueQuery.EnQue<half>(queryLocal);
        LocalTensor<half> queryLocal2 = inQueueQuery.DeQue<half>();
        Mul(queryLocal2[subDim2], queryLocal2[0], diffLocal[0], subDim2);
        inQueueQuery.EnQue<half>(queryLocal2);
        inQueueQueryMulDiff2.EnQue<half>(queryMulDiff2Local);
    }

    __aicore__ inline void SumQueryDiff2()
    {
        LocalTensor<half> queryMulDiff2Local = inQueueQueryMulDiff2.DeQue<half>();
        WholeReduceSum<half>(queryMulDiff2Local, queryMulDiff2Local, subDim2, 1, 1, 1, 0);
        inQueueQueryMulDiff2.EnQue<half>(queryMulDiff2Local);
    }

    __aicore__ inline void CopyPrecompute(const uint64_t codeWordBeginOffset, const int32_t moveTimes,
                                          const int32_t curCopyNum)
    {
        uint64_t src_offset = codeWordBeginOffset + moveTimes * (sizeCodeWordUBBuffer / subDim2);
        LocalTensor<half> precomputeLocal = inQueuePrecompute.AllocTensor<half>();
        DataCopy(precomputeLocal, precomputeGM[src_offset], curCopyNum);
        inQueuePrecompute.EnQue<half>(precomputeLocal);
    }

    __aicore__ inline void CopyMask(const uint64_t maskBeginOffset, const int32_t moveTimes, const int32_t curCopyNum)
    {
        uint64_t src_offset = maskBeginOffset + moveTimes * (sizeCodeWordUBBuffer / subDim2);
        LocalTensor<uint8_t> maskLocal = maskQueue.AllocTensor<uint8_t>();
        DataCopy(maskLocal, maskGlobal[src_offset], curCopyNum);
        maskQueue.EnQue<uint8_t>(maskLocal);
    }

    __aicore__ inline void PrecomputeAddScalar(LocalTensor<half> &queryMulDiff2Local, const int32_t curCopyNum)
    {
        LocalTensor<half> precomputeLocal = inQueuePrecompute.DeQue<half>();
        Adds(precomputeLocal, precomputeLocal, queryMulDiff2Local.GetValue(0), curCopyNum);
        inQueuePrecompute.EnQue<half>(precomputeLocal);
    }

    __aicore__ inline void CopyCodeWord(const uint64_t codeWordBeginOffset, const int32_t moveTimes,
                                        const int32_t curCopySize)
    {
        uint64_t src_offset = codeWordBeginOffset + moveTimes * sizeCodeWordUBBuffer;
        LocalTensor<uint8_t> codeWordU8Local = inQueueCodeWordU8.AllocTensor<uint8_t>();
        DataCopy(codeWordU8Local, codeWordGM[src_offset], curCopySize / sizeof(uint8_t));
        inQueueCodeWordU8.EnQue<uint8_t>(codeWordU8Local);
    }

    __aicore__ inline void CastCodeWordFp16(const int32_t curCopySize)
    {
        LocalTensor<half> codeWordFp16Local = inQueueCodeWordFp16.AllocTensor<half>();
        LocalTensor<uint8_t> codeWordU8Local = inQueueCodeWordU8.DeQue<uint8_t>();
        Cast(codeWordFp16Local, codeWordU8Local, RoundMode::CAST_NONE, curCopySize / sizeof(uint8_t));

        inQueueCodeWordFp16.EnQue<half>(codeWordFp16Local);
        inQueueCodeWordU8.FreeTensor(codeWordU8Local);
    }

    __aicore__ inline void CopyInA()
    {
        LocalTensor<half> queryLocal = inQueueQuery.DeQue<half>();
        LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
        DataCopy(a1Local, queryLocal[subDim2], {static_cast<uint16_t>(subDim2 / 16), 1, 0, 15});

        inQueueA1.EnQue(a1Local);
        inQueueQuery.FreeTensor(queryLocal);
    }

    __aicore__ inline void SplitA(LocalTensor<half> &a1Local)
    {
        LocalTensor<half> a2LocalVSM = inQueueA2.AllocTensor<half>();

        LoadData2dParams loadDataParams;
        loadDataParams.repeatTimes = subDim2 / 16;
        loadDataParams.srcStride = 1;
        loadDataParams.ifTranspose = false;
        LoadData(a2LocalVSM, a1Local, loadDataParams);
        inQueueA2.EnQue<half>(a2LocalVSM);
    }

    __aicore__ inline void CopyInB(const int32_t curComputeSize)
    {
        LocalTensor<half> b1LocalVSM = inQueueB1.AllocTensor<half>();
        LocalTensor<half> codeWordFp16Local = inQueueCodeWordFp16.DeQue<half>();
        DataCopy(b1LocalVSM, codeWordFp16Local, curComputeSize / sizeof(half));

        inQueueCodeWordFp16.FreeTensor(codeWordFp16Local);
        inQueueB1.EnQue(b1LocalVSM);
    }

    __aicore__ inline void SplitB(const int32_t curComputeSize)
    {
        int32_t srcOffset = 0;
        int32_t dstOffset = 0;
        int32_t column_block = curComputeSize / sizeof(half) / subDim2 / CUBE_ALIGN;
        LocalTensor<half> b2LocalVSM = inQueueB2.AllocTensor<half>();
        LocalTensor<half> b1LocalVSM = inQueueB1.DeQue<half>();
        for (int32_t i = 0; i < subDim2 / CUBE_ALIGN; ++i) {
            LoadData2dParams loadDataParams;
            loadDataParams.repeatTimes = column_block;
            loadDataParams.srcStride = subDim2 / CUBE_ALIGN;
            loadDataParams.ifTranspose = false;

            LoadData(b2LocalVSM[dstOffset], b1LocalVSM[srcOffset], loadDataParams);
            srcOffset += CUBE_ALIGN * CUBE_ALIGN;
            dstOffset += CUBE_ALIGN * CUBE_ALIGN * column_block;
        }

        inQueueB1.FreeTensor(b1LocalVSM);
        inQueueB2.EnQue<half>(b2LocalVSM);
    }

    __aicore__ inline void MatMulCompute(const int32_t curComputeSize)
    {
        uint16_t mComputeSize = 16;
        uint16_t nComputeSize = curComputeSize / sizeof(half) / subDim2;
        uint16_t kComputeSize = subDim2;
        LocalTensor<half> b2LocalVSM = inQueueB2.DeQue<half>();
        LocalTensor<half> c1LocalVSM = outQueueCO1.AllocTensor<half>();
        LocalTensor<half> a2LocalVSM = inQueueA2.DeQue<half>();

        Mmad(c1LocalVSM, a2LocalVSM, b2LocalVSM,
            {mComputeSize, nComputeSize, kComputeSize, false, 0, false, false, false});

        inQueueB2.FreeTensor(b2LocalVSM);
        outQueueCO1.EnQue<half>(c1LocalVSM);
        inQueueA2.FreeTensor(a2LocalVSM);
    }

    __aicore__ inline void Aggregate(const int32_t curComputeSize)
    {
        LocalTensor<half> c1Local = outQueueCO1.DeQue<half>();
        LocalTensor<half> c2Local = outQueueCO2.AllocTensor<half>();
        int32_t column_block = curComputeSize / sizeof(half) / subDim2 / 16;

        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = column_block;
        DataCopyEnhancedParams enhancedParams;
        enhancedParams.blockMode = BlockMode::BLOCK_MODE_VECTOR;

        DataCopy(c2Local, c1Local, dataCopyParams, enhancedParams);

        outQueueCO1.FreeTensor(c1Local);
        outQueueCO2.EnQue<half>(c2Local);
    }

    __aicore__ inline void AddPrecompute(const int32_t curCopyNum)
    {
        LocalTensor<half> c2Local = outQueueCO2.DeQue<half>();
        LocalTensor<half> precomputeLocal = inQueuePrecompute.DeQue<half>();
        LocalTensor<half> computeLocal = outQueueCompute.AllocTensor<half>();
        Add(computeLocal, c2Local, precomputeLocal, curCopyNum);

        outQueueCO2.FreeTensor(c2Local);
        inQueuePrecompute.FreeTensor(precomputeLocal);
        outQueueCompute.EnQue<half>(computeLocal);
    }

    __aicore__ inline void ComputeAndCopyOut(const uint64_t dstOffset, const uint64_t vmDstOffset,
                                             const int32_t curCopyNum)
    {
        LocalTensor<half> computeLocalVSM = outQueueCompute.DeQue<half>();
        LocalTensor<half> vcMinLocalVSM = outQueueVcMin.AllocTensor<half>();

        SelectWithBytesMaskShapeInfo info;
        info.firstAxis = 1;
        info.srcLastAxis = curCopyNum;
        info.maskLastAxis = curCopyNum;

        LocalTensor<uint8_t> maskLocal = maskQueue.DeQue<uint8_t>();
        LocalTensor<uint8_t> maskTmpLocal = maskTmpQueue.AllocTensor<uint8_t>();
        SelectWithBytesMask(computeLocalVSM, maskScalar, computeLocalVSM, maskLocal, maskTmpLocal, info);
        pipe_barrier(PIPE_ALL);
        maskQueue.FreeTensor(maskLocal);
        maskTmpQueue.FreeTensor(maskTmpLocal);

        uint32_t curCopyVcMinSize = sizeof(half) * curCopyNum * 2 / segSizeVcMin;
        uint32_t curCopyVcMinSizeAlign = 32 * ((curCopyVcMinSize + 31) / 32);
        const half zero = 0.0f;
        if (curCopyVcMinSize != curCopyVcMinSizeAlign) {
            Duplicate(vcMinLocalVSM, zero, curCopyVcMinSizeAlign / sizeof(half));
        }
        pipe_barrier(PIPE_ALL);

        WholeReduceMin<half>(vcMinLocalVSM, computeLocalVSM, segSizeVcMin, curCopyNum / segSizeVcMin, 1, 1,
                             segSizeVcMin * sizeof(half) / 32); // WholeRediceMin需要32bit对齐
        pipe_barrier(PIPE_ALL);
        for (int i = 0; i < curCopyVcMinSizeAlign / sizeof(half); i++) {
            float vcMinTemp = vcMinLocalVSM.GetValue(i);
            if (!(vcMinTemp >= -65504.0f && vcMinTemp <= 65504.0f)) {
                Duplicate(vcMinLocalVSM, zero, curCopyVcMinSizeAlign / sizeof(half));
                break;
            }
        }
        // 原子加
        SetAtomicAdd<half>();
        DataCopy(vcMinGM[vmDstOffset * 2 / segSizeVcMin], vcMinLocalVSM, curCopyVcMinSizeAlign / sizeof(half));
        SetAtomicNone();
        pipe_barrier(PIPE_ALL);

        DataCopy(outDistsGM[dstOffset], computeLocalVSM, curCopyNum);

        outQueueVcMin.FreeTensor(vcMinLocalVSM);
        outQueueCompute.FreeTensor(computeLocalVSM);
    }

    __aicore__ inline void CopyFlagOut()
    {
        LocalTensor<uint16_t> outflagUbVSM = outQueueFlag.AllocTensor<uint16_t>();
        outflagUbVSM.SetValue(0, 1);
        DataCopy(opFlagGM, outflagUbVSM, 16);
        outQueueFlag.FreeTensor(outflagUbVSM);
    }

private:
    static const uint32_t bufferNumL0B = 1;
    TPipe pipe;
    TQue<QuePosition::A1, 1> inQueueA1;
    TQue<QuePosition::A2, 1> inQueueA2;
    TQue<QuePosition::B1, 1> inQueueB1;
    TQue<QuePosition::B2, bufferNumL0B> inQueueB2;
    // dst queue
    TQue<QuePosition::CO1, bufferNumL0B> outQueueCO1;
    TQue<QuePosition::CO2, 1> outQueueCO2;

    TQue<QuePosition::VECIN, 1> inQueueDiff;
    TQue<QuePosition::VECIN, 1> inQueueQuery;
    TQue<QuePosition::VECIN, 1> inQueueQueryMulDiff2;
    TQue<QuePosition::VECIN, 1> inQueuePrecompute;
    TQue<QuePosition::VECOUT, 1> outQueueCompute;
    TQue<QuePosition::VECIN, 1> inQueueCodeWordU8;
    TQue<QuePosition::VECOUT, 1> inQueueCodeWordFp16;

    TQue<QuePosition::VECOUT, 1> inQueueL2Indices;
    TQue<QuePosition::VECIN, 1> outQueueFlag;
    TQue<QuePosition::VECOUT, 1> outQueueVcMin;

    TQue<QuePosition::VECIN, 1> maskQueue;
    TQue<QuePosition::VECIN, 1> maskTmpQueue;

    GlobalTensor<half> queryCodeGM;
    GlobalTensor<uint8_t> codeWordGM;
    GlobalTensor<uint64_t> l2IndicesGM;
    GlobalTensor<half> outDistsGM;
    GlobalTensor<half> diff1GM;
    GlobalTensor<half> diff2GM;
    GlobalTensor<half> precomputeGM;
    GlobalTensor<uint16_t> opFlagGM;
    GlobalTensor<half> vcMinGM;
    GlobalTensor<uint8_t> maskGlobal;

    uint32_t n;
    uint32_t nlist1;
    uint32_t nlist2;
    uint32_t subDim1;
    uint32_t subDim2;

    uint32_t nprobe2;
    uint32_t baseNum;
    uint32_t segmentSize;
    uint32_t segmentNum;

    uint32_t formerBlkNum;
    uint32_t probePerBlockFormer;
    uint32_t probePerBlockLatter;
    uint32_t sizeCodeWordUBBuffer;
    uint32_t sizeCodeWordL0BBuffer;
    uint32_t sizeCodeWordL1BBuffer;

    uint32_t blkIdx;
    uint32_t probePerBlock;
    uint32_t cubeAlign;
    uint32_t blockDim;
    uint32_t segSizeVcMin;

    uint32_t tmpMaskSize;
    half maskScalar{100.0};
};

#ifndef __CCE_KT_TEST__
extern "C" __global__ __aicore__ void vsm3(GM_ADDR queries, GM_ADDR shaped, GM_ADDR l2Indices, GM_ADDR diff1,
                                           GM_ADDR diff2, GM_ADDR precompute, GM_ADDR mask, GM_ADDR attr_nlistl1,
                                           GM_ADDR attr_nlistl2, GM_ADDR attr_segmentl3, GM_ADDR dist,
                                           GM_ADDR opflag, GM_ADDR vcmin, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    VSM3 op;
    op.Init(queries, shaped, l2Indices, diff1, diff2, precompute, mask, dist, opflag, vcmin,
            tiling_data.subDim1,
            tiling_data.subDim2,
            tiling_data.nlist1,
            tiling_data.nlist2,
            tiling_data.n,
            tiling_data.nprobe2,
            tiling_data.segmentSize,
            tiling_data.segSizeVcMin,
            tiling_data.tmpMaskSize,
            tiling_data.segmentNum,
            tiling_data.formerBlkNum,
            tiling_data.probePerBlockFormer,
            tiling_data.probePerBlockLatter,
            tiling_data.sizeCodeWordUBBuffer,
            tiling_data.sizeCodeWordL0BBuffer,
            tiling_data.sizeCodeWordL1BBuffer,
            tiling_data.cubeAlign,
            tiling_data.blockDim);
    op.Process();
}
#else
extern "C" __global__ __aicore__ void vsm3(GM_ADDR queryCode,
                                                GM_ADDR codeWord,
                                                GM_ADDR l2Indices,
                                                GM_ADDR diff1, GM_ADDR diff2, GM_ADDR precompute, GM_ADDR mask,
                                                GM_ADDR attr_nlistl1, GM_ADDR attr_nlistl2, GM_ADDR attr_segmentl3,
                                                GM_ADDR outDists, GM_ADDR opFlag, GM_ADDR vcMin)
{
    VSM3 op;
    op.Init(queryCode, codeWord, l2Indices, diff1, diff2, precompute, mask, outDists, opFlag, vcMin,
            32,
            16,
            1024,
            32,
            8,
            128,
            64,
            64,
            32,
            0,
            16,
            16,
            32 * 1024,
            64 * 1024,
            64 * 1024,
            16,
            8);
    op.Process();
}
#endif