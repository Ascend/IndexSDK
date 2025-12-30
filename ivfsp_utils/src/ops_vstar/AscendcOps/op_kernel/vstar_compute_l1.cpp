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

class KernelVstarComputeL1 {
public:
    __aicore__ inline KernelVstarComputeL1()
    {
        this->qTileSize = 16; // 数据以16位对齐
    }

    __aicore__ inline void Init(GM_ADDR x0,  // query (nq, dim)
                                GM_ADDR x1,  // codebook (nlist * subSpaceDim /16, dim /16, 16, 16)
                                GM_ADDR y0,  // qc (nq, nlist * subSpaceDim)
                                GM_ADDR y1,  // dist
                                GM_ADDR y2,  // flag
                                uint32_t subSpaceDim,
                                uint32_t dim,
                                uint32_t nq,
                                uint32_t nlist,
                                uint32_t blockSize,
                                uint32_t cbTileSizeL1,
                                uint32_t cbTileSizeB2,
                                uint32_t cbLoopsL1,
                                uint32_t cbLoopsB2)
    {
        this->subSpaceDim = subSpaceDim;       // 子空间维度
        this->nq = nq;                         //  query条数，可能不与16对齐
        this->nlist = nlist;                   // 总粗桶个数
        this->blockSize = blockSize;           // 当前计算的block内需要计算的粗桶个数
        this->dim = dim;
        this->cbTileSizeL1 = cbTileSizeL1;   // 当前tile 计算的粗通个数 * subSpaceDim
        this->cbTileSizeB2 = cbTileSizeB2;
        this->cbLoopsL1 = cbLoopsL1;
        this->cbLoopsB2 = cbLoopsB2;
        this->cbHeight = blockSize * subSpaceDim;

        auto blkIdx = GetBlockIdx();

#ifdef __CCE_KT_TEST__
        blockId=blkIdx;
printf("blockId = %d\n", blockId);
#endif
        auto cbTotalSize = cbHeight * dim;

        queryGM.SetGlobalBuffer((__gm__ float*)x0);
        codeBooksGM.SetGlobalBuffer((__gm__ half*)x1 + blkIdx * cbTotalSize);
        qcGM.SetGlobalBuffer((__gm__ half*)y0 + blkIdx * cbHeight);
        distsGM.SetGlobalBuffer((__gm__ half*)y1 + blkIdx * blockSize);
        flagsGM.SetGlobalBuffer((__gm__ uint16_t*)y2 + blkIdx * 16);

#ifdef __CCE_KT_TEST__

        printf("Init Start\n");
#endif
        pipe.InitBuffer(queryInQueL1, 1, 16 * dim * sizeof(half));
        pipe.InitBuffer(queryInQueA2, 1, 16 * dim * sizeof(half));
        pipe.InitBuffer(cbInQueL1, 1, cbTileSizeL1 * dim * sizeof(half));
        pipe.InitBuffer(cbInQueB2, 1, cbTileSizeB2 * dim * sizeof(half));
        pipe.InitBuffer(queryInUb, 1, nq * dim * sizeof(float));
        pipe.InitBuffer(queryFp16InUb, 1, nq * dim * sizeof(half));

        pipe.InitBuffer(qcShapedCo1Que, 1, qTileSize * cbTileSizeB2 * sizeof(half));
        pipe.InitBuffer(qcShapedCo2Que, 1, qTileSize * cbTileSizeL1 * sizeof(half));
        pipe.InitBuffer(qcNDCo2Que, 1, qTileSize * cbTileSizeL1 * sizeof(half));
        pipe.InitBuffer(distsQue, 1, nq * dim *sizeof(half));
        pipe.InitBuffer(flagsQue, 1, 16 * sizeof(uint16_t));

#ifdef __CCE_KT_TEST__

        printf("Init End\n");

#endif
    }

    __aicore__ inline void Process()
    {
        // step1 compute query matrix X codebook matrix
        this->dimBlocks = dim / 16;
        this->cbTileSizeL1Blocks = cbTileSizeL1 / 16;
        this->qTileSizeL1Blocks = 1;
        uint16_t cbNumL1 = cbTileSizeL1 / subSpaceDim;
        CopyQueryIn();
        pipe_barrier(PIPE_ALL);
        LocalTensor<half> a1Local = queryInQueL1.DeQue<half>();
        LocalTensor<half> distLocal = distsQue.AllocTensor<half>();
        SplitQuery(a1Local);
        queryInQueL1.FreeTensor(a1Local);
        LocalTensor<half> a2Local = queryInQueA2.DeQue<half>();
        for (int32_t m = 0; m < cbLoopsL1; m ++) {
            CopyCodeBookIn(m, cbTileSizeL1);
            LocalTensor<half> c2Local = qcShapedCo2Que.AllocTensor<half>();
            LocalTensor<half> b1Local = cbInQueL1.DeQue<half>();
            for (int32_t k = 0; k < cbLoopsB2; ++k) {
                SplitCodeBook(k, b1Local);
                Compute(a2Local);
                Aggregate(c2Local, k);
            }
            cbInQueL1.FreeTensor(b1Local);

            qcShapedCo2Que.EnQue(c2Local);

// copy to ND format and copy qc to GM
            CopyQCShaped2ND(m);
            Reduce(distLocal, m, cbNumL1);
        }
        queryInQueA2.FreeTensor(a2Local);

        distsQue.EnQue(distLocal);
        LocalTensor<half> distDeQueLocal = distsQue.DeQue<half>();
        CopyDistsOut(distDeQueLocal);
        distsQue.FreeTensor(distDeQueLocal);

        LocalTensor<uint16_t> flagsUb = flagsQue.AllocTensor<uint16_t>();
        flagsUb.SetValue(0, 1);
        DataCopy(flagsGM, flagsUb, {1, 1, 0, 0});
        flagsQue.FreeTensor(flagsUb);
    }

private:

    __aicore__ inline void CopyND2NZ(LocalTensor<half>& dst, const LocalTensor<half>& src, const uint16_t height,
                                     const uint16_t heightAlign, const uint16_t width)
    {
        for (int i = 0; i < width / 16; ++i) {
            int srcOffset = i * 16;
            int dstOffset = i * 16 * heightAlign;
            DataCopy(dst[dstOffset], src[srcOffset], { height, 1, uint16_t(width / 16 - 1), 0 });
        }
    }

/**
* CodeBook Move From GM to L1
* CodeBook is ZN Format Stored in GM
* @param splitIdx
* @return
*/
    __aicore__ inline void CopyCodeBookIn(int32_t splitIdx, int32_t copySizeL1)
    {
        int srcOffset = splitIdx * cbTileSizeL1 * dim;
        LocalTensor<half> b1Local = cbInQueL1.AllocTensor<half>(); // cbTileSizeL1 * dim * sizeof(half)
        DataCopy(b1Local, codeBooksGM[srcOffset], { 1, uint16_t(copySizeL1 * dim / 16), 0, 0 });
        cbInQueL1.EnQue(b1Local);
    }

/**
* copy query data from gm to A1(L1)
* transform ND format to NZ Format
* a1Local's shape: [queryAlign, dim] with NZ Format
* @return
*/
    __aicore__ inline void CopyQueryIn()
    {
        int numElement = nq * dim;
        {
            LocalTensor<float> queryFp32  = queryInUb.AllocTensor<float>();
            DataCopy(queryFp32, queryGM, {1, uint16_t(numElement * sizeof(float)/ 32), 0, 0});
            queryInUb.EnQue(queryFp32);
        }
        {
            LocalTensor<float> queryFp32  = queryInUb.DeQue<float>();
            LocalTensor<half> queryFp16  = queryFp16InUb.AllocTensor<half>();
            Cast(queryFp16, queryFp32, RoundMode::CAST_NONE, numElement);
            queryFp16InUb.EnQue(queryFp16);
            queryInUb.FreeTensor(queryFp32);
        }
        {
            LocalTensor<half> queryFp16  = queryFp16InUb.DeQue<half>();
            LocalTensor<half> a1Local = queryInQueL1.AllocTensor<half>(); // qTileSize *  dim
            CopyND2NZ(a1Local, queryFp16, nq, qTileSize, uint16_t(dim));
            queryInQueL1.EnQue(a1Local);
            queryFp16InUb.FreeTensor(queryFp16);
        }
    }

/**
* split query and copy query data from A1(L1) to A2(L0A Buffer)
* transform NZ format to Zz Format
* @param splitIdx
* @return
*/
    __aicore__ inline void SplitQuery(const LocalTensor<half>& a1Local)
    {
        LocalTensor<half> a2Local = queryInQueA2.AllocTensor<half>(); // qTileSizeA2 * dim * sizeof(half) Byte Size
        // transform nz to zz
        LoadData2dParams loadDataParams;
        loadDataParams.repeatTimes = dimBlocks;
        loadDataParams.srcStride = 1;
        loadDataParams.ifTranspose = false;

        LoadData(a2Local, a1Local, loadDataParams);

        queryInQueA2.EnQue<half>(a2Local);
    }

    __aicore__ inline void SplitCodeBook(int32_t splitIdx, const LocalTensor<half>& b1Local)
    {
        LocalTensor<half> b2Local = cbInQueB2.AllocTensor<half>(); // cbTileSizeB2 * dim * sizeof(half) Byte Size
        LoadData2dParams loadDataParams;
        loadDataParams.repeatTimes = cbTileSizeB2 / 16;
        loadDataParams.srcStride = dimBlocks;
        loadDataParams.ifTranspose = false;
        int srcOffset = splitIdx * cbTileSizeB2 * dim;
        int dstOffset = 0;
        for (int i = 0; i < dimBlocks; i ++) {
            LoadData(b2Local[dstOffset], b1Local[srcOffset], loadDataParams);
            srcOffset += 256;
            dstOffset += cbTileSizeB2 * 16;
        }
        cbInQueB2.EnQue<half>(b2Local);
    }

/**
* Copy Query Code From CO2 To GM
* @return
*/

    __aicore__ inline void CopyQCShaped2ND(int32_t cbLoopL1Idx)
    {
        LocalTensor<half> qcShapedCo2Local = qcShapedCo2Que.DeQue<half>();
        LocalTensor<half> qcNDCo2Local = qcNDCo2Que.AllocTensor<half>();
        for (int i = 0; i < cbTileSizeL1Blocks; ++i) {
            DataCopy(qcNDCo2Local[i * CUBE_ALIGN], qcShapedCo2Local[i * CUBE_ALIGN * CUBE_ALIGN],
                     { uint16_t(nq), 1, 0, uint16_t(cbTileSizeL1 * sizeof(half) / 32 - 1)});
        }
        qcNDCo2Que.EnQue(qcNDCo2Local);

        int32_t nb1 = nlist * subSpaceDim;
        int32_t dstOffset = cbLoopL1Idx * cbTileSizeL1;
        uint16_t dstStride = nb1 * sizeof(half) / 32 - 1;
        for (int i = 0; i < cbTileSizeL1Blocks; ++i) {
            DataCopy(qcGM[dstOffset + i * CUBE_ALIGN],
                     qcShapedCo2Local[i * CUBE_ALIGN * CUBE_ALIGN], { uint16_t(nq), 1, 0, dstStride });
        }
        qcShapedCo2Que.FreeTensor(qcShapedCo2Local);
    }

    __aicore__ inline void CopyDistsOut(LocalTensor<half>& distsUb)
    {
        uint16_t blockLen = uint16_t(blockSize * sizeof(half) / 32);
        uint16_t dstStride = nlist * sizeof(half) / 32 - blockLen;
        DataCopy(distsGM, distsUb, {uint16_t(nq), blockLen, 0, dstStride });
    }

/**
* Move data From CO1 to CO2, Data Format from NZ to ND
* @param c2Local
* @param bSplitIdx
* @param dstOffset
* @return
*/
    __aicore__ inline void Aggregate(const LocalTensor<half>& c2Local, const int bSplitIdx)
    {
        LocalTensor<half> c1Local = qcShapedCo1Que.DeQue<half>();
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = qTileSize * cbTileSizeB2 / 256;
        DataCopyEnhancedParams enhancedParams;
        enhancedParams.blockMode = BlockMode::BLOCK_MODE_MATRIX;

        int dstOffset = qTileSize * cbTileSizeB2 * bSplitIdx;

        DataCopy(c2Local[dstOffset], c1Local, dataCopyParams, enhancedParams);

        qcShapedCo1Que.FreeTensor(c1Local);
    }

    __aicore__ inline void Compute(const LocalTensor<half>& a2Local)
    {
        LocalTensor<half> b2Local = cbInQueB2.DeQue<half>();
        LocalTensor<half> c1Local = qcShapedCo1Que.AllocTensor<half>();

        Mmad(c1Local, a2Local, b2Local,
             { uint16_t(qTileSize), uint16_t(cbTileSizeB2), uint16_t(dim), false, 0, false, false, false });

        qcShapedCo1Que.EnQue<half>(c1Local);
        cbInQueB2.FreeTensor(b2Local);
    }

    __aicore__ inline void Reduce(LocalTensor<half>& distLocal, const int m, const int cbNumL1)
    {
        LocalTensor<half> c2LocalND = qcNDCo2Que.DeQue<half>();
        Mul(c2LocalND, c2LocalND, c2LocalND, nq * cbTileSizeL1); // 二级接口

        for (int i = 0; i < nq; i ++) {
            int srcOffset = i * cbTileSizeL1;
            int distOffset = m * cbNumL1 + i * blockSize;

            int wholeReduceSize = subSpaceDim * sizeof(half);
            int wholeReduceNum = subSpaceDim;
            int wholeReduceMaxSize = 256;
            int wholeReduceMaxNum = wholeReduceMaxSize / sizeof(half);
            int wholeReduceMinSize = 32;
            int wholeReduceMinNum = wholeReduceMinSize / sizeof(half);
            int maxRepeatTimes = 255;
            while (wholeReduceNum > wholeReduceMaxNum) {
                int curRepeatTimes = (cbTileSizeL1 / subSpaceDim) * wholeReduceNum / wholeReduceMinNum;
                int j;
                for (j=0; curRepeatTimes > maxRepeatTimes; j++) {
                    WholeReduceSum<half>(c2LocalND[srcOffset + j * maxRepeatTimes],
                        c2LocalND[srcOffset + j * maxRepeatTimes * wholeReduceMinNum],
                        wholeReduceMinNum, maxRepeatTimes, 1, 1, wholeReduceMinNum * sizeof(half) / wholeReduceMinSize);
                    curRepeatTimes -= maxRepeatTimes;
                }
                WholeReduceSum<half>(c2LocalND[srcOffset+ j * maxRepeatTimes],
                    c2LocalND[srcOffset + j * maxRepeatTimes * wholeReduceMinNum],
                    wholeReduceMinNum, curRepeatTimes, 1, 1, wholeReduceMinNum * sizeof(half) / wholeReduceMinSize);
                wholeReduceNum /= wholeReduceMinNum;
            }
            WholeReduceSum<half>(distLocal[distOffset], c2LocalND[srcOffset], wholeReduceNum,
                (cbTileSizeL1 / subSpaceDim), 1, 1, wholeReduceNum * sizeof(half) / wholeReduceMinSize);
        }

        qcNDCo2Que.FreeTensor(c2LocalND);
    }

private:
    TPipe pipe;

    TQue<QuePosition::A1, 1> queryInQueL1;
    TQue<QuePosition::A2, 1> queryInQueA2;
    TQue<QuePosition::B1, 1> cbInQueL1;
    TQue<QuePosition::B2, 1> cbInQueB2;
    TQue<QuePosition::VECIN, 1> queryInUb;
    TQue<QuePosition::VECOUT, 1> queryFp16InUb;

// dst queue
    TQue<QuePosition::CO1, 1> qcShapedCo1Que;
    TQue<QuePosition::CO2, 1> qcShapedCo2Que;
    TQue<QuePosition::CO2, 1> qcNDCo2Que;
    TQue<QuePosition::VECOUT, 1> flagsQue; // outQueueCO2Reduce: dists + flag
    TQue<QuePosition::VECOUT, 1> distsQue;

    GlobalTensor<float> queryGM;
    GlobalTensor<half> codeBooksGM;
    GlobalTensor<half> qcGM;
    GlobalTensor<half> distsGM;
    GlobalTensor<uint16_t> flagsGM;

    uint32_t subSpaceDim;
    uint32_t nlist;
    uint32_t nq;
    uint32_t blockSize;
    uint32_t dim;
    uint32_t cbTileSizeL1;
    uint32_t cbLoopsL1;
    uint32_t cbTileSizeB2;
    uint32_t cbLoopsB2;
    uint32_t cbHeight;
    uint16_t qTileSize;

    int32_t dimBlocks;
    int32_t qTileSizeL1Blocks;
    int32_t cbTileSizeL1Blocks;

#ifdef __CCE_KT_TEST__
    uint32_t blockId;
#endif
};

#ifndef __CCE_KT_TEST__
#ifdef __ONLINE_TEST__
extern "C" __global__ __aicore__ void vstar_compute_l1(GM_ADDR x0, GM_ADDR x1, GM_ADDR y0, GM_ADDR y1, GM_ADDR y2) {
    KernelVstarComputeL1 op;
// subSpaceDim,dim,nq,nlist,blockSize,cbTileSizeL1,cbTileSizeB2,cbLoopsL1,cbLoopsB2
op.Init(x0, x1, y0, y1, y2, 32, 256, 1, 1024, 128, 32, 16, 128, 2);
op.Process();
}

void vstar_compute_l1_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* x0, uint8_t* x1, uint8_t* y0,
                         uint8_t* y1, uint8_t* y2)
{
    vstar_compute_l1<<<blockDim, l2ctrl, stream>>>(x0, x1, y0, y1, y2);
}
#else
extern "C" __global__ __aicore__ void vstar_compute_l1(GM_ADDR x0, GM_ADDR x1, GM_ADDR y0, GM_ADDR y1, GM_ADDR y2,
                                                       GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelVstarComputeL1 op;
// subSpaceDim,dim,nq,nlist,blockSize,cbTileSizeL1,cbTileSizeB2,cbLoopsL1,cbLoopsB2
    op.Init(x0, x1, y0, y1, y2, tiling_data.subSpaceDim, tiling_data.dim, tiling_data.nq, tiling_data.nlist,
            tiling_data.blockSize, tiling_data.cbTileSizeL1, tiling_data.cbTileSizeB2, tiling_data.cbLoopsL1,
            tiling_data.cbLoopsB2);
    op.Process();
}
#endif
#else
extern "C" __global__ __aicore__ void vstar_compute_l1(GM_ADDR x0, GM_ADDR x1, GM_ADDR y0, GM_ADDR y1, GM_ADDR y2) {
    KernelVstarComputeL1 op;
// subSpaceDim,dim,nq,nlist,blockSize,cbTileSizeL1,cbTileSizeB2,cbLoopsL1,cbLoopsB2
op.Init(x0, x1, y0, y1, y2, 32, 256, 1, 1024, 128, 512, 64, 8, 8);
op.Process();
}
#endif
