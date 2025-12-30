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
using namespace Utils;

namespace {
constexpr uint8_t BUFFER_NUM = 1;
}

namespace IndexOps {
class AscendcL2Norm {
public:
    __aicore__ inline AscendcL2Norm(const AscendcL2NormTilingData &tilingData)
        : dim(tilingData.dim), vecCoreNum(tilingData.vecCoreNum), tiling(tilingData.cubeTilingData) {}
    __aicore__ inline void Init(GM_ADDR data, GM_ADDR transfer, GM_ADDR actualSize, GM_ADDR normResult);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseActualSize(GM_ADDR actualSize, uint32_t &actualNum);

    __aicore__ inline void CalcComputeNumAndCoreOffset(const uint32_t &actualNum);

    __aicore__ inline void SetGlobalMemory(GM_ADDR data, GM_ADDR transfer, GM_ADDR normResult);

    __aicore__ inline void InitBuffer();

    __aicore__ inline void ComputeOneLoop(const uint32_t &loopIndex, LocalTensor<int8_t> &aMatrixLocal,
        uint32_t curNum);

    __aicore__ inline void MoveCubeOutDiagonal(LocalTensor<half> &cubeOutLocal, LocalTensor<half> &diagonalLocal);

    __aicore__ inline void MoveTransferFromGM();

    __aicore__ inline void PrepareOnesLocal();

    __aicore__ inline void CalcResultInLoops(LocalTensor<int8_t> &aMatrixLocal);

    __aicore__ inline void MoveAMatrixFromGmToL1(LocalTensor<int8_t> &aMatrixLocal, uint32_t loopOffset,
        uint32_t curNum);

    __aicore__ inline void ComputeX2(LocalTensor<half> &cubeOutLocal);

    __aicore__ inline void GetDiagonal(uint32_t curNum);

    __aicore__ inline void ComputeRsqrt(uint32_t curNum);

    __aicore__ inline void CopyResultFromUb2Gm(uint32_t loopOffset, uint32_t curNumRound16);

private:
    Matmul<MatmulType<AscendC::TPosition::TSCM, CubeFormat::NZ, int8_t>,
           MatmulType<AscendC::TPosition::TSCM, CubeFormat::NZ, int8_t, true>,
           MatmulType<AscendC::TPosition::LCM, CubeFormat::NZ, half>> mm;
    TPipe pipe;
    TCubeTiling tiling;

    GlobalTensor<int8_t> aGlobal;
    GlobalTensor<half> outputGlobal;
    GlobalTensor<half> transferGlobal;

    TSCM<TPosition::GM, BUFFER_NUM> aMatrixQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> transferQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> diagonalQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> onsQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> cubeOutQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQue;

    LocalTensor<half> transferLocal;
    LocalTensor<half> onesLocal;
    LocalTensor<half> outputLocal;

    uint32_t dim {0};
    uint32_t computeNum {0};
    uint32_t vecCoreNum {0};
    uint32_t coreOffset {0};
    uint32_t coreIdx {static_cast<uint32_t>(GetBlockIdx())};
    // 受限于half的范围，matmul的结果需要缩小100倍，否则数据越界；但实际上100倍数还不够，仍然存在数据越界的情况。
    float scale {0.01};
    // 同时，matmul路随精度转换要求传入uint64_t的参数，其float->uint64_t转换流程固定为下方公式。
    const uint64_t quantScalar {static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&scale))};
};


__aicore__ inline void AscendcL2Norm::ParseActualSize(GM_ADDR actualSize, uint32_t &actualNum)
{
    actualNum = *(reinterpret_cast<__gm__ uint32_t*>(actualSize));
}

__aicore__ inline void AscendcL2Norm::CalcComputeNumAndCoreOffset(const uint32_t &actualNum)
{
    uint32_t computeNumEachCore = actualNum / vecCoreNum / tiling.M * tiling.M;
    uint32_t computeNumLastCore = actualNum - (vecCoreNum - 1) * computeNumEachCore;
    uint32_t extraComputeCoreNum = 0;
    if (computeNumLastCore > computeNumEachCore) {
        uint32_t extraBaseNum = computeNumLastCore - computeNumEachCore;
        extraComputeCoreNum = extraBaseNum / tiling.M;
        computeNumLastCore = computeNumLastCore - extraComputeCoreNum * tiling.M;
    }

    if (coreIdx < vecCoreNum - 1) {
        computeNum = (coreIdx < extraComputeCoreNum) ? (computeNumEachCore + tiling.M) : computeNumEachCore;
    } else {
        computeNum = computeNumLastCore;
    }

    if (coreIdx < extraComputeCoreNum) {
        coreOffset = coreIdx * (computeNumEachCore + tiling.M);
    } else {
        coreOffset = coreIdx * computeNumEachCore + extraComputeCoreNum * tiling.M;
    }
}

__aicore__ inline void AscendcL2Norm::SetGlobalMemory(GM_ADDR data, GM_ADDR transfer, GM_ADDR normResult)
{
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(data) + coreOffset * dim);
    transferGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(transfer));
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(normResult) + coreOffset);
}

__aicore__ inline void AscendcL2Norm::InitBuffer()
{
    pipe.InitBuffer(transferQue, BUFFER_NUM, tiling.M * BLOCK_HALF_NUM * sizeof(half));
    pipe.InitBuffer(aMatrixQue, BUFFER_NUM, tiling.M * tiling.Ka * sizeof(int8_t));
    pipe.InitBuffer(diagonalQue, BUFFER_NUM, tiling.M * BLOCK_HALF_NUM * sizeof(half));
    pipe.InitBuffer(onsQue, BUFFER_NUM, tiling.M * sizeof(half));
    pipe.InitBuffer(outputQue, BUFFER_NUM, tiling.M * sizeof(half));
    pipe.InitBuffer(cubeOutQue, BUFFER_NUM, tiling.M * tiling.M * sizeof(half));
}

__aicore__ inline void AscendcL2Norm::Init(GM_ADDR data, GM_ADDR transfer, GM_ADDR actualSize, GM_ADDR normResult)
{
    uint32_t actualNum = 0;
    ParseActualSize(actualSize, actualNum);

    CalcComputeNumAndCoreOffset(actualNum);

    if (computeNum == 0) {
        return;
    }

    SetGlobalMemory(data, transfer, normResult);

    InitBuffer();
}

__aicore__ inline void AscendcL2Norm::MoveAMatrixFromGmToL1(LocalTensor<int8_t> &aMatrixLocal,
    uint32_t loopOffset, uint32_t curNum)
{
    Nd2NzParams copyParam = {
        1, // 传输nd矩阵的数目：1个128的nd
        static_cast<uint16_t>(curNum), // nd矩阵的行数：输入的行数
        static_cast<uint16_t>(dim), // nd矩阵的列数：输入的列数
        0, // 源操作数相邻nd矩阵起始地址间的偏移：nd数目为1，该数无意义
        static_cast<uint16_t>(dim), // 源操作数同一nd矩阵的相邻行起始地址间的偏移：内存连续的，和列数保持一致即可
        // 目的nz矩阵中，来自源操作数同一行的多行数据相邻行起始地址间的偏移：按128行对齐，每行是32B
        static_cast<uint16_t>(tiling.M),
        1, // 目的nz矩阵中，Z型矩阵相邻行起始地址之间的偏移：一行32B
        0 // 目的nz矩阵中，相邻nz矩阵起始地址间的偏移：nz数目为1，该数无意义
    };
    DataCopy(aMatrixLocal, aGlobal[loopOffset * dim], copyParam);
    aMatrixQue.EnQue(aMatrixLocal);
}

__aicore__ inline void AscendcL2Norm::ComputeX2(LocalTensor<half> &cubeOutLocal)
{
    // 先申请cube的结果矩阵
    auto aMatrixLocal = aMatrixQue.DeQue<int8_t>();
    mm.SetTensorA(aMatrixLocal);
    mm.SetTensorB(aMatrixLocal, true);
    mm.IterateAll(cubeOutLocal);
}

__aicore__ inline void AscendcL2Norm::GetDiagonal(uint32_t curNum)
{
    auto diagonalLocal = diagonalQue.DeQue<half>();

    // 1. 保留对角线结果，其他值置0
    diagonalLocal = diagonalLocal * transferLocal;

    // 2. 逐行求和，取对角线结果
    WholeReduceSum<half>(outputLocal, diagonalLocal, BLOCK_HALF_NUM, curNum, 1, 1, 1);
}

__aicore__ inline void AscendcL2Norm::ComputeRsqrt(uint32_t curNum)
{
    Sqrt(outputLocal, outputLocal, curNum);
    Div(outputLocal, onesLocal, outputLocal, curNum);
    outputQue.EnQue(outputLocal);
}

__aicore__ inline void AscendcL2Norm::CopyResultFromUb2Gm(uint32_t loopOffset, uint32_t curNumRound16)
{
    outputLocal = outputQue.DeQue<half>();
    DataCopy(outputGlobal[loopOffset], outputLocal, curNumRound16);
}

__aicore__ inline void AscendcL2Norm::MoveCubeOutDiagonal(LocalTensor<half> &cubeOutLocal,
    LocalTensor<half> &diagonalLocal)
{
    DataCopyParams param {
        static_cast<uint16_t>(tiling.M / BLOCK_HALF_NUM),
        BLOCK_HALF_NUM, // 固定搬运BLOCK_HALF_NUM*BLOCK_HALF_NUM/BLOCK_HALF_NUM个数，即16*16矩阵
        static_cast<uint16_t>(tiling.M), // 每次搬运src间隔tiling.M*BLOCK_HALF_NUM/BLOCK_HALF_NUM个block
        0
    };
    DataCopy(diagonalLocal, cubeOutLocal, param);
    diagonalQue.EnQue(diagonalLocal);
}

__aicore__ inline void AscendcL2Norm::ComputeOneLoop(const uint32_t &loopIndex, LocalTensor<int8_t> &aMatrixLocal,
    uint32_t curNum)
{
    uint32_t loopOffset = loopIndex * tiling.M;

    // 1. 搬运本次计算底库至L1
    MoveAMatrixFromGmToL1(aMatrixLocal, loopOffset, curNum);

    // 2. matmul计算x^2
    auto cubeOutLocal = cubeOutQue.AllocTensor<half>();
    ComputeX2(cubeOutLocal);

    // 3. 按16维度的方形矩阵截取对角线
    auto diagonalLocal = diagonalQue.AllocTensor<half>();
    MoveCubeOutDiagonal(cubeOutLocal, diagonalLocal);
    cubeOutQue.FreeTensor(cubeOutLocal);

    // 4. 利用transfer获取对角线
    outputLocal = outputQue.AllocTensor<half>();
    GetDiagonal(curNum);
    diagonalQue.FreeTensor(diagonalLocal);

    // 5. 开根号求倒
    ComputeRsqrt(curNum);

    // 6. 搬运结果
    uint32_t curNumRound16 = RoundUp(curNum, ALIGN_16);
    CopyResultFromUb2Gm(loopOffset, curNumRound16);
    outputQue.FreeTensor(outputLocal);
}

__aicore__ inline void AscendcL2Norm::MoveTransferFromGM()
{
    DataCopy(transferLocal, transferGlobal, tiling.M * BLOCK_HALF_NUM);
    transferQue.EnQue(transferLocal);
}

__aicore__ inline void AscendcL2Norm::PrepareOnesLocal()
{
    const half one = 1;
    Duplicate(onesLocal, one, tiling.M);
}

__aicore__ inline void AscendcL2Norm::CalcResultInLoops(LocalTensor<int8_t> &aMatrixLocal)
{
    uint32_t loopNum = computeNum / tiling.M;
    uint32_t remainNum = computeNum % tiling.M;
    uint32_t loopOffset = 0;
    for (uint32_t i = 0; i < loopNum; i++) {
        ComputeOneLoop(i, aMatrixLocal, tiling.M);
    }
    if (remainNum != 0) {
        ComputeOneLoop(loopNum, aMatrixLocal, remainNum);
    }
}

__aicore__ inline void AscendcL2Norm::Process()
{
    // 1. 先注册matmul，后再处理无效计算量，否则AIC可能会卡住
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm);
    if (computeNum == 0) {
        return;
    }
    mm.Init(&tiling);
    mm.SetQuantScalar(quantScalar);

    // 2. 常驻内存数据的搬运与赋值
    transferLocal = transferQue.AllocTensor<half>();
    MoveTransferFromGM();
    onesLocal = onsQue.AllocTensor<half>();
    PrepareOnesLocal();
    transferLocal = transferQue.DeQue<half>();

    // 3. 分片计算
    auto aMatrixLocal = aMatrixQue.AllocTensor<int8_t>();
    CalcResultInLoops(aMatrixLocal);

    // 4. 释放内存
    mm.End();
    transferQue.FreeTensor(transferLocal);
    onsQue.FreeTensor(onesLocal);
    aMatrixQue.FreeTensor(aMatrixLocal);
}
}

extern "C" __global__ __aicore__ void ascendc_l2_norm(GM_ADDR data, GM_ADDR transfer, GM_ADDR actualSize,
    GM_ADDR normResult, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);
    IndexOps::AscendcL2Norm l2Norm(tilingData);
    if ASCEND_IS_AIV {
        l2Norm.Init(data, transfer, actualSize, normResult);
    }
    l2Norm.Process();
}