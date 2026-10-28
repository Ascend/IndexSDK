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

// CUBE_ONLY：matmul 仅在 AIC 上执行，AIV 完全不调用 matmul API。
// 这样 AIC/AIV 之间才能用裸 CrossCore flag 做显式同步（参考 ascendc_ivfpq_subspace_distance.cpp）。
#define ASCENDC_CUBE_ONLY

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/matmul.h"
#include "op_kernel_common.h"

using namespace AscendC;
using namespace matmul;
using namespace Utils;

namespace
{
constexpr uint8_t BUFFER_NUM = 1;
// AIC 将 cube 结果写入 GM workspace 后置该 flag，通知同组两个 AIV 可以读取
constexpr uint16_t L2NORM_CUBE_READY_ID = 1;
// AIV 读完本轮的 cube 结果后置该 flag，AIC 收到后才会写下一轮。
// 只有这一条反向握手才能保证「AIC 写 GM」与「AIV 读 GM」不重叠：单向 flag 下 AIV 刚通过
// wait 就开始读 32KB，而 AIC 已经在打下一轮的 fixpipe，AIV 会读到半刷新的矩阵。
// 注意 mode2 的语义是「同组两个 AIV 都 set 之后，AIC 的 wait 才放行」，因此两个次块都必须
// 置位；只由一个次块置位会让 AIC 永远等不到，表现为 add 超时（result 2010）。
constexpr uint16_t L2NORM_AIV_READ_DONE_ID = 2;
constexpr uint8_t SYNC_MODE2 = 2;
// cube 结果在 GM 上的双缓冲槽数：槽位 = coreIdx * L2NORM_CUBE_SLOT_NUM + (loopIndex & 1)
constexpr uint32_t L2NORM_CUBE_SLOT_NUM = 2;
}  // namespace

namespace IndexOps
{
class AscendcL2NormWith950
{
   public:
    __aicore__ inline AscendcL2NormWith950(const AscendcL2NormWith950TilingData &tilingData)
        : dim(tilingData.dim),
          vecCoreNum(tilingData.vecCoreNum),
          aicNum(tilingData.aicNum),
          tiling(tilingData.cubeTilingData)
    {
    }
    __aicore__ inline void Init(GM_ADDR data, GM_ADDR transfer, GM_ADDR actualSize, GM_ADDR normResult,
                                GM_ADDR usrWorkspace);

    // AIC 侧接口
    __aicore__ inline void RegisterMatmul();
    __aicore__ inline void InitCubeBuffer();
    __aicore__ inline void ProcessCube();

    // AIV 侧接口
    __aicore__ inline void InitVectorBuffer();
    __aicore__ inline void ProcessVector();

   private:
    __aicore__ inline void InitCoreIdx();
    __aicore__ inline void ParseActualSize(GM_ADDR actualSize, uint32_t &actualNum);

    __aicore__ inline void CalcComputeNumAndCoreOffset(const uint32_t &actualNum);

    __aicore__ inline void SetGlobalMemory(GM_ADDR data, GM_ADDR transfer, GM_ADDR normResult, GM_ADDR usrWorkspace);

    __aicore__ inline void ComputeCubeOneLoop(const uint32_t &loopIndex, LocalTensor<int8_t> &aMatrixLocal,
                                              uint32_t curNum);

    __aicore__ inline void VectorOneLoop(const uint32_t &loopIndex, uint32_t curNum);

    __aicore__ inline void PostProcessOneLoop(const uint32_t &loopIndex, uint32_t curNum);

    __aicore__ inline void MoveCubeOutDiagonal(LocalTensor<half> &cubeOutLocal, LocalTensor<half> &diagonalLocal);

    __aicore__ inline void SetCubeOutSlot(uint32_t loopIndex);

    __aicore__ inline void MoveTransferFromGM();

    __aicore__ inline void PrepareOnesLocal();

    __aicore__ inline void MoveAMatrixFromGmToL1(LocalTensor<int8_t> &aMatrixLocal, uint32_t loopOffset,
                                                 uint32_t curNum);

    __aicore__ inline void ComputeX2();

    __aicore__ inline void GetDiagonal(uint32_t curNum);

    __aicore__ inline void ComputeRsqrt(uint32_t curNum);

    __aicore__ inline void CopyResultFromUb2Gm(uint32_t loopOffset, uint32_t curNumRound16);

   private:
    Matmul<MatmulType<AscendC::TPosition::TSCM, CubeFormat::NZ, int8_t>,
           MatmulType<AscendC::TPosition::TSCM, CubeFormat::NZ, int8_t, true>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::NZ, half>>
        mm;
    TPipe pipe;
    TCubeTiling tiling;

    GlobalTensor<int8_t> aGlobal;
    GlobalTensor<half> outputGlobal;
    GlobalTensor<half> transferGlobal;
    GlobalTensor<half> cubeOutGlobal;
    // cubeOutGlobal 的基址（按 coreIdx 划分），每轮通过 SetCubeOutSlot 切到具体槽位
    __gm__ half *cubeOutBase{nullptr};

    TSCM<TPosition::GM, BUFFER_NUM> aMatrixQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> transferQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> diagonalQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> onsQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> cubeOutQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQue;

    LocalTensor<half> transferLocal;
    LocalTensor<half> onesLocal;
    LocalTensor<half> outputLocal;

    uint32_t dim{0};
    uint32_t computeNum{0};
    uint32_t vecCoreNum{0};
    uint32_t aicNum{0};
    uint32_t coreOffset{0};
    uint32_t coreIdx{0};
    // 受限于half的范围，matmul的结果需要缩小100倍，否则数据越界；但实际上100倍数还不够，仍然存在数据越界的情况。
    float scale{0.01};
    // 同时，matmul路随精度转换要求传入uint64_t的参数，其float->uint64_t转换流程固定为下方公式。
    const uint64_t quantScalar{static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&scale))};
};

__aicore__ inline void AscendcL2NormWith950::InitCoreIdx()
{
    // MIX 1:2模式：AIC blockIdx=0-27(即cube组号), AIV blockIdx=硬件索引
    // 统一映射到cube组号0-27用于任务分配和地址索引
    if ASCEND_IS_AIV
    {
        coreIdx = GetBlockIdx() / 2;
    }
    else
    {
        coreIdx = GetBlockIdx();
    }
}

__aicore__ inline void AscendcL2NormWith950::ParseActualSize(GM_ADDR actualSize, uint32_t &actualNum)
{
    actualNum = *(reinterpret_cast<__gm__ uint32_t *>(actualSize));
}

__aicore__ inline void AscendcL2NormWith950::CalcComputeNumAndCoreOffset(const uint32_t &actualNum)
{
    uint32_t computeNumEachCore = actualNum / vecCoreNum / tiling.M * tiling.M;
    uint32_t computeNumLastCore = actualNum - (vecCoreNum - 1) * computeNumEachCore;
    uint32_t extraComputeCoreNum = 0;
    if (computeNumLastCore > computeNumEachCore)
    {
        uint32_t extraBaseNum = computeNumLastCore - computeNumEachCore;
        extraComputeCoreNum = extraBaseNum / tiling.M;
        computeNumLastCore = computeNumLastCore - extraComputeCoreNum * tiling.M;
    }

    if (coreIdx < vecCoreNum - 1)
    {
        computeNum = (coreIdx < extraComputeCoreNum) ? (computeNumEachCore + tiling.M) : computeNumEachCore;
    }
    else
    {
        computeNum = computeNumLastCore;
    }

    if (coreIdx < extraComputeCoreNum)
    {
        coreOffset = coreIdx * (computeNumEachCore + tiling.M);
    }
    else
    {
        coreOffset = coreIdx * computeNumEachCore + extraComputeCoreNum * tiling.M;
    }
}

__aicore__ inline void AscendcL2NormWith950::SetGlobalMemory(GM_ADDR data, GM_ADDR transfer, GM_ADDR normResult,
                                                             GM_ADDR usrWorkspace)
{
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(data) + coreOffset * dim);
    transferGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(transfer));
    outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(normResult) + coreOffset);
    // GM 结果按「cube 组 × 双缓冲槽」划分，具体槽位由 SetCubeOutSlot 每轮切换
    cubeOutBase = reinterpret_cast<__gm__ half *>(usrWorkspace);
}

__aicore__ inline void AscendcL2NormWith950::SetCubeOutSlot(uint32_t loopIndex)
{
    uint32_t slot = coreIdx * L2NORM_CUBE_SLOT_NUM + (loopIndex % L2NORM_CUBE_SLOT_NUM);
    cubeOutGlobal.SetGlobalBuffer(cubeOutBase + static_cast<uint64_t>(slot) * tiling.M * tiling.M);
}

__aicore__ inline void AscendcL2NormWith950::InitCubeBuffer()
{
    // AIC 只需 L1 上的 A 矩阵缓冲
    pipe.InitBuffer(aMatrixQue, BUFFER_NUM, tiling.M * tiling.Ka * sizeof(int8_t));
}

__aicore__ inline void AscendcL2NormWith950::InitVectorBuffer()
{
    // AIV 侧 UB 缓冲
    pipe.InitBuffer(transferQue, BUFFER_NUM, tiling.M * BLOCK_HALF_NUM * sizeof(half));
    pipe.InitBuffer(diagonalQue, BUFFER_NUM, tiling.M * BLOCK_HALF_NUM * sizeof(half));
    pipe.InitBuffer(onsQue, BUFFER_NUM, tiling.M * sizeof(half));
    pipe.InitBuffer(outputQue, BUFFER_NUM, tiling.M * sizeof(half));
    pipe.InitBuffer(cubeOutQue, BUFFER_NUM, tiling.M * tiling.M * sizeof(half));
}

__aicore__ inline void AscendcL2NormWith950::Init(GM_ADDR data, GM_ADDR transfer, GM_ADDR actualSize,
                                                  GM_ADDR normResult, GM_ADDR usrWorkspace)
{
    InitCoreIdx();

    uint32_t actualNum = 0;
    ParseActualSize(actualSize, actualNum);

    CalcComputeNumAndCoreOffset(actualNum);

    // MIX模式：所有核必须调用SetGlobalMemory，即使computeNum==0，否则REGIST_MATMUL_OBJ会访问未初始化的地址
    SetGlobalMemory(data, transfer, normResult, usrWorkspace);
}

__aicore__ inline void AscendcL2NormWith950::MoveAMatrixFromGmToL1(LocalTensor<int8_t> &aMatrixLocal,
                                                                   uint32_t loopOffset, uint32_t curNum)
{
    Nd2NzParams copyParam = {
        1,                              // 传输nd矩阵的数目：1个128的nd
        static_cast<uint16_t>(curNum),  // nd矩阵的行数：输入的行数
        static_cast<uint16_t>(dim),     // nd矩阵的列数：输入的列数
        0,  // 源操作数相邻nd矩阵起始地址间的偏移：nd数目为1，该数无意义
        static_cast<uint16_t>(dim),  // 源操作数同一nd矩阵的相邻行起始地址间的偏移：内存连续的，和列数保持一致即可
        // 目的nz矩阵中，来自源操作数同一行的多行数据相邻行起始地址间的偏移：按128行对齐，每行是32B
        static_cast<uint16_t>(tiling.M),
        1,  // 目的nz矩阵中，Z型矩阵相邻行起始地址之间的偏移：一行32B
        0   // 目的nz矩阵中，相邻nz矩阵起始地址间的偏移：nz数目为1，该数无意义
    };
    DataCopy(aMatrixLocal, aGlobal[loopOffset * dim], copyParam);
    aMatrixQue.EnQue(aMatrixLocal);
}

__aicore__ inline void AscendcL2NormWith950::ComputeX2()
{
    auto aMatrixLocal = aMatrixQue.DeQue<int8_t>();
    mm.SetTensorA(aMatrixLocal);
    mm.SetTensorB(aMatrixLocal, true);
    mm.SetQuantScalar(quantScalar);
    // CUBE_ONLY：第4参数负责等待 fixpipe 写 GM 完成，MatmulImpl 无 WaitIterateAll 成员；
    // AIV 侧读取由 CrossCore flag 同步（见 ComputeCubeOneLoop/VectorOneLoop）。
    mm.IterateAll<false>(cubeOutGlobal, 0, false, true);
    // 初始化统一由 REGIST_MATMUL_OBJ 完成，循环内不调用 mm.Init。
    mm.End();
}

__aicore__ inline void AscendcL2NormWith950::GetDiagonal(uint32_t curNum)
{
    auto diagonalLocal = diagonalQue.DeQue<half>();

    // 1. 保留对角线结果，其他值置0
    diagonalLocal = diagonalLocal * transferLocal;

    // 2. 逐行求和，取对角线结果
    WholeReduceSum<half>(outputLocal, diagonalLocal, BLOCK_HALF_NUM, curNum, 1, 1, 1);
}

__aicore__ inline void AscendcL2NormWith950::ComputeRsqrt(uint32_t curNum)
{
    // 兜底：范数 = 1/sqrt(0.01*Σx^2)，Σx^2 理论上非负；一旦上游异常（例如 GM 读写竞态）让这里
    // 取到非对角元素就会出现负数，Sqrt(负) = NaN 会一路污染到 TopK；零向量又会让 Div(1,0) = +Inf。
    // 这里统一把开方输入压到最小正规格化 half 2^-14，保证输出恒为有限正数。
    const half minNorm = static_cast<half>(0.000061035156f);  // 2^-14
    Maxs(outputLocal, outputLocal, minNorm, curNum);
    Sqrt(outputLocal, outputLocal, curNum);
    Div(outputLocal, onesLocal, outputLocal, curNum);
    outputQue.EnQue(outputLocal);
}

__aicore__ inline void AscendcL2NormWith950::CopyResultFromUb2Gm(uint32_t loopOffset, uint32_t curNumRound16)
{
    outputLocal = outputQue.DeQue<half>();
    DataCopy(outputGlobal[loopOffset], outputLocal, curNumRound16);
}

__aicore__ inline void AscendcL2NormWith950::MoveCubeOutDiagonal(LocalTensor<half> &cubeOutLocal,
                                                                 LocalTensor<half> &diagonalLocal)
{
    DataCopyParams param{
        static_cast<uint16_t>(tiling.M / BLOCK_HALF_NUM),
        BLOCK_HALF_NUM,  // 固定搬运BLOCK_HALF_NUM*BLOCK_HALF_NUM/BLOCK_HALF_NUM个数，即16*16矩阵
        static_cast<uint16_t>(tiling.M),  // 每次搬运src间隔tiling.M*BLOCK_HALF_NUM/BLOCK_HALF_NUM个block
        0};
    DataCopy(diagonalLocal, cubeOutLocal, param);
    diagonalQue.EnQue(diagonalLocal);
}

__aicore__ inline void AscendcL2NormWith950::ComputeCubeOneLoop(const uint32_t &loopIndex,
                                                                LocalTensor<int8_t> &aMatrixLocal, uint32_t curNum)
{
    uint32_t loopOffset = loopIndex * tiling.M;

    // 反向握手：等同组两个 AIV 都读完上一轮（首轮令牌由 AIV 进入循环时先置出）。
    // 没有这一步时，AIC 打完本轮的 fixpipe 后置 CUBE_READY 就继续跑下一轮，
    // 同组 AIV 会读到「半刷新」的 cubeOutGlobal，掩码取对角线时可能取到 X*X^T 的
    // 非对角元素（可负），Sqrt(负) 就是 NaN（且不会出现 Inf，与现场现象一致）。
    AscendC::CrossCoreWaitFlag(L2NORM_AIV_READ_DONE_ID);

    // 双缓冲：本轮换到另一个槽位，避免覆盖 AIV 正在读的槽
    SetCubeOutSlot(loopIndex);

    // AIC：搬运本 M-tile 底库到 L1 并完成 matmul，结果由 fixpipe 写入 cubeOutGlobal
    MoveAMatrixFromGmToL1(aMatrixLocal, loopOffset, curNum);
    ComputeX2();

    // 结果已落 GM，置 flag 通知同组两个 AIV 可以读取
    AscendC::CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(L2NORM_CUBE_READY_ID);
}

__aicore__ inline void AscendcL2NormWith950::VectorOneLoop(const uint32_t &loopIndex, uint32_t curNum)
{
    // 反向握手：mode2 下 AIC 的 wait 需要「两个 AIV 都 set」才放行，因此两个次块都要置位。
    // 置位放在本轮 wait 之前（首轮即"令牌"），与 AIC 每轮一次 wait 严格 1:1 配平；
    // 此时上一轮的 PostProcessOneLoop 已返回，说明上一轮的 GM 读（PIPE_MTE2）已结束，AIC 才可安全写该槽。
    AscendC::CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE2>(L2NORM_AIV_READ_DONE_ID);

    // 两个次块都必须 wait，保证与 AIC 的 set 次数配平；只有次块1做后处理
    AscendC::CrossCoreWaitFlag(L2NORM_CUBE_READY_ID);
    if (GetSubBlockIdx() == 1)
    {
        PostProcessOneLoop(loopIndex, curNum);
    }
}

__aicore__ inline void AscendcL2NormWith950::PostProcessOneLoop(const uint32_t &loopIndex, uint32_t curNum)
{
    uint32_t loopOffset = loopIndex * tiling.M;

    // 与 AIC 用同一个槽位公式，保证读到的是本轮刚写入的结果
    SetCubeOutSlot(loopIndex);

    // 1. 从GM搬运matmul结果到UB
    auto cubeOutLocal = cubeOutQue.AllocTensor<half>();
    DataCopyParams copyParam{1, static_cast<uint16_t>(tiling.M * tiling.M / BLOCK_HALF_NUM), 0, 0};
    DataCopy(cubeOutLocal, cubeOutGlobal, copyParam);

    // 2. 按16维度的方形矩阵截取对角线
    auto diagonalLocal = diagonalQue.AllocTensor<half>();
    MoveCubeOutDiagonal(cubeOutLocal, diagonalLocal);
    cubeOutQue.FreeTensor(cubeOutLocal);

    // 3. 利用transfer获取对角线
    outputLocal = outputQue.AllocTensor<half>();
    GetDiagonal(curNum);
    diagonalQue.FreeTensor(diagonalLocal);

    // 4. 开根号求倒
    ComputeRsqrt(curNum);

    // 5. 搬运结果
    uint32_t curNumRound16 = RoundUp(curNum, ALIGN_16);
    CopyResultFromUb2Gm(loopOffset, curNumRound16);
    outputQue.FreeTensor(outputLocal);
}

__aicore__ inline void AscendcL2NormWith950::MoveTransferFromGM()
{
    DataCopy(transferLocal, transferGlobal, tiling.M * BLOCK_HALF_NUM);
    transferQue.EnQue(transferLocal);
}

__aicore__ inline void AscendcL2NormWith950::PrepareOnesLocal()
{
    const half one = 1;
    Duplicate(onesLocal, one, tiling.M);
}

__aicore__ inline void AscendcL2NormWith950::RegisterMatmul()
{
    // MIX模式：tiling必须传给REGIST_MATMUL_OBJ（第4参数），初始化由该宏完成，
    // 与 flat_ip/flat_l2 950 一致，不在计算循环内调用 mm.Init。
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
}

__aicore__ inline void AscendcL2NormWith950::ProcessCube()
{
    // AIC：逐 M-tile 完成 matmul 并写 GM，每轮结束后置 flag 唤醒同组 AIV
    if (computeNum != 0)
    {
        auto aMatrixLocal = aMatrixQue.AllocTensor<int8_t>();
        uint32_t loopNum = computeNum / tiling.M;
        uint32_t remainNum = computeNum % tiling.M;
        for (uint32_t i = 0; i < loopNum; i++)
        {
            ComputeCubeOneLoop(i, aMatrixLocal, tiling.M);
        }
        if (remainNum != 0)
        {
            ComputeCubeOneLoop(loopNum, aMatrixLocal, remainNum);
        }
        aMatrixQue.FreeTensor(aMatrixLocal);
    }
}

__aicore__ inline void AscendcL2NormWith950::ProcessVector()
{
    if (computeNum != 0)
    {
        // 次块1：准备常驻内存数据（transfer矩阵和全1向量）
        if (GetSubBlockIdx() == 1)
        {
            transferLocal = transferQue.AllocTensor<half>();
            MoveTransferFromGM();
            onesLocal = onsQue.AllocTensor<half>();
            PrepareOnesLocal();
            transferLocal = transferQue.DeQue<half>();
        }

        // 两个次块都参与 wait，与 AIC 的 set 次数配平；次块1在后处理
        uint32_t loopNum = computeNum / tiling.M;
        uint32_t remainNum = computeNum % tiling.M;
        for (uint32_t i = 0; i < loopNum; i++)
        {
            VectorOneLoop(i, tiling.M);
        }
        if (remainNum != 0)
        {
            VectorOneLoop(loopNum, remainNum);
        }

        if (GetSubBlockIdx() == 1)
        {
            transferQue.FreeTensor(transferLocal);
            onsQue.FreeTensor(onesLocal);
        }
    }

    // 与 ivfpq 一致：AIV 侧做一次全局同步，保证所有 AIV 完成后 kernel 再退出
    AscendC::SyncAll();
}
}  // namespace IndexOps

extern "C" __global__ __aicore__ void ascendc_l2_norm_with950(GM_ADDR data, GM_ADDR transfer, GM_ADDR actualSize,
                                                              GM_ADDR normResult, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr)
    {
        return;
    }

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    GET_TILING_DATA(tilingData, tiling);
    IndexOps::AscendcL2NormWith950 l2Norm(tilingData);

    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);

    // AC 分离：AIC 只跑 matmul，AIV 只做后处理，两者用裸 CrossCore flag 同步
    if ASCEND_IS_AIC
    {
        l2Norm.Init(data, transfer, actualSize, normResult, usrWorkspace);
        l2Norm.RegisterMatmul();
        l2Norm.InitCubeBuffer();
        l2Norm.ProcessCube();
    }

    if ASCEND_IS_AIV
    {
        l2Norm.Init(data, transfer, actualSize, normResult, usrWorkspace);
        l2Norm.InitVectorBuffer();
        l2Norm.ProcessVector();
    }
}
