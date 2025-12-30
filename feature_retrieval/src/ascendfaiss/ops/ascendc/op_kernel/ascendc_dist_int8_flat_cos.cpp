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
constexpr uint32_t FLAG_SIZE = 16;
constexpr uint32_t BURST_SIZE = 64;
constexpr uint32_t BURST_RESULT_SIZE = 2;
constexpr uint32_t BURST_RESULT_RATIO = BURST_SIZE / BURST_RESULT_SIZE;
constexpr uint32_t VIC_REPEAT_MAX = 255;
constexpr uint32_t ALIGN_128 = 128;
constexpr uint16_t VALID_FLAG = 1;
constexpr uint8_t TOTAL_BASE_OFFSET_IDX = 1;
}

namespace IndexOps {
class AscendcDistInt8FlatCos {
public:
    __aicore__ inline AscendcDistInt8FlatCos(const AscendcDistInt8FlatCosTilingData &tilingData)
        : queryNum(tilingData.queryNum), dim(tilingData.dim), blockSize(tilingData.baseBlockSize),
          vecCoreNum(tilingData.vecCoreNum), tiling(tilingData.cubeTilingData),
          onceComputeBaseNum(tilingData.onceComputeBaseNum) {}
    // 分配内存、设置输入输出地址等
    __aicore__ inline void Init(GM_ADDR queryData, GM_ADDR mask, GM_ADDR baseData, GM_ADDR queryNormData,
        GM_ADDR baseNormData, GM_ADDR actualSize, GM_ADDR result, GM_ADDR resultMax, GM_ADDR flag);
    // 内存搬运、核心接口调用等
    __aicore__ inline void Process();

    __aicore__ inline void SetFlag();

private:
    __aicore__ inline void ParseActualSize(GM_ADDR actualSize, uint32_t &actualNum);

    __aicore__ inline void CalcComputeNumAndCoreOffset(const uint32_t &actualNum);

    __aicore__ inline void SetGlobalMemory(GM_ADDR queryData, GM_ADDR mask, GM_ADDR baseData, GM_ADDR queryNormData,
        GM_ADDR baseNormData, GM_ADDR result, GM_ADDR resultMax, GM_ADDR flag);

    __aicore__ inline void InitBuffer();

    __aicore__ inline void CalcConstants();

    __aicore__ inline void SetFlagGlobalMemory(GM_ADDR flag);

    __aicore__ inline void ComputeOneLoop(const uint32_t &loopIndex, const uint32_t &curBaseNumAlign16,
        const uint32_t &curBaseNumAlign128, const bool &calcNext);

    __aicore__ inline void MoveAMatrixFromGM(LocalTensor<int8_t> &aMatrixLocal);

    __aicore__ inline void MoveQueryNormFromGM();

    __aicore__ inline void CalcResultInLoops();

    __aicore__ inline void MoveMatmulRetFromGM();

    __aicore__ inline void DoMask(const uint32_t &loopOffset);

    __aicore__ inline void CalcAB(const uint32_t &bMatrixOffset);

    __aicore__ inline void CalcABFirstTime(LocalTensor<int8_t> &aMatrixLocal);

    __aicore__ inline void MoveBaseNormFromGM(LocalTensor<half> &baseNormLocal, const uint32_t &loopOffset,
        const uint32_t &curBaseNumAlign16);

    __aicore__ inline void CalcResult(const uint32_t &curBaseNumAlign128);

    __aicore__ inline void CalcResultMax(LocalTensor<half> &resultMaxLocal);

    __aicore__ inline void MoveResultToGM(const uint32_t &loopOffset, const uint32_t &curBaseNumAlign16);

    __aicore__ inline void MoveResultMaxToGM(const uint32_t &loopOffset);

private:
    Matmul<MatmulType<AscendC::TPosition::TSCM, CubeFormat::NZ, int8_t>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, int8_t, true>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>> mm;
    TCubeTiling tiling;
    TPipe pipe;

    GlobalTensor<int8_t> aGlobal;
    GlobalTensor<int8_t> bGlobal;
    GlobalTensor<half> queryNormGlobal;
    GlobalTensor<half> baseNormGlobal;
    GlobalTensor<half> resultGlobal;
    GlobalTensor<half> resultMaxGlobal;
    GlobalTensor<uint16_t> flagGlobal;
    GlobalTensor<half> cubeOutGlobal;
    GlobalTensor<uint8_t> maskGlobal;

    TSCM<TPosition::GM, BUFFER_NUM> aMatrixQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> queryNormQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> baseNormQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> maskQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> resultQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> resultMaxQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> normProductQue;

    LocalTensor<half> resultLocal;
    LocalTensor<half> queryNormLocal;

    uint32_t dim {0};
    uint32_t queryNum {0};
    uint32_t queryNumAlign16 {0};
    uint32_t computeNum {0};
    uint32_t coreOffset {0};
    uint32_t blockSize {0};
    uint32_t coreIdx {static_cast<uint32_t>(GetBlockIdx())};
    uint32_t onceComputeBaseNum {512}; // 默认按实测512性能最佳
    uint32_t totalBaseOffset {0};
    uint32_t maskLen {0};
    uint32_t maskFlag {0};
    uint32_t vecCoreNum {0};
    // 受限于half的范围，matmul的结果需要缩小100倍，否则数据越界；但实际上100倍数还不够，仍然存在数据越界的情况。
    float scale {0.01};
    // 同时，matmul路随精度转换要求传入uint64_t的参数，其float->uint64_t转换流程固定为下方公式。
    const uint64_t quantScalar {static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&scale))};

    // 预计算值，避免反复计算
    uint32_t validResultNum {0};
    uint32_t validResultCopyBlocks {0};
    uint32_t validResultVecRepeats {0};

    uint32_t reduceMaxLoops {0};
    uint32_t reduceMaxRemainder {0};
    uint32_t reduceMaxLoopDstOffset {0};
    uint32_t reduceMaxLoopSrcOffset {0};
    uint32_t reduceMaxRemainderDstOffset {0};
    uint32_t reduceMaxRemainderSrcOffset {0};

    uint32_t resultMaxCopyBlockLen {0};
    uint32_t resultMaxCopyDstStride {0};

    uint32_t onceComputeBaseNumMaskLen {0};
    uint32_t onceComputeBaseNumAlign16 {0};
    uint32_t onceComputeBaseNumAlign128 {0};
};

__aicore__ inline void AscendcDistInt8FlatCos::ParseActualSize(GM_ADDR actualSize, uint32_t &actualNum)
{
    actualNum = *(reinterpret_cast<__gm__ uint32_t *>(actualSize));
    totalBaseOffset = *(reinterpret_cast<__gm__ uint32_t *>(actualSize) + TOTAL_BASE_OFFSET_IDX);
    maskLen = *(reinterpret_cast<__gm__ uint32_t *>(actualSize) + MASK_LEN_IDX);
    maskFlag = *(reinterpret_cast<__gm__ uint32_t *>(actualSize) + MASK_FLAG_IDX);
}

__aicore__ inline void AscendcDistInt8FlatCos::CalcComputeNumAndCoreOffset(const uint32_t &actualNum)
{
    uint32_t computeNumEachCore = actualNum / vecCoreNum / onceComputeBaseNum * onceComputeBaseNum;
    uint32_t computeNumLastCore = actualNum - (vecCoreNum - 1) * computeNumEachCore;
    uint32_t extraComputeCoreNum = 0;
    if (computeNumLastCore > computeNumEachCore) {
        uint32_t extraBaseNum = computeNumLastCore - computeNumEachCore;
        extraComputeCoreNum = extraBaseNum / onceComputeBaseNum;
        computeNumLastCore = computeNumLastCore - extraComputeCoreNum * onceComputeBaseNum;
    }

    if (coreIdx < vecCoreNum - 1) {
        computeNum = (coreIdx < extraComputeCoreNum) ? (computeNumEachCore + onceComputeBaseNum) : computeNumEachCore;
    } else {
        computeNum = computeNumLastCore;
    }

    if (coreIdx < extraComputeCoreNum) {
        coreOffset = coreIdx * (computeNumEachCore + onceComputeBaseNum);
    } else {
        coreOffset = coreIdx * computeNumEachCore + extraComputeCoreNum * onceComputeBaseNum;
    }
}

__aicore__ inline void AscendcDistInt8FlatCos::SetFlagGlobalMemory(GM_ADDR flag)
{
    flagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(flag) + coreIdx * FLAG_SIZE);
}

__aicore__ inline void AscendcDistInt8FlatCos::SetGlobalMemory(GM_ADDR queryData, GM_ADDR mask, GM_ADDR baseData,
    GM_ADDR queryNormData, GM_ADDR baseNormData, GM_ADDR result, GM_ADDR resultMax, GM_ADDR flag)
{
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(queryData));
    maskGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(mask) +
        (totalBaseOffset + coreOffset) / MASK_BIT_NUM);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(baseData) + coreOffset * dim);
    queryNormGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(queryNormData));
    baseNormGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(baseNormData) + coreOffset);
    resultGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(result) + coreOffset);
    resultMaxGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(resultMax) + coreOffset / BURST_RESULT_RATIO);
    SetFlagGlobalMemory(flag);
    GM_ADDR workSpace = GetSysWorkSpacePtr();
    GM_ADDR userWorkSpace = GetUserWorkspace(workSpace);
    if (userWorkSpace == nullptr) {
        return;
    }
    cubeOutGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(userWorkSpace) +
        coreIdx * tiling.M * onceComputeBaseNum);
}

__aicore__ inline void AscendcDistInt8FlatCos::InitBuffer()
{
    pipe.InitBuffer(aMatrixQue, BUFFER_NUM, tiling.M * tiling.Ka * sizeof(int8_t));
    pipe.InitBuffer(queryNormQue, BUFFER_NUM, tiling.M * sizeof(half));
    pipe.InitBuffer(resultQue, BUFFER_NUM, onceComputeBaseNum * tiling.M * sizeof(half));
    pipe.InitBuffer(baseNormQue, BUFFER_NUM, onceComputeBaseNum * sizeof(half));
    pipe.InitBuffer(normProductQue, BUFFER_NUM, onceComputeBaseNum * sizeof(half));
    pipe.InitBuffer(resultMaxQue, BUFFER_NUM, validResultNum / BURST_RESULT_RATIO * sizeof(half));
    pipe.InitBuffer(maskQue, BUFFER_NUM, queryNum * onceComputeBaseNum / MASK_BIT_NUM * sizeof(uint8_t));
}

__aicore__ inline void AscendcDistInt8FlatCos::CalcConstants()
{
    validResultNum = onceComputeBaseNum * queryNum;
    validResultCopyBlocks = validResultNum / BLOCK_HALF_NUM;
    validResultVecRepeats = validResultNum / VIC_HALF_FULL_MASK;

    uint32_t reduceMaxTimes = validResultNum / BURST_SIZE;
    reduceMaxLoops = reduceMaxTimes / VIC_REPEAT_MAX;
    reduceMaxRemainder = reduceMaxTimes % VIC_REPEAT_MAX;
    reduceMaxLoopDstOffset = VIC_REPEAT_MAX * BURST_RESULT_SIZE;
    reduceMaxLoopSrcOffset = VIC_REPEAT_MAX * BURST_SIZE;
    reduceMaxRemainderDstOffset = reduceMaxLoops * reduceMaxLoopDstOffset;
    reduceMaxRemainderSrcOffset = reduceMaxLoops * reduceMaxLoopSrcOffset;

    resultMaxCopyBlockLen = onceComputeBaseNum / BURST_RESULT_RATIO / BLOCK_HALF_NUM;
    resultMaxCopyDstStride = (blockSize - onceComputeBaseNum) / BURST_RESULT_RATIO / BLOCK_HALF_NUM;

    onceComputeBaseNumMaskLen = onceComputeBaseNum / MASK_BIT_NUM;
    onceComputeBaseNumAlign16 = RoundUp(onceComputeBaseNum, ALIGN_16);
    onceComputeBaseNumAlign128 = RoundUp(onceComputeBaseNum, ALIGN_128);
}

__aicore__ inline void AscendcDistInt8FlatCos::Init(GM_ADDR queryData, GM_ADDR mask, GM_ADDR baseData,
    GM_ADDR queryNormData, GM_ADDR baseNormData, GM_ADDR actualSize, GM_ADDR result, GM_ADDR resultMax, GM_ADDR flag)
{
    uint32_t actualNum = 0;
    ParseActualSize(actualSize, actualNum);

    CalcComputeNumAndCoreOffset(actualNum);

    if (computeNum == 0) {
        SetFlagGlobalMemory(flag);
        return;
    }

    CalcConstants();

    SetGlobalMemory(queryData, mask, baseData, queryNormData, baseNormData, result, resultMax, flag);

    InitBuffer();
}

__aicore__ inline void AscendcDistInt8FlatCos::CalcAB(const uint32_t &bMatrixOffset)
{
    mm.SetTensorB(bGlobal[bMatrixOffset], true);
    mm.IterateAll<false>(cubeOutGlobal, 0, false, true);
}

__aicore__ inline void AscendcDistInt8FlatCos::MoveBaseNormFromGM(LocalTensor<half> &baseNormLocal,
    const uint32_t &loopOffset, const uint32_t &curBaseNumAlign16)
{
    DataCopyParams param {
        1, // 1片连续内存
        static_cast<uint16_t>(curBaseNumAlign16 / BLOCK_HALF_NUM),
        0,
        0
    };
    DataCopy(baseNormLocal, baseNormGlobal[loopOffset], param);
    baseNormQue.EnQue(baseNormLocal);
}

__aicore__ inline void AscendcDistInt8FlatCos::CalcResult(const uint32_t &baseNumAlign128)
{
    // 1和8均表示内存是连续的
    UnaryRepeatParams mulsParam {
        1,
        1,
        8,
        8
    };
    BinaryRepeatParams mulParam {
        1,
        1,
        1,
        8,
        8,
        8
    };
    auto baseNormLocal = baseNormQue.DeQue<half>();
    auto normProductLocal = normProductQue.AllocTensor<half>();
    uint8_t repeatTime = baseNumAlign128 / VIC_HALF_FULL_MASK;
    for (uint32_t i = 0; i < queryNum; i++) {
        Muls(normProductLocal, baseNormLocal, queryNormLocal.GetValue(i), VIC_HALF_FULL_MASK, repeatTime, mulsParam);
        uint32_t offset = i * onceComputeBaseNum;
        Mul(resultLocal[offset], resultLocal[offset], normProductLocal, VIC_HALF_FULL_MASK, repeatTime, mulParam);
    }
    normProductQue.FreeTensor(normProductLocal);
}

__aicore__ inline void AscendcDistInt8FlatCos::CalcResultMax(LocalTensor<half> &resultMaxLocal)
{
    for (uint32_t i = 0; i < reduceMaxLoops; i++) {
        WholeReduceMax(resultMaxLocal[i * reduceMaxLoopDstOffset],
                       resultLocal[i * reduceMaxLoopSrcOffset],
                       BURST_SIZE, // 每64算一次
                       VIC_REPEAT_MAX, // 最多255次
                       1, // dst每次4B，内存连续
                       1, // src单次256B内存连续
                       4); // 每次平移64个数，64*2/32=4
    }
    WholeReduceMax(resultMaxLocal[reduceMaxRemainderDstOffset],
                   resultLocal[reduceMaxRemainderSrcOffset],
                   BURST_SIZE, // 每64算一次
                   reduceMaxRemainder, // 剩余次数
                   1, // dst每次4B，内存连续
                   1, // src单次256B内存连续
                   4); // 每次平移64个数，64*2/32=4
    resultMaxQue.EnQue(resultMaxLocal);
    resultQue.EnQue(resultLocal);
}

__aicore__ inline void AscendcDistInt8FlatCos::MoveResultToGM(const uint32_t &loopOffset,
    const uint32_t &curBaseNumAlign16)
{
    resultLocal = resultQue.DeQue<half>();
    DataCopyParams copyPara {
        static_cast<uint16_t>(queryNum),
        static_cast<uint16_t>(curBaseNumAlign16 / BLOCK_HALF_NUM),
        static_cast<uint16_t>((onceComputeBaseNum - curBaseNumAlign16) / BLOCK_HALF_NUM),
        static_cast<uint16_t>((blockSize - curBaseNumAlign16) / BLOCK_HALF_NUM),
    };
    DataCopy(resultGlobal[loopOffset], resultLocal, copyPara);
}

__aicore__ inline void AscendcDistInt8FlatCos::MoveResultMaxToGM(const uint32_t &loopOffset)
{
    auto resultMaxLocal = resultMaxQue.DeQue<half>();
    DataCopyParams copyPara {
        static_cast<uint16_t>(queryNum),
        static_cast<uint16_t>(resultMaxCopyBlockLen),
        0,
        static_cast<uint16_t>(resultMaxCopyDstStride),
    };
    DataCopy(resultMaxGlobal[loopOffset / BURST_RESULT_RATIO], resultMaxLocal, copyPara);
}

__aicore__ inline void AscendcDistInt8FlatCos::MoveMatmulRetFromGM()
{
    mm.WaitIterateAll();
    DataCopyParams copyPara {
        1, // 1片连续内存
        static_cast<uint16_t>(validResultCopyBlocks),
        0,
        0,
    };
    DataCopy(resultLocal, cubeOutGlobal, copyPara);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
}

__aicore__ inline void AscendcDistInt8FlatCos::DoMask(const uint32_t &loopOffset)
{
    if (maskFlag == 0) {
        return;
    }
    auto maskLocal = maskQue.AllocTensor<uint8_t>();
    // 因为maskLen并不是32字节对齐的，因此只能使用for
    for (uint32_t i = 0; i < queryNum; i++) {
        DataCopy(maskLocal[i * onceComputeBaseNumMaskLen], maskGlobal[i * static_cast<uint64_t>(maskLen) +
            loopOffset / MASK_BIT_NUM], onceComputeBaseNumMaskLen);
    }
    maskQue.EnQue(maskLocal);
    maskLocal = maskQue.DeQue<uint8_t>();
    BinaryRepeatParams param {
        1,
        1,
        1,
        8,
        8,
        8
    };

    uint32_t selectLoopTimes = validResultVecRepeats / SELECT_REPEAT_TIME;
    uint8_t selectRemainderTimes = static_cast<uint8_t>(validResultVecRepeats % SELECT_REPEAT_TIME);
    for (uint32_t i = 0; i < selectLoopTimes; i++) {
        Select(resultLocal[i * SELECT_REPEAT_TIME * VIC_HALF_FULL_MASK],
               maskLocal[i * SELECT_REPEAT_TIME * VIC_HALF_FULL_MASK / MASK_BIT_NUM],
               resultLocal[i * SELECT_REPEAT_TIME * VIC_HALF_FULL_MASK],
               HALF_MIN,
               SELMODE::VSEL_TENSOR_SCALAR_MODE,
               VIC_HALF_FULL_MASK,
               static_cast<uint8_t>(SELECT_REPEAT_TIME),
               param);
    }
    if (selectRemainderTimes != 0) {
        Select(resultLocal[selectLoopTimes * SELECT_REPEAT_TIME * VIC_HALF_FULL_MASK],
               maskLocal[selectLoopTimes * SELECT_REPEAT_TIME * VIC_HALF_FULL_MASK / MASK_BIT_NUM],
               resultLocal[selectLoopTimes * SELECT_REPEAT_TIME * VIC_HALF_FULL_MASK],
               HALF_MIN,
               SELMODE::VSEL_TENSOR_SCALAR_MODE,
               VIC_HALF_FULL_MASK,
               selectRemainderTimes,
               param);
    }

    maskQue.FreeTensor(maskLocal);
}

__aicore__ inline void AscendcDistInt8FlatCos::ComputeOneLoop(const uint32_t &loopIndex,
    const uint32_t &curBaseNumAlign16, const uint32_t &curBaseNumAlign128, const bool &calcNext)
{
    uint32_t loopOffset = loopIndex * onceComputeBaseNum;

    // 1. 搬运本次计算baseNorm至UB
    auto baseNormLocal = baseNormQue.AllocTensor<half>();
    MoveBaseNormFromGM(baseNormLocal, loopOffset, curBaseNumAlign16);

    // 2. 等待上一次vec->gm搬运完成
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);

    // 3. 搬运GM缓冲区
    resultLocal = resultQue.AllocTensor<half>();
    MoveMatmulRetFromGM();

    // 4. 启动下一次matmul
    if (calcNext) {
        uint32_t bMatrixOffset = (loopOffset + onceComputeBaseNum) * dim;
        CalcAB(bMatrixOffset);
    }

    // 5. 计算query*base*baseNorm*queryNorm
    CalcResult(curBaseNumAlign128);
    baseNormQue.FreeTensor(baseNormLocal);

    // 6. 作用mask
    DoMask(loopOffset);

    // 7. 计算burstMax
    auto resultMaxLocal = resultMaxQue.AllocTensor<half>();
    CalcResultMax(resultMaxLocal);

    // 8. 搬运result
    MoveResultToGM(loopOffset, curBaseNumAlign16);

    // 9. 搬运resultMax
    MoveResultMaxToGM(loopOffset);
    resultQue.FreeTensor(resultLocal);
    resultMaxQue.FreeTensor(resultMaxLocal);
}

__aicore__ inline void AscendcDistInt8FlatCos::MoveAMatrixFromGM(LocalTensor<int8_t> &aMatrixLocal)
{
    Nd2NzParams copyParam = {
        1, // 传输nd矩阵的数目：1个nd
        static_cast<uint16_t>(queryNum), // nd矩阵的行数：输入的行数
        static_cast<uint16_t>(dim), // nd矩阵的列数：输入的列数
        0, // 源操作数相邻nd矩阵起始地址间的偏移：nd数目为1，该数无意义
        static_cast<uint16_t>(dim), // 源操作数同一nd矩阵的相邻行起始地址间的偏移：内存连续的，和列数保持一致即可
        static_cast<uint16_t>(tiling.M), // 目的nz矩阵中，来自源操作数同一行的多行数据相邻行起始地址间的偏移：按128行对齐，每行是32B
        1, // 目的nz矩阵中，Z型矩阵相邻行起始地址之间的偏移：一行32B
        0 // 目的nz矩阵中，相邻nz矩阵起始地址间的偏移：nz数目为1，该数无意义
    };
    DataCopy(aMatrixLocal, aGlobal, copyParam);
    aMatrixQue.EnQue(aMatrixLocal);
    aMatrixLocal = aMatrixQue.DeQue<int8_t>();
}

__aicore__ inline void AscendcDistInt8FlatCos::CalcABFirstTime(LocalTensor<int8_t> &aMatrixLocal)
{
    MoveAMatrixFromGM(aMatrixLocal);

    mm.SetTensorA(aMatrixLocal);
    mm.SetQuantScalar(quantScalar);
    CalcAB(0);
}

__aicore__ inline void AscendcDistInt8FlatCos::MoveQueryNormFromGM()
{
    DataCopyParams copyPara {
        1, // 1片连续内存
        static_cast<uint16_t>(queryNormLocal.GetSize() / BLOCK_HALF_NUM),
        0,
        0,
    };
    DataCopy(queryNormLocal, queryNormGlobal, copyPara);
    queryNormQue.EnQue(queryNormLocal);
    queryNormLocal = queryNormQue.DeQue<half>();
}

__aicore__ inline void AscendcDistInt8FlatCos::CalcResultInLoops()
{
    uint32_t loopNum = computeNum / onceComputeBaseNum;
    uint32_t remainNum = computeNum % onceComputeBaseNum;
    bool calcNext = true;
    for (uint32_t i = 0; i < loopNum; i++) {
        if (i == loopNum - 1) {
            calcNext = (remainNum != 0);
        }
        ComputeOneLoop(i, onceComputeBaseNumAlign16, onceComputeBaseNumAlign128, calcNext);
    }
    if (remainNum != 0) {
        uint32_t remainNumAlign16 = RoundUp(remainNum, ALIGN_16);
        uint32_t remainNumAlign128 = RoundUp(remainNum, ALIGN_128);
        ComputeOneLoop(loopNum, remainNumAlign16, remainNumAlign128, false);
    }
}

__aicore__ inline void AscendcDistInt8FlatCos::Process()
{
    // 1. 先注册matmul，后再处理无效计算量，否则AIC可能会卡住
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm);
    if (computeNum == 0) {
        SetFlag();
        return;
    }
    mm.Init(&tiling);

    // 2. 仅在首次异步matmul前需要搬运A矩阵
    auto aMatrixLocal = aMatrixQue.AllocTensor<int8_t>();
    CalcABFirstTime(aMatrixLocal);

    // 3. queryNorm搬运至UB，常住内存，因此入口处搬运
    queryNormLocal = queryNormQue.AllocTensor<half>();
    MoveQueryNormFromGM();

    // 4. 分片计算
    CalcResultInLoops();

    // 5. 释放内存
    mm.End();
    SetFlag();
    aMatrixQue.FreeTensor(aMatrixLocal);
    queryNormQue.FreeTensor(queryNormLocal);
}

__aicore__ inline void AscendcDistInt8FlatCos::SetFlag()
{
    pipe_barrier(PIPE_MTE3);
    TBuf<> flagBuf;
    pipe.InitBuffer(flagBuf, DEFAULT_C0_SIZE);
    LocalTensor<uint16_t> flagLocal = flagBuf.Get<uint16_t>(DEFAULT_C0_SIZE / sizeof(uint16_t));
    flagLocal.SetValue(0, VALID_FLAG);
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    DataCopy(flagGlobal, flagLocal, flagLocal.GetSize());
}
}

extern "C" __global__ __aicore__ void ascendc_dist_int8_flat_cos(GM_ADDR queryData,
                                                                 GM_ADDR mask,
                                                                 GM_ADDR baseData,
                                                                 GM_ADDR queryNormData,
                                                                 GM_ADDR baseNormData,
                                                                 GM_ADDR actualSize,
                                                                 GM_ADDR result,
                                                                 GM_ADDR resultMax,
                                                                 GM_ADDR flag,
                                                                 GM_ADDR workspace,
                                                                 GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);
    IndexOps::AscendcDistInt8FlatCos op(tilingData);

    if ASCEND_IS_AIV {
        op.Init(queryData, mask, baseData, queryNormData, baseNormData, actualSize, result, resultMax, flag);
    }

    op.Process();
}